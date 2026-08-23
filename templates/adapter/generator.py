"""Piece 3 of 10 — the synthetic world: the demo's data, and the ML spine's training set.

WHAT YOU WRITE HERE
    A **hybrid** generator, which is the pattern that makes label-consistent synthetic
    data cheap. Three layers, and all three matter:

    1. **Procedural, seeded structure.** Every *feature-bearing* field is drawn from a
       seeded ``random.Random``, so a fixed ``seed`` pins the whole world.
    2. **LLM-fabricated text.** Only the prose — titles, details, corpus documents —
       comes from the model gateway, requested by **role** (``ModelRole.CHEAP`` for
       bulk record text, ``ModelRole.GENERATION`` for richer documents), never by a
       hard-coded model id, and parsed defensively.
    3. **Graceful degradation.** With no LLM available (or a malformed response) the
       generator falls back to deterministic templated text and *still* returns
       schema-valid data. On the day this is what makes the system demonstrable while
       the model key is still being sorted out.

    Also here, because it is domain content and used to be a core constant: the
    **client-facing demand series** ``/forecast`` charts.

THE CONTRACT (aegis.adapter.GeneratorModule) — these names must survive
    generate_synthetic()          async, optional LLM
    generate_synthetic_sync()     no LLM, no await — safe inside a running event loop
    DOMAIN_SERIES_LABEL, DOMAIN_SERIES_UNIT, domain_series_events()

    Plus, by convention and by the registry's re-exports: ``GeneratorConfig`` and
    ``assess_quality``.

╔══════════════════════════════════════════════════════════════════════════════╗
║ THE TRAP — this is the one that costs the demo, and nothing in the platform  ║
║ catches it.                                                                  ║
║                                                                              ║
║ The label MUST be drawn around ``ml_spec``'s latent function. Never           ║
║ independently, never "roughly similar", never a second copy of the formula    ║
║ inlined here.                                                                ║
║                                                                              ║
║   right:  mean = ml_spec.latent_cycle_time_hours(features, confounder=z)      ║
║           label = mean + rng.gauss(0.0, sigma)                                ║
║                                                                              ║
║   wrong:  label = rng.uniform(1, 80)          # a plausible-looking number    ║
║   wrong:  label = 12 + 0.7 * backlog + ...    # the formula, typed twice      ║
║                                                                              ║
║ If the label is not a function of the features, the target is noise: R² ≈ 0,  ║
║ the conformal interval is honestly enormous, SHAP has nothing to attribute,   ║
║ and the agent's "ML decision-support" block is a random number in a           ║
║ confident sentence. **The 14 conformance checks all pass.** The adapter suite ║
║ passes. Ruff passes. The only native symptom is ``distinct=False`` on the     ║
║ last line of ``python -m app.ml`` — and if you inline a second copy of the    ║
║ formula it will say ``distinct=True`` right up until someone edits a          ║
║ coefficient in piece 2 and the two copies drift apart in silence.             ║
║                                                                              ║
║ The other half of the trap is noise that is too SMALL. See                    ║
║ ``ml_spec.calibrated_noise_sigma``: this module measures the variance of the  ║
║ latent values it just computed and derives sigma from it, so the achievable   ║
║ R² lands at ``ml_spec.TARGET_R2`` instead of at 0.99.                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

VERIFY
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        tests/adapter/test_generator.py -q)
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml | tail -1)
    aegis-ml contract        # the held-out-R² floor, in seconds, before anything expensive

Targeted API: ``app.core.llm.complete(role, messages, *, tools, temperature,
response_format) -> LLMResult`` (``LLMResult`` has a ``.content: str``).
"""

from __future__ import annotations

import json
import random
import statistics
from datetime import datetime, timedelta
from typing import Protocol

from pydantic import BaseModel, Field

from app.adapter import ml_spec
from app.adapter.schema import (
    DatasetMetadata,
    Document,
    DocumentKind,
    IntakePath,
    Operator,
    Party,
    PartyTier,
    SyntheticDataset,
    UrgencyBand,
    WidgetKind,
    WorkItem,
    WorkItemStage,
    Zone,
)
from app.core.models import ModelRole

_EPOCH = datetime(2024, 1, 1, 9, 0, 0)
"""Base instant for deterministic timestamps, fixed so a seed pins the world.

TODO(domain): keep it far enough in the past that every generated record predates
"now" — a record dated in the future makes ``age`` negative in the lookup tool and
makes the forecast series end after today.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Injected LLM contract (structural — avoids a hard import of app.core.llm)
# ─────────────────────────────────────────────────────────────────────────────


class _LLMResultLike(Protocol):
    """Structural view of ``app.core.llm.LLMResult`` (only ``.content`` is used)."""

    content: str


class CompleteFn(Protocol):
    """The subset of ``app.core.llm.complete`` this generator depends on."""

    async def __call__(
        self,
        role: ModelRole,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        response_format: dict | None = None,
    ) -> _LLMResultLike:
        """Complete a chat request for the given model role."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class GeneratorConfig(BaseModel):
    """Config-driven knobs for one generation run.

    TODO(domain): rename the counts to your entities. Two constraints that are not
    style: every count must be a **positive integer field** (the host's demo graph
    scales the world by introspecting this model's integer fields), and every knob
    must have a working default, because callers construct ``GeneratorConfig()`` bare.
    """

    num_parties: int = Field(default=12, ge=1)
    num_operators: int = Field(default=6, ge=1)
    num_items: int = Field(default=40, ge=1)
    num_documents: int = Field(default=6, ge=0)
    completed_fraction: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Share of records that are finished (i.e. ML-labelled).",
    )
    seed: int | None = Field(
        default=None, description="RNG seed; set for a fully reproducible structure."
    )
    target_r2: float = Field(
        default=ml_spec.TARGET_R2,
        gt=0.0,
        lt=1.0,
        description="Held-out R² the label noise is calibrated for. See the module's "
        "trap block: this is what keeps the target learnable but not trivial.",
    )
    noise_scale: float | None = Field(
        default=None,
        ge=0.0,
        description="Explicit std-dev of Gaussian noise on the target, in the target's "
        "unit. Leave None (the default) to DERIVE it from target_r2 and the measured "
        "variance of the latent signal — the derived value stays correct when a piece-2 "
        "coefficient changes, and a hardcoded one silently stops being.",
    )
    use_llm: bool = Field(
        default=True, description="If False, skip the LLM and use templated text."
    )
    llm_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature for text."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────


async def generate_synthetic(
    config: GeneratorConfig | None = None,
    *,
    complete: CompleteFn | None = None,
) -> SyntheticDataset:
    """Fabricate a complete, schema-valid synthetic world (optionally with LLM prose).

    Args:
        config: Generation knobs; defaults to :class:`GeneratorConfig` defaults.
        complete: The LLM completion function (dependency injection). If ``None`` and
            ``config.use_llm`` is set, ``app.core.llm.complete`` is imported lazily.
            Pass a stub in tests to avoid any network.

    Returns:
        A :class:`SyntheticDataset` whose records seed the data layer and whose
        documents seed retrieval. Every record is pydantic-validated on construction.
    """
    cfg = config or GeneratorConfig()
    rng = random.Random(cfg.seed)

    resolved_complete = _resolve_complete(complete, cfg)

    parties = _build_parties(rng, cfg.num_parties)
    operators = _build_operators(rng, cfg.num_operators)

    item_text = await _fabricate_item_text(resolved_complete, rng, cfg)
    documents = await _fabricate_documents(resolved_complete, rng, cfg)

    return _assemble(
        cfg,
        rng,
        parties,
        operators,
        item_text,
        documents,
        llm_used=resolved_complete is not None,
    )


def generate_synthetic_sync(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Fabricate the synthetic world **synchronously**, with deterministic templated text.

    Identical structure and identical labels to :func:`generate_synthetic`, but with no
    LLM and no ``await`` — so it is safe to call from synchronous code *and* from inside
    a running event loop, where ``asyncio.run`` raises. This is what seeds the
    process-wide record store and what ``ml_spec.training_frame`` calls: neither needs
    LLM-written prose, only schema-valid records whose label is the real latent function
    of the features.

    Args:
        config: Generation knobs; ``use_llm`` is forced off. Defaults apply otherwise.

    Returns:
        A schema-valid :class:`SyntheticDataset` (deterministic under a fixed seed).
    """
    cfg = (config or GeneratorConfig()).model_copy(update={"use_llm": False})
    rng = random.Random(cfg.seed)

    parties = _build_parties(rng, cfg.num_parties)
    operators = _build_operators(rng, cfg.num_operators)
    item_text = _template_item_pool(rng, cfg)
    documents = _template_documents(rng, cfg)

    return _assemble(cfg, rng, parties, operators, item_text, documents, llm_used=False)


def _assemble(
    cfg: GeneratorConfig,
    rng: random.Random,
    parties: list[Party],
    operators: list[Operator],
    item_text: dict[WidgetKind, list[dict]],
    documents: list[Document],
    *,
    llm_used: bool,
) -> SyntheticDataset:
    """Assemble records + metadata into a dataset (the shared sync/async core)."""
    items, noise_sigma = _build_items(rng, cfg, parties, operators, item_text)
    num_labelled = sum(1 for i in items if i.is_labelled)
    metadata = DatasetMetadata(
        seed=cfg.seed,
        llm_used=llm_used,
        num_parties=len(parties),
        num_operators=len(operators),
        num_items=len(items),
        num_documents=len(documents),
        num_labelled=num_labelled,
        target_r2=cfg.target_r2,
        noise_sigma=noise_sigma,
    )
    return SyntheticDataset(
        metadata=metadata,
        parties=parties,
        operators=operators,
        items=items,
        documents=documents,
    )


def _resolve_complete(
    complete: CompleteFn | None, cfg: GeneratorConfig
) -> CompleteFn | None:
    """Pick the completion function to use (injected, lazily imported, or none)."""
    if not cfg.use_llm:
        return None
    if complete is not None:
        return complete
    try:  # Lazy import: app.core.llm may not exist yet during parallel builds.
        from app.core.llm import complete as core_complete  # noqa: PLC0415
    except ImportError:
        return None
    return core_complete


# ─────────────────────────────────────────────────────────────────────────────
# Structural (procedural, seeded) generation
#
# TODO(domain): replace these name pools and builders with your entities. Keep the
# seeded-RNG discipline — every draw goes through ``rng``, never ``random.*`` at module
# level — or the "deterministic under a fixed seed" promise quietly stops being true.
# ─────────────────────────────────────────────────────────────────────────────

_FIRST_NAMES = ("Ava", "Liam", "Noah", "Mia", "Ravi", "Sara", "Chen", "Ines", "Omar", "Kai")
_LAST_NAMES = ("Kim", "Patel", "Silva", "Nguyen", "Haddad", "Rossi", "Meyer", "Costa")
_ORG_STEMS = ("Northwind", "Contoso", "Globex", "Initech", "Umbrella", "Acme", "Hooli")


def _pick(rng: random.Random, seq: tuple) -> str:
    """Return a uniformly random element of ``seq``."""
    return rng.choice(seq)


def _build_parties(rng: random.Random, n: int) -> list[Party]:
    """Create ``n`` deterministic parties."""
    tiers = list(PartyTier)
    zones = list(Zone)
    parties: list[Party] = []
    for i in range(n):
        stem = _pick(rng, _ORG_STEMS)
        org = f"{stem} {_pick(rng, ('Ltd', 'Inc', 'GmbH', 'Group'))}"
        parties.append(
            Party(
                id=f"party-{i:04d}",
                name=org,
                # ``.example`` is a reserved TLD — the quality gate checks for it, and
                # it is what makes "PII-free by construction" a fact rather than a hope.
                contact_email=f"contact{i:04d}@{stem.lower()}.example",
                zone=rng.choice(zones),
                tier=rng.choices(tiers, weights=[6, 3, 1])[0],
                onboarded_at=_EPOCH - timedelta(days=rng.randint(30, 900)),
            )
        )
    return parties


def _build_operators(rng: random.Random, n: int) -> list[Operator]:
    """Create ``n`` deterministic operators."""
    zones = list(Zone)
    kinds = list(WidgetKind)
    operators: list[Operator] = []
    for i in range(n):
        name = f"{_pick(rng, _FIRST_NAMES)} {_pick(rng, _LAST_NAMES)}"
        skills = rng.sample(kinds, k=rng.randint(1, 2))
        operators.append(
            Operator(
                id=f"op-{i:03d}",
                name=name,
                line=f"Line-{rng.randint(1, 3)} {skills[0].value.title()}",
                tenure_months=rng.randint(1, 60),
                zone=rng.choice(zones),
                skills=skills,
            )
        )
    return operators


def _build_items(
    rng: random.Random,
    cfg: GeneratorConfig,
    parties: list[Party],
    operators: list[Operator],
    item_text: dict[WidgetKind, list[dict]],
) -> tuple[list[WorkItem], float]:
    """Assemble records and label the finished ones from the latent signal.

    **Two passes, and the reason is the trap in the module docstring.** The first pass
    draws every structural field and computes the noise-free latent value for the
    records that will be labelled. Only then is the variance of those latent values
    known — and only then can :func:`~app.adapter.ml_spec.calibrated_noise_sigma`
    derive the sigma that lands the achievable R² at ``cfg.target_r2``. A single pass
    would have to guess sigma, which is how "the label is learnable" becomes a claim
    nobody re-checks after the next coefficient edit.

    Returns:
        A tuple ``(items, noise_sigma)`` — the records, and the sigma actually applied
        (recorded onto the dataset metadata).
    """
    urgencies = list(UrgencyBand)
    intakes = list(IntakePath)
    kinds = list(WidgetKind)
    text_cursor: dict[WidgetKind, int] = dict.fromkeys(kinds, 0)

    drafts: list[tuple[WorkItem, Party, Operator | None, bool, float, float]] = []
    latent_values: list[float] = []

    for i in range(cfg.num_items):
        # Coverage guarantee (class balance): the first pass round-robins every kind so
        # no class is ever missing even for a small N; the remainder is drawn at random
        # for realistic imbalance. Deterministic under a fixed seed either way.
        kind = kinds[i] if i < len(kinds) else rng.choice(kinds)
        party = rng.choice(parties)
        specialists = [o for o in operators if kind in o.skills]
        operator = rng.choice(specialists or operators)

        urgency = rng.choices(urgencies, weights=[5, 3, 1])[0]
        intake = rng.choice(intakes)
        backlog = rng.randint(0, 40)
        rework = rng.choices([0, 1, 2], weights=[8, 2, 1])[0]

        title, detail = _next_text(item_text, kind, text_cursor, rng)
        created_at = _EPOCH - timedelta(hours=rng.randint(1, 24 * 120))
        will_complete = rng.random() < cfg.completed_fraction

        draft = WorkItem(
            id=f"item-{i:06d}",
            title=title,
            detail=detail,
            kind=kind,
            urgency=urgency,
            intake=intake,
            zone=party.zone,
            stage=WorkItemStage.IN_STAGE,
            party_id=party.id,
            assigned_operator_id=operator.id,
            created_at=created_at,
            updated_at=created_at,
            backlog_at_intake=backlog,
            rework_count=rework,
            first_touch_minutes=rng.randint(2, 240),
            target_cycle_hours=float(rng.choice([8, 24, 48, 72])),
        )

        # THE COUPLING. The label's mean comes from piece 2's latent function and from
        # nowhere else. ``confounder`` is the unobserved driver: drawn here, never
        # written onto the record, never a feature.
        features = ml_spec.features_for_item(draft, operator=operator, party=party)
        confounder = rng.gauss(0.0, 1.0)
        latent = ml_spec.latent_cycle_time_hours(features, confounder=confounder)
        # Reserve one standard-normal draw per record whether or not it is labelled, so
        # the RNG stream does not depend on the completion coin-flip.
        noise_draw = rng.gauss(0.0, 1.0)

        drafts.append((draft, party, operator, will_complete, latent, noise_draw))
        if will_complete:
            latent_values.append(latent)

    var_signal = statistics.pvariance(latent_values) if len(latent_values) > 1 else 0.0
    sigma = (
        cfg.noise_scale
        if cfg.noise_scale is not None
        else ml_spec.calibrated_noise_sigma(var_signal, target_r2=cfg.target_r2)
    )

    items: list[WorkItem] = []
    for draft, _party, _operator, will_complete, latent, noise_draw in drafts:
        if will_complete:
            items.append(_finalise_completed(draft, latent, sigma, noise_draw, rng))
        else:
            items.append(
                draft.model_copy(
                    update={
                        "stage": rng.choice(
                            [WorkItemStage.RECEIVED, WorkItemStage.QUEUED]
                        )
                    }
                )
            )
    return items, round(sigma, 4)


def _finalise_completed(
    item: WorkItem,
    latent: float,
    sigma: float,
    noise_draw: float,
    rng: random.Random,
) -> WorkItem:
    """Stamp the target and completion timestamps onto one finished record."""
    labelled = max(0.25, latent + sigma * noise_draw)
    cycle_time = round(labelled, 2)
    completed_at = item.created_at + timedelta(hours=cycle_time)
    return item.model_copy(
        update={
            "stage": rng.choices(
                [WorkItemStage.COMPLETED, WorkItemStage.CLOSED], weights=[3, 2]
            )[0],
            "completed_at": completed_at,
            "updated_at": completed_at,
            "cycle_time_hours": cycle_time,
            "quality_score": rng.randint(3, 5),
        }
    )


def _next_text(
    item_text: dict[WidgetKind, list[dict]],
    kind: WidgetKind,
    cursor: dict[WidgetKind, int],
    rng: random.Random,
) -> tuple[str, str]:
    """Pull the next (title, detail) for ``kind``, cycling the pool if needed."""
    pool = item_text.get(kind) or []
    if not pool:
        return _template_item_text(kind, rng)
    idx = cursor[kind] % len(pool)
    cursor[kind] += 1
    entry = pool[idx]
    title = str(entry.get("title") or "").strip() or _template_item_text(kind, rng)[0]
    body = str(entry.get("detail") or "").strip()
    if not body:
        body = _template_item_text(kind, rng)[1]
    return title, body


# ─────────────────────────────────────────────────────────────────────────────
# Quality gate
# ─────────────────────────────────────────────────────────────────────────────


class DatasetQualityReport(BaseModel):
    """A quick, dependency-free quality gate over a generated dataset.

    These are the checks worth running *before* trusting synthetic data: referential
    integrity, class coverage, a learnable label present, temporal consistency, and
    PII-free-by-construction.

    TODO(domain): add the checks your world needs and delete the ones it does not. The
    one to keep no matter what is ``has_labels`` — an empty training frame is the
    failure that looks like a model problem for an hour.
    """

    referential_integrity: bool = Field(description="Every FK resolves to a record.")
    kind_coverage: bool = Field(description="Every kind appears at least once.")
    has_labels: bool = Field(description="At least one record carries an ML target.")
    temporal_consistency: bool = Field(description="completed_at ≥ created_at everywhere.")
    pii_free: bool = Field(
        description="Reserved .example addresses AND the generated free text scanned "
        "clean by the guardrail PII detector."
    )
    num_labelled: int = Field(description="Count of ML-labelled records.")
    kind_counts: dict[str, int] = Field(description="Records per kind (class balance).")

    @property
    def ok(self) -> bool:
        """Whether every hard quality check passed."""
        return (
            self.referential_integrity
            and self.kind_coverage
            and self.has_labels
            and self.temporal_consistency
            and self.pii_free
        )


def _is_pii_free(dataset: SyntheticDataset) -> bool:
    """Whether the dataset carries no real-looking PII — actually scanned, not assumed.

    Two checks, and the second is the one that earns its keep: (1) every contact address
    is a reserved ``.example`` one, and (2) the **generated free text** contains no
    detectable PII per the guardrail detector. When text is LLM-fabricated a model can
    slip a real-looking email or phone number into a detail field, which "addresses end
    in .example" would never catch.
    """
    from app.guardrails.pii import contains_pii

    if not all(p.contact_email.endswith(".example") for p in dataset.parties):
        return False
    texts: list[str] = []
    for item in dataset.items:
        texts.append(item.title)
        texts.append(item.detail)
    for doc in dataset.documents:
        texts.append(doc.title)
        texts.append(doc.body)
    return not any(contains_pii(t) for t in texts)


def assess_quality(dataset: SyntheticDataset) -> DatasetQualityReport:
    """Run the synthetic-data quality checks over ``dataset`` and report the verdict.

    Pure, offline and cheap — safe to call after every generation run (and asserted in
    tests), so a malformed world is caught before it seeds the stores.

    Args:
        dataset: The generated dataset to inspect.

    Returns:
        A :class:`DatasetQualityReport` with per-check booleans and balance counts.
    """
    party_ids = {p.id for p in dataset.parties}
    operator_ids = {o.id for o in dataset.operators}
    referential = all(
        item.party_id in party_ids
        and (
            item.assigned_operator_id is None
            or item.assigned_operator_id in operator_ids
        )
        for item in dataset.items
    )
    counts: dict[str, int] = dict.fromkeys((k.value for k in WidgetKind), 0)
    for item in dataset.items:
        counts[item.kind.value] = counts.get(item.kind.value, 0) + 1
    temporal = all(
        item.completed_at is None or item.completed_at >= item.created_at
        for item in dataset.items
    )
    return DatasetQualityReport(
        referential_integrity=referential,
        kind_coverage=all(v > 0 for v in counts.values()),
        has_labels=dataset.metadata.num_labelled > 0,
        temporal_consistency=temporal,
        pii_free=_is_pii_free(dataset),
        num_labelled=dataset.metadata.num_labelled,
        kind_counts=counts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM-fabricated content (with deterministic fallbacks)
#
# TODO(domain): rewrite both prompts. They are the only place the *flavour* of your
# records comes from, and a prompt still asking for the placeholder world produces
# records whose prose contradicts their own fields.
# ─────────────────────────────────────────────────────────────────────────────

_KIND_HINTS: dict[WidgetKind, str] = {
    WidgetKind.ALPHA: "TODO(domain): the themes an 'alpha' item is usually about",
    WidgetKind.BETA: "TODO(domain): the themes a 'beta' item is usually about",
    WidgetKind.GAMMA: "TODO(domain): the themes a 'gamma' item is usually about",
    WidgetKind.DELTA: "TODO(domain): the themes a 'delta' item is usually about",
}


async def _fabricate_item_text(
    complete: CompleteFn | None,
    rng: random.Random,
    cfg: GeneratorConfig,
) -> dict[WidgetKind, list[dict]]:
    """Fetch realistic (title, detail) pairs per kind via the LLM.

    Falls back to templated text for any kind the LLM cannot supply, so the caller
    always receives a full pool.
    """
    per_kind = max(3, cfg.num_items // (len(WidgetKind) or 1))
    out: dict[WidgetKind, list[dict]] = {}
    for kind in WidgetKind:
        entries: list[dict] = []
        if complete is not None:
            entries = await _llm_item_text(complete, kind, per_kind, cfg)
        if not entries:
            entries = [
                {"title": t, "detail": d}
                for t, d in (_template_item_text(kind, rng) for _ in range(per_kind))
            ]
        out[kind] = entries
    return out


async def _llm_item_text(
    complete: CompleteFn,
    kind: WidgetKind,
    count: int,
    cfg: GeneratorConfig,
) -> list[dict]:
    """Ask the CHEAP model for ``count`` record title/detail pairs of one kind."""
    system = (
        "TODO(domain): one sentence describing what you fabricate. Keep the "
        "'entirely fictional, never reference real people or companies' clause — the "
        "quality gate scans this text for PII and a model will happily invent a real "
        "-looking address if nothing tells it not to."
    )
    user = (
        f"Produce {count} distinct work items of kind '{kind.value}' "
        f"(themes: {_KIND_HINTS[kind]}). Return JSON of the form "
        '{"items": [{"title": "...", "detail": "..."}, ...]}. '
        "Titles under 80 chars; details 1–3 sentences."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return await _call_and_parse(complete, ModelRole.CHEAP, messages, "items", cfg)


async def _fabricate_documents(
    complete: CompleteFn | None,
    rng: random.Random,
    cfg: GeneratorConfig,
) -> list[Document]:
    """Fabricate a small knowledge-document corpus via the GENERATION model."""
    if cfg.num_documents <= 0:
        return []
    kinds = list(WidgetKind)
    docs: list[Document] = []
    llm_docs: list[dict] = []
    if complete is not None:
        llm_docs = await _llm_documents(complete, cfg.num_documents, cfg)

    for i in range(cfg.num_documents):
        kind = kinds[i % len(kinds)]
        source = llm_docs[i] if i < len(llm_docs) else {}
        title = str(source.get("title") or "").strip()
        body = str(source.get("body") or "").strip()
        if not title or not body:
            title, body = _template_document(kind, i)
        docs.append(
            Document(
                id=f"doc-{i:04d}",
                kind=rng.choice(list(DocumentKind)),
                title=title,
                body=body,
                kind_scope=kind,
                tags=[kind.value, "synthetic"],
                source="synthetic",
            )
        )
    return docs


async def _llm_documents(
    complete: CompleteFn, count: int, cfg: GeneratorConfig
) -> list[dict]:
    """Ask the GENERATION model for ``count`` short knowledge documents."""
    system = (
        "TODO(domain): who writes these documents, and for whom. Keep them "
        "self-contained, generic and fictional."
    )
    user = (
        f"Write {count} short knowledge documents spanning "
        f"{', '.join(k.value for k in WidgetKind)}. Return JSON of the form "
        '{"documents": [{"title": "...", "body": "..."}, ...]}. '
        "Each body 3–6 sentences of actionable guidance."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return await _call_and_parse(complete, ModelRole.GENERATION, messages, "documents", cfg)


async def _call_and_parse(
    complete: CompleteFn,
    role: ModelRole,
    messages: list[dict],
    key: str,
    cfg: GeneratorConfig,
) -> list[dict]:
    """Call the gateway for JSON, parse defensively, and return ``result[key]``.

    Any transport or parsing failure returns ``[]`` so the caller falls back to
    templated content. **The generator must never raise on an LLM problem** — that is
    the whole graceful-degradation guarantee, and it is what makes the demo survive a
    missing API key.

    Args:
        complete: The injected completion function.
        role: Which model role to bill the call to.
        messages: Chat messages to send.
        key: The top-level JSON key holding the list of items.
        cfg: Generation config (supplies the temperature).

    Returns:
        The parsed list of dict items, or ``[]`` on any failure.
    """
    try:
        result = await complete(
            role,
            messages,
            temperature=cfg.llm_temperature,
            response_format={"type": "json_object"},
        )
        payload = json.loads(result.content)
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError, KeyError):
        return []
    entries = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic templated fallbacks (guarantee schema-valid output offline)
#
# TODO(domain): write real placeholder prose for your world. This path runs whenever
# there is no LLM — which is every test, every offline run, every training frame, and
# quite possibly the demo. It is not a second-class path.
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_TITLES: dict[WidgetKind, tuple[str, ...]] = {
    WidgetKind.ALPHA: (
        "TODO(domain): a typical alpha item",
        "TODO(domain): another alpha item",
        "TODO(domain): a third alpha item",
    ),
    WidgetKind.BETA: (
        "TODO(domain): a typical beta item",
        "TODO(domain): another beta item",
        "TODO(domain): a third beta item",
    ),
    WidgetKind.GAMMA: (
        "TODO(domain): a typical gamma item",
        "TODO(domain): another gamma item",
        "TODO(domain): a third gamma item",
    ),
    WidgetKind.DELTA: (
        "TODO(domain): a typical delta item",
        "TODO(domain): another delta item",
        "TODO(domain): a third delta item",
    ),
}


def _template_item_text(kind: WidgetKind, rng: random.Random) -> tuple[str, str]:
    """Return a deterministic (title, detail) for ``kind`` (the LLM fallback)."""
    title = rng.choice(_TEMPLATE_TITLES[kind])
    detail = (
        f"A {kind.value} work item was raised: {title.lower()}. "
        "TODO(domain): write the sentence a real record would carry."
    )
    return title, detail


def _template_document(kind: WidgetKind, index: int) -> tuple[str, str]:
    """Return a deterministic (title, body) knowledge document (the LLM fallback)."""
    title = f"{kind.value.title()} handling guide #{index}"
    body = (
        f"TODO(domain): a short guide covering common {kind.value} scenarios "
        f"({_KIND_HINTS[kind]}). Three to six sentences of actionable guidance — long "
        "enough that the chunker produces at least one chunk, which conformance check "
        "#13 verifies for seed documents."
    )
    return title, body


def _template_item_pool(
    rng: random.Random, cfg: GeneratorConfig
) -> dict[WidgetKind, list[dict]]:
    """Deterministic templated (title, detail) pool — the no-LLM path."""
    per_kind = max(3, cfg.num_items // (len(WidgetKind) or 1))
    return {
        kind: [
            {"title": t, "detail": d}
            for t, d in (_template_item_text(kind, rng) for _ in range(per_kind))
        ]
        for kind in WidgetKind
    }


def _template_documents(rng: random.Random, cfg: GeneratorConfig) -> list[Document]:
    """Deterministic templated knowledge-document corpus — the no-LLM path."""
    if cfg.num_documents <= 0:
        return []
    kinds = list(WidgetKind)
    docs: list[Document] = []
    for i in range(cfg.num_documents):
        kind = kinds[i % len(kinds)]
        title, body = _template_document(kind, i)
        docs.append(
            Document(
                id=f"doc-{i:04d}",
                kind=rng.choice(list(DocumentKind)),
                title=title,
                body=body,
                kind_scope=kind,
                tags=[kind.value, "synthetic"],
                source="synthetic",
            )
        )
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# The demand series — what /forecast forecasts, in this domain's words
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_SERIES_LABEL = "TODO(domain): Work items received per day"
"""What the client-facing demand series measures, in the client's language.

**This string used to be a constant in the core**, which made it the one domain
sentence no retarget ever changed: the ``/forecast`` response carried it to the
console, the console drew it on the chart, and a completely different deployment
charted the shipped domain's words over its own data forever.

TODO(domain): it is a **sentence a jury reads**. Write it as the client would say it,
and delete the ``TODO(domain):`` prefix — the prefix is there so a forgotten edit is
visible on screen instead of merely wrong.
"""

DOMAIN_SERIES_UNIT = "items"
"""The unit of :func:`domain_series_events`' values, for the forecast's y-axis.

TODO(domain): the plural noun the y-axis is counted in ("shipments", "claims",
"readings"). It is rendered next to the numbers, so it must read as a unit.
"""


def domain_series_events(
    *, num_records: int = 1400, seed: int = 11
) -> list[tuple[datetime, float]]:
    """Return one ``(timestamp, 1.0)`` arrival event per generated record.

    The **arrival** series, deliberately, not a completion series: arrivals are the
    quantity a client plans capacity against, and the series is complete at the recent
    end — whereas completions silently truncate it and bias the trend downwards for no
    reason a reader could see.

    This is the whole of the domain's contribution to ``/forecast``. The core buckets,
    fits and refuses honestly; it never names a record type or a timestamp field. Before
    this function existed it did both — it read the shipped domain's record collection
    and timestamp field by attribute name — so a retarget that renamed the collection
    made ``/forecast`` raise ``AttributeError`` with nothing in any checklist pointing
    at the file.

    Args:
        num_records: How many records to fabricate. Large enough that a daily bucket
            over the generator's span is a countable volume rather than a sparse 0/1
            rattle no model (and no reader) could learn from.
        seed: RNG seed, so the demo series is identical across processes and reloads.

    Returns:
        Arrival events, unordered.
    """
    dataset = generate_synthetic_sync(
        GeneratorConfig(num_items=num_records, seed=seed, use_llm=False)
    )
    return [(item.created_at, 1.0) for item in dataset.items]


__all__ = [
    "DOMAIN_SERIES_LABEL",
    "DOMAIN_SERIES_UNIT",
    "CompleteFn",
    "DatasetQualityReport",
    "GeneratorConfig",
    "assess_quality",
    "domain_series_events",
    "generate_synthetic",
    "generate_synthetic_sync",
]
