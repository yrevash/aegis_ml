"""Piece 2 of 10 — the supervised problem: which columns, which target, which frame.

WHAT YOU WRITE HERE
    The single source of truth for what is predictable in this domain:

      * :data:`FEATURES` — the ordered, typed feature contract (name, dtype,
        description, levels). Every level string comes from a piece 1 ``StrEnum``.
      * :data:`TARGET` — what is predicted, its task, and its **unit** (the unit is
        printed by ``python -m app.ml`` and by :func:`describe_prediction`).
      * :func:`latent_cycle_time_hours` — the **ground-truth generative signal**.
        Piece 3 calls this to SET each finished record's label. Rename it for your
        domain; keep it here rather than in the generator, so the drivers of the
        target are declared in one place and the data cannot drift from the spec.
      * :func:`features_for_item` — one record (joined to its operator and party)
        → a flat feature dict.
      * :func:`feature_matrix` — a dataset → ``(X, y)``.
      * :func:`training_frame` — a fresh synthetic world → the labelled DataFrame the
        spine trains on.
      * :func:`describe_prediction` — one prediction → the domain's own
        decision-support sentence, injected into the planner prompt.

    **Pure Python at import time.** No numpy, no pandas, no xgboost at module scope —
    pandas is imported *inside* :func:`training_frame`. This module is imported by
    ``aegis.ml.spec.resolve_spec`` in contexts where the ML extra may not be
    installed, and by the conformance suite, which runs with no ML stack at all.

THE CONTRACT (aegis.adapter.MLSpecModule) — these five names must survive
    FEATURES, FEATURE_NAMES, TARGET, training_frame(), describe_prediction()

    ``training_frame``'s keyword is ``num_records``, deliberately domain-neutral: the
    core Protocol names it, and spelling it ``num_items`` here breaks the contract.

THE TRAP
    ``aegis.ml.spec.resolve_spec`` reads ``FEATURE_NAMES`` and ``TARGET.name``
    leniently and returns ``FALLBACK_SPEC`` — four columns of generated noise called
    ``feature_0…3`` — when it finds neither. **Nothing raises.** Your model trains on
    noise and serves it as domain evidence, and the only native symptom is
    ``distinct=False`` on the last line of ``python -m app.ml``.

THE OTHER TRAP — the one that costs the demo
    The label must be sampled *around* :func:`latent_cycle_time_hours` (piece 3's job),
    and the noise must be **calibrated**. Both directions are failures:

      * No coupling at all ⇒ the target is noise, R² ≈ 0, the conformal interval is
        honestly enormous, and the model has nothing to explain.
      * Latent function + tiny noise ⇒ R² ≈ 0.99, which is not a triumph. It says the
        label is a closed-form function of the inputs; SHAP just re-reads your
        coefficients back to you, and a reviewer who asks "so what does the model add?"
        is correct.

    Aim for a held-out R² of roughly **0.45–0.80**. Two devices below get you there
    honestly rather than by hand-tuning a magic number:

      1. **Calibrated Gaussian noise.** For a signal with variance ``var_signal``,
         ``sigma = sqrt(var_signal * (1 - target_r2) / target_r2)`` gives an
         irreducible-error floor that lands the achievable R² at ``target_r2``. See
         :func:`calibrated_noise_sigma` — piece 3 calls it after measuring the
         variance of the latent values it just computed, so the number is derived from
         your coefficients rather than guessed.
      2. **An unobserved confounder.** :data:`CONFOUNDER_WEIGHT` scales a per-record
         latent draw that moves the target and is *not* in :data:`FEATURES`. This is
         what real data has and a closed-form generator does not: a genuine reason the
         best possible model still cannot reach 1.0.

    One **interaction term** is included on purpose too (urgency × backlog): a purely
    additive latent function is learnable by a linear model, which makes the gradient-
    boosted ensemble in the spine pointless and the SHAP plot flat.

VERIFY
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        tests/adapter/test_ml_spec.py -q)
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        --pyargs aegis.conformance --aegis-adapter app.adapter -q -k ml_spec)
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml | tail -1)
        ↑ read the last line. ``distinct=False`` means the spine learned nothing.
"""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import pandas as pd

    from app.api.schemas import MLExplainResponse

from app.adapter.schema import (
    IntakePath,
    Operator,
    Party,
    PartyTier,
    SyntheticDataset,
    UrgencyBand,
    WidgetKind,
    WorkItem,
    Zone,
)

FeatureDType = Literal["categorical", "numeric", "boolean"]


class FeatureSpec(BaseModel):
    """One model feature: its name, dtype and a human description.

    ``dtype`` is load-bearing beyond documentation: ``resolve_spec`` derives the
    one-hot subset from ``dtype == "categorical"`` whenever the adapter declares no
    explicit :data:`CATEGORICAL_FEATURES`. A numeric column mislabelled categorical
    becomes hundreds of one-hot columns; a categorical mislabelled numeric is handed
    to the estimator as a meaningless integer ordering.
    """

    name: str
    dtype: FeatureDType
    description: str
    levels: list[str] | None = Field(
        default=None, description="Allowed values for categorical features."
    )
    unit: str | None = Field(default=None, description="Unit for numeric features.")


class TargetSpec(BaseModel):
    """The prediction target the ML spine learns."""

    name: str
    task: Literal["regression", "classification"]
    unit: str | None = None
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# The contract
#
# TODO(domain): replace every FeatureSpec below with the columns from your Domain
# Brief. Rules that are not style:
#   * ORDER IS THE CONTRACT — FEATURE_NAMES preserves it and the training frame's
#     columns are built from it.
#   * levels= must come from the piece 1 enum, never a hand-typed list, or the two
#     drift and an unseen level one-hot-encodes to all-zeros in silence.
#   * every feature must be knowable BEFORE the target is observed.
# ─────────────────────────────────────────────────────────────────────────────

FEATURES: list[FeatureSpec] = [
    FeatureSpec(
        name="urgency",
        dtype="categorical",
        description="Urgency band; more urgent items are worked sooner.",
        levels=[u.value for u in UrgencyBand],
    ),
    FeatureSpec(
        name="kind",
        dtype="categorical",
        description="Subject area of the work item; some kinds are intrinsically slower.",
        levels=[k.value for k in WidgetKind],
    ),
    FeatureSpec(
        name="intake",
        dtype="categorical",
        description="How the item arrived; batched intake waits for the next batch.",
        levels=[i.value for i in IntakePath],
    ),
    FeatureSpec(
        name="zone",
        dtype="categorical",
        description="Site/geography; affects staffing windows and handover cost.",
        levels=[z.value for z in Zone],
    ),
    FeatureSpec(
        name="party_tier",
        dtype="categorical",
        description="Commercial tier of the counterparty; higher tiers are handled sooner.",
        levels=[t.value for t in PartyTier],
    ),
    FeatureSpec(
        name="operator_tenure_months",
        dtype="numeric",
        unit="months",
        description="Experience of the assigned operator; experienced operators are faster.",
    ),
    FeatureSpec(
        name="backlog_at_intake",
        dtype="numeric",
        unit="items",
        description="Queue depth at intake; a deeper queue means a longer wait.",
    ),
    FeatureSpec(
        name="rework_count",
        dtype="numeric",
        unit="passes",
        description="Times sent back for rework; each pass adds handling time.",
    ),
    FeatureSpec(
        name="detail_length",
        dtype="numeric",
        unit="characters",
        description="Characters in the item detail; a weak complexity proxy.",
    ),
]
"""The ordered, typed feature contract the ML spine consumes."""

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
"""Ordered feature-column names. ``resolve_spec`` reads THIS NAME — do not rename it."""

CATEGORICAL_FEATURES: list[str] = [f.name for f in FEATURES if f.dtype == "categorical"]
NUMERIC_FEATURES: list[str] = [f.name for f in FEATURES if f.dtype != "categorical"]

TARGET: TargetSpec = TargetSpec(
    name="cycle_time_hours",
    task="regression",
    unit="hours",
    description=(
        "TODO(domain): what is predicted, in the client's words, plus the sentence "
        "that says WHY it is learnable. Here: wall-clock hours from intake to "
        "completion — a smooth, monotone function of the operational features with "
        "calibrated noise, so it is genuinely learnable and the conformal interval "
        "means something."
    ),
)
"""The prediction target. ``resolve_spec`` reads ``TARGET.name`` — do not rename it."""


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth latent signal (piece 3 samples labels around this)
#
# TODO(domain): these coefficient tables ARE the domain's causal story. Write them
# from the Brief's "latent drivers" section, keep every driver MONOTONE (more backlog
# is never faster), and keep the magnitudes in the target's own unit so a reader can
# sanity-check them by eye.
# ─────────────────────────────────────────────────────────────────────────────

_INTERCEPT: float = 10.0

_URGENCY_SPEEDUP: dict[str, float] = {
    UrgencyBand.ROUTINE.value: 0.0,
    UrgencyBand.ELEVATED.value: 6.0,
    UrgencyBand.URGENT.value: 11.0,
}

_KIND_BASE: dict[str, float] = {
    WidgetKind.ALPHA.value: 8.0,
    WidgetKind.BETA.value: 26.0,
    WidgetKind.GAMMA.value: 15.0,
    WidgetKind.DELTA.value: 19.0,
}

_INTAKE_DELAY: dict[str, float] = {
    IntakePath.STREAM.value: 1.0,
    IntakePath.MANUAL.value: 3.0,
    IntakePath.BATCH.value: 8.0,
}

_ZONE_DELAY: dict[str, float] = {
    Zone.ZONE_NORTH.value: 2.0,
    Zone.ZONE_SOUTH.value: 4.5,
    Zone.ZONE_EAST.value: 6.0,
}

_TIER_SPEEDUP: dict[str, float] = {
    PartyTier.BASIC.value: 0.0,
    PartyTier.PLUS.value: 2.5,
    PartyTier.PRIME.value: 5.0,
}

_BACKLOG_PER_ITEM: float = 0.7
_REWORK_PER_PASS: float = 5.5
_TENURE_PER_MONTH: float = 0.45
_DETAIL_PER_CHAR: float = 0.012

_URGENCY_BACKLOG_INTERACTION: dict[str, float] = {
    UrgencyBand.ROUTINE.value: 1.0,
    UrgencyBand.ELEVATED.value: 0.6,
    UrgencyBand.URGENT.value: 0.25,
}
"""The ONE interaction term: how much of the backlog an item actually queues behind.

TODO(domain): keep an interaction of some kind, and keep it one you can justify in a
sentence. Here: an urgent item jumps most of the queue, so backlog costs it a quarter
of what it costs a routine one. A purely additive latent function is exactly recoverable
by linear regression, which makes the spine's gradient-boosted ensemble decorative and
flattens the SHAP plot into a restatement of the coefficients above.
"""

CONFOUNDER_WEIGHT: float = 7.0
"""Scale of the **unobserved confounder** — the honest reason R² cannot reach 1.0.

Piece 3 draws one standard-normal value per record and passes it as
:func:`latent_cycle_time_hours`' ``confounder`` argument. It moves the target and it is
deliberately **not** in :data:`FEATURES`, exactly as an unmeasured driver would be in
real data (the operator's day, an upstream supplier's delay, a machine warming up).

TODO(domain): keep a term like this, sized so it is a real fraction of the signal's
spread — a confounder worth 1% of the variance is a rounding error pretending to be
epistemic humility.
"""

TARGET_R2: float = 0.65
"""The held-out R² the label noise is calibrated FOR — the data's own ceiling.

TODO(domain): 0.45–0.80 is the band worth being in. Below it the model looks broken;
above ~0.9 the label is a closed-form function of the inputs and the whole ML story
collapses into "we wrote a formula and then fitted it".

Recorded onto :class:`~app.adapter.schema.DatasetMetadata` by piece 3, so a model card
can state the ceiling rather than leaving a 0.62 looking like an under-fit.
"""

_FLOOR_HOURS: float = 0.25
"""Lower clamp — the target is a duration and durations are positive."""


def calibrated_noise_sigma(var_signal: float, *, target_r2: float = TARGET_R2) -> float:
    """Return the Gaussian sigma that lands the achievable R² at ``target_r2``.

    For a label ``y = signal + noise`` with independent noise, the best achievable
    ``R² = var_signal / (var_signal + sigma²)``. Solving for sigma::

        sigma = sqrt(var_signal * (1 - target_r2) / target_r2)

    Piece 3 computes the latent value for every record it is about to label, measures
    the variance of *those* values, and calls this — so the noise is derived from the
    coefficient table above rather than being a magic constant that silently stops
    being right the moment a coefficient changes.

    Args:
        var_signal: Variance of the latent (noise-free) target across the batch.
        target_r2: The held-out R² to calibrate for; defaults to :data:`TARGET_R2`.

    Returns:
        The standard deviation to add as Gaussian noise. ``0.0`` for a degenerate
        batch (zero variance), where any noise at all would be pure noise.

    Raises:
        ValueError: If ``target_r2`` is not strictly inside ``(0, 1)`` — a request for
            a perfectly learnable or perfectly unlearnable label is a spec error, not
            something to silently clamp.
    """
    if not 0.0 < target_r2 < 1.0:
        raise ValueError(f"target_r2 must be in (0, 1), got {target_r2!r}")
    if var_signal <= 0.0:
        return 0.0
    return sqrt(var_signal * (1.0 - target_r2) / target_r2)


def latent_cycle_time_hours(features: dict, *, confounder: float = 0.0) -> float:
    """Compute the noise-free ground-truth target for one feature row.

    **This is the linchpin of the whole adapter's "predictable" claim.** Piece 3 calls
    it to set every finished record's label, so the target is a real function of the
    features by construction — which is what makes the spine's ensemble learnable and
    its conformal intervals meaningful rather than fitted to noise.

    Monotone in every driver (more urgent → faster; deeper backlog → slower; more
    experienced operator → faster), with one interaction
    (:data:`_URGENCY_BACKLOG_INTERACTION`) and one unobserved term
    (:data:`CONFOUNDER_WEIGHT`).

    TODO(domain): rename this function for your target and rewrite the body from your
    Brief's latent drivers. ``adapter/__init__.py`` re-exports it by name, so update
    that import too.

    Args:
        features: A feature dict as produced by :func:`features_for_item`. Missing
            keys fall back to neutral values so a partial row still scores rather than
            raising mid-generation.
        confounder: A standard-normal draw representing everything that moves the
            target and is not measured. Zero (the default) gives the observable-only
            signal, which is what an analysis or a what-if should use.

    Returns:
        The expected value of the target, floored at a small positive value.
    """
    hours = _INTERCEPT
    hours += _KIND_BASE.get(features.get("kind", ""), 15.0)
    hours += _INTAKE_DELAY.get(features.get("intake", ""), 3.0)
    hours += _ZONE_DELAY.get(features.get("zone", ""), 4.0)
    hours -= _URGENCY_SPEEDUP.get(features.get("urgency", ""), 0.0)
    hours -= _TIER_SPEEDUP.get(features.get("party_tier", ""), 0.0)

    # The interaction: how much of the queue this item actually waits behind.
    queue_exposure = _URGENCY_BACKLOG_INTERACTION.get(features.get("urgency", ""), 1.0)
    hours += _BACKLOG_PER_ITEM * queue_exposure * float(
        features.get("backlog_at_intake", 0) or 0
    )

    hours += _REWORK_PER_PASS * float(features.get("rework_count", 0) or 0)
    hours -= _TENURE_PER_MONTH * float(features.get("operator_tenure_months", 0) or 0)
    hours += _DETAIL_PER_CHAR * float(features.get("detail_length", 0) or 0)

    # The unobserved driver. Not a feature, on purpose.
    hours += CONFOUNDER_WEIGHT * float(confounder)

    return max(_FLOOR_HOURS, round(hours, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Record → features
# ─────────────────────────────────────────────────────────────────────────────


def features_for_item(
    item: WorkItem,
    *,
    operator: Operator | None,
    party: Party | None,
) -> dict:
    """Extract the model feature dict for one record.

    TODO(domain): rename to ``features_for_<your record>`` and return exactly the keys
    in :data:`FEATURE_NAMES` — no more (an extra key is a column the spine never sees)
    and no fewer (a missing key becomes a NaN column at training time). Categorical
    values are the enum ``.value`` strings, never the enum members: a ``StrEnum``
    member survives a DataFrame round-trip but not a JSON one.

    Args:
        item: The record to featurise.
        operator: The assigned operator (joined), or None if unassigned.
        party: The owning party (joined), or None if unknown.

    Returns:
        A flat ``{feature_name: value}`` dict covering every entry in :data:`FEATURES`.
    """
    return {
        "urgency": item.urgency.value,
        "kind": item.kind.value,
        "intake": item.intake.value,
        "zone": item.zone.value,
        "party_tier": party.tier.value if party else PartyTier.BASIC.value,
        "operator_tenure_months": operator.tenure_months if operator else 0,
        "backlog_at_intake": item.backlog_at_intake,
        "rework_count": item.rework_count,
        "detail_length": len(item.detail),
    }


def describe_prediction(resp: MLExplainResponse, *, top_k: int = 3) -> str:
    """Render one ML prediction as decision-support text for the agent's reasoning.

    This is the **domain's** framing of the spine's output. The core injects the
    returned block into the planner and generate prompts, so the agent plans *with*
    the model (predict-then-act) and can explain itself from the model's actual
    drivers rather than from its own narration.

    TODO(domain): re-voice this. It is the sentence a jury reads. Left unedited it
    names the placeholder target and unit out loud in the middle of a demo — and
    because it reads :data:`TARGET`, the *numbers* will be right while the *words*
    are wrong, which is the version nobody notices in rehearsal.

    Args:
        resp: The spine response (prediction, conformal fields, SHAP attribution).
        top_k: How many top SHAP drivers to surface.

    Returns:
        A compact, human-readable multi-line summary safe to embed in a prompt.
    """
    unit = f" {TARGET.unit}" if TARGET.unit else ""
    if isinstance(resp.prediction, (int, float)):
        head = f"Predicted {TARGET.name}: {float(resp.prediction):.1f}{unit}"
    else:
        head = f"Predicted {TARGET.name}: {resp.prediction}"

    lines = [f"ML decision-support ({TARGET.task}):", f"- {head}"]
    if resp.conformal_interval is not None and resp.conformal_confidence is not None:
        low, high = resp.conformal_interval
        lines.append(
            f"- {resp.conformal_confidence:.0%} confidence interval "
            f"[{low:.1f}, {high:.1f}]{unit}"
        )
    elif resp.prediction_set_size is not None and resp.conformal_confidence is not None:
        lines.append(
            f"- {resp.conformal_confidence:.0%} conformal set size "
            f"{resp.prediction_set_size} (1 = confident)"
        )
    drivers = [
        f"{f.feature} ({'+' if f.contribution >= 0 else '−'}{abs(f.contribution):.2f})"
        for f in resp.shap_attribution[:top_k]
    ]
    if drivers:
        lines.append("- Top drivers (SHAP): " + ", ".join(drivers))
    lines.append(
        "TODO(domain): close with what the reader should DO with this number."
    )
    return "\n".join(lines)


def feature_matrix(dataset: SyntheticDataset) -> tuple[list[dict], list[float]]:
    """Build the training matrix ``(X, y)`` from a dataset's labelled records.

    Only records carrying a measured outcome contribute rows (see
    :meth:`~app.adapter.schema.SyntheticDataset.labelled_items`), each joined to its
    operator and party for the full feature set.

    Args:
        dataset: The synthetic world to draw rows from.

    Returns:
        A tuple ``(X, y)``: a list of feature dicts and the aligned list of targets.
    """
    features: list[dict] = []
    targets: list[float] = []
    for item in dataset.labelled_items():
        operator = (
            dataset.operator_by_id(item.assigned_operator_id)
            if item.assigned_operator_id
            else None
        )
        party = dataset.party_by_id(item.party_id)
        features.append(features_for_item(item, operator=operator, party=party))
        # labelled_items() guarantees the target is not None.
        targets.append(float(item.cycle_time_hours or 0.0))
    return features, targets


def training_frame(*, num_records: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Build the ML spine's labelled training frame from a fresh synthetic world.

    This is the ``frame_provider`` the spine resolves at (offline) train time: it
    generates a deterministic synthetic dataset **synchronously** — no LLM, no network,
    no event loop — and turns its labelled records into a ``DataFrame`` with one column
    per :data:`FEATURE_NAMES` plus the :data:`TARGET` column.

    ``pandas`` is imported **inside** the function on purpose: this module must stay
    importable with no ML stack installed (see the module docstring).

    Args:
        num_records: Records to synthesise (more ⇒ more labelled training rows). The
            keyword is deliberately domain-neutral: ``aegis.adapter.MLSpecModule``
            names it, so spelling it ``num_items`` here breaks the contract.
        seed: Seed for the synthetic world (a fixed seed pins the frame exactly).

    Returns:
        A labelled training frame; feature columns keep their native dtypes
        (categoricals are the enum ``.value`` strings, numerics are numbers).
    """
    import pandas as pd

    from app.adapter.generator import GeneratorConfig, generate_synthetic_sync

    dataset = generate_synthetic_sync(GeneratorConfig(seed=seed, num_items=num_records))
    rows, targets = feature_matrix(dataset)
    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    frame[TARGET.name] = targets
    return frame


__all__ = [
    "CATEGORICAL_FEATURES",
    "CONFOUNDER_WEIGHT",
    "FEATURES",
    "FEATURE_NAMES",
    "NUMERIC_FEATURES",
    "TARGET",
    "TARGET_R2",
    "FeatureSpec",
    "TargetSpec",
    "calibrated_noise_sigma",
    "describe_prediction",
    "feature_matrix",
    "features_for_item",
    "latent_cycle_time_hours",
    "training_frame",
]
