"""Piece 1 of 10 — the vocabulary: the entities and enums everything else shares.

WHAT YOU WRITE HERE
    The record types of the new world, as pydantic v2 models, and every categorical
    as a ``StrEnum`` rather than a free string. This module is edited FIRST because
    every other piece imports its names: ``ml_spec`` featurises these records,
    ``generator`` fabricates them, ``tools`` mutates them, ``corpus`` returns them.

    Concretely, from your Domain Brief:
      * one enum per categorical vocabulary (the ML spec's ``levels`` come from these);
      * one model per entity the domain talks about;
      * ONE central record that carries the ML features and the ML target;
      * a document type for the retrieval corpus;
      * ``DatasetMetadata`` + ``SyntheticDataset`` — the container piece 3 returns and
        piece 2 reads.

    Delete every placeholder below. The widget/work-item world here is a shape to
    fill, not a domain to keep: it exists so the file imports and so you can see how
    the enums, the feature-bearing fields and the target hang together.

THE CONTRACT (aegis.adapter.SchemaModule) — these names must survive
    SCHEMA_VERSION      (the only Protocol member; the platform passes records opaquely)
    SyntheticDataset    (not in SchemaModule, but SKILL.md names it explicitly: it is
                         the container generate_synthetic* returns and feature_matrix
                         reads, and adapter/__init__.py re-exports it)

THE TRAP
    The platform never introspects a domain record, so nothing here fails loudly when
    it is wrong — it fails three files later. Two specific ones:

    * A categorical spelled as ``str`` instead of a ``StrEnum`` means ``ml_spec``'s
      ``FeatureSpec.levels`` has nothing to derive from, and an unseen level reaches
      ``OneHotEncoder(handle_unknown="ignore")``, which encodes it to all-zeros and
      does not raise.
    * The ML target field must be **nullable** and populated only on finished records.
      A target that is always present makes every row "labelled", including rows whose
      outcome has not happened yet — which is target leakage dressed as a bigger
      training set.

    Mutation goes through ``model_copy(update=...)`` in ``tools.py``, never in-place,
    which is what lets every tool report ``changed`` honestly and hand back an inverse.

VERIFY
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        tests/adapter/test_schema.py -q)

Targeted API: pydantic ``2.x``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0-template"
"""Bumped whenever the record shapes change; embedded in generated datasets.

TODO(domain): set this to your own version and bump it on every shape change. It is
written onto :class:`DatasetMetadata`, so a corpus or a trained model produced against
an older shape can be told apart from one produced against this shape. The string
``-template`` is here so a forgotten edit is visible in an artifact.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Categorical vocabularies (shared by generator, tools, ml_spec, prompts)
#
# TODO(domain): replace every enum below. One enum per categorical the Brief names.
# The ``.value`` strings are what land in the feature frame, the tool schemas and the
# prompt, so write them as the domain would say them, lower_snake, and keep the set
# small — every level is a one-hot column.
# ─────────────────────────────────────────────────────────────────────────────


class UrgencyBand(StrEnum):
    """How urgently a work item must be finished (drives routing and the target)."""

    ROUTINE = "routine"
    ELEVATED = "elevated"
    URGENT = "urgent"


class WidgetKind(StrEnum):
    """The subject area of a work item; picks the responsible line and corpus slice."""

    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    DELTA = "delta"


class IntakePath(StrEnum):
    """How the work item arrived (a real operational driver, not decoration)."""

    BATCH = "batch"
    STREAM = "stream"
    MANUAL = "manual"


class Zone(StrEnum):
    """Coarse site/geography split (affects staffing windows and handover cost)."""

    ZONE_NORTH = "zone_north"
    ZONE_SOUTH = "zone_south"
    ZONE_EAST = "zone_east"


class PartyTier(StrEnum):
    """Commercial tier of the counterparty; higher tiers are handled sooner."""

    BASIC = "basic"
    PLUS = "plus"
    PRIME = "prime"


class WorkItemStage(StrEnum):
    """Lifecycle stage of a work item.

    TODO(domain): the two stages that mean "finished and measured" go in
    :data:`COMPLETED_STAGES` below — those are the only rows the ML spine may learn
    from.
    """

    RECEIVED = "received"
    QUEUED = "queued"
    IN_STAGE = "in_stage"
    HELD = "held"
    COMPLETED = "completed"
    CLOSED = "closed"
    REWORKED = "reworked"


class DocumentKind(StrEnum):
    """Type of knowledge document ingested into retrieval."""

    GUIDE = "guide"
    POLICY = "policy"
    FAQ = "faq"
    RUNBOOK = "runbook"


COMPLETED_STAGES: frozenset[WorkItemStage] = frozenset(
    {WorkItemStage.COMPLETED, WorkItemStage.CLOSED}
)
"""Stages that mean "finished and measured" — the rows carrying an ML label.

TODO(domain): name the stages at which your target is actually observed. A stage
included here whose target is still ``None`` simply produces an unlabelled row (see
:attr:`WorkItem.is_labelled`); a stage *omitted* here silently shrinks the training set.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Entities
#
# TODO(domain): replace all five. Keep them small, fully typed, and validated on
# construction — an invalid record must be impossible to build, so a malformed
# generator cannot leak downstream.
# ─────────────────────────────────────────────────────────────────────────────


class Party(BaseModel):
    """The counterparty a work item is raised for (customer / account / requester)."""

    id: str = Field(description="Stable party id, e.g. 'party-0001'.")
    name: str = Field(description="Display name of the party/organisation.")
    contact_email: str = Field(
        description="Primary contact address. MUST be a reserved '.example' address in "
        "generated data — the quality gate in piece 3 checks it."
    )
    zone: Zone
    tier: PartyTier
    onboarded_at: datetime = Field(description="When the party relationship opened.")


class Operator(BaseModel):
    """A person who works items (the 'assignee' of an action)."""

    id: str = Field(description="Stable operator id, e.g. 'op-007'.")
    name: str = Field(description="Display name of the operator.")
    line: str = Field(description="Team/line the operator belongs to, e.g. 'Line-2 Beta'.")
    tenure_months: int = Field(ge=0, description="Months of experience on the line.")
    zone: Zone
    skills: list[WidgetKind] = Field(
        default_factory=list, description="Kinds this operator handles best."
    )


class ItemNote(BaseModel):
    """A single note appended to a work item's timeline.

    Notes are keyed by a **deterministic** :attr:`id` derived from their content, so
    appending the same note twice is a no-op — that is what makes ``add_item_note`` in
    piece 4 idempotent. Keep this property in whatever you replace it with.
    """

    id: str = Field(description="Deterministic note id (dedupes identical notes).")
    author: str = Field(description="Persona or operator id that wrote the note.")
    body: str = Field(description="Free-text note content.")
    created_at: datetime = Field(description="When the note was recorded.")


class WorkItem(BaseModel):
    """The central record of the domain — the thing the agent reasons and acts on.

    TODO(domain): this is the model to get right. Three groups of fields, and the
    grouping matters:

    1. **Identity / lifecycle** — ids, stage, timestamps, foreign keys.
    2. **Feature-bearing operational fields** — every column ``ml_spec.FEATURES``
       names must be derivable from this record joined to its operator and party, and
       must be knowable *at prediction time*. A field only filled in after the outcome
       is a leak, not a feature.
    3. **The target** — nullable, populated only once the outcome is observed.
    """

    id: str = Field(description="Stable work-item id, e.g. 'item-000123'.")
    title: str = Field(description="Short human summary of the item.")
    detail: str = Field(description="Full free-text description of the item.")

    kind: WidgetKind
    urgency: UrgencyBand
    intake: IntakePath
    zone: Zone
    stage: WorkItemStage

    party_id: str = Field(description="FK → :class:`Party`.")
    assigned_operator_id: str | None = Field(
        default=None, description="FK → :class:`Operator`, or None if unassigned."
    )

    created_at: datetime = Field(description="When the item was opened.")
    updated_at: datetime = Field(description="Last time the record changed.")
    completed_at: datetime | None = Field(
        default=None, description="When it was completed/closed, else None."
    )

    # ── Feature-bearing operational fields ───────────────────────────────────
    # TODO(domain): every one of these must be observable BEFORE the target is.
    backlog_at_intake: int = Field(
        ge=0, description="Queue depth when the item arrived (the load signal)."
    )
    rework_count: int = Field(
        default=0, ge=0, description="How many times the item was sent back for rework."
    )
    first_touch_minutes: int | None = Field(
        default=None, ge=0, description="Time to first human touch, if measured."
    )
    target_cycle_hours: float = Field(
        gt=0, description="Contractual/agreed completion target for this item."
    )
    quality_score: int | None = Field(
        default=None, ge=1, le=5, description="Post-completion quality rating, if given."
    )

    # ── Target ───────────────────────────────────────────────────────────────
    cycle_time_hours: float | None = Field(
        default=None,
        ge=0,
        description="ML TARGET — wall-clock hours from intake to completion "
        "(None while the item is still open). TODO(domain): rename to your target and "
        "keep it nullable.",
    )

    tags: list[str] = Field(default_factory=list, description="Free-form labels.")
    notes: list[ItemNote] = Field(
        default_factory=list, description="Timeline of item notes."
    )

    @property
    def is_labelled(self) -> bool:
        """Whether this record carries a measured outcome (i.e. is ML-trainable)."""
        return self.stage in COMPLETED_STAGES and self.cycle_time_hours is not None


class Document(BaseModel):
    """A knowledge document for the retrieval corpus (graph/vector ingest).

    Conformance check #13 reads what ``corpus.load_seed_corpus`` returns and requires a
    unique, non-empty :attr:`id` and a chunkable :attr:`body` on every record. Keep both
    fields (or keep names the platform recognises) whatever else you change here.
    """

    id: str = Field(description="Stable document id, e.g. 'doc-seed-0001'.")
    kind: DocumentKind
    title: str = Field(description="Document title.")
    body: str = Field(description="Full document text to index.")
    kind_scope: WidgetKind | None = Field(
        default=None, description="Primary kind the doc pertains to, if any."
    )
    tags: list[str] = Field(default_factory=list, description="Retrieval tags.")
    source: str = Field(
        default="synthetic", description="Provenance, e.g. 'seed' or 'synthetic'."
    )


class DatasetMetadata(BaseModel):
    """Provenance for a generated dataset (what was made, how, and how learnable).

    TODO(domain): rename the counts to your entities. Keep ``schema_version``,
    ``seed``, ``llm_used`` and ``num_labelled`` — they are what makes a generated
    artifact traceable, and ``num_labelled`` is the number that tells you at a glance
    whether the training frame will be empty.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    seed: int | None = Field(default=None, description="RNG seed, if deterministic.")
    llm_used: bool = Field(description="Whether an LLM fabricated the text content.")
    num_parties: int
    num_operators: int
    num_items: int
    num_documents: int
    num_labelled: int = Field(description="Items carrying an ML target.")
    target_r2: float | None = Field(
        default=None,
        description="The held-out R² the label noise was CALIBRATED for (piece 2's "
        "TARGET_R2). Recorded so a model card can state the ceiling the data itself "
        "imposes, rather than leaving a 0.62 R² looking like an under-fit.",
    )
    noise_sigma: float | None = Field(
        default=None, description="The std-dev actually applied to the latent signal."
    )


class SyntheticDataset(BaseModel):
    """A complete, typed synthetic world ready to seed records + retrieval.

    **Keep this class name.** ``SKILL.md`` names it as one of the three things that
    must survive a retarget by name: the generator returns it, ``ml_spec.feature_matrix``
    reads it, ``adapter/__init__.py`` re-exports it, and the host's demo/seed paths
    bind to it. Change every field inside it; do not rename the container.

    TODO(domain): rename the collections to your entities and keep the three accessor
    methods pointed at them. ``labelled_*`` is the one piece 2 depends on.
    """

    metadata: DatasetMetadata
    parties: list[Party]
    operators: list[Operator]
    items: list[WorkItem]
    documents: list[Document]

    def party_by_id(self, party_id: str) -> Party | None:
        """Return the party with ``party_id`` (or None if absent)."""
        return next((p for p in self.parties if p.id == party_id), None)

    def operator_by_id(self, operator_id: str) -> Operator | None:
        """Return the operator with ``operator_id`` (or None if absent)."""
        return next((o for o in self.operators if o.id == operator_id), None)

    def labelled_items(self) -> list[WorkItem]:
        """Return only the records that carry an ML target (the training rows)."""
        return [i for i in self.items if i.is_labelled]


__all__ = [
    "COMPLETED_STAGES",
    "SCHEMA_VERSION",
    "DatasetMetadata",
    "Document",
    "DocumentKind",
    "IntakePath",
    "ItemNote",
    "Operator",
    "Party",
    "PartyTier",
    "SyntheticDataset",
    "UrgencyBand",
    "WidgetKind",
    "WorkItem",
    "WorkItemStage",
    "Zone",
]
