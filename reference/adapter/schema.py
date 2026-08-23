"""Piece 1 of 10 — the vocabulary: the entities and enums everything else shares.

WHAT THIS FILE IS
    The record types of the pharmaceutical cold-chain world, as pydantic v2 models, with
    every categorical spelled as a ``StrEnum`` rather than a free string. This module is
    written FIRST because every other piece imports its names: :mod:`~reference.adapter.ml_spec`
    featurises these records, :mod:`~reference.adapter.generator` fabricates them,
    :mod:`~reference.adapter.tools` mutates them, :mod:`~reference.adapter.corpus` returns them.

    The world: a pharmaceutical distributor moves temperature-controlled shipments —
    vaccines, biologics, diagnostic kits — from origin depots through transfer hubs to
    clinics and cold stores. Each shipment rides with a :class:`Carrier` under a declared
    :class:`PackagingType`, is handed off some number of times, and is instrumented with
    data loggers whose readings arrive as :class:`SensorReading` records.

THE CONTRACT (aegis.adapter.SchemaModule) — these names must survive
    SCHEMA_VERSION      (the only Protocol member; the platform passes records opaquely)
    SyntheticDataset    (not in SchemaModule, but named explicitly by the retargeting
                         procedure: it is the container ``generate_synthetic*`` returns,
                         ``feature_matrix`` reads, and ``adapter/__init__.py`` re-exports)

THE TRAP
    The platform never introspects a domain record, so nothing here fails loudly when it
    is wrong — it fails three files later. Two specific ones, both live in this file:

    * A categorical spelled as ``str`` instead of a ``StrEnum`` means ``ml_spec``'s
      ``FeatureSpec.levels`` has nothing to derive from, and an unseen level reaches
      ``OneHotEncoder(handle_unknown="ignore")``, which encodes it to all-zeros and does
      not raise.
    * Both ML targets — :attr:`Shipment.spoilage_risk_pct` and
      :attr:`Shipment.excursion_flag` — are **nullable** and populated only once the
      shipment has been received and assayed. A target that is always present makes every
      row "labelled", including shipments still in the air whose outcome has not happened
      yet, which is target leakage dressed as a bigger training set.

    Mutation goes through ``model_copy(update=...)`` in :mod:`~reference.adapter.tools`,
    never in place, which is what lets every tool report ``changed`` honestly and hand
    back an inverse.

Targeted API: pydantic ``2.x``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = [
    "DELIVERED_STAGES",
    "SCHEMA_VERSION",
    "Carrier",
    "CarrierTier",
    "DatasetMetadata",
    "Document",
    "DocumentKind",
    "ExcursionFlag",
    "Facility",
    "FacilityKind",
    "OriginRegion",
    "PackagingType",
    "ProductClass",
    "RouteClass",
    "SensorReading",
    "Shipment",
    "ShipmentNote",
    "ShipmentStage",
    "SyntheticDataset",
]

SCHEMA_VERSION = "1.0.0"
"""Bumped whenever the record shapes change; embedded in every generated dataset.

Written onto :class:`DatasetMetadata`, so a corpus or a trained model produced against an
older shape of this domain can be told apart from one produced against this shape.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Categorical vocabularies (shared by generator, tools, ml_spec, prompts)
#
# Every ``.value`` string below lands in the feature frame, in the tool schemas and in the
# rendered prompt, so they are written as a cold-chain planner would say them: lower_snake,
# small sets, no synonyms.
# ─────────────────────────────────────────────────────────────────────────────


class CarrierTier(StrEnum):
    """Service tier a carrier is contracted at.

    Ascending order of cold-chain discipline. ``VALIDATED`` means the lane has been
    qualified end-to-end under GDP (Good Distribution Practice) with documented
    temperature mapping; ``ECONOMY`` means a general-freight carrier with a cold box on
    board and no lane qualification at all.
    """

    ECONOMY = "economy"
    STANDARD = "standard"
    PREMIUM = "premium"
    VALIDATED = "validated"


class RouteClass(StrEnum):
    """Shape of the physical journey — how many times custody changes hands.

    The single largest structural driver of cold-chain exposure: every transfer is a door
    opening, a tarmac wait and a chance to be left on a dock.
    """

    DIRECT = "direct"
    SINGLE_TRANSFER = "single_transfer"
    MULTI_LEG = "multi_leg"
    LAST_MILE_POOL = "last_mile_pool"


class PackagingType(StrEnum):
    """The thermal system protecting the payload.

    ``PASSIVE_GEL`` is a gel-pack shipper with a short qualified duration; ``PASSIVE_PCM``
    uses phase-change material and holds far longer; ``ACTIVE_ELECTRIC`` is a powered
    reefer container; ``DRY_ICE`` is sublimation cooling for frozen product.
    """

    PASSIVE_GEL = "passive_gel"
    PASSIVE_PCM = "passive_pcm"
    ACTIVE_ELECTRIC = "active_electric"
    DRY_ICE = "dry_ice"


class OriginRegion(StrEnum):
    """Coarse geography the shipment was dispatched from.

    Carried on every shipment because operations reports are cut by it — **not** because
    it predicts spoilage. See :data:`~reference.adapter.ml_spec.IRRELEVANT_FEATURES`: this
    is one of the two columns the generator deliberately draws independently of the target,
    so a well-behaved SHAP report shows it near zero.
    """

    EMEA = "emea"
    AMER = "amer"
    APAC = "apac"
    LATAM = "latam"


class ProductClass(StrEnum):
    """What is inside, at the granularity that changes thermal sensitivity."""

    VACCINE = "vaccine"
    BIOLOGIC = "biologic"
    SMALL_MOLECULE = "small_molecule"
    DIAGNOSTIC_KIT = "diagnostic_kit"


class ShipmentStage(StrEnum):
    """Lifecycle stage of a shipment.

    The two stages that mean "received and assayed" are in :data:`DELIVERED_STAGES` —
    those are the only rows the ML spine may learn from, because they are the only ones on
    which spoilage risk has actually been measured.
    """

    BOOKED = "booked"
    DISPATCHED = "dispatched"
    IN_TRANSIT = "in_transit"
    HELD_AT_HUB = "held_at_hub"
    QUARANTINED = "quarantined"
    DELIVERED = "delivered"
    RELEASED = "released"


class ExcursionFlag(StrEnum):
    """Whether the shipment left its qualified temperature range at any point.

    The domain's **secondary, classification** target. Deliberately two-valued and
    deliberately imbalanced in generated data (see
    :data:`~reference.adapter.generator.EXCURSION_SHARE`): most shipments arrive clean, and
    a classifier that beats the majority-class rate has genuinely learned something.
    """

    NO_EXCURSION = "no_excursion"
    EXCURSION = "excursion"


class FacilityKind(StrEnum):
    """What a facility does in the network."""

    ORIGIN_DEPOT = "origin_depot"
    TRANSFER_HUB = "transfer_hub"
    COLD_STORE = "cold_store"
    CLINIC = "clinic"


class DocumentKind(StrEnum):
    """Type of knowledge document ingested into retrieval."""

    SOP = "sop"
    POLICY = "policy"
    RUNBOOK = "runbook"
    FAQ = "faq"


DELIVERED_STAGES: frozenset[ShipmentStage] = frozenset(
    {ShipmentStage.DELIVERED, ShipmentStage.RELEASED}
)
"""Stages at which the outcome is observed — the rows carrying an ML label.

A shipment is only scored once it has been physically received and the receiving site has
run its stability assessment. ``QUARANTINED`` is deliberately absent: a quarantined
shipment has a *suspected* problem and no assay yet, so including it would train the model
on the operator's suspicion rather than on the measured outcome.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Entities
# ─────────────────────────────────────────────────────────────────────────────


class Carrier(BaseModel):
    """A logistics provider moving shipments under contract."""

    id: str = Field(description="Stable carrier id, e.g. 'carrier-003'.")
    name: str = Field(description="Display name of the carrier.")
    tier: CarrierTier = Field(description="Contracted service tier.")
    gdp_certified: bool = Field(
        description="Whether the carrier holds a current GDP certificate. Reported to "
        "auditors; not an ML feature, because tier already carries the signal."
    )
    on_time_rate: float = Field(
        ge=0.0, le=1.0, description="Rolling share of shipments delivered inside window."
    )
    hub_region: OriginRegion = Field(description="Region the carrier's main hub sits in.")


class Facility(BaseModel):
    """A physical site a shipment departs from, passes through, or arrives at."""

    id: str = Field(description="Stable facility id, e.g. 'fac-0007'.")
    name: str = Field(description="Display name of the site.")
    kind: FacilityKind = Field(description="What the site does in the network.")
    region: OriginRegion = Field(description="Coarse geography of the site.")
    has_backup_power: bool = Field(
        description="Whether cold rooms here survive a mains failure. Operationally "
        "important and reported in audits; not a shipment-level feature."
    )


class SensorReading(BaseModel):
    """One data-logger reading taken from a shipment in transit.

    Readings are what an excursion is ultimately *proved* from, and they are what a
    quality auditor asks to see. They are deliberately **not** featurised: a reading taken
    at hour 40 of a 60-hour journey is not knowable at booking time, so using it would be
    leakage. The gap *between* readings is knowable and is a feature — see
    :attr:`Shipment.sensor_gap_minutes`.
    """

    id: str = Field(description="Stable reading id, e.g. 'read-000412'.")
    shipment_id: str = Field(description="FK → :class:`Shipment`.")
    recorded_at: datetime = Field(description="When the logger took the sample.")
    temperature_c: float = Field(description="Payload temperature in degrees Celsius.")
    battery_pct: int = Field(ge=0, le=100, description="Logger battery remaining.")


class ShipmentNote(BaseModel):
    """A single note appended to a shipment's timeline.

    Notes carry a **deterministic** :attr:`id` derived from their content, so appending
    the same note twice is a no-op — that is what makes
    :func:`~reference.adapter.tools.add_shipment_note` genuinely idempotent rather than
    merely declared so.
    """

    id: str = Field(description="Deterministic note id (dedupes identical notes).")
    author: str = Field(description="Persona or operator id that wrote the note.")
    body: str = Field(description="Free-text note content.")
    created_at: datetime = Field(description="When the note was recorded.")


class Shipment(BaseModel):
    """The central record of the domain — the thing the agent reasons and acts on.

    Three groups of fields, and the grouping is what keeps the ML honest:

    1. **Identity / lifecycle** — ids, stage, timestamps, foreign keys.
    2. **Feature-bearing operational fields** — every column
       :data:`~reference.adapter.ml_spec.FEATURE_NAMES` names is derivable from this record
       joined to its carrier and origin facility, and every one of them is knowable at
       **booking time**, before the shipment moves. A field only filled in after arrival is
       a leak, not a feature.
    3. **The targets** — both nullable, both populated only once the shipment has been
       received and assayed.
    """

    id: str = Field(description="Stable shipment id, e.g. 'ship-000123'.")
    reference: str = Field(description="Customer-facing consignment reference.")
    summary: str = Field(description="Short human summary of the consignment.")
    detail: str = Field(description="Full free-text description of the consignment.")

    stage: ShipmentStage = Field(description="Where the shipment is in its lifecycle.")
    route_class: RouteClass = Field(description="Shape of the physical journey.")
    packaging_type: PackagingType = Field(description="Thermal system protecting the payload.")
    product_class: ProductClass = Field(description="What is inside.")

    carrier_id: str = Field(description="FK → :class:`Carrier`.")
    origin_facility_id: str = Field(description="FK → :class:`Facility` (departure site).")
    destination_facility_id: str = Field(description="FK → :class:`Facility` (arrival site).")
    shipper_id: str = Field(
        description="Owning shipper account. This is the field the OWN data scope "
        "narrows on — see reference.adapter.personas.SHIPPER_CLIENT."
    )

    booked_at: datetime = Field(description="When the consignment was booked.")
    dispatched_at: datetime = Field(description="When it physically left the origin site.")
    updated_at: datetime = Field(description="Last time the record changed.")
    delivered_at: datetime | None = Field(
        default=None, description="When it was received, else None."
    )

    # ── Feature-bearing operational fields (all knowable at booking time) ─────
    transit_hours: float = Field(
        gt=0.0, description="Planned door-to-door duration of the journey, in hours."
    )
    ambient_temp_c: float = Field(
        description="Forecast mean ambient temperature along the lane, in Celsius."
    )
    handoff_count: int = Field(
        ge=0, description="Planned custody transfers between origin and destination."
    )
    payload_kg: float = Field(gt=0.0, description="Gross weight of the consignment.")
    sensor_gap_minutes: float | None = Field(
        default=None,
        ge=0.0,
        description="Contracted interval between data-logger transmissions. Nullable: "
        "economy carriers frequently do not publish a telemetry interval at all, which "
        "is the domain's MAR missingness (see reference.problem.LATENT).",
    )

    # ── Targets ──────────────────────────────────────────────────────────────
    spoilage_risk_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="PRIMARY ML TARGET — assessed probability, in percent, that the "
        "consignment has lost potency and must be written off. None until the receiving "
        "site has run its stability assessment.",
    )
    excursion_flag: ExcursionFlag | None = Field(
        default=None,
        description="SECONDARY ML TARGET — whether the logger record shows the payload "
        "left its qualified range at any point. None until the loggers are downloaded.",
    )

    tags: list[str] = Field(default_factory=list, description="Free-form labels.")
    notes: list[ShipmentNote] = Field(
        default_factory=list, description="Timeline of shipment notes."
    )

    @property
    def is_labelled(self) -> bool:
        """Whether this shipment carries a measured outcome (i.e. is ML-trainable)."""
        return self.stage in DELIVERED_STAGES and self.spoilage_risk_pct is not None


class Document(BaseModel):
    """A knowledge document for the retrieval corpus (graph/vector ingest).

    The conformance suite reads what :func:`~reference.adapter.corpus.load_seed_corpus`
    returns and requires a unique, non-empty :attr:`id` and a chunkable :attr:`body` on
    every record: chunks are written with the record's id as their ``doc_id``, so a record
    with no id — or two records sharing one — produces chunks whose citation resolves to
    the wrong document or to nothing at all.
    """

    id: str = Field(description="Stable document id, e.g. 'doc-seed-0001'.")
    kind: DocumentKind = Field(description="What sort of document this is.")
    title: str = Field(description="Document title.")
    body: str = Field(description="Full document text to index.")
    product_scope: ProductClass | None = Field(
        default=None, description="Primary product class the doc pertains to, if any."
    )
    tags: list[str] = Field(default_factory=list, description="Retrieval tags.")
    source: str = Field(
        default="synthetic", description="Provenance, e.g. 'seed' or 'synthetic'."
    )


class DatasetMetadata(BaseModel):
    """Provenance for a generated dataset — what was made, how, and how learnable.

    ``target_r2``, ``noise_sigma`` and ``excursion_share`` are here so a model card can
    state the ceiling the *data itself* imposes. Without them a held-out R² of 0.66 reads
    as an under-fit; with them it reads as 95% of everything attainable.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    seed: int | None = Field(default=None, description="RNG seed, if deterministic.")
    llm_used: bool = Field(description="Whether an LLM fabricated the text content.")
    num_carriers: int = Field(ge=0, description="Carriers in the world.")
    num_facilities: int = Field(ge=0, description="Facilities in the world.")
    num_shipments: int = Field(ge=0, description="Shipments generated.")
    num_sensor_readings: int = Field(ge=0, description="Data-logger readings generated.")
    num_documents: int = Field(ge=0, description="Knowledge documents generated.")
    num_labelled: int = Field(ge=0, description="Shipments carrying an ML target.")
    target_r2: float | None = Field(
        default=None,
        description="The held-out R² the label noise was CALIBRATED for. Recorded so a "
        "model card can state the ceiling the data imposes.",
    )
    noise_sigma: float | None = Field(
        default=None, description="Std-dev of the i.i.d. noise actually applied, in percent."
    )
    confounder_sigma: float | None = Field(
        default=None,
        description="Std-dev of the combined UNOBSERVED drivers actually applied. Never a "
        "column; this is the structured half of the irreducible error.",
    )
    excursion_share: float | None = Field(
        default=None, description="Realised share of labelled shipments flagged as excursions."
    )
    missing_sensor_gap_share: float | None = Field(
        default=None, description="Realised share of labelled rows with no telemetry interval."
    )


class SyntheticDataset(BaseModel):
    """A complete, typed synthetic world ready to seed records + retrieval.

    **Keep this class name.** It is one of the three things the retargeting procedure names
    as surviving by name: the generator returns it, ``ml_spec.feature_matrix`` reads it,
    ``adapter/__init__.py`` re-exports it, and the host's demo/seed paths bind to it.
    """

    metadata: DatasetMetadata
    carriers: list[Carrier]
    facilities: list[Facility]
    shipments: list[Shipment]
    readings: list[SensorReading]
    documents: list[Document]

    def carrier_by_id(self, carrier_id: str) -> Carrier | None:
        """Return the carrier with ``carrier_id`` (or None if absent)."""
        return next((c for c in self.carriers if c.id == carrier_id), None)

    def facility_by_id(self, facility_id: str) -> Facility | None:
        """Return the facility with ``facility_id`` (or None if absent)."""
        return next((f for f in self.facilities if f.id == facility_id), None)

    def readings_for(self, shipment_id: str) -> list[SensorReading]:
        """Return every logger reading recorded against ``shipment_id``, oldest first."""
        matched = [r for r in self.readings if r.shipment_id == shipment_id]
        return sorted(matched, key=lambda r: r.recorded_at)

    def labelled_shipments(self) -> list[Shipment]:
        """Return only the shipments that carry an ML target (the training rows)."""
        return [s for s in self.shipments if s.is_labelled]
