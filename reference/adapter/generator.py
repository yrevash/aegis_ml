"""Piece 3 of 10 — the synthetic world: the demo's data, and the ML spine's training set.

WHAT THIS FILE IS
    A **hybrid** generator, which is the pattern that makes label-consistent synthetic data
    cheap. Three layers, and all three matter:

    1. **Procedural, seeded structure.** Every feature-bearing field is drawn from a seeded
       :class:`random.Random`, so a fixed ``seed`` pins the whole world exactly.
    2. **LLM-fabricated text.** Only the prose — consignment summaries, details, corpus
       documents — comes from the model gateway, requested by **role** (a cheap model for
       bulk record text, a generation model for documents), never by a hard-coded model id,
       and parsed defensively.
    3. **Graceful degradation.** With no LLM available (or a malformed response) the
       generator falls back to deterministic templated text and *still* returns schema-valid
       data with identical structure and identical labels. That is what makes the system
       demonstrable while a model key is still being sorted out — and it is the path every
       test, every offline run and every training frame actually takes.

    Also here, because it is domain content: the **client-facing demand series**
    ``/forecast`` charts.

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
║ inlined here.                                                                 ║
║                                                                              ║
║   right:  mean  = ml_spec.latent_spoilage_risk(features, confounder=u)        ║
║           label = mean + sigma * multiplier * z                               ║
║                                                                              ║
║   wrong:  label = rng.uniform(0, 100)         # a plausible-looking number    ║
║   wrong:  label = 34 + 8.0 * tanh(...) + ...  # the formula, typed twice      ║
║                                                                              ║
║ If the label is not a function of the features, the target is noise: R² ≈ 0,  ║
║ the conformal interval is honestly enormous, SHAP has nothing to attribute,   ║
║ and the agent's "ML decision-support" block is a random number in a           ║
║ confident sentence. Every structural conformance check still passes.          ║
║                                                                               ║
║ The other half of the trap is noise that is too SMALL. This module measures   ║
║ the variance of the latent values it just computed and derives sigma from it  ║
║ through ``ml_spec.noise_budget``, so the achievable R² lands at               ║
║ ``ml_spec.TARGET_R2`` (0.74) instead of at 0.99. A hardcoded ``noise_scale``  ║
║ would be correct exactly until the next coefficient edit, and then silently   ║
║ stop being.                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

FIVE REALISM DEVICES, ALL APPLIED HERE
    Calibrated σ · two unobserved confounders · heteroscedastic noise on lane length ·
    MAR missingness on ``sensor_gap_minutes`` conditioned on ``carrier_tier`` ·
    boundary label flips on ``excursion_flag``. Each one is a reason the model cannot reach
    a suspicious score, and each one gives a downstream Aegis feature something real to
    report. They are described where they are applied, in :func:`_build_shipments`.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from reference.adapter import ml_spec
from reference.adapter.schema import (
    Carrier,
    CarrierTier,
    DatasetMetadata,
    Document,
    DocumentKind,
    ExcursionFlag,
    Facility,
    FacilityKind,
    OriginRegion,
    PackagingType,
    ProductClass,
    RouteClass,
    SensorReading,
    Shipment,
    ShipmentStage,
    SyntheticDataset,
)

__all__ = [
    "DOMAIN_SERIES_LABEL",
    "DOMAIN_SERIES_UNIT",
    "CompleteFn",
    "DatasetQualityReport",
    "GeneratorConfig",
    "ModelRole",
    "assess_quality",
    "domain_series_events",
    "generate_synthetic",
    "generate_synthetic_sync",
]

_EPOCH = datetime(2025, 6, 2, 8, 0, 0)
"""Base instant for deterministic timestamps, fixed so a seed pins the world.

Far enough in the past that every generated shipment predates "now": a record dated in the
future makes the lookup tool's age negative and makes the forecast series end after today.
"""


class ModelRole(StrEnum):
    """Which *job* an LLM call is billed to — a role, never a model id.

    Values match ``aegis.core.models.ModelRole`` exactly, and both are ``StrEnum``, so a
    host that passes its own member into :class:`CompleteFn` and this module's own member
    are indistinguishable at the gateway. Declared locally rather than imported so this
    package remains importable with no Aegis checkout present, which is what lets the
    reference domain be run, tested and audited on its own.
    """

    CHEAP = "cheap"
    REASONING = "reasoning"
    GENERATION = "generation"
    EMBEDDING = "embedding"


# ─────────────────────────────────────────────────────────────────────────────
# Injected LLM contract (structural — no hard import of the host's gateway)
# ─────────────────────────────────────────────────────────────────────────────


class _LLMResultLike(Protocol):
    """Structural view of the host's LLM result (only ``.content`` is used)."""

    content: str


class CompleteFn(Protocol):
    """The subset of the host's ``complete`` this generator depends on."""

    async def __call__(
        self,
        role: ModelRole,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> _LLMResultLike:
        """Complete a chat request for the given model role."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class GeneratorConfig(BaseModel):
    """Config-driven knobs for one generation run.

    Every count is a **positive integer field** with a working default, because callers
    construct ``GeneratorConfig()`` bare and the host's demo graph scales the world by
    introspecting this model's integer fields.

    The realism knobs default to the values ``ml_spec`` declares, so the generator and the
    spec cannot describe different worlds without someone typing an override on purpose.
    """

    num_carriers: int = Field(default=9, ge=1, description="Carriers in the network.")
    num_facilities: int = Field(default=14, ge=2, description="Sites in the network.")
    num_shipments: int = Field(default=40, ge=1, description="Consignments to fabricate.")
    num_sensor_readings: int = Field(
        default=120, ge=0, description="Data-logger readings to fabricate across shipments."
    )
    num_documents: int = Field(default=6, ge=0, description="Knowledge documents.")
    delivered_fraction: float = Field(
        default=0.78,
        ge=0.0,
        le=1.0,
        description="Share of shipments already received and assayed (i.e. ML-labelled).",
    )
    seed: int | None = Field(
        default=None, description="RNG seed; set for a fully reproducible structure."
    )
    target_r2: float = Field(
        default=ml_spec.TARGET_R2,
        gt=0.0,
        lt=1.0,
        description="Held-out R² the spoilage-risk noise is calibrated for. See the "
        "module's trap block: this is what keeps the target learnable but not trivial.",
    )
    excursion_signal_r2: float = Field(
        default=ml_spec.EXCURSION_SIGNAL_R2,
        gt=0.0,
        lt=1.0,
        description="Signal fidelity of the score the excursion class boundary is cut "
        "from. Not the achieved accuracy — the cut and the boundary flips both cost some "
        "of it — but the knob that puts the achieved accuracy inside its band.",
    )
    confounder_share: float = Field(
        default=ml_spec.CONFOUNDER_SHARE,
        ge=0.0,
        lt=1.0,
        description="Fraction of the irreducible error carried by the two UNOBSERVED "
        "drivers rather than by i.i.d. measurement noise.",
    )
    noise_scale: float | None = Field(
        default=None,
        ge=0.0,
        description="Explicit total irreducible std-dev, in percentage points. Leave None "
        "(the default) to DERIVE it from target_r2 and the measured variance of the latent "
        "signal — the derived value stays correct when an ml_spec coefficient changes, and "
        "a hardcoded one silently stops being correct.",
    )
    missing_gap_base_rate: float = Field(
        default=ml_spec.MISSING_GAP_BASE_RATE,
        ge=0.0,
        le=1.0,
        description="Missingness rate for sensor_gap_minutes on non-economy carriers.",
    )
    missing_gap_peak_rate: float = Field(
        default=ml_spec.MISSING_GAP_PEAK_RATE,
        ge=0.0,
        le=1.0,
        description="Missingness rate for sensor_gap_minutes on economy carriers. Higher "
        "than the base rate is what makes the holes MAR rather than MCAR.",
    )
    label_flip_rate: float = Field(
        default=ml_spec.LABEL_FLIP_RATE,
        ge=0.0,
        le=0.4,
        description="Share of excursion labels corrupted, drawn from the rows nearest the "
        "class boundary — measurement error where measurement error actually happens.",
    )
    use_llm: bool = Field(
        default=True, description="If False, skip the LLM and use templated text."
    )
    llm_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature for fabricated text."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────


async def generate_synthetic(
    config: GeneratorConfig | None = None,
    *,
    complete: CompleteFn | None = None,
) -> SyntheticDataset:
    """Fabricate a complete, schema-valid cold-chain world (optionally with LLM prose).

    Args:
        config: Generation knobs; defaults to :class:`GeneratorConfig` defaults.
        complete: The LLM completion function (dependency injection). ``None`` with
            ``config.use_llm`` set means "no gateway is wired in", and the templated path
            runs — this module never reaches for a network client on its own.

    Returns:
        A :class:`~reference.adapter.schema.SyntheticDataset` whose shipments seed the data
        layer and whose documents seed retrieval. Every record is pydantic-validated on
        construction, so a malformed draw cannot leak downstream.
    """
    cfg = config or GeneratorConfig()
    rng = random.Random(cfg.seed)

    resolved = complete if cfg.use_llm else None

    carriers = _build_carriers(rng, cfg.num_carriers)
    facilities = _build_facilities(rng, cfg.num_facilities)

    shipment_text = await _fabricate_shipment_text(resolved, rng, cfg)
    documents = await _fabricate_documents(resolved, rng, cfg)

    return _assemble(
        cfg,
        rng,
        carriers,
        facilities,
        shipment_text,
        documents,
        llm_used=resolved is not None,
    )


def generate_synthetic_sync(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Fabricate the cold-chain world **synchronously**, with deterministic templated text.

    Identical structure and identical labels to :func:`generate_synthetic`, but with no LLM
    and no ``await`` — so it is safe to call from synchronous code *and* from inside a
    running event loop, where ``asyncio.run`` raises. This is what seeds the process-wide
    record store and what :func:`reference.adapter.ml_spec.training_frame` calls: neither
    needs LLM-written prose, only schema-valid records whose label is the real latent
    function of the features.

    Args:
        config: Generation knobs; ``use_llm`` is forced off. Defaults apply otherwise.

    Returns:
        A schema-valid dataset, byte-for-byte reproducible under a fixed seed.
    """
    cfg = (config or GeneratorConfig()).model_copy(update={"use_llm": False})
    rng = random.Random(cfg.seed)

    carriers = _build_carriers(rng, cfg.num_carriers)
    facilities = _build_facilities(rng, cfg.num_facilities)
    shipment_text = _template_shipment_pool(rng, cfg)
    documents = _template_documents(rng, cfg)

    return _assemble(cfg, rng, carriers, facilities, shipment_text, documents, llm_used=False)


def _assemble(
    cfg: GeneratorConfig,
    rng: random.Random,
    carriers: list[Carrier],
    facilities: list[Facility],
    shipment_text: dict[ProductClass, list[dict[str, Any]]],
    documents: list[Document],
    *,
    llm_used: bool,
) -> SyntheticDataset:
    """Assemble records + metadata into a dataset (the shared sync/async core).

    Args:
        cfg: The generation knobs.
        rng: The seeded generator, already advanced past carrier/facility construction.
        carriers: The carriers of this world.
        facilities: The sites of this world.
        shipment_text: Per-product-class pools of ``{"summary", "detail"}`` entries.
        documents: The knowledge corpus.
        llm_used: Whether the prose came from a model.

    Returns:
        The assembled :class:`~reference.adapter.schema.SyntheticDataset`.
    """
    shipments, calibration = _build_shipments(rng, cfg, carriers, facilities, shipment_text)
    readings = _build_readings(rng, cfg, shipments)
    labelled = [s for s in shipments if s.is_labelled]
    excursions = sum(1 for s in labelled if s.excursion_flag is ExcursionFlag.EXCURSION)
    missing_gap = sum(1 for s in labelled if s.sensor_gap_minutes is None)

    metadata = DatasetMetadata(
        seed=cfg.seed,
        llm_used=llm_used,
        num_carriers=len(carriers),
        num_facilities=len(facilities),
        num_shipments=len(shipments),
        num_sensor_readings=len(readings),
        num_documents=len(documents),
        num_labelled=len(labelled),
        target_r2=cfg.target_r2,
        noise_sigma=round(calibration["noise_sigma"], 4),
        confounder_sigma=round(calibration["confounder_sigma"], 4),
        excursion_share=round(excursions / len(labelled), 4) if labelled else None,
        missing_sensor_gap_share=round(missing_gap / len(labelled), 4) if labelled else None,
    )
    return SyntheticDataset(
        metadata=metadata,
        carriers=carriers,
        facilities=facilities,
        shipments=shipments,
        readings=readings,
        documents=documents,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Structural (procedural, seeded) generation
#
# Every draw goes through ``rng``, never ``random.*`` at module level, or the
# "deterministic under a fixed seed" promise quietly stops being true.
# ─────────────────────────────────────────────────────────────────────────────

_CARRIER_STEMS = (
    "Polarlane",
    "Cryomar",
    "Nordkyl",
    "Aeris Cold",
    "Vitrex Freight",
    "Glacier Reach",
    "Thermolink",
    "Sable Cold Chain",
    "Meridian Chill",
)
_SITE_STEMS = (
    "Harbour",
    "Fennel Street",
    "Northgate",
    "Rivermouth",
    "Kestrel Park",
    "Old Mill",
    "Sandford",
    "Beacon Hill",
    "Longacre",
    "Wexley",
    "Ashgrove",
    "Trentham",
    "Ivybridge",
    "Caldera",
)

_CARRIER_TIER_WEIGHTS: tuple[int, ...] = (3, 4, 2, 1)
"""Book share per :class:`CarrierTier`, in enum order — economy is common, validated rare."""

_ROUTE_WEIGHTS: tuple[int, ...] = (3, 4, 2, 2)
"""Book share per :class:`RouteClass`, in enum order."""

_REGION_WEIGHTS: tuple[int, ...] = (4, 3, 3, 2)
"""Book share per :class:`OriginRegion`, in enum order."""

_PRODUCT_WEIGHTS: tuple[int, ...] = (3, 3, 3, 2)
"""Book share per :class:`ProductClass`, in enum order."""

_PACKAGING_BY_PRODUCT: dict[ProductClass, tuple[int, ...]] = {
    # Weights per PackagingType, in enum order:
    #   passive_gel, passive_pcm, active_electric, dry_ice
    ProductClass.VACCINE: (2, 4, 3, 2),
    ProductClass.BIOLOGIC: (2, 3, 3, 3),
    ProductClass.SMALL_MOLECULE: (4, 3, 2, 1),
    ProductClass.DIAGNOSTIC_KIT: (5, 3, 1, 1),
}
"""Which thermal system a product class usually ships under.

A mild, honest correlation between two genuine drivers — nobody packs a frozen biologic in
gel packs, and nobody puts a diagnostic kit in a powered reefer. Correlation *between
drivers* is realism; correlation between a driver and one of
``ml_spec.IRRELEVANT_FEATURES`` would be a bug, which is why ``origin_region`` and
``payload_kg`` are drawn independently of everything.
"""

_ROUTE_PROFILE: dict[RouteClass, tuple[float, float, int, int]] = {
    # route -> (transit_hours low, transit_hours high, handoff low, handoff high)
    RouteClass.DIRECT: (8.0, 36.0, 0, 1),
    RouteClass.SINGLE_TRANSFER: (18.0, 64.0, 1, 2),
    RouteClass.MULTI_LEG: (48.0, 132.0, 3, 7),
    RouteClass.LAST_MILE_POOL: (12.0, 54.0, 2, 4),
}
"""Physical envelope of each journey shape: duration range and custody-transfer range."""

_GAP_SCALE_BY_TIER: dict[CarrierTier, float] = {
    CarrierTier.ECONOMY: 70.0,
    CarrierTier.STANDARD: 50.0,
    CarrierTier.PREMIUM: 35.0,
    CarrierTier.VALIDATED: 25.0,
}
"""Gamma scale for the contracted telemetry interval, by carrier tier (minutes)."""

_TRANSIT_MIN, _TRANSIT_MAX = 6.0, 132.0
_AMBIENT_MIN, _AMBIENT_MODE, _AMBIENT_MAX = -4.0, 18.0, 40.0
_PAYLOAD_MIN, _PAYLOAD_MAX = 5.0, 900.0
_GAP_MAX = 540.0
"""Hard envelopes, matching the ``minimum``/``maximum`` declared on ``ml_spec.FEATURES``.

They are repeated as module constants rather than read back off the specs so the clamps
read plainly here; the data contract check is what proves the two agree, and it runs on
every ``data_flow``.
"""


def _weighted(rng: random.Random, members: list[Any], weights: tuple[int, ...]) -> Any:  # noqa: ANN401
    """Return one member of ``members`` drawn with ``weights``.

    Args:
        rng: The seeded generator.
        members: The population, in the order ``weights`` describes.
        weights: Relative weights, one per member.

    Returns:
        The drawn member.
    """
    return rng.choices(members, weights=list(weights))[0]


def _build_carriers(rng: random.Random, n: int) -> list[Carrier]:
    """Create ``n`` deterministic carriers spanning every tier.

    The first four are round-robined across :class:`CarrierTier` so no tier is ever missing
    even for a small network; the remainder are drawn with :data:`_CARRIER_TIER_WEIGHTS` for
    a realistic mix. Shipment-level tier balance does not depend on this — the shipment
    builder draws a *tier* first and then a carrier holding it — but a network with no
    validated carrier at all would make a whole one-hot column constant.

    Args:
        rng: The seeded generator.
        n: How many carriers to create.

    Returns:
        The carriers, in id order.
    """
    tiers = list(CarrierTier)
    regions = list(OriginRegion)
    carriers: list[Carrier] = []
    for index in range(n):
        tier = tiers[index] if index < len(tiers) else _weighted(rng, tiers, _CARRIER_TIER_WEIGHTS)
        stem = _CARRIER_STEMS[index % len(_CARRIER_STEMS)]
        carriers.append(
            Carrier(
                id=f"carrier-{index:03d}",
                name=f"{stem} Logistics",
                tier=tier,
                gdp_certified=tier in (CarrierTier.PREMIUM, CarrierTier.VALIDATED),
                on_time_rate=round(rng.uniform(0.72, 0.99), 3),
                hub_region=rng.choice(regions),
            )
        )
    return carriers


def _build_facilities(rng: random.Random, n: int) -> list[Facility]:
    """Create ``n`` deterministic facilities, guaranteeing at least one of every kind.

    Args:
        rng: The seeded generator.
        n: How many facilities to create (at least two).

    Returns:
        The facilities, in id order.
    """
    kinds = list(FacilityKind)
    regions = list(OriginRegion)
    facilities: list[Facility] = []
    for index in range(n):
        kind = kinds[index % len(kinds)]
        stem = _SITE_STEMS[index % len(_SITE_STEMS)]
        facilities.append(
            Facility(
                id=f"fac-{index:04d}",
                name=f"{stem} {kind.value.replace('_', ' ').title()}",
                kind=kind,
                region=rng.choice(regions),
                has_backup_power=kind is not FacilityKind.CLINIC or rng.random() < 0.4,
            )
        )
    return facilities


class _ShipmentDraw(BaseModel):
    """One shipment's structural draw, before the label exists.

    A typed carrier for the first pass so the second pass cannot quietly read the wrong
    tuple element. Every random value the second pass needs is drawn **here**, in the first
    pass, so the RNG stream never depends on a branch — the delivered/undelivered coin flip
    or the missingness draw shifting the stream would make "deterministic under a fixed
    seed" quietly false for every record after the first divergence.
    """

    model_config = {"arbitrary_types_allowed": True}

    shipment: Shipment
    features: dict[str, Any]
    signal: float
    delivered: bool
    confounder_unit: float
    noise_unit: float
    excursion_confounder_unit: float
    excursion_noise_unit: float
    missing_draw: float
    stage_draw: float


def _draw_shipment(
    rng: random.Random,
    cfg: GeneratorConfig,
    index: int,
    carriers: list[Carrier],
    facilities: list[Facility],
    shipment_text: dict[ProductClass, list[dict[str, Any]]],
    text_cursor: dict[ProductClass, int],
) -> _ShipmentDraw:
    """Draw one shipment's structure and its noise-free latent signal.

    Args:
        rng: The seeded generator.
        cfg: The generation knobs.
        index: The shipment's ordinal (drives its id and the class round-robin).
        carriers: Carriers available to book.
        facilities: Sites available as origin/destination.
        shipment_text: Per-product-class prose pools.
        text_cursor: Mutable per-class cursor into those pools.

    Returns:
        The populated :class:`_ShipmentDraw`.
    """
    products = list(ProductClass)
    tiers = list(CarrierTier)
    routes = list(RouteClass)
    regions = list(OriginRegion)
    packagings = list(PackagingType)

    # Coverage guarantee: the first pass round-robins every product class so no class is
    # ever missing even for a small N; the remainder is weighted for realistic imbalance.
    product = (
        products[index]
        if index < len(products)
        else _weighted(rng, products, _PRODUCT_WEIGHTS)
    )
    tier = _weighted(rng, tiers, _CARRIER_TIER_WEIGHTS)
    of_tier = [c for c in carriers if c.tier is tier]
    carrier = rng.choice(of_tier or carriers)

    region = _weighted(rng, regions, _REGION_WEIGHTS)
    depots = [f for f in facilities if f.kind is FacilityKind.ORIGIN_DEPOT and f.region is region]
    origin = rng.choice(depots or [f for f in facilities if f.region is region] or facilities)
    endpoints = [
        f
        for f in facilities
        if f.id != origin.id and f.kind in (FacilityKind.CLINIC, FacilityKind.COLD_STORE)
    ]
    destination = rng.choice(endpoints or [f for f in facilities if f.id != origin.id])

    route = _weighted(rng, routes, _ROUTE_WEIGHTS)
    packaging = _weighted(rng, packagings, _PACKAGING_BY_PRODUCT[product])
    low_hours, high_hours, low_hops, high_hops = _ROUTE_PROFILE[route]

    drawn_hours = rng.uniform(low_hours, high_hours)
    transit_hours = round(min(_TRANSIT_MAX, max(_TRANSIT_MIN, drawn_hours)), 2)
    handoff_count = rng.randint(low_hops, high_hops)
    # ``ambient_temp_c`` and ``payload_kg`` are drawn with no reference to origin_region.
    # Real regions do correlate with real temperatures; letting them correlate here would
    # leak signal into origin_region, which is supposed to be a clean negative control.
    ambient_temp_c = round(rng.triangular(_AMBIENT_MIN, _AMBIENT_MAX, _AMBIENT_MODE), 2)
    payload_kg = round(min(_PAYLOAD_MAX, max(_PAYLOAD_MIN, rng.lognormvariate(3.6, 0.9))), 2)
    sensor_gap_minutes = round(min(_GAP_MAX, rng.gammavariate(1.6, _GAP_SCALE_BY_TIER[tier])), 1)

    summary, detail = _next_text(shipment_text, product, text_cursor, rng)
    booked_at = _EPOCH - timedelta(hours=rng.randint(24, 24 * 150))
    dispatched_at = booked_at + timedelta(hours=round(rng.uniform(2.0, 48.0), 2))

    shipment = Shipment(
        id=f"ship-{index:06d}",
        reference=f"CCL-{_EPOCH.year}-{index:06d}",
        summary=summary,
        detail=detail,
        stage=ShipmentStage.IN_TRANSIT,
        route_class=route,
        packaging_type=packaging,
        product_class=product,
        carrier_id=carrier.id,
        origin_facility_id=origin.id,
        destination_facility_id=destination.id,
        shipper_id=f"shipper-{index % 12:03d}",
        booked_at=booked_at,
        dispatched_at=dispatched_at,
        updated_at=dispatched_at,
        transit_hours=transit_hours,
        ambient_temp_c=ambient_temp_c,
        handoff_count=handoff_count,
        payload_kg=payload_kg,
        sensor_gap_minutes=sensor_gap_minutes,
    )

    features = ml_spec.features_for_shipment(shipment, carrier=carrier, origin_facility=origin)
    return _ShipmentDraw(
        shipment=shipment,
        features=features,
        # THE COUPLING. The label's mean comes from ml_spec's latent function and from
        # nowhere else. ``clamp=False`` because this module clamps once, after the noise.
        signal=ml_spec.latent_spoilage_risk(features, clamp=False),
        delivered=rng.random() < cfg.delivered_fraction,
        confounder_unit=_combined_unit_confounder(rng),
        noise_unit=rng.gauss(0.0, 1.0),
        excursion_confounder_unit=_combined_unit_confounder(rng),
        excursion_noise_unit=rng.gauss(0.0, 1.0),
        missing_draw=rng.random(),
        stage_draw=rng.random(),
    )


def _combined_unit_confounder(rng: random.Random) -> float:
    """Draw the two unobserved drivers and combine them at their declared weights.

    Returns a **unit-scaled** value: :func:`_build_shipments` rescales the whole vector so
    the confounders occupy exactly their declared share of the solved noise budget. That
    separation is what lets ``target_r2`` stay a setting — if the weights in
    :data:`~reference.adapter.ml_spec.CONFOUNDERS` were taken at face value, whatever
    magnitude someone happened to type would silently dictate the achievable R².

    Args:
        rng: The seeded generator.

    Returns:
        The combined draw, divided by its own standard deviation so the vector it belongs
        to has unit variance in expectation.
    """
    weights = [weight for _name, weight in ml_spec.CONFOUNDERS]
    combined = sum(weight * rng.gauss(0.0, 1.0) for weight in weights)
    norm = math.sqrt(sum(weight * weight for weight in weights))
    return combined / norm if norm > 0.0 else 0.0


def _build_shipments(
    rng: random.Random,
    cfg: GeneratorConfig,
    carriers: list[Carrier],
    facilities: list[Facility],
    shipment_text: dict[ProductClass, list[dict[str, Any]]],
) -> tuple[list[Shipment], dict[str, float]]:
    """Assemble shipments and label the received ones from the latent signal.

    **Two passes, and the reason is the trap in the module docstring.** The first pass draws
    every structural field and computes the noise-free latent value for the shipments that
    will be labelled. Only then is the variance of those latent values known — and only then
    can :func:`~reference.adapter.ml_spec.noise_budget` derive the σ that lands the
    achievable R² at ``cfg.target_r2``. A single pass would have to guess σ, which is how
    "the label is learnable" becomes a claim nobody re-checks after the next coefficient
    edit.

    The second pass applies, in this order:

    1. the **structured unobserved** term (two confounders, rescaled to their share);
    2. the **heteroscedastic** i.i.d. noise (wider on long lanes, normalised so the total
       budget is redistributed rather than enlarged);
    3. the clamp into ``[0, 100]``, applied exactly once;
    4. the **excursion** score, cut at the :data:`EXCURSION_SHARE` quantile, with the rows
       nearest the cut flipped;
    5. the **MAR holes** in ``sensor_gap_minutes`` — punched *last*, after the label was
       computed from the complete row. That ordering is the definition of MAR: the interval
       existed and moved the outcome, it simply was never published.

    Args:
        rng: The seeded generator.
        cfg: The generation knobs.
        carriers: Carriers available to book.
        facilities: Sites available as origin/destination.
        shipment_text: Per-product-class prose pools.

    Returns:
        ``(shipments, calibration)`` — the records, and the σ figures actually applied,
        which are recorded onto the dataset metadata so a model card can quote them.
    """
    text_cursor: dict[ProductClass, int] = dict.fromkeys(ProductClass, 0)
    draws = [
        _draw_shipment(rng, cfg, index, carriers, facilities, shipment_text, text_cursor)
        for index in range(cfg.num_shipments)
    ]

    labelled = [draw for draw in draws if draw.delivered]
    signals = [draw.signal for draw in labelled]
    var_signal = statistics.pvariance(signals) if len(signals) > 1 else 0.0

    if cfg.noise_scale is not None:
        total = cfg.noise_scale
        noise_sigma = total * math.sqrt(1.0 - cfg.confounder_share)
        confounder_sigma = total * math.sqrt(cfg.confounder_share)
    else:
        noise_sigma, confounder_sigma = ml_spec.noise_budget(
            var_signal, target_r2=cfg.target_r2, confounder_share=cfg.confounder_share
        )
    excursion_noise_sigma, excursion_confounder_sigma = ml_spec.noise_budget(
        var_signal, target_r2=cfg.excursion_signal_r2, confounder_share=cfg.confounder_share
    )

    multipliers = ml_spec.heteroscedastic_multipliers(
        [float(draw.features[ml_spec.HETEROSCEDASTIC_FEATURE]) for draw in labelled]
    )

    risks: list[float] = []
    scores: list[float] = []
    for draw, multiplier in zip(labelled, multipliers, strict=True):
        risks.append(
            max(
                ml_spec.TARGET_FLOOR,
                min(
                    ml_spec.TARGET_CEILING,
                    draw.signal
                    + confounder_sigma * draw.confounder_unit
                    + noise_sigma * multiplier * draw.noise_unit,
                ),
            )
        )
        scores.append(
            draw.signal
            + excursion_confounder_sigma * draw.excursion_confounder_unit
            + excursion_noise_sigma * multiplier * draw.excursion_noise_unit
        )

    flags = _excursion_labels(
        scores, share=ml_spec.EXCURSION_SHARE, flip_rate=cfg.label_flip_rate
    )

    finished: dict[str, tuple[float, ExcursionFlag]] = {
        draw.shipment.id: (risks[position], flags[position])
        for position, draw in enumerate(labelled)
    }

    shipments: list[Shipment] = []
    for draw in draws:
        outcome = finished.get(draw.shipment.id)
        shipments.append(_finalise(draw, outcome, cfg))

    return shipments, {
        "noise_sigma": noise_sigma,
        "confounder_sigma": confounder_sigma,
        "signal_variance": var_signal,
    }


def _excursion_labels(
    scores: list[float], *, share: float, flip_rate: float
) -> list[ExcursionFlag]:
    """Cut a noisy score into excursion labels, then corrupt the rows nearest the boundary.

    Cutting at a **quantile** rather than at a fixed score is what makes ``share`` exact. A
    fixed threshold on an unknown score distribution gives whatever balance it gives —
    usually far more extreme than intended — and an accidentally 99/1 target is the single
    easiest way to produce a classifier that "scores 0.99" while predicting one class
    forever.

    The flips are not uniform, and that is the point. The shipment nobody could call either
    way is the one that gets mislabelled — a two-degree touch at hour 50 that one reviewer
    writes up and another does not — not a shipment three standard deviations into the clean
    region. Flipping the closest-to-boundary rows caps achievable accuracy exactly where a
    calibrated classifier should already be reporting low confidence.

    Args:
        scores: The noisy excursion scores, in row order.
        share: Target share of rows labelled ``excursion``.
        flip_rate: Share of rows to corrupt.

    Returns:
        One :class:`~reference.adapter.schema.ExcursionFlag` per row, aligned to ``scores``.
    """
    count = len(scores)
    if count == 0:
        return []
    ordered = sorted(scores)
    position = min(count - 1, max(0, int(round((1.0 - share) * count)) - 1))
    cut = ordered[position]
    labels = [
        ExcursionFlag.EXCURSION if score > cut else ExcursionFlag.NO_EXCURSION
        for score in scores
    ]
    flips = int(round(flip_rate * count))
    if flips <= 0:
        return labels
    nearest = sorted(range(count), key=lambda i: (abs(scores[i] - cut), i))[:flips]
    for index in nearest:
        labels[index] = (
            ExcursionFlag.NO_EXCURSION
            if labels[index] is ExcursionFlag.EXCURSION
            else ExcursionFlag.EXCURSION
        )
    return labels


def _finalise(
    draw: _ShipmentDraw,
    outcome: tuple[float, ExcursionFlag] | None,
    cfg: GeneratorConfig,
) -> Shipment:
    """Stamp the outcome (or the in-flight stage) and the MAR hole onto one shipment.

    Args:
        draw: The first-pass draw.
        outcome: ``(spoilage_risk_pct, excursion_flag)`` for a received shipment, or None.
        cfg: The generation knobs (supplies the missingness rates).

    Returns:
        The finalised record.
    """
    update: dict[str, Any] = {}
    if outcome is None:
        update["stage"] = (
            ShipmentStage.HELD_AT_HUB if draw.stage_draw < 0.12 else ShipmentStage.IN_TRANSIT
        )
    else:
        risk, flag = outcome
        delivered_at = draw.shipment.dispatched_at + timedelta(
            hours=draw.shipment.transit_hours
        )
        update.update(
            {
                "stage": (
                    ShipmentStage.RELEASED if draw.stage_draw < 0.6 else ShipmentStage.DELIVERED
                ),
                "delivered_at": delivered_at,
                "updated_at": delivered_at,
                "spoilage_risk_pct": round(risk, 3),
                "excursion_flag": flag,
            }
        )

    # MAR, punched last: the interval existed and moved the outcome, it was simply never
    # published. Punching it before the label would make the *recording process* part of the
    # causal story, which is a different and much nastier problem than the one we want the
    # spine's imputation to demonstrate.
    economy = draw.features["carrier_tier"] == ml_spec.MISSING_GAP_TRIGGER_LEVEL
    rate = cfg.missing_gap_peak_rate if economy else cfg.missing_gap_base_rate
    if draw.missing_draw < rate:
        update["sensor_gap_minutes"] = None

    return draw.shipment.model_copy(update=update)


def _build_readings(
    rng: random.Random, cfg: GeneratorConfig, shipments: list[Shipment]
) -> list[SensorReading]:
    """Fabricate data-logger readings, spread deterministically across shipments.

    Readings are evidence, not features: a reading taken mid-journey is not knowable when
    the question is asked at booking time, so featurising one would be leakage. They exist
    because a quality auditor asks to see them and because
    :func:`~reference.adapter.tools.find_shipments` reports how many a shipment carries.

    A shipment already flagged as an excursion gets at least one reading outside its
    qualified band, so the readings do not contradict the label they sit beside.

    Args:
        rng: The seeded generator.
        cfg: The generation knobs.
        shipments: The finalised shipments.

    Returns:
        The readings, in id order.
    """
    if cfg.num_sensor_readings <= 0 or not shipments:
        return []
    readings: list[SensorReading] = []
    for index in range(cfg.num_sensor_readings):
        shipment = shipments[index % len(shipments)]
        offset = rng.uniform(0.5, max(1.0, shipment.transit_hours))
        excursed = shipment.excursion_flag is ExcursionFlag.EXCURSION
        temperature = (
            round(rng.uniform(8.6, 17.0), 2) if excursed and index % 3 == 0
            else round(rng.uniform(2.1, 7.9), 2)
        )
        readings.append(
            SensorReading(
                id=f"read-{index:06d}",
                shipment_id=shipment.id,
                recorded_at=shipment.dispatched_at + timedelta(hours=offset),
                temperature_c=temperature,
                battery_pct=rng.randint(18, 100),
            )
        )
    return readings


def _next_text(
    shipment_text: dict[ProductClass, list[dict[str, Any]]],
    product: ProductClass,
    cursor: dict[ProductClass, int],
    rng: random.Random,
) -> tuple[str, str]:
    """Pull the next ``(summary, detail)`` for ``product``, cycling the pool if needed.

    Args:
        shipment_text: The per-class prose pools.
        product: The class to pull for.
        cursor: Mutable per-class cursor.
        rng: The seeded generator (used only for the templated fallback).

    Returns:
        The summary and detail strings, never empty.
    """
    pool = shipment_text.get(product) or []
    if not pool:
        return _template_shipment_text(product, rng)
    entry = pool[cursor[product] % len(pool)]
    cursor[product] += 1
    summary = str(entry.get("summary") or "").strip()
    detail = str(entry.get("detail") or "").strip()
    fallback_summary, fallback_detail = _template_shipment_text(product, rng)
    return summary or fallback_summary, detail or fallback_detail


# ─────────────────────────────────────────────────────────────────────────────
# Quality gate
# ─────────────────────────────────────────────────────────────────────────────


class DatasetQualityReport(BaseModel):
    """A quick, dependency-free quality gate over a generated dataset.

    These are the checks worth running *before* trusting synthetic data: referential
    integrity, class coverage on both targets, a learnable label present, temporal
    consistency, and PII-free-by-construction. The one that earns its keep every single time
    is ``has_labels`` — an empty training frame is the failure that looks like a model
    problem for an hour.
    """

    referential_integrity: bool = Field(description="Every FK resolves to a record.")
    product_coverage: bool = Field(description="Every product class appears at least once.")
    excursion_coverage: bool = Field(
        description="Both excursion classes appear among the labelled rows — without this "
        "the classification target is constant and every metric on it is meaningless."
    )
    has_labels: bool = Field(description="At least one shipment carries an ML target.")
    temporal_consistency: bool = Field(
        description="booked_at ≤ dispatched_at ≤ delivered_at everywhere."
    )
    pii_free: bool = Field(
        description="No email address or phone-shaped string in any generated free text."
    )
    num_labelled: int = Field(ge=0, description="Count of ML-labelled shipments.")
    product_counts: dict[str, int] = Field(description="Shipments per product class.")
    excursion_counts: dict[str, int] = Field(description="Labelled shipments per excursion class.")
    missing_gap_share: float = Field(
        ge=0.0, le=1.0, description="Share of shipments with no published telemetry interval."
    )

    @property
    def ok(self) -> bool:
        """Whether every hard quality check passed."""
        return (
            self.referential_integrity
            and self.product_coverage
            and self.excursion_coverage
            and self.has_labels
            and self.temporal_consistency
            and self.pii_free
        )


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:\+\d[\d\s().-]{7,}\d)|(?:\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b)")
"""Detectors for the two PII shapes a language model actually invents in freight prose.

Deliberately implemented here rather than deferred to the host's guardrail detector: this
gate has to run with no Aegis checkout present, and "the corpus was scanned" must be a fact
rather than a hope. When a deployed host wants its own detector's verdict as well, it can
run one over the same text — these findings are additive, never a substitute.
"""


def _contains_pii(text: str) -> bool:
    """Return whether ``text`` carries an email address or a phone-shaped string.

    Args:
        text: The generated free text to scan.

    Returns:
        True when something PII-shaped was found.
    """
    return bool(_EMAIL_RE.search(text) or _PHONE_RE.search(text))


def assess_quality(dataset: SyntheticDataset) -> DatasetQualityReport:
    """Run the synthetic-data quality checks over ``dataset`` and report the verdict.

    Pure, offline and cheap — safe to call after every generation run, so a malformed world
    is caught before it seeds the stores.

    Args:
        dataset: The generated dataset to inspect.

    Returns:
        A :class:`DatasetQualityReport` with per-check booleans and balance counts.
    """
    carrier_ids = {c.id for c in dataset.carriers}
    facility_ids = {f.id for f in dataset.facilities}
    shipment_ids = {s.id for s in dataset.shipments}

    referential = all(
        s.carrier_id in carrier_ids
        and s.origin_facility_id in facility_ids
        and s.destination_facility_id in facility_ids
        for s in dataset.shipments
    ) and all(r.shipment_id in shipment_ids for r in dataset.readings)

    product_counts: dict[str, int] = dict.fromkeys((p.value for p in ProductClass), 0)
    for shipment in dataset.shipments:
        product_counts[shipment.product_class.value] += 1

    excursion_counts: dict[str, int] = dict.fromkeys((e.value for e in ExcursionFlag), 0)
    for shipment in dataset.labelled_shipments():
        flag = shipment.excursion_flag or ExcursionFlag.NO_EXCURSION
        excursion_counts[flag.value] += 1

    temporal = all(
        s.booked_at <= s.dispatched_at
        and (s.delivered_at is None or s.delivered_at >= s.dispatched_at)
        for s in dataset.shipments
    )

    texts: list[str] = []
    for shipment in dataset.shipments:
        texts.extend((shipment.summary, shipment.detail))
    for document in dataset.documents:
        texts.extend((document.title, document.body))

    missing_gap = sum(1 for s in dataset.shipments if s.sensor_gap_minutes is None)
    total = max(len(dataset.shipments), 1)

    return DatasetQualityReport(
        referential_integrity=referential,
        product_coverage=all(count > 0 for count in product_counts.values()),
        excursion_coverage=all(count > 0 for count in excursion_counts.values()),
        has_labels=dataset.metadata.num_labelled > 0,
        temporal_consistency=temporal,
        pii_free=not any(_contains_pii(text) for text in texts),
        num_labelled=dataset.metadata.num_labelled,
        product_counts=product_counts,
        excursion_counts=excursion_counts,
        missing_gap_share=round(missing_gap / total, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM-fabricated content (with deterministic fallbacks)
# ─────────────────────────────────────────────────────────────────────────────

_PRODUCT_HINTS: dict[ProductClass, str] = {
    ProductClass.VACCINE: (
        "multi-dose vials on a 2-8 °C lane, campaign stock, clinic delivery windows, "
        "diluent shipped alongside"
    ),
    ProductClass.BIOLOGIC: (
        "monoclonal antibody and cell-therapy consignments, frozen or deep-refrigerated, "
        "patient-specific batches with named destination pharmacies"
    ),
    ProductClass.SMALL_MOLECULE: (
        "controlled-room-temperature tablets and ampoules with an upper excursion limit, "
        "bulk pallet moves between depots"
    ),
    ProductClass.DIAGNOSTIC_KIT: (
        "assay cartridges and reagent packs, tolerant of short excursions but ruined by "
        "freezing, shipped in volume to laboratories"
    ),
}


async def _fabricate_shipment_text(
    complete: CompleteFn | None,
    rng: random.Random,
    cfg: GeneratorConfig,
) -> dict[ProductClass, list[dict[str, Any]]]:
    """Fetch realistic ``(summary, detail)`` pairs per product class via the LLM.

    Falls back to templated text for any class the LLM cannot supply, so the caller always
    receives a full pool.

    Args:
        complete: The injected completion function, or None for the templated path.
        rng: The seeded generator (used by the fallback).
        cfg: The generation knobs.

    Returns:
        Per-class pools of ``{"summary", "detail"}`` entries.
    """
    per_class = max(3, cfg.num_shipments // (len(ProductClass) or 1))
    pools: dict[ProductClass, list[dict[str, Any]]] = {}
    for product in ProductClass:
        entries: list[dict[str, Any]] = []
        if complete is not None:
            entries = await _llm_shipment_text(complete, product, per_class, cfg)
        if not entries:
            entries = [
                {"summary": summary, "detail": detail}
                for summary, detail in (
                    _template_shipment_text(product, rng) for _ in range(per_class)
                )
            ]
        pools[product] = entries
    return pools


async def _llm_shipment_text(
    complete: CompleteFn,
    product: ProductClass,
    count: int,
    cfg: GeneratorConfig,
) -> list[dict[str, Any]]:
    """Ask the cheap model for ``count`` consignment summary/detail pairs of one class.

    Args:
        complete: The injected completion function.
        product: The product class to write about.
        count: How many entries to request.
        cfg: The generation knobs (supplies the temperature).

    Returns:
        The parsed entries, or ``[]`` on any failure.
    """
    system = (
        "You write consignment records for a pharmaceutical cold-chain distributor's "
        "operations system. Everything you write is entirely fictional: never reference "
        "a real company, a real person, a real clinic, a real email address or a real "
        "phone number. The generated text is scanned for PII and a record carrying any "
        "will be rejected."
    )
    user = (
        f"Produce {count} distinct temperature-controlled consignments of product class "
        f"'{product.value}' (typical content: {_PRODUCT_HINTS[product]}). Return JSON of "
        'the form {"shipments": [{"summary": "...", "detail": "..."}, ...]}. '
        "Summaries under 80 characters; details 1-3 sentences naming the packout, the "
        "lane and what the receiving site is expecting."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return await _call_and_parse(complete, ModelRole.CHEAP, messages, "shipments", cfg)


async def _fabricate_documents(
    complete: CompleteFn | None,
    rng: random.Random,
    cfg: GeneratorConfig,
) -> list[Document]:
    """Fabricate a small knowledge-document corpus via the generation model.

    Args:
        complete: The injected completion function, or None for the templated path.
        rng: The seeded generator.
        cfg: The generation knobs.

    Returns:
        The documents, in id order.
    """
    if cfg.num_documents <= 0:
        return []
    products = list(ProductClass)
    entries: list[dict[str, Any]] = []
    if complete is not None:
        entries = await _llm_documents(complete, cfg.num_documents, cfg)

    documents: list[Document] = []
    for index in range(cfg.num_documents):
        product = products[index % len(products)]
        source = entries[index] if index < len(entries) else {}
        title = str(source.get("title") or "").strip()
        body = str(source.get("body") or "").strip()
        if not title or not body:
            title, body = _template_document(product, index)
        documents.append(
            Document(
                id=f"doc-gen-{index:04d}",
                kind=rng.choice(list(DocumentKind)),
                title=title,
                body=body,
                product_scope=product,
                tags=[product.value, "synthetic"],
                source="synthetic",
            )
        )
    return documents


async def _llm_documents(
    complete: CompleteFn, count: int, cfg: GeneratorConfig
) -> list[dict[str, Any]]:
    """Ask the generation model for ``count`` short cold-chain knowledge documents.

    Args:
        complete: The injected completion function.
        count: How many documents to request.
        cfg: The generation knobs.

    Returns:
        The parsed entries, or ``[]`` on any failure.
    """
    system = (
        "You are a cold-chain quality lead writing short internal reference notes for "
        "logistics coordinators. Keep every note self-contained, generic and fictional; "
        "name no real organisation, person or contact detail."
    )
    user = (
        f"Write {count} short cold-chain reference notes spanning "
        f"{', '.join(p.value for p in ProductClass)}. Return JSON of the form "
        '{"documents": [{"title": "...", "body": "..."}, ...]}. '
        "Each body 3-6 sentences of actionable guidance with concrete thresholds."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return await _call_and_parse(complete, ModelRole.GENERATION, messages, "documents", cfg)


async def _call_and_parse(
    complete: CompleteFn,
    role: ModelRole,
    messages: list[dict[str, Any]],
    key: str,
    cfg: GeneratorConfig,
) -> list[dict[str, Any]]:
    """Call the gateway for JSON, parse defensively, and return ``result[key]``.

    Any transport or parsing failure returns ``[]`` so the caller falls back to templated
    content. **The generator must never raise on an LLM problem** — that is the whole
    graceful-degradation guarantee, and it is what makes the demo survive a missing API key.
    The degradation is not silent: ``DatasetMetadata.llm_used`` records which path ran.

    Args:
        complete: The injected completion function.
        role: Which model role to bill the call to.
        messages: Chat messages to send.
        key: The top-level JSON key holding the list of entries.
        cfg: The generation knobs (supplies the temperature).

    Returns:
        The parsed list of dict entries, or ``[]`` on any failure.
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
        return [entry for entry in entries if isinstance(entry, dict)]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic templated fallbacks (guarantee schema-valid output offline)
#
# This path runs on every test, every offline run, every training frame and quite possibly
# the demo. It is not a second-class path, so the prose here is real prose.
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_SUMMARIES: dict[ProductClass, tuple[str, ...]] = {
    ProductClass.VACCINE: (
        "Campaign vaccine stock to regional clinic network",
        "Multi-dose vial consignment, 2-8 C lane",
        "Second-dose replenishment for immunisation programme",
    ),
    ProductClass.BIOLOGIC: (
        "Patient-specific cell therapy, deep-frozen packout",
        "Monoclonal antibody batch to hospital pharmacy",
        "Frozen biologic transfer between cold stores",
    ),
    ProductClass.SMALL_MOLECULE: (
        "Bulk tablet pallet move, controlled room temperature",
        "Ampoule consignment with upper excursion limit",
        "Depot-to-depot rebalance of oral solid stock",
    ),
    ProductClass.DIAGNOSTIC_KIT: (
        "Assay cartridge shipment to laboratory network",
        "Reagent pack replenishment, freeze-sensitive",
        "Point-of-care test kits to community sites",
    ),
}

_TEMPLATE_DETAILS: dict[ProductClass, tuple[str, ...]] = {
    ProductClass.VACCINE: (
        "Vials are packed with the diluent in the same shipper and must arrive together. "
        "The receiving clinic has a single cold-room door and a two-hour receiving window.",
        "Stock is drawn from qualified inventory and the lane is booked against the "
        "campaign schedule, so a missed slot pushes the whole clinic day.",
    ),
    ProductClass.BIOLOGIC: (
        "The packout is patient-specific and cannot be substituted; a write-off means the "
        "treatment slot is lost as well as the product.",
        "Deep-frozen product is transferred hand-to-hand at the destination cold store and "
        "the receiving pharmacist signs for the logger download at the same time.",
    ),
    ProductClass.SMALL_MOLECULE: (
        "Product tolerates the low end of the range but is written off above 30 C, so the "
        "risk on this lane is heat rather than freezing.",
        "A bulk pallet move between depots; the receiving depot will re-palletise before "
        "the stock is released to picking.",
    ),
    ProductClass.DIAGNOSTIC_KIT: (
        "Cartridges survive short warm excursions but are destroyed by freezing, so the "
        "winter risk on this lane is the tarmac rather than the truck.",
        "Reagent packs are volume stock for a laboratory network and are released against "
        "a batch certificate on arrival.",
    ),
}


def _template_shipment_text(product: ProductClass, rng: random.Random) -> tuple[str, str]:
    """Return a deterministic ``(summary, detail)`` for ``product`` (the LLM fallback).

    Args:
        product: The product class to write about.
        rng: The seeded generator.

    Returns:
        The summary and the detail.
    """
    summary = rng.choice(_TEMPLATE_SUMMARIES[product])
    detail = rng.choice(_TEMPLATE_DETAILS[product])
    return summary, detail


def _template_document(product: ProductClass, index: int) -> tuple[str, str]:
    """Return a deterministic ``(title, body)`` knowledge document (the LLM fallback).

    Args:
        product: The product class the note is scoped to.
        index: The document's ordinal, used to keep titles distinct.

    Returns:
        The title and the body — long enough that the chunker produces at least one chunk,
        which the corpus conformance check verifies for seed documents.
    """
    label = product.value.replace("_", " ")
    title = f"Cold-chain handling note {index}: {label}"
    body = (
        f"This note covers routine handling of {label} consignments "
        f"({_PRODUCT_HINTS[product]}). Confirm the packout matches the booked packaging "
        "type before dispatch: a substitution made on the dock is the single most common "
        "cause of an unexplained excursion. Record the contracted telemetry interval on "
        "the consignment; where the carrier publishes none, treat the lane as unmonitored "
        "and shorten the review cadence to every four hours. On arrival, download the "
        "logger before the shipper is broken down, because a logger separated from its "
        "packout can no longer evidence anything. If the record shows time outside the "
        "qualified range, quarantine the consignment and raise the assessment before the "
        "stock is released to picking."
    )
    return title, body


def _template_shipment_pool(
    rng: random.Random, cfg: GeneratorConfig
) -> dict[ProductClass, list[dict[str, Any]]]:
    """Build the deterministic templated prose pool — the no-LLM path.

    Args:
        rng: The seeded generator.
        cfg: The generation knobs.

    Returns:
        Per-class pools of ``{"summary", "detail"}`` entries.
    """
    per_class = max(3, cfg.num_shipments // (len(ProductClass) or 1))
    return {
        product: [
            {"summary": summary, "detail": detail}
            for summary, detail in (
                _template_shipment_text(product, rng) for _ in range(per_class)
            )
        ]
        for product in ProductClass
    }


def _template_documents(rng: random.Random, cfg: GeneratorConfig) -> list[Document]:
    """Build the deterministic templated knowledge corpus — the no-LLM path.

    Args:
        rng: The seeded generator.
        cfg: The generation knobs.

    Returns:
        The documents, in id order.
    """
    if cfg.num_documents <= 0:
        return []
    products = list(ProductClass)
    documents: list[Document] = []
    for index in range(cfg.num_documents):
        product = products[index % len(products)]
        title, body = _template_document(product, index)
        documents.append(
            Document(
                id=f"doc-gen-{index:04d}",
                kind=rng.choice(list(DocumentKind)),
                title=title,
                body=body,
                product_scope=product,
                tags=[product.value, "synthetic"],
                source="synthetic",
            )
        )
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# The demand series — what /forecast forecasts, in this domain's words
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_SERIES_LABEL = "Shipments dispatched per day"
"""What the client-facing demand series measures, in the client's language.

**This string used to be a constant in the Aegis core**, which made it the one domain
sentence no retarget ever changed: the ``/forecast`` response carried it to the console, the
console drew it on the chart, and a completely different deployment charted the shipped
domain's words over its own data forever. It is a sentence a jury reads.
"""

DOMAIN_SERIES_UNIT = "shipments"
"""The unit of :func:`domain_series_events`' values, for the forecast's y-axis."""


def domain_series_events(
    *, num_records: int = 1400, seed: int = 11
) -> list[tuple[datetime, float]]:
    """Return one ``(timestamp, 1.0)`` dispatch event per generated shipment.

    The **dispatch** series, deliberately, not a delivery series: dispatches are the
    quantity a cold-chain planner books capacity, packouts and dry ice against, and the
    series is complete at the recent end — whereas deliveries silently truncate it by the
    length of the longest lane and bias the trend downwards for no reason a reader could see.

    This is the whole of the domain's contribution to ``/forecast``. The core buckets, fits
    and refuses honestly; it never names a record type or a timestamp field.

    Args:
        num_records: How many shipments to fabricate. Large enough that a daily bucket over
            the generator's span is a countable volume rather than a sparse 0/1 rattle that
            no model — and no reader — could learn from.
        seed: RNG seed, so the demo series is identical across processes and reloads.

    Returns:
        Dispatch events, unordered.
    """
    dataset = generate_synthetic_sync(
        GeneratorConfig(num_shipments=num_records, seed=seed, use_llm=False)
    )
    return [(shipment.dispatched_at, 1.0) for shipment in dataset.shipments]
