"""Piece 2 of 10 — the supervised problem: which columns, which target, which frame.

WHAT THIS FILE IS
    The single source of truth for what is predictable in cold-chain logistics:

      * :data:`FEATURES` — the ordered, typed feature contract. Every categorical's
        ``levels`` list comes from a piece 1 ``StrEnum``, never from a hand-typed list.
      * :data:`TARGET` — the primary regression target, ``spoilage_risk_pct``, and its
        **unit** (``%``), which :func:`describe_prediction` prints.
      * :data:`SECONDARY_TARGET` — the classification target, ``excursion_flag``.
      * :data:`LATENT_INTERCEPT` / :data:`CATEGORICAL_EFFECTS` / :data:`NUMERIC_DRIVERS` /
        :data:`INTERACTION` / :data:`CONFOUNDERS` — the **declared causal story**, as plain
        data. :func:`latent_spoilage_risk` evaluates it row by row for the generator;
        :mod:`reference.problem` re-expresses the *same tables* as an
        :class:`aegis_ml.data.latent.LatentModel` for the pipelines. One table, two
        evaluators, no possibility of drift.
      * :func:`features_for_shipment` — one record (joined to its carrier and origin
        facility) → a flat feature dict.
      * :func:`feature_matrix` / :func:`training_frame` — the labelled frame the spine
        trains on.
      * :func:`describe_prediction` — one prediction → this domain's own decision-support
        sentence, injected into the planner prompt.

    **Pure Python at import time.** No numpy, no pandas, no scikit-learn at module scope —
    ``pandas`` is imported *inside* :func:`training_frame`. The one non-stdlib import is
    ``pydantic`` (through :mod:`aegis_ml.contracts.spec`, which itself imports nothing
    else), because this module is loaded by the conformance suite and by spec resolution in
    contexts where the ML extra is not installed at all.

THE CONTRACT (aegis.adapter.MLSpecModule) — these five names must survive
    FEATURES, FEATURE_NAMES, TARGET, training_frame(), describe_prediction()

    ``training_frame``'s keyword is ``num_records``, deliberately domain-neutral: the core
    Protocol names it, and spelling it ``num_shipments`` here breaks the contract.

THE TRAP
    ``aegis.ml.spec.resolve_spec`` reads ``FEATURE_NAMES`` and ``TARGET.name`` leniently
    and returns a four-column noise fallback when it finds neither. **Nothing raises.** The
    model trains on noise and serves it as domain evidence. :data:`PROBLEM` exists partly to
    close that: it is the one declarative object the data contract, the feature pipeline and
    this module's own names are all derived from, so there is no second place to typo.

THE OTHER TRAP — the one that costs the demo
    The label must be sampled *around* :func:`latent_spoilage_risk`, and the noise must be
    **calibrated**. Both directions are failures:

      * No coupling at all ⇒ the target is noise, R² ≈ 0, the conformal interval is
        honestly enormous, and the model has nothing to explain.
      * Latent function + a whisper of noise ⇒ R² ≈ 0.99, which is not a triumph. It says
        the label is a closed-form function of the inputs; SHAP re-reads the coefficient
        table back to you, and a reviewer who asks "so what does the model add?" is right.

    :data:`TARGET_R2` sits at 0.65, in the middle of the 0.45–0.80 band, and four devices
    put it there honestly rather than by hand-tuning a magic constant:

      1. :func:`calibrated_noise_sigma` — σ solved from the *measured* variance of the
         latent values the generator just computed, so it stays correct when a coefficient
         below changes.
      2. :data:`CONFOUNDERS` — two drivers that genuinely move the target and are never
         emitted as columns. This is the honest ceiling no model can climb past.
      3. :func:`heteroscedastic_multipliers` — the noise is wider on long lanes, so an
         adaptive conformal interval has a real reason to breathe.
      4. :data:`INTERACTION` plus the ``tanh``/``log1p`` shapes — a purely additive latent
         function is exactly recoverable by ridge regression, which would make the spine's
         boosted ensemble decorative and flatten the SHAP plot into a restatement of the
         table.

    And :data:`IRRELEVANT_FEATURES` names the two columns with **no driver at all**. A SHAP
    report that correctly puts them near zero is far better evidence than one where every
    column happens to matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from aegis_ml.contracts.spec import FeatureSpec, MLProblem, TargetSpec

from reference.adapter.schema import (
    CarrierTier,
    ExcursionFlag,
    OriginRegion,
    PackagingType,
    ProductClass,
    RouteClass,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only, never a runtime dependency
    import pandas as pd

    from reference.adapter.schema import Carrier, Facility, Shipment, SyntheticDataset

__all__ = [
    "CATEGORICAL_EFFECTS",
    "CATEGORICAL_FEATURES",
    "CONFOUNDERS",
    "CONFOUNDER_SHARE",
    "EXCURSION_PROBLEM",
    "EXCURSION_TARGET_ACCURACY",
    "FEATURES",
    "FEATURE_NAMES",
    "HETEROSCEDASTIC_FEATURE",
    "HETEROSCEDASTIC_STRENGTH",
    "INTERACTION",
    "IRRELEVANT_FEATURES",
    "LABEL_FLIP_RATE",
    "LATENT_INTERCEPT",
    "MISSING_GAP_BASE_RATE",
    "MISSING_GAP_PEAK_RATE",
    "MISSING_GAP_TRIGGER_LEVEL",
    "NUMERIC_DRIVERS",
    "NUMERIC_FEATURES",
    "PROBLEM",
    "SECONDARY_TARGET",
    "TARGET",
    "TARGET_CEILING",
    "TARGET_FLOOR",
    "TARGET_R2",
    "Interaction",
    "NumericDriver",
    "TransformName",
    "calibrated_noise_sigma",
    "describe_prediction",
    "excursion_frame",
    "excursion_matrix",
    "feature_matrix",
    "features_for_shipment",
    "heteroscedastic_multipliers",
    "latent_spoilage_risk",
    "noise_budget",
    "training_frame",
]


# ─────────────────────────────────────────────────────────────────────────────
# The feature contract
#
# ORDER IS THE CONTRACT — FEATURE_NAMES preserves it and the training frame's columns are
# built from it. Every ``levels`` list comes from a piece 1 enum, never a hand-typed list,
# or the two drift and an unseen level one-hot-encodes to all-zeros in silence.
#
# Every feature here is knowable at BOOKING TIME, before the shipment moves. That is the
# rule that keeps this a forecasting problem rather than a leakage problem: the logger
# readings that ultimately *prove* an excursion are not features, because they do not exist
# when the question is asked.
# ─────────────────────────────────────────────────────────────────────────────

FEATURES: list[FeatureSpec] = [
    FeatureSpec(
        name="carrier_tier",
        dtype="categorical",
        description=(
            "Contracted service tier of the carrier. A GDP-validated lane is temperature "
            "mapped end to end; an economy lane is general freight with a cold box."
        ),
        levels=[t.value for t in CarrierTier],
    ),
    FeatureSpec(
        name="route_class",
        dtype="categorical",
        description=(
            "Shape of the journey. Every custody transfer is a door opening, a tarmac "
            "wait and a chance to be left on a dock, so more legs means more exposure."
        ),
        levels=[r.value for r in RouteClass],
    ),
    FeatureSpec(
        name="packaging_type",
        dtype="categorical",
        description=(
            "Thermal system protecting the payload. Passive gel packs hold for hours; "
            "phase-change material and powered reefers hold for days."
        ),
        levels=[p.value for p in PackagingType],
    ),
    FeatureSpec(
        name="origin_region",
        dtype="categorical",
        description=(
            "Coarse geography the shipment departed from. Carried because operations "
            "reports are cut by it; it is NOT a driver of spoilage in this domain and is "
            "generated independently of the target on purpose (see IRRELEVANT_FEATURES)."
        ),
        levels=[r.value for r in OriginRegion],
    ),
    FeatureSpec(
        name="product_class",
        dtype="categorical",
        description=(
            "What is inside, at the granularity that changes thermal sensitivity. Live "
            "vaccine is the least forgiving; a diagnostic kit is the most."
        ),
        levels=[p.value for p in ProductClass],
    ),
    FeatureSpec(
        name="transit_hours",
        dtype="numeric",
        unit="hours",
        description=(
            "Planned door-to-door duration. The dominant continuous driver, and a "
            "saturating one: past the packaging's qualified duration the marginal hour "
            "adds much less than the first hour over."
        ),
        minimum=6.0,
        maximum=132.0,
    ),
    FeatureSpec(
        name="ambient_temp_c",
        dtype="numeric",
        unit="degrees_celsius",
        description="Forecast mean ambient temperature along the lane. Hotter lanes eat "
        "thermal reserve faster.",
        minimum=-4.0,
        maximum=40.0,
    ),
    FeatureSpec(
        name="handoff_count",
        dtype="numeric",
        unit="transfers",
        description="Planned custody transfers between origin and destination.",
        minimum=0.0,
        maximum=7.0,
    ),
    FeatureSpec(
        name="payload_kg",
        dtype="numeric",
        unit="kilograms",
        description=(
            "Gross weight of the consignment. Billed on, reported on, and genuinely NOT "
            "predictive of spoilage once packaging and duration are known — the second "
            "deliberately irrelevant column (see IRRELEVANT_FEATURES)."
        ),
        minimum=5.0,
        maximum=900.0,
    ),
    FeatureSpec(
        name="sensor_gap_minutes",
        dtype="numeric",
        unit="minutes",
        description=(
            "Contracted interval between data-logger transmissions. A long gap means an "
            "excursion can run unnoticed for that long before anyone can intervene. "
            "Nullable: economy carriers often publish no interval at all, which is this "
            "domain's missing-at-random hole."
        ),
        minimum=0.0,
        maximum=540.0,
        nullable=True,
    ),
]
"""The ordered, typed feature contract the ML spine consumes."""

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
"""Ordered feature-column names. ``resolve_spec`` reads THIS NAME — do not rename it."""

CATEGORICAL_FEATURES: list[str] = [f.name for f in FEATURES if f.dtype == "categorical"]
"""The one-hot subset, in declaration order."""

NUMERIC_FEATURES: list[str] = [f.name for f in FEATURES if f.dtype != "categorical"]
"""Everything passed through to the estimator unencoded, in declaration order."""

TARGET: TargetSpec = TargetSpec(
    name="spoilage_risk_pct",
    task="regression",
    unit="%",
    description=(
        "Assessed probability, in percent, that a received consignment has lost potency "
        "and must be written off. It is learnable because it is a smooth, monotone "
        "function of the lane's booked characteristics — carrier discipline, journey "
        "shape, packaging, duration, ambient heat and telemetry cadence — with calibrated "
        "noise and two unobserved drivers, so the model has real signal to find and the "
        "conformal interval has a real reason to be wide."
    ),
    minimum=0.0,
    maximum=100.0,
)
"""The primary prediction target. ``resolve_spec`` reads ``TARGET.name`` — do not rename."""

SECONDARY_TARGET: TargetSpec = TargetSpec(
    name="excursion_flag",
    task="classification",
    description=(
        "Whether the data-logger record shows the payload left its qualified temperature "
        "range at any point in the journey. Deliberately imbalanced — most shipments "
        "arrive clean — so a classifier that beats the majority-class rate has learned "
        "something rather than learned to say 'no'."
    ),
    levels=[e.value for e in ExcursionFlag],
)
"""The secondary, classification target. Same features, different question, harder metric."""

IRRELEVANT_FEATURES: tuple[str, ...] = ("origin_region", "payload_kg")
"""The two columns with **no driver at all** — one categorical, one numeric.

They are declared, generated, contract-checked and fed to the model exactly like the other
eight, and they move the target by precisely zero. A SHAP report that puts them near the
bottom is evidence the attribution is reading the data rather than reading the analyst's
expectations; a report that ranks either of them highly is a finding worth chasing.

``origin_region`` is drawn **independently** of ``ambient_temp_c`` in the generator for
exactly this reason. Real regions do correlate with real temperatures, and letting them
correlate here would have leaked signal into a column that is supposed to have none,
quietly destroying the only clean negative control in the feature set.
"""


# ─────────────────────────────────────────────────────────────────────────────
# The declared causal story
#
# These tables ARE the domain's claim about what drives spoilage. Every effect is in the
# target's own unit (percentage points), so a reader can sanity-check them by eye, and
# every driver is monotone: more heat is never safer, a longer lane is never safer.
#
# reference.problem re-expresses these same tables as aegis_ml LatentDriver / Interaction /
# Confounder objects. They are read from here rather than re-typed there, so the pure-Python
# evaluator below and the vectorised one the pipelines use cannot disagree.
# ─────────────────────────────────────────────────────────────────────────────

TransformName = Literal["identity", "tanh", "log1p"]
"""Shape functions used by the numeric drivers.

All three are **monotone**, which is what keeps the latent function explainable ("a longer
lane is always at least as risky") and its SHAP attribution readable as a domain claim.
``tanh`` and ``log1p`` are the non-linear pair: they are why a ridge baseline visibly loses
to the boosted ensemble instead of matching it.

The names and their semantics match ``aegis_ml.data.latent.TransformName`` exactly,
including the ``log1p`` domain clamp, so the two evaluators agree to the last bit.
"""


@dataclass(frozen=True)
class NumericDriver:
    """One numeric feature's declared contribution to the latent spoilage signal.

    The contribution is ``coefficient * transform((value - center) / scale)``. Centring and
    scaling are what make the coefficients comparable to each other: without them a driver
    measured in minutes and a driver measured in degrees need coefficients two orders of
    magnitude apart and nobody reviewing the table can tell which one dominates.

    Attributes:
        feature: Column this driver reads.
        coefficient: Signed magnitude in percentage points. Its sign is the domain claim.
        transform: Shape function applied after standardisation.
        center: Subtracted before ``scale``. ``None`` leaves the value uncentred — which
            is required for ``log1p``, whose domain starts just above −1.
        scale: Divides after centring. ``None`` leaves the value unscaled.
    """

    feature: str
    coefficient: float
    transform: TransformName = "identity"
    center: float | None = None
    scale: float | None = None

    def contribution(self, value: object) -> float:
        """Return this driver's signed contribution for one raw cell value.

        A missing or unparseable value contributes exactly ``0.0`` — the neutral element,
        matching ``aegis_ml.data.latent.LatentDriver``. That matters once the MAR holes are
        punched: the label was computed from the *complete* row before the hole appeared, so
        the label stays honest and only the observation is incomplete.

        Args:
            value: The raw cell value; anything float-coercible, or ``None``.

        Returns:
            The signed contribution in percentage points.
        """
        if value is None:
            return 0.0
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(numeric):
            return 0.0
        if self.center is not None:
            numeric -= self.center
        if self.scale is not None:
            numeric /= self.scale
        return self.coefficient * _apply_transform(self.transform, numeric)


@dataclass(frozen=True)
class Interaction:
    """A numeric driver gated on one level of a categorical feature.

    The one interaction term, and it is one that can be justified in a sentence: a long
    lane costs far more under passive gel packs, whose qualified duration is measured in
    hours, than under a powered reefer that simply keeps running. A purely additive latent
    function is exactly recoverable by linear regression, which would make the spine's
    gradient-boosted ensemble decorative and flatten the SHAP plot.

    Attributes:
        left: The numeric feature standardised and used as the magnitude.
        right: The categorical feature whose level gates the term.
        right_level: The level on which the term fires.
        coefficient: Signed magnitude in percentage points.
        left_center: Centring for ``left``.
        left_scale: Scaling for ``left``.
    """

    left: str
    right: str
    right_level: str
    coefficient: float
    left_center: float | None = None
    left_scale: float | None = None

    def contribution(self, row: dict[str, object]) -> float:
        """Return this interaction's signed contribution for one feature row.

        Args:
            row: The raw feature mapping.

        Returns:
            The signed contribution, ``0.0`` when the gate level is not held.
        """
        if str(row.get(self.right)) != self.right_level:
            return 0.0
        magnitude = NumericDriver(
            feature=self.left,
            coefficient=1.0,
            transform="identity",
            center=self.left_center,
            scale=self.left_scale,
        ).contribution(row.get(self.left))
        return self.coefficient * magnitude


def _apply_transform(name: TransformName, value: float) -> float:
    """Apply one shape function, with the same domain guards ``aegis_ml`` uses.

    Args:
        name: The transform to apply.
        value: The already-standardised value.

    Returns:
        The shaped value. ``log1p`` clamps its argument just above ``-1`` so an
        out-of-range feature yields a finite contribution rather than a ``nan`` that would
        silently delete the row from every downstream fit.

    Raises:
        KeyError: If ``name`` is not a known transform — a spec error, not something to
            silently treat as the identity.
    """
    if name == "identity":
        return value
    if name == "tanh":
        return math.tanh(value)
    if name == "log1p":
        return math.log1p(max(value, -0.999_999))
    raise KeyError(f"unknown transform {name!r}; expected one of identity, tanh, log1p")


LATENT_INTERCEPT: float = 34.0
"""Baseline spoilage risk, in percent, before any driver fires.

Chosen so the *labelled* distribution sits comfortably inside ``[0, 100]``: with a signal
standard deviation near 11 points and a calibrated noise budget on top, a baseline of 34
puts the floor at roughly 2.4 standard deviations away. Clamping is a realism feature — a
risk of −6% is not a thing — but a baseline low enough to clamp a fifth of the rows would
destroy recoverable signal and quietly depress the achievable R².
"""

CATEGORICAL_EFFECTS: dict[str, dict[str, float]] = {
    "carrier_tier": {
        CarrierTier.ECONOMY.value: 8.0,
        CarrierTier.STANDARD.value: 2.0,
        CarrierTier.PREMIUM.value: -2.5,
        CarrierTier.VALIDATED.value: -7.0,
    },
    "route_class": {
        RouteClass.DIRECT.value: -6.0,
        RouteClass.SINGLE_TRANSFER.value: -1.0,
        RouteClass.MULTI_LEG.value: 6.5,
        RouteClass.LAST_MILE_POOL.value: 3.5,
    },
    "packaging_type": {
        PackagingType.PASSIVE_GEL.value: 6.0,
        PackagingType.PASSIVE_PCM.value: 1.0,
        PackagingType.ACTIVE_ELECTRIC.value: -6.0,
        PackagingType.DRY_ICE.value: -2.5,
    },
    "product_class": {
        ProductClass.VACCINE.value: 4.5,
        ProductClass.BIOLOGIC.value: 2.0,
        ProductClass.SMALL_MOLECULE.value: -3.5,
        ProductClass.DIAGNOSTIC_KIT.value: -3.0,
    },
}
"""Categorical feature → level → effect on spoilage risk, in percentage points.

``origin_region`` is absent by design: it is one of :data:`IRRELEVANT_FEATURES`. A level
absent from one of these maps contributes zero, which is deliberate — an unseen level must
not silently inherit another level's effect.
"""

NUMERIC_DRIVERS: tuple[NumericDriver, ...] = (
    NumericDriver(
        feature="transit_hours",
        coefficient=8.0,
        transform="tanh",
        center=69.0,
        scale=63.0,
    ),
    NumericDriver(
        feature="ambient_temp_c",
        coefficient=7.0,
        transform="identity",
        center=18.0,
        scale=22.0,
    ),
    NumericDriver(
        feature="handoff_count",
        coefficient=4.5,
        transform="identity",
        center=3.0,
        scale=3.0,
    ),
    NumericDriver(
        feature="sensor_gap_minutes",
        coefficient=4.0,
        transform="log1p",
        center=None,
        scale=110.0,
    ),
)
"""The numeric half of the causal story, in declaration order.

``payload_kg`` is absent by design — the second of :data:`IRRELEVANT_FEATURES`.

``sensor_gap_minutes`` is deliberately **uncentred**: ``log1p`` is only defined above −1,
and centring a 0–540 minute column at its midpoint would push its low end straight into the
clamp, turning a gentle saturating driver into a cliff. Uncentred, ``log1p(gap / 110)`` runs
from 0 to about 1.8, which is exactly the "the first hour of blindness matters far more
than the sixth" shape the domain claims.
"""

INTERACTION: Interaction = Interaction(
    left="transit_hours",
    right="packaging_type",
    right_level=PackagingType.PASSIVE_GEL.value,
    coefficient=7.0,
    left_center=69.0,
    left_scale=63.0,
)
"""The one interaction: duration costs far more under gel packs than under a reefer."""

CONFOUNDERS: tuple[tuple[str, float], ...] = (
    ("unrecorded_tarmac_delay", 1.0),
    ("undocumented_precool_quality", 0.8),
)
"""The **unobserved** drivers: ``(name, relative weight)``, never emitted as columns.

Two of them, and they are the honest reason held-out R² cannot approach 1.0. Both are real
things that move real cold-chain outcomes and that nobody records at booking time: how long
the pallet sat on hot tarmac waiting for a slot, and how thoroughly the shipper pre-cooled
the packout before sealing it.

The *weights* here set the confounders' relative shape. Their *size* is solved from
:data:`CONFOUNDER_SHARE` and the calibrated noise budget by :func:`noise_budget`, so the
declared R² stays a setting rather than becoming a wish that whatever coefficients someone
happened to type quietly overrode.
"""

TARGET_R2: float = 0.65
"""The held-out R² the label noise is calibrated FOR — the data's own ceiling.

Dead centre of the 0.45–0.80 band. Below it the model looks broken; above roughly 0.9 the
label is a closed-form function of the inputs and the whole ML story collapses into "we
wrote a formula and then fitted it". Recorded onto
:class:`~reference.adapter.schema.DatasetMetadata` by the generator, so a model card can
state the ceiling rather than leaving a 0.66 looking like an under-fit.
"""

CONFOUNDER_SHARE: float = 0.4
"""Fraction of the irreducible error that is *structured but unobserved* rather than i.i.d.

At 0.4, two fifths of everything the model cannot explain is the two named
:data:`CONFOUNDERS` and three fifths is plain measurement noise. That split is what lets a
model card say "31% of target variance is irreducible, and most of it has a name" instead
of "the residual is noise", which is a far weaker claim.
"""

HETEROSCEDASTIC_FEATURE: str = "transit_hours"
"""The feature whose percentile rank scales the noise width.

Long lanes are not merely riskier on average, they are *less predictable*: more of the
journey is outside anyone's direct control. Making the residual spread depend on lane
length is what gives an adaptive conformal interval something real to adapt to, instead of
a constant band that is too wide everywhere or too narrow everywhere.
"""

HETEROSCEDASTIC_STRENGTH: float = 0.6
"""Multiplier range for the above: σ runs from ``σ/1.6`` at the shortest lanes to ``σ·1.6``
at the longest, normalised to unit mean square so the total noise budget is redistributed
across rows rather than enlarged."""

LABEL_FLIP_RATE: float = 0.03
"""Share of ``excursion_flag`` labels corrupted, chosen from the rows nearest the boundary.

Uniform random label noise is the wrong model. The shipment nobody could call either way is
the one that gets mislabelled — a two-degree touch at hour 50 that one reviewer writes up
and another does not — not a shipment three standard deviations into the clean region.
"""

EXCURSION_TARGET_ACCURACY: float = 0.86
"""Signal fidelity requested for the classification score before the boundary cut.

Not the achieved accuracy: the boundary cut and :data:`LABEL_FLIP_RATE` both cost some of
it, so the realised held-out accuracy lands lower. The measured value is what
:func:`aegis_ml.data.latent.measure_learnability` reports and what the demo prints; this
number is the knob that puts it inside the 0.65–0.88 band.
"""

MISSING_GAP_TRIGGER_LEVEL: str = CarrierTier.ECONOMY.value
"""Carrier tier at which telemetry-interval reporting collapses.

The missingness is **MAR, not MCAR**, and this is what makes it so: whether
``sensor_gap_minutes`` is recorded depends on ``carrier_tier``, which is itself a driver of
the target. Under MCAR, median imputation would be unbiased and demonstrating it would
prove nothing; under MAR the imputed rows are systematically riskier than the observed
ones, which is when ``MLExplainResponse.imputed_features`` becomes information a reviewer
can act on.
"""

MISSING_GAP_BASE_RATE: float = 0.015
"""Share of non-economy shipments with no published telemetry interval."""

MISSING_GAP_PEAK_RATE: float = 0.12
"""Share of economy shipments with no published telemetry interval.

With economy carriers at roughly 30% of the book these two rates give an overall
missingness near 4%, which is the rate ``config/contracts.toml`` asks for.
"""

TARGET_FLOOR: float = 0.0
"""Lower clamp — spoilage risk is a percentage and percentages are not negative."""

TARGET_CEILING: float = 100.0
"""Upper clamp."""


# ─────────────────────────────────────────────────────────────────────────────
# The declarative problem — one object, three consumers
# ─────────────────────────────────────────────────────────────────────────────

PROBLEM: MLProblem = MLProblem(
    domain_id="cold_chain_logistics",
    features=FEATURES,
    target=TARGET,
    primary_metric="r2",
    requested_coverage=0.9,
)
"""The whole supervised problem as one declarative object.

Constructed here, beside :data:`FEATURES` and :data:`TARGET`, so there is exactly one place
those columns are spelled. :mod:`reference.problem` re-exports it under the same name and
adds the latent model; the pipelines and the demo import it from there.

``requested_coverage`` is 0.9 and is named "requested" on purpose: the coverage the
conformal interval *achieves* on held-out rows is a separate, measured number everywhere in
Aegis, and collapsing the two into one field is how a calibration failure becomes invisible.
"""

EXCURSION_PROBLEM: MLProblem = MLProblem(
    domain_id="cold_chain_logistics_excursion",
    features=FEATURES,
    target=SECONDARY_TARGET,
    primary_metric="accuracy",
    requested_coverage=0.9,
)
"""The same ten features against the classification target.

A separate ``domain_id`` because a trained artifact must never be confusable with one
trained on the regression target: they answer different questions from identical columns,
which is precisely the pair a registry lookup could otherwise mix up.
"""


# ─────────────────────────────────────────────────────────────────────────────
# The latent function and its noise calibration
# ─────────────────────────────────────────────────────────────────────────────


def latent_spoilage_risk(
    features: dict[str, Any],
    *,
    confounder: float = 0.0,
    clamp: bool = True,
) -> float:
    """Compute the noise-free ground-truth spoilage risk for one feature row.

    **This is the linchpin of the whole adapter's "predictable" claim.** The generator calls
    it to set every received shipment's label, so the target is a real function of the
    features by construction — which is what makes the spine's ensemble learnable and its
    conformal intervals meaningful rather than fitted to noise.

    Monotone in every driver (hotter lane → riskier; longer lane → riskier; better carrier
    tier → safer), with one interaction (:data:`INTERACTION`) and two shapes that a linear
    model cannot represent.

    Args:
        features: A feature dict as produced by :func:`features_for_shipment`. Missing keys
            contribute zero rather than raising, so a partial row still scores.
        confounder: The combined contribution of the unobserved drivers for this row, in
            percentage points and **already scaled** (see :func:`noise_budget`). Zero — the
            default — gives the observable-only signal, which is what an analysis, a what-if
            or an oracle comparison should use.
        clamp: Whether to clamp the result into ``[TARGET_FLOOR, TARGET_CEILING]``. The
            generator passes ``False`` because it clamps once, after adding noise; clamping
            twice would put a floor under the signal that the vectorised
            :class:`~aegis_ml.data.latent.LatentModel` does not apply, and the two
            evaluators would silently stop agreeing.

    Returns:
        Spoilage risk in percent.
    """
    total = LATENT_INTERCEPT
    for feature, effects in CATEGORICAL_EFFECTS.items():
        level = features.get(feature)
        if level is not None:
            total += effects.get(str(level), 0.0)
    for driver in NUMERIC_DRIVERS:
        total += driver.contribution(features.get(driver.feature))
    total += INTERACTION.contribution(features)
    total += float(confounder)
    if clamp:
        return max(TARGET_FLOOR, min(TARGET_CEILING, total))
    return total


def calibrated_noise_sigma(var_signal: float, *, target_r2: float = TARGET_R2) -> float:
    """Return the total irreducible standard deviation that lands the ceiling at ``target_r2``.

    For a label ``y = signal + unexplained`` with independent unexplained error, the best
    achievable ``R² = var_signal / (var_signal + var_unexplained)``. Solving::

        sigma_total = sqrt(var_signal * (1 - target_r2) / target_r2)

    The generator computes the latent value for every shipment it is about to label,
    measures the variance of *those* values, and calls this — so the noise is derived from
    the coefficient tables above rather than being a magic constant that silently stops
    being right the moment a coefficient changes. That is the specific failure this function
    exists to prevent: Aegis's own reference generator hardcodes ``noise_scale=4.0`` against
    a signal that spreads far wider, and lands at an R² of 0.97 that nobody re-checks.

    Args:
        var_signal: Variance of the observable, noise-free target across the batch.
        target_r2: The held-out R² to calibrate for; defaults to :data:`TARGET_R2`.

    Returns:
        The total unexplained standard deviation, in percentage points. ``0.0`` for a
        degenerate batch, where any noise at all would be pure noise.

    Raises:
        ValueError: If ``target_r2`` is not strictly inside ``(0, 1)`` — a request for a
            perfectly learnable or perfectly unlearnable label is a spec error, not
            something to silently clamp.
    """
    if not 0.0 < target_r2 < 1.0:
        raise ValueError(f"target_r2 must be in (0, 1), got {target_r2!r}")
    if var_signal <= 0.0:
        return 0.0
    return math.sqrt(var_signal * (1.0 - target_r2) / target_r2)


def noise_budget(
    var_signal: float,
    *,
    target_r2: float = TARGET_R2,
    confounder_share: float = CONFOUNDER_SHARE,
) -> tuple[float, float]:
    """Split the irreducible error into its structured and i.i.d. halves.

    The total budget comes from :func:`calibrated_noise_sigma`; ``confounder_share`` of the
    *variance* goes to the unobserved drivers and the rest to measurement noise. Solving
    both together is what lets both knobs be honoured at once. If the declared confounder
    weights were taken at face value instead, whatever magnitude someone happened to type
    would silently dictate the achievable R² and :data:`TARGET_R2` would become a wish.

    Args:
        var_signal: Variance of the observable latent signal across the batch.
        target_r2: The R² ceiling to calibrate for.
        confounder_share: Fraction of the unexplained *variance* the confounders occupy.

    Returns:
        ``(noise_sigma, confounder_sigma)`` — the standard deviation of the i.i.d. noise
        term and the standard deviation of the combined unobserved drivers, both in
        percentage points.

    Raises:
        ValueError: If ``confounder_share`` is outside ``[0, 1)``.
    """
    if not 0.0 <= confounder_share < 1.0:
        raise ValueError(f"confounder_share must be in [0, 1), got {confounder_share!r}")
    total = calibrated_noise_sigma(var_signal, target_r2=target_r2)
    budget = total * total
    return math.sqrt(budget * (1.0 - confounder_share)), math.sqrt(budget * confounder_share)


def heteroscedastic_multipliers(
    values: list[float], *, strength: float = HETEROSCEDASTIC_STRENGTH
) -> list[float]:
    """Return per-row noise multipliers driven by the percentile rank of ``values``.

    Mirrors ``aegis_ml.data.latent.LatentModel._noise_multiplier`` exactly, in pure Python,
    so the generator and the pipelines describe the same world.

    Two properties, and both are load-bearing:

    * The multiplier is driven by the **percentile rank** rather than the raw value, so the
      effect is identical whether the feature is measured in hours or in millions and no
      single outlier can blow the noise up by 400×.
    * It is normalised to **unit mean square**. Without that, switching heteroscedasticity
      on would quietly *add* variance — the mean of a geometric spread's square is above one
      — and the R² :func:`calibrated_noise_sigma` solved for would come out low by however
      much ``strength`` happened to be. Normalising keeps the two knobs orthogonal: this one
      redistributes the noise budget across rows, it never enlarges it.

    Args:
        values: The driving feature's values, one per row, in row order.
        strength: Multiplier range; ``0.0`` returns a flat list of ones.

    Returns:
        One multiplier per row, aligned to ``values``.
    """
    count = len(values)
    if count == 0:
        return []
    if strength <= 0.0 or len({*values}) < 2:
        return [1.0] * count
    order = sorted(range(count), key=lambda i: values[i])
    ranks = [0.0] * count
    position = 0
    while position < count:
        end = position
        while end + 1 < count and values[order[end + 1]] == values[order[position]]:
            end += 1
        # Average rank across ties, then convert to a percentile in (0, 1] — the same
        # convention as pandas' ``rank(pct=True)``, which the vectorised path uses.
        average = (position + end + 2) / 2.0
        for index in order[position : end + 1]:
            ranks[index] = average / count
        position = end + 1
    raw = [math.pow(1.0 + strength, 2.0 * rank - 1.0) for rank in ranks]
    mean_square = sum(value * value for value in raw) / count
    if mean_square <= 0.0:
        return [1.0] * count
    root = math.sqrt(mean_square)
    return [value / root for value in raw]


# ─────────────────────────────────────────────────────────────────────────────
# Record → features → frame
# ─────────────────────────────────────────────────────────────────────────────


def features_for_shipment(
    shipment: Shipment,
    *,
    carrier: Carrier | None,
    origin_facility: Facility | None,
) -> dict[str, Any]:
    """Extract the model feature dict for one shipment.

    Returns exactly the keys in :data:`FEATURE_NAMES` — no more (an extra key is a column
    the spine never sees) and no fewer (a missing key becomes an all-NaN column at training
    time). Categorical values are the enum ``.value`` strings, never the enum members: a
    ``StrEnum`` member survives a DataFrame round-trip but not a JSON one.

    Args:
        shipment: The shipment to featurise.
        carrier: The contracted carrier (joined), or None if unknown.
        origin_facility: The departure site (joined), or None if unknown.

    Returns:
        A flat ``{feature_name: value}`` dict covering every entry in :data:`FEATURES`.
        ``sensor_gap_minutes`` may legitimately be ``None`` — that is the MAR hole, and it
        is the one value in here that a consumer must be ready to impute.
    """
    return {
        "carrier_tier": carrier.tier.value if carrier else CarrierTier.STANDARD.value,
        "route_class": shipment.route_class.value,
        "packaging_type": shipment.packaging_type.value,
        "origin_region": (
            origin_facility.region.value if origin_facility else OriginRegion.EMEA.value
        ),
        "product_class": shipment.product_class.value,
        "transit_hours": shipment.transit_hours,
        "ambient_temp_c": shipment.ambient_temp_c,
        "handoff_count": shipment.handoff_count,
        "payload_kg": shipment.payload_kg,
        "sensor_gap_minutes": shipment.sensor_gap_minutes,
    }


def _labelled_rows(
    dataset: SyntheticDataset,
) -> tuple[list[dict[str, Any]], list[float], list[str]]:
    """Join every labelled shipment to its carrier and origin site, once.

    Args:
        dataset: The synthetic world to draw rows from.

    Returns:
        ``(rows, spoilage, excursion)`` — the feature dicts and both aligned label lists.
    """
    rows: list[dict[str, Any]] = []
    spoilage: list[float] = []
    excursion: list[str] = []
    for shipment in dataset.labelled_shipments():
        carrier = dataset.carrier_by_id(shipment.carrier_id)
        origin = dataset.facility_by_id(shipment.origin_facility_id)
        rows.append(features_for_shipment(shipment, carrier=carrier, origin_facility=origin))
        # ``labelled_shipments`` guarantees the primary target is not None.
        spoilage.append(float(shipment.spoilage_risk_pct or 0.0))
        flag = shipment.excursion_flag or ExcursionFlag.NO_EXCURSION
        excursion.append(flag.value)
    return rows, spoilage, excursion


def feature_matrix(dataset: SyntheticDataset) -> tuple[list[dict[str, Any]], list[float]]:
    """Build the primary training matrix ``(X, y)`` from a dataset's labelled shipments.

    Args:
        dataset: The synthetic world to draw rows from.

    Returns:
        A tuple ``(X, y)``: a list of feature dicts and the aligned list of spoilage-risk
        percentages.
    """
    rows, spoilage, _ = _labelled_rows(dataset)
    return rows, spoilage


def excursion_matrix(dataset: SyntheticDataset) -> tuple[list[dict[str, Any]], list[str]]:
    """Build the classification matrix ``(X, y)`` for the secondary target.

    Args:
        dataset: The synthetic world to draw rows from.

    Returns:
        A tuple ``(X, y)``: the same feature dicts as :func:`feature_matrix`, and the
        aligned list of ``excursion_flag`` level strings.
    """
    rows, _, excursion = _labelled_rows(dataset)
    return rows, excursion


def training_frame(*, num_records: int = 1400, seed: int = 11) -> pd.DataFrame:
    """Build the ML spine's labelled training frame from a fresh synthetic world.

    This is the ``frame_provider`` the spine resolves at (offline) train time: it generates
    a deterministic synthetic dataset **synchronously** — no LLM, no network, no event loop
    — and turns its labelled shipments into a ``DataFrame`` with one column per
    :data:`FEATURE_NAMES` plus the :data:`TARGET` column.

    ``pandas`` is imported **inside** the function on purpose: this module must stay
    importable with no ML stack installed at all.

    Args:
        num_records: Shipments to synthesise. Only the received-and-assayed ones carry a
            label, so the returned frame has roughly
            ``num_records * GeneratorConfig.delivered_fraction`` rows. The keyword is
            deliberately domain-neutral: ``aegis.adapter.MLSpecModule`` names it, so
            spelling it ``num_shipments`` here would break the contract.
        seed: Seed for the synthetic world (a fixed seed pins the frame exactly).

    Returns:
        A labelled training frame; feature columns keep their native dtypes (categoricals
        are the enum ``.value`` strings, numerics are numbers, and ``sensor_gap_minutes``
        carries genuine nulls).
    """
    import pandas as pd

    from reference.adapter.generator import GeneratorConfig, generate_synthetic_sync

    dataset = generate_synthetic_sync(GeneratorConfig(seed=seed, num_shipments=num_records))
    rows, targets = feature_matrix(dataset)
    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    frame[TARGET.name] = targets
    return frame


def excursion_frame(*, num_records: int = 1400, seed: int = 11) -> pd.DataFrame:
    """Build the labelled frame for the **secondary** classification target.

    Identical features and an identical synthetic world to :func:`training_frame` under the
    same seed — only the label column differs. Kept as a separate frame rather than a third
    column on the primary one because a target sitting in the feature frame of another
    target is perfect leakage waiting for someone to forget to drop it.

    Args:
        num_records: Shipments to synthesise (see :func:`training_frame`).
        seed: Seed for the synthetic world.

    Returns:
        A frame of :data:`FEATURE_NAMES` plus the ``excursion_flag`` column.
    """
    import pandas as pd

    from reference.adapter.generator import GeneratorConfig, generate_synthetic_sync

    dataset = generate_synthetic_sync(GeneratorConfig(seed=seed, num_shipments=num_records))
    rows, labels = excursion_matrix(dataset)
    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    frame[SECONDARY_TARGET.name] = labels
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Prediction → the sentence a human reads
# ─────────────────────────────────────────────────────────────────────────────


def describe_prediction(resp: Any, *, top_k: int = 3) -> str:
    """Render one ML prediction as decision-support text for the agent's reasoning.

    This is the **domain's** framing of the spine's output. The core injects the returned
    block into the planner and generate prompts, so the agent plans *with* the model
    (predict-then-act) and can explain itself from the model's actual drivers rather than
    from its own narration.

    Every field is read through :func:`getattr`, because the response object belongs to the
    host: it is ``app.api.schemas.MLExplainResponse`` in a deployed Aegis and
    ``aegis_ml``'s own shape in a standalone run. Reading defensively is what lets one
    adapter serve both without a hard import of either.

    Args:
        resp: The spine response (prediction, conformal fields, SHAP attribution).
        top_k: How many top SHAP drivers to surface.

    Returns:
        A compact, human-readable multi-line summary safe to embed in a prompt.
    """
    unit = f" {TARGET.unit}" if TARGET.unit else ""
    prediction = getattr(resp, "prediction", None)
    if isinstance(prediction, int | float) and not isinstance(prediction, bool):
        head = f"Predicted spoilage risk: {float(prediction):.1f}{unit}"
    else:
        head = f"Predicted spoilage risk: {prediction}"

    lines = ["ML decision-support for this consignment (regression):", f"- {head}"]

    interval = getattr(resp, "conformal_interval", None)
    confidence = getattr(resp, "conformal_confidence", None)
    set_size = getattr(resp, "prediction_set_size", None)
    if interval is not None and confidence is not None:
        low, high = interval
        lines.append(
            f"- {float(confidence):.0%} conformal interval [{float(low):.1f}, "
            f"{float(high):.1f}]{unit} — quote this whenever you quote the number."
        )
    elif set_size is not None and confidence is not None:
        lines.append(
            f"- {float(confidence):.0%} conformal set size {set_size} (1 = confident)"
        )

    drivers = [
        f"{getattr(f, 'feature', '?')} "
        f"({'+' if float(getattr(f, 'contribution', 0.0)) >= 0 else '−'}"
        f"{abs(float(getattr(f, 'contribution', 0.0))):.2f})"
        for f in (getattr(resp, "shap_attribution", None) or [])[:top_k]
    ]
    if drivers:
        lines.append("- Top drivers (SHAP): " + ", ".join(drivers))

    imputed = list(getattr(resp, "imputed_features", None) or [])
    if imputed:
        lines.append(
            "- Imputed from training data (not this consignment): "
            + ", ".join(str(name) for name in imputed)
            + ". A prediction assembled mostly from imputed values describes the average "
            "lane, not this one."
        )

    lines.append(
        "Use this to decide whether to re-ice, reroute or pre-book a quarantine bay — and "
        "say which. It is evidence for a recommendation, never an authorisation: the "
        "quarantine and reroute tools still stop at their own risk gate."
    )
    return "\n".join(lines)
