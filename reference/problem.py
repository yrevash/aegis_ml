"""The cold-chain domain as ``aegis_ml`` sees it: one problem, two targets, one latent model.

This module is the seam between the **adapter** (which is a domain, written for the Aegis
platform) and the **pipelines** (which are ML machinery, written for any domain). Everything
here is derived from :mod:`reference.adapter.ml_spec` rather than re-typed beside it, and
that is the entire design:

* :data:`PROBLEM` is the :class:`~aegis_ml.contracts.spec.MLProblem` the adapter already
  built out of its own ``FEATURES`` and ``TARGET`` — re-exported, not rebuilt.
* :data:`LATENT` is the **same declared causal story** as
  :func:`reference.adapter.ml_spec.latent_spoilage_risk`, re-expressed as an
  :class:`~aegis_ml.data.latent.LatentModel` so the vectorised pipeline code can evaluate
  it over a whole frame. Its drivers are *constructed from*
  :data:`~reference.adapter.ml_spec.CATEGORICAL_EFFECTS`,
  :data:`~reference.adapter.ml_spec.NUMERIC_DRIVERS` and
  :data:`~reference.adapter.ml_spec.INTERACTION`, so the two evaluators cannot drift apart.

**Why two evaluators at all.** The generator must run with no numpy, no pandas and no
scikit-learn, because ``ml_spec`` is imported by spec resolution and by the conformance
suite in environments where the ML extra is not installed. The pipelines, by contrast, want
a vectorised model that can calibrate σ over a frame, punch MAR holes and report an oracle
R². Two evaluators of one table is the honest resolution; two *tables* would be the trap
the adapter's own docstring warns about — the formula typed twice, drifting silently the
first time somebody edits a coefficient.

The proof that they agree is not a comment. It is the ``oracle_r2`` and ``headroom`` fields
:func:`aegis_ml.data.latent.realism_report` prints: ``oracle_r2`` scores :data:`LATENT`'s
signal against the labels the *generator* wrote. If the two ever diverged, that number
would collapse and the demo would say so on its own front page.
"""

from __future__ import annotations

from aegis_ml.contracts.spec import MLProblem
from aegis_ml.data.latent import (
    Confounder,
    Interaction,
    LatentDriver,
    LatentModel,
    MissingnessRule,
    RealismConfig,
)
from reference.adapter import ml_spec

__all__ = [
    "EXCURSION_LATENT",
    "EXCURSION_PROBLEM",
    "LATENT",
    "PROBLEM",
    "SEED",
]

SEED: int = 11
"""Default seed for every stochastic step here, so a frame regenerates identically.

The same value :func:`reference.adapter.ml_spec.training_frame` defaults to, so a latent
model calibrated in one process describes the frame produced in another.
"""

PROBLEM: MLProblem = ml_spec.PROBLEM
"""The primary supervised problem: predict ``spoilage_risk_pct`` from ten booked columns.

Constructed in :mod:`reference.adapter.ml_spec`, beside the ``FEATURES`` and ``TARGET``
lists it is built from, and re-bound here because this is the module the pipelines and the
demo import. One object, three consumers: the pandera data contract
(:mod:`aegis_ml.contracts.frames`), the feature pipeline
(:mod:`aegis_ml.features.pipeline`) and the adapter's own five Protocol names.
"""

EXCURSION_PROBLEM: MLProblem = ml_spec.EXCURSION_PROBLEM
"""The secondary problem: classify ``excursion_flag`` from the same ten columns."""


def _drivers() -> list[LatentDriver]:
    """Re-express the adapter's declared drivers as ``aegis_ml`` latent drivers.

    Read from :data:`~reference.adapter.ml_spec.CATEGORICAL_EFFECTS` and
    :data:`~reference.adapter.ml_spec.NUMERIC_DRIVERS` rather than re-typed, so a
    coefficient edit lands in both evaluators at once.

    The categorical drivers carry ``coefficient=1.0`` and put the whole effect in
    ``level_effects``, in percentage points. That keeps the numbers readable as domain
    claims — "an economy carrier adds eight points of spoilage risk" — rather than as a
    magnitude multiplied by a shape nobody can hold in their head.

    Returns:
        Every driver the latent model fires, categoricals first, in spec order.
    """
    drivers: list[LatentDriver] = [
        LatentDriver(feature=feature, coefficient=1.0, level_effects=dict(effects))
        for feature, effects in ml_spec.CATEGORICAL_EFFECTS.items()
    ]
    drivers.extend(
        LatentDriver(
            feature=driver.feature,
            coefficient=driver.coefficient,
            transform=driver.transform,
            center=driver.center,
            scale=driver.scale,
        )
        for driver in ml_spec.NUMERIC_DRIVERS
    )
    return drivers


def _interactions() -> list[Interaction]:
    """Re-express the adapter's one interaction term.

    Returns:
        A single-element list: transit duration gated on gel-pack packaging.
    """
    term = ml_spec.INTERACTION
    return [
        Interaction(
            left=term.left,
            right=term.right,
            right_level=term.right_level,
            coefficient=term.coefficient,
            left_center=term.left_center,
            left_scale=term.left_scale,
        )
    ]


def _confounders() -> list[Confounder]:
    """Re-express the two unobserved drivers.

    Their coefficients here set only the confounders' **shape**;
    :attr:`~aegis_ml.data.latent.RealismConfig.confounder_share` sets their size, by
    rescaling the whole vector to occupy exactly its declared fraction of the solved noise
    budget. That separation is what keeps ``target_r2`` a setting rather than a wish.

    Returns:
        One :class:`~aegis_ml.data.latent.Confounder` per declared unobserved driver.
    """
    return [
        Confounder(
            name=name,
            coefficient=weight,
            distribution="normal",
            location=0.0,
            scale=1.0,
        )
        for name, weight in ml_spec.CONFOUNDERS
    ]


def _class_weights() -> dict[str, float]:
    """The declared excursion class balance, keyed by level.

    Read from :data:`~reference.adapter.ml_spec.EXCURSION_SHARE` so the pure-Python
    generator's quantile cut and this model's quantile cut describe the same split.

    Returns:
        Level → share, summing to one.
    """
    share = ml_spec.EXCURSION_SHARE
    return {"no_excursion": 1.0 - share, "excursion": share}


def _missingness() -> list[MissingnessRule]:
    """The one MAR rule: telemetry intervals go unpublished on economy lanes.

    ``depends_on`` is a *driver* of the target, which is what makes this
    missing-at-random rather than missing-completely-at-random. Under MCAR, median
    imputation is unbiased and demonstrating it proves nothing; under MAR the imputed rows
    are systematically riskier than the observed ones, which is exactly when the spine
    telling you which features it filled in becomes information a reviewer can act on.

    Returns:
        The single declared rule.
    """
    return [
        MissingnessRule(
            feature="sensor_gap_minutes",
            depends_on="carrier_tier",
            depends_on_level=ml_spec.MISSING_GAP_TRIGGER_LEVEL,
            rate=ml_spec.MISSING_GAP_BASE_RATE,
            max_rate=ml_spec.MISSING_GAP_PEAK_RATE,
        )
    ]


LATENT: LatentModel = LatentModel(
    intercept=ml_spec.LATENT_INTERCEPT,
    drivers=_drivers(),
    interactions=_interactions(),
    confounders=_confounders(),
    realism=RealismConfig(
        target_r2=ml_spec.TARGET_R2,
        target_accuracy=None,
        confounder_share=ml_spec.CONFOUNDER_SHARE,
        heteroscedastic_feature=ml_spec.HETEROSCEDASTIC_FEATURE,
        heteroscedastic_strength=ml_spec.HETEROSCEDASTIC_STRENGTH,
        # Regression labels are not flipped — flipping is a classification device, and it
        # is declared on EXCURSION_LATENT instead.
        label_flip_rate=0.0,
        missingness=_missingness(),
        irrelevant_share=len(ml_spec.IRRELEVANT_FEATURES) / len(ml_spec.FEATURE_NAMES),
    ),
    floor=ml_spec.TARGET_FLOOR,
    ceiling=ml_spec.TARGET_CEILING,
    task="regression",
    seed=SEED,
)
"""The vectorised spelling of the adapter's declared causal story, for the pipelines.

Handed to :func:`aegis_ml.pipelines.flows.data_flow` as its ``latent`` argument, which is
what turns the realism stage from "what can we measure about this frame" into "how does
this frame compare to the function that generated it". Without it the report omits the
noise, confounder, ceiling and oracle figures rather than guessing them.
"""

EXCURSION_LATENT: LatentModel = LatentModel(
    intercept=ml_spec.LATENT_INTERCEPT,
    drivers=_drivers(),
    interactions=_interactions(),
    confounders=_confounders(),
    realism=RealismConfig(
        target_r2=None,
        target_accuracy=ml_spec.EXCURSION_SIGNAL_R2,
        confounder_share=ml_spec.CONFOUNDER_SHARE,
        heteroscedastic_feature=ml_spec.HETEROSCEDASTIC_FEATURE,
        heteroscedastic_strength=ml_spec.HETEROSCEDASTIC_STRENGTH,
        label_flip_rate=ml_spec.LABEL_FLIP_RATE,
        class_weights=_class_weights(),
        missingness=_missingness(),
        irrelevant_share=len(ml_spec.IRRELEVANT_FEATURES) / len(ml_spec.FEATURE_NAMES),
    ),
    floor=None,
    ceiling=None,
    task="classification",
    levels=[level for level in ml_spec.SECONDARY_TARGET.levels],
    seed=SEED,
)
"""The same drivers read as a logit, cut into excursion classes at the declared balance.

Identical coefficients to :data:`LATENT` on purpose: a shipment at high spoilage risk is a
shipment whose lane was hard on the payload, which is the same lane that produces a logged
excursion. What differs is the *question*, the noise budget, and the boundary label flips —
so the classification frame is genuinely harder than the regression one rather than being a
thresholded copy of it.
"""
