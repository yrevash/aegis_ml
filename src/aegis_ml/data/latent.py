"""Latent target functions, and the proof that the label they produce is learnable.

## Why this module exists at all

``SKILL.md`` names the trap in one sentence: *"the generator must sample labels around
your latent function; if it does not the target is noise, the model finds nothing, and the
conformal interval is honestly enormous."*

Nothing in the platform enforces it. All 14 Aegis conformance checks pass on a frame whose
target is ``random.gauss(0, 1)`` — they check adapter *structure*, and a noise label is
structurally perfect. The only native symptom is ``distinct=False`` on the final line of
``python -m app.ml``, which is read minutes before a demo, at which point the fix is a
regenerate-and-retrain cycle you no longer have time for.

:func:`assert_learnable` is that check, run in seconds, and :class:`LatentModel` is the
thing that makes it pass by construction: a declared function of the features that the
generator samples around, exactly as ``app.adapter.ml_spec.latent_resolution_hours`` does
for the reference domain.

## Why the target must ALSO be hard

The opposite failure is quieter and just as damaging. A latent function plus a whisper of
Gaussian noise yields held-out R² ≈ 0.98, and that number is a tell: it announces that the
data is a toy, and it collapses the entire "uncertainty you can audit" story, because a
conformal interval calibrated on near-zero residuals is a hairline that impresses nobody
and informs no decision. Aegis's own reference generator sets ``noise_scale=4.0`` against a
~12-hour intercept for precisely this reason — roughly a third of the target's scale is
deliberately irreducible.

So realism here is a *declared property*, not an accident of hand-tuned constants:

* :attr:`RealismConfig.target_r2` calibrates the noise σ from the signal's own variance
  (``σ = sqrt(var_signal · (1 − r²) / r²)``) so the achievable ceiling lands where you said.
* :class:`Confounder` drivers genuinely move the target and are **never emitted as
  columns** — an irreducible ceiling no model can cheat past, which is what real data has.
* :attr:`RealismConfig.heteroscedastic_feature` makes the noise wider in some regions, so
  an adaptive conformal interval has an honest reason to breathe.
* :attr:`RealismConfig.label_flip_rate` corrupts labels *near the decision boundary*,
  which is where real measurement error lives.
* :class:`MissingnessRule` punches MAR holes, giving the spine's median/mode imputation and
  the ``MLExplainResponse.imputed_features`` field something real to report.
* :attr:`RealismConfig.class_weights` sets the class balance away from a tidy 50/50.
* :func:`default_latent_model` deliberately leaves some declared features with **no
  driver** — a model that correctly assigns them near-zero SHAP is a far better
  demonstration than one where every column is load-bearing.

The band this aims for: held-out **R² 0.45–0.80**, **accuracy 0.65–0.88**. Clearly learned,
obviously not memorised.

## Both bounds are enforced

:func:`assert_learnable` refuses a score below the floor (the target is noise) *and* flags
a score above :data:`R2_CEILING` / :data:`ACCURACY_CEILING`. A held-out R² of 0.99 on
synthetic data is not a good result, it is a bug report — almost always a leaked feature or
a target trivially recoverable from one column.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from aegis_ml._require import require
from aegis_ml.contracts.errors import (
    AegisMLError,
    InsufficientLabelsError,
    LabelNotLearnableError,
)
from aegis_ml.contracts.spec import TaskType
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable
    from types import ModuleType

    import pandas as pd

    from aegis_ml.contracts.spec import MLProblem

__all__ = [
    "ACCURACY_CEILING",
    "MIN_LEARNABILITY_ROWS",
    "R2_CEILING",
    "Confounder",
    "Interaction",
    "LatentCalibration",
    "LatentDriver",
    "LatentModel",
    "LearnabilityReport",
    "MissingnessRule",
    "RealismConfig",
    "TransformName",
    "assert_learnable",
    "default_latent_model",
    "measure_learnability",
    "realism_report",
]

logger = logging.getLogger(__name__)

_EXTRA = "aegis-ml[serve]"
"""Install target quoted when pandas/scikit-learn are missing."""

R2_CEILING = 0.95
"""Held-out R² above which a regression target is treated as suspiciously easy.

Real tabular problems in this class do not reach 0.95 out of a gradient-boosted ensemble on
a few thousand rows. A score here means one of three things, all of which are bugs: a
leaked feature, a target that is an arithmetic function of one column, or a generator whose
noise term never fired.
"""

ACCURACY_CEILING = 0.98
"""Held-out accuracy above which a classification target is treated as suspiciously easy."""

MIN_LEARNABILITY_ROWS = 60
"""Fewest rows for which a held-out learnability measurement means anything.

Below this the test split is a handful of rows and the measured score is dominated by which
rows happened to land in it — reporting it would be a coin flip dressed as evidence.
"""

_MAJORITY_MARGIN = 0.02
"""How far a classifier must beat the majority-class rate to count as having learned.

A 95/5 binary target hands a constant predictor 0.95 accuracy, which sails past the
configured floor of 0.55 while learning precisely nothing. The floor is therefore raised to
the majority share plus this margin whenever the majority share is the higher bar.
"""

_SIGMA_SEARCH_ITERATIONS = 40
"""Bisection steps used to hit a requested classification accuracy. Cheap and exact enough."""

TransformName = Literal["identity", "log1p", "sqrt", "tanh", "square", "abs"]
"""Shape functions available to a driver.

``identity``, ``log1p``, ``sqrt`` and ``tanh`` are monotone — the property that makes a
latent function explainable ("deeper queue is always slower") and its SHAP attribution
readable. ``square`` and ``abs`` are **not** monotone; they are offered because U-shaped
effects are real (both very short and very long descriptions are hard to triage), but a
driver that uses one is making a domain claim and the spec's description should say so.
"""

_MONOTONE_TRANSFORMS = frozenset({"identity", "log1p", "sqrt", "tanh"})
"""The subset of :data:`TransformName` that preserves ordering; see its docstring."""


def _pandas() -> ModuleType:
    """Import pandas through :func:`~aegis_ml._require.require`."""
    return require(_EXTRA, "pandas")


def _numpy() -> ModuleType:
    """Import numpy through :func:`~aegis_ml._require.require`."""
    return require(_EXTRA, "numpy")


def _sklearn(submodule: str) -> ModuleType:
    """Import a scikit-learn submodule through :func:`~aegis_ml._require.require`.

    Args:
        submodule: Dotted path below ``sklearn``, e.g. ``"ensemble"``.

    Returns:
        The imported submodule.
    """
    return require(_EXTRA, f"sklearn.{submodule}")


def _transform(name: TransformName) -> Callable[[Any], Any]:
    """Resolve a transform name to a callable that works on scalars *and* numpy arrays.

    One table serves both :meth:`LatentModel.evaluate` (one row, pure Python) and
    :meth:`LatentModel.evaluate_frame` (vectorised over a column). That is not a
    micro-optimisation — two implementations of the same latent function drift, and when
    they drift the generator writes labels the learnability check cannot reproduce, which
    is the exact failure this module exists to prevent.

    Domain guards are part of the definition: ``log1p`` is clamped just above ``-1`` and
    ``sqrt`` at ``0`` so an out-of-range feature yields a finite contribution instead of a
    ``nan`` that silently deletes the row from the fit.

    Args:
        name: A member of :data:`TransformName`.

    Returns:
        The callable implementing it.
    """
    np = _numpy()
    table: dict[str, Callable[[Any], Any]] = {
        "identity": lambda v: v,
        "log1p": lambda v: np.log1p(np.maximum(v, -0.999_999)),
        "sqrt": lambda v: np.sqrt(np.maximum(v, 0.0)),
        "tanh": np.tanh,
        "square": np.square,
        "abs": np.abs,
    }
    return table[name]


class LatentDriver(BaseModel):
    """One feature's declared contribution to the latent target.

    A driver is either **numeric** (``coefficient`` times a shaped, optionally standardised
    value) or **categorical** (``coefficient`` times a per-level effect). Standardisation
    through ``center``/``scale`` is what makes coefficients comparable to each other:
    without it a feature measured in characters and a feature measured in months need
    coefficients three orders of magnitude apart, and nobody reviewing the spec can tell
    which driver actually dominates the target.

    A missing or unparseable value contributes exactly zero — the neutral element, matching
    ``latent_resolution_hours``'s ``float(features.get(...) or 0)`` convention. That matters
    once :class:`MissingnessRule` starts punching holes: the latent value was computed from
    the *complete* row before the hole was punched, so the label stays honest and only the
    observation is incomplete.

    Attributes:
        feature: Column name this driver reads.
        coefficient: Signed magnitude. Its sign is the domain claim ("more backlog is
            slower"), so keep it meaningful rather than absorbing it into level effects.
        level_effects: For a categorical feature, level → effect. A level absent from this
            map contributes zero, which is deliberate: an unseen level should not silently
            inherit another level's effect.
        transform: Shape function applied to the standardised numeric value.
        center: Subtracted before ``scale``; usually the feature's midpoint or mean.
        scale: Divides after ``center``; usually half the declared range or the std.
    """

    feature: str = Field(description="Feature column this driver reads.")
    coefficient: float = Field(default=1.0, description="Signed magnitude of the effect.")
    level_effects: dict[str, float] = Field(
        default_factory=dict, description="Categorical level → effect; absent level ⇒ 0."
    )
    transform: TransformName = Field(default="identity", description="Shape function.")
    center: float | None = Field(default=None, description="Subtracted before scaling.")
    scale: float | None = Field(default=None, gt=0.0, description="Divisor after centring.")

    @property
    def is_categorical(self) -> bool:
        """Whether this driver reads levels rather than a number."""
        return bool(self.level_effects)

    @property
    def is_monotone(self) -> bool:
        """Whether this driver preserves the ordering of its feature.

        Reported by :func:`realism_report` because a latent function built only from
        monotone drivers is one a reviewer can check by reading it, and a non-monotone one
        needs its domain justification spelled out.
        """
        return self.is_categorical or self.transform in _MONOTONE_TRANSFORMS

    def _standardise(self, values: Any) -> Any:  # noqa: ANN401 - float or ndarray, one code path
        """Centre and scale a value or column, leaving it untouched when undeclared."""
        if self.center is not None:
            values = values - self.center
        if self.scale is not None:
            values = values / self.scale
        return values

    def contribution(self, value: object) -> float:
        """Contribution of one raw feature value to the latent score.

        Args:
            value: The raw cell value: a level string for a categorical driver, anything
                float-coercible for a numeric one.

        Returns:
            The signed contribution; ``0.0`` for a null, an unparseable value, or a level
            this driver does not name.
        """
        if self.is_categorical:
            if value is None:
                return 0.0
            return self.coefficient * self.level_effects.get(str(value), 0.0)
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(numeric):
            return 0.0
        shaped = _transform(self.transform)(self._standardise(numeric))
        return float(self.coefficient * shaped)

    def contribution_series(self, column: pd.Series) -> pd.Series:
        """Vectorised :meth:`contribution` over a whole column.

        Args:
            column: The raw feature column.

        Returns:
            A float Series aligned to ``column``'s index, nulls contributing ``0.0``.
        """
        pd = _pandas()
        if self.is_categorical:
            mapped = column.astype("object").map(
                lambda v: self.level_effects.get(str(v), 0.0) if v is not None else 0.0
            )
            return pd.to_numeric(mapped, errors="coerce").fillna(0.0) * self.coefficient
        numeric = pd.to_numeric(column, errors="coerce")
        shaped = _transform(self.transform)(self._standardise(numeric))
        return (self.coefficient * pd.Series(shaped, index=column.index)).fillna(0.0)


class Interaction(BaseModel):
    """A product term between two features — the reason a linear model must lose.

    Without at least one interaction (or a non-monotone shape) a ridge regression matches
    the gradient-boosted ensemble, the leaderboard's tiers are indistinguishable, and the
    entire AutoML stack has nothing to prove on the demo data. One honest interaction fixes
    that: "a deep queue costs much more on a technical case than on a billing case" is both
    a real operational claim and something only a model with interactions can represent.

    Attributes:
        left: The numeric feature standardised and used as the first factor.
        right: The second feature. Numeric unless ``right_level`` is set.
        coefficient: Signed magnitude of the product term.
        right_level: When set, ``right`` is treated as categorical and the term fires only
            on rows holding this level — a gated effect rather than a product.
        left_center: Centring for ``left``; keeps the term's scale comparable to main effects.
        left_scale: Scaling for ``left``.
        right_center: Centring for ``right`` (numeric form only).
        right_scale: Scaling for ``right`` (numeric form only).
    """

    left: str = Field(description="First factor's feature name.")
    right: str = Field(description="Second factor's feature name.")
    coefficient: float = Field(default=1.0, description="Signed magnitude.")
    right_level: str | None = Field(
        default=None, description="Gate on this level instead of multiplying."
    )
    left_center: float | None = Field(default=None)
    left_scale: float | None = Field(default=None, gt=0.0)
    right_center: float | None = Field(default=None)
    right_scale: float | None = Field(default=None, gt=0.0)

    def _left_driver(self) -> LatentDriver:
        """The left factor expressed as a plain driver, so standardisation has one owner."""
        return LatentDriver(
            feature=self.left, coefficient=1.0, center=self.left_center, scale=self.left_scale
        )

    def _right_driver(self) -> LatentDriver:
        """The right factor expressed as a plain driver (numeric form only)."""
        return LatentDriver(
            feature=self.right,
            coefficient=1.0,
            center=self.right_center,
            scale=self.right_scale,
        )

    def contribution(self, row: dict[str, object]) -> float:
        """Contribution of this interaction to one row's latent score.

        Args:
            row: The raw feature mapping.

        Returns:
            The signed contribution, ``0.0`` when either factor is missing.
        """
        left = self._left_driver().contribution(row.get(self.left))
        if self.right_level is not None:
            gate = 1.0 if str(row.get(self.right)) == self.right_level else 0.0
            return self.coefficient * left * gate
        right = self._right_driver().contribution(row.get(self.right))
        return self.coefficient * left * right

    def contribution_series(self, frame: pd.DataFrame) -> pd.Series:
        """Vectorised :meth:`contribution` over a frame.

        Args:
            frame: The raw feature frame; must carry both factor columns.

        Returns:
            A float Series aligned to ``frame``'s index.
        """
        left = self._left_driver().contribution_series(frame[self.left])
        if self.right_level is not None:
            gate = (frame[self.right].astype("object").astype(str) == self.right_level).astype(
                float
            )
            return self.coefficient * left * gate
        right = self._right_driver().contribution_series(frame[self.right])
        return self.coefficient * left * right


class Confounder(BaseModel):
    """A driver that genuinely moves the target and is never emitted as a column.

    This is what separates realistic synthetic data from a puzzle with the answer printed
    on the back. Real targets are moved by things nobody recorded — the technician who was
    off sick, the port that was congested that week — and their combined variance is a hard
    ceiling on achievable R² that no amount of AutoML can climb past.

    Having that ceiling be *declared* is the point. It means the model card can say "the
    irreducible share of target variance is 31%, so an R² of 0.66 is 96% of what is
    attainable", which is a far stronger claim than a bare score, and it is why a wide
    conformal interval on this data is correct rather than embarrassing.

    Attributes:
        name: Label for reporting only; no column of this name is created.
        coefficient: Signed magnitude applied to the drawn value.
        distribution: Shape of the unobserved variable.
        location: Mean (``normal``/``lognormal``) or lower bound (``uniform``).
        scale: Standard deviation (``normal``/``lognormal``) or width (``uniform``).
    """

    name: str = Field(description="Reporting label; never becomes a column.")
    coefficient: float = Field(default=1.0, description="Signed magnitude.")
    distribution: Literal["normal", "uniform", "lognormal"] = "normal"
    location: float = Field(default=0.0, description="Mean, or lower bound for uniform.")
    scale: float = Field(default=1.0, gt=0.0, description="Std-dev, or width for uniform.")

    def draw(self, n: int, rng: Any) -> Any:  # noqa: ANN401 - numpy Generator, imported lazily
        """Draw ``n`` unobserved values.

        Args:
            n: How many rows to draw for.
            rng: A seeded ``numpy.random.Generator``.

        Returns:
            An ndarray of contributions (already multiplied by ``coefficient``).
        """
        if self.distribution == "normal":
            raw = rng.normal(self.location, self.scale, size=n)
        elif self.distribution == "lognormal":
            raw = rng.lognormal(self.location, self.scale, size=n)
        else:
            raw = rng.uniform(self.location, self.location + self.scale, size=n)
        return self.coefficient * raw


class MissingnessRule(BaseModel):
    """A MAR hole punched in one feature, conditioned on another.

    ``depends_on`` is required, and that is a deliberate refusal to support MCAR. Under
    missing-completely-at-random, median/mode imputation is unbiased and demonstrates
    nothing — the spine's ``imputed_features`` list would be honest and boring. Under
    missing-at-random the imputed rows are systematically different from the observed ones,
    which is when "we told you which features we filled in for you" becomes information a
    reviewer can act on.

    The rule is applied **after** the target is computed, from the complete row. That
    ordering is the definition of MAR: the value existed and moved the label, it simply was
    not recorded. Punching the hole first would make the label a function of the recording
    process, which is a different (and much nastier) problem.

    Attributes:
        feature: The column that receives nulls. Must be declared ``nullable`` in the spec.
        rate: Share missing among the least-affected rows.
        depends_on: The column that drives the missingness.
        depends_on_level: When set, ``depends_on`` is categorical and rows holding this
            level get ``max_rate``; every other row gets ``rate``.
        max_rate: Share missing among the most-affected rows.
    """

    feature: str = Field(description="Column that receives nulls.")
    rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Floor missingness share.")
    depends_on: str = Field(description="Column the missingness probability depends on.")
    depends_on_level: str | None = Field(default=None, description="Categorical trigger level.")
    max_rate: float = Field(default=0.25, ge=0.0, le=1.0, description="Peak missingness share.")

    @model_validator(mode="after")
    def _rates_ordered(self) -> MissingnessRule:
        """Reject a rule whose peak is below its floor — almost always a swapped pair."""
        if self.max_rate < self.rate:
            raise ValueError(
                f"missingness rule for {self.feature!r} has max_rate {self.max_rate} below "
                f"rate {self.rate}; the two are swapped"
            )
        if self.feature == self.depends_on:
            raise ValueError(
                f"missingness rule for {self.feature!r} depends on itself — that is MNAR, "
                f"not MAR, and imputation cannot be justified under it"
            )
        return self


class RealismConfig(BaseModel):
    """The knobs that keep a generated target hard enough to be worth predicting.

    Defaults are chosen to land inside the target band (R² 0.45–0.80, accuracy 0.65–0.88)
    rather than at the top of it: a demo whose headline number is 0.62 with a stated
    irreducible ceiling of 0.70 is more credible than one reporting 0.99, and it leaves the
    conformal interval a real job to do.

    Attributes:
        target_r2: Requested achievable R² for a regression target. The noise σ is solved
            from the signal's own variance to land here, so realism is declared rather than
            hand-tuned. ``None`` means "use ``LatentModel.noise_scale`` as an absolute σ".
        target_accuracy: The classification analogue, hit by bisecting on σ until the share
            of rows whose noisy score crosses its class boundary matches ``1 − accuracy``.
        heteroscedastic_feature: Feature whose percentile rank scales the noise, so the
            residual spread differs by region and an adaptive interval has something to
            adapt to. ``None`` keeps the noise homoscedastic.
        heteroscedastic_strength: Multiplier range for the above: σ runs from
            ``σ/(1+s)`` at the bottom of the feature's range to ``σ·(1+s)`` at the top.
        label_flip_rate: Share of classification labels flipped to the adjacent class,
            chosen from the rows closest to their boundary — measurement error where
            measurement error actually happens.
        class_weights: Level → share, defining the class balance. Realised exactly by
            cutting the noisy score at the corresponding quantiles.
        missingness: MAR rules applied after the label is computed.
        irrelevant_share: Fraction of declared features :func:`default_latent_model` leaves
            with no driver at all.
    """

    target_r2: float | None = Field(default=0.62, gt=0.0, lt=1.0)
    target_accuracy: float | None = Field(default=0.78, gt=0.0, lt=1.0)
    heteroscedastic_feature: str | None = Field(default=None)
    heteroscedastic_strength: float = Field(default=0.6, ge=0.0, le=4.0)
    label_flip_rate: float = Field(default=0.03, ge=0.0, le=0.4)
    class_weights: dict[str, float] = Field(default_factory=dict)
    missingness: list[MissingnessRule] = Field(default_factory=list)
    irrelevant_share: float = Field(default=0.2, ge=0.0, le=0.6)


class LatentCalibration(BaseModel):
    """The solved noise scale and class boundaries for one latent model on one frame.

    Kept as an explicit, printable object rather than hidden state on the model for two
    reasons. It is *evidence* — ``implied_r2_ceiling`` is the number that turns "R² 0.63"
    from a mediocre score into "90% of everything attainable on this data" — and it keeps
    :class:`LatentModel` pure, so the same model handed the same frame and seed produces
    the same labels in the generator, in the test, and in the doctor command.

    Attributes:
        noise_sigma: Standard deviation of the irreducible noise term actually used.
        signal_variance: Variance of the observable latent signal across the frame.
        confounder_variance: Variance contributed by unobserved drivers.
        implied_r2_ceiling: ``signal_var / (signal_var + confounder_var + σ²)`` — the best
            R² any model could achieve on this data, confounders included.
        cut_points: Score thresholds separating classification levels, in level order.
        class_shares: Realised level → share, measured on the frame.
        notes: Anything a reader must know, e.g. that confounder variance alone already
            exceeded the noise budget so the requested R² was unreachable.
    """

    noise_sigma: float = Field(default=0.0, ge=0.0)
    signal_variance: float = Field(default=0.0, ge=0.0)
    confounder_variance: float = Field(default=0.0, ge=0.0)
    implied_r2_ceiling: float = Field(default=0.0)
    cut_points: list[float] = Field(default_factory=list)
    class_shares: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class LatentModel(BaseModel):
    """A declared latent function plus the realism layer that keeps it honest.

    The regression path is ``clip(intercept + Σ drivers + Σ interactions + Σ confounders +
    ε, floor, ceiling)``. The classification path treats the same sum as a **logit** and
    cuts it at the quantiles implied by :attr:`RealismConfig.class_weights`, which realises
    the requested class balance exactly while leaving the ordering of the underlying score
    intact — so the classes stay learnable from the same drivers.

    Attributes:
        intercept: Baseline score before any driver fires.
        drivers: Per-feature contributions.
        interactions: Product terms; at least one is what makes tree models beat linear ones.
        confounders: Unobserved drivers — real effect, no column, hard R² ceiling.
        noise_scale: Absolute σ, used only when the matching ``target_*`` knob is ``None``.
        realism: The knobs described in :class:`RealismConfig`.
        floor: Inclusive lower clamp for a regression target (ignored for classification —
            clamping a logit would destroy the ordering the class cuts depend on).
        ceiling: Inclusive upper clamp for a regression target.
        task: Which of the two paths above applies.
        levels: Class labels in order, low score to high. Required for classification.
        seed: Default seed for every stochastic step, so a frame regenerates identically.
    """

    intercept: float = Field(default=0.0, description="Baseline before any driver.")
    drivers: list[LatentDriver] = Field(default_factory=list)
    interactions: list[Interaction] = Field(default_factory=list)
    confounders: list[Confounder] = Field(default_factory=list)
    noise_scale: float = Field(default=0.0, ge=0.0, description="Absolute σ fallback.")
    realism: RealismConfig = Field(default_factory=RealismConfig)
    floor: float | None = Field(default=None, description="Regression lower clamp.")
    ceiling: float | None = Field(default=None, description="Regression upper clamp.")
    task: TaskType = Field(default="regression")
    levels: list[str] = Field(default_factory=list, description="Class labels, low to high.")
    seed: int = Field(default_factory=lambda: settings.random_seed)

    @model_validator(mode="after")
    def _coherent(self) -> LatentModel:
        """Reject a model that cannot produce the target it claims to produce."""
        if not self.drivers and not self.interactions:
            raise ValueError(
                "a latent model with no drivers and no interactions produces a constant "
                "plus noise — which is precisely the noise target this module exists to "
                "detect. Declare at least one driver."
            )
        if self.task == "classification":
            if len(self.levels) < 2:
                raise ValueError("a classification latent model needs at least 2 levels")
            weights = self.realism.class_weights
            unknown = sorted(set(weights) - set(self.levels))
            if unknown:
                raise ValueError(f"class_weights names levels not in `levels`: {unknown}")
        elif self.levels:
            raise ValueError("a regression latent model must not declare class levels")
        if self.floor is not None and self.ceiling is not None and self.floor >= self.ceiling:
            raise ValueError(f"floor {self.floor} is not below ceiling {self.ceiling}")
        return self

    @property
    def driven_features(self) -> list[str]:
        """Every feature name that appears in a driver or an interaction."""
        names = {d.feature for d in self.drivers}
        for term in self.interactions:
            names.update({term.left, term.right})
        return sorted(names)

    # ── the deterministic signal ──────────────────────────────────────────────

    def signal(self, row: dict[str, object]) -> float:
        """The observable, noiseless, unclamped latent score for one row.

        This is the quantity a perfect model could recover. It excludes confounders and
        noise on purpose: those are exactly the parts no model can see, and keeping them
        out of this method is what lets :meth:`calibrate` compute an honest ceiling.

        Args:
            row: Raw feature mapping.

        Returns:
            The observable latent score.
        """
        total = self.intercept
        for driver in self.drivers:
            total += driver.contribution(row.get(driver.feature))
        for term in self.interactions:
            total += term.contribution(row)
        return float(total)

    def evaluate(self, row: dict[str, object]) -> float:
        """The noiseless target value for one row: :meth:`signal`, clamped.

        For a classification model this returns the logit, unclamped — see
        :attr:`floor`'s docstring for why clamping it would be wrong.

        Args:
            row: Raw feature mapping.

        Returns:
            The clamped latent value (regression) or the raw logit (classification).
        """
        return self._clamp_scalar(self.signal(row))

    def signal_frame(self, frame: pd.DataFrame) -> pd.Series:
        """Vectorised :meth:`signal` over a frame.

        Args:
            frame: Raw feature frame carrying every driven column.

        Returns:
            A float Series of observable latent scores.

        Raises:
            AegisMLError: When a driven column is absent from the frame — a silent zero
                here would quietly delete a driver from the label.
        """
        pd = _pandas()
        missing = [c for c in self.driven_features if c not in frame.columns]
        if missing:
            raise AegisMLError(
                f"Latent model drives columns the frame does not have: {missing}. "
                f"The label would be computed with those drivers silently set to zero, "
                f"producing a target that no model can recover from the columns present."
            )
        total = pd.Series(float(self.intercept), index=frame.index, dtype="float64")
        for driver in self.drivers:
            total = total + driver.contribution_series(frame[driver.feature])
        for term in self.interactions:
            total = total + term.contribution_series(frame)
        return total

    def evaluate_frame(self, frame: pd.DataFrame) -> pd.Series:
        """Vectorised :meth:`evaluate` over a frame.

        Args:
            frame: Raw feature frame.

        Returns:
            A float Series of clamped, noiseless target values.
        """
        return self._clamp_series(self.signal_frame(frame))

    def _clamp_scalar(self, value: float) -> float:
        """Apply the regression clamps to a scalar (no-op for classification)."""
        if self.task == "classification":
            return value
        if self.floor is not None:
            value = max(self.floor, value)
        if self.ceiling is not None:
            value = min(self.ceiling, value)
        return value

    def _clamp_series(self, values: pd.Series) -> pd.Series:
        """Apply the regression clamps to a Series (no-op for classification)."""
        if self.task == "classification":
            return values
        return values.clip(lower=self.floor, upper=self.ceiling)

    # ── the realism layer ─────────────────────────────────────────────────────

    def _rng(self, seed: int | None) -> Any:  # noqa: ANN401 - numpy Generator, imported lazily
        """Build the seeded numpy generator every stochastic step draws from."""
        np = _numpy()
        return np.random.default_rng(self.seed if seed is None else seed)

    def _confounder_values(self, n: int, rng: Any) -> Any:  # noqa: ANN401 - ndarray
        """Sum every confounder's drawn contribution for ``n`` rows."""
        np = _numpy()
        total = np.zeros(n, dtype="float64")
        for confounder in self.confounders:
            total = total + confounder.draw(n, rng)
        return total

    def _noise_multiplier(self, frame: pd.DataFrame) -> Any:  # noqa: ANN401 - ndarray
        """Per-row scaling of σ, implementing the declared heteroscedasticity.

        The multiplier is driven by the *percentile rank* of the chosen feature rather than
        its raw value, so the effect is identical whether the feature is measured in
        minutes or in millions and no single outlier can blow the noise up by 400×.
        """
        np = _numpy()
        pd = _pandas()
        feature = self.realism.heteroscedastic_feature
        strength = self.realism.heteroscedastic_strength
        if feature is None or strength <= 0.0 or feature not in frame.columns:
            return np.ones(len(frame), dtype="float64")
        column = pd.to_numeric(frame[feature], errors="coerce")
        if column.notna().sum() < 2 or column.nunique(dropna=True) < 2:
            column = pd.Series(
                frame[feature].astype("object").astype(str).factorize()[0], index=frame.index
            )
        ranks = column.rank(pct=True, na_option="keep").fillna(0.5).to_numpy(dtype="float64")
        # ranks ∈ [0, 1] → multiplier ∈ [1/(1+s), 1+s], geometric so the mean stays ~1.
        return np.power(1.0 + strength, 2.0 * ranks - 1.0)

    def calibrate(self, frame: pd.DataFrame, *, seed: int | None = None) -> LatentCalibration:
        """Solve the noise scale (and class cuts) that realise the requested difficulty.

        Regression solves in closed form. With ``V`` the variance of the observable signal
        and ``C`` the variance the confounders add, the achievable R² is
        ``V / (V + C + σ²)``, so hitting ``target_r2 = r`` needs ``σ² = V(1−r)/r − C``. When
        the confounders alone already exceed that budget the requested R² is simply not
        reachable, σ is pinned at zero and a note says so — the ceiling is reported, never
        quietly moved.

        Classification has no closed form, so σ is bisected until the share of rows whose
        noisy score lands on the far side of its class boundary matches ``1 − accuracy``.
        That share is the Bayes error of the generated problem, which is the accuracy
        ceiling a classifier can approach but not exceed.

        Args:
            frame: The feature frame the labels will be drawn for.
            seed: Overrides :attr:`seed` for the confounder draw.

        Returns:
            The solved :class:`LatentCalibration`.
        """
        np = _numpy()
        signal = self.signal_frame(frame)
        signal_var = float(np.nanvar(signal.to_numpy(dtype="float64")))
        confounders = self._confounder_values(len(frame), self._rng(seed))
        confounder_var = float(np.var(confounders)) if self.confounders else 0.0
        notes: list[str] = []

        if self.task == "regression":
            sigma = self._regression_sigma(signal_var, confounder_var, notes)
        else:
            sigma = self._classification_sigma(frame, signal, confounders, notes)

        total_unexplained = confounder_var + sigma**2
        ceiling = (
            signal_var / (signal_var + total_unexplained)
            if signal_var + total_unexplained > 0.0
            else 0.0
        )
        calibration = LatentCalibration(
            noise_sigma=sigma,
            signal_variance=signal_var,
            confounder_variance=confounder_var,
            implied_r2_ceiling=float(ceiling),
            notes=notes,
        )
        if self.task == "classification":
            noisy = self._noisy_score(frame, signal, confounders, sigma, seed)
            calibration.cut_points = self._cut_points(noisy)
            calibration.class_shares = self._shares_from_cuts(noisy, calibration.cut_points)
        return calibration

    def _regression_sigma(
        self, signal_var: float, confounder_var: float, notes: list[str]
    ) -> float:
        """Solve σ for the requested R², or fall back to the declared absolute σ."""
        target = self.realism.target_r2
        if target is None:
            return self.noise_scale
        if signal_var <= 0.0:
            notes.append(
                "observable signal has zero variance on this frame — every driver is "
                "constant here, so no noise level can produce a learnable target"
            )
            return self.noise_scale
        budget = signal_var * (1.0 - target) / target
        remaining = budget - confounder_var
        if remaining <= 0.0:
            notes.append(
                f"confounder variance {confounder_var:.4g} already exceeds the noise budget "
                f"{budget:.4g} for target_r2={target}; σ pinned at 0 and the achievable "
                f"ceiling is BELOW the request — lower the confounder coefficients or "
                f"accept the lower ceiling, but do not read the requested value as achieved"
            )
            return 0.0
        return math.sqrt(remaining)

    def _classification_sigma(
        self,
        frame: pd.DataFrame,
        signal: pd.Series,
        confounders: Any,  # noqa: ANN401 - ndarray
        notes: list[str],
    ) -> float:
        """Bisect σ until the boundary-crossing share matches ``1 − target_accuracy``."""
        np = _numpy()
        target = self.realism.target_accuracy
        if target is None:
            return self.noise_scale
        clean = signal.to_numpy(dtype="float64")
        spread = float(np.nanstd(clean))
        if spread <= 0.0:
            notes.append(
                "observable signal has zero variance on this frame — the class labels "
                "would be pure noise regardless of σ"
            )
            return self.noise_scale
        wanted_error = 1.0 - target
        low, high = 0.0, spread * 8.0
        for _ in range(_SIGMA_SEARCH_ITERATIONS):
            mid = 0.5 * (low + high)
            if self._crossing_share(frame, signal, confounders, mid) < wanted_error:
                low = mid
            else:
                high = mid
        sigma = 0.5 * (low + high)
        achieved = 1.0 - self._crossing_share(frame, signal, confounders, sigma)
        notes.append(
            f"σ={sigma:.4g} bisected to a Bayes-error ceiling of {achieved:.3f} accuracy "
            f"(requested {target:.3f}); a classifier that scores above this is reading "
            f"something it should not have"
        )
        return sigma

    def _crossing_share(
        self,
        frame: pd.DataFrame,
        signal: pd.Series,
        confounders: Any,  # noqa: ANN401 - ndarray
        sigma: float,
    ) -> float:
        """Share of rows whose noisy label differs from their noiseless one, at this σ."""
        np = _numpy()
        clean_cuts = self._cut_points(signal)
        clean_labels = np.digitize(signal.to_numpy(dtype="float64"), clean_cuts)
        noisy = self._noisy_score(frame, signal, confounders, sigma, seed=None)
        noisy_labels = np.digitize(noisy.to_numpy(dtype="float64"), self._cut_points(noisy))
        return float(np.mean(clean_labels != noisy_labels))

    def _noisy_score(
        self,
        frame: pd.DataFrame,
        signal: pd.Series,
        confounders: Any,  # noqa: ANN401 - ndarray
        sigma: float,
        seed: int | None,
    ) -> pd.Series:
        """``signal + confounders + heteroscedastic ε`` as a Series."""
        pd = _pandas()
        rng = self._rng(seed)
        noise = rng.normal(0.0, 1.0, size=len(frame)) * sigma * self._noise_multiplier(frame)
        return pd.Series(
            signal.to_numpy(dtype="float64") + confounders + noise, index=frame.index
        )

    def _weights_in_level_order(self) -> list[float]:
        """Normalised class weights in :attr:`levels` order, defaulting to uniform."""
        declared = self.realism.class_weights
        raw = [float(declared.get(level, 0.0)) for level in self.levels]
        total = sum(raw)
        if total <= 0.0:
            return [1.0 / len(self.levels)] * len(self.levels)
        return [value / total for value in raw]

    def _cut_points(self, scores: pd.Series) -> list[float]:
        """Score thresholds realising the declared class balance on ``scores``.

        Cutting at quantiles rather than at fixed logit values is what makes
        :attr:`RealismConfig.class_weights` exact. A fixed threshold on an unknown score
        distribution gives whatever balance it gives — usually far more extreme than
        intended — and an accidentally 99/1 target is the single easiest way to produce a
        classifier that "scores 0.99" while predicting one class forever.
        """
        np = _numpy()
        weights = self._weights_in_level_order()
        cumulative: list[float] = []
        running = 0.0
        for weight in weights[:-1]:
            running += weight
            cumulative.append(running)
        values = scores.to_numpy(dtype="float64")
        return [float(np.quantile(values, q)) for q in cumulative]

    def _shares_from_cuts(self, scores: pd.Series, cuts: list[float]) -> dict[str, float]:
        """Measure the realised level shares for a scored frame."""
        np = _numpy()
        indices = np.digitize(scores.to_numpy(dtype="float64"), cuts)
        n = max(len(scores), 1)
        return {
            level: float(np.sum(indices == position) / n)
            for position, level in enumerate(self.levels)
        }

    def sample_frame(
        self,
        frame: pd.DataFrame,
        *,
        calibration: LatentCalibration | None = None,
        seed: int | None = None,
    ) -> pd.Series:
        """Draw the actual target column for a frame — the thing the generator writes.

        Regression returns clamped floats; classification returns level strings, cut at the
        calibrated quantiles and then corrupted by :attr:`RealismConfig.label_flip_rate` on
        the rows closest to their boundary.

        Args:
            frame: Raw feature frame.
            calibration: A calibration to reuse (recomputed when ``None``). Reuse it when
                labelling several frames that must share one noise scale — a test split
                labelled with its own σ is not the same problem as the training split.
            seed: Overrides :attr:`seed` for this draw.

        Returns:
            The target Series, aligned to ``frame``'s index.
        """
        signal = self.signal_frame(frame)
        resolved = calibration or self.calibrate(frame, seed=seed)
        confounders = self._confounder_values(len(frame), self._rng(seed))
        noisy = self._noisy_score(frame, signal, confounders, resolved.noise_sigma, seed)
        if self.task == "regression":
            return self._clamp_series(noisy).rename(None)
        return self._labels_from_scores(noisy, resolved)

    def _labels_from_scores(
        self, scores: pd.Series, calibration: LatentCalibration
    ) -> pd.Series:
        """Cut a noisy score into class labels and apply boundary label noise."""
        np = _numpy()
        pd = _pandas()
        cuts = calibration.cut_points or self._cut_points(scores)
        values = scores.to_numpy(dtype="float64")
        indices = np.digitize(values, cuts)
        indices = self._flip_boundary_labels(values, indices, cuts)
        return pd.Series([self.levels[i] for i in indices], index=scores.index, dtype="object")

    def _flip_boundary_labels(
        self,
        values: Any,  # noqa: ANN401 - ndarray
        indices: Any,  # noqa: ANN401 - ndarray
        cuts: list[float],
    ) -> Any:  # noqa: ANN401 - ndarray
        """Flip the rows nearest a class boundary to the adjacent class.

        Uniform random label noise is the wrong model: a shipment nobody could call either
        way is the one that gets mislabelled, not a shipment three standard deviations into
        the safe region. Flipping the closest-to-boundary rows reproduces that, and it caps
        achievable accuracy in the region where a calibrated classifier should already be
        reporting low confidence — which is exactly the behaviour the conformal
        prediction-set size is supposed to demonstrate.
        """
        np = _numpy()
        rate = self.realism.label_flip_rate
        if rate <= 0.0 or not cuts:
            return indices
        count = int(round(rate * len(values)))
        if count <= 0:
            return indices
        cut_array = np.asarray(cuts, dtype="float64")
        distances = np.min(np.abs(values[:, None] - cut_array[None, :]), axis=1)
        nearest_cut = np.argmin(np.abs(values[:, None] - cut_array[None, :]), axis=1)
        chosen = np.argsort(distances, kind="stable")[:count]
        flipped = indices.copy()
        for row in chosen:
            cut_index = int(nearest_cut[row])
            above = values[row] >= cut_array[cut_index]
            flipped[row] = cut_index if above else cut_index + 1
        # A flip must land on a real level even when a row sat outside every cut.
        return np.clip(flipped, 0, len(self.levels) - 1)

    def apply_missingness(
        self,
        frame: pd.DataFrame,
        *,
        problem: MLProblem | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Punch the declared MAR holes into a copy of ``frame``.

        Args:
            frame: The complete frame, already carrying its target column.
            problem: When given, each rule's feature is checked against its
                ``FeatureSpec.nullable`` declaration — a hole in a column the data contract
                says is non-nullable makes every later pandera validation fail, which is a
                confusing way to discover a mismatch between two things you control.
            seed: Overrides :attr:`seed`.

        Returns:
            A copy of ``frame`` with nulls inserted.

        Raises:
            AegisMLError: When a rule targets a column that is absent, or one the spec
                declares non-nullable.
        """
        np = _numpy()
        pd = _pandas()
        rules = self.realism.missingness
        if not rules:
            return frame
        nullable = (
            {f.name: f.nullable for f in problem.features} if problem is not None else None
        )
        result = frame.copy()
        rng = self._rng(seed)
        for rule in rules:
            for column in (rule.feature, rule.depends_on):
                if column not in result.columns:
                    raise AegisMLError(
                        f"Missingness rule references column {column!r}, which the frame "
                        f"does not have. Rules are declared against the feature spec, so "
                        f"this means the generator and the spec disagree about the schema."
                    )
            if nullable is not None and not nullable.get(rule.feature, False):
                raise AegisMLError(
                    f"Missingness rule punches holes in {rule.feature!r}, but the spec "
                    f"declares it nullable=False. Either mark the FeatureSpec nullable "
                    f"(and let the pandera contract expect nulls) or drop the rule — a "
                    f"frame that violates its own contract fails at the boundary, not here."
                )
            probability = self._missing_probability(result, rule)
            mask = rng.random(len(result)) < probability
            if mask.any():
                result.loc[mask, rule.feature] = np.nan
                result[rule.feature] = pd.Series(result[rule.feature], index=result.index)
        return result

    def _missing_probability(self, frame: pd.DataFrame, rule: MissingnessRule) -> Any:  # noqa: ANN401
        """Per-row missingness probability implied by one MAR rule."""
        np = _numpy()
        pd = _pandas()
        driver = frame[rule.depends_on]
        if rule.depends_on_level is not None:
            hit = driver.astype("object").astype(str) == rule.depends_on_level
            return np.where(hit.to_numpy(), rule.max_rate, rule.rate)
        numeric = pd.to_numeric(driver, errors="coerce")
        if numeric.notna().sum() < 2 or numeric.nunique(dropna=True) < 2:
            raise AegisMLError(
                f"Missingness rule for {rule.feature!r} depends on {rule.depends_on!r}, "
                f"which is constant or unparseable on this frame — the resulting holes "
                f"would be MCAR, and imputation on MCAR data demonstrates nothing. Point "
                f"the rule at a varying column, or set depends_on_level for a categorical."
            )
        ranks = numeric.rank(pct=True, na_option="keep").fillna(0.5).to_numpy(dtype="float64")
        return rule.rate + (rule.max_rate - rule.rate) * ranks

    def attach_target(
        self,
        frame: pd.DataFrame,
        problem: MLProblem,
        *,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Label a feature frame and then punch its MAR holes — in that order.

        The ordering is the whole point of MAR and is stated here because it is easy to
        reverse by accident. The label is computed from the **complete** row, so the
        unrecorded value genuinely influenced the outcome; only afterwards is the
        observation degraded. Punching holes first would compute a label from a row with
        zeros where data should be, which makes the recording process part of the causal
        story and quietly turns an imputation demo into a misspecification demo.

        Args:
            frame: Feature frame carrying every column the spec declares.
            problem: The problem whose target column name and nullability rules apply.
            seed: Overrides :attr:`seed`.

        Returns:
            A new frame with the target column appended and missingness applied.
        """
        labelled = frame.copy()
        labelled[problem.target.name] = self.sample_frame(frame, seed=seed)
        return self.apply_missingness(labelled, problem=problem, seed=seed)


def default_latent_model(
    problem: MLProblem,
    *,
    target_r2: float | None = None,
    target_accuracy: float | None = None,
    irrelevant_share: float | None = None,
    seed: int | None = None,
) -> LatentModel:
    """Derive a realistic latent model from a problem spec alone.

    This is the starting point a domain author edits, not a substitute for thinking about
    the domain: real coefficients encode real claims ("technical cases take longest"), and
    a model card is more convincing when its drivers read like operations rather than like
    a random draw. What this does give you is a structurally correct model — standardised
    numeric drivers, level effects spread across each categorical, one interaction, one
    unobserved confounder, MAR holes on the nullable columns, and a deliberate handful of
    features left with no driver at all — that lands inside the realism band on the first
    run, so the pipeline is green before the coefficients are argued about.

    Coefficient magnitudes are solved from the declared target range: with ``k`` drivers
    each contributing roughly ``±coefficient`` on a standardised scale, the total signal
    standard deviation is about ``0.5·coefficient·√k``, so ``coefficient`` is set to put
    that near a sixth of the target's declared span. A target with no declared bounds is
    treated as unit-scaled and says so through the model's ``floor``/``ceiling`` staying
    ``None``.

    Args:
        problem: The declarative problem.
        target_r2: Requested achievable R²; defaults to :class:`RealismConfig`'s.
        target_accuracy: Requested achievable accuracy; defaults to :class:`RealismConfig`'s.
        irrelevant_share: Fraction of features deliberately left undriven.
        seed: Seed for the deterministic coefficient draw.

    Returns:
        A :class:`LatentModel` ready to label frames for this problem.

    Raises:
        AegisMLError: When the spec has too few features to build a non-degenerate model.
    """
    import random

    resolved_seed = settings.random_seed if seed is None else seed
    rng = random.Random(resolved_seed)
    realism_defaults = RealismConfig()
    share = realism_defaults.irrelevant_share if irrelevant_share is None else irrelevant_share

    features = list(problem.features)
    if len(features) < 2:
        raise AegisMLError(
            f"Domain {problem.domain_id!r} declares {len(features)} feature(s); a latent "
            f"model needs at least 2 so that some features can carry signal and others can "
            f"honestly carry none. Add features to the spec before generating data."
        )
    keep = max(2, int(round(len(features) * (1.0 - share))))
    order = list(features)
    rng.shuffle(order)
    driving, irrelevant = order[:keep], order[keep:]

    span = _target_span(problem)
    base = span / (3.0 * math.sqrt(max(len(driving), 1)))
    drivers = [_driver_for(feature, base, rng) for feature in driving]
    interactions = _interactions_for(driving, base, rng)

    numeric_driving = [f for f in driving if f.dtype == "numeric"]
    heteroscedastic = numeric_driving[0].name if numeric_driving else None
    missingness = _missingness_for(problem, driving)

    realism = RealismConfig(
        target_r2=realism_defaults.target_r2 if target_r2 is None else target_r2,
        target_accuracy=(
            realism_defaults.target_accuracy if target_accuracy is None else target_accuracy
        ),
        heteroscedastic_feature=heteroscedastic,
        label_flip_rate=realism_defaults.label_flip_rate,
        class_weights=_default_class_weights(problem),
        missingness=missingness,
        irrelevant_share=share,
    )
    logger.debug(
        "default_latent_model(%s): %d driven features, %d deliberately irrelevant (%s)",
        problem.domain_id,
        len(driving),
        len(irrelevant),
        ", ".join(f.name for f in irrelevant) or "none",
    )
    return LatentModel(
        intercept=_target_midpoint(problem),
        drivers=drivers,
        interactions=interactions,
        confounders=[
            Confounder(
                name="unrecorded_operating_conditions",
                coefficient=base * 1.2,
                distribution="normal",
                location=0.0,
                scale=1.0,
            )
        ],
        realism=realism,
        floor=problem.target.minimum,
        ceiling=problem.target.maximum,
        task=problem.target.task,
        levels=list(problem.target.levels),
        seed=resolved_seed,
    )


def _target_span(problem: MLProblem) -> float:
    """The target's declared range, or ``1.0`` when it declares none."""
    target = problem.target
    if target.task == "classification":
        return 1.0
    if target.minimum is not None and target.maximum is not None:
        return max(float(target.maximum) - float(target.minimum), 1e-6)
    return 1.0


def _target_midpoint(problem: MLProblem) -> float:
    """The intercept a spec implies: the midpoint of a bounded range, else zero."""
    target = problem.target
    if target.task == "classification":
        return 0.0
    if target.minimum is not None and target.maximum is not None:
        return (float(target.minimum) + float(target.maximum)) / 2.0
    return 0.0


def _driver_for(feature: Any, base: float, rng: Any) -> LatentDriver:  # noqa: ANN401
    """Build one driver for a feature, with a deterministic sign and shape."""
    magnitude = base * rng.uniform(0.6, 1.4)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    if feature.dtype == "categorical":
        levels = list(feature.levels)
        rng.shuffle(levels)
        step = 2.0 / max(len(levels) - 1, 1)
        effects = {level: -1.0 + step * index for index, level in enumerate(levels)}
        return LatentDriver(
            feature=feature.name, coefficient=sign * magnitude, level_effects=effects
        )
    center, scale = _bounds_to_center_scale(feature)
    # Mild, monotone non-linearity on a minority of drivers: enough that a linear model
    # visibly underperforms the boosted ensemble, not so much that SHAP stops reading.
    transform: TransformName = rng.choice(["identity", "identity", "tanh", "sqrt"])
    return LatentDriver(
        feature=feature.name,
        coefficient=sign * magnitude,
        transform=transform,
        center=center,
        scale=scale,
    )


def _bounds_to_center_scale(feature: Any) -> tuple[float | None, float | None]:  # noqa: ANN401
    """Derive standardisation from a feature's declared bounds, when it has them."""
    if feature.minimum is None or feature.maximum is None:
        return None, None
    low, high = float(feature.minimum), float(feature.maximum)
    half = (high - low) / 2.0
    return (low + high) / 2.0, half if half > 0.0 else None


def _interactions_for(driving: list[Any], base: float, rng: Any) -> list[Interaction]:  # noqa: ANN401
    """Build one interaction term, preferring numeric × categorical-level gating."""
    numerics = [f for f in driving if f.dtype == "numeric"]
    categoricals = [f for f in driving if f.dtype == "categorical"]
    if not numerics:
        return []
    left = numerics[0]
    center, scale = _bounds_to_center_scale(left)
    magnitude = base * rng.uniform(0.8, 1.5)
    if categoricals:
        gate = categoricals[0]
        return [
            Interaction(
                left=left.name,
                right=gate.name,
                right_level=gate.levels[0],
                coefficient=magnitude,
                left_center=center,
                left_scale=scale,
            )
        ]
    if len(numerics) < 2:
        return []
    right = numerics[1]
    right_center, right_scale = _bounds_to_center_scale(right)
    return [
        Interaction(
            left=left.name,
            right=right.name,
            coefficient=magnitude,
            left_center=center,
            left_scale=scale,
            right_center=right_center,
            right_scale=right_scale,
        )
    ]


def _missingness_for(problem: MLProblem, driving: list[Any]) -> list[MissingnessRule]:  # noqa: ANN401
    """One MAR rule per nullable feature, conditioned on a different driven column."""
    nullable = [f for f in problem.features if f.nullable]
    if not nullable:
        return []
    rules: list[MissingnessRule] = []
    for feature in nullable:
        candidates = [f for f in driving if f.name != feature.name and f.dtype == "numeric"]
        if not candidates:
            candidates = [f for f in problem.features if f.name != feature.name]
        if not candidates:
            continue
        driver = candidates[0]
        rules.append(
            MissingnessRule(
                feature=feature.name,
                depends_on=driver.name,
                depends_on_level=(driver.levels[0] if driver.dtype == "categorical" else None),
                rate=0.03,
                max_rate=0.22,
            )
        )
    return rules


def _default_class_weights(problem: MLProblem) -> dict[str, float]:
    """A deliberately uneven class balance for a classification target.

    A tidy 50/50 split is the one balance real operational data never has, and it hides the
    single most common evaluation mistake in the stack: a classifier that beats the floor
    purely by predicting the majority class. Weighting the levels geometrically produces a
    majority class near 60% for a binary target and a long tail for a multiclass one, which
    is enough imbalance to make the majority-baseline guard in :func:`assert_learnable`
    earn its place without making a rare class unlearnable.
    """
    levels = list(problem.target.levels)
    if len(levels) < 2:
        return {}
    weights = [0.6**index for index in range(len(levels))]
    total = sum(weights)
    return {level: weight / total for level, weight in zip(levels, weights, strict=True)}


class LearnabilityReport(BaseModel):
    """What a held-out fit measured, and both verdicts it supports.

    Attributes:
        metric_name: ``"r2"`` or ``"accuracy"``, matching ``ModelCard.metric_name``.
        metric_value: Measured on rows the model never saw.
        floor: The bar below which the target is treated as noise.
        ceiling: The bar above which it is treated as leaked or trivial.
        effective_floor: ``floor`` raised to the majority-class rate for classification;
            equal to ``floor`` for regression.
        majority_share: Share held by the most common class, or ``None`` for regression.
        learnable: Whether ``metric_value`` cleared ``effective_floor``.
        suspiciously_easy: Whether ``metric_value`` exceeded ``ceiling``. A held-out R² of
            0.99 on synthetic data is not a good result, it is a bug report.
        n_train: Rows the probe fitted on.
        n_test: Rows it scored on.
        n_classes: Distinct label values seen, or ``None`` for regression.
        notes: Human-readable findings, each carrying its number.
    """

    metric_name: str
    metric_value: float
    floor: float
    ceiling: float
    effective_floor: float
    majority_share: float | None = None
    learnable: bool
    suspiciously_easy: bool = False
    n_train: int = 0
    n_test: int = 0
    n_classes: int | None = None
    notes: list[str] = Field(default_factory=list)


def measure_learnability(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    floor: float | None = None,
    ceiling: float | None = None,
    test_size: float = 0.25,
    seed: int | None = None,
) -> LearnabilityReport:
    """Fit a fast probe on a train split and score it on held-out rows.

    ``HistGradientBoosting*`` is the probe because it is the cheapest estimator that can
    represent what the real spine represents. The question being answered is "does signal
    exist in these columns", and a linear probe would answer "no" for a target built from
    interactions and saturating transforms — a false alarm that sends a domain author back
    to rewrite a generator that was correct. It also consumes ``NaN`` natively, which
    matters once :class:`MissingnessRule` has done its work.

    This function measures and reports; it never raises on the score itself. Use
    :func:`assert_learnable` for the gate.

    Args:
        frame: Labelled frame carrying every feature and the target.
        problem: The problem describing them.
        floor: Overrides ``settings.learnable_r2_floor`` / ``learnable_accuracy_floor``.
        ceiling: Overrides :data:`R2_CEILING` / :data:`ACCURACY_CEILING`.
        test_size: Share of rows held out for the measurement.
        seed: Split and estimator seed.

    Returns:
        The :class:`LearnabilityReport`.

    Raises:
        InsufficientLabelsError: When the frame has too few rows for the measurement to
            mean anything — a score from 12 rows is a coin flip dressed as evidence.
        AegisMLError: When the target column is absent, or is constant.
    """
    from aegis_ml.data.splits import stratified_split
    from aegis_ml.features.pipeline import encode_frame

    pd = _pandas()
    metrics = _sklearn("metrics")
    ensemble = _sklearn("ensemble")

    task = problem.target.task
    name = problem.target.name
    resolved_seed = settings.random_seed if seed is None else seed
    resolved_floor = _resolved_floor(task, floor)
    resolved_ceiling = _resolved_ceiling(task, ceiling)

    if name not in frame.columns:
        raise AegisMLError(
            f"Frame has no target column {name!r} (columns: {sorted(frame.columns)[:12]}). "
            f"The generator and the spec disagree about the label's name — which is the "
            f"failure aegis_ml.contracts.spec exists to make impossible, so the frame was "
            f"probably not built from the generated ml_spec module."
        )
    labelled = frame[frame[name].notna()]
    if len(labelled) < MIN_LEARNABILITY_ROWS:
        raise InsufficientLabelsError(
            have=len(labelled), need=MIN_LEARNABILITY_ROWS, what="The learnability check"
        )

    distinct = int(labelled[name].nunique(dropna=True))
    if distinct < 2:
        raise AegisMLError(
            f"Target {name!r} takes a single value across all {len(labelled)} rows. Every "
            f"metric on a constant target is meaningless — accuracy reads 1.0 and R² reads "
            f"0.0 — so this is refused rather than scored. This is the same condition "
            f"`python -m app.ml` reports as distinct=False on its last line."
        )

    train, test = stratified_split(labelled, problem, test_size=test_size, seed=resolved_seed)
    x_train = encode_frame(train, problem)
    x_test = encode_frame(test, problem).reindex(columns=x_train.columns, fill_value=0.0)
    y_train, y_test = train[name], test[name]

    notes: list[str] = []
    majority: float | None = None
    if task == "regression":
        model = ensemble.HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, random_state=resolved_seed
        )
        model.fit(x_train, pd.to_numeric(y_train, errors="coerce"))
        value = float(
            metrics.r2_score(pd.to_numeric(y_test, errors="coerce"), model.predict(x_test))
        )
        metric_name, effective_floor = "r2", resolved_floor
    else:
        model = ensemble.HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=resolved_seed
        )
        model.fit(x_train, y_train.astype(str))
        value = float(metrics.accuracy_score(y_test.astype(str), model.predict(x_test)))
        majority = float(y_test.astype(str).value_counts(normalize=True).max())
        effective_floor = max(resolved_floor, majority + _MAJORITY_MARGIN)
        metric_name = "accuracy"
        notes.append(
            f"majority class holds {majority:.3f} of the held-out rows, so the floor is "
            f"raised to {effective_floor:.3f} — a constant predictor must not pass"
        )

    learnable = value >= effective_floor
    easy = value > resolved_ceiling
    if not learnable:
        notes.append(
            f"{metric_name}={value:.4f} did not clear {effective_floor:.4f}: the label is "
            f"not a recoverable function of these columns"
        )
    if easy:
        notes.append(
            f"{metric_name}={value:.4f} exceeds the sanity ceiling {resolved_ceiling:.4f}. "
            f"On synthetic data that is not a good result, it is a bug report: look for a "
            f"leaked feature (aegis_ml.features.leakage), a target that is arithmetic on "
            f"one column, or a noise term that never fired."
        )
    return LearnabilityReport(
        metric_name=metric_name,
        metric_value=value,
        floor=resolved_floor,
        ceiling=resolved_ceiling,
        effective_floor=effective_floor,
        majority_share=majority,
        learnable=learnable,
        suspiciously_easy=easy,
        n_train=int(len(train)),
        n_test=int(len(test)),
        n_classes=distinct if task == "classification" else None,
        notes=notes,
    )


def _resolved_floor(task: TaskType, floor: float | None) -> float:
    """The configured learnability floor for a task, unless overridden."""
    if floor is not None:
        return floor
    return (
        settings.learnable_r2_floor
        if task == "regression"
        else settings.learnable_accuracy_floor
    )


def _resolved_ceiling(task: TaskType, ceiling: float | None) -> float:
    """The sanity ceiling for a task, unless overridden."""
    if ceiling is not None:
        return ceiling
    return R2_CEILING if task == "regression" else ACCURACY_CEILING


def assert_learnable(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    floor: float | None = None,
    ceiling: float | None = None,
    strict_ceiling: bool = False,
    seed: int | None = None,
) -> float:
    """Refuse a frame whose label is noise — and shout about one that is too easy.

    The lower bound is the gate: below the floor, training is pointless and the conformal
    interval it produces will be honestly enormous, so this raises
    :class:`~aegis_ml.contracts.errors.LabelNotLearnableError` naming the measured number.

    The upper bound is deliberately *not* fatal by default. A too-easy target is nearly
    always leakage, but the diagnosis belongs to
    :func:`aegis_ml.features.leakage.detect_leakage`, which can name the offending column,
    and a domain genuinely does occasionally have a near-deterministic target. So the
    default is a logged warning plus ``suspiciously_easy`` on the report, and
    ``strict_ceiling=True`` escalates: it runs the leakage detector, raises
    :class:`~aegis_ml.contracts.errors.TargetLeakageError` naming the feature when one
    explains the score, and otherwise refuses with the reason spelled out.

    Args:
        frame: Labelled frame.
        problem: The problem describing it.
        floor: Overrides the configured floor.
        ceiling: Overrides the sanity ceiling.
        strict_ceiling: Whether exceeding the ceiling is fatal.
        seed: Split and estimator seed.

    Returns:
        The measured score (R² or accuracy).

    Raises:
        LabelNotLearnableError: When the score is below the floor.
        TargetLeakageError: When ``strict_ceiling`` is set and one feature explains the score.
        AegisMLError: When ``strict_ceiling`` is set, the score is above the ceiling, and no
            single feature explains it.
        InsufficientLabelsError: When there are too few labelled rows to measure.
    """
    report = measure_learnability(
        frame, problem, floor=floor, ceiling=ceiling, seed=seed
    )
    if not report.learnable:
        raise LabelNotLearnableError(
            report.metric_name, report.metric_value, report.effective_floor
        )
    if report.suspiciously_easy:
        message = (
            f"{report.metric_name}={report.metric_value:.4f} on held-out rows exceeds the "
            f"sanity ceiling {report.ceiling:.4f}. On generated data this is a bug report, "
            f"not a result: the interval it calibrates will be a hairline that informs no "
            f"decision."
        )
        if strict_ceiling:
            _refuse_too_easy(frame, problem, report, message, seed)
        logger.warning("%s Run aegis_ml.features.leakage.detect_leakage to find out why.", message)
    return report.metric_value


def _refuse_too_easy(
    frame: pd.DataFrame,
    problem: MLProblem,
    report: LearnabilityReport,
    message: str,
    seed: int | None,
) -> None:
    """Escalate a too-easy score into the most specific refusal the evidence supports.

    Raises:
        TargetLeakageError: When a single feature scores above the leakage threshold.
        AegisMLError: When no single feature explains it — the target is trivially
            recoverable from the feature set as a whole.
    """
    from aegis_ml.features.leakage import detect_leakage

    signals = detect_leakage(frame, problem, seed=seed)
    if signals:
        worst = max(signals, key=lambda signal: signal.score)
        from aegis_ml.contracts.errors import TargetLeakageError

        raise TargetLeakageError(worst.feature, worst.score, worst.threshold)
    raise AegisMLError(
        f"{message} No single feature clears the leakage threshold, so the target is "
        f"trivially recoverable from the feature set as a whole — usually an arithmetic "
        f"identity spread over two columns, or a generator whose noise term never fired. "
        f"Raise LatentModel.realism.target_r2's noise budget (see aegis_ml.data.latent) "
        f"or pass strict_ceiling=False once you have confirmed the target really is this "
        f"deterministic. Measured: {report.metric_name}={report.metric_value:.4f}."
    )


def realism_report(
    frame: pd.DataFrame,
    problem: MLProblem,
    latent: LatentModel | None = None,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Assemble the evidence that a generated frame is realistic, rather than asserting it.

    Every number here answers a question a reviewer will actually ask. *Is the model
    memorising?* — ``achieved`` against ``implied_r2_ceiling``. *Is the interval wide
    because the model is bad or because the world is uncertain?* — ``confounder_share`` and
    ``noise_to_signal``. *Does the imputation story have anything behind it?* —
    ``missingness``. *Is the classifier just predicting the majority?* — ``class_balance``
    against the report's ``majority_share``. *Do the SHAP values distinguish real drivers
    from filler?* — ``undriven_features``.

    Args:
        frame: The labelled frame to describe.
        problem: The problem it belongs to.
        latent: The model that produced it. Without it, the calibration-derived fields are
            omitted rather than guessed — this function never infers a latent function it
            was not given.
        seed: Seed for the learnability probe and the calibration draw.

    Returns:
        A JSON-safe dictionary of measured evidence.
    """
    pd = _pandas()
    report = measure_learnability(frame, problem, seed=seed)
    target = problem.target.name

    evidence: dict[str, Any] = {
        "domain_id": problem.domain_id,
        "n_rows": int(len(frame)),
        "task": problem.target.task,
        "achieved": {
            "metric": report.metric_name,
            "value": report.metric_value,
            "floor": report.effective_floor,
            "ceiling": report.ceiling,
            "learnable": report.learnable,
            "suspiciously_easy": report.suspiciously_easy,
        },
        "missingness": {
            column: float(frame[column].isna().mean())
            for column in problem.feature_names
            if column in frame.columns and frame[column].isna().any()
        },
        "notes": list(report.notes),
    }

    if problem.target.task == "classification" and target in frame.columns:
        shares = frame[target].astype(str).value_counts(normalize=True)
        evidence["class_balance"] = {str(k): float(v) for k, v in shares.items()}
        evidence["achieved"]["majority_share"] = report.majority_share

    if latent is None:
        evidence["notes"].append(
            "no LatentModel supplied: the noise, confounder and ceiling figures are "
            "omitted rather than inferred"
        )
        return evidence

    calibration = latent.calibrate(frame, seed=seed)
    signal_var = calibration.signal_variance
    unexplained = calibration.confounder_variance + calibration.noise_sigma**2
    evidence["latent"] = {
        "driven_features": latent.driven_features,
        "undriven_features": [
            name for name in problem.feature_names if name not in latent.driven_features
        ],
        "non_monotone_drivers": [
            driver.feature for driver in latent.drivers if not driver.is_monotone
        ],
        "n_interactions": len(latent.interactions),
        "confounders": [c.name for c in latent.confounders],
    }
    evidence["noise"] = {
        "sigma": calibration.noise_sigma,
        "signal_variance": signal_var,
        "confounder_variance": calibration.confounder_variance,
        "noise_to_signal": (
            float(math.sqrt(unexplained / signal_var)) if signal_var > 0.0 else None
        ),
        "confounder_share": (
            float(calibration.confounder_variance / (signal_var + unexplained))
            if signal_var + unexplained > 0.0
            else None
        ),
        "implied_r2_ceiling": calibration.implied_r2_ceiling,
        "heteroscedastic_feature": latent.realism.heteroscedastic_feature,
        "label_flip_rate": latent.realism.label_flip_rate,
        "notes": list(calibration.notes),
    }
    if problem.target.task == "regression" and target in frame.columns:
        residual = pd.to_numeric(frame[target], errors="coerce") - latent.signal_frame(frame)
        quartile = latent.signal_frame(frame).rank(pct=True)
        low = residual[quartile <= 0.25].std()
        high = residual[quartile >= 0.75].std()
        evidence["noise"]["heteroscedasticity_ratio"] = (
            float(high / low) if low and float(low) > 0.0 else None
        )
        evidence["achieved"]["headroom"] = (
            float(report.metric_value / calibration.implied_r2_ceiling)
            if calibration.implied_r2_ceiling > 0.0
            else None
        )
    return evidence
