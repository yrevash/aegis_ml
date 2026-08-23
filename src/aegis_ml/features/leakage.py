"""Find the feature that already knows the answer.

Target leakage is the failure that produces the *best* numbers and the worst outcome. A
column that is a restatement of the label — a ``resolved_at`` timestamp, a
``final_status``, a risk score computed downstream from the very thing being predicted —
sends held-out R² to 0.99, sails through every conformance check, produces a conformal
interval a hair wide, and then collapses the moment it meets a row where that column is not
yet populated. Which, by construction, is every row at prediction time.

Aegis catches exactly one form of this: ``MLProblem`` refuses a target that is also listed
as a feature, because that is perfect leakage spelled out. Everything subtler is invisible
to the platform, and the symptom — an implausibly good score — reads as success.

So detection here is empirical rather than name-based. Each feature is fitted *alone*
against the target and scored on rows it did not see. A single column that recovers the
label on its own is either the label wearing a hat, or a genuine deterministic relationship
the domain author must consciously declare. Two cheaper checks run alongside it: a near-±1
correlation with a numeric target, and outright value-level duplication after string
normalisation.

The detector deliberately reports rather than assumes. ``allow`` exists because leakage is
sometimes real and intended — a feature genuinely available at prediction time can look
identical to a leak from here — and the honest handling of that is a declaration in the
config, named in the model card, not a threshold quietly nudged upward.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError, TargetLeakageError
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence
    from types import ModuleType

    import pandas as pd

    from aegis_ml.contracts.spec import FeatureSpec, MLProblem

__all__ = [
    "MIN_LEAKAGE_ROWS",
    "LeakKind",
    "LeakSignal",
    "assert_no_leakage",
    "detect_leakage",
]

logger = logging.getLogger(__name__)

_EXTRA = "aegis-ml[serve]"
"""Install target quoted when scikit-learn or pandas are missing."""

MIN_LEAKAGE_ROWS = 40
"""Fewest rows for which a single-feature held-out score is worth reporting.

Below this a single feature can hit 1.0 by coincidence often enough that the detector would
cry wolf, and a leakage warning nobody believes is worse than no warning at all.
"""

LeakKind = Literal["single_feature", "correlation", "duplicate"]
"""How a leak was found.

``single_feature`` is the general case: a model fitted on this column alone recovers the
label. ``correlation`` is the linear special case, kept separate because a near-±1 Pearson
coefficient names the relationship ("this is the target times a constant") in a way a
tree's held-out score does not. ``duplicate`` means the column's *values* are the label's
values — the same fact stored twice, usually under a different name after a join.
"""


class LeakSignal(BaseModel):
    """One feature flagged as knowing too much, with the number behind the flag.

    ``score`` and ``threshold`` are both carried because "flagged" without the figure is
    indistinguishable from a bug in the detector — the same reason
    :class:`~aegis_ml.contracts.protocols.GateDecision` reports its metrics on a pass as
    well as a failure.

    Attributes:
        feature: The column that leaks.
        kind: How it was detected; see :data:`LeakKind`.
        score: The measured statistic — held-out R²/accuracy, ``|Pearson r|``, or the share
            of rows where feature and target are the same value.
        threshold: The bar it crossed.
        detail: One sentence a human can act on.
    """

    feature: str
    kind: LeakKind
    score: float
    threshold: float
    detail: str = Field(default="", description="Actionable one-line explanation.")


def _sklearn(submodule: str) -> ModuleType:
    """Import a scikit-learn submodule through :func:`~aegis_ml._require.require`."""
    return require(_EXTRA, f"sklearn.{submodule}")


def _pandas() -> ModuleType:
    """Import pandas through :func:`~aegis_ml._require.require`."""
    return require(_EXTRA, "pandas")


def _single_feature_problem(problem: MLProblem, feature: FeatureSpec) -> MLProblem:
    """A copy of ``problem`` carrying exactly one feature.

    Reusing :class:`~aegis_ml.contracts.spec.MLProblem` rather than hand-rolling a
    one-column encoder keeps the single-feature probe on the same representation the spine
    uses — a categorical is one-hot encoded here exactly as it would be there, so a score
    measured on one column is comparable to the full model's.
    """
    return problem.model_copy(update={"features": [feature]})


def _score_single_feature(
    frame: pd.DataFrame,
    problem: MLProblem,
    feature: FeatureSpec,
    seed: int,
) -> tuple[float, float | None]:
    """Fit and score a model on one feature alone.

    Returns:
        ``(score, majority_share)`` — held-out R² or accuracy, and for classification the
        held-out majority-class rate, which the caller needs in order not to mistake an
        imbalanced target for a leaking feature.
    """
    from aegis_ml.data.splits import stratified_split
    from aegis_ml.features.pipeline import encode_frame

    pd = _pandas()
    ensemble = _sklearn("ensemble")
    metrics = _sklearn("metrics")

    solo = _single_feature_problem(problem, feature)
    columns = [feature.name, problem.target.name]
    train, test = stratified_split(frame[columns], problem, test_size=0.25, seed=seed)
    x_train = encode_frame(train, solo)
    x_test = encode_frame(test, solo).reindex(columns=x_train.columns, fill_value=0.0)
    y_train, y_test = train[problem.target.name], test[problem.target.name]

    if problem.target.task == "regression":
        model = ensemble.HistGradientBoostingRegressor(max_iter=120, random_state=seed)
        model.fit(x_train, pd.to_numeric(y_train, errors="coerce"))
        score = float(
            metrics.r2_score(pd.to_numeric(y_test, errors="coerce"), model.predict(x_test))
        )
        return score, None
    model = ensemble.HistGradientBoostingClassifier(max_iter=120, random_state=seed)
    model.fit(x_train, y_train.astype(str))
    score = float(metrics.accuracy_score(y_test.astype(str), model.predict(x_test)))
    majority = float(y_test.astype(str).value_counts(normalize=True).max())
    return score, majority


def _duplicate_share(frame: pd.DataFrame, feature: str, target: str) -> float:
    """Share of rows where the feature and the target hold the same value.

    Compared as normalised strings so that ``3`` and ``3.0``, or ``"OK"`` and ``"ok "``,
    still register as the same fact stored twice — which is what a join usually produces
    and what a dtype-sensitive comparison would miss.
    """

    def _normalise(column: pd.Series) -> pd.Series:
        return column.astype("object").astype(str).str.strip().str.casefold()

    left = _normalise(frame[feature])
    right = _normalise(frame[target])
    return float((left == right).mean())


def _correlation(frame: pd.DataFrame, feature: str, target: str) -> float | None:
    """``|Pearson r|`` between a numeric feature and a numeric target, or ``None``."""
    pd = _pandas()
    left = pd.to_numeric(frame[feature], errors="coerce")
    right = pd.to_numeric(frame[target], errors="coerce")
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    value = left.corr(right)
    return None if value is None or value != value else abs(float(value))


def detect_leakage(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    threshold: float | None = None,
    allow: Iterable[str] = (),
    seed: int | None = None,
) -> list[LeakSignal]:
    """Score every feature on its own and report the ones that already know the answer.

    Three passes run per feature, cheapest first, and a feature is reported once per pass it
    trips — the duplicate finding and the single-feature finding say different things about
    the same column, and collapsing them would lose the diagnosis.

    Cost is one small gradient-boosted fit per feature, which on a hackathon-sized frame
    (1k–10k rows, a dozen features) is a couple of seconds. That is cheap next to running an
    AutoML search against a leaked target and promoting the result.

    For classification the raw accuracy is compared against the held-out majority-class
    rate before anything is flagged. On a 97/3 target every feature "predicts" the label
    with 0.97 accuracy by saying the majority word, and a detector that flagged all of them
    would be reporting the imbalance, not a leak.

    Args:
        frame: The labelled frame.
        problem: The problem describing it.
        threshold: Score above which a feature is flagged; defaults to
            ``settings.leakage_threshold``.
        allow: Features declared as intentionally near-deterministic. They are skipped, and
            skipping is logged — a suppressed check that leaves no trace is how a real leak
            eventually ships.
        seed: Seed for the probe splits and estimators.

    Returns:
        Every :class:`LeakSignal` found, highest score first. An empty list is a pass.

    Raises:
        AegisMLError: When the target column is absent, or there are too few rows for a
            single-feature score to mean anything.
    """
    resolved_threshold = settings.leakage_threshold if threshold is None else threshold
    resolved_seed = settings.random_seed if seed is None else seed
    target = problem.target.name
    allowed = set(allow)

    if target not in frame.columns:
        raise AegisMLError(
            f"Cannot check leakage without the target column {target!r}. "
            f"Present: {sorted(frame.columns)[:12]}"
        )
    labelled = frame[frame[target].notna()]
    if len(labelled) < MIN_LEAKAGE_ROWS:
        raise AegisMLError(
            f"Leakage detection needs at least {MIN_LEAKAGE_ROWS} labelled rows; "
            f"{len(labelled)} are available. A single-feature score on fewer rows hits 1.0 "
            f"by coincidence often enough to be worthless, and this refuses to emit a "
            f"finding it cannot stand behind."
        )
    if allowed:
        logger.info(
            "Leakage detection is skipping declared-intentional features %s for domain %s; "
            "this must be stated on the model card.",
            sorted(allowed),
            problem.domain_id,
        )

    signals: list[LeakSignal] = []
    for feature in problem.features:
        if feature.name in allowed or feature.name not in labelled.columns:
            continue
        signals.extend(
            _signals_for_feature(labelled, problem, feature, resolved_threshold, resolved_seed)
        )
    return sorted(signals, key=lambda signal: signal.score, reverse=True)


def _signals_for_feature(
    frame: pd.DataFrame,
    problem: MLProblem,
    feature: FeatureSpec,
    threshold: float,
    seed: int,
) -> list[LeakSignal]:
    """Run all three passes for one feature and collect whatever they find."""
    target = problem.target.name
    found: list[LeakSignal] = []

    duplicate = _duplicate_share(frame, feature.name, target)
    if duplicate >= threshold:
        found.append(
            LeakSignal(
                feature=feature.name,
                kind="duplicate",
                score=duplicate,
                threshold=threshold,
                detail=(
                    f"{duplicate:.1%} of rows hold the same value in {feature.name!r} and "
                    f"{target!r}. This is the label stored twice, almost always the result "
                    f"of a join; drop the column."
                ),
            )
        )

    if feature.dtype != "categorical" and problem.target.task == "regression":
        correlation = _correlation(frame, feature.name, target)
        if correlation is not None and correlation >= threshold:
            found.append(
                LeakSignal(
                    feature=feature.name,
                    kind="correlation",
                    score=correlation,
                    threshold=threshold,
                    detail=(
                        f"|Pearson r| = {correlation:.4f} against {target!r}: the feature is "
                        f"an affine restatement of the target, not a driver of it."
                    ),
                )
            )

    score, majority = _score_single_feature(frame, problem, feature, seed)
    baseline = 0.0 if majority is None else majority
    if score >= threshold and score > baseline:
        found.append(
            LeakSignal(
                feature=feature.name,
                kind="single_feature",
                score=score,
                threshold=threshold,
                detail=(
                    f"A model fitted on {feature.name!r} alone recovers {target!r} with "
                    f"{'r2' if problem.target.task == 'regression' else 'accuracy'}="
                    f"{score:.4f} on held-out rows. Either it is unavailable at prediction "
                    f"time (drop it) or the relationship is genuinely deterministic and "
                    f"must be declared intentional."
                ),
            )
        )
    return found


def assert_no_leakage(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    threshold: float | None = None,
    allow: Iterable[str] = (),
    seed: int | None = None,
) -> Sequence[LeakSignal]:
    """Refuse a frame in which any feature already knows the target.

    Raises on the strongest signal, because the strongest one is the one whose removal
    changes the model most and therefore the one to fix first. The remaining signals are
    logged at warning level before the raise, so a frame with four leaking columns does not
    take four runs to clean up.

    Args:
        frame: The labelled frame.
        problem: The problem describing it.
        threshold: Overrides ``settings.leakage_threshold``.
        allow: Features declared as intentionally near-deterministic.
        seed: Seed for the probe splits and estimators.

    Returns:
        The empty sequence, so a passing call can be asserted on directly.

    Raises:
        TargetLeakageError: When any feature crosses the threshold.
        AegisMLError: When the frame cannot support the measurement at all.
    """
    signals = detect_leakage(frame, problem, threshold=threshold, allow=allow, seed=seed)
    if not signals:
        return []
    for signal in signals[1:]:
        logger.warning("Additional leakage signal: %s", signal.detail)
    worst = signals[0]
    logger.warning("Leakage: %s", worst.detail)
    raise TargetLeakageError(worst.feature, worst.score, worst.threshold)
