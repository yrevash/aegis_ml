"""Per-segment performance — the aggregate score's blind spot, made visible.

One number for a whole dataset is an average over populations that do not experience the
model the same way. A model that gains two points of R² overall while losing fifteen on the
APAC segment has *improved* by every headline measure and has become materially worse for
everyone in APAC. The aggregate is not merely silent about that — it is the specific
instrument that cannot detect it, because the segment's error is diluted by its own small
share of the rows.

So the slice sweep exists, and the promotion gate reads its **worst** entry.

Two design rules follow from the same reasoning:

* **A skipped slice is recorded, never dropped.** Segments below ``min_rows`` produce a
  metric too noisy to act on, so they are not scored — but they are returned in
  :class:`SliceReport.skipped` with their row counts. A slice that vanishes silently is
  indistinguishable from a slice that passed, and the small segments are usually the ones a
  fairness question is about.
* **Declared-but-absent levels are recorded too.** A categorical level in the spec with no
  rows in the evaluation frame means the model was never evaluated on it. That is a fact
  about the evidence, not an empty result.

Numeric features are sliced by quantile buckets rather than fixed thresholds so each bucket
holds a comparable number of rows; the bucket edges are measured from the data and printed
in the level name, so a reader can see what "q3" actually covers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from aegis_ml.contracts.protocols import SliceMetric
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.evaluate.metrics import higher_is_better, primary, score

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    import numpy.typing as npt
    import pandas as pd

__all__ = [
    "SkippedSlice",
    "SliceReport",
    "slice_metrics",
    "slice_report",
    "worst_slice",
]

_DEFAULT_MIN_ROWS = 30
"""Below ~30 rows a segment metric is dominated by which rows happened to land there."""


class SkippedSlice(BaseModel):
    """A segment that was NOT scored, and why.

    This type exists so that "we did not look" can never be mistaken for "we looked and it
    was fine". The gate ignores these (there is nothing to compare), but the model card
    prints them, because an unevaluated segment is a limitation of the evidence.
    """

    feature: str
    level: str
    n_rows: int = Field(ge=0)
    min_rows: int = Field(ge=0)
    reason: str = Field(
        description="Either 'too few rows to measure' or 'declared level absent from the "
        "evaluation frame' — the second means the model was never tested on it at all."
    )


class SliceReport(BaseModel):
    """Every segment that was measured, every segment that was not, and the worst one.

    ``overall_metric_value`` is carried alongside so the *spread* is readable without
    recomputation: on genuinely noisy data a worst slice below the overall score is normal
    and expected, and the finding is the size of the gap, not its existence.
    """

    metric_name: str
    higher_is_better: bool
    min_rows: int = Field(ge=0)
    n_rows_total: int = Field(ge=0)
    overall_metric_value: float | None = None
    slices: list[SliceMetric] = Field(default_factory=list)
    skipped: list[SkippedSlice] = Field(default_factory=list)
    worst: SliceMetric | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def worst_gap(self) -> float | None:
        """How far the worst segment falls below the overall score, in metric units.

        Returns:
            A non-negative gap in the "worse" direction, or ``None`` when either the worst
            slice or the overall value is unknown. Sign is normalised so a larger number is
            always a bigger problem, whichever way the metric points.
        """
        if self.worst is None or self.overall_metric_value is None:
            return None
        if self.higher_is_better:
            return float(self.overall_metric_value - self.worst.metric_value)
        return float(self.worst.metric_value - self.overall_metric_value)


def _as_level(value: object) -> str | None:
    """Render one categorical cell as a level name, or ``None`` when it is missing.

    Missing values are mapped to ``None`` rather than to the string ``"nan"`` so a
    missingness pattern is never scored as if it were a real level of the feature — though
    it is worth measuring separately, which is why the caller counts the rows it drops.

    Args:
        value: One cell of a categorical or boolean column.

    Returns:
        The level name, or ``None`` for a null.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN is the only self-unequal float.
        return None
    return str(value)


def _quantile_levels(
    column: pd.Series,
    *,
    n_quantiles: int,
) -> pd.Series | None:
    """Bucket a numeric column into labelled quantile bands.

    Args:
        column: The numeric (or datetime) column to bucket.
        n_quantiles: Requested number of buckets.

    Returns:
        A string Series of bucket labels aligned with ``column``, or ``None`` when the
        column has too few distinct values to bucket at all (a near-constant feature slices
        into one band, which is the whole dataset again and tells a reader nothing).
    """
    import pandas as pd

    numeric = pd.to_numeric(column, errors="coerce")
    if numeric.notna().sum() == 0 or numeric.nunique(dropna=True) < 2:
        return None
    try:
        bands = pd.qcut(numeric, q=n_quantiles, duplicates="drop")
    except ValueError:
        return None
    if bands.dropna().nunique() < 2:
        return None
    names = {
        index: f"q{index + 1} {interval}"
        for index, interval in enumerate(bands.cat.categories)
    }
    # Code -1 marks a row pandas could not place (a null); it maps to None and is skipped.
    return bands.cat.codes.map(lambda code: names.get(int(code))).astype("object")


def _segment_frames(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    n_quantiles: int,
) -> list[tuple[str, pd.Series]]:
    """Build the (feature, per-row segment label) pairs the sweep iterates over.

    Args:
        frame: The evaluation frame.
        problem: The declared problem; dtypes decide categorical vs quantile slicing.
        n_quantiles: Buckets per numeric feature.

    Returns:
        One ``(feature_name, labels)`` pair per sliceable feature. Features that cannot be
        sliced (a constant numeric column) are simply not returned; the caller records them.
    """
    pairs: list[tuple[str, pd.Series]] = []
    for spec in problem.features:
        if spec.name not in frame.columns:
            continue
        column = frame[spec.name]
        if spec.dtype in {"categorical", "boolean"}:
            pairs.append((spec.name, column.astype("object").map(_as_level)))
        else:
            labels = _quantile_levels(column, n_quantiles=n_quantiles)
            if labels is not None:
                pairs.append((spec.name, labels))
    return pairs


def slice_metrics(
    frame: pd.DataFrame,
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    problem: MLProblem,
    *,
    min_rows: int = _DEFAULT_MIN_ROWS,
    n_quantiles: int = 4,
    y_proba: npt.ArrayLike | None = None,
) -> list[SliceMetric]:
    """Compute the primary metric inside every sufficiently populated segment.

    Categorical and boolean features slice by level; numeric and datetime features slice by
    quantile band, with the measured band edges written into the level name so ``q3`` is
    self-describing.

    This returns only the *measured* slices. Use :func:`slice_report` when the skipped ones
    matter — and they usually do, because a segment too small to measure is exactly the
    segment nobody has evidence about.

    Args:
        frame: Evaluation frame, one row per prediction, carrying the feature columns.
        y_true: Measured targets aligned with ``frame`` by position.
        y_pred: Predictions aligned with ``frame`` by position.
        problem: The declared problem; supplies features, dtypes and the primary metric.
        min_rows: Minimum rows for a segment to be scored.
        n_quantiles: Quantile bands per numeric feature.
        y_proba: Optional probabilities, so probability-based primary metrics
            (``roc_auc``, ``log_loss``) can be sliced too.

    Returns:
        One :class:`~aegis_ml.contracts.protocols.SliceMetric` per measured segment.

    Raises:
        ValueError: When ``frame``, ``y_true`` and ``y_pred`` are not the same length —
            a misalignment would attribute rows to the wrong segment and produce entirely
            plausible numbers.
    """
    return slice_report(
        frame,
        y_true,
        y_pred,
        problem,
        min_rows=min_rows,
        n_quantiles=n_quantiles,
        y_proba=y_proba,
    ).slices


def slice_report(
    frame: pd.DataFrame,
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    problem: MLProblem,
    *,
    min_rows: int = _DEFAULT_MIN_ROWS,
    n_quantiles: int = 4,
    y_proba: npt.ArrayLike | None = None,
) -> SliceReport:
    """Run the full slice sweep and return measured slices, skipped slices and the worst.

    Args:
        frame: Evaluation frame, one row per prediction, carrying the feature columns.
        y_true: Measured targets aligned with ``frame`` by position.
        y_pred: Predictions aligned with ``frame`` by position.
        problem: The declared problem.
        min_rows: Minimum rows for a segment to be scored.
        n_quantiles: Quantile bands per numeric feature.
        y_proba: Optional probabilities for probability-based metrics.

    Returns:
        A :class:`SliceReport`. ``skipped`` is populated for both under-populated segments
        and declared categorical levels absent from the frame.

    Raises:
        ValueError: When the inputs are not aligned by length.
    """
    import numpy as np

    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    proba = None if y_proba is None else np.asarray(y_proba, dtype=float)
    n_rows = int(len(frame))
    if not (true.shape[0] == pred.shape[0] == n_rows):
        raise ValueError(
            f"Misaligned inputs: frame has {n_rows} rows, y_true {true.shape[0]}, y_pred "
            f"{pred.shape[0]}. Slicing on a misaligned frame attributes each row's error to "
            f"the wrong segment and every resulting number looks reasonable."
        )
    if proba is not None and proba.shape[0] != n_rows:
        raise ValueError(f"y_proba has {proba.shape[0]} rows for {n_rows} frame rows.")

    metric_name = problem.metric
    direction = higher_is_better(metric_name)

    overall_value: float | None = None
    notes: list[str] = []
    try:
        overall_value = primary(problem, score(problem, true, pred, y_proba=proba))[1]
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        notes.append(
            f"Overall {metric_name} could not be computed on this frame ({exc}); slice "
            f"values are still comparable to each other but not to a headline score."
        )

    labels_declared = {
        spec.name: list(spec.levels) for spec in problem.features if spec.levels
    }
    measured: list[SliceMetric] = []
    skipped: list[SkippedSlice] = []

    for feature, labels in _segment_frames(frame, problem, n_quantiles=n_quantiles):
        label_values = labels.to_numpy()
        seen: set[str] = set()
        for level in sorted({str(v) for v in label_values if v is not None and v == v}):
            seen.add(level)
            mask = np.asarray([str(v) == level for v in label_values], dtype=bool)
            count = int(mask.sum())
            if count < min_rows:
                skipped.append(
                    SkippedSlice(
                        feature=feature,
                        level=level,
                        n_rows=count,
                        min_rows=min_rows,
                        reason="too few rows to measure",
                    )
                )
                continue
            sub_proba = None if proba is None else proba[mask]
            try:
                metrics = score(problem, true[mask], pred[mask], y_proba=sub_proba)
                value = primary(problem, metrics)[1]
            except Exception as exc:  # noqa: BLE001 - recorded as a skip, never hidden
                skipped.append(
                    SkippedSlice(
                        feature=feature,
                        level=level,
                        n_rows=count,
                        min_rows=min_rows,
                        reason=f"{metric_name} undefined on this segment: {exc}",
                    )
                )
                continue
            measured.append(
                SliceMetric(
                    feature=feature,
                    level=level,
                    n_rows=count,
                    metric_name=metric_name,
                    metric_value=float(value),
                )
            )
        for declared in labels_declared.get(feature, []):
            if str(declared) not in seen:
                skipped.append(
                    SkippedSlice(
                        feature=feature,
                        level=str(declared),
                        n_rows=0,
                        min_rows=min_rows,
                        reason="declared level absent from the evaluation frame",
                    )
                )

    worst = worst_slice(measured)
    if skipped:
        notes.append(
            f"{len(skipped)} segment(s) were not scored (see `skipped`). They are listed "
            f"rather than dropped: an unmeasured segment is a gap in the evidence, not a "
            f"pass."
        )
    if worst is not None and overall_value is not None:
        gap = (
            overall_value - worst.metric_value
            if direction
            else worst.metric_value - overall_value
        )
        notes.append(
            f"Worst segment {worst.feature}={worst.level} (n={worst.n_rows}) scores "
            f"{worst.metric_value:.4f} against an overall {overall_value:.4f} — a gap of "
            f"{gap:.4f} in the worse direction. The gate compares THIS number against the "
            f"champion's worst, because an average improvement that hides a collapsed "
            f"segment is a regression for everyone in it."
        )
    return SliceReport(
        metric_name=metric_name,
        higher_is_better=direction,
        min_rows=min_rows,
        n_rows_total=n_rows,
        overall_metric_value=overall_value,
        slices=measured,
        skipped=skipped,
        worst=worst,
        notes=notes,
    )


def worst_slice(slices: Sequence[SliceMetric]) -> SliceMetric | None:
    """Return the worst-performing segment, respecting the metric's direction.

    Direction comes from :data:`~aegis_ml.evaluate.metrics.HIGHER_IS_BETTER`, never from an
    assumption: "worst" for ``r2`` is the minimum and for ``rmse`` the maximum, and getting
    that backwards hands the gate the *best* segment as if it were the worst, which passes
    criterion 4 exactly when it should fail.

    Args:
        slices: Measured slices. They must all name the same metric — a mixed list has no
            worst element, only an incomparable one.

    Returns:
        The worst slice, or ``None`` when the sequence is empty (no segment was measurable,
        which the caller should surface as missing evidence rather than as a pass).

    Raises:
        ValueError: When the slices name more than one metric.
        UnknownMetricError: When the metric has no declared direction.
    """
    items = list(slices)
    if not items:
        return None
    names = {s.metric_name for s in items}
    if len(names) > 1:
        raise ValueError(
            f"Slices name multiple metrics {sorted(names)}; there is no 'worst' across two "
            f"scales. Compute one sweep per metric."
        )
    direction = higher_is_better(next(iter(names)))
    if direction:
        return min(items, key=lambda s: s.metric_value)
    return max(items, key=lambda s: s.metric_value)
