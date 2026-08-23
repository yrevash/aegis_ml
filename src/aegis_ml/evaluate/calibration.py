"""Calibration: does a stated confidence mean what it says?

Accuracy answers "how often is the model right". Calibration answers the different and, for
a decision-support system, more consequential question: **when the model says 90%, is it
right 90% of the time?** A model can be accurate and badly calibrated, and the failure mode
is specific — an agent that reads a confident-looking interval and acts on it.

Two families live here because Aegis states two kinds of confidence:

* **Probabilistic** (classification): Brier score, expected calibration error (ECE) and the
  reliability curve behind it.
* **Conformal** (both tasks): the *measured* rate at which the interval or prediction set
  actually contained the truth.

The naming rule is absolute and inherited from Aegis (``ModelCard.conformal_coverage`` vs
``conformal_coverage_empirical``, ``BacktestReport.requested_coverage`` vs
``empirical_coverage``): **the level requested and the level measured are always two
separate fields.** :class:`CoverageReport` never collapses them, and ``meets_request`` is a
derived boolean so no reader has to reconstruct the comparison — or reconstruct it
differently from the gate.

Why this module matters more than usual on heteroscedastic data: split-conformal calibrates
one interval width for the whole distribution. Where the noise is larger than average, that
fixed width under-covers; where it is smaller, it over-covers. The *marginal* coverage can
land exactly on 0.90 while the high-variance tail is covered at 0.70. Marginal coverage is
therefore reported as what it is — a marginal quantity — and :func:`coverage_by_slice`
exists to expose where it is being paid for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from aegis_ml.contracts.errors import InsufficientLabelsError
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "CalibrationReport",
    "CoverageReport",
    "ReliabilityBin",
    "brier_score",
    "calibration_report",
    "coverage",
    "coverage_by_slice",
    "coverage_report",
    "expected_calibration_error",
    "mean_interval_width",
    "mean_set_size",
    "reliability_curve",
]

_MIN_COVERAGE_ROWS = 20
"""Below this, a coverage *measurement* is not a measurement.

At 10 rows the only achievable values are multiples of 0.1, so "0.9 measured against 0.9
requested" is as likely to be arithmetic as evidence. Callers with fewer rows get
:class:`~aegis_ml.contracts.errors.InsufficientLabelsError`, which points at NannyML's
label-free *estimate* — a number that says it is an estimate.
"""


class ReliabilityBin(BaseModel):
    """One confidence bucket of the reliability curve.

    The gap between ``mean_confidence`` and ``empirical_accuracy`` is the calibration error
    *for this bucket*; ``n_rows`` says how much to trust it. A bin holding four rows can
    show a 0.5 gap from noise alone, which is why ECE weights bins by population rather
    than averaging the gaps evenly.
    """

    lower: float = Field(ge=0.0, le=1.0, description="Inclusive lower edge of the bucket.")
    upper: float = Field(ge=0.0, le=1.0, description="Exclusive upper edge (inclusive at 1).")
    n_rows: int = Field(ge=0)
    mean_confidence: float = Field(
        ge=0.0, le=1.0, description="Mean predicted probability of the predicted class."
    )
    empirical_accuracy: float = Field(
        ge=0.0, le=1.0, description="MEASURED share of rows in this bucket that were correct."
    )

    @property
    def gap(self) -> float:
        """Signed over-confidence: predicted minus measured.

        Returns:
            Positive when the model claimed more confidence than it earned, which is the
            direction that misleads a downstream decision.
        """
        return self.mean_confidence - self.empirical_accuracy


class CalibrationReport(BaseModel):
    """Probabilistic calibration of a classifier on a held-out split.

    ``expected_calibration_error`` is the population-weighted mean bucket gap and
    ``max_calibration_error`` the worst single bucket. Both are reported because they fail
    differently: a model can have a tiny ECE while one sparsely populated high-confidence
    bucket — precisely the bucket a downstream action keys off — is badly wrong.
    """

    n_rows: int = Field(ge=0)
    n_bins: int = Field(ge=1)
    binning: Literal["uniform", "quantile"] = "uniform"
    brier_score: float = Field(
        ge=0.0, description="MEASURED mean squared error of the probability vector."
    )
    expected_calibration_error: float = Field(ge=0.0, le=1.0)
    max_calibration_error: float = Field(ge=0.0, le=1.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    bins: list[ReliabilityBin] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def overconfident(self) -> bool:
        """Whether mean stated confidence exceeds measured accuracy overall.

        Returns:
            ``True`` when the model's average claim is larger than its average performance.
        """
        return self.mean_confidence > self.accuracy


class CoverageReport(BaseModel):
    """Requested versus measured conformal coverage — two fields, never one.

    ``requested_coverage`` is what was asked for; ``empirical_coverage`` is what was
    measured on held-out rows; ``meets_request`` is the comparison, computed once here so
    the gate, the model card and the CLI cannot disagree about it.

    ``tolerance`` is not slack granted to the model — it is the sampling error of the
    measurement itself. With 200 held-out rows the standard error of a 0.90 coverage
    estimate is about 0.021, so a measured 0.88 is not evidence of under-coverage. The
    tolerance exists so that noise is not read as failure; it is deliberately *not* wide
    enough to let a genuinely under-covering interval through.
    """

    requested_coverage: float = Field(
        gt=0.0, lt=1.0, description="The level ASKED FOR. Not a measurement."
    )
    empirical_coverage: float = Field(
        ge=0.0, le=1.0, description="The rate ACTUALLY ACHIEVED on held-out rows."
    )
    tolerance: float = Field(ge=0.0, description="Allowed shortfall, i.e. sampling error.")
    floor: float = Field(description="requested_coverage - tolerance; the pass threshold.")
    meets_request: bool = Field(description="empirical_coverage >= floor.")
    n_rows: int = Field(ge=0, description="Rows the empirical rate was measured on.")
    kind: Literal["interval", "set"] = Field(
        description="'interval' for regression, 'set' for classification."
    )
    mean_width: float | None = Field(
        default=None,
        description="Mean interval width (regression) — coverage bought with an unusable "
        "interval is not a win, and only the width shows that.",
    )
    mean_set_size: float | None = Field(
        default=None,
        description="Mean prediction-set size (classification); 1.0 means confident.",
    )
    notes: list[str] = Field(default_factory=list)

    @property
    def gap(self) -> float:
        """Signed shortfall: measured minus requested.

        Returns:
            Negative when the interval covered less often than it promised.
        """
        return self.empirical_coverage - self.requested_coverage


def _confidence_and_correct(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    labels: Sequence[object] | None,
) -> tuple:
    """Reduce any probability shape to (confidence, correct) for the predicted class.

    Args:
        y_true: Measured labels.
        y_proba: ``(n,)`` positive-class probabilities, or ``(n, k)`` class probabilities.
        labels: Column ordering of ``y_proba``. Inferred from ``y_true`` when omitted —
            which is only safe if every class appears in this split.

    Returns:
        ``(confidence, correct, classes, proba)`` where ``confidence`` is the probability
        assigned to the model's own top class and ``correct`` is the 0/1 hit vector.

    Raises:
        ValueError: When shapes disagree.
    """
    import numpy as np

    true = np.asarray(y_true).reshape(-1)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    classes = np.asarray(list(labels)) if labels is not None else np.unique(true)
    if proba.shape[0] != true.shape[0]:
        raise ValueError(
            f"y_proba has {proba.shape[0]} rows and y_true has {true.shape[0]}."
        )
    if proba.shape[1] != classes.shape[0]:
        raise ValueError(
            f"y_proba has {proba.shape[1]} columns but {classes.shape[0]} classes were "
            f"resolved ({list(classes)}). Pass labels= with the estimator's classes_ "
            f"ordering; an inferred ordering permutes the columns silently."
        )
    top = proba.argmax(axis=1)
    confidence = proba[np.arange(proba.shape[0]), top]
    predicted = classes[top]
    correct = (predicted.astype(str) == true.astype(str)).astype(float)
    return confidence, correct, classes, proba


def brier_score(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    *,
    labels: Sequence[object] | None = None,
) -> float:
    """Mean squared error between the probability vector and the one-hot truth.

    The multiclass (Brier–Gneiting) form is used for every class count, including two, so a
    binary and a three-class model are scored on the same scale. The binary-only convention
    that squares a single probability is half this value, and mixing the two conventions
    across a leaderboard makes one model look twice as good as another for free.

    Args:
        y_true: Measured labels.
        y_proba: ``(n,)`` or ``(n, k)`` probabilities.
        labels: Column ordering of ``y_proba``.

    Returns:
        The measured Brier score. Lower is better; 0 is perfect.

    Raises:
        InsufficientLabelsError: When there are no rows.
    """
    import numpy as np

    _, _, classes, proba = _confidence_and_correct(y_true, y_proba, labels)
    true = np.asarray(y_true).reshape(-1)
    if true.shape[0] == 0:
        raise InsufficientLabelsError(0, 1, "Brier score")
    onehot = (classes.astype(str)[None, :] == true.astype(str)[:, None]).astype(float)
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def reliability_curve(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    *,
    n_bins: int = 10,
    binning: Literal["uniform", "quantile"] = "uniform",
    labels: Sequence[object] | None = None,
) -> list[ReliabilityBin]:
    """Bucket predictions by stated confidence and measure accuracy inside each bucket.

    Empty buckets are dropped rather than reported as ``0.0`` accuracy: a bucket with no
    rows has no measured accuracy, and rendering zero there draws a reliability curve that
    plunges to the floor in regions the model simply never predicted.

    Args:
        y_true: Measured labels.
        y_proba: ``(n,)`` or ``(n, k)`` probabilities.
        n_bins: Number of confidence buckets.
        binning: ``uniform`` splits the [0,1] confidence range evenly — the standard, and
            the one whose bin edges are comparable across models. ``quantile`` splits by
            equal population, which is more stable when confidences cluster (a
            well-trained model puts most of its mass above 0.8, leaving the uniform low
            bins nearly empty).
        labels: Column ordering of ``y_proba``.

    Returns:
        The populated bins, in ascending confidence order.

    Raises:
        InsufficientLabelsError: When there are no rows.
        ValueError: When ``n_bins`` is below 1.
    """
    import numpy as np

    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1; got {n_bins}.")
    confidence, correct, _, _ = _confidence_and_correct(y_true, y_proba, labels)
    if confidence.shape[0] == 0:
        raise InsufficientLabelsError(0, 1, "Reliability curve")

    if binning == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(confidence, quantiles))
        if edges.shape[0] < 2:
            edges = np.asarray([confidence.min(), confidence.max() + 1e-12])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: list[ReliabilityBin] = []
    for i in range(edges.shape[0] - 1):
        low, high = float(edges[i]), float(edges[i + 1])
        last = i == edges.shape[0] - 2
        mask = (confidence >= low) & ((confidence <= high) if last else (confidence < high))
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=max(0.0, min(1.0, low)),
                upper=max(0.0, min(1.0, high)),
                n_rows=count,
                mean_confidence=float(confidence[mask].mean()),
                empirical_accuracy=float(correct[mask].mean()),
            )
        )
    return bins


def expected_calibration_error(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    *,
    n_bins: int = 10,
    binning: Literal["uniform", "quantile"] = "uniform",
    labels: Sequence[object] | None = None,
) -> float:
    """Population-weighted mean absolute gap between stated confidence and measured accuracy.

    Weighting by bin population is the whole point: an unweighted mean lets a bucket holding
    three rows contribute as much as one holding three hundred, and on a class-imbalanced
    problem the sparse buckets are exactly the noisy ones.

    Args:
        y_true: Measured labels.
        y_proba: ``(n,)`` or ``(n, k)`` probabilities.
        n_bins: Number of confidence buckets.
        binning: ``uniform`` or ``quantile`` — see :func:`reliability_curve`.
        labels: Column ordering of ``y_proba``.

    Returns:
        ECE in [0, 1]. Lower is better; 0 means every stated confidence was earned.
    """
    bins = reliability_curve(
        y_true, y_proba, n_bins=n_bins, binning=binning, labels=labels
    )
    total = sum(b.n_rows for b in bins)
    if total == 0:
        raise InsufficientLabelsError(0, 1, "Expected calibration error")
    return float(sum(b.n_rows * abs(b.gap) for b in bins) / total)


def calibration_report(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    *,
    n_bins: int = 10,
    binning: Literal["uniform", "quantile"] = "uniform",
    labels: Sequence[object] | None = None,
) -> CalibrationReport:
    """Compute the full probabilistic calibration picture in one pass.

    Args:
        y_true: Measured labels from the held-out split.
        y_proba: ``(n,)`` or ``(n, k)`` probabilities aligned with ``y_true``.
        n_bins: Number of confidence buckets.
        binning: ``uniform`` or ``quantile``.
        labels: Column ordering of ``y_proba``; pass the estimator's ``classes_``.

    Returns:
        A :class:`CalibrationReport`. ``notes`` carries the honest caveats that a number
        alone cannot: sparse bins, and whether the model is over- or under-confident.

    Raises:
        InsufficientLabelsError: When there are no rows.
    """
    confidence, correct, _, _ = _confidence_and_correct(y_true, y_proba, labels)
    n_rows = int(confidence.shape[0])
    if n_rows == 0:
        raise InsufficientLabelsError(0, 1, "Calibration report")
    bins = reliability_curve(y_true, y_proba, n_bins=n_bins, binning=binning, labels=labels)
    total = sum(b.n_rows for b in bins)
    ece = float(sum(b.n_rows * abs(b.gap) for b in bins) / total)
    mce = max((abs(b.gap) for b in bins), default=0.0)

    notes: list[str] = []
    sparse = [b for b in bins if b.n_rows < 10]
    if sparse:
        notes.append(
            f"{len(sparse)} of {len(bins)} populated bins hold fewer than 10 rows; their "
            f"gaps are dominated by sampling noise and the max-calibration-error may come "
            f"from one of them."
        )
    mean_conf = float(confidence.mean())
    accuracy = float(correct.mean())
    if mean_conf > accuracy:
        notes.append(
            f"Over-confident overall: mean stated confidence {mean_conf:.3f} exceeds "
            f"measured accuracy {accuracy:.3f}. A downstream action keyed off the stated "
            f"probability will fire more often than the evidence supports."
        )
    else:
        notes.append(
            f"Under-confident overall: mean stated confidence {mean_conf:.3f} is below "
            f"measured accuracy {accuracy:.3f}. Safe for decisions, but it suppresses "
            f"actions the model was in fact right about."
        )
    return CalibrationReport(
        n_rows=n_rows,
        n_bins=n_bins,
        binning=binning,
        brier_score=brier_score(y_true, y_proba, labels=labels),
        expected_calibration_error=ece,
        max_calibration_error=float(mce),
        mean_confidence=mean_conf,
        accuracy=accuracy,
        bins=bins,
        notes=notes,
    )


def _is_interval(item: object) -> bool:
    """Decide whether one conformal element is an interval rather than a set.

    Args:
        item: One element of the ``intervals_or_sets`` sequence.

    Returns:
        ``True`` for a 2-element numeric ``(low, high)``; ``False`` for a set/list of labels.
    """
    if isinstance(item, tuple | list) and len(item) == 2:
        return all(isinstance(x, int | float) and not isinstance(x, bool) for x in item)
    return False


def coverage(
    y_true: npt.ArrayLike,
    intervals_or_sets: Sequence[object],
    *,
    min_rows: int = _MIN_COVERAGE_ROWS,
) -> float:
    """Measure the EMPIRICAL rate at which the conformal output contained the truth.

    This is the measurement half of the pair. It never sees the requested level and cannot
    be tuned by it — which is the point: the requested level is a promise, and this function
    is the only thing entitled to say whether the promise was kept.

    Both conformal shapes are accepted and detected per element:

    * regression — ``(low, high)`` numeric pairs; a hit is ``low <= y <= high``
    * classification — sets/lists of candidate labels; a hit is ``y in set``

    Args:
        y_true: Measured targets from held-out rows the conformal predictor was NOT
            calibrated on. Measuring coverage on the calibration split reproduces the
            requested level by construction and proves nothing.
        intervals_or_sets: One interval or prediction set per row, aligned with ``y_true``.
        min_rows: Refuse below this many rows. At 10 rows the measurement can only take
            multiples of 0.1, so agreement with the requested level is arithmetic.

    Returns:
        The measured coverage in [0, 1].

    Raises:
        InsufficientLabelsError: When fewer than ``min_rows`` rows are available.
        ValueError: When lengths disagree, or an element is neither an interval nor a set.
    """
    import numpy as np

    true = np.asarray(y_true).reshape(-1)
    items = list(intervals_or_sets)
    if len(items) != true.shape[0]:
        raise ValueError(
            f"{len(items)} conformal outputs for {true.shape[0]} labels — a misalignment "
            f"here produces a coverage number computed against the wrong rows."
        )
    if true.shape[0] < min_rows:
        raise InsufficientLabelsError(int(true.shape[0]), min_rows, "Conformal coverage")

    hits = 0
    for value, item in zip(true.tolist(), items, strict=True):
        if _is_interval(item):
            low, high = float(item[0]), float(item[1])  # type: ignore[index]
            hits += int(low <= float(value) <= high)
        elif isinstance(item, set | frozenset | list | tuple):
            members = {str(x) for x in item}
            hits += int(str(value) in members)
        else:
            raise ValueError(
                f"Conformal output {item!r} is neither a numeric (low, high) interval nor "
                f"a set of candidate labels. Coverage of an unrecognised shape cannot be "
                f"measured, and guessing would report a number for something unmeasured."
            )
    return hits / len(items)


def mean_interval_width(intervals: Sequence[object]) -> float:
    """Mean width of conformal intervals — the price paid for the coverage.

    Coverage alone can always be met by widening: an interval spanning the whole target
    range covers 100% and supports no decision. The width is what makes a coverage number
    interpretable, which is why :class:`CoverageReport` carries it.

    Args:
        intervals: ``(low, high)`` pairs.

    Returns:
        The mean ``high - low``.

    Raises:
        ValueError: When the sequence is empty or an element is not an interval.
    """
    items = list(intervals)
    if not items:
        raise ValueError("mean_interval_width needs at least one interval.")
    widths = []
    for item in items:
        if not _is_interval(item):
            raise ValueError(f"{item!r} is not a numeric (low, high) interval.")
        widths.append(float(item[1]) - float(item[0]))  # type: ignore[index]
    return float(sum(widths) / len(widths))


def mean_set_size(sets: Sequence[object]) -> float:
    """Mean cardinality of conformal prediction sets; 1.0 means confident.

    Args:
        sets: Sets/lists of candidate labels, one per row.

    Returns:
        The mean number of candidate labels per row. A set size equal to the class count
        means the predictor declined to exclude anything at that confidence level.

    Raises:
        ValueError: When the sequence is empty or an element is not a set-like.
    """
    items = list(sets)
    if not items:
        raise ValueError("mean_set_size needs at least one prediction set.")
    sizes = []
    for item in items:
        if not isinstance(item, set | frozenset | list | tuple):
            raise ValueError(f"{item!r} is not a set of candidate labels.")
        sizes.append(len(set(map(str, item))))
    return float(sum(sizes) / len(sizes))


def coverage_report(
    requested: float,
    empirical: float,
    tolerance: float | None = None,
    *,
    n_rows: int = 0,
    kind: Literal["interval", "set"] = "interval",
    mean_width: float | None = None,
    mean_size: float | None = None,
) -> CoverageReport:
    """Pair a requested coverage with a measured one and decide whether it was met.

    The comparison lives here, once, so the gate, the model card and the CLI cannot each
    implement it slightly differently — the failure where the card prints "coverage met"
    and the gate rejects for coverage in the same run.

    Args:
        requested: The level ASKED FOR (``MLProblem.requested_coverage``).
        empirical: The level MEASURED by :func:`coverage` on held-out rows.
        tolerance: Allowed shortfall; defaults to ``settings.coverage_tolerance``. It stands
            for the sampling error of the measurement, not for slack granted to the model.
        n_rows: Rows the empirical rate was measured on. Carried so a reader can judge the
            measurement, and used to flag a rate measured on too few rows.
        kind: ``interval`` (regression) or ``set`` (classification).
        mean_width: Mean interval width, when known.
        mean_size: Mean prediction-set size, when known.

    Returns:
        A :class:`CoverageReport` with ``meets_request`` already decided.

    Raises:
        ValueError: When ``requested`` is not a proper probability.
    """
    if not 0.0 < requested < 1.0:
        raise ValueError(f"requested coverage must be in (0, 1); got {requested}.")
    tol = settings.coverage_tolerance if tolerance is None else tolerance
    floor = requested - tol
    meets = empirical >= floor

    notes: list[str] = []
    if not meets:
        notes.append(
            f"UNDER-COVERAGE: measured {empirical:.3f} against a requested {requested:.3f} "
            f"(floor {floor:.3f}). The stated confidence is not being delivered; every "
            f"interval quoted downstream is narrower than the evidence supports."
        )
    elif empirical - requested > tol:
        notes.append(
            f"Over-coverage: measured {empirical:.3f} against a requested {requested:.3f}. "
            f"The promise is kept, but conservatively — the intervals are wider than they "
            f"need to be and support weaker decisions than the data allows."
        )
    else:
        notes.append(
            f"Coverage met: measured {empirical:.3f} against a requested {requested:.3f} "
            f"(floor {floor:.3f}), on {n_rows} held-out rows."
        )
    if 0 < n_rows < _MIN_COVERAGE_ROWS:
        notes.append(
            f"Measured on only {n_rows} rows; at this size the estimate moves in steps of "
            f"{1 / n_rows:.2f} and agreement with the requested level is not evidence."
        )
    return CoverageReport(
        requested_coverage=requested,
        empirical_coverage=empirical,
        tolerance=tol,
        floor=floor,
        meets_request=meets,
        n_rows=n_rows,
        kind=kind,
        mean_width=mean_width,
        mean_set_size=mean_size,
        notes=notes,
    )


def coverage_by_slice(
    y_true: npt.ArrayLike,
    intervals_or_sets: Sequence[object],
    segments: Sequence[object],
    *,
    requested: float,
    tolerance: float | None = None,
    min_rows: int = _MIN_COVERAGE_ROWS,
) -> dict[str, CoverageReport]:
    """Measure coverage separately inside each segment — where marginal coverage hides.

    Split conformal guarantees *marginal* coverage: averaged over the whole distribution.
    Under heteroscedastic noise a single calibrated width over-covers the quiet regions and
    under-covers the loud ones, and the average lands exactly where it was asked to. The
    only way to see it is to measure per segment, which is what this does.

    Segments with fewer than ``min_rows`` rows get no measurement — but they are never
    silently absent either: they are named in the companion ``"__skipped__"`` entry, because
    a segment that quietly disappears from the report is the same blind spot the marginal
    average creates.

    Args:
        y_true: Measured targets, held out.
        intervals_or_sets: Conformal outputs aligned with ``y_true``.
        segments: A segment label per row (e.g. a categorical feature's value).
        requested: The requested coverage level.
        tolerance: Allowed shortfall; defaults to ``settings.coverage_tolerance``.
        min_rows: Minimum rows for a segment to be measured.

    Returns:
        Segment label → :class:`CoverageReport`. An extra key ``"__skipped__"`` is present
        when any segment was too small; its report carries the skipped names in ``notes``.

    Raises:
        ValueError: When the three sequences are not aligned.
    """
    import numpy as np

    true = np.asarray(y_true).reshape(-1)
    items = list(intervals_or_sets)
    labels = [str(s) for s in segments]
    if not (len(items) == len(labels) == true.shape[0]):
        raise ValueError(
            f"Misaligned inputs: {true.shape[0]} labels, {len(items)} conformal outputs, "
            f"{len(labels)} segment labels."
        )
    kind: Literal["interval", "set"] = (
        "interval" if items and _is_interval(items[0]) else "set"
    )

    out: dict[str, CoverageReport] = {}
    skipped: list[str] = []
    for name in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == name]
        if len(idx) < min_rows:
            skipped.append(f"{name} (n={len(idx)})")
            continue
        sub_true = true[idx]
        sub_items = [items[i] for i in idx]
        empirical = coverage(sub_true, sub_items, min_rows=min_rows)
        width = None
        size = None
        if kind == "interval":
            width = mean_interval_width(sub_items)
        else:
            size = mean_set_size(sub_items)
        out[name] = coverage_report(
            requested,
            empirical,
            tolerance,
            n_rows=len(idx),
            kind=kind,
            mean_width=width,
            mean_size=size,
        )
    if skipped:
        out["__skipped__"] = CoverageReport(
            requested_coverage=requested,
            empirical_coverage=0.0,
            tolerance=settings.coverage_tolerance if tolerance is None else tolerance,
            floor=requested - (settings.coverage_tolerance if tolerance is None else tolerance),
            meets_request=False,
            n_rows=0,
            kind=kind,
            notes=[
                "NOT A MEASUREMENT — this entry records segments too small to measure, so "
                "they are visible rather than absent: " + ", ".join(skipped)
            ],
        )
    return out
