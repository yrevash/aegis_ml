"""Measurement: metrics, cross-validation, calibration, slices and the promotion gate.

Everything a promotion decision rests on is computed here, and the package is organised
around one conviction: **an aggregate score is not evidence on its own.** It hides
fold-to-fold instability (:mod:`~aegis_ml.evaluate.cv` reports std next to mean), it hides
mis-stated confidence (:mod:`~aegis_ml.evaluate.calibration` separates the coverage
requested from the coverage measured, always as two fields), and it hides a collapsed
segment (:mod:`~aegis_ml.evaluate.slices` finds it, and
:mod:`~aegis_ml.evaluate.gate` compares the *worst* one, not the mean).

:mod:`~aegis_ml.evaluate.metrics` owns :data:`~aegis_ml.evaluate.metrics.HIGHER_IS_BETTER`
because every other module here needs a metric's *direction* before it can say which of two
numbers is better, and a wrong direction promotes the worse model without a single
suspicious-looking figure anywhere in the record.

Re-exporting is safe from an import-cost point of view: every heavy dependency (numpy,
pandas, sklearn) is imported inside the function that needs it, so importing this package
costs a pydantic import, matching ``aegis/ml/__init__.py``'s discipline.
"""

from __future__ import annotations

from aegis_ml.evaluate.calibration import (
    CalibrationReport,
    CoverageReport,
    ReliabilityBin,
    brier_score,
    calibration_report,
    coverage,
    coverage_by_slice,
    coverage_report,
    expected_calibration_error,
    mean_interval_width,
    mean_set_size,
    reliability_curve,
)
from aegis_ml.evaluate.cv import (
    CVReport,
    CVStrategy,
    FoldScore,
    TemporalShuffleError,
    cross_validate,
    nested_cv,
    resolve_strategy,
)
from aegis_ml.evaluate.gate import (
    CRITERIA,
    GateConfig,
    evaluate_gate,
    format_decision,
    promote_or_raise,
)
from aegis_ml.evaluate.metrics import (
    HIGHER_IS_BETTER,
    MetricNotComputedError,
    UnknownMetricError,
    classification_metrics,
    higher_is_better,
    primary,
    regression_metrics,
    score,
)
from aegis_ml.evaluate.slices import (
    SkippedSlice,
    SliceReport,
    slice_metrics,
    slice_report,
    worst_slice,
)

__all__ = [
    "CRITERIA",
    "HIGHER_IS_BETTER",
    "CVReport",
    "CVStrategy",
    "CalibrationReport",
    "CoverageReport",
    "FoldScore",
    "GateConfig",
    "MetricNotComputedError",
    "ReliabilityBin",
    "SkippedSlice",
    "SliceReport",
    "TemporalShuffleError",
    "UnknownMetricError",
    "brier_score",
    "calibration_report",
    "classification_metrics",
    "coverage",
    "coverage_by_slice",
    "coverage_report",
    "cross_validate",
    "evaluate_gate",
    "expected_calibration_error",
    "format_decision",
    "higher_is_better",
    "mean_interval_width",
    "mean_set_size",
    "nested_cv",
    "primary",
    "promote_or_raise",
    "regression_metrics",
    "reliability_curve",
    "resolve_strategy",
    "score",
    "slice_metrics",
    "slice_report",
    "worst_slice",
]
