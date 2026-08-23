"""Monitoring: what the inputs did, what the model is probably doing, and what to do.

Four modules, in the order a real monitoring loop uses them:

* :mod:`~aegis_ml.monitor.log` — prediction logging. Without it there is no *current*
  frame, and every drift number is computed on data assembled by hand for a screenshot.
* :mod:`~aegis_ml.monitor.drift` — Evidently 0.7+ against the reference frame stored at
  training time. Reports the **share of drifted features**, because with a dozen features
  at p<0.05 one of them firing is the expected outcome under no drift at all.
* :mod:`~aegis_ml.monitor.perf` — NannyML CBPE/DLE: estimated performance **without ground
  truth**. Every estimated quantity is named ``estimated_*`` so it can never be read as a
  measurement.
* :mod:`~aegis_ml.monitor.alerts` — thresholds to actions. A blocking verdict blocks
  *promotion*, never *serving*: Aegis serves the model it has and flags it.

Nothing here imports pandas, Evidently or NannyML at module scope — the heavy work happens
inside the functions, via :func:`aegis_ml._require.require`, so importing this package
costs pydantic and the standard library.
"""

from __future__ import annotations

from aegis_ml.monitor.alerts import (
    Alert,
    AlertConfig,
    AlertLevel,
    evaluate_alerts,
    raise_if_blocking,
)
from aegis_ml.monitor.drift import EvidentlySurfaceError, drift_report, frame_digest
from aegis_ml.monitor.log import (
    compact_to_parquet,
    feature_digest,
    log_path,
    log_prediction,
    log_prediction_async,
    read_log,
)
from aegis_ml.monitor.perf import NannyMLSurfaceError, estimate_performance

__all__ = [
    "Alert",
    "AlertConfig",
    "AlertLevel",
    "EvidentlySurfaceError",
    "NannyMLSurfaceError",
    "compact_to_parquet",
    "drift_report",
    "estimate_performance",
    "evaluate_alerts",
    "feature_digest",
    "frame_digest",
    "log_path",
    "log_prediction",
    "log_prediction_async",
    "raise_if_blocking",
    "read_log",
]
