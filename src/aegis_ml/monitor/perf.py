"""Label-free performance estimation with NannyML — the strongest claim in this stack.

Every other monitoring signal answers "has the input changed?". This one answers the
question a business actually asks — *"is the model still any good?"* — **before the ground
truth arrives**. In the domains this package targets, labels land days or weeks after the
prediction; a model that quietly degraded on Monday is otherwise discovered on Friday, by
a human, from a complaint.

Two estimators, one per task:

* **CBPE** (Confidence-Based Performance Estimation) for classification. It uses the
  model's own calibrated probabilities: given a well-calibrated ``P(y=1|x)``, the expected
  confusion matrix over a chunk of unlabelled predictions is computable, and every metric
  derived from it follows. Its assumption — no concept drift, i.e. ``P(y|x)`` is stable
  while ``P(x)`` may move — is stated in the returned payload, because a violated
  assumption is exactly the case where a confident estimate is most dangerous.
* **DLE** (Direct Loss Estimation) for regression. It fits a second model to predict the
  *loss* of the first from the same features, then applies it to unlabelled data.

**Naming rule, enforced everywhere in this module.** Every estimated quantity is prefixed
``estimated_``. When labels *are* present in the current frame, the realised metric is
computed too and returned alongside under ``realised_`` keys, so an estimate and a
measurement can be read side by side and can never be mistaken for one another. This is
the same discipline ``contracts/protocols.py`` applies to requested versus empirical
coverage, applied to the estimate/measurement boundary instead.

The reference frame must be **labelled and large enough**. CBPE calibrates on it, DLE
trains a loss model on it, and both return a confident-looking number from far too few
rows. Below :data:`MIN_REFERENCE_LABELS` this module raises
:class:`~aegis_ml.contracts.errors.InsufficientLabelsError` with the real counts rather
than estimating anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.contracts.errors import InsufficientLabelsError
from aegis_ml.contracts.spec import MLProblem

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "MIN_CHUNK_ROWS",
    "MIN_REFERENCE_LABELS",
    "NannyMLSurfaceError",
    "PREDICTION_COLUMN_CANDIDATES",
    "PROBA_COLUMN_CANDIDATES",
    "estimate_performance",
]

_LOG = logging.getLogger(__name__)

MIN_REFERENCE_LABELS = 500
"""Labelled reference rows required before an estimate is produced at all.

Both estimators are *fitted* on the reference: CBPE calibrates probabilities on it, DLE
trains a loss model on it. Both will happily return a number from 50 rows, and that number
will look exactly as authoritative as one from 50,000. Given the deliberately noisy data
this package targets (held-out R² in the 0.45–0.80 band, unobserved confounders,
heteroscedastic noise), a loss model fitted on a few hundred rows is fitting noise — so the
floor is enforced and the error carries the real counts.
"""

MIN_CHUNK_ROWS = 50
"""Smallest chunk NannyML is asked to estimate over.

Chunks are the unit of estimation; a chunk of 5 rows produces a confidence band wider than
the metric's own range, which reads as "unknown" but plots as a value.
"""

PREDICTION_COLUMN_CANDIDATES: tuple[str, ...] = (
    "y_pred",
    "prediction",
    "predicted",
)
"""Column names accepted for the model's point prediction, in priority order."""

PROBA_COLUMN_CANDIDATES: tuple[str, ...] = (
    "y_pred_proba",
    "prediction_proba",
    "predicted_proba",
    "proba",
)
"""Column names accepted for the positive-class probability (binary classification)."""


class NannyMLSurfaceError(RuntimeError):
    """The installed NannyML does not expose ``CBPE``/``DLE`` as this module expects."""

    def __init__(self, missing: str, version: str) -> None:
        """Name what was missing and which version is installed."""
        super().__init__(
            f"nannyml {version} does not provide {missing}. This module is written against "
            f"the documented estimator surface (`nannyml.CBPE` for classification, "
            f"`nannyml.DLE` for regression). Install a supported build:\n"
            f"    uv pip install 'aegis-ml[serve]'\n"
            f"Nothing here substitutes a different estimator on your behalf: a realised "
            f"metric and an estimated one are different claims."
        )
        self.missing = missing
        self.version = version


@dataclass(frozen=True)
class _Columns:
    """The resolved column roles for one estimation."""

    features: list[str]
    y_pred: str
    y_true: str
    y_pred_proba: str | dict[str, str] | None


def _resolve_columns(
    reference: pd.DataFrame, current: pd.DataFrame, problem: MLProblem
) -> _Columns:
    """Work out which columns carry predictions, probabilities and labels.

    Resolved from what is actually in the frames rather than assumed, and *refused* when
    ambiguous. A missing ``y_pred`` column is the single most common way this call fails,
    and the error says which names were looked for — guessing one would silently estimate
    performance for a column that is not the model's output.

    Raises:
        ValueError: When a required column is absent from either frame.
    """
    shared = [c for c in reference.columns if c in set(current.columns)]
    # Datetime features are excluded from the estimator's feature list. DLE fits a
    # LightGBM loss model over these columns, and a datetime64 column raises
    # DTypePromotionError inside LightGBM's frame conversion — several frames deep, with a
    # message about dtypes that says nothing about the calendar column that caused it.
    # They are also useless to the loss model: a timestamp is monotone in the analysis
    # period, so it encodes "which chunk is this" rather than anything about the input.
    features = [
        f.name for f in problem.features if f.name in shared and f.dtype != "datetime"
    ]
    if not features:
        raise ValueError(
            f"no usable declared feature column is present in both frames (shared: "
            f"{shared[:10]}…; datetime features are excluded by design). Both estimators "
            f"need the features: DLE trains a loss model on them and CBPE chunks by them."
        )

    y_pred = next((c for c in PREDICTION_COLUMN_CANDIDATES if c in shared), None)
    if y_pred is None:
        raise ValueError(
            f"no prediction column found in both frames; looked for "
            f"{list(PREDICTION_COLUMN_CANDIDATES)}. Label-free estimation is a function of "
            f"the model's OUTPUT on unlabelled rows — without it there is nothing to "
            f"estimate from."
        )

    y_true = problem.target.name
    if y_true not in reference.columns:
        raise ValueError(
            f"the reference frame has no target column {y_true!r}. The reference must be "
            f"LABELLED: CBPE calibrates on it and DLE fits its loss model on it. The "
            f"current frame is the one that may be unlabelled."
        )

    proba: str | dict[str, str] | None = None
    if problem.target.task == "classification":
        levels = list(problem.target.levels)
        if len(levels) > 2:
            mapping = {
                level: column
                for level in levels
                for column in (f"proba_{level}", f"y_pred_proba_{level}")
                if column in shared
            }
            if len(mapping) != len(levels):
                raise ValueError(
                    f"multiclass CBPE needs one probability column per class. Expected "
                    f"proba_<level> for each of {levels}; found {sorted(mapping.values())}. "
                    f"Estimating from a subset would silently renormalise over the classes "
                    f"that happen to be present."
                )
            proba = mapping
        else:
            proba = next((c for c in PROBA_COLUMN_CANDIDATES if c in shared), None)
            if proba is None:
                raise ValueError(
                    f"binary CBPE needs the positive-class probability; looked for "
                    f"{list(PROBA_COLUMN_CANDIDATES)} in both frames. CBPE estimates the "
                    f"confusion matrix from calibrated probabilities — a hard 0/1 "
                    f"prediction carries no confidence to estimate from."
                )
    return _Columns(features=features, y_pred=y_pred, y_true=y_true, y_pred_proba=proba)


def _prepare(frame: pd.DataFrame, columns: _Columns, problem: MLProblem) -> pd.DataFrame:
    """Return the subset of ``frame`` the estimator needs, with dtypes it can consume.

    Two conversions, both required by DLE's internal LightGBM loss model and harmless to
    CBPE:

    * Declared categorical (and boolean) features are cast to pandas ``category``, which
      LightGBM consumes natively. Left as ``object``, they raise on fit — and one-hot
      encoding them here instead would silently change what the loss model sees compared
      with what the served model saw.
    * Only the columns actually used are kept, so an unrelated datetime or free-text
      column travelling in the frame cannot break the fit.

    The frames are copied; the caller's data is never mutated. A monitoring call that
    quietly re-typed the frame it was handed would corrupt whatever the caller does next.
    """
    categorical = {
        f.name for f in problem.features if f.dtype in ("categorical", "boolean")
    }
    wanted = list(columns.features)
    for extra in (columns.y_pred, columns.y_true):
        if extra in frame.columns and extra not in wanted:
            wanted.append(extra)
    if isinstance(columns.y_pred_proba, str):
        if columns.y_pred_proba in frame.columns and columns.y_pred_proba not in wanted:
            wanted.append(columns.y_pred_proba)
    elif isinstance(columns.y_pred_proba, dict):
        wanted.extend(c for c in columns.y_pred_proba.values() if c not in wanted)

    prepared = frame.loc[:, [c for c in wanted if c in frame.columns]].copy()
    for column in prepared.columns:
        if column in categorical:
            prepared[column] = prepared[column].astype("category")
    return prepared


def _chunk_size(n_current: int) -> int:
    """Pick a chunk size giving ~10 estimation points, floored at :data:`MIN_CHUNK_ROWS`."""
    return max(MIN_CHUNK_ROWS, n_current // 10)


def _metrics_for(problem: MLProblem) -> list[str]:
    """The metric set each estimator is asked for.

    The problem's primary metric leads the list so the headline estimate answers the same
    question the promotion gate ranks on; the rest are context. Only metrics the estimator
    actually supports are requested — asking CBPE for ``r2`` raises inside NannyML with a
    message about metric names, several frames from anything the caller wrote.
    """
    if problem.target.task == "classification":
        supported = {"roc_auc", "f1", "precision", "recall", "accuracy"}
        primary = problem.metric if problem.metric in supported else "roc_auc"
        return [primary] + [m for m in ("roc_auc", "accuracy", "f1") if m != primary]
    supported = {"mae", "mape", "mse", "rmse", "msle", "rmsle"}
    primary = problem.metric if problem.metric in supported else "rmse"
    return [primary] + [m for m in ("rmse", "mae") if m != primary]


def _estimator(
    problem: MLProblem, columns: _Columns, chunk: int
) -> tuple[Any, str, list[str], str]:
    """Build the estimator for this task, verifying the surface first.

    Returns:
        ``(estimator, kind, metrics, nannyml_version)`` — the version travels with the
        estimator so a later surface mismatch can name it without re-importing.
    """
    nannyml = require("aegis-ml[serve]", "nannyml")
    version = str(getattr(nannyml, "__version__", "unknown"))
    metrics = _metrics_for(problem)

    if problem.target.task == "classification":
        if not hasattr(nannyml, "CBPE"):
            raise NannyMLSurfaceError("nannyml.CBPE", version)
        levels = list(problem.target.levels)
        problem_type = (
            "classification_multiclass" if len(levels) > 2 else "classification_binary"
        )
        estimator = nannyml.CBPE(
            y_pred=columns.y_pred,
            y_pred_proba=columns.y_pred_proba,
            y_true=columns.y_true,
            metrics=metrics,
            chunk_size=chunk,
            problem_type=problem_type,
        )
        return estimator, "CBPE", metrics, version

    if not hasattr(nannyml, "DLE"):
        raise NannyMLSurfaceError("nannyml.DLE", version)
    estimator = nannyml.DLE(
        feature_column_names=columns.features,
        y_pred=columns.y_pred,
        y_true=columns.y_true,
        metrics=metrics,
        chunk_size=chunk,
    )
    return estimator, "DLE", metrics, version


def _results_frame(results: Any) -> pd.DataFrame:  # noqa: ANN401 - a nannyml Result
    """Return the analysis-period results as a DataFrame, however this build exposes it."""
    filtered = results
    filter_fn = getattr(results, "filter", None)
    if callable(filter_fn):
        filtered = filter_fn(period="analysis")
    to_df = getattr(filtered, "to_df", None)
    if not callable(to_df):
        raise NannyMLSurfaceError("a Result with .to_df()", "unknown")
    return to_df()


def _column(frame: pd.DataFrame, metric: str, field: str) -> Any:  # noqa: ANN401
    """Pull one ``(metric, field)`` column out of NannyML's MultiIndex result frame.

    NannyML returns a two-level column index — ``(metric, 'value')``,
    ``(metric, 'upper_confidence_boundary')`` and so on — but flattens to
    ``metric_value``-style names in some contexts. Both layouts are handled, and a missing
    column returns ``None`` rather than raising: a build that omits, say, the sampling
    error should still yield the estimate itself.
    """
    if hasattr(frame.columns, "nlevels") and frame.columns.nlevels > 1:
        if (metric, field) in frame.columns:
            return frame[(metric, field)]
        return None
    for candidate in (f"{metric}_{field}", f"{metric}.{field}"):
        if candidate in frame.columns:
            return frame[candidate]
    return None


def _last(series: Any) -> float | None:  # noqa: ANN401 - a pandas Series or None
    """Return the final value of a series as a float, or ``None`` if unavailable."""
    if series is None or len(series) == 0:
        return None
    value = series.iloc[-1]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # NaN check without importing math


def _realised(
    current: pd.DataFrame, problem: MLProblem, columns: _Columns, metric: str
) -> tuple[str | None, float | None]:
    """Compute the realised metric when the current frame happens to carry labels.

    Reported *beside* the estimate, never instead of it. The whole point of this module is
    the case where labels are absent; when some arrive, showing both is what lets a reader
    judge whether the estimator is trustworthy on this data — which is a far more useful
    thing to publish than either number alone.
    """
    if columns.y_true not in current.columns:
        return None, None
    labelled = current.dropna(subset=[columns.y_true, columns.y_pred])
    if labelled.empty:
        return None, None

    sklearn_metrics = require("aegis-ml[serve]", "sklearn.metrics")
    y_true = labelled[columns.y_true]
    y_pred = labelled[columns.y_pred]
    try:
        if problem.target.task == "classification":
            if metric == "accuracy":
                return "accuracy", float(sklearn_metrics.accuracy_score(y_true, y_pred))
            if metric == "f1":
                # The averaging is decided from the labels actually present, not from the
                # declared level count: sklearn's average="binary" needs a pos_label that
                # exists, and a two-level spec whose data arrived as strings is exactly the
                # case that raises. The averaging is named in the returned metric name so a
                # reader is never left guessing which f1 they are looking at.
                observed = set(y_true.unique()) | set(y_pred.unique())
                if observed <= {0, 1} and len(observed) <= 2:
                    return "f1_binary", float(
                        sklearn_metrics.f1_score(y_true, y_pred, average="binary")
                    )
                return "f1_macro", float(
                    sklearn_metrics.f1_score(y_true, y_pred, average="macro")
                )
            if metric == "roc_auc" and isinstance(columns.y_pred_proba, str):
                proba = labelled[columns.y_pred_proba]
                return "roc_auc", float(sklearn_metrics.roc_auc_score(y_true, proba))
            return "accuracy", float(sklearn_metrics.accuracy_score(y_true, y_pred))
        if metric == "mae":
            return "mae", float(sklearn_metrics.mean_absolute_error(y_true, y_pred))
        if metric == "mse":
            return "mse", float(sklearn_metrics.mean_squared_error(y_true, y_pred))
        return "rmse", float(sklearn_metrics.mean_squared_error(y_true, y_pred) ** 0.5)
    except ValueError as exc:
        # A genuinely undefined metric (one class present, for instance) — reported as
        # absent rather than as a number, and the reason travels in the payload.
        _LOG.info("realised %s could not be computed on the current frame: %s", metric, exc)
        return None, None


def estimate_performance(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    problem: MLProblem,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Estimate live performance from unlabelled data, and measure it too when possible.

    The reference must be labelled (it is what the estimator is fitted on); the current
    frame need not be. Both frames must carry the model's predictions — that is what the
    estimate is computed *from*.

    Args:
        reference: Labelled frame from training time, with features, predictions and the
            target. For classification it must also carry calibrated probabilities.
        current: Live frame, labels optional.
        problem: The declared problem — supplies the feature list, the target name, the
            task and the primary metric.
        run_id: The model being monitored; echoed into the payload.

    Returns:
        A dict in which **every estimated quantity is named ``estimated_*``**:

        ``estimated_metric_name``, ``estimated_metric_value``,
        ``estimated_confidence_low`` / ``_high``, ``estimated_alert``,
        ``estimated_metrics`` (one entry per requested metric),
        ``estimated_chunks`` (the per-chunk series, for plotting), plus
        ``realised_metric_name`` / ``realised_metric_value`` when the current frame
        carried labels, and ``estimator`` (``"CBPE"`` or ``"DLE"``), row counts, the
        chunk size, and ``assumptions``.

    Raises:
        InsufficientLabelsError: When the labelled reference is smaller than
            :data:`MIN_REFERENCE_LABELS`, or the current frame cannot fill one chunk.
        ValueError: When a required prediction/probability/target column is missing.
        NannyMLSurfaceError: When the installed NannyML lacks CBPE or DLE.
    """
    columns = _resolve_columns(reference, current, problem)

    labelled = reference.dropna(subset=[columns.y_true, columns.y_pred])
    have = int(len(labelled))
    if have < MIN_REFERENCE_LABELS:
        raise InsufficientLabelsError(
            have,
            MIN_REFERENCE_LABELS,
            f"NannyML {'CBPE' if problem.target.task == 'classification' else 'DLE'} "
            f"estimation for run {run_id!r} (the reference is what the estimator is "
            f"FITTED on, so an under-powered reference produces a confident-looking "
            f"estimate with no support behind it)",
        )
    if len(current) < MIN_CHUNK_ROWS:
        raise InsufficientLabelsError(
            int(len(current)),
            MIN_CHUNK_ROWS,
            f"NannyML estimation for run {run_id!r} needs at least one full chunk of "
            f"CURRENT rows (labels not required for these — only the predictions)",
        )

    chunk = _chunk_size(int(len(current)))
    estimator, kind, metrics, nannyml_version = _estimator(problem, columns, chunk)

    _LOG.info(
        "perf: fitting %s on %d labelled reference rows; estimating over %d current rows "
        "in chunks of %d (metrics: %s)",
        kind,
        have,
        len(current),
        chunk,
        ", ".join(metrics),
    )
    estimator.fit(_prepare(labelled, columns, problem))
    results = estimator.estimate(_prepare(current, columns, problem))
    frame = _results_frame(results)

    estimated: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        value = _last(_column(frame, metric, "value"))
        if value is None:
            continue
        estimated[metric] = {
            "estimated_value": value,
            "estimated_confidence_low": _last(
                _column(frame, metric, "lower_confidence_boundary")
            ),
            "estimated_confidence_high": _last(
                _column(frame, metric, "upper_confidence_boundary")
            ),
            "estimated_sampling_error": _last(_column(frame, metric, "sampling_error")),
            "estimated_threshold_low": _last(_column(frame, metric, "lower_threshold")),
            "estimated_threshold_high": _last(_column(frame, metric, "upper_threshold")),
        }

    if not estimated:
        raise NannyMLSurfaceError(
            f"per-metric 'value' columns for {metrics} in the estimation result frame "
            f"(columns seen: {list(frame.columns)[:8]}…)",
            nannyml_version,
        )

    headline = metrics[0] if metrics[0] in estimated else next(iter(estimated))
    alert_series = _column(frame, headline, "alert")
    alert = bool(alert_series.iloc[-1]) if alert_series is not None and len(alert_series) else False

    realised_name, realised_value = _realised(current, problem, columns, headline)

    payload: dict[str, Any] = {
        "run_id": run_id,
        "estimator": kind,
        "task": problem.target.task,
        "estimated_metric_name": f"estimated_{headline}",
        "estimated_metric_value": estimated[headline]["estimated_value"],
        "estimated_confidence_low": estimated[headline]["estimated_confidence_low"],
        "estimated_confidence_high": estimated[headline]["estimated_confidence_high"],
        "estimated_alert": alert,
        "estimated_metrics": {f"estimated_{k}": v for k, v in estimated.items()},
        "estimated_chunks": _chunk_series(frame, list(estimated)),
        "realised_metric_name": realised_name,
        "realised_metric_value": realised_value,
        "labels_present_in_current": realised_value is not None,
        "n_reference_labelled_rows": have,
        "n_current_rows": int(len(current)),
        "chunk_size": chunk,
        "prediction_column": columns.y_pred,
        "probability_column": columns.y_pred_proba,
        "assumptions": [
            "This is an ESTIMATE, not a measurement. Every estimated field is named "
            "estimated_* for that reason; a realised_* field, when present, is the "
            "measured value on the labelled subset of the current frame.",
            "CBPE and DLE both assume no concept drift: P(y|x) is stable while P(x) may "
            "move. If the relationship itself changed, the estimate stays confident and "
            "becomes wrong — pair this with aegis_ml.monitor.drift, which sees P(x).",
            "CBPE additionally assumes the model's probabilities are calibrated. The "
            "Aegis spine's conformal layer calibrates intervals, not class probabilities.",
        ],
    }
    _LOG.info(
        "perf: run %s — %s = %.4f (%s) [realised %s: %s]",
        run_id,
        payload["estimated_metric_name"],
        payload["estimated_metric_value"],
        kind,
        realised_name,
        realised_value,
    )
    return payload


def _chunk_series(frame: pd.DataFrame, metrics: list[str]) -> list[dict[str, Any]]:
    """Flatten the per-chunk estimates into plain dicts for JSON and plotting."""
    rows: list[dict[str, Any]] = []
    start = _column(frame, "chunk", "start_index")
    end = _column(frame, "chunk", "end_index")
    for position in range(len(frame)):
        row: dict[str, Any] = {"chunk": position}
        if start is not None:
            row["start_index"] = int(start.iloc[position])
        if end is not None:
            row["end_index"] = int(end.iloc[position])
        for metric in metrics:
            values = _column(frame, metric, "value")
            if values is not None:
                row[f"estimated_{metric}"] = float(values.iloc[position])
        rows.append(row)
    return rows
