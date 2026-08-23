"""Data drift against the reference frame stored at training time — Evidently 0.7+.

**Modern API only.** Evidently removed the ``evidently.report.Report`` /
``evidently.metric_preset`` surface that most of the material written before 2025 targets.
This module is written against the current one::

    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset, DataSummaryPreset
    from evidently.metrics import ValueDrift

and it *verifies* that surface at import time (:func:`_load_evidently`), raising an error
that names the installed version rather than failing somewhere inside a preset with an
``AttributeError`` nobody can act on.

Three measurement decisions, each of which changes what the numbers mean:

1. **Column types are declared, never inferred.** The training data this package targets
   is deliberately hard — heteroscedastic noise, MAR missingness, irrelevant features — so
   the reference frame contains NaNs. Type inference over a column that is 30% null
   guesses ``object`` and silently switches the statistical test. Every column is passed
   explicitly through Evidently's ``DataDefinition``, derived from the
   :class:`~aegis_ml.contracts.spec.MLProblem` that also generated the adapter's spec.

2. **The per-column tests are pinned to p-value tests** — Kolmogorov–Smirnov for numeric
   columns, chi-square for categorical ones — with one threshold. Evidently's default
   chooses its test by row count, so a run over 900 rows and a run over 1100 rows return
   numbers with *opposite* directions (a p-value versus a distance). Pinning means
   "drifted" has exactly one meaning across every column and every run size, and it is the
   meaning stated here: ``p < threshold``.

3. **The headline statistic is the share of drifted features, not any single p-value.**
   With 12 features and a 0.05 threshold, roughly one feature crossing that line is the
   *expected* outcome when nothing has drifted at all — 46% of the time at least one of 12
   independent tests fires by chance. A monitor that alerts on "some feature had p<0.05"
   alerts every day and is switched off within a week. The share (compared against
   :attr:`~aegis_ml.settings.Settings.drift_share_warn` and ``drift_share_block``) is the
   statistic that survives multiple comparisons; the per-feature scores are kept in the
   JSON side-file for the human who wants to look.

Target and prediction drift are measured and reported *separately* from the feature share,
because they mean different things: features drifting is the world changing, the target
drifting is the thing you are predicting changing, and the prediction distribution
drifting is your model's behaviour changing. The Aegis rule that a requested number and a
measured number never share a field applies here too.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.contracts.protocols import DriftReport
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "CATEGORICAL_TEST",
    "NUMERIC_TEST",
    "PREDICTION_COLUMN_CANDIDATES",
    "EvidentlySurfaceError",
    "drift_report",
    "frame_digest",
]

_LOG = logging.getLogger(__name__)

NUMERIC_TEST = "ks"
"""Kolmogorov–Smirnov, two-sample. Returns a **p-value**: small means drifted."""

CATEGORICAL_TEST = "chisquare"
"""Chi-square over the level counts. Also a **p-value**: small means drifted."""

PREDICTION_COLUMN_CANDIDATES: tuple[str, ...] = (
    "prediction",
    "prediction_label",
    "y_pred",
    "predicted",
)
"""Column names treated as "the model's output" when present in both frames.

Matches what :mod:`aegis_ml.monitor.log` writes, so a current frame built by
``read_log(run_id)`` gets prediction drift measured with no extra configuration.
"""

_MIN_ROWS = 30
"""Below this, neither KS nor chi-square says anything about a distribution.

Refused loudly rather than reported as ``drifted_share=0.0``: "no drift" and "not enough
data to tell" are the two answers a monitoring dashboard must never confuse.
"""


class EvidentlySurfaceError(RuntimeError):
    """The installed Evidently does not expose the API this module is written against.

    Raised instead of guessing at an alternative import path. The pre-0.7 surface
    (``evidently.report.Report``, ``evidently.metric_preset``) was *removed*, not
    deprecated, so silently falling back to it is not an option that exists — and the
    version number is the one fact that makes the failure actionable.
    """

    def __init__(self, missing: str, version: str) -> None:
        """Name what was missing, what is installed, and the version that works."""
        super().__init__(
            f"evidently {version} does not provide {missing}. This module is written "
            f"against the modern (0.7+) API: `from evidently import Report, Dataset, "
            f"DataDefinition`, `from evidently.presets import DataDriftPreset, "
            f"DataSummaryPreset`, `from evidently.metrics import ValueDrift`. The pre-0.7 "
            f"`evidently.report.Report` + `evidently.metric_preset` surface was removed "
            f"upstream, so there is nothing to fall back to. Install a supported build:\n"
            f"    uv pip install 'evidently>=0.7'"
        )
        self.missing = missing
        self.version = version


@dataclass(frozen=True)
class _Evidently:
    """The verified Evidently symbols, resolved once per call."""

    version: str
    Report: Any  # noqa: N815 - mirrors the upstream class name exactly
    Dataset: Any  # noqa: N815
    DataDefinition: Any  # noqa: N815
    DataDriftPreset: Any  # noqa: N815
    DataSummaryPreset: Any  # noqa: N815
    ValueDrift: Any  # noqa: N815


def _load_evidently() -> _Evidently:
    """Import Evidently and verify every symbol this module uses.

    Checked eagerly and all at once, so a mismatch is one clear error at the start of the
    report rather than an ``AttributeError`` raised after the expensive comparison has
    already run.

    Returns:
        The verified symbols.

    Raises:
        ImportError: If evidently is not installed, naming the install command.
        EvidentlySurfaceError: If it is installed but exposes a different API.
    """
    evidently = require("aegis-ml[serve]", "evidently")
    version = str(getattr(evidently, "__version__", "unknown"))

    for name in ("Report", "Dataset", "DataDefinition"):
        if not hasattr(evidently, name):
            raise EvidentlySurfaceError(f"evidently.{name}", version)

    presets = require("aegis-ml[serve]", "evidently.presets")
    for name in ("DataDriftPreset", "DataSummaryPreset"):
        if not hasattr(presets, name):
            raise EvidentlySurfaceError(f"evidently.presets.{name}", version)

    metrics = require("aegis-ml[serve]", "evidently.metrics")
    if not hasattr(metrics, "ValueDrift"):
        raise EvidentlySurfaceError("evidently.metrics.ValueDrift", version)

    value_drift = metrics.ValueDrift
    if not _accepts(value_drift, "method"):
        raise EvidentlySurfaceError(
            "evidently.metrics.ValueDrift(method=...) — this module pins the statistical "
            "test so that 'drifted' means p < threshold for every column and every run "
            "size, and cannot do that without the parameter",
            version,
        )

    return _Evidently(
        version=version,
        Report=evidently.Report,
        Dataset=evidently.Dataset,
        DataDefinition=evidently.DataDefinition,
        DataDriftPreset=presets.DataDriftPreset,
        DataSummaryPreset=presets.DataSummaryPreset,
        ValueDrift=value_drift,
    )


def _accepts(target: Any, name: str) -> bool:  # noqa: ANN401 - any callable or model
    """Return whether ``target`` accepts a keyword argument called ``name``.

    Evidently's metrics are pydantic-style models whose ``__init__`` is generated, so both
    the signature and the declared fields are consulted, and a ``**kwargs`` signature is
    taken at its word. This is the "verify, do not guess" rule applied to a parameter
    rather than an import.
    """
    fields = getattr(target, "model_fields", None)
    if isinstance(fields, dict) and name in fields:
        return True
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name:
            return True
    return False


def frame_digest(frame: pd.DataFrame) -> str:
    """Content-address a DataFrame: same rows and columns → same digest.

    Used for ``DriftReport.reference_digest`` so a report can be tied to the exact
    reference frame it was computed against. Hashes the column names *and* a stable
    rendering of the values, because two frames with identical numbers under different
    column names are not the same reference — one of them is a bug.

    Args:
        frame: Any DataFrame.

    Returns:
        64-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(f"|rows={len(frame)}|".encode())
    for column in frame.columns:
        series = frame[column]
        digest.update(str(column).encode("utf-8"))
        digest.update(series.astype("string").fillna("<NA>").str.cat(sep="\x1f").encode("utf-8"))
    return digest.hexdigest()


def _column_kinds(
    problem: MLProblem, frame_columns: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Split the columns to analyse into numeric, categorical and skipped.

    Types come from the ``MLProblem`` — the same declaration that generated the adapter's
    ``ml_spec.py`` and the pandera contract — not from the frame. A column that is 30%
    null infers as ``object`` and would then be chi-square tested as if its float values
    were category labels, which produces a p-value that is not wrong so much as
    meaningless.

    ``datetime`` features are skipped: a two-sample distribution test on a timestamp column
    always drifts, because time has passed. Reporting that as a drifted feature inflates
    the share on every single run.

    Returns:
        ``(numeric, categorical, skipped)`` — all restricted to columns actually present.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    skipped: list[str] = []
    for feature in problem.features:
        if feature.name not in frame_columns:
            skipped.append(feature.name)
            continue
        if feature.dtype == "datetime":
            skipped.append(feature.name)
        elif feature.dtype in ("categorical", "boolean"):
            categorical.append(feature.name)
        else:
            numeric.append(feature.name)
    return numeric, categorical, skipped


def _score_for(snapshot: Any, payload: dict[str, Any], metric: Any) -> float | None:  # noqa: ANN401
    """Read back the score of one *specific* metric instance this module constructed.

    Evidently identifies a metric by a fingerprint — ``metric.metric_id``, a hash of its
    type and its parameters — and keys both ``Snapshot.metric_results`` and the ``id``
    field of ``Snapshot.dict()["metrics"]`` by it. Looking a score up by that fingerprint
    is exact, and exactness matters here for one specific reason: ``DataDriftPreset``
    emits its *own* ``ValueDrift`` for every column using Evidently's default test, whose
    statistic is chosen by row count and may be a distance rather than a p-value. Matching
    on a column name would collide with it, and a distance read as a p-value inverts the
    drift verdict — high drift would score as "stable".

    Both lookup routes are tried because they come from different layers: the live result
    objects, and the serialisable dict. A build that exposes only one still works.

    Args:
        snapshot: The object returned by ``Report.run``.
        payload: ``snapshot.dict()``.
        metric: The metric instance whose score is wanted.

    Returns:
        The score, or ``None`` when this build reports it in neither place.
    """
    metric_id = getattr(metric, "metric_id", None)
    if metric_id is None:
        return None

    results = getattr(snapshot, "metric_results", None)
    if isinstance(results, dict):
        value = getattr(results.get(metric_id), "value", None)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)

    for entry in payload.get("metrics", []) or []:
        if entry.get("id") != metric_id:
            continue
        value = entry.get("value")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _run_snapshot(report: Any, current: Any, reference: Any) -> Any:  # noqa: ANN401
    """Call ``Report.run`` with current/reference in the right places, whatever they are.

    Evidently 0.7's ``run`` takes the *current* dataset first and the reference second.
    Getting that backwards does not raise — it silently reports the reference as if it were
    live traffic — so the parameter names are inspected and keywords are used when they
    exist, falling back to the documented positional order only when the signature does not
    expose them.
    """
    try:
        parameters = inspect.signature(report.run).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "current_data" in parameters and "reference_data" in parameters:
        return report.run(current_data=current, reference_data=reference)
    return report.run(current, reference)


def _snapshot_payload(snapshot: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the snapshot's dict form, verifying the method exists first."""
    as_dict = getattr(snapshot, "dict", None)
    if not callable(as_dict):
        raise EvidentlySurfaceError(
            "a Snapshot with a .dict() method (returned by Report.run)",
            "unknown",
        )
    payload = as_dict()
    if not isinstance(payload, dict):
        raise EvidentlySurfaceError("Snapshot.dict() returning a mapping", "unknown")
    return payload


def _save_html(snapshot: Any, destination: Path) -> None:  # noqa: ANN401
    """Write the self-contained HTML report, verifying the method exists first."""
    save = getattr(snapshot, "save_html", None)
    if not callable(save):
        raise EvidentlySurfaceError("Snapshot.save_html(path)", "unknown")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save(str(destination))


def _verdict(share: float) -> str:
    """Map a drifted-feature share onto ``pass`` / ``warn`` / ``block``."""
    if share >= settings.drift_share_block:
        return "block"
    if share >= settings.drift_share_warn:
        return "warn"
    return "pass"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    problem: MLProblem,
    *,
    run_id: str,
    html_out: Path | str | None = None,
    p_value_threshold: float = 0.05,
) -> DriftReport:
    """Measure drift of ``current`` against ``reference`` and write an HTML report.

    Any two frames are accepted — the reference frame stored by a training run, a frame
    reconstructed from :func:`aegis_ml.monitor.log.read_log`, or a deliberately shifted
    frame produced by a demo. Nothing about the *provenance* of the two frames is assumed;
    only their columns are, and those come from ``problem``.

    Args:
        reference: The distribution the model was calibrated on.
        current: The distribution now being served.
        problem: The declared problem, supplying column types (see :func:`_column_kinds`)
            and the target column name.
        run_id: The model these frames belong to; recorded on the report.
        html_out: Where to write the HTML. Defaults to
            ``<reports_dir>/drift/<run_id>-<UTC stamp>.html``. A JSON side-file with the
            per-feature scores is written next to it.
        p_value_threshold: The pinned significance level. A feature counts as drifted when
            its KS/chi-square p-value falls below this. Raising it makes the monitor more
            sensitive per feature — which is usually the wrong knob: the share thresholds
            in settings are the ones that control alerting.

    Returns:
        A :class:`~aegis_ml.contracts.protocols.DriftReport`. ``drifted_share`` is over
        **features only**; target and prediction drift are separate fields, and both carry
        the p-value (small = drifted), as documented on this module.

    Raises:
        ValueError: If either frame has fewer than 30 rows, or if no declared feature is
            present in both frames — both cases produce a number that looks like "no
            drift" and is not.
        EvidentlySurfaceError: If the installed Evidently exposes a different API.
    """
    evidently = _load_evidently()

    if len(reference) < _MIN_ROWS or len(current) < _MIN_ROWS:
        raise ValueError(
            f"drift needs at least {_MIN_ROWS} rows on each side; got "
            f"reference={len(reference)}, current={len(current)}. Below that a "
            f"Kolmogorov–Smirnov or chi-square result is noise, and reporting it as "
            f"drifted_share=0.0 would be indistinguishable from a genuinely stable model."
        )

    shared = [c for c in reference.columns if c in set(current.columns)]
    numeric, categorical, skipped = _column_kinds(problem, shared)
    if not numeric and not categorical:
        raise ValueError(
            f"none of the {len(problem.features)} declared features appear in both frames "
            f"(shared columns: {shared[:10]}…). Drift cannot be measured on zero columns; "
            f"check that the current frame was built with store_features=True, or that it "
            f"is the right domain's log."
        )

    target = problem.target.name if problem.target.name in shared else None
    prediction = next((c for c in PREDICTION_COLUMN_CANDIDATES if c in shared), None)

    definition_numeric = list(numeric)
    definition_categorical = list(categorical)
    classification = problem.target.task == "classification"
    if target is not None:
        bucket = definition_categorical if classification else definition_numeric
        bucket.append(target)
    if prediction is not None:
        is_label = classification or prediction == "prediction_label"
        bucket = definition_categorical if is_label else definition_numeric
        bucket.append(prediction)

    analysed = definition_numeric + definition_categorical
    reference_slice = reference.loc[:, analysed].copy()
    current_slice = current.loc[:, analysed].copy()

    # Column types are DECLARED here, never inferred: the reference frame legitimately
    # contains NaNs (MAR missingness by design), and inference over a 30%-null column
    # picks `object` and silently switches the statistical test underneath us.
    definition = evidently.DataDefinition(
        numerical_columns=definition_numeric,
        categorical_columns=definition_categorical,
    )
    reference_ds = evidently.Dataset.from_pandas(reference_slice, data_definition=definition)
    current_ds = evidently.Dataset.from_pandas(current_slice, data_definition=definition)

    # Each pinned metric is kept, keyed by its column, so its score can be read back by
    # fingerprint rather than by name-matching against the preset's own metrics.
    pinned: dict[str, Any] = {
        column: _value_drift(evidently, column, NUMERIC_TEST, p_value_threshold)
        for column in definition_numeric
    }
    pinned.update(
        {
            column: _value_drift(evidently, column, CATEGORICAL_TEST, p_value_threshold)
            for column in definition_categorical
        }
    )

    report = evidently.Report(
        metrics=[
            evidently.DataDriftPreset(),
            evidently.DataSummaryPreset(),
            *pinned.values(),
        ]
    )
    snapshot = _run_snapshot(report, current_ds, reference_ds)
    payload = _snapshot_payload(snapshot)

    scores: dict[str, float] = {}
    for column, metric in pinned.items():
        score = _score_for(snapshot, payload, metric)
        if score is not None:
            scores[column] = score

    feature_columns = numeric + categorical
    measured = {c: scores[c] for c in feature_columns if c in scores}
    unmeasured = [c for c in feature_columns if c not in scores]
    if not measured:
        raise EvidentlySurfaceError(
            f"a readable ValueDrift score for any of {feature_columns[:5]}… — neither "
            f"Snapshot.metric_results nor Snapshot.dict()['metrics'] was keyed by the "
            f"metric_id of the metrics this module constructed",
            evidently.version,
        )

    drifted = sorted(c for c, score in measured.items() if score < p_value_threshold)
    share = len(drifted) / len(measured)
    verdict = _verdict(share)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    html_path = (
        Path(html_out)
        if html_out is not None
        else Path(settings.reports_dir) / "drift" / f"{run_id}-{stamp}.html"
    )
    _save_html(snapshot, html_path)

    detail = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "evidently_version": evidently.version,
        "numeric_test": NUMERIC_TEST,
        "categorical_test": CATEGORICAL_TEST,
        "p_value_threshold": p_value_threshold,
        "convention": (
            "Every per-column score is a p-value from the pinned test: SMALL means "
            "drifted. A feature counts as drifted when p < p_value_threshold. The headline "
            f"statistic is the SHARE of drifted features ({len(drifted)}/{len(measured)}), "
            "not any single p-value — across N features, roughly N x threshold of them "
            "cross the line by chance alone, so a single p<0.05 among a dozen features is "
            "the expected outcome under no drift at all."
        ),
        "share_thresholds": {
            "warn": settings.drift_share_warn,
            "block": settings.drift_share_block,
        },
        "feature_scores": measured,
        "drifted_features": drifted,
        "features_not_measured": unmeasured,
        "features_skipped": skipped,
        "target_column": target,
        "prediction_column": prediction,
        "target_p_value": scores.get(target) if target else None,
        "prediction_p_value": scores.get(prediction) if prediction else None,
        "n_reference_rows": int(len(reference)),
        "n_current_rows": int(len(current)),
        "html_report_path": str(html_path),
    }
    json_path = html_path.with_suffix(".json")
    from aegis_ml.registry.store import atomic_write_json

    atomic_write_json(json_path, detail)

    _LOG.info(
        "drift: run %s — %d/%d features drifted (share %.3f) → %s [%s]",
        run_id,
        len(drifted),
        len(measured),
        share,
        verdict,
        html_path,
    )

    return DriftReport(
        run_id=run_id,
        reference_digest=frame_digest(reference),
        n_reference_rows=int(len(reference)),
        n_current_rows=int(len(current)),
        dataset_drift=share >= settings.drift_share_warn,
        drifted_share=share,
        drifted_features=drifted,
        target_drift=scores.get(target) if target else None,
        prediction_drift=scores.get(prediction) if prediction else None,
        verdict=verdict,  # type: ignore[arg-type]
        html_report_path=str(html_path),
    )


def _value_drift(
    evidently: _Evidently, column: str, method: str, threshold: float
) -> Any:  # noqa: ANN401 - an Evidently metric instance
    """Construct one pinned ``ValueDrift`` metric.

    ``threshold`` is passed only when the installed build accepts it. The drift verdict is
    computed here from the returned p-value either way, so the parameter affects only
    Evidently's own colouring in the HTML — but passing it keeps the report a human reads
    and the report a gate reads from disagreeing about which columns are red.
    """
    kwargs: dict[str, Any] = {"column": column, "method": method}
    if _accepts(evidently.ValueDrift, "threshold"):
        kwargs["threshold"] = threshold
    return evidently.ValueDrift(**kwargs)
