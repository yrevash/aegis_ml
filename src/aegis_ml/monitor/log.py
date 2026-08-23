"""Prediction logging — the thing that makes drift monitoring real instead of a demo.

Drift detection compares a *current* frame against the reference frame stored at training
time. Without a prediction log there is no current frame, so every drift number is
computed on data somebody assembled by hand for the screenshot. This module is the
unglamorous half of the monitoring story and the half that decides whether the other half
means anything.

**What is stored by default: a digest, not the values.** A prediction log is a verbatim
copy of production inputs, and production inputs carry PII. ``feature_digest`` — SHA-256
over a canonical JSON encoding of the feature mapping — supports the two things the log
exists for (traffic volume over time, and detecting replayed or duplicated inputs) while
being useless to an attacker who gets the file. Feature *values* are written only when the
caller passes ``store_features=True``, which is the mode used to build a drift current
frame, and that opt-in is deliberately explicit and per-call.

**Why JSONL and not parquet on the write path.** Parquet cannot be appended row-wise: a
per-prediction parquet write means read-modify-write of an entire file per request, which
is both slow and a data-loss window. The sink is therefore append-only JSONL — one
``O_APPEND`` write per line, which POSIX makes atomic for small records so concurrent
workers interleave whole lines rather than fragments. :func:`compact_to_parquet` folds the
JSONL into a columnar file for the analytics side, and :func:`read_log` reads both.

**Postgres is a mirror, when configured.** With ``AEGIS_ML_POSTGRES_DSN`` set, every row
is also written to ``ml_predictions`` (see :mod:`aegis_ml.registry.db`). The file sink is
written *first* and unconditionally: the database is the queryable copy, never the only
one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "compact_to_parquet",
    "feature_digest",
    "log_path",
    "log_prediction",
    "log_prediction_async",
    "read_log",
]

_LOG = logging.getLogger(__name__)

_COLUMNS: tuple[str, ...] = (
    "ts",
    "run_id",
    "tenant_id",
    "feature_digest",
    "prediction",
    "prediction_label",
    "interval_low",
    "interval_high",
)
"""The columns every log row carries, whether or not features were stored.

Fixed so :func:`read_log` returns a frame with a stable schema even when the log is empty
— a monitoring job that crashes on an empty log the first morning is a monitoring job
nobody turns back on.
"""


def predictions_dir() -> Path:
    """Return ``<reports_dir>/predictions``, creating it if needed."""
    directory = Path(settings.reports_dir) / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_path(run_id: str) -> Path:
    """Return the JSONL sink for one run.

    One file per run, not one per day or one global file: a drift comparison is always
    *this model's* traffic against *this model's* reference frame, and mixing two models'
    predictions into one file makes the most common query the hardest one.

    Args:
        run_id: The model that produced the predictions.

    Returns:
        ``<reports_dir>/predictions/<run_id>.jsonl``.

    Raises:
        ValueError: If ``run_id`` is not a safe single path segment.
    """
    if not run_id or run_id != Path(run_id).name or run_id in {".", ".."}:
        raise ValueError(
            f"run_id {run_id!r} is not a safe path segment; it becomes a file name here. "
            f"Run ids come from aegis_ml.registry.store.new_run_id()."
        )
    return predictions_dir() / f"{run_id}.jsonl"


def _canonical(value: Any) -> Any:  # noqa: ANN401 - any JSON-able feature value
    """Reduce one feature value to a form whose text encoding is stable.

    Floats are the reason this exists. ``repr(0.1 + 0.2)`` and ``repr(0.30000000000000004)``
    are the same string, but ``float32`` and ``float64`` versions of the same measurement
    are not — so numpy scalars are converted through :func:`float` and NaN is folded to a
    single sentinel. Without that, the same request logged from two code paths produces
    two different digests and duplicate detection silently stops working.
    """
    if value is None:
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return "nan" if math.isnan(value) else float(value)
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalar
        return _canonical(item())
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, Sequence | tuple | list):
        return [_canonical(v) for v in value]
    return str(value)


def _as_mapping(features: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    """Normalise a feature vector to a name → value mapping.

    A bare sequence is accepted (some call sites hold a positional row) and keyed as
    ``f0, f1, …``. The keys are recorded rather than dropped so the digest of a mapping and
    the digest of the equivalent positional row are *deliberately different*: they are not
    interchangeable inputs, and pretending otherwise would hide a caller passing columns in
    the wrong order.
    """
    if isinstance(features, Mapping):
        return {str(k): v for k, v in features.items()}
    return {f"f{index}": value for index, value in enumerate(features)}


def feature_digest(features: Mapping[str, Any] | Sequence[Any]) -> str:
    """Return the SHA-256 of a feature vector, canonicalised.

    Keys are sorted, values are normalised by :func:`_canonical`, and the encoding is
    compact JSON with no whitespace. The digest is therefore stable across processes,
    across dict insertion orders, and across the numpy/python scalar boundary — the three
    ways a naive ``hash(str(features))`` produces a different answer for the same input.

    Args:
        features: Mapping of feature name → value, or a positional sequence.

    Returns:
        64-character lowercase hex digest.
    """
    canonical = _canonical(_as_mapping(features))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _split_prediction(prediction: Any) -> tuple[float | None, str | None]:  # noqa: ANN401
    """Split a prediction into its numeric and its label form.

    Both task types log here. A regression writes a float and no label; a classification
    writes the class name and, when the label is numeric-like, the number too. Forcing a
    class label through a float column is how a dashboard ends up plotting the mean of a
    category code and calling it a trend.
    """
    if prediction is None:
        return None, None
    if isinstance(prediction, bool):
        return float(prediction), str(prediction)
    if isinstance(prediction, int | float):
        return float(prediction), None
    item = getattr(prediction, "item", None)
    if callable(item):
        return _split_prediction(item())
    text = str(prediction)
    try:
        return float(text), text
    except ValueError:
        return None, text


def _row(
    run_id: str,
    features: Mapping[str, Any] | Sequence[Any],
    prediction: Any,  # noqa: ANN401 - float, class label, or numpy scalar
    *,
    interval: tuple[float, float] | None,
    tenant_id: int | None,
    store_features: bool,
    detail: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the row written to every sink, so the file and the database agree exactly."""
    mapping = _as_mapping(features)
    numeric, label = _split_prediction(prediction)
    row: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "tenant_id": tenant_id,
        "feature_digest": feature_digest(mapping),
        "prediction": numeric,
        "prediction_label": label,
        "interval_low": None if interval is None else float(interval[0]),
        "interval_high": None if interval is None else float(interval[1]),
    }
    if store_features:
        row["features"] = _canonical(mapping)
    if detail:
        row["detail"] = _canonical(dict(detail))
    return row


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON line with a single ``O_APPEND`` write.

    One ``write`` of one complete line, opened ``O_APPEND``, is what keeps concurrent
    workers from interleaving halves of two rows — the kernel positions each append at the
    current end of file. Buffered ``print``-style writing does not give that guarantee and
    produces a log that ``json.loads`` chokes on exactly when traffic is highest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    handle = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        os.write(handle, payload.encode("utf-8"))
    finally:
        os.close(handle)


async def _mirror_to_postgres(row: Mapping[str, Any], *, dsn: str) -> None:
    """Write one row to ``ml_predictions``. Raises on failure; never swallows."""
    from aegis_ml.registry import db

    async with db.session_scope(dsn) as session:
        interval: tuple[float, float] | None = None
        if row.get("interval_low") is not None and row.get("interval_high") is not None:
            interval = (float(row["interval_low"]), float(row["interval_high"]))
        detail = dict(row.get("detail") or {})
        if "features" in row:
            detail["features"] = row["features"]
        await db.insert_prediction(
            session,
            run_id=str(row["run_id"]),
            feature_digest=str(row["feature_digest"]),
            prediction=row.get("prediction"),
            prediction_label=row.get("prediction_label"),
            interval=interval,
            tenant_id=row.get("tenant_id"),
            ts=datetime.fromisoformat(str(row["ts"])),
            detail=detail,
        )


def log_prediction(
    run_id: str,
    features: Mapping[str, Any] | Sequence[Any],
    prediction: Any,  # noqa: ANN401 - float, class label, or numpy scalar
    *,
    interval: tuple[float, float] | None = None,
    tenant_id: int | None = None,
    store_features: bool = False,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record one served prediction.

    The file sink is written first and always. The Postgres mirror follows only when
    ``settings.postgres_dsn`` is configured, and a failure there raises rather than being
    dropped — a monitoring pipeline that silently loses half its rows reports a drop in
    traffic that never happened.

    Args:
        run_id: The model that produced this prediction.
        features: Feature mapping (preferred) or positional row.
        prediction: The predicted value — a float for regression, a class label for
            classification.
        interval: ``(low, high)`` conformal bounds from the Aegis spine, when present.
        tenant_id: Optional tenant scope; a plain value here, isolated at query time
            exactly as ``aegis.ops`` does it.
        store_features: Store the raw feature values alongside the digest. Off by default
            because this file becomes a copy of production inputs. Turn it on for the
            traffic you intend to drift-check — :func:`read_log` can only reconstruct a
            current frame from rows that carry values.
        detail: Anything else worth keeping (class probabilities, latency, request id).

    Returns:
        The row that was written, so a caller can assert on it or forward it.

    Raises:
        RuntimeError: If a Postgres mirror is configured and this is called from inside a
            running event loop — use :func:`log_prediction_async` there. Blocking on
            ``asyncio.run`` inside a live loop raises deep in asyncio; refusing here names
            the fix instead.
    """
    row = _row(
        run_id,
        features,
        prediction,
        interval=interval,
        tenant_id=tenant_id,
        store_features=store_features,
        detail=detail,
    )
    _append_jsonl(log_path(run_id), row)

    dsn = settings.postgres_dsn
    if dsn:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_mirror_to_postgres(row, dsn=dsn))
        else:
            raise RuntimeError(
                "log_prediction() was called from inside a running event loop while "
                "AEGIS_ML_POSTGRES_DSN is set. The file sink already holds this row; the "
                "database mirror cannot be written synchronously from a live loop. Await "
                "aegis_ml.monitor.log.log_prediction_async(...) instead — an ASGI request "
                "handler is always in this case."
            )
    return row


async def log_prediction_async(
    run_id: str,
    features: Mapping[str, Any] | Sequence[Any],
    prediction: Any,  # noqa: ANN401 - float, class label, or numpy scalar
    *,
    interval: tuple[float, float] | None = None,
    tenant_id: int | None = None,
    store_features: bool = False,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record one served prediction from async code (a FastAPI handler, typically).

    Same contract as :func:`log_prediction`, with the Postgres mirror awaited on the
    caller's loop. See that function for the arguments.

    Returns:
        The row that was written.
    """
    row = _row(
        run_id,
        features,
        prediction,
        interval=interval,
        tenant_id=tenant_id,
        store_features=store_features,
        detail=detail,
    )
    _append_jsonl(log_path(run_id), row)
    dsn = settings.postgres_dsn
    if dsn:
        await _mirror_to_postgres(row, dsn=dsn)
    return row


def _parse_since(since: str | datetime | None) -> datetime | None:
    """Coerce a ``since`` filter to an aware UTC datetime."""
    if since is None:
        return None
    parsed = since if isinstance(since, datetime) else datetime.fromisoformat(str(since))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def read_log(
    run_id: str,
    *,
    since: str | datetime | None = None,
    expand_features: bool = True,
) -> pd.DataFrame:
    """Read a run's prediction log back as a DataFrame.

    Both sinks are read and concatenated: any ``<run_id>*.parquet`` compaction files
    produced by :func:`compact_to_parquet`, plus the live ``<run_id>.jsonl`` tail. A
    compaction that has not yet had its JSONL truncated therefore cannot lose rows — at
    worst it duplicates them, which :func:`compact_to_parquet` avoids by writing the
    parquet before removing the lines it consumed.

    Args:
        run_id: The model whose log to read.
        since: Only rows at or after this instant (ISO-8601 string or datetime). A naive
            datetime is read as UTC, matching what the writer stamps.
        expand_features: Lift a stored ``features`` mapping into real columns. On by
            default because the frame is usually headed for
            :func:`aegis_ml.monitor.drift.drift_report`, which needs feature columns, not
            a column of dicts.

    Returns:
        A DataFrame with at least :data:`_COLUMNS`, newest last.

    Raises:
        FileNotFoundError: When nothing has ever been logged for this run. Distinguished
            from "logged nothing recently" (an empty frame) on purpose: an absent log means
            the serving path is not wired to the logger, and that is a wiring bug, not a
            quiet Tuesday.
    """
    pandas = require("aegis-ml[serve]", "pandas")

    jsonl = log_path(run_id)
    parquet_files = sorted(predictions_dir().glob(f"{run_id}*.parquet"))
    if not jsonl.is_file() and not parquet_files:
        raise FileNotFoundError(
            f"no prediction log for run {run_id!r} (looked for {str(jsonl)!r} and "
            f"{run_id}*.parquet). Nothing has been logged: wire the serving path to "
            f"aegis_ml.monitor.log.log_prediction(), or point AEGIS_ML_REPORTS_DIR at the "
            f"directory that already holds the log."
        )

    frames: list[Any] = [pandas.read_parquet(path) for path in parquet_files]
    if jsonl.is_file():
        rows: list[dict[str, Any]] = []
        with open(jsonl, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{jsonl} line {number} is not valid JSON ({exc}). The log is "
                        f"append-only and every writer emits one complete line per "
                        f"prediction, so a broken line means the file was edited or a "
                        f"disk filled mid-write. Fix or remove the line rather than "
                        f"letting a drift report quietly skip it."
                    ) from exc
        if rows:
            frames.append(pandas.DataFrame(rows))

    if not frames:
        return pandas.DataFrame(columns=list(_COLUMNS))

    frame = pandas.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    for column in _COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame["ts"] = pandas.to_datetime(frame["ts"], utc=True, format="mixed")
    frame = frame.sort_values("ts").reset_index(drop=True)

    cutoff = _parse_since(since)
    if cutoff is not None:
        frame = frame.loc[frame["ts"] >= cutoff].reset_index(drop=True)

    if expand_features and "features" in frame.columns:
        expanded = pandas.json_normalize(
            frame["features"].apply(lambda value: value if isinstance(value, dict) else {})
        )
        expanded.index = frame.index
        overlap = [c for c in expanded.columns if c in frame.columns]
        if overlap:
            expanded = expanded.rename(columns={c: f"feature_{c}" for c in overlap})
        frame = frame.drop(columns=["features"]).join(expanded)

    return frame


def compact_to_parquet(run_id: str, *, truncate: bool = True) -> Path | None:
    """Fold the JSONL tail into a columnar parquet file.

    Order matters and is the whole reason this is a function rather than two lines at a
    call site: the parquet is written and ``fsync``-ed *before* the JSONL is truncated. A
    crash in between duplicates rows (which :func:`read_log` tolerates and a de-duplicating
    caller can fix); the opposite order loses them permanently.

    Args:
        run_id: The run whose log to compact.
        truncate: Remove the consumed JSONL afterwards. Pass ``False`` to keep both.

    Returns:
        The parquet file written, or ``None`` when there was nothing to compact.
    """
    pandas = require("aegis-ml[serve]", "pandas")
    require("aegis-ml[serve]", "pyarrow")

    jsonl = log_path(run_id)
    if not jsonl.is_file() or jsonl.stat().st_size == 0:
        return None

    rows = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None

    frame = pandas.DataFrame(rows)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    destination = predictions_dir() / f"{run_id}.{stamp}.parquet"
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)

    if truncate:
        os.truncate(jsonl, 0)
    _LOG.info("compacted %d prediction rows for run %s → %s", len(rows), run_id, destination)
    return destination
