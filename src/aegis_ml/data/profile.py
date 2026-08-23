"""Profile a frame: a machine-readable summary always, a skrub ``TableReport`` on request.

Two audiences, two artefacts, and the split between them is deliberate.

The **JSON summary** is what the pipeline consumes. It is computed with pandas alone, so it
runs anywhere the package runs, and it is the reference snapshot a drift report is later
compared against — which means it must be small, stable and serialisable, not pretty.
Everything in it is a plain Python scalar or ``None``; a ``numpy.float64`` or a ``NaN`` in a
registry JSON file is a deserialisation failure waiting for the worst possible moment.

The **HTML report** is what a human reads, and skrub's ``TableReport`` is the best thing
available for it — distributions, top values, associations and missingness in one
self-contained file. It is only built when a path is given, which keeps skrub genuinely
optional: a machine that can compute the summary is never blocked from doing so by a
reporting dependency it does not have.

What the summary is looking for, and why each field earns its place:

* ``null_share`` — the spine imputes silently (median for numerics, mode for categoricals)
  and reports what it filled in through ``MLExplainResponse.imputed_features``. A column at
  60% null is being invented for most rows.
* ``n_unique`` / ``cardinality_share`` — a column with a distinct value per row is an
  identifier. One-hot encoding one produces a matrix as wide as the frame is long, and a
  tree that splits on it memorises the training set outright.
* ``constant`` — a column with one value contributes nothing, but it also silently reduces
  the feature count the model card claims to have used.
* ``top_levels`` — the levels actually present, which is what an ``isin`` violation in
  :mod:`aegis_ml.contracts.frames` will be diagnosed against.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

    import pandas as pd

__all__ = ["DEFAULT_TOP_LEVELS", "profile", "summarize_column", "summarize_columns"]

logger = logging.getLogger(__name__)

_EXTRA = "aegis-ml[serve]"
"""Install target quoted when pandas or skrub are missing."""

DEFAULT_TOP_LEVELS = 10
"""How many most-frequent values a categorical summary carries.

Enough to diagnose a level-set mismatch by eye, few enough that a high-cardinality column
does not turn a registry JSON file into a copy of the data.
"""


def _pandas() -> ModuleType:
    """Import pandas through :func:`~aegis_ml._require.require`."""
    return require(_EXTRA, "pandas")


def _jsonable(value: Any) -> Any:  # noqa: ANN401 - narrows anything numpy hands back
    """Convert a pandas/numpy scalar into something ``json.dumps`` accepts.

    ``NaN`` becomes ``None`` rather than surviving as a float. JSON has no NaN literal, so
    the alternative is a file that only Python's non-strict decoder can read — and the
    registry's whole point is that its files outlive the process that wrote them.
    """
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def summarize_column(column: pd.Series) -> dict[str, Any]:
    """Summarise one column into JSON-safe scalars.

    Numeric and non-numeric columns get different fields rather than a union of both with
    nulls: a ``mean`` of ``None`` on a string column reads as "the mean could not be
    computed", which invites someone to go and fix it.

    Args:
        column: The column to describe.

    Returns:
        A dictionary of plain scalars, always carrying ``dtype``, ``n_null``,
        ``null_share``, ``n_unique``, ``cardinality_share`` and ``constant``.
    """
    pd = _pandas()
    n_rows = int(len(column))
    n_null = int(column.isna().sum())
    n_unique = int(column.nunique(dropna=True))
    summary: dict[str, Any] = {
        "dtype": str(column.dtype),
        "n_rows": n_rows,
        "n_null": n_null,
        "null_share": (n_null / n_rows) if n_rows else 0.0,
        "n_unique": n_unique,
        "cardinality_share": (n_unique / n_rows) if n_rows else 0.0,
        "constant": n_unique <= 1,
    }
    numeric = pd.to_numeric(column, errors="coerce")
    is_numeric = numeric.notna().sum() > 0 and not pd.api.types.is_bool_dtype(column)
    if is_numeric and pd.api.types.is_numeric_dtype(column):
        summary["kind"] = "numeric"
        summary.update(
            {
                "min": _jsonable(numeric.min()),
                "max": _jsonable(numeric.max()),
                "mean": _jsonable(numeric.mean()),
                "std": _jsonable(numeric.std()),
                "median": _jsonable(numeric.median()),
            }
        )
    elif pd.api.types.is_datetime64_any_dtype(column):
        summary["kind"] = "datetime"
        summary.update({"min": _jsonable(column.min()), "max": _jsonable(column.max())})
    else:
        summary["kind"] = "categorical"
        counts = column.astype("object").value_counts(dropna=True).head(DEFAULT_TOP_LEVELS)
        summary["top_levels"] = [
            {
                "value": _jsonable(level),
                "count": int(count),
                "share": (int(count) / n_rows) if n_rows else 0.0,
            }
            for level, count in counts.items()
        ]
    return summary


def summarize_columns(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarise every column in a frame.

    Args:
        frame: The frame to describe.

    Returns:
        Column name → the dictionary :func:`summarize_column` produces.
    """
    return {str(name): summarize_column(frame[name]) for name in frame.columns}


def profile(
    frame: pd.DataFrame,
    *,
    out_html: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Describe a frame, optionally writing a skrub ``TableReport`` beside the summary.

    Args:
        frame: The frame to profile.
        out_html: Where to write the human-readable report. ``None`` skips it entirely, and
            skrub is then never imported — the summary must remain computable on a machine
            that only has the base install.
        title: Heading for the HTML report; defaults to a row/column count.

    Returns:
        A JSON-safe dictionary with ``n_rows``, ``n_columns``, ``duplicate_rows``,
        ``memory_bytes``, ``columns`` (per-column summaries), ``findings`` (the audit lines
        a human should read first) and ``html_path`` (``None`` when no report was written).

    Raises:
        AegisMLError: When the frame has no columns — a profile of nothing is not a profile,
            and the caller has almost certainly handed in the wrong object.
        ImportError: When ``out_html`` is given and skrub is not installed.
    """
    if frame.columns.empty:
        raise AegisMLError(
            "Cannot profile a frame with no columns. This usually means a generator "
            "returned an empty DataFrame rather than raising, so nothing downstream will "
            "report the real failure either."
        )
    columns = summarize_columns(frame)
    summary: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "columns": columns,
        "findings": _findings(columns, n_rows=len(frame)),
        "html_path": None,
    }
    if out_html is not None:
        summary["html_path"] = _write_table_report(frame, Path(out_html), title)
    return summary


def _findings(columns: dict[str, dict[str, Any]], *, n_rows: int) -> list[str]:
    """Turn the per-column numbers into the sentences a reviewer should read first.

    Thresholds here are advisory and stay advisory — this function never raises. The gate
    that refuses a frame is :func:`aegis_ml.data.contract_check.check`; a profile that
    started rejecting data would give two components veto power over the same decision.
    """
    findings: list[str] = []
    for name, summary in columns.items():
        if summary["constant"]:
            findings.append(
                f"{name}: constant across all {n_rows} rows — it contributes nothing to any "
                f"model and inflates the feature count the model card reports"
            )
        if summary["null_share"] >= 0.5:
            findings.append(
                f"{name}: {summary['null_share']:.1%} null — the spine imputes these "
                f"silently, so most rows would be scored on an invented value"
            )
        if summary["kind"] == "categorical" and summary["cardinality_share"] >= 0.5:
            findings.append(
                f"{name}: {summary['n_unique']} distinct values over {n_rows} rows — this "
                f"behaves like an identifier; one-hot encoding it makes the matrix as wide "
                f"as the frame is long and lets a tree memorise the training set"
            )
    return findings


def _write_table_report(frame: pd.DataFrame, path: Path, title: str | None) -> str:
    """Render skrub's ``TableReport`` to ``path`` and return the path as a string.

    Args:
        frame: The frame to report on.
        path: Destination ``.html`` file; parent directories are created.
        title: Report heading.

    Returns:
        The absolute path written.

    Raises:
        ImportError: When skrub is not installed, naming the exact install command.
    """
    skrub = require(_EXTRA, "skrub")
    path.parent.mkdir(parents=True, exist_ok=True)
    heading = title or f"Data profile — {len(frame)} rows × {len(frame.columns)} columns"
    report = skrub.TableReport(frame, title=heading)
    path.write_text(report.html(), encoding="utf-8")
    logger.info("Wrote skrub TableReport to %s", path)
    return str(path.resolve())
