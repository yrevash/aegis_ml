"""Time-series forecasting: a pass-through to ``aegis.forecast``, plus extra candidates.

Three modules, one rule between them — **the coverage a caller asked for and the coverage
the data achieved are never the same field**:

* :mod:`aegis_ml.forecast.engine` wraps ``aegis.forecast.forecast_series`` (Nixtla
  StatsForecast, conformal intervals, rolling-origin backtest, three typed refusals and no
  naive-line fallback) and normalises its answer into :class:`~.engine.ForecastRun`.
* :mod:`aegis_ml.forecast.ml_forecast` scores Nixtla ``mlforecast`` global-ML candidates on
  the *same* windows, so "would gradient boosting on lags have beaten AutoETS?" gets a
  measured answer instead of an opinion.
* :mod:`aegis_ml.forecast.backtest` is the measurement on its own, and the module that
  refuses to random-split a series at all.

Everything is imported lazily: the forecasting stack lives behind the ``aegis[forecast]``
extra, and importing this package must not require it.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BacktestSummary",
    "ForecastCandidate",
    "ForecastPoint",
    "ForecastRun",
    "RandomSplitRefusedError",
    "SeriesObservation",
    "backtest",
    "forecast",
    "ml_candidates",
    "rank_candidates",
    "summarise",
    "to_forecast_run",
]

_LAZY: dict[str, tuple[str, str]] = {
    "BacktestSummary": ("aegis_ml.forecast.backtest", "BacktestSummary"),
    "ForecastCandidate": ("aegis_ml.forecast.engine", "ForecastCandidate"),
    "ForecastPoint": ("aegis_ml.forecast.engine", "ForecastPoint"),
    "ForecastRun": ("aegis_ml.forecast.engine", "ForecastRun"),
    "RandomSplitRefusedError": ("aegis_ml.forecast.backtest", "RandomSplitRefusedError"),
    "SeriesObservation": ("aegis_ml.forecast.engine", "SeriesObservation"),
    "backtest": ("aegis_ml.forecast.backtest", "backtest"),
    "forecast": ("aegis_ml.forecast.engine", "forecast"),
    "ml_candidates": ("aegis_ml.forecast.ml_forecast", "ml_candidates"),
    "rank_candidates": ("aegis_ml.forecast.backtest", "rank_candidates"),
    "summarise": ("aegis_ml.forecast.backtest", "summarise"),
    "to_forecast_run": ("aegis_ml.forecast.engine", "to_forecast_run"),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a lazy re-export of many shapes
    """Resolve a forecasting symbol on first access, keeping import cost at pydantic-only.

    Args:
        name: Attribute being accessed on this package.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: If ``name`` is not one of this package's exports.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> list[str]:
    """List the lazy exports so tab-completion and ``help()`` see them."""
    return sorted(__all__)
