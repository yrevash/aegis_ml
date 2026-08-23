"""Nixtla ``mlforecast`` candidates — global ML forecasting, scored on the same windows.

``aegis.forecast`` fits one statistical model per series: AutoARIMA, AutoETS and a
SeasonalNaive baseline, each estimated from that series' own history. That is the right
default and it is not the whole roster. Gradient boosting over lag features learns from
*every* series at once (hence "global") and picks up exogenous structure — day-of-week,
month, rolling levels — that a univariate ARIMA cannot express. Sometimes it wins by a
wide margin; sometimes it loses to SeasonalNaive, which is itself worth knowing.

So this module adds candidates. It does not replace the engine, and it does not get to
declare a winner on its own terms: everything here is backtested with the **same**
rolling-origin cutoffs, the **same** horizon and the **same** requested level as
``aegis.forecast``'s roster, because scores measured on different windows are not
comparable and presenting them side by side would imply they are.

Intervals come from ``mlforecast``'s ``PredictionIntervals`` with
``method="conformal_distribution"`` — the same distribution-free construction as the
statsforecast side: residuals from windows that lie chronologically *before* the point
being predicted. A tree ensemble has no predictive distribution to quote, so a parametric
band is not even available here; conformal is the only honest option, which is why it is
the only one offered.

``mlforecast`` and ``lightgbm`` live in the ``strong`` extra and are therefore normally
present only in the trainer venv. That is why scoring these candidates is opt-in on
:func:`aegis_ml.forecast.engine.forecast` rather than on by default: a serving-venv caller
who did not ask for them must not be handed an ImportError for a roster addition.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.forecast.engine import ForecastCandidate

__all__ = [
    "ML_FORECAST_EXTRA",
    "MLForecastScoringError",
    "ml_candidates",
    "to_long_frame",
]

ML_FORECAST_EXTRA = "aegis-ml[strong]"
"""Install target carrying mlforecast + lightgbm; named verbatim in every ImportError."""

#: Aegis frequency alias → the exact pandas offset the bucketer produces. Mirrors
#: ``aegis.forecast.engine._PANDAS_FREQ`` deliberately: ``aegis.forecast.series`` floors
#: weekly buckets to Monday, so plain ``W`` (week-ending-Sunday) would shift every
#: timestamp by six days and score the ML candidates against misaligned actuals.
_PANDAS_FREQ: dict[str, str] = {"h": "h", "D": "D", "W": "W-MON", "MS": "MS"}

#: Calendar features worth giving a tree per frequency. Nothing here is inferred at run
#: time: an hourly series gets hour-of-day, a daily one gets day-of-week, and a monthly one
#: gets month — handing a monthly series ``dayofweek`` is a constant column and pure noise.
_DATE_FEATURES: dict[str, list[str]] = {
    "h": ["hour", "dayofweek"],
    "D": ["dayofweek", "week"],
    "W": ["week", "month"],
    "MS": ["month", "quarter"],
}

_EPS = 1e-9


class MLForecastScoringError(AegisMLError):
    """No ``mlforecast`` candidate could be backtested on this series.

    Raised rather than returning an empty list. An empty roster and a roster that could not
    be scored look identical to a caller, and one of them means "gradient boosting was not
    better here" while the other means "we never found out".
    """

    def __init__(self, reasons: dict[str, str]) -> None:
        """Name every candidate that failed and the exception it failed with."""
        detail = "; ".join(f"{model}: {why}" for model, why in reasons.items()) or "no candidates"
        super().__init__(
            f"No mlforecast candidate produced a scoreable backtest ({detail}). Global ML "
            f"forecasting needs materially more history than a univariate model: each "
            f"lag feature costs one observation at the head of the series, and the "
            f"conformal windows are carved out after that. Either supply more history or "
            f"drop include_ml_candidates — the statsforecast roster is unaffected."
        )
        self.reasons = reasons


def _level_pct(level: float) -> int:
    """Convert a coverage level to the whole percent mlforecast's columns are named with.

    Args:
        level: Requested coverage, e.g. ``0.9``.

    Returns:
        The level as an integer percentage, e.g. ``90``.

    Raises:
        ValueError: If ``level`` is outside ``(0, 1)`` or is not a whole percent — the
            interval column names are built from the integer, so a fractional level would
            silently look up a column that does not exist.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level!r}")
    pct = level * 100.0
    if abs(pct - round(pct)) > 1e-6:
        raise ValueError(f"level must be a whole percent (e.g. 0.9, 0.95), got {level!r}")
    return int(round(pct))


def to_long_frame(
    points: Sequence[tuple[datetime, float]],
    *,
    series_id: str,
) -> Any:  # noqa: ANN401 - a pandas.DataFrame, imported lazily
    """Render ``(timestamp, value)`` pairs as the long frame Nixtla consumes.

    Args:
        points: Observed history. Sorted and de-duplicated here (same-timestamp rows are
            **summed**, matching ``aegis.forecast.series.normalise_points``, so the two
            engines never see different histories for the same input).
        series_id: Value for the ``unique_id`` column.

    Returns:
        A ``unique_id`` / ``ds`` / ``y`` :class:`pandas.DataFrame`, oldest first.
    """
    pd = require(ML_FORECAST_EXTRA, "pandas")
    merged: dict[datetime, float] = {}
    for ts, value in points:
        stamp = ts.replace(tzinfo=None) if ts.tzinfo is not None else ts
        merged[stamp] = merged.get(stamp, 0.0) + float(value)
    ordered = sorted(merged.items())
    return pd.DataFrame(
        {
            "unique_id": [series_id] * len(ordered),
            "ds": [ts for ts, _ in ordered],
            "y": [value for _, value in ordered],
        }
    )


def _lags(season_length: int, n_rows: int, horizon: int) -> list[int]:
    """Choose lag features that the available history can actually support.

    Every lag of order *k* costs the first *k* rows of the series, and the conformal
    windows are carved out of what remains. A roster that asks for a two-season lag on a
    series with three seasons of history does not fail loudly — it silently trains on a
    handful of rows and reports a confident, meaningless score. So the roster is trimmed to
    what the data supports, and the trim is visible in the returned candidate count.

    Args:
        season_length: Seasonal period for the frequency.
        n_rows: Observations available.
        horizon: Forecast horizon (its windows are held out of training).

    Returns:
        Ascending lag orders, always including lag 1.
    """
    budget = max(1, n_rows - horizon * 2)
    wanted = {1, 2, 3, season_length, season_length + 1, 2 * season_length}
    usable = sorted(lag for lag in wanted if 0 < lag <= max(1, budget // 3))
    return usable or [1]


def _models(seed: int) -> dict[str, Any]:
    """Build the candidate estimators.

    Three deliberately different biases: a boosted tree (non-linear, interaction-heavy), a
    random forest (variance-reducing, robust to a noisy target) and ridge regression on the
    same lag matrix (linear, and the one that tells you whether the trees are earning their
    complexity). A roster of three boosters would agree with itself and prove nothing.

    Args:
        seed: Random state, so a re-run reproduces the leaderboard exactly.

    Returns:
        Model name → estimator. The keys become the frame's prediction column names.
    """
    lgb = require(ML_FORECAST_EXTRA, "lightgbm")
    linear = require("aegis-ml[serve]", "sklearn.linear_model")
    ensemble = require("aegis-ml[serve]", "sklearn.ensemble")
    return {
        "LGBMRegressor": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            random_state=seed,
            verbosity=-1,
        ),
        "RandomForest": ensemble.RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "Ridge": linear.Ridge(alpha=1.0, random_state=seed),
    }


def _smape(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Symmetric MAPE (%), with a both-sides-zero pair contributing 0 rather than dividing.

    Identical to ``aegis.forecast.engine._smape`` on purpose: a candidate scored with a
    different sMAPE definition is not comparable with the statsforecast roster, and the
    whole point of this module is that it is.
    """
    total = 0.0
    for a, p in zip(actual, pred, strict=True):
        denom = abs(a) + abs(p)
        total += 0.0 if denom < _EPS else 200.0 * abs(p - a) / denom
    return total / len(actual)


def _mape(actual: Sequence[float], pred: Sequence[float]) -> float | None:
    """MAPE (%), or ``None`` when any actual is ~0 and it is undefined rather than large."""
    if any(abs(a) < _EPS for a in actual):
        return None
    return sum(abs(p - a) / abs(a) for a, p in zip(actual, pred, strict=True)) / len(actual) * 100.0


def _mae(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Mean absolute error over paired actual/predicted values."""
    return sum(abs(p - a) for a, p in zip(actual, pred, strict=True)) / len(actual)


def ml_candidates(
    df: Any,  # noqa: ANN401 - a pandas.DataFrame in Nixtla long format
    *,
    horizon: int,
    freq: str,
    season_length: int,
    level: float,
    n_windows: int = 3,
    seed: int = 7,
) -> list[ForecastCandidate]:
    """Backtest global-ML forecasters and return their scores as comparable candidates.

    Args:
        df: History in Nixtla long format (``unique_id``/``ds``/``y``) — build it with
            :func:`to_long_frame`.
        horizon: Steps forecast from each cutoff. Must match the statsforecast run's
            horizon, or the two score sets are not comparable.
        freq: Aegis frequency alias (``'h'``/``'D'``/``'W'``/``'MS'``), as reported on
            ``ForecastResult.freq``. Mapped here to the exact pandas offset.
        season_length: Seasonal period for that frequency; drives the lag roster.
        level: Coverage level to REQUEST, e.g. ``0.9``. What is achieved is measured and
            returned on each candidate as ``empirical_coverage``.
        n_windows: Rolling-origin cutoffs — both the backtest windows and the windows the
            conformal band calibrates on. Same value as the statsforecast run.
        seed: Random state for the tree ensembles, so the leaderboard reproduces.

    Returns:
        One :class:`~aegis_ml.forecast.engine.ForecastCandidate` per model that produced a
        scoreable backtest, ascending by sMAPE. ``selected`` is always ``False``: these are
        roster entries, and the forecast actually served comes from the engine that fitted
        and returned points.

    Raises:
        ValueError: If ``freq`` is not a supported alias or ``level`` is not a whole percent.
        MLForecastScoringError: If no candidate could be scored — never an empty list, which
            would read as "the trees lost".
        ImportError: If the ``strong`` extra is not installed, naming the install command.
    """
    if freq not in _PANDAS_FREQ:
        raise ValueError(f"unsupported frequency {freq!r}; expected one of {sorted(_PANDAS_FREQ)}")
    level_pct = _level_pct(level)

    mlf = require(ML_FORECAST_EXTRA, "mlforecast")
    utils = require(ML_FORECAST_EXTRA, "mlforecast.utils")
    lag_transforms = require(ML_FORECAST_EXTRA, "mlforecast.lag_transforms")

    n_rows = int(len(df))
    lags = _lags(season_length, n_rows, horizon)
    window = max(2, min(season_length, max(2, n_rows // 6)))

    engine = mlf.MLForecast(
        models=_models(seed),
        freq=_PANDAS_FREQ[freq],
        lags=lags,
        lag_transforms={1: [lag_transforms.RollingMean(window_size=window)]},
        date_features=_DATE_FEATURES[freq],
    )
    intervals = utils.PredictionIntervals(
        n_windows=n_windows, h=horizon, method="conformal_distribution"
    )

    try:
        cv = engine.cross_validation(
            df=df,
            h=horizon,
            n_windows=n_windows,
            step_size=horizon,
            level=[level_pct],
            prediction_intervals=intervals,
            refit=False,
        )
    except Exception as exc:  # noqa: BLE001 - converted to a typed refusal with the reason
        raise MLForecastScoringError({"<cross_validation>": f"{type(exc).__name__}: {exc}"}) from exc

    candidates: list[ForecastCandidate] = []
    reasons: dict[str, str] = {}
    for name in _models(seed):
        lo_col, hi_col = f"{name}-lo-{level_pct}", f"{name}-hi-{level_pct}"
        if not {name, lo_col, hi_col} <= set(cv.columns):
            reasons[name] = "produced no prediction or interval columns on the backtest"
            continue
        usable = cv[["y", name, lo_col, hi_col]].dropna()
        if usable.empty:
            reasons[name] = "every backtest row was non-finite after alignment"
            continue
        actual = [float(v) for v in usable["y"]]
        pred = [float(v) for v in usable[name]]
        inside = sum(
            1
            for a, lo, hi in zip(actual, usable[lo_col], usable[hi_col], strict=True)
            if float(lo) <= a <= float(hi)
        )
        candidates.append(
            ForecastCandidate(
                model=name,
                family="mlforecast",
                smape=_smape(actual, pred),
                mape=_mape(actual, pred),
                mae=_mae(actual, pred),
                empirical_coverage=inside / len(actual),
                selected=False,
            )
        )

    if not candidates:
        raise MLForecastScoringError(reasons)
    candidates.sort(key=lambda c: c.smape)
    return candidates
