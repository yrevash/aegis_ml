"""A thin pass-through to ``aegis.forecast`` — deliberately not a second forecaster.

``aegis.forecast.engine.forecast_series`` is already the honest thing: Nixtla
StatsForecast (AutoARIMA / AutoETS / SeasonalNaive) fitted per series, intervals from
``ConformalIntervals`` calibrated on chronologically earlier windows, a rolling-origin
backtest whose ``empirical_coverage`` is *counted* rather than assumed, and three typed
refusals — :class:`InsufficientHistoryError`, :class:`DegenerateSeriesError`,
:class:`ForecastFitError` — instead of a naive line drawn through noise.

Re-implementing any of that here would produce a second forecaster with a second set of
bugs and a second coverage story, and the demo would have to explain which one it was
looking at. So this module does exactly two things:

1. **Locates** the forecasting stack through :func:`aegis_ml._require.require`, so a
   deployment without the extra gets the install command rather than a mystery.
2. **Normalises** the result into :class:`ForecastRun`, a shape this package's registry,
   CLI and serving router can store and render alongside a
   :class:`~aegis_ml.contracts.protocols.TrainResult`.

Every refusal propagates untouched. In particular ``DegenerateSeriesError`` is *not* caught
and turned into a flat forecast with 100% coverage: a constant series fits perfectly,
forecasts a flat line and reports full coverage from a zero-width interval — all
arithmetically true, all describing the absence of data rather than a prediction.

The requested/measured split survives the normalisation intact and by construction:
:attr:`ForecastRun.requested_coverage` is what the caller asked for and
:attr:`ForecastRun.empirical_coverage` is what the backtest counted. They are two fields
here for the same reason they are two fields in ``aegis.forecast.types.BacktestReport`` and
in :class:`~aegis_ml.contracts.protocols.TrainResult` — one field would mean whichever the
reader assumed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from aegis_ml._require import require
from aegis_ml.contracts.spec import MLProblem

__all__ = [
    "FORECAST_EXTRA",
    "ForecastCandidate",
    "ForecastPoint",
    "ForecastRun",
    "SeriesObservation",
    "forecast",
    "to_forecast_run",
]

FORECAST_EXTRA = "aegis[forecast]"
"""The install target that carries statsforecast — quoted verbatim in every ImportError."""


class SeriesObservation(BaseModel):
    """One observed point of history, in this package's own JSON-safe shape."""

    ts: datetime = Field(description="Observation timestamp (naive UTC, as Aegis stores).")
    value: float = Field(description="The observed value.")


class ForecastPoint(BaseModel):
    """One forecast step: point prediction plus interval bounds.

    ``lo``/``hi`` mean only what :attr:`ForecastRun.interval_method` says they mean. Read
    that field before quoting these numbers: a parametric band is a model assumption, and
    only a conformal one was calibrated on out-of-sample errors.
    """

    ts: datetime
    point: float
    lo: float
    hi: float
    step: int = Field(description="1-based index into the horizon.")


class ForecastCandidate(BaseModel):
    """One scored candidate model, kept whether it won or lost.

    Losers are retained on purpose, mirroring ``ForecastResult.candidates``: a selection is
    only auditable if the baseline it beat is visible. "AutoETS won" says nothing; "AutoETS
    won at sMAPE 8.1 against SeasonalNaive's 8.4" says the extra machinery bought 0.3
    points, and lets a reader decide whether that was worth it.
    """

    model: str = Field(description="Model name, e.g. 'AutoETS' or 'LGBMRegressor'.")
    family: Literal["statsforecast", "mlforecast"] = Field(
        default="statsforecast",
        description="Which engine produced this score — the two are comparable only "
        "because they were backtested on the same windows at the same horizon.",
    )
    smape: float = Field(description="MEASURED symmetric MAPE (%) on held-out points.")
    mape: float | None = Field(default=None, description="MEASURED MAPE (%), None if undefined.")
    mae: float = Field(description="MEASURED mean absolute error on held-out points.")
    empirical_coverage: float = Field(description="MEASURED interval coverage, 0–1.")
    selected: bool = Field(default=False)


class ForecastRun(BaseModel):
    """A forecast normalised for this package's registry, CLI and serving router.

    Field-for-field a lossless projection of ``aegis.forecast.types.ForecastResult`` plus
    the backtest numbers hoisted to the top level, because every consumer here reads
    ``empirical_coverage`` and none of them should have to know it lives one level down.
    """

    series_id: str
    label: str
    unit: str | None = None
    data_source: str = Field(description="Provenance tag; a demo series must never read as live.")
    freq: str
    season_length: int
    history_points: int
    history: list[SeriesObservation] = Field(default_factory=list)
    horizon: int
    points: list[ForecastPoint] = Field(default_factory=list)
    model: str = Field(description="The selected model.")
    selection_metric: str = "smape"
    candidates: list[ForecastCandidate] = Field(default_factory=list)
    excluded_models: dict[str, str] = Field(
        default_factory=dict,
        description="Model → why it could not be scored. Never empty-and-silent: a "
        "candidate that vanished and a candidate that failed must look different.",
    )
    interval_method: Literal["conformal", "parametric"]
    interval_method_detail: str
    requested_coverage: float = Field(description="The coverage level ASKED FOR.")
    empirical_coverage: float = Field(description="The coverage rate ACHIEVED on held-out data.")
    coverage_meets_request: bool = Field(description="Measured >= requested, with no rounding up.")
    backtest_windows: int
    backtest_points: int
    smape: float
    mape: float | None = None
    mae: float
    model_selected_on_backtest_windows: bool = True
    generated_at: datetime

    @property
    def coverage_gap(self) -> float:
        """Requested minus achieved coverage; positive means the band under-covered.

        Surfaced as a property rather than a stored field so it can never disagree with the
        two numbers it is derived from.
        """
        return self.requested_coverage - self.empirical_coverage


def _resolve_label(problem_or_label: MLProblem | str) -> tuple[str, str, str | None]:
    """Derive ``(series_id, label, unit)`` from a problem or a bare label.

    Accepting an :class:`~aegis_ml.contracts.spec.MLProblem` is a convenience with a point:
    the target's ``unit`` is what turns a bare float in a decision-support sentence into a
    quantity, and the problem already carries it. Passing a string is the escape hatch for
    a series that is not the supervised target (dispatch counts, ticket volume).

    Args:
        problem_or_label: The ML problem whose target names the series, or a plain label.

    Returns:
        ``(series_id, label, unit)``.
    """
    if isinstance(problem_or_label, MLProblem):
        target = problem_or_label.target
        return (
            f"{problem_or_label.domain_id}:{target.name}",
            target.description or target.name,
            target.unit,
        )
    return (problem_or_label, problem_or_label, None)


def to_forecast_run(
    result: Any,  # noqa: ANN401 - aegis.forecast.types.ForecastResult, imported lazily
    *,
    extra_candidates: Sequence[ForecastCandidate] = (),
) -> ForecastRun:
    """Project an ``aegis.forecast`` result into :class:`ForecastRun`.

    Args:
        result: An ``aegis.forecast.types.ForecastResult``.
        extra_candidates: Additional scored candidates — normally the mlforecast global-ML
            entries from :func:`aegis_ml.forecast.ml_forecast.ml_candidates`. They are
            appended to the roster and ranked alongside the statsforecast ones, which is
            only legitimate because both were backtested on the same windows at the same
            horizon and the same requested level.

    Returns:
        The normalised run.

    Note:
        ``selected`` is left exactly as ``aegis.forecast`` set it. A better-scoring
        mlforecast candidate is reported as the better score and is **not** silently
        promoted to ``model``: the forecast that was actually fitted and returned is the
        statsforecast winner, and relabelling it would make the returned points and the
        named model disagree.
    """
    backtest = result.backtest
    candidates = [
        ForecastCandidate(
            model=c.model,
            family="statsforecast",
            smape=c.smape,
            mape=c.mape,
            mae=c.mae,
            empirical_coverage=c.empirical_coverage,
            selected=c.selected,
        )
        for c in result.candidates
    ]
    candidates.extend(extra_candidates)
    candidates.sort(key=lambda c: c.smape)

    return ForecastRun(
        series_id=result.series_id,
        label=result.label,
        unit=result.unit,
        data_source=result.data_source,
        freq=result.freq,
        season_length=result.season_length,
        history_points=result.history_points,
        history=[SeriesObservation(ts=p.ts, value=p.value) for p in result.history],
        horizon=result.horizon,
        points=[
            ForecastPoint(ts=p.ts, point=p.point, lo=p.lo, hi=p.hi, step=p.step)
            for p in result.points
        ],
        model=result.model,
        selection_metric=result.selection_metric,
        candidates=candidates,
        excluded_models={e.model: e.reason for e in result.excluded_models},
        interval_method=result.interval_method,
        interval_method_detail=result.interval_method_detail,
        requested_coverage=backtest.requested_coverage,
        empirical_coverage=backtest.empirical_coverage,
        coverage_meets_request=backtest.coverage_meets_request,
        backtest_windows=backtest.windows,
        backtest_points=backtest.n_points,
        smape=backtest.smape,
        mape=backtest.mape,
        mae=backtest.mae,
        model_selected_on_backtest_windows=result.model_selected_on_backtest_windows,
        generated_at=result.generated_at.replace(tzinfo=UTC)
        if result.generated_at.tzinfo is None
        else result.generated_at,
    )


def forecast(
    points: Iterable[tuple[datetime, float] | SeriesObservation],
    problem_or_label: MLProblem | str,
    *,
    horizon: int,
    series_id: str | None = None,
    data_source: str = "adapter",
    freq: str | None = None,
    unit: str | None = None,
    level: float | None = None,
    interval: Literal["conformal", "parametric"] = "conformal",
    backtest_windows: int = 3,
    conformal_windows: int = 3,
    include_history: bool = True,
    include_ml_candidates: bool = False,
) -> ForecastRun:
    """Forecast a series through ``aegis.forecast``, normalised for this package.

    Args:
        points: Observed history as ``(timestamp, value)`` pairs or
            :class:`SeriesObservation`s. Order is irrelevant; ``aegis.forecast`` sorts,
            de-duplicates (summing same-timestamp rows) and infers the frequency.
        problem_or_label: An :class:`~aegis_ml.contracts.spec.MLProblem` (whose target
            supplies the label and unit) or a plain human label.
        horizon: Steps to forecast beyond the last observation.
        series_id: Override for the derived series id.
        data_source: Provenance tag, e.g. ``"usage_ledger"`` or ``"synthetic"``. Recorded
            on the result so a demo series is never read as live data.
        freq: Pandas frequency alias; inferred from the observed spacing when ``None``.
        unit: Override for the derived unit.
        level: Coverage level to REQUEST. Defaults to ``settings.requested_coverage``, so
            the forecast and the supervised spine ask for the same number by default.
        interval: ``"conformal"`` (calibrated) or ``"parametric"`` (the model's own
            predictive distribution). Recorded on the result either way.
        backtest_windows: Rolling-origin cutoffs the metrics are measured on.
        conformal_windows: Windows the conformal band calibrates on.
        include_history: Whether to echo the observed history on the result.
        include_ml_candidates: Also score Nixtla ``mlforecast`` global-ML candidates on the
            same windows and add them to the roster. Off by default because it needs the
            ``strong`` extra and materially more time; on, it answers "would gradient
            boosting on lags have beaten AutoETS here?" with a measured number.

    Returns:
        A :class:`ForecastRun` whose every claim was measured.

    Raises:
        ImportError: If the forecasting stack is not installed, naming the install command.
        InsufficientHistoryError: Propagated — the series is too short to fit, calibrate
            and backtest at this horizon, with the arithmetic attached.
        DegenerateSeriesError: Propagated — a flat series would report 100% coverage from a
            zero-width interval, and saying "no variation recorded" is the honest answer.
        ForecastFitError: Propagated — every candidate failed; there is no naive fallback.
        ValueError: Propagated for an invalid horizon, level or frequency.
    """
    from aegis_ml.settings import settings

    engine = require(FORECAST_EXTRA, "aegis.forecast")
    derived_id, derived_label, derived_unit = _resolve_label(problem_or_label)
    normalised = [
        (p.ts, p.value) if isinstance(p, SeriesObservation) else p  # type: ignore[misc]
        for p in points
    ]
    requested = settings.requested_coverage if level is None else level

    result = engine.forecast_series(
        normalised,
        series_id=series_id or derived_id,
        label=derived_label,
        horizon=horizon,
        data_source=data_source,
        freq=freq,
        unit=unit if unit is not None else derived_unit,
        level=requested,
        interval=interval,
        backtest_windows=backtest_windows,
        conformal_windows=conformal_windows,
        include_history=include_history,
    )

    extra: list[ForecastCandidate] = []
    if include_ml_candidates:
        from aegis_ml.forecast.ml_forecast import ml_candidates, to_long_frame

        extra = ml_candidates(
            to_long_frame(normalised, series_id=series_id or derived_id),
            horizon=horizon,
            freq=result.freq,
            season_length=result.season_length,
            level=requested,
            n_windows=backtest_windows,
        )

    return to_forecast_run(result, extra_candidates=extra)
