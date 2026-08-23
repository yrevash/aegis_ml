"""Rolling-origin backtesting for time series — and a hard refusal to random-split one.

The single most common way to publish a fraudulent time-series score is to call
``train_test_split(shuffle=True)``. It does not error, it does not warn, and it produces
numbers that are better than the honest ones — which is exactly why it survives review. A
randomly held-out row sits *between* rows the model trained on, so the model interpolates
where production would have to extrapolate, and any interval calibrated on those residuals
inherits the same optimism. The reported coverage is then a statement about a world in
which the future is available at training time.

So this module offers exactly one splitting strategy and :func:`split_series` raises
:class:`RandomSplitRefusedError` on any other. There is no ``allow_random=True`` escape
hatch, because there is no case where the escape hatch is the right answer for a series.

What it *does* provide is the honest analogue: rolling-origin cross-validation. For each
cutoff the model is fitted strictly on data before it and scored on the ``horizon``
observations after it, so nothing scored was ever seen in fitting or calibration. That
measurement already exists inside ``aegis.forecast`` and is not re-implemented here —
:func:`backtest` drives it, merges in the ``mlforecast`` roster when asked, and projects
the whole thing into :class:`BacktestSummary`, which mirrors
``aegis.forecast.types.BacktestReport``'s field names (``requested_coverage``,
``empirical_coverage``, ``coverage_meets_request``) so a reader moving between the two
never has to ask which coverage a number is.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.forecast.engine import ForecastCandidate, ForecastRun, SeriesObservation, forecast

__all__ = [
    "BacktestSummary",
    "RandomSplitRefusedError",
    "backtest",
    "rank_candidates",
    "rolling_origin_cutoffs",
    "split_series",
    "summarise",
]

SplitStrategy = Literal["chronological", "rolling-origin"]
"""The only two strategies this module will perform. Both respect the arrow of time."""


class RandomSplitRefusedError(AegisMLError):
    """A caller asked for a shuffled or random split of a time series.

    Refused rather than performed. The resulting score would be higher than the truth and
    indistinguishable from it, and a conformal interval calibrated on those residuals would
    carry a coverage claim that does not hold for a single future observation.
    """

    def __init__(self, strategy: str) -> None:
        """Name the rejected strategy and the correct alternative."""
        super().__init__(
            f"Split strategy {strategy!r} is refused for a time series. A random or "
            f"shuffled split places held-out rows BETWEEN training rows, so the model "
            f"interpolates where production must extrapolate: the score comes out better "
            f"than the truth and the conformal interval calibrated on those residuals "
            f"makes a coverage claim that holds for no future observation. Use "
            f"strategy='chronological' for a single cut, or "
            f"aegis_ml.forecast.backtest.backtest() for the rolling-origin measurement "
            f"the coverage number is supposed to come from. There is no override."
        )
        self.strategy = strategy


class BacktestSummary(BaseModel):
    """Measured accuracy and interval coverage for one series, plus the full roster.

    Mirrors ``aegis.forecast.types.BacktestReport`` field for field on the coverage triple,
    and adds the ranked candidate list so the winner's margin over the seasonal-naive
    baseline is visible. A summary showing only the winner cannot say whether the winner
    won by a nose or a mile, and the margin is the part that says whether the complexity
    was worth carrying.
    """

    series_id: str
    label: str
    freq: str
    season_length: int
    horizon: int
    windows: int = Field(description="Rolling-origin cutoffs evaluated.")
    n_points: int = Field(description="Held-out (cutoff, step) pairs actually scored.")
    history_points: int
    model: str = Field(description="The model whose numbers the top-level metrics are.")
    smape: float = Field(description="MEASURED symmetric MAPE (%) on held-out points.")
    mape: float | None = Field(default=None, description="MEASURED MAPE (%), None if undefined.")
    mae: float = Field(description="MEASURED mean absolute error on held-out points.")
    requested_coverage: float = Field(description="The coverage level ASKED FOR.")
    empirical_coverage: float = Field(description="The coverage rate ACHIEVED on held-out data.")
    coverage_meets_request: bool = Field(description="Measured >= requested; never rounded up.")
    interval_method: Literal["conformal", "parametric"]
    interval_method_detail: str
    candidates: list[ForecastCandidate] = Field(
        default_factory=list, description="Every scored candidate, best sMAPE first."
    )
    excluded_models: dict[str, str] = Field(
        default_factory=dict, description="Candidate → why it could not be scored."
    )
    split_strategy: SplitStrategy = "rolling-origin"
    generated_at: datetime

    @property
    def coverage_gap(self) -> float:
        """Requested minus achieved coverage; positive means the band under-covered."""
        return self.requested_coverage - self.empirical_coverage

    @property
    def baseline_margin(self) -> float | None:
        """Winner's sMAPE minus the best non-selected candidate's, or ``None`` if alone.

        Negative means the winner is genuinely better. Reported because "the model beat the
        baseline" is a claim, and this is the number behind it.
        """
        others = [c.smape for c in self.candidates if c.model != self.model]
        return None if not others else self.smape - min(others)


def rolling_origin_cutoffs(n_observations: int, *, horizon: int, windows: int) -> list[int]:
    """Return the index of the last training observation for each rolling-origin window.

    Args:
        n_observations: Length of the series.
        horizon: Steps scored after each cutoff.
        windows: Number of cutoffs, most recent last.

    Returns:
        Ascending cutoff indices. Window *i* trains on ``[0, cutoff_i)`` and scores
        ``[cutoff_i, cutoff_i + horizon)``, so the windows are non-overlapping in what they
        score (``step_size == horizon``) and every scored point lies strictly after every
        point its model saw.

    Raises:
        ValueError: If ``windows < 2`` (coverage measured on one window is a coin flip
            with no error bar), or if the series is too short to carve the windows out
            while leaving any history to fit on.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon!r}")
    if windows < 2:
        raise ValueError(
            f"windows must be >= 2 to measure coverage, got {windows!r} — a coverage rate "
            f"from a single window is one Bernoulli draw per step and carries no evidence."
        )
    needed = horizon * windows + 1
    if n_observations < needed + 1:
        raise ValueError(
            f"{n_observations} observations cannot support {windows} window(s) of horizon "
            f"{horizon}: that consumes {horizon * windows} points and leaves "
            f"{max(0, n_observations - horizon * windows)} to fit on."
        )
    last = n_observations - horizon
    return [last - i * horizon for i in range(windows - 1, -1, -1)]


def split_series(
    points: Sequence[SeriesObservation] | Sequence[tuple[datetime, float]],
    *,
    strategy: str = "chronological",
    test_size: float = 0.2,
) -> tuple[list[tuple[datetime, float]], list[tuple[datetime, float]]]:
    """Cut a series into train/test **by time**, or refuse.

    Args:
        points: The observed history in any order; sorted here.
        strategy: Must be ``"chronological"``. Any other value — ``"random"``,
            ``"shuffle"``, ``"stratified"``, ``"kfold"`` — raises.
        test_size: Fraction of observations held out at the **end** of the series.

    Returns:
        ``(train, test)`` as ``(timestamp, value)`` pairs, both oldest first, with every
        test timestamp strictly after every train timestamp.

    Raises:
        RandomSplitRefusedError: For any strategy that does not respect the arrow of time.
        ValueError: If ``test_size`` leaves either side empty.
    """
    if strategy != "chronological":
        raise RandomSplitRefusedError(strategy)
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size!r}")

    pairs = [
        (p.ts, p.value) if isinstance(p, SeriesObservation) else (p[0], float(p[1]))
        for p in points
    ]
    pairs.sort(key=lambda item: item[0])
    n_test = int(round(len(pairs) * test_size))
    if n_test < 1 or n_test >= len(pairs):
        raise ValueError(
            f"test_size={test_size} over {len(pairs)} observations leaves "
            f"{n_test} test rows and {len(pairs) - n_test} training rows; both must be >= 1"
        )
    return pairs[:-n_test], pairs[-n_test:]


def rank_candidates(candidates: Iterable[ForecastCandidate]) -> list[ForecastCandidate]:
    """Rank candidates best-first on measured sMAPE.

    Args:
        candidates: Scored candidates from either engine.

    Returns:
        A new list, ascending by sMAPE, ties broken by MAE then by name so the ordering is
        deterministic — a leaderboard that reshuffles between identical runs cannot be
        diffed, and diffing leaderboards is how a regression gets noticed.
    """
    return sorted(candidates, key=lambda c: (c.smape, c.mae, c.model))


def summarise(run: ForecastRun) -> BacktestSummary:
    """Project a :class:`~aegis_ml.forecast.engine.ForecastRun` into a backtest summary.

    Args:
        run: A completed forecast run.

    Returns:
        The measured half of that run, with the roster ranked.
    """
    return BacktestSummary(
        series_id=run.series_id,
        label=run.label,
        freq=run.freq,
        season_length=run.season_length,
        horizon=run.horizon,
        windows=run.backtest_windows,
        n_points=run.backtest_points,
        history_points=run.history_points,
        model=run.model,
        smape=run.smape,
        mape=run.mape,
        mae=run.mae,
        requested_coverage=run.requested_coverage,
        empirical_coverage=run.empirical_coverage,
        coverage_meets_request=run.coverage_meets_request,
        interval_method=run.interval_method,
        interval_method_detail=run.interval_method_detail,
        candidates=rank_candidates(run.candidates),
        excluded_models=dict(run.excluded_models),
        split_strategy="rolling-origin",
        generated_at=datetime.now(UTC),
    )


def backtest(
    points: Iterable[tuple[datetime, float] | SeriesObservation],
    problem_or_label: MLProblem | str,
    *,
    horizon: int,
    freq: str | None = None,
    level: float | None = None,
    windows: int = 3,
    conformal_windows: int = 3,
    data_source: str = "adapter",
    include_ml_candidates: bool = False,
) -> BacktestSummary:
    """Measure a series' forecastability by rolling-origin cross-validation.

    This is the measurement, not the forecast: it runs the same fit-select-score path
    :func:`aegis_ml.forecast.engine.forecast` runs and keeps only the measured half. Use it
    to answer "is this series forecastable at this horizon, and does the 90% band actually
    cover 90%?" before wiring a forecast into a decision.

    Args:
        points: The observed history, in any order.
        problem_or_label: An :class:`~aegis_ml.contracts.spec.MLProblem` or a plain label.
        horizon: Steps forecast from each cutoff.
        freq: Frequency alias; inferred from the observed spacing when ``None``.
        level: Coverage level to REQUEST; defaults to ``settings.requested_coverage``.
        windows: Rolling-origin cutoffs. Must be >= 2.
        conformal_windows: Windows the conformal band calibrates on, inside training only.
        data_source: Provenance tag recorded on the run.
        include_ml_candidates: Also score the ``mlforecast`` roster on the same windows.

    Returns:
        A :class:`BacktestSummary`. ``coverage_meets_request`` is the headline: a band that
        under-covers is a finding to report, not a number to round up.

    Raises:
        ValueError: If ``windows < 2``, or for an invalid horizon/level/frequency.
        InsufficientHistoryError: Propagated — too little history to fit, calibrate and
            score at this horizon, with the arithmetic attached.
        DegenerateSeriesError: Propagated — a flat series' 100% coverage from a zero-width
            band is not a measurement.
        ForecastFitError: Propagated — no candidate could be scored.
    """
    if windows < 2:
        raise ValueError(
            f"windows must be >= 2 to measure coverage, got {windows!r} — one window "
            f"gives one Bernoulli draw per step and no error bar on the rate."
        )
    run = forecast(
        points,
        problem_or_label,
        horizon=horizon,
        data_source=data_source,
        freq=freq,
        level=level,
        interval="conformal",
        backtest_windows=windows,
        conformal_windows=conformal_windows,
        include_history=False,
        include_ml_candidates=include_ml_candidates,
    )
    return summarise(run)
