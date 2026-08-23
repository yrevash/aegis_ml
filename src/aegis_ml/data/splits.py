"""Splits that match how Aegis actually splits — and refuse the ones that leak.

Three of these functions exist because a split is the cheapest place in the whole pipeline
to invalidate a conformal guarantee without producing an error.

**The split must match the spine's.** ``aegis.ml.model.train`` carves the test set off
first, then splits the remainder into a training set and a *disjoint* calibration set, and
stratifies both cuts for classification. :func:`three_way_split` reproduces that exactly. It
is not decoration: an AutoML search that scores candidates on a differently-shaped split is
ranking them on a different problem than the one the promoted model will be measured on,
and the leaderboard number a demo quotes stops describing the served model.

**The calibration set must be large enough for the level you asked for.** Split conformal
takes the ``ceil((n + 1) · level)``-th smallest calibration residual as the interval
half-width. When that rank exceeds ``n`` there is no such residual, and the requested
coverage is unattainable no matter what the data looks like — a 5-row calibration split
cannot support a 90% interval. :func:`min_calibration_rows` replicates Aegis's guard so the
refusal happens before an expensive search rather than at ``fit`` time.

**A time series must never be shuffled.** ``aegis.forecast.engine`` states the rule
outright: *"``aegis.ml.model`` calibrates its conformal predictor on a random
``train_test_split``. On a time series that is a leak: calibration rows drawn from after
the rows the model trained on make the residual distribution optimistic and void the
coverage guarantee."* :func:`time_ordered_split` therefore raises rather than accepting a
``shuffle=True`` it would have to ignore.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, NamedTuple

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError, InsufficientLabelsError
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

    import pandas as pd

    from aegis_ml.contracts.spec import MLProblem

__all__ = [
    "ThreeWaySplit",
    "grouped_split",
    "min_calibration_rows",
    "stratified_split",
    "three_way_split",
    "time_ordered_split",
]

logger = logging.getLogger(__name__)

_EXTRA = "aegis-ml[serve]"
"""Install target quoted when scikit-learn or pandas are missing."""


def _model_selection() -> ModuleType:
    """Import ``sklearn.model_selection`` through :func:`~aegis_ml._require.require`."""
    return require(_EXTRA, "sklearn.model_selection")


class ThreeWaySplit(NamedTuple):
    """The three disjoint frames a conformalised fit needs.

    A plain tuple rather than a pydantic model because it carries DataFrames, and because
    every caller immediately unpacks it. The names are the ones ``TrainResult`` reports
    (``training_size`` / ``calibration_size`` / ``test_size``), so the numbers on a model
    card can be traced back to the object that produced them.

    Attributes:
        train: Rows the estimator fits on.
        calibration: Rows MAPIE calibrates on. Disjoint from ``train`` — that disjointness
            is the entire basis of the coverage guarantee.
        test: Rows nothing has seen, used to measure the metric and the *empirical* coverage.
    """

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def min_calibration_rows(confidence_level: float, n: int | None = None) -> int:
    """Smallest calibration set for which ``confidence_level`` is even attainable.

    Replicates ``aegis.ml.model._min_calibration_rows``. Split conformal needs the
    ``ceil((n + 1) · level)``-th smallest residual to exist, so it searches for the smallest
    ``n`` where that rank is in range. At 90% that is 9 rows; at 99% it is 99.

    Passing ``n`` turns the function into the guard as well as the calculation: it is the
    number of calibration rows you actually have, and an insufficient count is refused with
    :class:`~aegis_ml.contracts.errors.InsufficientLabelsError` rather than allowed through
    to produce an interval that cannot mean what it claims.

    Args:
        confidence_level: Requested marginal coverage, strictly in ``(0, 1)``.
        n: Calibration rows available. When given and too small, this raises.

    Returns:
        The minimum number of calibration rows.

    Raises:
        ValueError: When ``confidence_level`` is not strictly between 0 and 1.
        InsufficientLabelsError: When ``n`` is given and falls below the minimum.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1); got {confidence_level!r}")
    minimum = 1
    while math.ceil((minimum + 1) * confidence_level) > minimum:
        minimum += 1
    if n is not None and n < minimum:
        raise InsufficientLabelsError(
            have=n,
            need=minimum,
            what=f"A {confidence_level:.0%} split-conformal calibration split",
        )
    return minimum


def stratified_split(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    test_size: float = 0.2,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a labelled frame in two, stratifying on the label for classification.

    Stratification is not a nicety here. Without it a rare class can land entirely outside
    the calibration split, and its conformal sets then carry no guarantee at all while still
    being rendered with a confidence percentage next to them. Aegis handles a frame too
    degenerate to stratify by logging and continuing rather than failing the whole fit; this
    matches that behaviour deliberately, because the alternative — refusing — would make a
    3-row rare class fatal to a pipeline that is otherwise fine.

    Args:
        frame: The labelled frame. Both halves keep every column, target included.
        problem: The problem, read for its task and target name.
        test_size: Fraction routed to the second frame.
        seed: Split seed; defaults to ``settings.random_seed``.

    Returns:
        ``(first, second)`` — conventionally ``(train, test)``.

    Raises:
        AegisMLError: When the target column is absent from ``frame``.
    """
    model_selection = _model_selection()
    resolved_seed = settings.random_seed if seed is None else seed
    target = problem.target.name
    if target not in frame.columns:
        raise AegisMLError(
            f"Cannot split on target {target!r}: the frame does not have that column. "
            f"Present: {sorted(frame.columns)[:12]}"
        )
    if problem.target.task == "classification":
        try:
            return model_selection.train_test_split(
                frame,
                test_size=test_size,
                random_state=resolved_seed,
                stratify=frame[target],
            )
        except ValueError:
            logger.warning(
                "Cannot stratify the split for %s (a class has fewer than 2 rows); falling "
                "back to an unstratified split — conformal sets for a class absent from "
                "calibration carry no coverage guarantee.",
                problem.domain_id,
            )
    return model_selection.train_test_split(frame, test_size=test_size, random_state=resolved_seed)


def grouped_split(
    frame: pd.DataFrame,
    *,
    group_column: str,
    test_size: float = 0.2,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that no group's rows appear on both sides.

    Use this whenever several rows describe the same real-world entity: three sensor
    readings from one shipment, five tickets from one customer, a facility that appears in
    forty rows. A random row-level split puts near-duplicate rows in train and test, the
    model memorises the entity rather than the relationship, and the held-out score
    overstates what will happen on an entity it has never seen. That is leakage with no
    leaking *column* — :mod:`aegis_ml.features.leakage` cannot see it, because the offending
    signal is the row's identity rather than its values.

    Args:
        frame: The frame to split.
        group_column: Column holding the entity id.
        test_size: Approximate fraction of *groups* routed to the second frame.
        seed: Split seed; defaults to ``settings.random_seed``.

    Returns:
        ``(first, second)``, disjoint in ``group_column``.

    Raises:
        AegisMLError: When the group column is absent, or holds a single group (in which
            case no group-respecting split exists at all).
    """
    model_selection = _model_selection()
    resolved_seed = settings.random_seed if seed is None else seed
    if group_column not in frame.columns:
        raise AegisMLError(
            f"Cannot group-split on {group_column!r}: the frame does not have that column."
        )
    groups = frame[group_column]
    if groups.nunique(dropna=False) < 2:
        raise AegisMLError(
            f"Column {group_column!r} holds a single group across all {len(frame)} rows, so "
            f"no split can keep groups disjoint. Either the wrong column was named or the "
            f"frame really is one entity, in which case a grouped split is not the tool."
        )
    splitter = model_selection.GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=resolved_seed
    )
    first_idx, second_idx = next(splitter.split(frame, groups=groups))
    return frame.iloc[first_idx].copy(), frame.iloc[second_idx].copy()


def time_ordered_split(
    frame: pd.DataFrame,
    *,
    time_column: str | None = None,
    test_size: float = 0.2,
    shuffle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series chronologically: everything before the cutoff, everything after.

    ``shuffle=True`` is refused, loudly, rather than honoured or quietly ignored.
    ``aegis.forecast.engine`` documents exactly why: calibration rows drawn from *after* the
    rows the model trained on make the residual distribution optimistic and void the
    coverage guarantee, so a randomly split time series produces an interval that is
    narrower than the truth and labelled with a confidence it does not have. A caller who
    asked for shuffling has a mistaken mental model, and silently doing the right thing
    would leave them holding it.

    Args:
        frame: The frame to split.
        time_column: Column to sort by. ``None`` trusts the frame's existing row order,
            which is what a generator that emitted rows chronologically produces.
        test_size: Fraction of rows routed to the later frame.
        shuffle: Must be ``False``. Present only so the mistake raises here.

    Returns:
        ``(earlier, later)`` — train on the first, measure on the second.

    Raises:
        ValueError: When ``shuffle`` is true, or ``test_size`` is not in ``(0, 1)``.
        AegisMLError: When ``time_column`` is named but absent, or the frame is too short
            to yield a non-empty split on both sides.
    """
    if shuffle:
        raise ValueError(
            "time_ordered_split refuses shuffle=True. aegis.forecast documents random "
            "splitting of a time series as a leak: calibration rows drawn from after the "
            "training rows make the residual distribution optimistic and void the coverage "
            "guarantee. Use stratified_split for i.i.d. rows, or drop the flag."
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size!r}")
    ordered = frame
    if time_column is not None:
        if time_column not in frame.columns:
            raise AegisMLError(
                f"Cannot order by {time_column!r}: the frame does not have that column."
            )
        ordered = frame.sort_values(time_column, kind="mergesort")
    cut = int(len(ordered) * (1.0 - test_size))
    if cut <= 0 or cut >= len(ordered):
        raise AegisMLError(
            f"A {test_size:.0%} chronological split of {len(ordered)} rows leaves one side "
            f"empty. A time-ordered split needs enough history to have both a past and a "
            f"future; supply more rows or lower test_size."
        )
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def three_way_split(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    test_size: float = 0.2,
    calibration_size: float = 0.25,
    seed: int | None = None,
    confidence_level: float | None = None,
) -> ThreeWaySplit:
    """Reproduce ``aegis.ml.model.train``'s split: test first, then train/calibration.

    The order matters and is easy to get subtly wrong. Carving the test set off the *whole*
    frame first means ``calibration_size`` is a fraction of what remains, not of the total —
    so the defaults (0.2 / 0.25) yield 60/20/20, not 55/25/20. Any other reading produces a
    calibration split of a different size from the spine's, and MAPIE's interval width is a
    direct function of that size.

    The guard at the end is the one Aegis raises at fit time, pulled forward: a calibration
    split too small for the requested conformal level makes that level unattainable, and
    finding that out after an AutoML search has run is an expensive way to learn it.

    Args:
        frame: The labelled frame.
        problem: The problem, read for its task (stratification) and target name.
        test_size: Fraction of *all* rows held out for measurement.
        calibration_size: Fraction of the *non-test* rows reserved for calibration.
        seed: Split seed; defaults to ``settings.random_seed``.
        confidence_level: Conformal level the calibration split must be able to support.
            Defaults to the problem's ``requested_coverage``.

    Returns:
        A :class:`ThreeWaySplit`.

    Raises:
        InsufficientLabelsError: When the calibration split cannot support the level.
        AegisMLError: When the target column is absent, or a split comes out empty.
        ValueError: When either fraction is outside ``(0, 1)``.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size!r}")
    if not 0.0 < calibration_size < 1.0:
        raise ValueError(f"calibration_size must be in (0, 1); got {calibration_size!r}")

    resolved_seed = settings.random_seed if seed is None else seed
    level = problem.requested_coverage if confidence_level is None else confidence_level

    fit_frame, test = stratified_split(frame, problem, test_size=test_size, seed=resolved_seed)
    train, calibration = stratified_split(
        fit_frame, problem, test_size=calibration_size, seed=resolved_seed
    )
    if train.empty or calibration.empty or test.empty:
        raise AegisMLError(
            f"A {1 - test_size:.0%}/{test_size:.0%} then {1 - calibration_size:.0%}/"
            f"{calibration_size:.0%} split of {len(frame)} rows left an empty part "
            f"(train={len(train)}, calibration={len(calibration)}, test={len(test)}). "
            f"Generate more rows before training."
        )
    min_calibration_rows(level, len(calibration))
    logger.debug(
        "three_way_split(%s): train=%d calibration=%d test=%d at level %.2f",
        problem.domain_id,
        len(train),
        len(calibration),
        len(test),
        level,
    )
    return ThreeWaySplit(train=train, calibration=calibration, test=test)
