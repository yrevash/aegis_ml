"""Splits and leakage — the two ways a held-out score stops meaning anything.

The conformal coverage guarantee rests entirely on the calibration split being disjoint from
the training split, and on the calibration split being large enough for the requested level
to be *attainable* at all. Leakage is the other half: a column that restates the target
produces a beautiful score and a model that cannot work.
"""

from __future__ import annotations

import math

import pytest

from aegis_ml.contracts.errors import AegisMLError, InsufficientLabelsError, TargetLeakageError
from aegis_ml.data.splits import (
    grouped_split,
    min_calibration_rows,
    stratified_split,
    three_way_split,
    time_ordered_split,
)
from aegis_ml.features.leakage import assert_no_leakage, detect_leakage
from aegis_ml.settings import settings
from tests.fixtures import frames as fx

# ── three-way split ───────────────────────────────────────────────────────────


def test_three_way_split_parts_are_disjoint_and_complete(frame, problem, seed) -> None:
    """Disjointness IS the coverage guarantee; completeness proves no rows were dropped."""
    train, calibration, test = three_way_split(frame, problem, seed=seed)

    train_ix, calib_ix, test_ix = (set(part.index) for part in (train, calibration, test))
    assert not train_ix & calib_ix
    assert not train_ix & test_ix
    assert not calib_ix & test_ix
    assert train_ix | calib_ix | test_ix == set(frame.index)
    assert len(train) + len(calibration) + len(test) == len(frame)


def test_three_way_split_sizes_follow_the_spine_convention(frame, problem, seed) -> None:
    """Test is carved off the WHOLE frame first, so 0.2/0.25 yields 60/20/20 — not 55/25/20."""
    n = len(frame)
    train, calibration, test = three_way_split(
        frame, problem, test_size=0.2, calibration_size=0.25, seed=seed
    )
    assert len(test) / n == pytest.approx(0.20, abs=0.02)
    assert len(calibration) / n == pytest.approx(0.20, abs=0.02)
    assert len(train) / n == pytest.approx(0.60, abs=0.02)


def test_three_way_split_is_deterministic_for_a_seed(frame, problem, seed) -> None:
    """A demo has to be repeatable, so the same seed must partition identically."""
    first = three_way_split(frame, problem, seed=seed)
    second = three_way_split(frame, problem, seed=seed)
    for a, b in zip(first, second, strict=True):
        assert list(a.index) == list(b.index)


def test_three_way_split_refuses_a_calibration_split_too_small_for_the_level(problem) -> None:
    """A level the calibration split cannot support is refused before the search runs."""
    import pandas as pd
    from reference.adapter import ml_spec

    tiny = ml_spec.training_frame(num_records=90, seed=5)
    assert isinstance(tiny, pd.DataFrame)
    with pytest.raises(InsufficientLabelsError):
        three_way_split(tiny, problem, confidence_level=0.995, seed=5)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_three_way_split_refuses_a_fraction_outside_the_unit_interval(frame, problem, bad) -> None:
    """A fraction of 0 or 1 leaves a part empty; it is refused, not clamped."""
    with pytest.raises(ValueError, match="must be in"):
        three_way_split(frame, problem, test_size=bad)


def test_stratified_split_preserves_class_balance(excursion_frame, excursion_problem) -> None:
    """Classification splits stratify, or a rare class can vanish from the test set."""
    train, test = stratified_split(excursion_frame, excursion_problem, test_size=0.25, seed=7)
    target = excursion_problem.target.name
    whole = excursion_frame[target].value_counts(normalize=True)
    for level, share in whole.items():
        assert test[target].value_counts(normalize=True).get(level, 0.0) == pytest.approx(
            share, abs=0.05
        )


# ── min_calibration_rows ──────────────────────────────────────────────────────


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9, 0.95, 0.99])
def test_min_calibration_rows_matches_the_conformal_rank_condition(level: float) -> None:
    """It must be the smallest n with ``ceil((n + 1) * level) <= n`` — MAPIE's own condition."""
    minimum = min_calibration_rows(level)
    assert math.ceil((minimum + 1) * level) <= minimum
    assert math.ceil(minimum * level) > minimum - 1
    for smaller in range(1, minimum):
        assert math.ceil((smaller + 1) * level) > smaller, f"{smaller} should not have qualified"


def test_min_calibration_rows_known_values() -> None:
    """The two the docstring quotes: 9 rows at 90%, 99 at 99%."""
    assert min_calibration_rows(0.9) == 9
    assert min_calibration_rows(0.99) == 99


def test_min_calibration_rows_guards_as_well_as_calculates() -> None:
    """Passing ``n`` turns the calculation into the refusal."""
    assert min_calibration_rows(0.9, 40) == 9
    with pytest.raises(InsufficientLabelsError) as excinfo:
        min_calibration_rows(0.9, 4)
    assert excinfo.value.need == 9
    assert excinfo.value.have == 4


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.2])
def test_min_calibration_rows_refuses_an_impossible_level(level: float) -> None:
    """A level outside (0, 1) is not a conformal level."""
    with pytest.raises(ValueError, match="confidence_level"):
        min_calibration_rows(level)


# ── grouped split ─────────────────────────────────────────────────────────────


def test_grouped_split_has_zero_group_overlap(frame, seed) -> None:
    """Leakage with no leaking column: the row's identity is the signal."""
    grouped = frame.copy()
    grouped["lane_id"] = [f"lane-{i % 40:02d}" for i in range(len(grouped))]

    left, right = grouped_split(grouped, group_column="lane_id", test_size=0.25, seed=seed)

    assert not set(left["lane_id"]) & set(right["lane_id"])
    assert set(left["lane_id"]) | set(right["lane_id"]) == set(grouped["lane_id"])
    assert len(left) + len(right) == len(grouped)
    assert 0.15 < len(right) / len(grouped) < 0.4


def test_grouped_split_refuses_an_absent_column(frame) -> None:
    """Naming the wrong column must raise, not silently split at random."""
    with pytest.raises(AegisMLError, match="does not have that column"):
        grouped_split(frame, group_column="not_a_column")


def test_grouped_split_refuses_a_single_group(frame) -> None:
    """One group means no group-respecting split exists at all."""
    single = frame.copy()
    single["lane_id"] = "one-and-only"
    with pytest.raises(AegisMLError, match="single group"):
        grouped_split(single, group_column="lane_id")


# ── time-ordered split ────────────────────────────────────────────────────────


def test_time_ordered_split_refuses_to_shuffle(frame) -> None:
    """Shuffling a time series voids the coverage guarantee, so the flag raises."""
    with pytest.raises(ValueError) as excinfo:
        time_ordered_split(frame, shuffle=True)
    message = str(excinfo.value)
    assert "refuses shuffle=True" in message
    assert "coverage guarantee" in message


def test_time_ordered_split_is_chronological(frame) -> None:
    """Everything in the earlier frame precedes everything in the later one."""
    earlier, later = time_ordered_split(frame, test_size=0.2)
    assert max(earlier.index) < min(later.index)
    assert len(earlier) + len(later) == len(frame)
    assert len(later) / len(frame) == pytest.approx(0.2, abs=0.01)


def test_time_ordered_split_sorts_by_a_named_column(frame) -> None:
    """With a time column the frame is ordered by it, whatever the row order was."""
    timed = frame.sample(frac=1.0, random_state=1).copy()
    timed["booked_at"] = range(len(timed) - 1, -1, -1)
    earlier, later = time_ordered_split(timed, time_column="booked_at", test_size=0.25)
    assert earlier["booked_at"].max() < later["booked_at"].min()


def test_time_ordered_split_refuses_an_absent_time_column(frame) -> None:
    """Ordering by a column that is not there would silently use row order instead."""
    with pytest.raises(AegisMLError, match="does not have that column"):
        time_ordered_split(frame, time_column="nope")


def test_time_ordered_split_refuses_a_frame_with_no_future(frame) -> None:
    """A split that leaves one side empty is refused rather than returned empty."""
    with pytest.raises(AegisMLError, match="empty"):
        time_ordered_split(frame.head(4), test_size=0.9)


# ── leakage ───────────────────────────────────────────────────────────────────


def test_leakage_detector_is_silent_on_a_clean_frame(frame, problem, seed) -> None:
    """A detector that cries wolf gets switched off; the real frame must produce nothing."""
    assert detect_leakage(frame, problem, seed=seed) == []
    assert assert_no_leakage(frame, problem, seed=seed) == []


def test_leakage_detector_finds_an_injected_affine_restatement(frame, problem, seed) -> None:
    """``target * 1.0001`` is the canonical leak and must be flagged by name."""
    leaky = fx.leaky_frame(frame, problem)
    declared = fx.with_leak_declared(problem)

    signals = detect_leakage(leaky, declared, seed=seed)

    assert signals, "an affine restatement of the target went undetected"
    flagged = {signal.feature for signal in signals}
    assert flagged == {fx.LEAK_COLUMN}, f"only the leak should be flagged, got {flagged}"
    worst = max(signals, key=lambda s: s.score)
    assert worst.score >= settings.leakage_threshold
    assert worst.threshold == pytest.approx(settings.leakage_threshold)


def test_assert_no_leakage_raises_naming_the_feature(frame, problem, seed) -> None:
    """The refusal carries the column name and the score behind it."""
    leaky = fx.leaky_frame(frame, problem)
    declared = fx.with_leak_declared(problem)

    with pytest.raises(TargetLeakageError) as excinfo:
        assert_no_leakage(leaky, declared, seed=seed)

    assert excinfo.value.feature == fx.LEAK_COLUMN
    assert "leakage, not signal" in str(excinfo.value)


def test_declared_leakage_can_be_allowed_explicitly(frame, problem, seed) -> None:
    """A genuinely-available-at-prediction-time column is allowed by naming it, not by luck."""
    leaky = fx.leaky_frame(frame, problem)
    declared = fx.with_leak_declared(problem)
    assert assert_no_leakage(leaky, declared, allow=[fx.LEAK_COLUMN], seed=seed) == []


def test_leakage_threshold_is_read_from_settings(frame, problem, seed) -> None:
    """Lowering the threshold catches more; the default comes from settings, not a literal."""
    leaky = fx.leaky_frame(frame, problem)
    declared = fx.with_leak_declared(problem)
    signals = detect_leakage(leaky, declared, seed=seed)
    assert all(s.threshold == pytest.approx(settings.leakage_threshold) for s in signals)
