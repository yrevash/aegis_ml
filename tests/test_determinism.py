"""Determinism — what makes the demo repeatable.

Every stochastic step in this package takes a seed, and the promise is narrow but total:
the same seed reproduces the frame, the split and the fitted model's predictions exactly;
a different seed does not. Both halves matter. A pipeline that always returns the same
thing regardless of seed is not deterministic, it is constant, and a constant is what a
broken generator looks like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from reference.adapter import ml_spec

from aegis_ml.automl import recipe as R
from aegis_ml.data.splits import three_way_split

ROWS = 500
"""Small enough to regenerate several times per test, large enough for the split to be real."""


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    """Two frames from one seed and one from another."""
    return {
        "a": ml_spec.training_frame(num_records=ROWS, seed=11),
        "a_again": ml_spec.training_frame(num_records=ROWS, seed=11),
        "b": ml_spec.training_frame(num_records=ROWS, seed=12),
    }


# ── frames ────────────────────────────────────────────────────────────────────


def test_same_seed_produces_an_identical_frame(frames) -> None:
    """Byte-for-byte, including the MAR holes and the categorical draw."""
    pd.testing.assert_frame_equal(frames["a"], frames["a_again"])


def test_different_seed_produces_a_different_frame(frames) -> None:
    """A generator that ignores its seed is a constant, not a generator."""
    a, b = frames["a"], frames["b"]
    assert not a.equals(b)
    common = min(len(a), len(b))
    target = "spoilage_risk_pct"
    assert not np.allclose(
        a[target].to_numpy()[:common], b[target].to_numpy()[:common], equal_nan=True
    )


def test_missingness_pattern_is_reproducible(frames) -> None:
    """The MAR holes are drawn from their own stream and must land in the same rows."""
    a, again = frames["a"], frames["a_again"]
    for column in a.columns:
        assert list(a[column].isna()) == list(again[column].isna())


# ── splits ────────────────────────────────────────────────────────────────────


def test_same_seed_produces_identical_splits(frames, problem) -> None:
    """The three-way partition is a function of the seed alone."""
    first = three_way_split(frames["a"], problem, seed=101)
    second = three_way_split(frames["a"], problem, seed=101)
    for left, right in zip(first, second, strict=True):
        assert list(left.index) == list(right.index)


def test_different_seed_produces_a_different_split(frames, problem) -> None:
    """Otherwise the "held-out" set is the same rows for every experiment ever run."""
    first = three_way_split(frames["a"], problem, seed=101)
    second = three_way_split(frames["a"], problem, seed=202)
    assert list(first.test.index) != list(second.test.index)
    assert set(first.test.index) != set(second.test.index)


# ── fitted predictions ────────────────────────────────────────────────────────


def test_same_seed_produces_identical_fitted_predictions(frames, problem) -> None:
    """Two fits of one recipe at one random_state must agree to the bit."""
    recipe = R.baseline_recipe(problem)
    features = frames["a"][problem.feature_names].head(50)

    first = R.fit_recipe(recipe, frames["a"], problem, random_state=7).predict(features)
    second = R.fit_recipe(recipe, frames["a"], problem, random_state=7).predict(features)

    assert np.array_equal(first, second), "the same recipe and seed produced two models"


def test_a_different_training_frame_produces_different_predictions(frames, problem) -> None:
    """The model is genuinely a function of the data, not of the recipe alone."""
    recipe = R.baseline_recipe(problem)
    features = frames["a"][problem.feature_names].head(50)

    from_a = R.fit_recipe(recipe, frames["a"], problem, random_state=7).predict(features)
    from_b = R.fit_recipe(recipe, frames["b"], problem, random_state=7).predict(features)

    assert not np.allclose(from_a, from_b)


def test_latent_target_draw_is_reproducible(frames, problem, latent) -> None:
    """``sample_frame`` must redraw the same labels for the same seed."""
    features = frames["a"].drop(columns=[problem.target.name])
    calibration = latent.calibrate(features, seed=5)
    first = latent.sample_frame(features, calibration=calibration, seed=5)
    second = latent.sample_frame(features, calibration=calibration, seed=5)
    assert np.allclose(first.to_numpy(), second.to_numpy())

    other = latent.sample_frame(features, calibration=calibration, seed=6)
    assert not np.allclose(first.to_numpy(), other.to_numpy())


def test_calibration_is_reproducible(frames, problem, latent) -> None:
    """The solved noise scale is a function of the frame and the seed, nothing else."""
    features = frames["a"].drop(columns=[problem.target.name])
    first = latent.calibrate(features, seed=5)
    second = latent.calibrate(features, seed=5)
    assert first.noise_sigma == pytest.approx(second.noise_sigma)
    assert first.confounder_scale == pytest.approx(second.confounder_scale)
    assert first.implied_r2_ceiling == pytest.approx(second.implied_r2_ceiling)


def test_learnability_probe_is_reproducible(frames, problem) -> None:
    """The headline number a demo quotes must not move between two runs of one command."""
    from aegis_ml.data.latent import measure_learnability

    first = measure_learnability(frames["a"], problem, seed=13)
    second = measure_learnability(frames["a"], problem, seed=13)
    assert first.metric_value == pytest.approx(second.metric_value)
    assert first.n_train == second.n_train and first.n_test == second.n_test
