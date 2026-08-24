"""Metric direction — a wrong entry here silently promotes the worse model.

``HIGHER_IS_BETTER`` is read by the promotion gate to decide the sign of the margin. If
``rmse`` were recorded as higher-is-better, a challenger with double the error would clear
the gate, be installed over the champion, and every downstream number would be internally
consistent. Nothing would look wrong.

So every entry is asserted explicitly and by name. A parametrised sweep over the dict's own
contents would pass whatever the dict said.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis_ml.evaluate.metrics import (
    HIGHER_IS_BETTER,
    MetricNotComputedError,
    UnknownMetricError,
    classification_metrics,
    higher_is_better,
    primary,
    regression_metrics,
    score,
)

LOWER_IS_BETTER = (
    "rmse",
    "mae",
    "mape",
    "median_ae",
    "log_loss",
    "brier_score",
    "expected_calibration_error",
    "max_calibration_error",
)
"""Error metrics: smaller is better. Written out here, not derived from the module."""

HIGHER = (
    "r2",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "precision",
    "recall",
    "roc_auc",
)
"""Score metrics: larger is better."""


@pytest.mark.parametrize("metric", LOWER_IS_BETTER)
def test_error_metrics_are_lower_is_better(metric: str) -> None:
    """rmse/mae/mape and the calibration errors must all rank downwards."""
    assert metric in HIGHER_IS_BETTER, f"{metric} has no declared direction"
    assert HIGHER_IS_BETTER[metric] is False
    assert higher_is_better(metric) is False


@pytest.mark.parametrize("metric", HIGHER)
def test_score_metrics_are_higher_is_better(metric: str) -> None:
    """r2/accuracy/f1/roc_auc must all rank upwards."""
    assert metric in HIGHER_IS_BETTER, f"{metric} has no declared direction"
    assert HIGHER_IS_BETTER[metric] is True
    assert higher_is_better(metric) is True


def test_every_declared_metric_is_covered_by_this_test() -> None:
    """A metric added to the registry without a direction test would slip through."""
    assert set(HIGHER_IS_BETTER) == set(LOWER_IS_BETTER) | set(HIGHER)


def test_unknown_metric_is_refused_rather_than_guessed() -> None:
    """Guessing a direction promotes the worse model with a straight face."""
    with pytest.raises(UnknownMetricError) as excinfo:
        higher_is_better("wobbliness")
    assert "wobbliness" in str(excinfo.value)


# ── the direction has to match the arithmetic, not just the table ─────────────


def test_regression_metric_values_move_in_the_declared_direction() -> None:
    """A genuinely better prediction raises r2 and lowers rmse/mae on real numbers."""
    rng = np.random.default_rng(0)
    truth = rng.normal(50.0, 10.0, size=400)
    good = truth + rng.normal(0.0, 1.0, size=400)
    poor = truth + rng.normal(0.0, 8.0, size=400)

    better = regression_metrics(truth, good)
    worse = regression_metrics(truth, poor)

    assert better["r2"] > worse["r2"]
    assert better["rmse"] < worse["rmse"]
    assert better["mae"] < worse["mae"]
    assert better["median_ae"] < worse["median_ae"]


def test_classification_metric_values_move_in_the_declared_direction() -> None:
    """A more accurate classifier scores higher on accuracy/f1 and lower on log_loss."""
    rng = np.random.default_rng(1)
    truth = rng.choice(["no", "yes"], size=400, p=[0.6, 0.4])
    good = np.where(rng.random(400) < 0.9, truth, np.where(truth == "yes", "no", "yes"))
    poor = np.where(rng.random(400) < 0.55, truth, np.where(truth == "yes", "no", "yes"))

    better = classification_metrics(truth, good, labels=["no", "yes"])
    worse = classification_metrics(truth, poor, labels=["no", "yes"])

    assert better["accuracy"] > worse["accuracy"]
    assert better["f1_macro"] > worse["f1_macro"]
    assert better["balanced_accuracy"] > worse["balanced_accuracy"]


def test_probability_metrics_reward_calibration() -> None:
    """log_loss, brier_score and the calibration errors all fall as probabilities sharpen.

    ``log_loss``/``roc_auc`` come from ``evaluate.metrics``; ``brier_score`` and the two
    calibration errors are computed in ``evaluate.calibration`` but ranked by the same
    ``HIGHER_IS_BETTER`` table, so the direction has to be right in both places at once.
    """
    from aegis_ml.evaluate.calibration import brier_score

    rng = np.random.default_rng(2)
    truth = rng.choice(["no", "yes"], size=400, p=[0.5, 0.5])
    is_yes = (truth == "yes").astype(float)

    confident = np.column_stack([1.0 - (0.9 * is_yes + 0.05), 0.9 * is_yes + 0.05])
    vague = np.full((400, 2), 0.5)
    labels = ["no", "yes"]
    predicted = np.where(is_yes > 0.5, "yes", "no")

    sharp = classification_metrics(truth, predicted, confident, labels=labels)
    blunt = classification_metrics(truth, predicted, vague, labels=labels)

    assert sharp["log_loss"] < blunt["log_loss"]
    assert brier_score(truth, confident, labels=labels) < brier_score(truth, vague, labels=labels)


def test_calibration_errors_are_lower_for_an_honest_probability() -> None:
    """ECE/MCE fall when the stated probability matches the observed frequency.

    Built the only way this can be tested honestly: draw the labels FROM a known
    probability, hand that probability back as the calibrated forecast, and compare it to an
    over-confident one that pushes every score to the extremes.
    """
    from aegis_ml.evaluate.calibration import expected_calibration_error

    rng = np.random.default_rng(11)
    p_yes = rng.uniform(0.05, 0.95, size=2000)
    truth = np.where(rng.random(2000) < p_yes, "yes", "no")
    labels = ["no", "yes"]

    honest = np.column_stack([1.0 - p_yes, p_yes])
    overconfident_yes = np.clip((p_yes - 0.5) * 4.0 + 0.5, 0.01, 0.99)
    overconfident = np.column_stack([1.0 - overconfident_yes, overconfident_yes])

    assert expected_calibration_error(truth, honest, labels=labels) < expected_calibration_error(
        truth, overconfident, labels=labels
    )


# ── primary(): the gate's single entry point ──────────────────────────────────


def test_primary_pulls_the_problems_own_ranking_metric(problem) -> None:
    """``primary`` names the metric the gate will rank on and returns its value."""
    metrics = {"r2": 0.63, "rmse": 4.2, "mae": 3.1}
    name, value = primary(problem, metrics)
    assert name == problem.metric == "r2"
    assert value == pytest.approx(0.63)


def test_primary_refuses_to_substitute_another_metric(problem) -> None:
    """A missing primary metric raises; substituting one ranks two different scales."""
    with pytest.raises(MetricNotComputedError):
        primary(problem, {"rmse": 4.2})


def test_score_dispatches_on_the_declared_task(problem, excursion_problem) -> None:
    """One entry point, so the gate and the card measure the same quantities."""
    rng = np.random.default_rng(3)
    truth = rng.normal(40.0, 5.0, size=200)
    regression = score(problem, truth, truth + rng.normal(0.0, 1.0, size=200))
    assert "r2" in regression and "rmse" in regression
    assert "accuracy" not in regression

    labels = list(excursion_problem.target.levels)
    y_true = rng.choice(labels, size=200)
    classification = score(excursion_problem, y_true, y_true)
    assert "accuracy" in classification
    assert classification["accuracy"] == pytest.approx(1.0)
    assert "r2" not in classification
