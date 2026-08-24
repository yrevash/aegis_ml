"""Realism — the property the whole package exists to defend.

A generated frame is only useful as evidence if the model has to *work* for its score. Four
things have to hold at once, and each has its own failure story:

* the held-out score lands **inside** the declared band, not above it;
* a noise-free target trips the ceiling guard rather than being celebrated;
* a target drawn independently of the features is refused outright;
* the confounder draw and the noise draw are **independent streams**.

That last one is a permanent regression test for a bug that shipped: both vectors were
rebuilt from the same seed, which made them perfectly correlated, doubled the effective
spread and put the realised R² far below the one ``calibrate`` had solved for. Nothing
raised. The frame was simply harder than it declared itself to be.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis_ml.contracts.errors import AegisMLError, LabelNotLearnableError
from aegis_ml.data.latent import (
    _STREAM_CONFOUNDER,
    _STREAM_MISSINGNESS,
    _STREAM_NOISE,
    R2_CEILING,
    assert_learnable,
    measure_learnability,
    realism_report,
)
from aegis_ml.pipelines.flows import realism_band_for
from aegis_ml.settings import settings
from tests.fixtures import frames as fx


def test_held_out_score_lands_inside_the_declared_band(frame, problem, seed) -> None:
    """The real generated frame scores inside ``realism_band_for``, not above it."""
    floor, ceiling = realism_band_for(problem)
    report = measure_learnability(frame, problem, seed=seed)

    assert report.metric_name == "r2"
    assert report.metric_value >= floor, (
        f"r2={report.metric_value:.4f} is below the realism floor {floor}: the target is "
        f"closer to noise than to signal."
    )
    assert report.metric_value <= ceiling, (
        f"r2={report.metric_value:.4f} is above the realism ceiling {ceiling}: a generated "
        f"frame this easy describes a world that does not exist."
    )
    assert report.learnable is True
    assert report.suspiciously_easy is False


def test_band_floor_is_read_from_settings_not_hardcoded(frame, problem, seed) -> None:
    """The learnability floor the probe applies is ``settings.learnable_r2_floor``."""
    report = measure_learnability(frame, problem, seed=seed)
    assert report.floor == pytest.approx(settings.learnable_r2_floor)
    assert report.effective_floor == pytest.approx(settings.learnable_r2_floor)
    assert report.ceiling == pytest.approx(R2_CEILING)


def test_noise_free_target_is_flagged_suspiciously_easy(frame, problem, latent, seed) -> None:
    """A deterministic target trips ``suspiciously_easy`` instead of scoring as a success."""
    deterministic = fx.noise_free_target(frame, problem, latent)
    report = measure_learnability(deterministic, problem, seed=seed)

    assert report.metric_value > R2_CEILING, "the no-noise frame should be near-perfectly fit"
    assert report.suspiciously_easy is True
    assert report.notes, "a suspiciously easy frame must say why in its notes"
    assert "ceiling" in " ".join(report.notes).lower()


def test_noise_free_target_is_refused_by_the_ceiling_guard(frame, problem, latent, seed) -> None:
    """``assert_learnable(strict_ceiling=True)`` refuses a frame nothing had to learn."""
    deterministic = fx.noise_free_target(frame, problem, latent)
    with pytest.raises(AegisMLError) as excinfo:
        assert_learnable(deterministic, problem, strict_ceiling=True, seed=seed)
    assert "ceiling" in str(excinfo.value).lower()


def test_noise_free_target_passes_when_the_ceiling_is_not_strict(
    frame, problem, latent, seed
) -> None:
    """Without ``strict_ceiling`` the score is returned, so the guard is opt-in and explicit."""
    deterministic = fx.noise_free_target(frame, problem, latent)
    score = assert_learnable(deterministic, problem, seed=seed)
    assert score > R2_CEILING


def test_pure_noise_target_raises_label_not_learnable(frame, problem, seed) -> None:
    """A label drawn independently of the features is refused, with the number in the message."""
    noise = fx.pure_noise_target(frame, problem, seed=3)
    with pytest.raises(LabelNotLearnableError) as excinfo:
        assert_learnable(noise, problem, seed=seed)

    error = excinfo.value
    assert error.metric == "r2"
    assert error.value < error.floor
    assert error.floor == pytest.approx(settings.learnable_r2_floor)
    assert "latent_fn(features)" in str(error)


def test_pure_noise_target_measures_as_not_learnable(frame, problem, seed) -> None:
    """The probe reports ``learnable=False`` rather than raising, for callers wanting the number."""
    noise = fx.pure_noise_target(frame, problem, seed=3)
    report = measure_learnability(noise, problem, seed=seed)
    assert report.learnable is False
    assert report.metric_value < settings.learnable_r2_floor


# ── realism_report ────────────────────────────────────────────────────────────


def test_realism_report_reports_missingness(frame, problem, latent, seed) -> None:
    """MAR holes punched by the generator are measured and named."""
    report = realism_report(frame, problem, latent, seed=seed)
    missingness = report["missingness"]
    assert missingness, "the reference generator declares a missingness rule; it must show up"
    assert "sensor_gap_minutes" in missingness
    assert 0.0 < missingness["sensor_gap_minutes"] < 0.5
    assert frame["sensor_gap_minutes"].isna().mean() == pytest.approx(
        missingness["sensor_gap_minutes"]
    )


def test_realism_report_reports_confounder_share(frame, problem, latent, seed) -> None:
    """The unobserved-confounder contribution is quantified, not asserted."""
    noise = realism_report(frame, problem, latent, seed=seed)["noise"]
    assert noise["confounder_variance"] > 0.0
    assert 0.0 < noise["confounder_share"] < 1.0
    assert noise["sigma"] > 0.0
    assert noise["noise_to_signal"] > 0.0

    named = realism_report(frame, problem, latent, seed=seed)["latent"]["confounders"]
    assert named == [c.name for c in latent.confounders]
    assert named, "the reference domain declares unobserved confounders; they must be named"


def test_realism_report_reports_heteroscedasticity(frame, problem, latent, seed) -> None:
    """Residual spread in the top quartile of the noise driver exceeds the bottom quartile."""
    noise = realism_report(frame, problem, latent, seed=seed)["noise"]
    assert noise["heteroscedastic_feature"] == latent.realism.heteroscedastic_feature
    ratio = noise["heteroscedasticity_ratio"]
    assert ratio is not None
    assert ratio > 1.0, (
        f"heteroscedasticity_ratio={ratio}: the declared noise driver is not actually "
        f"widening the residuals, so the conformal interval has nothing to be wide about."
    )


def test_realism_report_reports_class_balance(excursion_frame, excursion_problem, seed) -> None:
    """For a classification frame the report carries the class shares and the majority share."""
    report = realism_report(excursion_frame, excursion_problem, seed=seed)
    balance = report["class_balance"]
    assert set(balance) <= set(excursion_problem.target.levels)
    assert sum(balance.values()) == pytest.approx(1.0)
    assert report["achieved"]["majority_share"] == pytest.approx(max(balance.values()), abs=0.05)


def test_realism_report_omits_latent_figures_when_no_latent_is_supplied(
    frame, problem, seed
) -> None:
    """Without a latent model the calibration figures are absent, never inferred."""
    report = realism_report(frame, problem, None, seed=seed)
    assert "noise" not in report
    assert "latent" not in report
    assert any("omitted rather than inferred" in note for note in report["notes"])


def test_realism_report_names_undriven_features(frame, problem, latent, seed) -> None:
    """Filler features are named, so a SHAP story can be checked against the truth."""
    latent_block = realism_report(frame, problem, latent, seed=seed)["latent"]
    driven = set(latent_block["driven_features"])
    undriven = set(latent_block["undriven_features"])
    assert driven and undriven, "a realistic frame has both real drivers and filler"
    assert not driven & undriven
    assert driven | undriven == set(problem.feature_names)


# ── the RNG-stream regression test ────────────────────────────────────────────


def test_confounder_and_noise_streams_are_independent(frame, latent, seed) -> None:
    """REGRESSION: the confounder draw and the noise draw must not be the same vector.

    This shipped once. Both were rebuilt from ``default_rng(seed)``, so ``confounders`` and
    the standard normal used for measurement noise were bit-identical — perfectly
    correlated, which doubles the effective spread and drops the realised R² well below the
    one ``calibrate`` solved for, silently.
    """
    n = len(frame)
    confounder = np.asarray(
        latent._confounder_values(n, latent._rng(seed, _STREAM_CONFOUNDER)), dtype="float64"
    )
    noise = np.asarray(
        latent._rng(seed, _STREAM_NOISE).standard_normal(n), dtype="float64"
    )

    assert not np.allclose(confounder, noise), "confounder and noise are the SAME vector"
    correlation = float(np.corrcoef(confounder, noise)[0, 1])
    assert abs(correlation) < 0.2, (
        f"confounder/noise correlation is {correlation:+.4f}; the two streams are not "
        f"independent, so the realised R2 will sit below the calibrated one with nothing "
        f"raising."
    )


def test_every_declared_stream_is_distinct(frame, latent, seed) -> None:
    """All three named streams differ pairwise — not just confounder against noise."""
    n = len(frame)
    draws = {
        stream: np.asarray(latent._rng(seed, stream).standard_normal(n), dtype="float64")
        for stream in (_STREAM_CONFOUNDER, _STREAM_NOISE, _STREAM_MISSINGNESS)
    }
    streams = sorted(draws)
    for i, left in enumerate(streams):
        for right in streams[i + 1 :]:
            correlation = float(np.corrcoef(draws[left], draws[right])[0, 1])
            assert abs(correlation) < 0.2, (
                f"streams {left} and {right} correlate {correlation:+.4f}"
            )


def test_a_stream_is_reproducible_for_a_fixed_seed(frame, latent, seed) -> None:
    """Independence must not have been bought with non-determinism."""
    n = len(frame)
    first = latent._confounder_values(n, latent._rng(seed, _STREAM_CONFOUNDER))
    second = latent._confounder_values(n, latent._rng(seed, _STREAM_CONFOUNDER))
    assert np.allclose(np.asarray(first), np.asarray(second))


def test_realised_r2_tracks_the_calibrated_ceiling(frame, problem, latent, seed) -> None:
    """The measured score must be a believable fraction of the oracle score.

    This is the *consequence* the stream bug produced: a headroom far below 1 means the
    frame is harder than the latent model declared. A ratio near or above 1 would mean the
    probe is somehow beating the oracle, which is its own bug.
    """
    report = realism_report(frame, problem, latent, seed=seed)
    headroom = report["achieved"]["headroom"]
    oracle = report["noise"]["oracle_r2"]
    assert oracle is not None and oracle > 0.5, f"oracle_r2={oracle}: the two evaluators disagree"
    assert 0.5 < headroom < 1.05, (
        f"headroom={headroom:.3f}: the achieved score is {report['achieved']['value']:.4f} "
        f"against an oracle of {oracle:.4f}."
    )
