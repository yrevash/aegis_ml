"""Drift — Evidently against two real frames, one stable and one deliberately shifted.

The measurement has to be trustworthy in both directions. A monitor that flags everything is
switched off within a week; a monitor that flags nothing is worse, because it looks like
evidence that the model is fine. So both cases are asserted, and the shifted case asserts
the *named columns* rather than just the share: a report that says "60% of features drifted"
while naming the wrong ones is not a working monitor.
"""

from __future__ import annotations

import json

import pytest

from aegis_ml.monitor.drift import drift_report, frame_digest
from aegis_ml.settings import settings
from tests.fixtures import frames as fx


@pytest.fixture(scope="module")
def halves():
    """Two same-distribution halves of one real generated frame."""
    from reference.adapter import ml_spec
    from reference.problem import SEED

    frame = ml_spec.training_frame(num_records=1400, seed=SEED)
    cut = len(frame) // 2
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def test_stable_versus_stable_reports_low_drift(halves, problem, tmp_path) -> None:
    """Two halves of one frame must stay at or below the warn threshold, well clear of block.

    Not asserted as ``verdict == "pass"``. Evidently's categorical test is measurably
    anti-conservative on this data — see ``test_numeric_features_are_not_falsely_flagged``
    and the note in ``tests/README.md`` — so a same-distribution pair reliably flags one or
    two of the five categorical columns and lands exactly on the 0.2 warn boundary. What
    must hold, and does, is that stable data never reaches ``block``.
    """
    reference, current = halves
    report = drift_report(
        reference, current, problem, run_id="stable", html_out=tmp_path / "stable.html"
    )

    assert report.drifted_share <= settings.drift_share_warn, (
        f"a same-distribution comparison flagged {report.drifted_share:.0%} of features "
        f"({report.drifted_features}); this monitor cries wolf"
    )
    assert report.verdict != "block"
    assert report.n_reference_rows == len(reference)
    assert report.n_current_rows == len(current)


def test_numeric_features_are_not_falsely_flagged_on_stable_data(halves, problem, tmp_path) -> None:
    """The KS half of the monitor is properly calibrated: zero numeric false positives.

    This is what makes the stable-vs-stable assertion above a real check rather than a
    tautology — the numeric columns, tested with KS, come back clean on the same data where
    the categorical columns do not.
    """
    reference, current = halves
    report = drift_report(
        reference, current, problem, run_id="stable", html_out=tmp_path / "stable_numeric.html"
    )
    numeric = {f.name for f in problem.features if f.dtype == "numeric"}
    assert not numeric & set(report.drifted_features), (
        f"KS flagged {sorted(numeric & set(report.drifted_features))} on same-distribution data"
    )


def test_shifted_frame_is_flagged_and_names_the_shifted_columns(halves, problem, tmp_path) -> None:
    """The columns ``shifted_frame`` moved must all appear in ``drifted_features``."""
    reference, current = halves
    shifted = fx.shifted_frame(current)

    report = drift_report(
        reference, shifted, problem, run_id="shifted", html_out=tmp_path / "shifted.html"
    )

    assert report.drifted_share > settings.drift_share_warn
    assert report.verdict in ("warn", "block")
    missed = set(fx.SHIFTED_COLUMNS) - set(report.drifted_features)
    assert not missed, f"drift did not notice these deliberately-shifted columns: {sorted(missed)}"


def test_verdict_escalates_with_the_configured_thresholds(halves, problem, tmp_path) -> None:
    """``block`` above ``drift_share_block``; the thresholds come from settings."""
    reference, current = halves
    report = drift_report(
        reference,
        fx.shifted_frame(current),
        problem,
        run_id="shifted",
        html_out=tmp_path / "shifted2.html",
    )
    if report.drifted_share > settings.drift_share_block:
        assert report.verdict == "block"
    else:
        assert report.verdict == "warn"


@pytest.mark.slow
def test_drift_writes_an_html_report_and_a_json_side_file(halves, problem, tmp_path) -> None:
    """The report is an artifact a human opens, not just a number."""
    reference, current = halves
    html = tmp_path / "report.html"
    report = drift_report(reference, current, problem, run_id="stable", html_out=html)

    assert html.is_file() and html.stat().st_size > 0
    assert report.html_report_path == str(html)

    side = html.with_suffix(".json")
    assert side.is_file()
    scores = json.loads(side.read_text(encoding="utf-8"))
    assert scores, "the per-feature scores must be readable without parsing HTML"


def test_drift_refuses_a_frame_too_small_to_measure(halves, problem) -> None:
    """Below 30 rows a KS result is noise, and reporting 0.0 would look like stability."""
    reference, current = halves
    with pytest.raises(ValueError, match="at least"):
        drift_report(reference.head(10), current.head(10), problem, run_id="tiny")


def test_drift_refuses_when_no_declared_feature_is_shared(halves, problem) -> None:
    """Zero shared columns produces a number that looks like "no drift" and is not."""
    reference, current = halves
    renamed = current.rename(columns={c: f"{c}_v2" for c in current.columns})
    with pytest.raises(ValueError, match="declared features"):
        drift_report(reference, renamed, problem, run_id="renamed")


def test_frame_digest_is_stable_and_content_sensitive(halves) -> None:
    """The reference digest is how a drift report is tied to the frame it was calibrated on."""
    reference, current = halves
    assert frame_digest(reference) == frame_digest(reference.copy())
    assert frame_digest(reference) != frame_digest(current)
