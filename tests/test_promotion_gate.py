"""The promotion gate: five criteria, all required, every number recorded.

The gate is the last thing between an AutoML search and the file a live backend loads.
Every criterion covers a different failure, and the tests here are organised the same way,
one section per criterion, plus the two entry points.

The ``TrainResult`` objects come from ``tests.fixtures.builders``: the gate compares floats,
and producing those floats by fitting models would make a gate test fail whenever an
estimator changed. The types are the real ones; only the provenance of the numbers is
synthetic.
"""

from __future__ import annotations

import pytest

from aegis_ml.contracts.errors import PromotionRejectedError
from aegis_ml.evaluate.gate import (
    CRITERIA,
    GateConfig,
    evaluate_gate,
    format_decision,
    promote_or_raise,
)
from aegis_ml.settings import settings
from tests.fixtures.builders import slices, train_result


def _gate(challenger, champion=None, *, contract_ok=True, leakage=(), config=None):
    """Run the gate with the boring arguments defaulted, so each test shows only its variable."""
    return evaluate_gate(
        challenger, champion, contract_ok=contract_ok, leakage=leakage, config=config
    )


# ── the whole decision ────────────────────────────────────────────────────────


def test_every_criterion_is_evaluated_even_after_one_fails() -> None:
    """A decision that names only the first problem hides the one that reappears next."""
    challenger = train_result("chal", 0.40, empirical_coverage=None)
    champion = train_result("champ", 0.70)
    decision = _gate(challenger, champion, contract_ok=False, leakage=["settled_risk_pct"])

    assert set(decision.checks) == set(CRITERIA)
    assert decision.promoted is False
    assert sum(decision.checks.values()) == 0, "this challenger fails all five"


def test_reasons_are_populated_on_a_pass_as_well_as_a_fail() -> None:
    """"Promoted" with no figures is as opaque as "rejected" with no figures."""
    challenger = train_result("chal", 0.80, worst_slices=slices(0.75, 0.70))
    champion = train_result("champ", 0.60, worst_slices=slices(0.55, 0.50))
    decision = _gate(challenger, champion)

    assert decision.promoted is True
    assert len(decision.reasons) == len(CRITERIA) + 1, (
        "one sentence per criterion, plus the verdict"
    )
    assert all(r.startswith(("PASS", "PROMOTED")) for r in decision.reasons), decision.reasons
    assert decision.metrics["gain"] == pytest.approx(0.20)
    assert decision.metrics["challenger_r2"] == pytest.approx(0.80)
    assert decision.metrics["champion_r2"] == pytest.approx(0.60)


def test_gate_records_the_champion_run_id() -> None:
    """The decision is auditable: it names both sides."""
    decision = _gate(train_result("chal", 0.8), train_result("champ", 0.6))
    assert decision.challenger_run_id == "chal"
    assert decision.champion_run_id == "champ"


# ── criterion 1: beats_champion ───────────────────────────────────────────────


def test_worse_challenger_is_rejected() -> None:
    """A challenger that scores below the champion fails, with both numbers quoted."""
    decision = _gate(train_result("chal", 0.50), train_result("champ", 0.70))
    assert decision.checks["beats_champion"] is False
    assert decision.promoted is False
    assert decision.metrics["gain"] == pytest.approx(-0.20)
    fail = next(r for r in decision.reasons if r.startswith("FAIL beats_champion"))
    assert "0.5000" in fail and "0.7000" in fail


def test_better_challenger_is_accepted() -> None:
    """A genuinely better challenger clears every criterion and is promoted."""
    challenger = train_result("chal", 0.80, worst_slices=slices(0.70, 0.65))
    champion = train_result("champ", 0.60, worst_slices=slices(0.55, 0.50))
    decision = _gate(challenger, champion)
    assert decision.promoted is True
    assert all(decision.checks[key] for key in CRITERIA)


def test_a_gain_inside_the_noise_margin_is_rejected() -> None:
    """The margin exists because run-to-run movement of that size is noise."""
    margin = settings.promote_min_gain
    challenger = train_result("chal", 0.60 + margin / 2, worst_slices=slices(0.70))
    champion = train_result("champ", 0.60, worst_slices=slices(0.55))
    decision = _gate(challenger, champion)
    assert decision.checks["beats_champion"] is False
    assert decision.metrics["min_gain"] == pytest.approx(margin)


def test_direction_is_honoured_for_a_lower_is_better_metric() -> None:
    """On rmse, the SMALLER challenger wins — this is what a wrong direction breaks."""
    challenger = train_result(
        "chal", 3.0, metric_name="rmse", worst_slices=slices(4.0, metric_name="rmse")
    )
    champion = train_result(
        "champ", 5.0, metric_name="rmse", worst_slices=slices(6.0, metric_name="rmse")
    )
    decision = _gate(challenger, champion)
    assert decision.checks["beats_champion"] is True
    assert decision.metrics["gain"] == pytest.approx(2.0)
    assert decision.promoted is True

    reversed_decision = _gate(champion, challenger)
    assert reversed_decision.checks["beats_champion"] is False


def test_two_metrics_are_never_compared() -> None:
    """Comparing r2 against rmse produces a real number that means nothing."""
    challenger = train_result("chal", 0.9)
    champion = train_result("champ", 3.0, metric_name="rmse")
    decision = _gate(challenger, champion)
    assert decision.checks["beats_champion"] is False
    assert "two scales" in " ".join(decision.reasons)


def test_first_model_passes_explicitly_and_says_it_did_so_trivially() -> None:
    """The no-champion case is a documented PASS, never a silent one."""
    challenger = train_result("first", 0.55)
    decision = _gate(challenger, None)

    assert decision.champion_run_id is None
    assert decision.checks["beats_champion"] is True
    assert decision.checks["worst_slice_not_worse"] is True
    assert decision.promoted is True

    reason = next(r for r in decision.reasons if "beats_champion" in r)
    assert reason.startswith("PASS")
    assert "trivially" in reason
    assert "no champion exists" in reason
    assert "NOT" in reason, "it must say it did not outperform anything"


# ── criterion 2: coverage_meets_request ───────────────────────────────────────


def test_missing_empirical_coverage_fails() -> None:
    """Unmeasured is not met: the calibrated interval is the product being shipped."""
    challenger = train_result("chal", 0.90, empirical_coverage=None)
    decision = _gate(challenger, None)
    assert decision.checks["coverage_meets_request"] is False
    assert decision.promoted is False
    assert "NOTHING WAS MEASURED" in " ".join(decision.reasons)


def test_missing_coverage_can_be_waived_but_the_waiver_is_recorded() -> None:
    """Waiving it is allowed; hiding the waiver is not."""
    challenger = train_result("chal", 0.90, empirical_coverage=None)
    decision = _gate(challenger, None, config=GateConfig(require_empirical_coverage=False))
    assert decision.checks["coverage_meets_request"] is True
    assert "WAIVED" in " ".join(decision.reasons)
    assert "UNVERIFIED" in " ".join(decision.reasons)


def test_coverage_below_the_tolerance_floor_fails() -> None:
    """A measured coverage under requested-minus-tolerance is an under-covering interval."""
    tolerance = settings.coverage_tolerance
    challenger = train_result("chal", 0.90, empirical_coverage=0.90 - tolerance - 0.02)
    decision = _gate(challenger, None)
    assert decision.checks["coverage_meets_request"] is False
    assert decision.metrics["coverage_floor"] == pytest.approx(0.90 - tolerance)


def test_coverage_inside_the_tolerance_passes() -> None:
    """A small shortfall inside the tolerance stands for sampling error, not slack."""
    tolerance = settings.coverage_tolerance
    challenger = train_result("chal", 0.90, empirical_coverage=0.90 - tolerance / 2)
    decision = _gate(challenger, None)
    assert decision.checks["coverage_meets_request"] is True
    assert decision.metrics["coverage_gap"] == pytest.approx(-tolerance / 2)


# ── criterion 3: contracts_pass ───────────────────────────────────────────────


def test_failed_data_contract_blocks_promotion() -> None:
    """A frame that violated its contract cannot produce a promotable model."""
    decision = _gate(train_result("chal", 0.9), None, contract_ok=False)
    assert decision.checks["contracts_pass"] is False
    assert decision.promoted is False


def test_skipped_contract_check_is_reported_as_skipped_not_passed() -> None:
    """``require_contracts=False`` still leaves the key present and False."""
    decision = _gate(
        train_result("chal", 0.9),
        None,
        contract_ok=False,
        config=GateConfig(require_contracts=False),
    )
    assert "contracts_pass" in decision.checks
    assert decision.checks["contracts_pass"] is False
    assert "SKIPPED" in " ".join(decision.reasons)


# ── criterion 4: worst_slice_not_worse ────────────────────────────────────────


def test_a_challenger_with_no_slice_sweep_fails() -> None:
    """An unmeasured segment distribution is missing evidence, not a pass."""
    decision = _gate(train_result("chal", 0.9, worst_slices=[]), None)
    assert decision.checks["worst_slice_not_worse"] is False
    assert "no slice measured" in " ".join(decision.reasons)


def test_a_collapsed_slice_is_caught_even_when_the_headline_improves() -> None:
    """The exact case the criterion exists for: better on average, worse for one segment."""
    challenger = train_result("chal", 0.85, worst_slices=slices(0.85, 0.20))
    champion = train_result("champ", 0.60, worst_slices=slices(0.62, 0.55))
    decision = _gate(challenger, champion)

    assert decision.checks["beats_champion"] is True, "the headline really did improve"
    assert decision.checks["worst_slice_not_worse"] is False
    assert decision.promoted is False
    assert decision.metrics["challenger_worst_slice"] == pytest.approx(0.20)
    assert decision.metrics["champion_worst_slice"] == pytest.approx(0.55)


def test_slice_tolerance_defaults_to_no_regression_permitted() -> None:
    """The default tolerance is zero — widening it must be a deliberate act."""
    assert GateConfig().slice_tolerance == 0.0


# ── criterion 5: no_target_leakage ────────────────────────────────────────────


@pytest.mark.parametrize(
    "finding",
    [
        "settled_risk_pct",
        {"feature": "settled_risk_pct", "score": 1.0},
        pytest.param(object(), id="unrenderable-object"),
    ],
)
def test_any_leakage_finding_blocks_promotion(finding: object) -> None:
    """A finding that could not be rendered must still block, never be dropped."""
    decision = _gate(train_result("chal", 0.9), None, leakage=[finding])
    assert decision.checks["no_target_leakage"] is False
    assert decision.promoted is False


def test_no_leakage_findings_passes_and_says_the_audit_ran() -> None:
    """An empty findings list is a PASS with a sentence, not an absent criterion."""
    decision = _gate(train_result("chal", 0.9), None, leakage=[])
    assert decision.checks["no_target_leakage"] is True
    assert any("no_target_leakage" in r for r in decision.reasons)


# ── promote_or_raise ──────────────────────────────────────────────────────────


def test_promote_or_raise_raises_on_a_rejected_challenger() -> None:
    """A caller whose next statement overwrites the artifact cannot step over a rejection."""
    challenger = train_result("chal", 0.40)
    champion = train_result("champ", 0.80)
    with pytest.raises(PromotionRejectedError) as excinfo:
        promote_or_raise(challenger, champion, contract_ok=True, leakage=[])

    reasons = excinfo.value.reasons
    assert reasons, "a rejection with no reasons is indistinguishable from a gate bug"
    assert any(r.startswith("FAIL beats_champion") for r in reasons)
    assert "0.4000" in str(excinfo.value)


def test_promote_or_raise_returns_the_decision_on_a_pass() -> None:
    """On a pass it hands back the same decision ``evaluate_gate`` would have returned."""
    challenger = train_result("chal", 0.80, worst_slices=slices(0.70))
    champion = train_result("champ", 0.60, worst_slices=slices(0.55))
    decision = promote_or_raise(challenger, champion, contract_ok=True, leakage=[])
    assert decision.promoted is True
    assert decision.checks == _gate(challenger, champion).checks


# ── rendering ─────────────────────────────────────────────────────────────────


def test_format_decision_renders_every_criterion_and_number() -> None:
    """One renderer, so the terminal, the card and the registry cannot disagree."""
    decision = _gate(train_result("chal", 0.40), train_result("champ", 0.80))
    text = format_decision(decision)

    assert "Promotion gate: REJECTED" in text
    for key in CRITERIA:
        assert key in text
    assert "challenger: chal" in text
    assert "champion:   champ" in text
    assert "gain" in text


def test_format_decision_names_the_absent_champion_explicitly() -> None:
    """A blank champion line would read as a missing value rather than a first model."""
    text = format_decision(_gate(train_result("first", 0.6), None))
    assert "(none — first model in this domain)" in text


def test_leakage_audit_that_never_ran_is_not_a_pass():
    """`None` leakage means the audit did not run, which must never read as clean.

    Regression test for a real defect. `flows.promote_flow` fed the gate
    `gate_inputs.get("leakage", [])`, and an empty list means "the audit ran and found
    nothing". So a run with no `gate_inputs.json` — the exact case the surrounding code
    logged as UNPROVEN — was told it had passed criterion 5, and the decision printed
    "PASS no_target_leakage: the feature audit flagged nothing" about an audit that had
    never happened. Criterion 1 actively rewards a leaking feature, because leakage
    produces the best held-out score, so criterion 5 is the only thing standing in its way.
    """
    unknown = _gate(train_result("chal", 0.9), None, leakage=None)
    assert unknown.checks["no_target_leakage"] is False
    assert not unknown.promoted
    assert any("did not run" in r for r in unknown.reasons)

    clean = _gate(train_result("chal", 0.9), None, leakage=[])
    assert clean.checks["no_target_leakage"] is True
    assert any("flagged nothing" in r for r in clean.reasons)
