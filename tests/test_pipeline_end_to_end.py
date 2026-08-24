"""The chain a demo actually runs: contract → search → slices → coverage → gate → promote.

Each of the other test modules checks one component against its own contract. This one
checks that the components still fit together, because the failure that survives a full
unit suite is the one where two modules each behave correctly and disagree about the shape
between them.

Nothing here is mocked. The AutoML search really runs (baseline plus FLAML on a small time
budget), the recipe is really re-fitted, the gate really decides, and the artifact really
lands in ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis_ml.automl.search import run_search
from aegis_ml.automl.tiers import available_tiers
from aegis_ml.contracts.protocols import RegistryEntry, TrainResult
from aegis_ml.data.contract_check import check
from aegis_ml.data.splits import three_way_split
from aegis_ml.evaluate.calibration import coverage
from aegis_ml.evaluate.gate import evaluate_gate
from aegis_ml.evaluate.metrics import primary, score
from aegis_ml.evaluate.slices import slice_metrics, worst_slice
from aegis_ml.features.leakage import detect_leakage
from aegis_ml.registry import promote as P
from aegis_ml.registry import store
from aegis_ml.settings import settings
from tests.fixtures import frames as fx

# ── the data contract report ──────────────────────────────────────────────────


def test_contract_check_passes_on_the_reference_frame(frame, problem, seed) -> None:
    """One report covering schema, learnability and leakage — what the CLI prints."""
    report = check(frame, problem, seed=seed)

    assert report.domain_id == problem.domain_id
    assert report.schema_ok is True
    assert report.schema_error is None
    assert report.learnable is True
    assert report.suspiciously_easy is False
    assert report.leakage == []
    assert report.issues == []
    assert report.ok is True
    assert report.n_rows == len(frame)
    assert {c.column for c in report.columns} >= set(problem.feature_names)


def test_contract_check_reports_a_leak_without_raising(frame, problem, seed) -> None:
    """The report is a report: it collects every finding rather than stopping at the first."""
    leaky = fx.leaky_frame(frame, problem)
    report = check(leaky, fx.with_leak_declared(problem), seed=seed)

    assert report.ok is False
    assert any(fx.LEAK_COLUMN in str(item) for item in report.leakage)
    assert report.issues, "an un-ok report with no issues cannot be acted on"


def test_contract_check_reports_a_schema_violation(frame, problem, seed) -> None:
    """A frame the pandera contract rejects comes back with ``schema_ok=False`` and the reason."""
    bad = frame.copy()
    categorical = next(f for f in problem.features if f.dtype == "categorical")
    bad.loc[bad.index[0], categorical.name] = "not_a_declared_level"

    report = check(bad, problem, include_leakage=False, seed=seed)

    assert report.schema_ok is False
    assert report.schema_error is not None
    assert categorical.name in report.schema_error
    assert report.ok is False


# ── the AutoML search ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_run_search_returns_a_portable_recipe_and_a_full_leaderboard(frame, problem, seed) -> None:
    """The search's whole job: a recipe the serving venv can fit, and the losers beside it."""
    from aegis_ml.automl import recipe as R

    winner, board = run_search(frame, problem, tiers=["baseline"], time_budget=5, seed=seed)

    R.assert_portable(winner)
    assert winner.tier in ("baseline", "flaml")
    assert winner.members

    assert board.metric_name == problem.metric
    assert board.higher_is_better is True
    assert len(board.candidates) >= 2, "a leaderboard with one row cannot show a margin"
    assert sum(1 for c in board.candidates if c.selected) == 1
    assert "baseline" in board.tiers_run


@pytest.mark.slow
def test_a_skipped_tier_is_never_silent(frame, problem, seed) -> None:
    """An empty leaderboard slot and an unavailable dependency must not look identical."""
    unavailable = [t for t, ok in available_tiers().items() if not ok]
    assert unavailable, "every tier is available here; this assertion has nothing to check"

    _winner, board = run_search(
        frame, problem, tiers=["baseline", *unavailable], time_budget=5, seed=seed
    )

    for tier in unavailable:
        assert tier in board.tiers_skipped, f"tier {tier!r} was dropped without a reason"
        assert board.tiers_skipped[tier], f"tier {tier!r} was skipped with an empty reason"
        assert tier not in board.tiers_run


# ── slices ────────────────────────────────────────────────────────────────────


def test_slice_sweep_produces_measurable_segments(frame, problem, seed) -> None:
    """The gate reads the worst slice, so the sweep has to actually produce slices."""
    from aegis_ml.automl import recipe as R

    train, _calibration, test = three_way_split(frame, problem, seed=seed)
    pipeline = R.fit_recipe(R.baseline_recipe(problem), train, problem, random_state=seed)
    predictions = pipeline.predict(test[problem.feature_names])

    metrics = slice_metrics(test, test[problem.target.name], predictions, problem, min_rows=20)

    assert metrics, "no slice was measurable on a 190-row held-out split"
    assert all(m.metric_name == problem.metric for m in metrics)
    assert all(m.n_rows >= 20 for m in metrics)

    worst = worst_slice(metrics)
    assert worst is not None
    assert worst.metric_value == min(m.metric_value for m in metrics), (
        "for an higher-is-better metric the WORST slice is the minimum"
    )


# ── the whole chain ───────────────────────────────────────────────────────────


@pytest.mark.slow
def test_train_evaluate_gate_and_promote(frame, problem, seed) -> None:
    """Contract → split → fit → measure → slice → coverage → gate → registry → promote.

    The end state asserted is the one that matters to a live backend: the file
    ``aegis.ml.get_model()`` loads holds this run's model, and the registry says so.
    """
    from aegis_ml.automl import recipe as R

    # 1. contract
    report = check(frame, problem, seed=seed)
    assert report.ok is True

    # 2. split
    train, calibration, test = three_way_split(frame, problem, seed=seed)

    # 3. fit
    recipe = R.baseline_recipe(problem)
    pipeline = R.fit_recipe(recipe, train, problem, random_state=seed)

    # 4. measure on the untouched test split
    truth = test[problem.target.name]
    predictions = pipeline.predict(test[problem.feature_names])
    metric_name, metric_value = primary(problem, score(problem, truth, predictions))
    assert metric_value > 0.3, f"the chain produced an unusable model ({metric_value:.4f})"

    # 5. a real split-conformal interval from the calibration residuals
    calibration_residuals = np.abs(
        calibration[problem.target.name].to_numpy()
        - pipeline.predict(calibration[problem.feature_names])
    )
    level = problem.requested_coverage
    width = float(np.quantile(calibration_residuals, level, method="higher"))
    intervals = [(float(p) - width, float(p) + width) for p in predictions]
    empirical = coverage(truth, intervals)
    assert empirical == pytest.approx(level, abs=0.08), (
        f"a split-conformal interval calibrated at {level} covered {empirical:.3f}"
    )

    # 6. slices and leakage
    slices = slice_metrics(test, truth, predictions, problem, min_rows=20)
    leakage = detect_leakage(frame, problem, seed=seed)
    assert leakage == []

    # 7. the gate, with no champion
    result = TrainResult(
        run_id=store.new_run_id(problem.domain_id),
        domain_id=problem.domain_id,
        task="regression",
        target=problem.target.name,
        metric_name=metric_name,
        metric_value=metric_value,
        requested_coverage=level,
        empirical_coverage=empirical,
        training_size=len(train),
        calibration_size=len(calibration),
        test_size=len(test),
        recipe=recipe,
        slices=slices,
    )
    decision = evaluate_gate(result, None, contract_ok=report.ok, leakage=leakage)
    assert decision.promoted is True, decision.reasons

    # 8. registry + promotion
    from datetime import UTC, datetime

    entry = RegistryEntry(
        run_id=result.run_id,
        domain_id=problem.domain_id,
        created_at=datetime.now(UTC).isoformat(),
        result=result,
    )
    store.save_run(entry, model=pipeline)
    installed = P.promote(result.run_id, decision=decision)

    assert installed == settings.artifact_path
    assert P.sha256_file(installed) == P.sha256_file(
        store.artifact(result.run_id, "model.joblib")
    )
    champion = store.champion(problem.domain_id)
    assert champion is not None and champion.run_id == result.run_id
    assert champion.gate is not None and champion.gate.promoted is True

    # 9. and the served file predicts
    import joblib

    assert len(joblib.load(installed).predict(test[problem.feature_names].head(5))) == 5


@pytest.mark.slow
def test_a_second_worse_run_is_refused_and_the_champion_survives(frame, problem, seed) -> None:
    """The point of the gate: a worse challenger leaves the serving file untouched."""
    from aegis_ml.automl import recipe as R
    from aegis_ml.contracts.errors import PromotionRejectedError
    from tests.fixtures.builders import gate_decision, registry_entry, train_result

    pipeline = R.fit_recipe(R.baseline_recipe(problem), frame, problem, random_state=seed)

    champion_id = store.new_run_id(problem.domain_id)
    store.save_run(registry_entry(train_result(champion_id, 0.70)), model=pipeline)
    P.promote(champion_id, decision=gate_decision(champion_id, promoted=True))
    champion_digest = P.sha256_file(settings.artifact_path)

    challenger_id = store.new_run_id(problem.domain_id)
    store.save_run(registry_entry(train_result(challenger_id, 0.40)), model=pipeline)

    decision = evaluate_gate(
        train_result(challenger_id, 0.40),
        store.load_entry(champion_id).result,
        contract_ok=True,
        leakage=[],
    )
    assert decision.promoted is False

    with pytest.raises(PromotionRejectedError):
        P.promote(challenger_id, decision=decision)

    assert P.sha256_file(settings.artifact_path) == champion_digest
    survivor = store.champion(problem.domain_id)
    assert survivor is not None and survivor.run_id == champion_id
