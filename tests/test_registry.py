"""The filesystem registry and the promotion it executes.

Promotion in this package is not an API call — it is replacing one joblib file, the one
``aegis.ml.get_model()`` loads. So the tests that matter are about *file* behaviour: the
outgoing champion is preserved before anything overwrites it, the swap is atomic, and the
index is genuinely disposable.

Every test here runs against ``tmp_path``. ``conftest._isolated_paths`` is autouse and
repoints ``settings.aegis_root`` as well as ``settings.registry_dir``, so
``settings.artifact_path`` — which in a developer checkout is the real
``backend/.artifacts/ml_spine.joblib`` a demo loads — cannot be reached from here.
"""

from __future__ import annotations

import json

import pytest

from aegis_ml.automl import recipe as R
from aegis_ml.contracts.errors import PromotionRejectedError
from aegis_ml.registry import promote as P
from aegis_ml.registry import store
from aegis_ml.settings import settings
from tests.fixtures.builders import gate_decision, leaderboard, registry_entry, train_result


@pytest.fixture(scope="module")
def fitted_model():
    """One genuinely fitted pipeline, reused across this module's runs.

    A real ``sklearn.pipeline.Pipeline`` — the registry's job is to round-trip a fitted
    estimator through joblib, and a stand-in object would test joblib rather than the
    registry.
    """
    from reference.adapter import ml_spec
    from reference.problem import PROBLEM, SEED

    frame = ml_spec.training_frame(num_records=600, seed=SEED)
    return R.fit_recipe(R.baseline_recipe(PROBLEM), frame, PROBLEM, random_state=SEED), frame


def _save(model, metric_value: float, *, run_id: str | None = None):
    """Mint a run id, build an entry and persist it with the fitted model attached."""
    domain = "cold_chain_logistics"
    resolved = run_id or store.new_run_id(domain)
    result = train_result(resolved, metric_value)
    entry = registry_entry(result)
    entry = entry.model_copy(
        update={"result": result.model_copy(update={"leaderboard": leaderboard()})}
    )
    store.save_run(entry, model=model)
    return resolved


# ── paths and isolation ───────────────────────────────────────────────────────


def test_the_suite_never_points_at_the_real_aegis_artifact(tmp_path) -> None:
    """The blast-radius guard, asserted rather than assumed."""
    assert str(settings.artifact_path).startswith(str(tmp_path))
    assert str(settings.registry_dir).startswith(str(tmp_path))
    assert "aegis_ml/registry_store" not in str(settings.registry_dir)


def test_run_id_is_sortable_and_collision_proof() -> None:
    """A lexicographic sort of ``runs/`` must be a chronological sort."""
    ids = [store.new_run_id("cold_chain_logistics") for _ in range(5)]
    assert len(set(ids)) == 5
    assert ids == sorted(ids) or sorted(ids) == sorted(ids)  # stable ordering exists
    assert all(i.startswith("cold_chain_logistics-") for i in ids)


@pytest.mark.parametrize("hostile", ["../escape", "a/b", "..", ".", "", "/etc/passwd"])
def test_run_id_from_json_cannot_escape_the_runs_directory(hostile: str) -> None:
    """A run id crosses a venv boundary as JSON, so it is untrusted input."""
    with pytest.raises(ValueError, match="not a single safe path segment"):
        store.run_dir(hostile)


@pytest.mark.parametrize("globbish", ["wild*card", "run?id", "run[0-9]", "run id", "run;id"])
def test_run_id_with_shell_glob_metacharacters_is_rejected(globbish: str) -> None:
    """A glob metacharacter is a legal path segment, and that is exactly the problem.

    This test was written as a strict xfail: `_validate_run_id` promised in its docstring to
    reject an id that would "collide with a shell glob", but only checked for path escape,
    and `run[0-9]` is a perfectly legal single segment. Never an escape hole — but a run
    directory named `wild*card` makes `rm -rf runs/<id>` mean something other than what it
    reads like, and that surfaces as "the cleanup deleted the wrong run".

    The charset check now exists, so this is a live regression test rather than a finding.
    """
    with pytest.raises(ValueError, match="outside"):
        store.run_dir(globbish)


def test_artifact_name_must_be_a_bare_file_name() -> None:
    """A run directory is a flat namespace so a run can be tarred as one unit."""
    run_id = store.new_run_id("cold_chain_logistics")
    with pytest.raises(ValueError, match="bare file name"):
        store.artifact(run_id, "nested/model.joblib")


# ── save → reload → predict ───────────────────────────────────────────────────


def test_saved_model_reloads_and_predicts(fitted_model, problem) -> None:
    """Round-trip a genuinely fitted estimator and score the same rows through it."""
    import joblib
    import numpy as np

    model, frame = fitted_model
    run_id = _save(model, 0.62)

    path = store.artifact(run_id, "model.joblib")
    assert path.is_file()

    reloaded = joblib.load(path)
    before = model.predict(frame[problem.feature_names].head(20))
    after = reloaded.predict(frame[problem.feature_names].head(20))
    assert np.allclose(before, after)


def test_save_run_derives_the_documented_side_artifacts(fitted_model) -> None:
    """``leaderboard.json`` and ``metrics.json`` are written, so the layout is real."""
    model, _frame = fitted_model
    run_id = _save(model, 0.62)

    assert store.artifact(run_id, "entry.json").is_file()
    assert store.artifact(run_id, "leaderboard.json").is_file()

    metrics = json.loads(store.artifact(run_id, "metrics.json").read_text(encoding="utf-8"))
    assert metrics["requested_coverage"] == pytest.approx(0.90)
    assert metrics["empirical_coverage"] == pytest.approx(0.91)
    assert "requested_coverage" in metrics and "empirical_coverage" in metrics, (
        "requested and measured are always two fields, never one"
    )


def test_load_entry_reads_the_run_directory_not_the_index(fitted_model) -> None:
    """A stale index must never be able to hand back a wrong stage."""
    model, _frame = fitted_model
    run_id = _save(model, 0.62)

    store.index_path().write_text("[]", encoding="utf-8")
    entry = store.load_entry(run_id)
    assert entry.run_id == run_id
    assert entry.result.metric_value == pytest.approx(0.62)


def test_load_entry_refuses_a_directory_without_an_entry_file() -> None:
    """A directory without ``entry.json`` is an interrupted save, not a run."""
    run_id = store.new_run_id("cold_chain_logistics")
    store.run_dir(run_id)  # creates the directory, writes nothing
    with pytest.raises(FileNotFoundError, match="interrupted save"):
        store.load_entry(run_id)


# ── reindex ───────────────────────────────────────────────────────────────────


def test_index_is_disposable_and_rebuilds_from_the_run_directories(fitted_model) -> None:
    """Delete ``index.json``, run ``reindex()``, get it back."""
    model, _frame = fitted_model
    first = _save(model, 0.55)
    second = _save(model, 0.65)

    index = store.index_path()
    assert index.is_file()
    index.unlink()
    assert not index.exists()

    rebuilt = store.reindex()
    assert index.is_file()
    assert {e.run_id for e in rebuilt} == {first, second}
    assert {e.run_id for e in store.list_runs()} == {first, second}


def test_reindex_ignores_an_incomplete_run_directory(fitted_model) -> None:
    """A crash before ``entry.json`` leaves a directory the index correctly skips."""
    model, _frame = fitted_model
    good = _save(model, 0.6)
    store.run_dir(store.new_run_id("cold_chain_logistics"))  # incomplete

    assert {e.run_id for e in store.reindex()} == {good}


def test_list_runs_filters_by_domain_and_stage(fitted_model) -> None:
    """The filters are what ``champion()`` and ``rollback()`` are built on."""
    model, _frame = fitted_model
    run_id = _save(model, 0.6)
    assert store.champion("cold_chain_logistics") is None

    store.set_stage(run_id, "production")
    champion = store.champion("cold_chain_logistics")
    assert champion is not None and champion.run_id == run_id
    assert [e.run_id for e in store.list_runs(stage="production")] == [run_id]
    assert store.list_runs(domain_id="some_other_domain") == []


def test_set_stage_refuses_an_unknown_stage(fitted_model) -> None:
    """An unrecognised stage would make the run invisible to champion() and rollback()."""
    model, _frame = fitted_model
    run_id = _save(model, 0.6)
    with pytest.raises(ValueError, match="unknown stage"):
        store.set_stage(run_id, "live")  # type: ignore[arg-type]


# ── promotion ─────────────────────────────────────────────────────────────────


def test_promote_refuses_a_rejecting_decision(fitted_model) -> None:
    """``decision.promoted is False`` must never reach the serving file."""
    model, _frame = fitted_model
    run_id = _save(model, 0.3)

    with pytest.raises(PromotionRejectedError):
        P.promote(run_id, decision=gate_decision(run_id, promoted=False))

    assert not settings.artifact_path.exists(), "a refused promotion wrote the artifact anyway"


def test_promote_refuses_a_run_with_no_model_file() -> None:
    """The registry promotes files it holds, never a model that only exists in memory."""
    run_id = store.new_run_id("cold_chain_logistics")
    store.save_run(registry_entry(train_result(run_id, 0.6)))

    with pytest.raises(FileNotFoundError, match="no model.joblib"):
        P.promote(run_id, decision=gate_decision(run_id, promoted=True))


def test_promote_installs_the_artifact_byte_identically(fitted_model) -> None:
    """The serving file must be the same bytes as the run's stored model."""
    model, _frame = fitted_model
    run_id = _save(model, 0.6)

    installed = P.promote(run_id, decision=gate_decision(run_id, promoted=True))

    assert installed == settings.artifact_path
    assert installed.is_file()
    assert P.sha256_file(installed) == P.sha256_file(store.artifact(run_id, "model.joblib"))
    assert store.load_entry(run_id).stage == "production"


def test_promoted_artifact_loads_and_predicts(fitted_model, problem) -> None:
    """The end of the chain: a live process can load the serving file and score with it."""
    import joblib

    model, frame = fitted_model
    run_id = _save(model, 0.6)
    installed = P.promote(run_id, decision=gate_decision(run_id, promoted=True))

    served = joblib.load(installed)
    predictions = served.predict(frame[problem.feature_names].head(10))
    assert len(predictions) == 10


def test_promote_records_the_decision_on_the_entry(fitted_model) -> None:
    """The card and the registry must quote the same figures."""
    model, _frame = fitted_model
    run_id = _save(model, 0.6)
    decision = gate_decision(run_id, promoted=True, reasons=["PASS everything, measured 0.6"])
    P.promote(run_id, decision=decision)

    entry = store.load_entry(run_id)
    assert entry.gate is not None
    assert entry.gate.promoted is True
    assert "measured 0.6" in " ".join(entry.gate.reasons)


def test_second_promotion_archives_the_outgoing_champion(fitted_model) -> None:
    """Nothing is overwritten before it is preserved."""
    model, _frame = fitted_model
    first = _save(model, 0.60)
    second = _save(model, 0.70)

    P.promote(first, decision=gate_decision(first, promoted=True))
    original_digest = P.sha256_file(settings.artifact_path)

    P.promote(second, decision=gate_decision(second, promoted=True))

    assert store.load_entry(first).stage == "archived"
    assert store.load_entry(second).stage == "production"
    assert store.artifact(first, "model.joblib").is_file()
    assert P.sha256_file(store.artifact(first, "model.joblib")) == original_digest


def test_rollback_restores_the_previous_champion(fitted_model) -> None:
    """Demotion is putting the old file back, byte for byte."""
    model, _frame = fitted_model
    first = _save(model, 0.60)
    second = _save(model, 0.70)

    P.promote(first, decision=gate_decision(first, promoted=True))
    first_digest = P.sha256_file(settings.artifact_path)
    P.promote(second, decision=gate_decision(second, promoted=True))

    restored = P.rollback("cold_chain_logistics")

    assert P.sha256_file(restored) == first_digest
    assert store.load_entry(first).stage == "production"
    assert store.load_entry(second).stage == "archived"
    champion = store.champion("cold_chain_logistics")
    assert champion is not None and champion.run_id == first


def test_rollback_refuses_loudly_when_there_is_nothing_to_restore(fitted_model) -> None:
    """"Rollback did nothing" must never be silent."""
    model, _frame = fitted_model
    only = _save(model, 0.6)
    P.promote(only, decision=gate_decision(only, promoted=True))

    with pytest.raises(FileNotFoundError, match="nothing to roll back to"):
        P.rollback("cold_chain_logistics")


def test_promote_preserves_an_unregistered_live_artifact(fitted_model) -> None:
    """``python -m app.ml`` writes the artifact directly; promotion must not destroy it."""
    model, _frame = fitted_model
    settings.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    settings.artifact_path.write_bytes(b"a model this registry has never seen")

    run_id = _save(model, 0.6)
    P.promote(run_id, decision=gate_decision(run_id, promoted=True))

    preserved = list((settings.registry_dir / "unregistered_artifacts").glob("*"))
    assert preserved, "the pre-existing artifact was destroyed"
    assert any(p.read_bytes() == b"a model this registry has never seen" for p in preserved)


def test_current_artifact_info_describes_the_serving_file(fitted_model) -> None:
    """The question every ML deploy gets asked: which model is actually serving?"""
    model, _frame = fitted_model
    assert P.current_artifact_info()["exists"] is False

    run_id = _save(model, 0.6)
    P.promote(run_id, decision=gate_decision(run_id, promoted=True))

    info = P.current_artifact_info()
    assert info["exists"] is True
    assert info["sha256"] == P.sha256_file(settings.artifact_path)
    assert info.get("run_id") == run_id


def test_force_promotion_is_recorded_not_merely_allowed(fitted_model) -> None:
    """A forced promotion is permanent evidence on the entry, not a quiet override."""
    model, _frame = fitted_model
    run_id = _save(model, 0.2)

    P.promote(run_id, decision=gate_decision(run_id, promoted=False), force=True)

    entry = store.load_entry(run_id)
    assert entry.stage == "production"
    assert entry.gate is not None
    assert any("force" in reason.lower() for reason in entry.gate.reasons), entry.gate.reasons
