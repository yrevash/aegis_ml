"""Portability — the keystone of the two-venv split, and where the last real bug was.

The AutoML search runs in a trainer venv holding AutoGluon/TabPFN/torch. Its answer crosses
back into the serving venv as JSON and is **re-fitted** there. That only works if
``is_portable_kind`` means *fittable here*, not *importable here* — a distinction that is
currently load-bearing rather than theoretical:

    lightgbm 4.5 is installed and imports cleanly in this venv. Its sklearn wrapper calls
    ``check_X_y(..., force_all_finite=)``, a keyword scikit-learn removed in 1.8, and this
    venv has sklearn 1.9. ``import lightgbm`` succeeds; ``LGBMRegressor().fit()`` raises.

So the first test below is not a formality. It is the regression test for treating
importability as usability, and it would fail the moment that probe is weakened.
"""

from __future__ import annotations

import json

import pytest

from aegis_ml._require import is_available
from aegis_ml.automl import recipe as R
from aegis_ml.contracts.errors import RecipeNotPortableError
from aegis_ml.contracts.protocols import Recipe, RecipeMember

# ── the keystone ──────────────────────────────────────────────────────────────


def test_lightgbm_is_importable_but_not_portable_in_this_venv() -> None:
    """``is_portable_kind`` must mean FITTABLE, not IMPORTABLE.

    lightgbm is installed here, so a check that only asked ``is_available`` would say
    ``True`` and hand the serving venv a recipe it cannot fit.
    """
    assert is_available("lightgbm"), (
        "this test only has teeth while lightgbm is installed; if the venv changed, the "
        "importable-vs-fittable distinction must be re-asserted another way"
    )
    assert R.is_portable_kind("LGBMRegressor") is False
    assert R.is_portable_kind("LGBMClassifier") is False


def test_lightgbm_is_on_the_allowlist_despite_not_being_portable() -> None:
    """The refusal comes from the fit probe, not from an absent allowlist entry."""
    assert "LGBMRegressor" in R.PORTABLE_KINDS
    assert R.PORTABLE_KINDS["LGBMRegressor"] == "lightgbm"


@pytest.mark.parametrize(
    "kind",
    [
        "XGBRegressor",
        "XGBClassifier",
        "HistGradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "RandomForestRegressor",
        "ExtraTreesRegressor",
    ],
)
def test_serving_venv_estimators_are_portable(kind: str) -> None:
    """Everything the serving venv can actually fit reports as portable."""
    assert R.is_portable_kind(kind) is True


def test_portability_is_checked_against_the_task() -> None:
    """A classifier cannot slip into a regression recipe under a portable label."""
    assert R.is_portable_kind("XGBRegressor", task="regression") is True
    assert R.is_portable_kind("XGBRegressor", task="classification") is False
    assert R.is_portable_kind("XGBClassifier", task="classification") is True
    assert R.is_portable_kind("XGBClassifier", task="regression") is False


def test_unknown_kind_is_not_portable() -> None:
    """A class name that arrives as data is never imported on trust."""
    assert R.is_portable_kind("TabPFNRegressor") is False
    assert R.is_portable_kind("AutogluonTabularPredictor") is False
    assert R.is_portable_kind("os.system") is False


def test_estimator_class_refuses_a_non_allowlisted_name() -> None:
    """``estimator_class`` raises with the allowlist in the message."""
    with pytest.raises(RecipeNotPortableError) as excinfo:
        R.estimator_class("TabPFNRegressor")
    assert "not on the portable allowlist" in str(excinfo.value)
    assert excinfo.value.kind == "TabPFNRegressor"


def test_estimator_class_refuses_a_task_mismatch() -> None:
    """A classifier named in a regression recipe is refused before anything is fitted."""
    with pytest.raises(RecipeNotPortableError):
        R.estimator_class("XGBClassifier", task="regression")


def test_assert_portable_refuses_a_recipe_with_a_foreign_member(problem) -> None:
    """``assert_portable`` fails at the boundary, in milliseconds, not after a fit."""
    recipe = Recipe(
        task="regression",
        members=[RecipeMember(name="tabpfn", kind="TabPFNRegressor")],
        tier="tabpfn",
    )
    with pytest.raises(RecipeNotPortableError):
        R.assert_portable(recipe)


def test_assert_portable_accepts_the_baseline(problem) -> None:
    """The baseline recipe is portable by construction."""
    R.assert_portable(R.baseline_recipe(problem))


# ── JSON round-trip: the venv boundary itself ─────────────────────────────────


def test_recipe_round_trips_losslessly_through_json(problem) -> None:
    """``model_dump_json`` → ``model_validate_json`` returns an equal object.

    This IS the two-venv split: a recipe that does not survive the round-trip is a search
    that spent its whole time budget and accomplished nothing.
    """
    original = R.baseline_recipe(problem)
    restored = Recipe.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.model_dump() == original.model_dump()
    assert [m.kind for m in restored.members] == [m.kind for m in original.members]
    assert restored.categorical_features == original.categorical_features
    assert restored.numeric_features == original.numeric_features


def test_recipe_round_trips_through_a_file(problem, tmp_path) -> None:
    """``save_recipe``/``load_recipe`` is the on-disk form of the same guarantee."""
    original = R.baseline_recipe(problem)
    path = R.save_recipe(tmp_path / "nested" / "recipe.json", original)
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["tier"] == "baseline"
    assert R.load_recipe(path) == original


def test_numpy_scalars_are_coerced_before_serialisation() -> None:
    """FLAML and Optuna hand back numpy scalars; ``Recipe.model_dump_json`` raises on them."""
    import numpy as np

    params = {
        "n_estimators": np.int64(300),
        "learning_rate": np.float64(0.07),
        "flags": [np.bool_(True), np.int32(2)],
        "nested": {"depth": np.int64(5)},
    }
    clean = R.jsonable_params(params)
    assert json.loads(json.dumps(clean)) == {
        "n_estimators": 300,
        "learning_rate": pytest.approx(0.07),
        "flags": [True, 2],
        "nested": {"depth": 5},
    }
    recipe = Recipe(
        task="regression",
        members=[RecipeMember(name="xgboost", kind="XGBRegressor", params=clean)],
        tier="flaml",
    )
    assert Recipe.model_validate_json(recipe.model_dump_json()) == recipe


# ── to_aegis_members: the substitution point ──────────────────────────────────


def test_to_aegis_members_returns_name_estimator_pairs(problem) -> None:
    """The exact shape ``aegis.ml.model._regression_members`` returns."""
    from sklearn.base import BaseEstimator

    recipe = R.baseline_recipe(problem)
    members = R.to_aegis_members(recipe, random_state=7)

    assert isinstance(members, list)
    assert len(members) == len(recipe.members)
    for (name, estimator), declared in zip(members, recipe.members, strict=True):
        assert isinstance(name, str) and name == declared.name
        assert isinstance(estimator, BaseEstimator)
        assert type(estimator).__name__ == declared.kind
        assert not hasattr(estimator, "n_features_in_"), "members must come back UNFITTED"


def test_to_aegis_members_applies_the_random_state(problem) -> None:
    """Every member that accepts a seed gets one, so two runs of a recipe agree."""
    members = R.to_aegis_members(R.baseline_recipe(problem), random_state=123)
    for _name, estimator in members:
        assert estimator.get_params().get("random_state") == 123


def test_to_aegis_members_refuses_a_non_allowlisted_estimator() -> None:
    """A recipe naming something foreign is refused rather than imported."""
    recipe = Recipe(
        task="regression",
        members=[
            RecipeMember(name="ok", kind="XGBRegressor"),
            RecipeMember(name="bad", kind="CatBoostRegressor"),
        ],
        tier="autogluon",
    )
    with pytest.raises(RecipeNotPortableError) as excinfo:
        R.to_aegis_members(recipe, random_state=1)
    assert excinfo.value.kind == "CatBoostRegressor"


def test_build_estimator_wraps_members_in_the_spine_ensemble(problem) -> None:
    """Regression gets a ``VotingRegressor``; classification gets a SOFT ``VotingClassifier``."""
    from sklearn.ensemble import VotingClassifier, VotingRegressor

    regressor = R.build_estimator(R.baseline_recipe(problem), random_state=7)
    assert isinstance(regressor, VotingRegressor)

    classification = problem.model_copy(
        update={
            "target": problem.target.model_copy(
                update={"task": "classification", "levels": ["no", "yes"], "unit": None}
            )
        }
    )
    classifier = R.build_estimator(R.baseline_recipe(classification), random_state=7)
    assert isinstance(classifier, VotingClassifier)
    assert classifier.voting == "soft", "a hard vote gives MAPIE nothing continuous to threshold"


# ── fit_recipe: it genuinely fits ─────────────────────────────────────────────


def test_baseline_recipe_actually_trains_and_predicts(frame, problem) -> None:
    """``baseline_recipe`` produces something that fits real data and predicts sanely."""
    from sklearn.pipeline import Pipeline

    recipe = R.baseline_recipe(problem)
    pipeline = R.fit_recipe(recipe, frame, problem, random_state=7)

    assert isinstance(pipeline, Pipeline)
    assert [step for step, _ in pipeline.steps] == ["preprocess", "estimator"]

    predictions = pipeline.predict(frame[problem.feature_names])
    assert len(predictions) == len(frame)
    assert predictions.dtype.kind == "f"
    assert predictions.round(6).std() != 0.0, "a constant prediction is not a fitted model"


def test_fit_recipe_beats_predicting_the_mean(frame, problem) -> None:
    """The fitted point predictor genuinely learns: held-out R2 clears zero comfortably."""
    from sklearn.metrics import r2_score

    from aegis_ml.data.splits import stratified_split

    train, test = stratified_split(frame, problem, test_size=0.25, seed=7)
    pipeline = R.fit_recipe(R.baseline_recipe(problem), train, problem, random_state=7)
    score = r2_score(test[problem.target.name], pipeline.predict(test[problem.feature_names]))
    assert score > 0.3, f"held-out r2={score:.4f}: fit_recipe is not actually fitting"


def test_fit_recipe_refuses_a_frame_missing_a_declared_column(frame, problem) -> None:
    """A frame the contract should have refused fails here with the column named."""
    stripped = frame.drop(columns=[problem.feature_names[0]])
    with pytest.raises(KeyError) as excinfo:
        R.fit_recipe(R.baseline_recipe(problem), stripped, problem, random_state=7)
    assert problem.feature_names[0] in str(excinfo.value)


def test_baseline_recipe_reproduces_the_spine_members(problem) -> None:
    """The baseline is the honest floor: exactly the members ``aegis.ml.model`` builds."""
    recipe = R.baseline_recipe(problem)
    assert recipe.tier == "baseline"
    assert [m.kind for m in recipe.members] == [
        "XGBRegressor",
        "HistGradientBoostingRegressor",
    ]
    assert recipe.notes, "the baseline must say it is the baseline on the leaderboard"
    assert set(recipe.categorical_features) == {
        f.name for f in problem.features if f.dtype == "categorical"
    }
    assert set(recipe.numeric_features) | set(recipe.categorical_features) <= set(
        problem.feature_names
    )


def test_classification_baseline_pins_the_eval_metric(problem) -> None:
    """Without an explicit ``eval_metric``, two runs of "the same" recipe are incomparable."""
    classification = problem.model_copy(
        update={
            "target": problem.target.model_copy(
                update={"task": "classification", "levels": ["no", "yes"], "unit": None}
            )
        }
    )
    recipe = R.baseline_recipe(classification)
    xgb = next(m for m in recipe.members if m.kind == "XGBClassifier")
    assert xgb.params["eval_metric"] == "logloss"


def test_kind_for_maps_family_and_task_to_an_allowlisted_class() -> None:
    """The family→class mapping refuses an unknown family rather than guessing."""
    assert R.kind_for("xgboost", "regression") == "XGBRegressor"
    assert R.kind_for("xgboost", "classification") == "XGBClassifier"
    with pytest.raises(RecipeNotPortableError):
        R.kind_for("tabpfn", "regression")


def test_coerce_params_drops_what_the_estimator_cannot_accept() -> None:
    """A search-tuned parameter the serving class does not know is dropped and REPORTED."""
    params, dropped = R.coerce_params(
        "HistGradientBoostingRegressor", {"max_iter": 50, "not_a_real_hgb_param": 3}
    )
    assert params["max_iter"] == 50
    assert "not_a_real_hgb_param" not in params
    assert "not_a_real_hgb_param" in dropped, "a silently dropped parameter is a silent downgrade"
