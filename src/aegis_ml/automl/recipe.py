"""The portable recipe — the keystone that makes the two-venv split sound.

Decision D1 in ``finalplan.md`` splits the ML stack across two interpreters because
AutoGluon/TabPFN/torch will not resolve under the backend's ``pandas<2.4`` /
``numpy<2.5`` / ``numba==0.67.0`` caps. The search therefore runs *somewhere else*, and
the only thing that crosses back is JSON. This module is both ends of that bridge: it
decides what may cross (:data:`PORTABLE_KINDS`), and it rebuilds the crossing into live
estimators in the serving venv (:func:`to_aegis_members`, :func:`fit_recipe`).

**Why an explicit allowlist rather than dynamic import.** A recipe is data produced by
another process. ``getattr(importlib.import_module(mod), kind)`` on a class name that
arrived as data is an arbitrary-import primitive, and — the more immediate problem — it
succeeds for classes that cannot survive the crossing at all. AutoGluon's
``WeightedEnsembleModel`` imports fine *in the trainer venv* and is meaningless in the
serving venv. So a name that is not in the map is refused with
:class:`~aegis_ml.contracts.errors.RecipeNotPortableError` rather than imported, and its
score is reported as an accuracy ceiling instead.

**Why the output shape is a list of ``(name, estimator)`` tuples.** That is exactly what
``aegis.ml.model._regression_members()`` returns (``aegis/src/aegis/ml/model.py``). Matching
it means the Aegis spine can be handed an AutoML-tuned ensemble as a drop-in substitution
and keeps everything that makes it trustworthy: MAPIE split-conformal calibration on a
disjoint split, SHAP TreeExplainer attribution averaged by member weight, the SHA-256
dataset digest, and the ModelCard. Full AutoML benefit, zero changes inside ``aegis/``.

**Why every member must be a tree model.** The spine explains itself with
``shap.TreeExplainer``, which supports XGBoost, LightGBM and sklearn's forest/histogram
learners and nothing else. A linear or kNN member would fit and score perfectly well and
then make ``explain()`` raise at request time, which is the wrong place to discover it.
The allowlist is tree-only for that reason, not by accident.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml._require import is_available, require
from aegis_ml.contracts.errors import RecipeNotPortableError
from aegis_ml.contracts.protocols import Recipe, RecipeMember
from aegis_ml.contracts.spec import MLProblem

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import light
    import pandas as pd
    from sklearn.pipeline import Pipeline

__all__ = [
    "HGB_PARAMS",
    "KIND_EXTRAS",
    "PORTABLE_KINDS",
    "XGB_PARAMS",
    "assert_portable",
    "baseline_recipe",
    "build_estimator",
    "coerce_params",
    "estimator_class",
    "fit_recipe",
    "is_portable_kind",
    "jsonable_params",
    "kind_for",
    "load_recipe",
    "recipe_from_members",
    "save_recipe",
    "to_aegis_members",
]

PORTABLE_KINDS: dict[str, str] = {
    # xgboost — the Aegis spine's primary member; present in both venvs.
    "XGBRegressor": "xgboost",
    "XGBClassifier": "xgboost",
    # sklearn — the complementary histogram booster plus the two forest learners a
    # search commonly prefers on small, noisy, synthetic frames.
    "HistGradientBoostingRegressor": "sklearn.ensemble",
    "HistGradientBoostingClassifier": "sklearn.ensemble",
    "RandomForestRegressor": "sklearn.ensemble",
    "RandomForestClassifier": "sklearn.ensemble",
    "ExtraTreesRegressor": "sklearn.ensemble",
    "ExtraTreesClassifier": "sklearn.ensemble",
    # lightgbm — FLAML's and AutoGluon's favourite learner. Listed because it is
    # SHAP-TreeExplainer-compatible and pip-installable into the serving venv, but it is
    # NOT a [serve] dependency, so :func:`is_portable_kind` checks importability before a
    # search is allowed to select it.
    "LGBMRegressor": "lightgbm",
    "LGBMClassifier": "lightgbm",
}
"""Estimator class name → the module it may be imported from. The whole allowlist.

Every entry is a tree learner ``shap.TreeExplainer`` supports, because the Aegis spine
explains its ensemble member-by-member with exactly that explainer. Adding a non-tree
member here would produce a model that trains, scores, promotes — and then raises inside
``explain()`` on the first request that asks why.
"""

KIND_EXTRAS: dict[str, str] = {
    "xgboost": "aegis-ml[serve]",
    "sklearn.ensemble": "aegis-ml[serve]",
    "lightgbm": "aegis-ml[strong]",
}
"""Module → the install target that provides it, for the error message's remedy line."""

# ─────────────────────────────────────────────────────────────────────────────
# Mirrors of aegis/src/aegis/ml/model.py's _XGB_PARAMS / _HGB_PARAMS. Duplicated
# deliberately: `aegis` is NOT a dependency of this package (it is a sibling checkout the
# CLI writes artefacts into), so importing them would couple installation to a path.
# `baseline_recipe` therefore reproduces the spine's defaults exactly, which is what makes
# a "baseline" leaderboard row an honest floor: it is the model Aegis would have trained.
# ─────────────────────────────────────────────────────────────────────────────
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "n_jobs": 1,
    "tree_method": "hist",
}
"""The spine's XGBoost hyper-parameters — CPU-only, single-threaded, deterministic."""

HGB_PARAMS: dict[str, Any] = {
    "max_iter": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
}
"""The spine's HistGradientBoosting hyper-parameters — the complementary learner."""

_REGRESSION_SUFFIX = "Regressor"
_CLASSIFICATION_SUFFIX = "Classifier"


def kind_for(family: str, task: str) -> str:
    """Return the estimator class name for a learner family and task.

    Search tiers speak in families (``"xgboost"``, ``"random_forest"``); recipes speak in
    class names. Doing the mapping in one function keeps ``XGBRegressor`` from being
    spelled by hand in four modules, where the classification variant is the one that
    eventually gets forgotten.

    Args:
        family: One of ``xgboost``, ``hist_gbm``, ``random_forest``, ``extra_trees``,
            ``lightgbm``.
        task: ``"regression"`` or ``"classification"``.

    Returns:
        A class name that is a key of :data:`PORTABLE_KINDS`.

    Raises:
        RecipeNotPortableError: If the family has no portable estimator.
    """
    suffix = _CLASSIFICATION_SUFFIX if task == "classification" else _REGRESSION_SUFFIX
    stems = {
        "xgboost": "XGB",
        "hist_gbm": "HistGradientBoosting",
        "random_forest": "RandomForest",
        "extra_trees": "ExtraTrees",
        "lightgbm": "LGBM",
    }
    stem = stems.get(family)
    if stem is None:
        raise RecipeNotPortableError(family, f"no portable estimator for family {family!r}")
    return f"{stem}{suffix}"


def _task_of_kind(kind: str) -> str | None:
    """Return the task a kind is valid for, or ``None`` if the name says nothing."""
    if kind.endswith(_CLASSIFICATION_SUFFIX):
        return "classification"
    if kind.endswith(_REGRESSION_SUFFIX):
        return "regression"
    return None


def is_portable_kind(kind: str, *, task: str | None = None) -> bool:
    """Return whether ``kind`` may cross into this interpreter and be constructed here.

    Three conditions, all of them necessary:

    1. the name is on the allowlist (nothing else is ever imported);
    2. its module is importable *in this interpreter* — LightGBM is on the allowlist but
       is a ``[strong]`` extra, so a search running in the trainer venv must not select it
       as "portable" when the serving venv cannot construct it;
    3. it matches the task, so a ``XGBClassifier`` cannot end up in a regression recipe.

    Args:
        kind: Estimator class name from a search result.
        task: ``"regression"`` / ``"classification"`` to check against, or ``None`` to skip
            the task check.

    Returns:
        ``True`` if a recipe naming ``kind`` can be fitted here.
    """
    module = PORTABLE_KINDS.get(kind)
    if module is None or not is_available(module.split(".")[0]):
        return False
    return not (task is not None and _task_of_kind(kind) not in (None, task))


def estimator_class(kind: str, *, task: str | None = None) -> type:
    """Resolve an estimator class name from the allowlist, or refuse.

    Args:
        kind: Estimator class name carried by a :class:`~aegis_ml.contracts.protocols.RecipeMember`.
        task: Optional task to validate the class name against.

    Returns:
        The estimator class, ready to construct.

    Raises:
        RecipeNotPortableError: If the name is not on the allowlist, its module is not
            installed here, or it belongs to the other task.
    """
    module_path = PORTABLE_KINDS.get(kind)
    if module_path is None:
        raise RecipeNotPortableError(
            kind,
            f"not on the portable allowlist ({sorted(PORTABLE_KINDS)}). Class names that "
            f"arrive as data are never imported on trust",
        )
    if task is not None and _task_of_kind(kind) not in (None, task):
        raise RecipeNotPortableError(
            kind, f"is a {_task_of_kind(kind)} estimator but the recipe's task is {task!r}"
        )
    extra = KIND_EXTRAS.get(module_path, "aegis-ml[serve]")
    module = require(extra, module_path)
    cls = getattr(module, kind, None)
    if not inspect.isclass(cls):
        raise RecipeNotPortableError(
            kind, f"{module_path!r} is installed but exposes no class named {kind!r}"
        )
    return cls


def assert_portable(recipe: Recipe) -> None:
    """Raise unless every member of ``recipe`` can be constructed in this interpreter.

    Called before anything expensive (a full re-fit, a promotion) so a non-portable member
    fails in milliseconds at the boundary rather than after a training run.

    Args:
        recipe: The recipe to check.

    Raises:
        RecipeNotPortableError: On the first member that cannot cross.
    """
    for member in recipe.members:
        estimator_class(member.kind, task=recipe.task)


def jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Coerce hyper-parameters to JSON-native scalars, recursively.

    Not cosmetic. FLAML and Optuna hand back ``numpy.int64`` / ``numpy.float64`` values,
    and ``Recipe.model_dump_json()`` raises on them. The whole two-venv split is a JSON
    round-trip, so a recipe that cannot serialise is a search that silently accomplished
    nothing — after having spent the time budget.

    Args:
        params: Raw constructor kwargs from a search result.

    Returns:
        The same mapping with numpy scalars/arrays reduced to Python builtins. Anything
        that still is not JSON-native is stringified, which keeps the value visible in the
        recipe rather than dropping it.
    """
    return {str(key): _jsonable(value) for key, value in params.items()}


def _jsonable(value: Any) -> Any:  # noqa: ANN401 - deliberately accepts anything
    """Reduce one value to a JSON-native form, preserving it as a string if it cannot be."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)  # numpy scalar
    if callable(item) and getattr(value, "ndim", None) == 0:
        return _jsonable(item())
    tolist = getattr(value, "tolist", None)  # numpy array
    if callable(tolist):
        return _jsonable(tolist())
    return str(value)


def coerce_params(kind: str, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Filter search-supplied kwargs down to what ``kind``'s constructor actually accepts.

    A cross-venv recipe is the one place where hyper-parameters arrive from a *different
    library* than the one that will consume them: FLAML's LightGBM config carries
    ``log_max_bin``, AutoGluon's carries ``ag_args_fit``. Passing those to the sklearn
    estimator raises ``TypeError`` deep inside a re-fit.

    Dropped keys are *returned*, never discarded quietly — the caller writes them into
    ``Recipe.notes`` so the reader can see that the re-fitted model is not parameterised
    identically to the one the search scored.

    Args:
        kind: Estimator class name (must be on the allowlist).
        params: Candidate constructor kwargs.

    Returns:
        ``(kept, dropped)`` — the accepted kwargs and the names that were removed.
    """
    cls = estimator_class(kind)
    accepted = _accepted_param_names(cls)
    if accepted is None:  # constructor takes **kwargs (xgboost, lightgbm)
        return jsonable_params(params), []
    kept = {k: v for k, v in params.items() if k in accepted}
    dropped = sorted(set(params) - set(kept))
    return jsonable_params(kept), dropped


def _accepted_param_names(cls: type) -> set[str] | None:
    """Return a constructor's parameter names, or ``None`` when it accepts ``**kwargs``.

    ``None`` is a meaningful answer, not a failure: XGBoost and LightGBM forward unknown
    kwargs to their native boosters on purpose, and filtering them against the Python
    signature would silently strip legitimate booster settings.
    """
    signature = inspect.signature(cls.__init__)
    names: set[str] = set()
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.kind is not inspect.Parameter.VAR_POSITIONAL:
            names.add(name)
    return names


def _build_one(member: RecipeMember, task: str, random_state: int) -> Any:  # noqa: ANN401
    """Construct one fitted-ready estimator from a recipe member."""
    cls = estimator_class(member.kind, task=task)
    params, _dropped = coerce_params(member.kind, dict(member.params))
    accepted = _accepted_param_names(cls)
    if "random_state" not in params and (accepted is None or "random_state" in accepted):
        params["random_state"] = random_state
    if member.kind == "XGBClassifier" and "eval_metric" not in params:
        # Mirrors aegis.ml.model._classification_members: without an explicit eval_metric
        # XGBoost picks one per-version, which makes two runs of "the same" recipe
        # non-comparable across an xgboost upgrade.
        params["eval_metric"] = "logloss"
    return cls(**params)


def to_aegis_members(recipe: Recipe, *, random_state: int) -> list[tuple[str, Any]]:
    """Return the recipe as ``[(name, estimator), ...]`` — the Aegis spine's own shape.

    This is the substitution point. ``aegis.ml.model._regression_members(random_state)``
    returns a list of exactly this shape and ``TrustworthyModel`` wraps it in a
    ``VotingRegressor`` / soft ``VotingClassifier`` before conformalising and explaining
    it. Anything returned here can therefore be dropped in without touching ``aegis/``.

    Args:
        recipe: A portable recipe, typically from :func:`~aegis_ml.automl.search.search`.
        random_state: Seed applied to every member that accepts one, so two runs of the
            same recipe produce the same model.

    Returns:
        ``(name, unfitted estimator)`` pairs in recipe order.

    Raises:
        RecipeNotPortableError: If any member names a non-allowlisted estimator.
    """
    return [
        (member.name, _build_one(member, recipe.task, random_state)) for member in recipe.members
    ]


def build_estimator(recipe: Recipe, *, random_state: int) -> Any:  # noqa: ANN401
    """Wrap the recipe's members in the voting ensemble the spine uses.

    Soft voting for classification (probabilities averaged, not labels) because the
    conformal *prediction set* needs calibrated class scores; a hard vote gives MAPIE
    nothing continuous to threshold and the resulting set is degenerate.

    Args:
        recipe: The recipe to build.
        random_state: Seed for every member that accepts one.

    Returns:
        An unfitted ``VotingRegressor`` or soft ``VotingClassifier``.
    """
    ensemble = require("aegis-ml[serve]", "sklearn.ensemble")
    members = to_aegis_members(recipe, random_state=random_state)
    weights = [member.weight for member in recipe.members]
    # Equal weights are expressed as None so sklearn's own default path is used, keeping
    # the fitted object identical to the spine's when the recipe did not tune weights.
    use_weights = None if len({round(w, 9) for w in weights}) == 1 else weights
    if recipe.task == "classification":
        return ensemble.VotingClassifier(
            estimators=members, voting="soft", weights=use_weights, n_jobs=1
        )
    return ensemble.VotingRegressor(estimators=members, weights=use_weights, n_jobs=1)


def fit_recipe(
    recipe: Recipe,
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    random_state: int,
) -> Pipeline:
    """Fit ``recipe`` on ``frame`` and return the fitted preprocess + ensemble pipeline.

    The preprocessing comes from :func:`aegis_ml.features.pipeline.column_transformer`,
    which reproduces the spine's ``ColumnTransformer`` (one-hot the declared categoricals
    with ``handle_unknown="ignore"``, pass numerics through). It is imported *here* rather
    than at module scope so that importing this module — which the light contracts layer
    and the CLI both do — never drags sklearn in.

    Note what this function deliberately does **not** do: conformal calibration and SHAP.
    Those belong to ``aegis.ml.model.TrustworthyModel``, which must fit them on splits it
    controls. A pipeline returned from here is a *point predictor*; promoting it as a
    model without going through the spine would lose the calibrated interval, which is the
    entire trust story.

    Args:
        recipe: The portable recipe to fit.
        frame: Training frame containing every feature column and the target column.
        problem: The spec the frame conforms to; supplies the column split.
        random_state: Seed for every member that accepts one.

    Returns:
        A fitted ``sklearn.pipeline.Pipeline`` with steps ``("preprocess", "estimator")``.

    Raises:
        RecipeNotPortableError: If a member cannot be constructed here.
        KeyError: If ``frame`` is missing a declared feature or the target column.
    """
    pipeline_mod = require("aegis-ml[serve]", "sklearn.pipeline")
    features_mod = require("aegis-ml[serve]", "aegis_ml.features.pipeline")

    missing = [c for c in [*problem.feature_names, problem.target.name] if c not in frame.columns]
    if missing:
        raise KeyError(
            f"training frame is missing declared columns {missing}; the pandera contract "
            f"(aegis_ml.contracts.frames) should have refused this frame at the boundary"
        )

    x = frame[problem.feature_names]
    y = frame[problem.target.name]
    pipeline = pipeline_mod.Pipeline(
        steps=[
            ("preprocess", features_mod.column_transformer(problem)),
            ("estimator", build_estimator(recipe, random_state=random_state)),
        ]
    )
    pipeline.fit(x, y)
    return pipeline


def recipe_from_members(
    members: list[RecipeMember],
    problem: MLProblem,
    *,
    tier: str = "baseline",
    search_seconds: float = 0.0,
    notes: list[str] | None = None,
) -> Recipe:
    """Assemble a :class:`Recipe` around ``members``, filling the columns from ``problem``.

    Centralised because the feature lists on a recipe are not decoration: they are what
    lets a *stored* recipe be re-fitted months later against a frame whose column order
    has drifted. Deriving them from the problem in one place keeps a search tier from
    inventing its own ordering.

    Args:
        members: The ensemble members, already coerced and JSON-safe.
        problem: The spec the recipe was searched against.
        tier: Which tier produced it.
        search_seconds: Wall-clock the tier spent.
        notes: Anything a reader must know about this recipe.

    Returns:
        A fully-populated recipe.
    """
    return Recipe(
        task=problem.target.task,
        members=members,
        categorical_features=list(problem.categorical_features),
        numeric_features=list(problem.numeric_features),
        tier=tier,  # type: ignore[arg-type]
        search_seconds=max(0.0, float(search_seconds)),
        notes=list(notes or []),
    )


def baseline_recipe(problem: MLProblem) -> Recipe:
    """Return the Aegis-default recipe: XGBoost + HistGradientBoosting, soft-voted.

    This exists so that "fall back to the baseline" is always an *explicit, named* act
    with a leaderboard row attached, rather than an ``except: pass`` that leaves the
    reader unable to distinguish "AutoML found nothing better" from "AutoML never ran".
    It reproduces ``aegis.ml.model``'s members and hyper-parameters exactly, so its score
    is the honest floor the other tiers must beat.

    Args:
        problem: The spec to build the recipe for.

    Returns:
        A two-member recipe on tier ``baseline``.
    """
    task = problem.target.task
    if task == "classification":
        members = [
            RecipeMember(
                name="xgboost",
                kind="XGBClassifier",
                params={**XGB_PARAMS, "eval_metric": "logloss"},
            ),
            RecipeMember(name="hist_gbc", kind="HistGradientBoostingClassifier", params=HGB_PARAMS),
        ]
    else:
        members = [
            RecipeMember(name="xgboost", kind="XGBRegressor", params=dict(XGB_PARAMS)),
            RecipeMember(name="hist_gbr", kind="HistGradientBoostingRegressor", params=HGB_PARAMS),
        ]
    return recipe_from_members(
        members,
        problem,
        tier="baseline",
        notes=["Aegis spine defaults (aegis.ml.model); the floor every other tier must beat."],
    )


def save_recipe(path: str | Path, recipe: Recipe) -> Path:
    """Write ``recipe`` to ``path`` as indented JSON, creating parent directories.

    Args:
        path: Destination file, typically ``registry/runs/<run_id>/recipe.json``.
        recipe: The recipe to persist.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(recipe.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_recipe(path: str | Path) -> Recipe:
    """Read a recipe back from JSON, validating it against the contract.

    Validation on load is the point: a recipe file is the one artefact that crossed a
    process boundary, and reading it with :func:`json.load` into a dict would defer every
    shape error to the moment the estimator is constructed.

    Args:
        path: The ``recipe.json`` to read.

    Returns:
        The validated recipe.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the JSON does not satisfy the contract.
    """
    text = Path(path).read_text(encoding="utf-8")
    return Recipe.model_validate(json.loads(text))
