"""Optuna refinement of an already-chosen recipe — TPE + Hyperband, resumable on disk.

**Why this is a separate stage from the search.** :mod:`aegis_ml.automl.search` answers
"which family of model", over four tiers, on one holdout. That question is answered well
by fitting a handful of sensible configurations. "Which hyper-parameters" is a different
question with a different budget shape — hundreds of cheap trials over a continuous space —
and TPE only starts beating random search from roughly the 30th trial. Folding the two
together would spend the search's wall-clock on the wrong question.

**Why cross-validation here and a single holdout there.** The search compares four
*structurally different* tiers, where a 0.05 difference is real. HPO compares near-identical
configurations, where a 0.005 difference on a 25% holdout is noise the optimiser will
happily chase for 60 trials and hand back an overfitted configuration that scores worse in
production. K-fold costs k× the fits and is the only thing that makes the ranking mean
anything at this resolution. It also gives Hyberband something to prune on: each fold is
reported as a step, so a hopeless configuration dies after one fold instead of three.

**Why SQLite storage.** ``settings.registry_dir / "optuna"`` holds one database per domain,
and the study name is derived from domain + target + metric. Re-running ``aegis-ml tune``
therefore *continues* the study rather than restarting it — 60 trials on Tuesday plus 60
on Wednesday is a 120-trial study, which is exactly what you want the day before a demo.
The name must encode the metric because a study optimising R² and one optimising RMSE
cannot share trials: the direction is opposite.

**Why tuning can return the recipe it was given.** If no trial beats the incoming recipe on
the same cross-validation, this returns the original, with a note saying so. An HPO stage
that always hands back "the best trial" quietly ships a regression whenever the search's
choice was already good — and it looks identical to a successful tune.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.automl.recipe import (
    build_estimator,
    coerce_params,
    jsonable_params,
    recipe_from_members,
)
from aegis_ml.automl.search import METRIC_HIGHER_IS_BETTER, score_predictions
from aegis_ml.contracts.protocols import Recipe, RecipeMember
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import light
    import pandas as pd

__all__ = ["CV_FOLDS", "storage_url", "study_name_for", "tune"]

CV_FOLDS: int = 3
"""Folds per trial.

Three, not five: each fold is a full ensemble fit, and the budget is
``settings.hpo_trials`` (60) × folds. Five folds would buy a slightly tighter estimate at
the cost of 40% fewer configurations explored, and at this sample size the extra
configurations are worth more than the extra precision.
"""

_TUNABLE_KINDS = frozenset(
    {
        "XGBRegressor",
        "XGBClassifier",
        "HistGradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "RandomForestRegressor",
        "RandomForestClassifier",
        "ExtraTreesRegressor",
        "ExtraTreesClassifier",
        "LGBMRegressor",
        "LGBMClassifier",
    }
)
"""Kinds this module knows a search space for. A member outside it keeps its parameters
verbatim and the fact is recorded on the returned recipe — a silently untuned member would
otherwise look tuned."""


def study_name_for(problem: MLProblem) -> str:
    """Return the study name for a problem: domain, target and metric.

    All three are load-bearing. Two domains must not share trials; two targets inside one
    domain must not either; and a study optimising ``r2`` cannot be resumed as one
    optimising ``rmse`` because the direction is inverted and every stored trial would be
    read backwards.

    Args:
        problem: The spec being tuned against.

    Returns:
        A stable study name, safe to reuse across runs.
    """
    return f"{problem.domain_id}::{problem.target.name}::{problem.metric}"


def storage_url() -> str:
    """Return the SQLite URL for the shared Optuna store, creating its directory.

    Returns:
        A ``sqlite:///`` URL under ``settings.registry_dir / "optuna"``.
    """
    directory = settings.registry_dir / "optuna"
    directory.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{directory / 'studies.db'}"


def _suggest(trial: Any, member: RecipeMember) -> dict[str, Any]:  # noqa: ANN401
    """Suggest one member's hyper-parameters, define-by-run.

    Parameter names are prefixed with the member name because a two-member ensemble
    suggests ``learning_rate`` twice, and an unprefixed second suggestion would silently
    overwrite the first inside the trial — tying the two members' learning rates together
    without anybody asking for it.

    Args:
        trial: The live Optuna trial.
        member: The member whose space is being sampled.

    Returns:
        Constructor kwargs for this member in this trial. Unknown kinds return their
        existing parameters unchanged.
    """
    kind = member.kind
    prefix = f"{member.name}__"

    def name(param: str) -> str:
        return f"{prefix}{param}"

    if kind.startswith("XGB"):
        return {
            **member.params,
            "n_estimators": trial.suggest_int(name("n_estimators"), 100, 1200, step=50),
            "max_depth": trial.suggest_int(name("max_depth"), 2, 10),
            "learning_rate": trial.suggest_float(name("learning_rate"), 0.01, 0.3, log=True),
            "subsample": trial.suggest_float(name("subsample"), 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(name("colsample_bytree"), 0.5, 1.0),
            "min_child_weight": trial.suggest_int(name("min_child_weight"), 1, 20),
        }
    if kind.startswith("HistGradientBoosting"):
        return {
            **member.params,
            "max_iter": trial.suggest_int(name("max_iter"), 100, 1000, step=50),
            "max_depth": trial.suggest_int(name("max_depth"), 2, 12),
            "learning_rate": trial.suggest_float(name("learning_rate"), 0.01, 0.3, log=True),
            "l2_regularization": trial.suggest_float(
                name("l2_regularization"), 1e-8, 10.0, log=True
            ),
        }
    if kind.startswith(("RandomForest", "ExtraTrees")):
        return {
            **member.params,
            "n_estimators": trial.suggest_int(name("n_estimators"), 100, 800, step=50),
            "max_depth": trial.suggest_int(name("max_depth"), 3, 24),
            "min_samples_leaf": trial.suggest_int(name("min_samples_leaf"), 1, 16),
            "max_features": trial.suggest_categorical(
                name("max_features"), ["sqrt", "log2", 0.5, 1.0]
            ),
        }
    if kind.startswith("LGBM"):
        return {
            **member.params,
            "n_estimators": trial.suggest_int(name("n_estimators"), 100, 1200, step=50),
            "num_leaves": trial.suggest_int(name("num_leaves"), 8, 128, log=True),
            "learning_rate": trial.suggest_float(name("learning_rate"), 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int(name("min_child_samples"), 5, 60),
            "colsample_bytree": trial.suggest_float(name("colsample_bytree"), 0.5, 1.0),
        }
    return dict(member.params)


def _members_from_trial(trial: Any, recipe: Recipe) -> list[RecipeMember]:  # noqa: ANN401
    """Build the trial's candidate members, coercing each to its constructor's signature."""
    members: list[RecipeMember] = []
    for member in recipe.members:
        params = _suggest(trial, member)
        kept, _dropped = coerce_params(member.kind, params)
        members.append(
            RecipeMember(name=member.name, kind=member.kind, params=kept, weight=member.weight)
        )
    return members


def _cross_val_score(
    members: list[RecipeMember],
    recipe: Recipe,
    problem: MLProblem,
    frame: pd.DataFrame,
    *,
    folds: list[tuple[Any, Any]],
    seed: int,
    trial: Any = None,  # noqa: ANN401
) -> float:
    """Fit and score one configuration across the folds, reporting each to the pruner.

    The preprocessing is re-fitted inside every fold rather than once outside. That costs
    a few milliseconds per fold and removes the only leak available at this stage: an
    encoder fitted on the validation rows makes every configuration look slightly better,
    and it makes the configurations that exploit the leak look best of all.

    Args:
        members: The candidate members for this trial.
        recipe: The recipe being tuned (supplies task and column lists).
        problem: The spec.
        frame: The tuning frame.
        folds: Pre-computed ``(train_index, validation_index)`` pairs.
        seed: Estimator seed.
        trial: The live trial, for intermediate reporting and pruning; or ``None`` when
            scoring the incoming recipe outside the study.

    Returns:
        The mean fold score, in the metric's natural direction.

    Raises:
        optuna.TrialPruned: When the pruner stops this configuration early.
    """
    optuna = require("aegis-ml[serve]", "optuna")
    pipeline_mod = require("aegis-ml[serve]", "sklearn.pipeline")
    features_mod = require("aegis-ml[serve]", "aegis_ml.features.pipeline")

    draft = recipe_from_members(members, problem, tier=recipe.tier)
    x = frame[problem.feature_names]
    y = frame[problem.target.name]
    scores: list[float] = []

    for step, (train_index, valid_index) in enumerate(folds):
        model = pipeline_mod.Pipeline(
            steps=[
                ("preprocess", features_mod.column_transformer(problem)),
                ("estimator", build_estimator(draft, random_state=seed)),
            ]
        )
        model.fit(x.iloc[train_index], y.iloc[train_index])
        x_valid = x.iloc[valid_index]
        y_pred = model.predict(x_valid)
        y_proba = None
        if problem.target.task == "classification" and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_valid)
        labels = sorted(y.dropna().unique().tolist(), key=str) if y_proba is not None else None
        scores.append(
            score_predictions(
                problem.metric, y.iloc[valid_index], y_pred, y_proba=y_proba, labels=labels
            )
        )
        if trial is not None:
            running = sum(scores) / len(scores)
            trial.report(running, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return sum(scores) / len(scores)


def tune(
    frame: pd.DataFrame,
    problem: MLProblem,
    recipe: Recipe,
    *,
    n_trials: int | None = None,
    timeout: int | None = None,
    seed: int | None = None,
) -> Recipe:
    """Refine ``recipe``'s hyper-parameters with an Optuna study, and return a new recipe.

    The returned object is always a *new* :class:`~aegis_ml.contracts.protocols.Recipe` —
    never a mutation of the input — because the caller usually wants to keep the untuned
    one on the leaderboard for comparison, and an in-place tune destroys that evidence.

    Args:
        frame: Tuning data. Use the training portion, not the promotion gate's holdout:
            60 trials against the gate's split is 60 chances to overfit the number that
            decides promotion.
        problem: The spec; supplies columns, task and the metric being optimised.
        recipe: The recipe to refine, normally the winner from
            :func:`aegis_ml.automl.search.search`.
        n_trials: Trials this invocation adds to the (possibly resumed) study. Defaults to
            ``settings.hpo_trials``.
        timeout: Wall-clock seconds for this invocation. Defaults to
            ``settings.hpo_timeout``. Whichever limit is reached first stops the study.
        seed: Sampler and estimator seed; defaults to ``settings.random_seed``.

    Returns:
        A recipe with tuned parameters, or the input recipe unchanged when no trial beat
        it — in both cases with the outcome written into ``notes``.

    Raises:
        ImportError: If Optuna is not installed (message names the install command).
        RecipeNotPortableError: If a member of ``recipe`` cannot be constructed here.
    """
    optuna = require("aegis-ml[serve]", "optuna")
    selection = require("aegis-ml[serve]", "sklearn.model_selection")

    n_trials = settings.hpo_trials if n_trials is None else n_trials
    timeout = settings.hpo_timeout if timeout is None else timeout
    seed = settings.random_seed if seed is None else seed
    metric = problem.metric
    if metric not in METRIC_HIGHER_IS_BETTER:
        raise ValueError(
            f"cannot tune on {metric!r}: Optuna needs a direction and "
            f"{sorted(METRIC_HIGHER_IS_BETTER)} are the metrics whose direction is known."
        )
    higher_is_better = METRIC_HIGHER_IS_BETTER[metric]

    usable = frame.loc[frame[problem.target.name].notna()]
    y = usable[problem.target.name]
    if problem.target.task == "classification" and y.value_counts().min() >= CV_FOLDS:
        splitter = selection.StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        folds = list(splitter.split(usable, y))
    else:
        splitter = selection.KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        folds = list(splitter.split(usable))

    # Optuna's per-trial INFO log is one line per trial times 60; at that volume it buries
    # the search's own output. Warnings and errors still print.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name_for(problem),
        storage=storage_url(),
        direction="maximize" if higher_is_better else "minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=CV_FOLDS, reduction_factor=3
        ),
        load_if_exists=True,
    )

    started = time.perf_counter()
    incoming_score = _cross_val_score(
        list(recipe.members), recipe, problem, usable, folds=folds, seed=seed
    )

    untunable = sorted({m.kind for m in recipe.members if m.kind not in _TUNABLE_KINDS})

    def objective(trial: Any) -> float:  # noqa: ANN401 - optuna's Trial, imported lazily
        """Score one sampled configuration by cross-validation."""
        members = _members_from_trial(trial, recipe)
        return _cross_val_score(
            members, recipe, problem, usable, folds=folds, seed=seed, trial=trial
        )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    elapsed = time.perf_counter() - started

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    provenance = (
        f"Optuna study {study.study_name!r} at {storage_url()} — {len(completed)} complete "
        f"trials ({len(study.trials)} total, resumable), TPE + Hyperband, {CV_FOLDS}-fold CV "
        f"on {len(usable)} rows, {elapsed:.1f}s this invocation."
    )
    baseline_note = f"Untuned recipe scored {metric}={incoming_score:.4f} on the same folds."

    if not completed:
        tuned = recipe.model_copy(deep=True)
        tuned.notes.extend(
            [
                provenance,
                baseline_note,
                "NO TRIAL COMPLETED (every trial was pruned, failed, or the budget expired "
                "first), so the parameters are UNCHANGED from the search's choice.",
            ]
        )
        return tuned

    best = study.best_trial
    improved = best.value > incoming_score if higher_is_better else best.value < incoming_score
    if not improved:
        tuned = recipe.model_copy(deep=True)
        tuned.notes.extend(
            [
                provenance,
                baseline_note,
                f"Best trial scored {metric}={best.value:.4f}, which does NOT beat the "
                f"untuned recipe, so the search's parameters are kept unchanged. Tuning "
                f"that always returns 'the best trial' ships a regression whenever the "
                f"starting point was already good.",
            ]
        )
        return tuned

    members = [
        RecipeMember(
            name=member.name,
            kind=member.kind,
            params=jsonable_params(coerce_params(member.kind, _params_from_trial(best, member))[0]),
            weight=member.weight,
        )
        for member in recipe.members
    ]
    tuned = recipe_from_members(
        members,
        problem,
        tier=recipe.tier,
        search_seconds=recipe.search_seconds + elapsed,
        notes=[
            *recipe.notes,
            provenance,
            baseline_note,
            f"Tuned recipe scored {metric}={best.value:.4f} (trial #{best.number}), an "
            f"improvement of {abs(best.value - incoming_score):.4f}.",
        ],
    )
    if untunable:
        tuned.notes.append(
            f"Members {untunable} have no search space defined in aegis_ml.automl.hpo and "
            f"kept their parameters verbatim — they are UNTUNED, not optimally tuned."
        )
    return tuned


def _params_from_trial(trial: Any, member: RecipeMember) -> dict[str, Any]:  # noqa: ANN401
    """Recover one member's parameters from a finished trial's flat parameter dict.

    Optuna stores a trial's parameters flat and prefixed (``xgboost__max_depth``). Undoing
    the prefix here — rather than re-running :func:`_suggest` against a frozen trial — is
    what lets a *resumed* study's best trial be read back after the process that created it
    has exited.
    """
    prefix = f"{member.name}__"
    recovered = {
        key[len(prefix) :]: value for key, value in trial.params.items() if key.startswith(prefix)
    }
    return {**member.params, **recovered}
