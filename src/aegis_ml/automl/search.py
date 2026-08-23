"""Run every available tier, score every candidate on one held-out split, keep the losers.

**Why the losers stay.** ``aegis.forecast.ForecastResult.candidates`` reports SeasonalNaive's
score even when AutoARIMA beats it, and that is the pattern this module copies. A
leaderboard showing only the winner cannot answer the question an enterprise reviewer
actually asks — *was the extra machinery worth it?* A boosted ensemble that beats a ridge
regression by 0.30 R² and one that beats it by 0.004 look identical once the losers are
thrown away, and only one of them justifies its complexity.

**Why one split, computed once.** Every tier scores on the same held-out rows, encoded by
the same fitted preprocessor. Tiers that score themselves on their own internal splits
(FLAML's validation loss, AutoGluon's leaderboard ``score_val``) produce numbers that are
not comparable to each other — different splits, different metrics, sometimes different
directions. So their internal scores are never copied onto the leaderboard: every
candidate here is re-scored by :func:`score_predictions` on the *same* holdout, and the
tier's own opinion, where it is useful, is recorded in ``Candidate.detail``.

**Why the preprocessor is fitted on the training split only.** ``aegis.ml.model`` fits its
``ColumnTransformer`` on the whole frame before splitting, which leaks the holdout's
categorical levels into the encoder. That is harmless for the spine (the encoder is
unsupervised and the levels are declared in the spec anyway) but it is not harmless here,
because these numbers are used to *choose* between models. A search that leaks is a search
that prefers whichever model exploits the leak.

**Why failures become reasons rather than exceptions.** A tier that raises mid-search —
AutoGluon running out of disk, TabPFN refusing a frame that is too wide — must not destroy
the other tiers' results, and must not disappear either. Every failure lands in
``Leaderboard.tiers_skipped`` carrying its exception type and message. If *nothing*
produced a candidate, the search raises: an empty leaderboard is never returned as a
result.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.automl import tiers as tiers_mod
from aegis_ml.automl.recipe import (
    baseline_recipe,
    build_estimator,
    coerce_params,
    is_portable_kind,
    jsonable_params,
    kind_for,
    recipe_from_members,
)
from aegis_ml.contracts.errors import AegisMLError, InsufficientLabelsError
from aegis_ml.contracts.protocols import Candidate, Leaderboard, Recipe, RecipeMember, TierName
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import light
    import numpy as np
    import pandas as pd

__all__ = [
    "HOLDOUT_FRACTION",
    "METRIC_HIGHER_IS_BETTER",
    "MIN_SEARCH_ROWS",
    "score_predictions",
    "search",
]

HOLDOUT_FRACTION: float = 0.25
"""Share of rows withheld from every tier and used for all reported scores.

A quarter is a compromise the frame size forces: this factory generates 1k–10k rows, so
0.1 leaves a holdout small enough that a 0.02 R² difference is inside the noise, and 0.4
starves AutoGluon's own internal validation split.
"""

MIN_SEARCH_ROWS: int = 60
"""Below this the holdout cannot separate candidates and the search refuses to guess.

With 60 rows the holdout is 15. That is already thin; below it, ranking four tiers on the
resulting score is numerology, and picking a "winner" would launder noise into a model
card that reads as evidence.
"""

METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "r2": True,
    "mae": False,
    "rmse": False,
    "mape": False,
    "accuracy": True,
    "balanced_accuracy": True,
    "f1": True,
    "f1_macro": True,
    "roc_auc": True,
    "log_loss": False,
}
"""Metric name → whether a larger number is better.

Kept as data because the direction has to be carried into ``Leaderboard.higher_is_better``,
into Optuna's ``direction``, and into the promotion gate's comparison. Three places
inferring it independently is three chances for one of them to rank backwards, which
produces a leaderboard that looks fine and is upside down.
"""

_PROBA_METRICS = frozenset({"roc_auc", "log_loss"})
"""Metrics that need class probabilities, not labels."""

_TABPFN_MAX_ROWS = 10_000
"""Training rows handed to TabPFN before subsampling.

TabPFN-2.5 handles 50k rows, but its inference cost grows with the context it carries and
a hackathon box is 16 GB. Subsampling above this is recorded on the candidate, because a
score from 10k of 40k rows is not a score from 40k rows.
"""

_TABPFN_MAX_FEATURES = 2_000
"""Encoded columns TabPFN-2.5 accepts. Beyond it the tier is skipped with a reason."""


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def score_predictions(
    metric: str,
    y_true: Any,  # noqa: ANN401 - a pandas Series or 1-d array; typed loosely to stay light
    y_pred: Any,  # noqa: ANN401
    *,
    y_proba: Any = None,  # noqa: ANN401
    labels: list[Any] | None = None,
) -> float:
    """Score one candidate's holdout predictions with the problem's primary metric.

    Every tier's candidate goes through this one function so that the leaderboard's column
    means the same thing on every row. Tiers report their own internal scores in wildly
    different conventions — FLAML minimises a loss, AutoGluon negates its error metrics —
    and mixing those into one ranked list silently ranks some candidates backwards.

    Args:
        metric: A key of :data:`METRIC_HIGHER_IS_BETTER`.
        y_true: Ground truth for the holdout rows.
        y_pred: Predicted labels or values.
        y_proba: Class probabilities, required by ``roc_auc`` and ``log_loss``.
        labels: Class labels in the column order of ``y_proba``.

    Returns:
        The metric value, in its natural direction (see :data:`METRIC_HIGHER_IS_BETTER`).

    Raises:
        ValueError: If the metric is unknown, or needs probabilities that were not given.
    """
    if metric not in METRIC_HIGHER_IS_BETTER:
        raise ValueError(
            f"unknown metric {metric!r}; supported: {sorted(METRIC_HIGHER_IS_BETTER)}. "
            f"MLProblem.primary_metric must name one of these so the leaderboard, the "
            f"Optuna direction and the promotion gate agree on which way is better."
        )
    if metric in _PROBA_METRICS and y_proba is None:
        raise ValueError(
            f"metric {metric!r} scores probabilities, but this candidate exposes no "
            f"predict_proba. Choose 'accuracy' or 'f1_macro', or drop the estimator."
        )
    metrics = require("aegis-ml[serve]", "sklearn.metrics")
    np_mod = require("aegis-ml[serve]", "numpy")

    if metric == "r2":
        return float(metrics.r2_score(y_true, y_pred))
    if metric == "mae":
        return float(metrics.mean_absolute_error(y_true, y_pred))
    if metric == "rmse":
        return float(np_mod.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    if metric == "mape":
        return float(metrics.mean_absolute_percentage_error(y_true, y_pred))
    if metric == "accuracy":
        return float(metrics.accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(metrics.balanced_accuracy_score(y_true, y_pred))
    if metric == "f1":
        positive = _pos_label(y_true)
        return float(metrics.f1_score(y_true, y_pred, average="binary", pos_label=positive))
    if metric == "f1_macro":
        return float(metrics.f1_score(y_true, y_pred, average="macro"))
    if metric == "log_loss":
        return float(metrics.log_loss(y_true, y_proba, labels=labels))
    # roc_auc — binary takes the positive column, multiclass takes the full matrix.
    proba = np_mod.asarray(y_proba)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return float(metrics.roc_auc_score(y_true, proba[:, 1]))
    return float(metrics.roc_auc_score(y_true, proba, multi_class="ovr", labels=labels))


def _pos_label(y_true: Any) -> Any:  # noqa: ANN401
    """Return the positive class for binary ``f1``: the rarer label, deterministically.

    sklearn defaults ``pos_label=1``, which raises on string labels ("excursion"/"nominal")
    — the shape this factory's classification targets actually take. Choosing the rarer
    class is also the right default for the imbalanced targets these domains generate,
    where F1 on the majority class is uninformative.
    """
    pd_mod = require("aegis-ml[serve]", "pandas")
    counts = pd_mod.Series(y_true).value_counts()
    return counts.index[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Search context
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class _Scored:
    """One leaderboard row plus, when portable, the recipe that reproduces it."""

    candidate: Candidate
    recipe: Recipe | None = None


@dataclass(slots=True)
class _Context:
    """Everything a tier needs, computed once so every tier is scored identically."""

    problem: MLProblem
    metric: str
    higher_is_better: bool
    seed: int
    time_budget: int
    train_frame: pd.DataFrame
    holdout_frame: pd.DataFrame
    x_train: np.ndarray
    x_holdout: np.ndarray
    y_train: pd.Series
    y_holdout: pd.Series
    encoded_names: list[str]
    labels: list[Any] = field(default_factory=list)

    @property
    def task(self) -> str:
        """The supervised task, read off the problem's target spec."""
        return self.problem.target.task

    def score(self, y_pred: Any, y_proba: Any = None) -> float:  # noqa: ANN401
        """Score holdout predictions with this search's single metric."""
        return score_predictions(
            self.metric,
            self.y_holdout,
            y_pred,
            y_proba=y_proba,
            labels=self.labels or None,
        )


def _resolve_metric(problem: MLProblem) -> tuple[str, bool]:
    """Return ``(metric, higher_is_better)``, refusing a metric that cannot be ranked."""
    metric = problem.metric
    if metric not in METRIC_HIGHER_IS_BETTER:
        raise ValueError(
            f"MLProblem.primary_metric={metric!r} is not one of "
            f"{sorted(METRIC_HIGHER_IS_BETTER)}. The leaderboard cannot rank a metric "
            f"whose direction it does not know, and guessing 'higher is better' would "
            f"rank an error metric upside down."
        )
    return metric, METRIC_HIGHER_IS_BETTER[metric]


def _build_context(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    time_budget: int,
    seed: int,
) -> _Context:
    """Split, encode and package the data every tier will see.

    Raises:
        InsufficientLabelsError: If the frame is too small for the holdout to mean
            anything (see :data:`MIN_SEARCH_ROWS`).
        KeyError: If a declared column is absent from the frame.
    """
    pd_mod = require("aegis-ml[serve]", "pandas")
    selection = require("aegis-ml[serve]", "sklearn.model_selection")
    features_mod = require("aegis-ml[serve]", "aegis_ml.features.pipeline")

    columns = [*problem.feature_names, problem.target.name]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(
            f"frame is missing declared columns {missing}; validate it against "
            f"aegis_ml.contracts.frames before searching, not after"
        )

    usable = frame.loc[frame[problem.target.name].notna(), columns]
    if len(usable) < MIN_SEARCH_ROWS:
        raise InsufficientLabelsError(len(usable), MIN_SEARCH_ROWS, "AutoML search")

    metric, higher = _resolve_metric(problem)
    y_all = usable[problem.target.name]
    stratify = None
    if problem.target.task == "classification" and y_all.value_counts().min() >= 2:
        stratify = y_all
    train_frame, holdout_frame = selection.train_test_split(
        usable,
        test_size=HOLDOUT_FRACTION,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    preprocessor = features_mod.column_transformer(problem)
    x_train = preprocessor.fit_transform(train_frame[problem.feature_names])
    x_holdout = preprocessor.transform(holdout_frame[problem.feature_names])
    encoded_names = [str(n) for n in preprocessor.get_feature_names_out()]

    # Estimators are fitted on plain arrays, never on a named frame. One-hot column names
    # derived from domain levels routinely contain '[', ']' or '<', which XGBoost rejects
    # outright with a feature-name error — a failure that would look like "XGBoost cannot
    # handle this domain" rather than "the encoder named a column badly".
    np_mod = require("aegis-ml[serve]", "numpy")
    x_train = np_mod.asarray(x_train, dtype=float)
    x_holdout = np_mod.asarray(x_holdout, dtype=float)

    labels: list[Any] = []
    if problem.target.task == "classification":
        labels = sorted(pd_mod.Series(y_all).dropna().unique().tolist(), key=str)

    return _Context(
        problem=problem,
        metric=metric,
        higher_is_better=higher,
        seed=seed,
        time_budget=time_budget,
        train_frame=train_frame,
        holdout_frame=holdout_frame,
        x_train=x_train,
        x_holdout=x_holdout,
        y_train=train_frame[problem.target.name],
        y_holdout=holdout_frame[problem.target.name],
        encoded_names=encoded_names,
        labels=labels,
    )


def _fit_and_score(
    ctx: _Context,
    estimator: Any,  # noqa: ANN401 - any fitted-capable sklearn-style estimator
) -> tuple[float, float]:
    """Fit ``estimator`` on the training split and score it on the holdout.

    Returns:
        ``(metric_value, fit_seconds)``.
    """
    started = time.perf_counter()
    estimator.fit(ctx.x_train, ctx.y_train)
    fit_seconds = time.perf_counter() - started
    y_pred = estimator.predict(ctx.x_holdout)
    y_proba = None
    proba_fn = getattr(estimator, "predict_proba", None)
    if ctx.task == "classification" and callable(proba_fn):
        y_proba = proba_fn(ctx.x_holdout)
    return ctx.score(y_pred, y_proba), fit_seconds


def _portable_families(task: str) -> list[str]:
    """Return the learner families that are BOTH allowlisted and importable here.

    Availability is checked per family rather than assumed, because the two venvs differ:
    the trainer venv has LightGBM, the backend venv may not, and a search that marks a
    LightGBM candidate portable in the trainer venv hands back a recipe the serving venv
    cannot construct — which is precisely the failure the recipe allowlist exists to stop.
    """
    families = ["xgboost", "hist_gbm", "random_forest", "extra_trees", "lightgbm"]
    return [f for f in families if is_portable_kind(kind_for(f, task), task=task)]


def _member(family: str, task: str, params: dict[str, Any], name: str | None = None) -> RecipeMember:
    """Build one recipe member, coercing params to what the class actually accepts."""
    kind = kind_for(family, task)
    kept, _dropped = coerce_params(kind, params)
    return RecipeMember(name=name or family, kind=kind, params=kept)


def _estimator_for(members: list[RecipeMember], ctx: _Context) -> Any:  # noqa: ANN401
    """Construct the unfitted voting ensemble a list of members describes."""
    draft = recipe_from_members(members, ctx.problem, tier="baseline")
    return build_estimator(draft, random_state=ctx.seed)


# ─────────────────────────────────────────────────────────────────────────────
# Tier: baseline
# ─────────────────────────────────────────────────────────────────────────────
def _baseline_configs(ctx: _Context) -> list[tuple[str, list[RecipeMember]]]:
    """Return the named portable configurations the baseline tier fits.

    Three shapes, each answering a different question: the spine's own pair (what Aegis
    would have trained unaided), each family alone (which learner is carrying the
    ensemble), and every available family together (does more averaging help). All are
    cheap — the whole tier is seconds — and together they give the leaderboard a real
    spread rather than one row.
    """
    task = ctx.task
    available = _portable_families(task)
    hgb = {"max_iter": 200, "max_depth": 4, "learning_rate": 0.1}
    xgb = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    forest = {"n_estimators": 300, "n_jobs": 1}
    defaults: dict[str, dict[str, Any]] = {
        "xgboost": xgb,
        "hist_gbm": hgb,
        "random_forest": forest,
        "extra_trees": forest,
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "n_jobs": 1, "verbose": -1},
    }

    configs: list[tuple[str, list[RecipeMember]]] = []
    spine = [f for f in ("xgboost", "hist_gbm") if f in available]
    if spine:
        configs.append(
            ("aegis_spine", [_member(f, task, defaults[f]) for f in spine]),
        )
    for family in available:
        configs.append((family, [_member(family, task, defaults[family])]))
    if len(available) > 2:
        configs.append(
            ("wide_vote", [_member(f, task, defaults[f]) for f in available]),
        )
    return configs


def _linear_reference(ctx: _Context) -> tuple[Any, str]:  # noqa: ANN401
    """Return the linear reference model and its name.

    It exists to give the leaderboard a floor with a *shape* — how much of this target is
    linear in the encoded features — and it is deliberately marked non-portable. Not
    because the serving venv cannot fit a ridge, but because the Aegis spine explains its
    ensemble with ``shap.TreeExplainer``, which supports tree models only. Promoting a
    linear member would produce a model that trains and scores and then raises inside
    ``explain()`` on the first request that asks why.

    Imputation is explicit here: the frames this searches over carry MAR missingness, and
    unlike the tree learners a linear model has no native NaN path.
    """
    impute = require("aegis-ml[serve]", "sklearn.impute")
    pipeline = require("aegis-ml[serve]", "sklearn.pipeline")
    preprocessing = require("aegis-ml[serve]", "sklearn.preprocessing")
    linear = require("aegis-ml[serve]", "sklearn.linear_model")

    steps: list[tuple[str, Any]] = [
        ("impute", impute.SimpleImputer(strategy="median")),
        ("scale", preprocessing.StandardScaler()),
    ]
    if ctx.task == "classification":
        steps.append(("model", linear.LogisticRegression(max_iter=2000, n_jobs=1)))
        return pipeline.Pipeline(steps), "logistic_reference"
    steps.append(("model", linear.RidgeCV()))
    return pipeline.Pipeline(steps), "ridge_reference"


def _search_baseline(ctx: _Context) -> tuple[list[_Scored], dict[str, str]]:
    """Fit the always-available sklearn/xgboost configurations.

    This tier is the reason a search can always return *something* portable. It is also
    the only tier whose candidates come with recipes by construction: every configuration
    it fits was built out of allowlisted members in the first place, so the leaderboard
    entry and the recipe that reproduces it cannot drift apart.
    """
    scored: list[_Scored] = []
    failures: dict[str, str] = {}

    for name, members in _baseline_configs(ctx):
        try:
            value, fit_seconds = _fit_and_score(ctx, _estimator_for(members, ctx))
        except Exception as exc:  # audit-ok: one config failing must not lose the others
            failures[f"baseline:{name}"] = f"{type(exc).__name__}: {exc}"
            continue
        recipe = recipe_from_members(
            members,
            ctx.problem,
            tier="baseline",
            search_seconds=fit_seconds,
            notes=[f"baseline configuration {name!r}, fitted on {len(ctx.y_train)} rows"],
        )
        scored.append(
            _Scored(
                candidate=Candidate(
                    name=name,
                    tier="baseline",
                    metric_name=ctx.metric,
                    metric_value=value,
                    fit_seconds=fit_seconds,
                    portable=True,
                    detail={"members": [m.kind for m in members]},
                ),
                recipe=recipe,
            )
        )

    model, name = _linear_reference(ctx)
    try:
        value, fit_seconds = _fit_and_score(ctx, model)
    except Exception as exc:  # audit-ok: the reference is diagnostic, never the winner
        failures[f"baseline:{name}"] = f"{type(exc).__name__}: {exc}"
    else:
        scored.append(
            _Scored(
                candidate=Candidate(
                    name=name,
                    tier="baseline",
                    metric_name=ctx.metric,
                    metric_value=value,
                    fit_seconds=fit_seconds,
                    portable=False,
                    detail={
                        "role": "reference floor — how much of this target is linear",
                        "excluded_from_recipes": (
                            "shap.TreeExplainer, which the Aegis spine explains with, "
                            "supports tree models only"
                        ),
                    },
                ),
                recipe=None,
            )
        )

    if not scored and not failures:
        failures["baseline"] = (
            "no allowlisted estimator is importable here; install aegis-ml[serve] "
            "(scikit-learn) — the baseline tier is what guarantees a portable recipe"
        )
    return scored, failures


# ─────────────────────────────────────────────────────────────────────────────
# Tier: flaml
# ─────────────────────────────────────────────────────────────────────────────
_FLAML_TO_FAMILY: dict[str, str] = {
    "xgboost": "xgboost",
    "xgb_limitdepth": "xgboost",
    "lgbm": "lightgbm",
    "rf": "random_forest",
    "extra_tree": "extra_trees",
}
"""FLAML estimator id → portable learner family.

FLAML also searches ``catboost``, ``lrl1`` and ``kneighbor``. They are absent from this map
on purpose: none is a SHAP-TreeExplainer-compatible member of the Aegis spine, so a
recipe naming one could not be explained after promotion. Their scores still reach the
leaderboard — as non-portable rows.
"""

_FLAML_METRIC: dict[str, str] = {
    "r2": "r2",
    "mae": "mae",
    "rmse": "rmse",
    "mape": "mape",
    "accuracy": "accuracy",
    "f1": "f1",
    "f1_macro": "macro_f1",
    "roc_auc": "roc_auc",
    "log_loss": "log_loss",
    "balanced_accuracy": "accuracy",
}
"""Our metric → FLAML's spelling of it, used only to steer FLAML's own search.

``balanced_accuracy`` has no FLAML equivalent, so FLAML optimises plain accuracy while the
leaderboard still *reports* balanced accuracy on the holdout. That substitution is written
onto the candidate's detail, because a tier optimising a different objective than the one
it is judged on is exactly the kind of mismatch that makes a tier look unfairly weak.
"""


def _search_flaml(ctx: _Context) -> tuple[list[_Scored], dict[str, str]]:
    """Run FLAML's cost-frugal search under a wall-clock budget.

    FLAML's own answer is a fitted model with a validation loss. Neither is used directly:
    the model is re-scored on the shared holdout, and each per-estimator best config is
    *rebuilt as an allowlisted estimator and re-fitted*, which is the only way to know that
    the configuration survives the crossing into the serving venv. A config that FLAML
    liked but that cannot be reconstructed portably is reported as a non-portable row
    rather than promoted.
    """
    automl_mod = require("aegis-ml[serve]", "flaml.automl")
    scored: list[_Scored] = []
    failures: dict[str, str] = {}
    task = ctx.task
    flaml_metric = _FLAML_METRIC.get(ctx.metric, "r2" if task == "regression" else "accuracy")

    estimator_list = sorted(
        {
            flaml_name
            for flaml_name, family in _FLAML_TO_FAMILY.items()
            if is_portable_kind(kind_for(family, task), task=task)
        }
    )
    if not estimator_list:
        return [], {"flaml": "no portable learner is importable for FLAML to search over"}

    automl = automl_mod.AutoML()
    started = time.perf_counter()
    automl.fit(
        X_train=ctx.x_train,
        y_train=ctx.y_train.to_numpy(),
        task=task,
        metric=flaml_metric,
        time_budget=ctx.time_budget,
        estimator_list=estimator_list,
        seed=ctx.seed,
        n_jobs=1,
        verbose=0,
        early_stop=True,
    )
    search_seconds = time.perf_counter() - started

    detail_note = (
        f"FLAML optimised {flaml_metric!r}; the leaderboard reports {ctx.metric!r} on the "
        f"shared holdout"
    )
    per_estimator: dict[str, Any] = dict(getattr(automl, "best_config_per_estimator", {}) or {})
    for flaml_name, config in per_estimator.items():
        if not isinstance(config, dict):
            continue
        family = _FLAML_TO_FAMILY.get(flaml_name)
        if family is None:
            failures[f"flaml:{flaml_name}"] = (
                f"FLAML estimator {flaml_name!r} has no SHAP-explainable portable "
                f"equivalent, so its configuration is not re-fitted here"
            )
            continue
        try:
            member = _member(family, task, jsonable_params(config), name=flaml_name)
            value, fit_seconds = _fit_and_score(ctx, _estimator_for([member], ctx))
        except Exception as exc:  # audit-ok: one config failing must not lose the tier
            failures[f"flaml:{flaml_name}"] = f"{type(exc).__name__}: {exc}"
            continue
        scored.append(
            _Scored(
                candidate=Candidate(
                    name=f"flaml_{flaml_name}",
                    tier="flaml",
                    metric_name=ctx.metric,
                    metric_value=value,
                    fit_seconds=fit_seconds,
                    portable=True,
                    detail={"config": jsonable_params(config), "note": detail_note},
                ),
                recipe=recipe_from_members(
                    [member],
                    ctx.problem,
                    tier="flaml",
                    search_seconds=search_seconds,
                    notes=[detail_note],
                ),
            )
        )

    best_name = str(getattr(automl, "best_estimator", "") or "")
    if best_name and f"flaml_{best_name}" not in {s.candidate.name for s in scored}:
        try:
            y_pred = automl.predict(ctx.x_holdout)
            y_proba = automl.predict_proba(ctx.x_holdout) if task == "classification" else None
            value = ctx.score(y_pred, y_proba)
        except Exception as exc:  # audit-ok: recorded, never replaced with a number
            failures["flaml:best_model"] = f"{type(exc).__name__}: {exc}"
        else:
            scored.append(
                _Scored(
                    candidate=Candidate(
                        name=f"flaml_best_{best_name}",
                        tier="flaml",
                        metric_name=ctx.metric,
                        metric_value=value,
                        fit_seconds=search_seconds,
                        portable=False,
                        detail={
                            "reason_not_portable": (
                                f"FLAML's winning estimator {best_name!r} is not on the "
                                f"portable allowlist"
                            ),
                            "note": detail_note,
                        },
                    )
                )
            )
    return scored, failures


# ─────────────────────────────────────────────────────────────────────────────
# Tier: autogluon
# ─────────────────────────────────────────────────────────────────────────────
_AUTOGLUON_METRIC: dict[str, str] = {
    "r2": "r2",
    "mae": "mean_absolute_error",
    "rmse": "root_mean_squared_error",
    "mape": "mean_absolute_percentage_error",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "log_loss": "log_loss",
}
"""Our metric → AutoGluon's ``eval_metric`` name."""

_AUTOGLUON_TOP_MODELS = 6
"""How many of AutoGluon's fitted models get their own leaderboard row.

Its ``best_quality`` preset fits dozens. All of them on the leaderboard would bury the
other three tiers under one tier's internal detail; none of them would hide the fact that
the stack's advantage comes from bagging many learners, which is the honest story.
"""


def _search_autogluon(ctx: _Context) -> tuple[list[_Scored], dict[str, str]]:
    """Fit AutoGluon's stacked ensemble and re-score its top models on the shared holdout.

    Every candidate here is ``portable=False``. That is not a limitation of the allowlist
    but a property of the models: a ``best_quality`` predictor is a multi-layer stack of
    bagged learners whose out-of-fold structure has no representation as a list of
    constructor kwargs. Its score is the *accuracy ceiling* — what the problem admits when
    portability is not required — and the honest way to publish it is next to the recipe
    that was actually promoted, which is what the leaderboard does.
    """
    tabular = require("aegis-ml[strong]", "autogluon.tabular")
    scored: list[_Scored] = []
    failures: dict[str, str] = {}

    target = ctx.problem.target.name
    if ctx.task == "classification":
        problem_type = "binary" if len(ctx.labels) == 2 else "multiclass"
    else:
        problem_type = "regression"
    eval_metric = _AUTOGLUON_METRIC.get(ctx.metric)

    with tempfile.TemporaryDirectory(prefix="aegis_ml_autogluon_") as workdir:
        predictor = tabular.TabularPredictor(
            label=target,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=workdir,
            verbosity=1,
        )
        started = time.perf_counter()
        predictor.fit(
            train_data=ctx.train_frame,
            presets="best_quality",
            time_limit=ctx.time_budget,
        )
        search_seconds = time.perf_counter() - started

        holdout_x = ctx.holdout_frame[ctx.problem.feature_names]
        model_names = list(predictor.model_names())[:_AUTOGLUON_TOP_MODELS]
        for model_name in model_names:
            try:
                y_pred = predictor.predict(holdout_x, model=model_name)
                y_proba = None
                if ctx.task == "classification":
                    proba_frame = predictor.predict_proba(holdout_x, model=model_name)
                    y_proba = proba_frame[ctx.labels].to_numpy() if ctx.labels else None
                value = ctx.score(y_pred, y_proba)
            except Exception as exc:  # audit-ok: one AG model failing is recorded, not faked
                failures[f"autogluon:{model_name}"] = f"{type(exc).__name__}: {exc}"
                continue
            scored.append(
                _Scored(
                    candidate=Candidate(
                        name=f"autogluon_{model_name}",
                        tier="autogluon",
                        metric_name=ctx.metric,
                        metric_value=value,
                        fit_seconds=search_seconds,
                        portable=False,
                        detail={
                            "preset": "best_quality",
                            "reason_not_portable": (
                                "AutoGluon models are bagged/stacked wrappers; their "
                                "out-of-fold structure has no constructor-kwargs form the "
                                "serving venv could rebuild"
                            ),
                        },
                    )
                )
            )
    if not scored and not failures:
        failures["autogluon"] = "AutoGluon fitted no model within the time budget"
    return scored, failures


# ─────────────────────────────────────────────────────────────────────────────
# Tier: tabpfn
# ─────────────────────────────────────────────────────────────────────────────
def _search_tabpfn(ctx: _Context) -> tuple[list[_Scored], dict[str, str]]:
    """Fit TabPFN-2.5 (and AutoTabPFN when available) and score on the shared holdout.

    TabPFN is never portable, and the reason is worth stating precisely: the model *is* a
    pretrained transformer that performs in-context learning over the training rows. There
    are no hyper-parameters to carry across the venv boundary — carrying the model would
    mean carrying the weights, which the licence does not permit for production use
    anyway. Its number is an evaluation-only accuracy ceiling.

    Every candidate produced here carries :data:`~aegis_ml.automl.tiers.TABPFN_LICENSE_NOTICE`
    in its detail, and :func:`search` copies it into ``Recipe.notes`` whenever this tier
    contributed to the run.
    """
    tabpfn = require("aegis-ml[strong]", "tabpfn")
    np_mod = require("aegis-ml[serve]", "numpy")
    scored: list[_Scored] = []
    failures: dict[str, str] = {}

    n_features = ctx.x_train.shape[1]
    if n_features > _TABPFN_MAX_FEATURES:
        return [], {
            "tabpfn": (
                f"encoded width is {n_features} columns, above TabPFN-2.5's "
                f"{_TABPFN_MAX_FEATURES}-feature ceiling; reduce the categorical "
                f"cardinality or drop the tier"
            )
        }

    x_train, y_train = ctx.x_train, ctx.y_train.to_numpy()
    subsampled = False
    if len(x_train) > _TABPFN_MAX_ROWS:
        rng = np_mod.random.default_rng(ctx.seed)
        index = rng.choice(len(x_train), size=_TABPFN_MAX_ROWS, replace=False)
        x_train, y_train = x_train[index], y_train[index]
        subsampled = True

    detail_base: dict[str, Any] = {
        "license": tiers_mod.TABPFN_LICENSE_NOTICE,
        "reason_not_portable": (
            "the model is a pretrained in-context transformer; there is no set of "
            "constructor kwargs the serving venv could rebuild it from"
        ),
        "train_rows": int(len(x_train)),
    }
    if subsampled:
        detail_base["subsampled"] = (
            f"training rows subsampled to {_TABPFN_MAX_ROWS} of {len(ctx.x_train)}; this "
            f"score is not a score on the full training split"
        )

    variants: list[tuple[str, Any]] = []
    if ctx.task == "classification":
        variants.append(("tabpfn", tabpfn.TabPFNClassifier()))
    else:
        variants.append(("tabpfn", tabpfn.TabPFNRegressor()))

    if tiers_mod.has_autotabpfn():
        phe = require("aegis-ml[strong]", "tabpfn_extensions.post_hoc_ensembles.sklearn_interface")
        auto_cls = (
            phe.AutoTabPFNClassifier if ctx.task == "classification" else phe.AutoTabPFNRegressor
        )
        variants.append(("auto_tabpfn", auto_cls(max_time=ctx.time_budget)))
    else:
        failures["tabpfn:auto_tabpfn"] = (
            "tabpfn_extensions is not installed, so the post-hoc ensemble (AutoTabPFN) "
            "did not run; the plain TabPFN score below is NOT an AutoTabPFN score. "
            "Install with `uv pip install 'aegis-ml[strong]'`"
        )

    for name, model in variants:
        try:
            started = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - started
            y_pred = model.predict(ctx.x_holdout)
            y_proba = None
            if ctx.task == "classification" and hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(ctx.x_holdout)
            value = ctx.score(y_pred, y_proba)
        except Exception as exc:  # audit-ok: recorded with its message, never a number
            failures[f"tabpfn:{name}"] = f"{type(exc).__name__}: {exc}"
            continue
        scored.append(
            _Scored(
                candidate=Candidate(
                    name=name,
                    tier="tabpfn",
                    metric_name=ctx.metric,
                    metric_value=value,
                    fit_seconds=fit_seconds,
                    portable=False,
                    detail=dict(detail_base),
                )
            )
        )
    return scored, failures


_TIER_FUNCTIONS = {
    "baseline": _search_baseline,
    "flaml": _search_flaml,
    "autogluon": _search_autogluon,
    "tabpfn": _search_tabpfn,
}
"""Tier → its private implementation. Dispatch is a table so that adding a tier is one
entry plus one function, and so that :func:`search`'s loop has no tier-specific branch."""


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────
def _rank(scored: list[_Scored], higher_is_better: bool) -> list[_Scored]:
    """Rank candidates best-first, breaking ties towards the cheaper tier.

    The tie-break matters more than it looks: on a small holdout two configurations
    genuinely tie often, and preferring the earlier tier means preferring the simpler,
    faster, portable one. Preferring whichever happened to be appended first would make
    the selection depend on dict ordering.
    """
    sign = -1.0 if higher_is_better else 1.0
    return sorted(
        scored,
        key=lambda s: (
            sign * s.candidate.metric_value,
            tiers_mod.TIER_ORDER.index(s.candidate.tier),
            s.candidate.fit_seconds,
            s.candidate.name,
        ),
    )


def _ceiling_note(best: Candidate, chosen: Candidate, metric: str) -> str:
    """Describe, with both numbers, how much accuracy portability cost."""
    gap = best.metric_value - chosen.metric_value
    return (
        f"ACCURACY CEILING: {best.name!r} (tier {best.tier}) scored {metric}="
        f"{best.metric_value:.4f} but cannot be re-fitted in the serving venv, so it was "
        f"NOT promoted. The promoted recipe {chosen.name!r} (tier {chosen.tier}) scored "
        f"{metric}={chosen.metric_value:.4f} — a gap of {gap:+.4f} on the held-out split. "
        f"Report the ceiling as evidence of headroom, never as this model's performance."
    )


def search(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    tiers: list[TierName] | tuple[TierName, ...] | None = None,
    time_budget: int | None = None,
    seed: int | None = None,
) -> tuple[Recipe, Leaderboard]:
    """Search every available tier and return the best portable recipe plus the full board.

    The two return values answer two different questions and neither substitutes for the
    other. The :class:`~aegis_ml.contracts.protocols.Recipe` is what gets fitted and
    promoted; the :class:`~aegis_ml.contracts.protocols.Leaderboard` is what gets published
    in the model card, including the candidates that lost and the tiers that never ran.

    Args:
        frame: Training data containing every declared feature and the target. Rows with a
            null target are dropped — they are unusable for a supervised score, and
            dropping them here is visible in the candidate row counts.
        problem: The spec; supplies the column split, the task and the primary metric.
        tiers: Tiers to attempt, or ``None`` for all four. Unavailable ones are skipped
            *with a reason*, never dropped silently.
        time_budget: Wall-clock seconds granted to each *budgeted* tier (FLAML,
            AutoGluon) — per tier, not shared, because both take a ``time_limit`` and
            splitting one budget across them would make each tier's result depend on which
            other tiers happened to be installed. Defaults to
            ``settings.automl_time_budget``.
        seed: Split and estimator seed; defaults to ``settings.random_seed``.

    Returns:
        ``(recipe, leaderboard)``. The recipe is always portable and always fittable in
        this interpreter; its ``notes`` carry the accuracy ceiling when a non-portable
        candidate won, and the TabPFN licence notice when that tier contributed.

    Raises:
        InsufficientLabelsError: If the frame is too small to score a holdout on.
        AegisMLError: If every tier failed — an empty leaderboard is never returned as a
            result, because "nothing beat the baseline" and "nothing ran" must not look
            alike.
        ValueError: If ``problem.primary_metric`` is not a rankable metric.
    """
    seed = settings.random_seed if seed is None else seed
    time_budget = settings.automl_time_budget if time_budget is None else time_budget
    started = time.perf_counter()

    ctx = _build_context(frame, problem, time_budget=time_budget, seed=seed)
    to_run, skipped = tiers_mod.resolve_tiers(tiers)

    all_scored: list[_Scored] = []
    tiers_run: list[TierName] = []
    for tier in to_run:
        try:
            tier_scored, tier_failures = _TIER_FUNCTIONS[tier](ctx)
        except Exception as exc:  # audit-ok: a tier's failure is reported, never invented
            skipped[tier] = (
                f"failed after starting: {type(exc).__name__}: {exc}. The other tiers' "
                f"results below are unaffected."
            )
            continue
        skipped.update(tier_failures)
        if tier_scored:
            tiers_run.append(tier)
            all_scored.extend(tier_scored)
        else:
            skipped.setdefault(tier, "ran but produced no scored candidate")

    if not all_scored:
        raise AegisMLError(
            "AutoML search produced no scored candidate on any tier. Reasons collected:\n  - "
            + "\n  - ".join(f"{k}: {v}" for k, v in sorted(skipped.items()))
        )

    ranked = _rank(all_scored, ctx.higher_is_better)
    best = ranked[0]
    portable = next((s for s in ranked if s.candidate.portable and s.recipe is not None), None)

    notes: list[str] = [
        f"Selected on {ctx.metric} over {len(ctx.y_holdout)} held-out rows "
        f"({HOLDOUT_FRACTION:.0%} of {len(ctx.y_train) + len(ctx.y_holdout)}), seed={seed}.",
    ]
    if portable is None:
        chosen = baseline_recipe(problem)
        chosen.notes.append(
            "NO PORTABLE CANDIDATE was produced by the requested tiers, so this is the "
            "Aegis spine's default recipe, UNFITTED BY ANY SEARCH. Its leaderboard score "
            "is absent because it was never scored here — run the baseline tier to get one."
        )
        selected_candidate: Candidate | None = None
    else:
        chosen = portable.recipe  # type: ignore[assignment]
        selected_candidate = portable.candidate
        portable.candidate.selected = True

    if selected_candidate is not None and best.candidate is not selected_candidate:
        notes.append(_ceiling_note(best.candidate, selected_candidate, ctx.metric))

    if any(s.candidate.tier == "tabpfn" for s in all_scored):
        notes.extend(tiers_mod.tier_notes("tabpfn"))

    chosen.notes.extend(notes)
    chosen.search_seconds = time.perf_counter() - started

    leaderboard = Leaderboard(
        metric_name=ctx.metric,
        higher_is_better=ctx.higher_is_better,
        candidates=[s.candidate for s in ranked],
        tiers_run=tiers_run,
        tiers_skipped=skipped,
    )
    return chosen, leaderboard
