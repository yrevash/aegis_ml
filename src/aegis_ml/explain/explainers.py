"""Pick the right SHAP algorithm for the model in hand, rather than assuming one.

``shap`` is not one algorithm. ``TreeExplainer`` computes exact Shapley values by walking a
fitted ensemble's split structure; ``LinearExplainer`` computes them in closed form from
``coef_`` and a reference distribution; ``PermutationExplainer`` estimates them from model
evaluations alone and therefore works on anything callable. They agree on what they compute
and differ enormously in what they can be pointed at and what they cost.

**Why a dispatch rather than one explainer.** Both single-explainer answers are wrong in a
way that costs something real:

* *TreeExplainer for everything* is what this repository used to do, and it decided which
  models were allowed to win. A ridge regression scored best in a measured run and was refused
  promotion because the explainer could not explain it — a tooling limit deciding a
  modelling question, backwards. See :func:`~aegis_ml.automl.search._linear_reference`.
* *``shap.Explainer`` for everything* looks like the fix and is not. Handed a tree model
  **and background data**, ``shap.Explainer`` selects the interventional tree path, whose
  additivity check then fails against the model's own output — verified against shap 0.51
  with ``HistGradientBoostingRegressor``. Trees must be given no background at all, which
  a single generic call has no way to know.

So the dispatch is explicit, and each branch is chosen for a reason that belongs to the
model family:

``tree``
    ``shap.TreeExplainer(model)`` with **no background**. Exact, milliseconds, and the
    reference distribution is the tree's own node cover — supplying data instead switches
    the algorithm and breaks additivity. This is byte-identical to what the Aegis spine has
    always done for its boosters, so nothing about tree explanations changes here.
``linear``
    ``shap.LinearExplainer(model, background)``. Exact and closed-form, but a linear
    attribution is ``coef_j · (x_j − E[x_j])``: the expectation is part of the answer, so a
    background is required and there is no default that would not silently change every
    number.
``permutation``
    ``shap.PermutationExplainer(f, background)`` over the model's own ``predict`` /
    ``predict_proba``. Model-agnostic, so kNN, an SVM or a stacked meta-learner are
    explainable rather than unpromotable; it costs model evaluations instead of structure,
    which is the honest price of not knowing anything about the estimator.

**Pipelines.** A linear member on data with missing values must carry its own imputation
(scikit-learn's linear models have no NaN path), so it arrives here wrapped in a
``Pipeline``. When the prefix preserves the column count — an imputer, a scaler — the final
estimator is explained directly and the attributions still line up one-to-one with the
input columns, because an affine per-column transform leaves ``coef_j · (x_j − E[x_j])``
unchanged. When the prefix reshapes the columns, unwrapping would return attributions in a
column space the caller never handed in, so the whole pipeline is explained by permutation
instead. That check is made against real transformed data, never guessed from the class
name.

``shap`` is an optional dependency, imported through :func:`aegis_ml._require.require` so an
absent install raises with the exact command that fixes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import light
    from collections.abc import Sequence

__all__ = [
    "LINEAR_CLASS_PREFIXES",
    "TREE_CLASS_PREFIXES",
    "ExplainerKind",
    "ExplainerUnavailableError",
    "build_explainer",
    "model_family",
    "shap_values",
]

ExplainerKind = Literal["tree", "linear", "permutation"]
"""Which SHAP algorithm a model was routed to. Returned so callers can record it."""

TREE_CLASS_PREFIXES: tuple[str, ...] = (
    "XGB",
    "LGBM",
    "CatBoost",
    "HistGradientBoosting",
    "GradientBoosting",
    "RandomForest",
    "ExtraTree",
    "DecisionTree",
)
"""Class-name prefixes routed to ``shap.TreeExplainer``.

Matched on the class name rather than on ``isinstance``, because the families that matter
here live in three different libraries (xgboost, lightgbm, scikit-learn) and importing all
of them to run an ``isinstance`` check would make this module drag in every optional
dependency just to answer a question about one model.

``ExtraTree`` covers both ``ExtraTreesRegressor`` (the forest) and ``ExtraTreeRegressor``
(the single tree). The list errs towards being short: a tree family that is missing from it
falls through to the permutation branch and is explained correctly but slowly, whereas a
non-tree that matched by accident would raise inside ``TreeExplainer``.
"""

LINEAR_CLASS_PREFIXES: tuple[str, ...] = (
    "Ridge",
    "LinearRegression",
    "LogisticRegression",
    "ElasticNet",
    "Lasso",
    "Lars",
    "SGD",
    "BayesianRidge",
    "HuberRegressor",
    "PassiveAggressive",
    "Perceptron",
    "TheilSen",
)
"""Class-name prefixes routed to ``shap.LinearExplainer``.

A structural check on ``coef_``/``intercept_`` backs this up in :func:`model_family`, so a
linear estimator whose name is not listed is still recognised. The prefixes exist because
the structural check alone would also match models that merely *expose* coefficients
without being additive in them.
"""


class ExplainerUnavailableError(AegisMLError):
    """The model cannot be explained in the form the caller requires.

    Raised instead of falling back to a cruder attribution (feature permutation on the
    metric, say, or coefficient magnitudes). Those answer a different question, and a report
    that silently swapped one for the other would attribute a prediction to the wrong cause
    while looking identical.
    """

    def __init__(self, reason: str) -> None:
        """Name what was missing and what the caller can do about it."""
        super().__init__(
            f"Cannot compute SHAP attributions: {reason}. Nothing here substitutes a "
            f"different notion of importance on your behalf — permutation importance and "
            f"SHAP answer different questions, and a report that quietly swapped them would "
            f"attribute a prediction to the wrong cause with no visible symptom."
        )


def model_family(model: object) -> ExplainerKind:
    """Classify one *unwrapped* estimator into the SHAP algorithm that fits it.

    Args:
        model: A fitted estimator. A ``Pipeline`` is classified by its final step only —
            call :func:`_unwrap` first if the prefix matters.

    Returns:
        ``"tree"``, ``"linear"`` or ``"permutation"``. The last is not a failure: it is the
        model-agnostic branch, and it is the correct answer for every estimator whose
        structure SHAP has no closed form for.
    """
    name = type(model).__name__
    if name.startswith(TREE_CLASS_PREFIXES):
        return "tree"
    if name.startswith(LINEAR_CLASS_PREFIXES):
        return "linear"
    # A linear model this list has never heard of is still additive in its coefficients,
    # and LinearExplainer needs exactly `coef_` and `intercept_` to be exact on it.
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        return "linear"
    return "permutation"


def _unwrap(model: object, background: Any) -> tuple[object, Any, Any]:  # noqa: ANN401
    """Split a ``Pipeline`` into (final estimator, transform, transformed background).

    Only pipelines whose prefix preserves the column count are unwrapped, and that is
    decided by transforming the background and comparing widths — not by inspecting the
    step types. An imputer and a scaler keep the columns, so the final estimator's
    attributions still name the caller's own columns; a ``ColumnTransformer`` does not, and
    unwrapping past one would hand back a ``(n, n_encoded)`` array against an ``(n, n)``
    question.

    Args:
        model: Any fitted estimator, pipeline or not.
        background: Reference rows, or ``None``.

    Returns:
        ``(estimator, transform, background)`` — ``transform`` is a callable applied to
        explained rows before they reach ``estimator``, or ``None`` when nothing was
        unwrapped.
    """
    steps = getattr(model, "steps", None)
    if not steps or len(steps) < 2 or background is None:
        return model, None, background
    final = steps[-1][1]
    if model_family(final) == "permutation":
        # Nothing is gained by unwrapping into another model-agnostic explanation, and the
        # full pipeline is the object whose predictions the caller actually asked about.
        return model, None, background
    prefix = model[:-1]
    transformed = prefix.transform(background)
    width = getattr(background, "shape", (0, 0))[1]
    if getattr(transformed, "shape", (0, 0))[1] != width:
        return model, None, background
    return final, prefix.transform, transformed


def _predict_function(model: object) -> Any:  # noqa: ANN401 - a bound method
    """Return the continuous output the permutation branch should attribute.

    ``predict_proba`` is preferred wherever it exists: a hard class index has no gradient
    for SHAP to attribute, and explaining the integer label would produce signed numbers
    whose direction is an artefact of the label encoding.

    Args:
        model: A fitted estimator.

    Returns:
        The callable SHAP will evaluate.

    Raises:
        ExplainerUnavailableError: When the model exposes neither method.
    """
    proba = getattr(model, "predict_proba", None)
    if callable(proba):
        return proba
    predict = getattr(model, "predict", None)
    if callable(predict):
        return predict
    raise ExplainerUnavailableError(
        f"{type(model).__name__} exposes neither predict_proba nor predict, so there is no "
        f"output for a model-agnostic explainer to attribute"
    )


def build_explainer(
    model: object,
    background: Any = None,  # noqa: ANN401 - a DataFrame or 2-D array
    *,
    feature_names: Sequence[str] | None = None,
) -> tuple[Any, ExplainerKind]:
    """Build the SHAP explainer that fits ``model``, and say which one was chosen.

    The ``kind`` is returned rather than kept private because it belongs in the record: a
    permutation attribution carries Monte-Carlo noise an exact tree attribution does not,
    and a reader comparing two reports needs to know which they are looking at.

    A plain callable is accepted as ``model`` too — a wrapped ``predict`` over a masked
    matrix, as :mod:`aegis_ml.explain.shap_report` builds. It routes to the permutation
    branch by construction, since a function has no family.

    Args:
        model: A fitted estimator, a ``Pipeline``, or a callable taking a 2-D array.
        background: Reference rows defining what "feature absent" means. Required for the
            linear and permutation branches; **must not be passed through to trees**, and
            is deliberately dropped for them (see this module's docstring).
        feature_names: Column names, attached to the explainer so a ``shap.Explanation``
            labels itself. Ignored by ``TreeExplainer``, which reads them off the model.

    Returns:
        ``(explainer, kind)``.

    Raises:
        ImportError: When ``shap`` is not installed, naming the install command.
        ExplainerUnavailableError: When the chosen branch needs a background and none was
            given, or the model has no continuous output to attribute.
    """
    explainer, kind, _transform = _dispatch(model, background, feature_names)
    return explainer, kind


def _dispatch(
    model: object,
    background: Any,  # noqa: ANN401
    feature_names: Sequence[str] | None,
) -> tuple[Any, ExplainerKind, Any]:
    """Do the routing once, returning the row transform :func:`shap_values` also needs.

    Split out so that building an explainer and using it do not each pay for unwrapping the
    pipeline and transforming the background a second time.

    Args:
        model: A fitted estimator, a ``Pipeline``, or a callable taking a 2-D array.
        background: Reference rows, or ``None``.
        feature_names: Column names to attach to the explainer, or ``None``.

    Returns:
        ``(explainer, kind, transform)``, where ``transform`` maps explained rows into the
        space the explainer works in, or is ``None`` when they are already in it.

    Raises:
        ImportError: When ``shap`` is not installed.
        ExplainerUnavailableError: When the branch needs a background and none was given.
    """
    shap = require("aegis-ml[serve]", "shap")
    names = list(feature_names) if feature_names else None

    if not hasattr(model, "predict") and callable(model):
        if background is None:
            raise ExplainerUnavailableError(
                "a bare predict function can only be explained model-agnostically, and a "
                "permutation explanation is stated relative to a background distribution"
            )
        masker = shap.maskers.Independent(background, max_samples=len(background))
        return shap.PermutationExplainer(model, masker, feature_names=names), "permutation", None

    estimator, transform, reference = _unwrap(model, background)
    kind = model_family(estimator)

    if kind == "tree":
        # No background, deliberately. TreeExplainer's default path uses the fitted trees'
        # own node cover as the reference; handing it data switches it to the interventional
        # algorithm, whose additivity check then fails against the model's output.
        return shap.TreeExplainer(estimator), "tree", transform

    if reference is None:
        raise ExplainerUnavailableError(
            f"{type(estimator).__name__} is explained by the {kind!r} branch, which states "
            f"every contribution relative to a reference distribution — pass the training "
            f"rows as `background`"
        )

    if kind == "linear":
        return shap.LinearExplainer(estimator, reference, feature_names=names), "linear", transform

    masker = shap.maskers.Independent(reference, max_samples=len(reference))
    permutation = shap.PermutationExplainer(
        _predict_function(estimator), masker, feature_names=names
    )
    return permutation, "permutation", transform


def shap_values(
    model: object,
    x: Any,  # noqa: ANN401 - a DataFrame or 2-D array
    background: Any = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401 - a numpy ndarray; numpy is an optional dependency
    """Return SHAP values for ``x`` using whichever explainer fits ``model``.

    Args:
        model: A fitted estimator or pipeline.
        x: Rows to explain, ``(n_rows, n_features)``.
        background: Reference rows. Ignored for tree models — they need none and are
            actively harmed by one — and required for every other family.

    Returns:
        A ``(n_rows, n_features)`` float array for regression and binary classification.
        Multiclass keeps SHAP's trailing class axis, ``(n_rows, n_features, n_classes)``:
        averaging it away here would invent a single number for a quantity that genuinely
        has one value per class, and every caller in this package already branches on
        ``values.ndim == 3``.

    Raises:
        ImportError: When ``shap`` is not installed.
        ExplainerUnavailableError: When the model's branch needs a background and none was
            given.
    """
    np = require("aegis-ml[serve]", "numpy")
    explainer, _kind, transform = _dispatch(model, background, None)
    rows = transform(x) if transform is not None else x
    return np.asarray(explainer.shap_values(rows))
