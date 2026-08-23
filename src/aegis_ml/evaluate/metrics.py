"""The metric registry — values *and* the direction each one improves in.

Two things make this module load-bearing rather than a thin wrapper over sklearn.

**Direction is data, not folklore.** :data:`HIGHER_IS_BETTER` exists because the promotion
gate compares a challenger against a champion and must know which way is better. Hard-coding
``challenger > champion`` promotes the model with the *larger* RMSE — a worse model — and
nothing downstream can tell, because both numbers are real and both are printed. The gate
therefore never guesses: it asks this table, and a metric missing from the table raises
:class:`UnknownMetricError` instead of defaulting to "higher wins".

**Absence beats NaN.** MAPE is undefined where the actual is zero. Returning ``nan`` looks
like a number, survives ``json.dumps`` as ``NaN``, and compares ``False`` against every
threshold — so a gate criterion on it silently fails open or closed depending on which side
of the comparison it lands. Instead the excluded rows are counted in ``mape_n_excluded`` and
the key is simply **absent** when every actual is zero. A missing key is checkable; a NaN is
a trap.

The vocabulary here is aligned with Aegis: ``aegis.ml.types.ModelCard.metric_name`` only ever
carries ``"r2"`` or ``"accuracy"``, so those are the defaults :func:`primary` resolves to via
:attr:`~aegis_ml.contracts.spec.MLProblem.metric`. Everything else computed here is
supporting evidence printed alongside, never a substitute for the card's headline number.

Heavy imports (numpy, sklearn) live inside the functions: importing this module must stay as
cheap as importing pydantic, because ``aegis_ml.evaluate`` is imported by the CLI at startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aegis_ml.contracts.errors import AegisMLError, InsufficientLabelsError
from aegis_ml.contracts.spec import MLProblem

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy.typing as npt

__all__ = [
    "HIGHER_IS_BETTER",
    "MetricNotComputedError",
    "UnknownMetricError",
    "classification_metrics",
    "higher_is_better",
    "primary",
    "regression_metrics",
    "score",
]


HIGHER_IS_BETTER: dict[str, bool] = {
    # Regression.
    "r2": True,
    "rmse": False,
    "mae": False,
    "mape": False,
    "median_ae": False,
    # Classification.
    "accuracy": True,
    "balanced_accuracy": True,
    "f1_macro": True,
    "precision": True,
    "recall": True,
    "roc_auc": True,
    "log_loss": False,
    # Calibration (computed in aegis_ml.evaluate.calibration, ranked here).
    "brier_score": False,
    "expected_calibration_error": False,
    "max_calibration_error": False,
}
"""Metric name → whether a larger value is a better model.

Every metric this package can rank appears here. The gate reads it directly; getting
``rmse`` wrong would promote the worse model with a straight face.
"""


class UnknownMetricError(AegisMLError):
    """A metric was named that this package cannot rank.

    Raised rather than assuming a direction. "Higher is better" is right for most metrics
    and catastrophically wrong for the error metrics, and the failure is invisible: the
    gate reports a real number, a real champion and a real challenger, and promotes the
    wrong one.
    """

    def __init__(self, name: str) -> None:
        """Name the unrankable metric and the table that must learn about it."""
        known = ", ".join(sorted(HIGHER_IS_BETTER))
        super().__init__(
            f"Metric {name!r} has no declared direction, so nothing here can say whether a "
            f"larger value is a better model. Add it to "
            f"`aegis_ml.evaluate.metrics.HIGHER_IS_BETTER` with its direction. Known "
            f"metrics: {known}."
        )
        self.name = name


class MetricNotComputedError(AegisMLError):
    """The requested primary metric is absent from the computed metric dict.

    This is the honest report of a real gap — e.g. ``roc_auc`` asked for as the primary
    metric while no probabilities were passed, or ``mape`` on an all-zero target. Falling
    back to another metric here would let the gate rank two models on two different scales.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        """Name the missing metric and what was actually computed."""
        super().__init__(
            f"Primary metric {name!r} was not computed. Available: "
            f"{', '.join(sorted(available)) or '(none)'}. Pass the inputs it needs "
            f"(probabilities for roc_auc/log_loss) or change `MLProblem.primary_metric` — "
            f"ranking on a different metric than the one requested would compare two models "
            f"on two different scales."
        )
        self.name = name
        self.available = list(available)


def higher_is_better(metric_name: str) -> bool:
    """Return whether a larger value of ``metric_name`` means a better model.

    Args:
        metric_name: A key of :data:`HIGHER_IS_BETTER`.

    Returns:
        ``True`` when larger is better (``r2``, ``accuracy``), ``False`` for error metrics.

    Raises:
        UnknownMetricError: When the metric has no declared direction. Never guessed.
    """
    try:
        return HIGHER_IS_BETTER[metric_name]
    except KeyError as exc:
        raise UnknownMetricError(metric_name) from exc


def _as_arrays(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, what: str) -> tuple:
    """Coerce two label-shaped inputs to aligned 1-D numpy arrays.

    Args:
        y_true: Ground truth, any array-like (list, Series, ndarray).
        y_pred: Predictions, aligned by position with ``y_true``.
        what: Human name of the measurement, used in the refusal message.

    Returns:
        ``(y_true, y_pred)`` as 1-D numpy arrays of equal length.

    Raises:
        InsufficientLabelsError: When there are no rows to measure.
        ValueError: When the two inputs disagree on length — a misalignment that would
            otherwise produce a plausible-looking score computed against the wrong rows.
    """
    import numpy as np

    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if true.shape[0] != pred.shape[0]:
        raise ValueError(
            f"{what}: y_true has {true.shape[0]} rows and y_pred has {pred.shape[0]}. "
            f"A length mismatch does not raise inside numpy broadcasting for every shape — "
            f"it can produce a real-looking score computed against the wrong rows."
        )
    if true.shape[0] == 0:
        raise InsufficientLabelsError(0, 1, what)
    return true, pred


def regression_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """Compute the regression metric set on one aligned pair of arrays.

    ``r2`` leads because it is what ``ModelCard.metric_name`` carries for regression, but it
    is scale-free and therefore says nothing about whether an error is *operationally*
    large. ``rmse`` and ``mae`` are reported in the target's own unit for exactly that
    reason, and ``median_ae`` alongside ``mae`` exposes a heavy tail: when the two diverge,
    a minority of rows carries most of the error, which is the signature of the
    heteroscedastic noise the conformal interval has to absorb.

    Args:
        y_true: Measured target values from the held-out split.
        y_pred: Model predictions aligned with ``y_true``.

    Returns:
        ``{"r2", "rmse", "mae", "median_ae"}`` always; ``"mape"`` and ``"mape_n_excluded"``
        when at least one actual is non-zero. ``mape`` is **absent**, never NaN, when every
        actual is zero — see the module docstring.

    Raises:
        InsufficientLabelsError: When there are no rows.
        ValueError: When the inputs are not aligned.
    """
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    true, pred = _as_arrays(y_true, y_pred, "Regression metrics")
    out: dict[str, float] = {
        "r2": float(r2_score(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
        "median_ae": float(median_absolute_error(true, pred)),
    }
    nonzero = np.abs(true) > 0.0
    n_excluded = int((~nonzero).sum())
    if nonzero.any():
        ratios = np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])
        out["mape"] = float(ratios.mean())
        out["mape_n_excluded"] = float(n_excluded)
    return out


def classification_metrics(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    y_proba: npt.ArrayLike | None = None,
    *,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """Compute the classification metric set, adding probability metrics when available.

    ``accuracy`` is the card's headline (``ModelCard.metric_name``) and is also the metric
    most easily faked by class imbalance: predicting the majority class on a 85/15 split
    scores 0.85 while learning nothing. ``balanced_accuracy`` and ``f1_macro`` are therefore
    always reported next to it — on an imbalanced target the gap between them *is* the
    finding.

    ``roc_auc`` and ``log_loss`` need probabilities, so they appear only when ``y_proba`` is
    passed. They are omitted rather than approximated from hard labels: an AUC computed from
    0/1 predictions is a different quantity that happens to land in the same range.

    Args:
        y_true: Measured class labels from the held-out split.
        y_pred: Predicted class labels aligned with ``y_true``.
        y_proba: Optional predicted probabilities — shape ``(n,)`` or ``(n, 2)`` for binary,
            ``(n, k)`` for multiclass, with columns ordered as ``labels`` (or as
            ``numpy.unique(y_true)`` when ``labels`` is omitted).
        labels: Optional explicit class ordering, matching the columns of ``y_proba``.
            Pass it whenever a class may be absent from ``y_true`` in this split, otherwise
            the inferred ordering silently shifts the probability columns.

    Returns:
        ``{"accuracy", "balanced_accuracy", "f1_macro", "precision", "recall"}`` always,
        plus ``"roc_auc"`` and ``"log_loss"`` when probabilities were supplied and the
        class set supports them.

    Raises:
        InsufficientLabelsError: When there are no rows.
        ValueError: When the inputs are not aligned, or ``y_proba`` has the wrong width.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    true, pred = _as_arrays(y_true, y_pred, "Classification metrics")
    classes = np.asarray(labels) if labels is not None else np.unique(true)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "n_classes": float(len(classes)),
    }
    if y_proba is None:
        return out

    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[0] != true.shape[0]:
        raise ValueError(
            f"y_proba has {proba.shape[0]} rows but y_true has {true.shape[0]}."
        )
    if proba.shape[1] != len(classes):
        raise ValueError(
            f"y_proba has {proba.shape[1]} columns but {len(classes)} classes were "
            f"resolved ({list(classes)}). Pass `labels=` with the estimator's own "
            f"`classes_` ordering — an inferred ordering silently permutes the columns and "
            f"produces a real-looking but wrong AUC."
        )
    out["log_loss"] = float(log_loss(true, proba, labels=list(classes)))
    if len(classes) == 2:
        out["roc_auc"] = float(roc_auc_score(true == classes[1], proba[:, 1]))
    elif len(np.unique(true)) == len(classes):
        out["roc_auc"] = float(
            roc_auc_score(true, proba, multi_class="ovr", average="macro", labels=list(classes))
        )
    return out


def score(
    problem: MLProblem,
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    y_proba: npt.ArrayLike | None = None,
) -> dict[str, float]:
    """Compute the metric set appropriate to the problem's task.

    One entry point so that every caller — cross-validation, the slice sweep, the gate and
    the model card — measures the *same* quantities in the same way. Two callers computing
    "accuracy" through two code paths eventually disagree by a rounding rule, and the gate
    compares them anyway.

    Args:
        problem: The declared problem; only ``target.task`` and ``target.levels`` are read.
        y_true: Measured target values from the held-out split.
        y_pred: Predictions aligned with ``y_true``.
        y_proba: Optional class probabilities (classification only). Ignored for regression,
            where it has no meaning.

    Returns:
        The task's metric dict, as documented on :func:`regression_metrics` and
        :func:`classification_metrics`.
    """
    if problem.target.task == "regression":
        return regression_metrics(y_true, y_pred)
    labels = list(problem.target.levels) or None
    return classification_metrics(y_true, y_pred, y_proba, labels=labels)


def primary(problem: MLProblem, metrics: dict[str, float]) -> tuple[str, float]:
    """Pull the problem's ranking metric out of a computed metric dict.

    Args:
        problem: The declared problem; :attr:`~aegis_ml.contracts.spec.MLProblem.metric`
            resolves to ``r2``/``accuracy`` when the spec leaves ``primary_metric`` blank,
            which keeps the gate and ``ModelCard.metric_name`` naming the same number.
        metrics: Output of :func:`score`.

    Returns:
        ``(metric_name, metric_value)``.

    Raises:
        MetricNotComputedError: When the requested metric is absent. Substituting another
            metric would rank two models on two different scales.
        UnknownMetricError: When the requested metric has no declared direction, since a
            value nothing can rank is useless to the gate that consumes it.
    """
    name = problem.metric
    higher_is_better(name)  # Refuse now, not inside the gate three stages later.
    if name not in metrics:
        raise MetricNotComputedError(name, list(metrics))
    return name, float(metrics[name])
