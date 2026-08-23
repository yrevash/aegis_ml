"""Cross-validation that refuses to average away the thing you were testing for.

A single train/test split gives one number with no spread, and on the noisy, heteroscedastic
data this package is built for, that number moves by several points of R² depending on the
seed. The gate then promotes on noise. Cross-validation is how the spread becomes visible,
so :class:`CVReport` reports **std next to mean, always** — a mean of 0.62 with a std of 0.03
and a mean of 0.62 with a std of 0.18 are entirely different findings, and only one of them
supports a promotion.

Three refusals are wired in rather than documented as advice:

* **A temporal split is never shuffled.** ``strategy="time_series"`` with ``shuffle=True``
  raises :class:`TemporalShuffleError`. Shuffling a time-ordered frame trains on the future
  and validates on the past; the score goes *up*, which is exactly why nobody catches it.
* **Groups are never silently ignored.** Passing ``groups=`` with a non-group strategy
  raises. Rows from one entity landing on both sides of a split is leakage that inflates
  every fold equally, so no fold looks anomalous.
* **``auto`` announces what it picked.** The resolved strategy is a field on the report, not
  an implementation detail, because "stratified" and "kfold" answer different questions.

:func:`nested_cv` exists for the one question plain CV cannot answer honestly: how well the
*selection procedure* generalises. Picking the best of eight candidates by CV and then
quoting that CV score is selection bias — the maximum of eight noisy estimates is optimistic
by construction. The nested loop selects inside the training fold and scores on data the
selection never saw.

Heavy imports live inside functions: this module costs a pydantic import to load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from aegis_ml.contracts.errors import AegisMLError, InsufficientLabelsError
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.evaluate.metrics import higher_is_better, primary, score
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping, Sequence

    import numpy.typing as npt
    import pandas as pd

__all__ = [
    "CVReport",
    "CVStrategy",
    "FoldScore",
    "TemporalShuffleError",
    "cross_validate",
    "nested_cv",
    "resolve_strategy",
]

CVStrategy = Literal["auto", "kfold", "stratified", "group", "time_series"]
"""Splitting scheme. ``auto`` resolves to ``stratified`` for classification, ``kfold`` for
regression — and the resolution is recorded on the report, never left implicit."""


class TemporalShuffleError(AegisMLError):
    """A time-ordered split was asked to shuffle.

    The reason this is an error and not a warning: shuffling a temporal frame leaks future
    rows into training, and the resulting score is *higher* than the honest one. A check
    that fails in the direction of "looks better" is never caught by looking at results.
    """

    def __init__(self) -> None:
        """Explain why shuffling and a temporal split cannot be reconciled."""
        super().__init__(
            "strategy='time_series' cannot shuffle. TimeSeriesSplit exists precisely to "
            "keep every training fold strictly earlier than its validation fold; shuffling "
            "trains on the future and validates on the past, which inflates the score. "
            "Either drop shuffle=True, or use strategy='kfold' and state plainly that the "
            "frame is being treated as exchangeable."
        )


class FoldScore(BaseModel):
    """One fold's measured metrics, with the split sizes that produced them.

    Sizes are carried because an unstable fold is usually a *small* fold: a 40-row
    validation slice on an imbalanced target can miss a class entirely, and its metric is
    then a different quantity from its neighbours' rather than a worse one.
    """

    fold: int = Field(ge=0, description="Zero-based fold index, in split order.")
    n_train: int = Field(ge=0)
    n_test: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    selected: str | None = Field(
        default=None,
        description="For nested CV: which candidate the inner loop chose on this fold. "
        "Instability here is itself the result — a procedure that picks a different "
        "model every fold has not identified a winner.",
    )


class CVReport(BaseModel):
    """Per-fold and aggregate cross-validation results, spread included.

    ``mean`` without ``std`` is the failure this shape prevents. Everything the gate or the
    model card quotes from a CV run comes from here, so the spread travels with the point
    estimate instead of being dropped at the first hand-off.
    """

    strategy: CVStrategy = Field(description="The RESOLVED strategy, never 'auto'.")
    requested_strategy: CVStrategy = Field(description="What the caller asked for.")
    n_splits: int = Field(ge=2)
    shuffled: bool = Field(description="Whether the splitter shuffled before splitting.")
    seed: int
    task: Literal["regression", "classification"]
    metric_name: str = Field(description="The primary metric the folds are ranked on.")
    higher_is_better: bool
    primary_mean: float
    primary_std: float = Field(ge=0.0)
    primary_min: float
    primary_max: float
    folds: list[FoldScore] = Field(default_factory=list)
    mean: dict[str, float] = Field(default_factory=dict)
    std: dict[str, float] = Field(default_factory=dict)
    nested: bool = Field(default=False, description="True for a nested_cv report.")
    selections: dict[str, int] = Field(
        default_factory=dict,
        description="Nested CV only: candidate name → how many outer folds chose it.",
    )
    notes: list[str] = Field(default_factory=list)

    @property
    def spread_ratio(self) -> float:
        """Fold-to-fold std as a share of the absolute mean — instability at a glance.

        Returns:
            ``std / |mean|``, or ``0.0`` when the mean is exactly zero. A ratio above ~0.25
            means the folds disagree about the model as much as models usually differ from
            each other, and no promotion decision taken on the mean alone is defensible.
        """
        if self.primary_mean == 0.0:
            return 0.0
        return self.primary_std / abs(self.primary_mean)


def resolve_strategy(requested: CVStrategy, task: str) -> CVStrategy:
    """Turn ``auto`` into the concrete strategy for this task.

    Args:
        requested: What the caller asked for.
        task: ``"regression"`` or ``"classification"``.

    Returns:
        The concrete strategy. ``auto`` becomes ``stratified`` for classification (class
        proportions preserved per fold, without which a rare class can be absent from a
        validation fold and its metric silently measures something else) and ``kfold`` for
        regression.
    """
    if requested != "auto":
        return requested
    return "stratified" if task == "classification" else "kfold"


def _build_splitter(
    strategy: CVStrategy,
    *,
    n_splits: int,
    seed: int,
    shuffle: bool,
) -> object:
    """Construct the sklearn splitter for a resolved strategy.

    Args:
        strategy: A concrete strategy (never ``auto``).
        n_splits: Number of folds.
        seed: Random seed; only consulted when ``shuffle`` is true.
        shuffle: Whether to shuffle before splitting.

    Returns:
        A fitted-nothing sklearn splitter exposing ``.split(X, y, groups)``.

    Raises:
        ValueError: On an unknown strategy — never a default splitter, because the split
            scheme determines what the score means.
    """
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
    )

    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    if strategy == "stratified":
        return StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None
        )
    if strategy == "group":
        return GroupKFold(n_splits=n_splits)
    if strategy == "time_series":
        return TimeSeriesSplit(n_splits=n_splits)
    raise ValueError(f"Unknown cross-validation strategy {strategy!r}.")


def _check_inputs(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    n_splits: int,
    strategy: CVStrategy,
    groups: Sequence[object] | None,
) -> None:
    """Refuse the split configurations that produce a real-looking but wrong score.

    Args:
        frame: The labelled frame, features and target columns present.
        problem: The declared problem.
        n_splits: Number of folds.
        strategy: The RESOLVED strategy.
        groups: Group labels, or None.

    Raises:
        InsufficientLabelsError: When there are fewer rows than folds, or (for a stratified
            split) a class with fewer members than folds.
        ValueError: When a declared column is missing from the frame, or when groups are
            supplied to a strategy that would ignore them.
    """
    missing = [c for c in [*problem.feature_names, problem.target.name] if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Frame is missing declared columns {missing}. Cross-validating on the columns "
            f"that happen to be present measures a different model than the one the spec "
            f"describes."
        )
    n_rows = int(len(frame))
    if n_rows < n_splits:
        raise InsufficientLabelsError(n_rows, n_splits, f"{n_splits}-fold cross-validation")
    if groups is not None and strategy != "group":
        raise ValueError(
            f"groups= was supplied with strategy={strategy!r}, which ignores them. Rows "
            f"from one entity would land on both sides of a split; that leakage inflates "
            f"every fold equally, so no fold looks anomalous. Use strategy='group'."
        )
    if strategy == "group" and groups is None:
        raise ValueError("strategy='group' requires groups= (one group label per row).")
    if strategy == "stratified":
        if problem.target.task != "classification":
            raise ValueError(
                "strategy='stratified' needs discrete classes; this problem's target is "
                "regression. Use 'kfold', or bucket the target explicitly if stratifying "
                "on a continuous quantity is genuinely what you want."
            )
        counts = frame[problem.target.name].value_counts()
        if len(counts) and int(counts.min()) < n_splits:
            rare = str(counts.idxmin())
            raise InsufficientLabelsError(
                int(counts.min()),
                n_splits,
                f"{n_splits}-fold stratified CV (class {rare!r} is the limiting class)",
            )


def _fit_predict(
    estimator_factory: Callable[[], object],
    problem: MLProblem,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[npt.ArrayLike, npt.ArrayLike | None, list[str] | None]:
    """Fit one fresh estimator and predict a fold, returning probabilities when available.

    A *factory* rather than an estimator instance is required for a reason that bites in
    practice: pipelines built over skrub/ColumnTransformer carry fitted encoder state, and
    re-fitting the same object across folds is only safe if every step happens to reset
    cleanly. A fresh object per fold is unconditionally correct.

    Args:
        estimator_factory: Zero-argument callable returning an unfitted, sklearn-compatible
            estimator that accepts the RAW frame (encoding belongs inside the pipeline).
        problem: The declared problem.
        x_train: Training features for this fold.
        y_train: Training target for this fold.
        x_test: Validation features for this fold.

    Returns:
        ``(y_pred, y_proba_or_None, class_order_or_None)``.
    """
    estimator = estimator_factory()
    estimator.fit(x_train, y_train)  # type: ignore[attr-defined]
    y_pred = estimator.predict(x_test)  # type: ignore[attr-defined]
    y_proba = None
    classes = None
    if problem.target.task == "classification" and hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(x_test)
        raw = getattr(estimator, "classes_", None)
        if raw is not None:
            classes = [str(c) for c in raw]
    return y_pred, y_proba, classes


def _aggregate(folds: list[FoldScore]) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and population std of every metric present in EVERY fold.

    Metrics computed in only some folds (``roc_auc`` when one fold saw a single class,
    ``mape`` when one fold's actuals are all zero) are deliberately excluded from the
    aggregate rather than averaged over a subset: a mean over three of five folds is not
    the same quantity as a mean over five, and nothing downstream can tell them apart.

    Args:
        folds: The per-fold scores.

    Returns:
        ``(mean, std)`` keyed by metric name.
    """
    import numpy as np

    if not folds:
        return {}, {}
    shared = set(folds[0].metrics)
    for fold in folds[1:]:
        shared &= set(fold.metrics)
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in sorted(shared):
        values = np.asarray([f.metrics[key] for f in folds], dtype=float)
        mean[key] = float(values.mean())
        std[key] = float(values.std())
    return mean, std


def cross_validate(
    estimator_factory: Callable[[], object],
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    n_splits: int = 5,
    strategy: CVStrategy = "auto",
    groups: Sequence[object] | None = None,
    seed: int | None = None,
    shuffle: bool | None = None,
) -> CVReport:
    """Cross-validate one estimator configuration and report the spread with the mean.

    Args:
        estimator_factory: Zero-argument callable returning a *fresh unfitted* estimator
            that consumes the raw frame (its own preprocessing inside). See
            :func:`_fit_predict` for why an instance is not accepted.
        frame: Labelled frame carrying every declared feature and the target column.
        problem: The declared problem; supplies the columns, the task and the metric.
        n_splits: Number of folds. At least 2.
        strategy: ``auto`` | ``kfold`` | ``stratified`` | ``group`` | ``time_series``.
        groups: One group label per row. Required by, and only legal with, ``group``.
        seed: Random seed; defaults to ``settings.random_seed`` so two runs of the demo
            produce the same folds and therefore comparable numbers.
        shuffle: Whether to shuffle before splitting. Defaults to ``True`` for
            ``kfold``/``stratified`` and ``False`` for ``group``/``time_series``. Passing
            ``True`` with ``time_series`` raises :class:`TemporalShuffleError`.

    Returns:
        A :class:`CVReport` with per-fold metrics, means, standard deviations and the
        resolved strategy.

    Raises:
        TemporalShuffleError: When a temporal split was asked to shuffle.
        InsufficientLabelsError: When the frame (or a class) cannot fill the folds.
        ValueError: On a missing column, ignored groups, or an impossible strategy/task pair.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2; got {n_splits}.")
    resolved = resolve_strategy(strategy, problem.target.task)
    if resolved == "time_series" and shuffle:
        raise TemporalShuffleError()
    if shuffle is None:
        shuffle = resolved in {"kfold", "stratified"}
    if resolved == "group" and shuffle:
        raise ValueError(
            "GroupKFold does not shuffle; it partitions by group. Pass shuffle=False."
        )
    seed = settings.random_seed if seed is None else seed

    _check_inputs(frame, problem, n_splits=n_splits, strategy=resolved, groups=groups)

    import numpy as np

    features = list(problem.feature_names)
    x_all = frame[features]
    y_all = frame[problem.target.name]
    group_array = None if groups is None else np.asarray(list(groups))
    if group_array is not None and len(group_array) != len(frame):
        raise ValueError(
            f"groups has {len(group_array)} entries for {len(frame)} rows — a misaligned "
            f"group vector silently splits on the wrong entity."
        )

    splitter = _build_splitter(resolved, n_splits=n_splits, seed=seed, shuffle=shuffle)
    folds: list[FoldScore] = []
    notes: list[str] = []
    for index, (train_idx, test_idx) in enumerate(
        splitter.split(x_all, y_all, group_array)  # type: ignore[attr-defined]
    ):
        x_train = x_all.iloc[train_idx]
        y_train = y_all.iloc[train_idx]
        x_test = x_all.iloc[test_idx]
        y_test = y_all.iloc[test_idx]
        y_pred, y_proba, classes = _fit_predict(
            estimator_factory, problem, x_train, y_train, x_test
        )
        if problem.target.task == "classification" and y_proba is not None:
            from aegis_ml.evaluate.metrics import classification_metrics

            labels = list(problem.target.levels) or classes
            metrics = classification_metrics(y_test, y_pred, y_proba, labels=labels)
        else:
            metrics = score(problem, y_test, y_pred)
        folds.append(
            FoldScore(
                fold=index,
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                metrics=metrics,
            )
        )

    mean, std = _aggregate(folds)
    metric_name, _ = primary(problem, folds[0].metrics)
    per_fold = [f.metrics[metric_name] for f in folds if metric_name in f.metrics]
    if len(per_fold) != len(folds):
        notes.append(
            f"{metric_name} was computable on {len(per_fold)} of {len(folds)} folds; the "
            f"aggregate covers only those folds and is not comparable to a full-fold run."
        )
    values = np.asarray(per_fold, dtype=float)
    if resolved == "time_series":
        notes.append(
            "Temporal split: each training fold is strictly earlier than its validation "
            "fold, and the folds have different training sizes by construction — early "
            "folds are trained on less data and score lower for that reason alone."
        )
    return CVReport(
        strategy=resolved,
        requested_strategy=strategy,
        n_splits=n_splits,
        shuffled=bool(shuffle),
        seed=seed,
        task=problem.target.task,
        metric_name=metric_name,
        higher_is_better=higher_is_better(metric_name),
        primary_mean=float(values.mean()),
        primary_std=float(values.std()),
        primary_min=float(values.min()),
        primary_max=float(values.max()),
        folds=folds,
        mean=mean,
        std=std,
        notes=notes,
    )


def nested_cv(
    candidates: Mapping[str, Callable[[], object]],
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    outer_splits: int = 5,
    inner_splits: int = 3,
    strategy: CVStrategy = "auto",
    groups: Sequence[object] | None = None,
    seed: int | None = None,
) -> CVReport:
    """Score the *selection procedure*, not the selected model — the unbiased estimate.

    Choosing the best of ``k`` candidates by cross-validation and then quoting that
    candidate's CV score double-uses the same data: the maximum of ``k`` noisy estimates is
    optimistic by construction, and the optimism grows with ``k``. On the noisy data this
    package targets, that bias is comfortably large enough to promote a model that is not
    actually better.

    Here the inner loop selects a candidate using only the outer training fold, and the
    outer fold scores that choice on rows the selection never saw. The reported mean is
    therefore an estimate of "what this whole procedure produces", which is the number a
    promotion decision actually rests on.

    ``selections`` is part of the answer, not diagnostics: a procedure that picks a
    different candidate on every outer fold has not identified a winner, however good the
    mean looks.

    Args:
        candidates: Name → zero-argument factory of a fresh unfitted estimator.
        frame: Labelled frame carrying every declared feature and the target column.
        problem: The declared problem.
        outer_splits: Folds in the scoring loop.
        inner_splits: Folds in the selection loop, run inside each outer training fold.
        strategy: Split strategy, applied to both loops.
        groups: Group labels; only legal with ``strategy="group"``, and sliced consistently
            into the inner loop so the group barrier holds at both levels.
        seed: Random seed; defaults to ``settings.random_seed``.

    Returns:
        A :class:`CVReport` with ``nested=True``, per-outer-fold scores, the selected
        candidate per fold and the selection tally.

    Raises:
        ValueError: When ``candidates`` is empty, or any input is inconsistent (see
            :func:`cross_validate`).
    """
    if not candidates:
        raise ValueError(
            "nested_cv needs at least one candidate. An empty search space cannot produce "
            "an unbiased selection score — there is no selection."
        )
    resolved = resolve_strategy(strategy, problem.target.task)
    seed = settings.random_seed if seed is None else seed
    _check_inputs(frame, problem, n_splits=outer_splits, strategy=resolved, groups=groups)

    import numpy as np

    features = list(problem.feature_names)
    x_all = frame[features]
    y_all = frame[problem.target.name]
    group_array = None if groups is None else np.asarray(list(groups))

    shuffle = resolved in {"kfold", "stratified"}
    splitter = _build_splitter(resolved, n_splits=outer_splits, seed=seed, shuffle=shuffle)
    metric_name = problem.metric
    better_is_higher = higher_is_better(metric_name)

    folds: list[FoldScore] = []
    tally: dict[str, int] = {name: 0 for name in candidates}
    for index, (train_idx, test_idx) in enumerate(
        splitter.split(x_all, y_all, group_array)  # type: ignore[attr-defined]
    ):
        train_frame = frame.iloc[train_idx]
        inner_groups = None if group_array is None else list(group_array[train_idx])
        best_name: str | None = None
        best_value: float | None = None
        for name, factory in candidates.items():
            inner = cross_validate(
                factory,
                train_frame,
                problem,
                n_splits=inner_splits,
                strategy=strategy,
                groups=inner_groups,
                seed=seed,
            )
            value = inner.primary_mean
            if (
                best_value is None
                or (better_is_higher and value > best_value)
                or (not better_is_higher and value < best_value)
            ):
                best_name, best_value = name, value
        if best_name is None:  # pragma: no cover - candidates is non-empty, checked above
            raise ValueError("Inner selection produced no winner; candidates was empty.")
        tally[best_name] += 1

        y_pred, y_proba, classes = _fit_predict(
            candidates[best_name],
            problem,
            x_all.iloc[train_idx],
            y_all.iloc[train_idx],
            x_all.iloc[test_idx],
        )
        y_test = y_all.iloc[test_idx]
        if problem.target.task == "classification" and y_proba is not None:
            from aegis_ml.evaluate.metrics import classification_metrics

            labels = list(problem.target.levels) or classes
            metrics = classification_metrics(y_test, y_pred, y_proba, labels=labels)
        else:
            metrics = score(problem, y_test, y_pred)
        folds.append(
            FoldScore(
                fold=index,
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                metrics=metrics,
                selected=best_name,
            )
        )

    mean, std = _aggregate(folds)
    values = np.asarray(
        [f.metrics[metric_name] for f in folds if metric_name in f.metrics], dtype=float
    )
    distinct = sum(1 for count in tally.values() if count)
    notes = [
        "Nested CV: the inner loop selected a candidate using only the outer training "
        "fold, so this mean estimates the SELECTION PROCEDURE's generalisation, not the "
        "chosen model's best-case score.",
    ]
    if distinct > 1:
        notes.append(
            f"The inner loop chose {distinct} different candidates across {len(folds)} "
            f"outer folds ({tally}) — the procedure has not identified a single winner, "
            f"and the mean hides that disagreement."
        )
    return CVReport(
        strategy=resolved,
        requested_strategy=strategy,
        n_splits=outer_splits,
        shuffled=shuffle,
        seed=seed,
        task=problem.target.task,
        metric_name=metric_name,
        higher_is_better=better_is_higher,
        primary_mean=float(values.mean()),
        primary_std=float(values.std()),
        primary_min=float(values.min()),
        primary_max=float(values.max()),
        folds=folds,
        mean=mean,
        std=std,
        nested=True,
        selections=tally,
        notes=notes,
    )
