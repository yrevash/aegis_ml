"""One command that answers "is this frame fit to train on?" before anything expensive runs.

Three independent things can be wrong with a training frame, and each is invisible to the
checks that catch the other two:

1. **It violates its own contract.** An unseen categorical level, a value outside its
   declared range, a null in a column marked non-nullable. Caught by
   :mod:`aegis_ml.contracts.frames` — and by nothing else, because
   ``OneHotEncoder(handle_unknown="ignore")`` is contractually obliged not to raise.
2. **Its label is noise, or its label is too easy.** Caught by
   :mod:`aegis_ml.data.latent`. Both bounds matter: below the floor the model learns
   nothing and the conformal interval is honestly enormous; above the ceiling something is
   leaking and the interval is a hairline that informs no decision.
3. **Its columns are structurally unusable** — constant, mostly null, an identifier in
   disguise, or a declared ``datetime`` the spine will hand straight to a tree learner as
   ``datetime64[ns]``. Caught here.

Running them together is the point. A frame that passes the contract but has a noise label
trains a model that will pass all 14 Aegis conformance checks and print ``distinct=False``
on the last line of ``python -m app.ml`` — minutes before a demo, when the fix is a
regenerate-and-retrain cycle nobody has time for.

The default is to **report, not raise**. This is the diagnostic a domain author runs while
iterating, and an exception on the first problem hides the other three. ``strict=True``
turns it into a gate, and re-raises the most specific typed error the evidence supports
rather than a generic wrapper — the gate in ``aegis_ml.evaluate`` needs to know *which*
failure it hit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from aegis_ml.contracts.spec import MLProblem

__all__ = [
    "HIGH_CARDINALITY_SHARE",
    "HIGH_NULL_SHARE",
    "ColumnAudit",
    "ContractReport",
    "check",
]

logger = logging.getLogger(__name__)

HIGH_NULL_SHARE = 0.5
"""Null share above which a column is reported as mostly invented.

The spine imputes silently — median for numerics, mode for categoricals — and lists what it
filled in through ``MLExplainResponse.imputed_features``. Past half the rows, the model is
learning the imputed constant more than the feature.
"""

HIGH_CARDINALITY_SHARE = 0.5
"""Distinct-values-per-row ratio above which a categorical is reported as an identifier.

One-hot encoding an id produces a matrix as wide as the frame is long, and a tree that
splits on it memorises the training set — which then reads as an excellent held-out score
right up until a row arrives with an id the model has never seen.
"""


class ColumnAudit(BaseModel):
    """Structural findings for one column.

    Attributes:
        column: Column name.
        declared_dtype: What the spec says it is, or ``None`` for an undeclared column.
        pandas_dtype: What the frame actually holds.
        n_null: Null count.
        null_share: Nulls as a share of rows.
        n_unique: Distinct non-null values.
        cardinality_share: ``n_unique`` over row count.
        constant: Whether it holds a single value.
        issues: Human-readable problems, each carrying its number.
    """

    column: str
    declared_dtype: str | None = None
    pandas_dtype: str = ""
    n_null: int = 0
    null_share: float = 0.0
    n_unique: int = 0
    cardinality_share: float = 0.0
    constant: bool = False
    issues: list[str] = Field(default_factory=list)


class ContractReport(BaseModel):
    """Everything the three checks found, with every number that produced a verdict.

    ``ok`` is deliberately narrow: it means the frame is fit to *train* on. A frame can be
    ``ok`` while carrying issues — a 20%-null column is a finding, not a blocker — and the
    distinction is what makes the report worth reading rather than just its boolean.

    Attributes:
        domain_id: Which problem was checked.
        n_rows: Rows in the frame.
        n_columns: Columns in the frame.
        schema_ok: Whether the pandera contract passed.
        schema_error: The formatted violation list, when it did not.
        learnable: Whether the held-out probe cleared its floor.
        metric_name: ``"r2"`` or ``"accuracy"``.
        metric_value: What the probe measured, or ``None`` when it could not run.
        floor: The bar the probe had to clear, majority-adjusted for classification.
        ceiling: The bar above which the score is treated as a leak.
        suspiciously_easy: Whether the probe scored above the ceiling. Not a failure by
            itself — see :func:`aegis_ml.data.latent.assert_learnable` — but never silent.
        learnability_error: Why the probe could not run, when it could not.
        leakage: One line per feature flagged by the leakage detector.
        columns: Per-column structural audit.
        issues: Blocking problems — non-empty means ``ok`` is ``False``.
        warnings: Non-blocking findings a human should still read.
        ok: Whether the frame is fit to train on.
    """

    domain_id: str
    n_rows: int
    n_columns: int
    schema_ok: bool = False
    schema_error: str | None = None
    learnable: bool = False
    metric_name: str = ""
    metric_value: float | None = None
    floor: float = 0.0
    ceiling: float = 0.0
    suspiciously_easy: bool = False
    learnability_error: str | None = None
    leakage: list[str] = Field(default_factory=list)
    columns: list[ColumnAudit] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    ok: bool = False

    def summary(self) -> str:
        """One-line verdict suitable for a CLI's last line.

        Returns:
            A compact human-readable status carrying the measured metric.
        """
        head = "PASS" if self.ok else "FAIL"
        metric = (
            f"{self.metric_name}={self.metric_value:.4f}"
            if self.metric_value is not None
            else f"{self.metric_name or 'metric'}=unmeasured"
        )
        return (
            f"{head} contract[{self.domain_id}] rows={self.n_rows} "
            f"schema={'ok' if self.schema_ok else 'violated'} {metric} "
            f"issues={len(self.issues)} warnings={len(self.warnings)}"
        )


def check(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    strict: bool = False,
    floor: float | None = None,
    ceiling: float | None = None,
    include_leakage: bool = True,
    seed: int | None = None,
) -> ContractReport:
    """Run the schema, learnability and structural checks and return one report.

    Ordering is chosen so the cheapest, most diagnostic check runs first: a schema violation
    usually explains whatever the learnability probe would have reported, and reporting
    "R² = 0.02" for a frame whose target column is spelled wrong sends the reader looking in
    the wrong place. The probe still runs afterwards, on the original frame, because a
    contract violation in one column does not make the rest of the evidence worthless.

    Args:
        frame: The candidate training frame.
        problem: The problem it must satisfy.
        strict: Whether to raise on the first blocking finding instead of reporting it.
        floor: Overrides the configured learnability floor.
        ceiling: Overrides the sanity ceiling.
        include_leakage: Whether to run the per-feature leakage probe. Costs one small fit
            per feature; worth it before an AutoML search, skippable inside a tight loop.
        seed: Seed for every split and estimator involved.

    Returns:
        The :class:`ContractReport`.

    Raises:
        AegisMLError: When ``strict`` and the schema failed, or a structural issue blocks.
        LabelNotLearnableError: When ``strict`` and the label did not clear its floor.
        TargetLeakageError: When ``strict`` and a feature leaks.
        InsufficientLabelsError: When ``strict`` and there are too few rows to measure.
    """
    from aegis_ml.contracts.frames import validate
    from aegis_ml.data.latent import measure_learnability

    resolved_seed = settings.random_seed if seed is None else seed
    report = ContractReport(
        domain_id=problem.domain_id,
        n_rows=int(len(frame)),
        n_columns=int(len(frame.columns)),
        columns=_audit_columns(frame, problem),
    )

    schema_failure: Exception | None = None
    try:
        validate(frame, problem)
        report.schema_ok = True
    except AegisMLError as exc:
        schema_failure = exc
        report.schema_ok = False
        report.schema_error = str(exc)
        report.issues.append("data contract violated (see schema_error)")
    except ImportError as exc:
        report.schema_ok = False
        report.schema_error = str(exc)
        report.warnings.append(
            "pandera is not installed, so the data contract was NOT checked — this is an "
            "unchecked frame, not a clean one"
        )
        logger.warning("Skipping pandera validation: %s", exc)

    learnability_failure: Exception | None = None
    try:
        learnability = measure_learnability(
            frame, problem, floor=floor, ceiling=ceiling, seed=resolved_seed
        )
    except (AegisMLError, ImportError) as exc:
        learnability_failure = exc
        report.learnability_error = str(exc)
        report.issues.append("learnability could not be measured (see learnability_error)")
    else:
        report.learnable = learnability.learnable
        report.metric_name = learnability.metric_name
        report.metric_value = learnability.metric_value
        report.floor = learnability.effective_floor
        report.ceiling = learnability.ceiling
        report.suspiciously_easy = learnability.suspiciously_easy
        report.warnings.extend(learnability.notes)
        if not learnability.learnable:
            from aegis_ml.contracts.errors import LabelNotLearnableError

            learnability_failure = LabelNotLearnableError(
                learnability.metric_name,
                learnability.metric_value,
                learnability.effective_floor,
            )
            report.issues.append(
                f"label is not learnable: {learnability.metric_name}="
                f"{learnability.metric_value:.4f} below floor "
                f"{learnability.effective_floor:.4f}"
            )

    leak_failure: Exception | None = None
    if include_leakage:
        leak_failure = _run_leakage(frame, problem, report, resolved_seed)

    for audit in report.columns:
        for issue in audit.issues:
            report.warnings.append(f"{audit.column}: {issue}")

    report.ok = not report.issues
    if strict and not report.ok:
        _raise_most_specific(report, schema_failure, learnability_failure, leak_failure)
    return report


def _run_leakage(
    frame: pd.DataFrame,
    problem: MLProblem,
    report: ContractReport,
    seed: int,
) -> Exception | None:
    """Run the leakage detector, folding its findings into ``report``.

    Returns:
        The typed error to re-raise under ``strict``, or ``None``.
    """
    from aegis_ml.contracts.errors import TargetLeakageError
    from aegis_ml.features.leakage import detect_leakage

    try:
        signals = detect_leakage(frame, problem, seed=seed)
    except (AegisMLError, ImportError) as exc:
        report.warnings.append(f"leakage detection did not run: {exc}")
        return None
    report.leakage = [signal.detail for signal in signals]
    if not signals:
        return None
    report.issues.append(
        f"{len(signals)} leakage signal(s); strongest is {signals[0].feature!r} at "
        f"{signals[0].score:.4f}"
    )
    return TargetLeakageError(signals[0].feature, signals[0].score, signals[0].threshold)


def _audit_columns(frame: pd.DataFrame, problem: MLProblem) -> list[ColumnAudit]:
    """Audit every declared column plus the target for structural unusability."""
    declared: dict[str, Any] = {f.name: f for f in problem.features}
    audits: list[ColumnAudit] = []
    n_rows = max(int(len(frame)), 1)
    names = [*problem.feature_names, problem.target.name]
    for name in names:
        spec = declared.get(name)
        if name not in frame.columns:
            audits.append(
                ColumnAudit(
                    column=name,
                    declared_dtype=spec.dtype if spec is not None else "target",
                    issues=[
                        "declared in the spec but absent from the frame — the generator "
                        "and the spec disagree about the schema"
                    ],
                )
            )
            continue
        column = frame[name]
        n_null = int(column.isna().sum())
        n_unique = int(column.nunique(dropna=True))
        audit = ColumnAudit(
            column=name,
            declared_dtype=spec.dtype if spec is not None else problem.target.task,
            pandas_dtype=str(column.dtype),
            n_null=n_null,
            null_share=n_null / n_rows,
            n_unique=n_unique,
            cardinality_share=n_unique / n_rows,
            constant=n_unique <= 1,
        )
        _flag_column(audit, spec, n_rows)
        audits.append(audit)
    return audits


def _flag_column(audit: ColumnAudit, spec: Any, n_rows: int) -> None:  # noqa: ANN401
    """Attach the structural findings for one audited column."""
    if audit.constant:
        audit.issues.append(
            f"constant across all {n_rows} rows — contributes nothing, and inflates the "
            f"feature count the model card reports"
        )
    if audit.null_share >= HIGH_NULL_SHARE:
        audit.issues.append(
            f"{audit.null_share:.1%} null — past half the rows the model is learning the "
            f"imputed constant more than the feature"
        )
    if spec is not None and spec.dtype == "categorical":
        if audit.cardinality_share >= HIGH_CARDINALITY_SHARE:
            audit.issues.append(
                f"{audit.n_unique} distinct values over {n_rows} rows — this behaves like "
                f"an identifier; one-hot encoding it lets a tree memorise the training set"
            )
        declared_levels = set(spec.levels)
        if declared_levels and audit.n_unique > len(declared_levels):
            audit.issues.append(
                f"holds {audit.n_unique} distinct values but the spec declares "
                f"{len(declared_levels)} levels — the extra ones encode to all-zeros "
                f"without raising"
            )
    if spec is not None and spec.dtype == "datetime":
        audit.issues.append(
            "declared datetime: MLProblem.numeric_features (like Aegis's ResolvedSpec) "
            "treats every non-categorical feature as a passthrough, so this column reaches "
            "the estimator as datetime64[ns] and the tree learner will reject it. Convert "
            "it to engineered numerics (epoch seconds, day-of-week, hours-since) in the "
            "generator and declare those instead"
        )
    if spec is not None and not spec.nullable and audit.n_null > 0:
        audit.issues.append(
            f"{audit.n_null} nulls in a column the spec declares nullable=False — the "
            f"pandera contract rejects this frame"
        )


def _raise_most_specific(
    report: ContractReport,
    schema_failure: Exception | None,
    learnability_failure: Exception | None,
    leak_failure: Exception | None,
) -> None:
    """Re-raise the most specific typed error behind a failing report.

    Order matters. A schema violation usually causes whatever else was measured, leakage
    invalidates the learnability number rather than the reverse, and only when none of the
    three named failures fired is a generic refusal the honest answer.

    Raises:
        Exception: The most specific typed error available, or :class:`AegisMLError`.
    """
    for failure in (schema_failure, leak_failure, learnability_failure):
        if failure is not None:
            raise failure
    joined = "\n  - ".join(report.issues)
    raise AegisMLError(
        f"Data contract check failed for {report.domain_id!r}:\n  - {joined}\n{report.summary()}"
    )
