"""Hand-built result objects for tests that exercise decision logic, not estimators.

The promotion gate consumes ``TrainResult`` objects and compares numbers on them. Producing
those numbers by fitting two models would make a gate test fail when an estimator changed,
which is the opposite of what it is for. So the numbers are literals and the objects are the
real pydantic types — no shape is faked, only the provenance of the floats.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aegis_ml.contracts.protocols import (
    Candidate,
    GateDecision,
    Leaderboard,
    RegistryEntry,
    SliceMetric,
    TrainResult,
)

__all__ = [
    "DOMAIN",
    "gate_decision",
    "leaderboard",
    "registry_entry",
    "slices",
    "train_result",
]

DOMAIN = "cold_chain_logistics"
"""The reference domain id, so builders and the real adapter agree without a fixture."""


def slices(
    *values: float, metric_name: str = "r2", feature: str = "carrier_tier"
) -> list[SliceMetric]:
    """Build one ``SliceMetric`` per value, on synthetic level names.

    Args:
        *values: The metric value for each slice, in order.
        metric_name: The metric the slices are measured in.
        feature: The feature the slices segment on.

    Returns:
        One ``SliceMetric`` per value, levels named ``level_0``, ``level_1``, …
    """
    return [
        SliceMetric(
            feature=feature,
            level=f"level_{i}",
            n_rows=40 + i,
            metric_name=metric_name,
            metric_value=float(value),
        )
        for i, value in enumerate(values)
    ]


def train_result(
    run_id: str,
    metric_value: float,
    *,
    metric_name: str = "r2",
    empirical_coverage: float | None = 0.91,
    requested_coverage: float = 0.90,
    worst_slices: list[SliceMetric] | None = None,
    domain_id: str = DOMAIN,
    task: str = "regression",
    target: str = "spoilage_risk_pct",
) -> TrainResult:
    """Build a ``TrainResult`` carrying exactly the fields the gate reads.

    Args:
        run_id: Identifier the decision will quote.
        metric_value: The measured primary metric.
        metric_name: Which metric that is.
        empirical_coverage: Measured conformal coverage; ``None`` means "not measured",
            which the gate treats as a failure rather than a pass.
        requested_coverage: The level that was asked for.
        worst_slices: Slice sweep; ``None`` gives a default three-slice sweep whose worst
            value sits a little under ``metric_value``.
        domain_id: The domain.
        task: ``"regression"`` or ``"classification"``.
        target: Target column name.

    Returns:
        A populated ``TrainResult``.
    """
    if worst_slices is None:
        worst_slices = slices(
            metric_value, metric_value - 0.05, metric_value - 0.10, metric_name=metric_name
        )
    return TrainResult(
        run_id=run_id,
        domain_id=domain_id,
        task=task,  # type: ignore[arg-type]
        target=target,
        metric_name=metric_name,
        metric_value=float(metric_value),
        requested_coverage=float(requested_coverage),
        empirical_coverage=empirical_coverage,
        training_size=600,
        calibration_size=200,
        test_size=200,
        slices=worst_slices,
    )


def gate_decision(run_id: str, *, promoted: bool, reasons: list[str] | None = None) -> GateDecision:
    """Build a ``GateDecision`` directly, for tests of what ``promote`` does with one."""
    verdict = "PROMOTED" if promoted else "REJECTED"
    return GateDecision(
        promoted=promoted,
        challenger_run_id=run_id,
        reasons=reasons or [f"{verdict}: built by tests.fixtures.builders"],
        checks=dict.fromkeys(
            (
                "beats_champion",
                "coverage_meets_request",
                "contracts_pass",
                "worst_slice_not_worse",
                "no_target_leakage",
            ),
            promoted,
        ),
    )


def leaderboard(*, metric_name: str = "r2") -> Leaderboard:
    """Build a two-row leaderboard with one selected winner and one recorded loser."""
    return Leaderboard(
        metric_name=metric_name,
        higher_is_better=True,
        candidates=[
            Candidate(
                name="baseline_voting",
                tier="baseline",
                metric_name=metric_name,
                metric_value=0.62,
                selected=True,
            ),
            Candidate(
                name="linear_reference",
                tier="baseline",
                metric_name=metric_name,
                metric_value=0.41,
            ),
        ],
        tiers_run=["baseline"],
        tiers_skipped={"autogluon": "not importable in this interpreter"},
    )


def registry_entry(result: TrainResult, *, stage: str = "staging") -> RegistryEntry:
    """Wrap a ``TrainResult`` in the registry row ``store.save_run`` persists."""
    return RegistryEntry(
        run_id=result.run_id,
        domain_id=result.domain_id,
        created_at=datetime.now(UTC).isoformat(),
        stage=stage,  # type: ignore[arg-type]
        result=result,
    )
