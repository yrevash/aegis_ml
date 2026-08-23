"""Turning drift measurements into alerts — and being precise about what "block" means.

**A blocking drift verdict does NOT withdraw the serving model.** Aegis serves the model
it has and flags it. That is a deliberate position, and it is the opposite of what a
naively-built monitor does:

* Withdrawing the model on drift converts a *degraded* answer into **no answer at all**.
  For a decision-support system, a prediction carrying a wide conformal interval and a
  visible drift flag is strictly more useful than a 503 — the human can see the caveat and
  decide. An outage tells them nothing and looks like a bug.
* Drift is measured on the *inputs*. It says the world moved, not that the model broke.
  The model may still be fine; :mod:`aegis_ml.monitor.perf` is the module that estimates
  whether it is.
* Nothing here can know what the fallback would be. There is no "safe" model to switch to;
  the previous champion was calibrated on the *same* reference frame that just drifted.

What a block **does** stop is **promotion of anything calibrated on the drifted
reference**. A challenger whose gate metrics were measured against a reference the live
data has moved away from has not been evaluated on the world it is about to serve, and
promoting it on those numbers is promoting a measurement that no longer applies. Hence
:class:`~aegis_ml.contracts.errors.DriftThresholdExceededError`, whose own message says
exactly this, and hence :func:`raise_if_blocking` being called from the promotion path
rather than from the serving path.

Alert levels map to actions, not to feelings:

* ``info`` — recorded, nothing gates on it. Usually "not enough current rows to trust this".
* ``warn`` — visible in the console and the model card; serving and promotion both
  continue. This is where a drifting-but-usable model lives, and it is the common case
  with noisy real data.
* ``block`` — :func:`raise_if_blocking` refuses promotion. Serving is untouched.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from aegis_ml.contracts.errors import DriftThresholdExceededError
from aegis_ml.contracts.protocols import DriftReport
from aegis_ml.settings import settings

__all__ = [
    "Alert",
    "AlertConfig",
    "AlertLevel",
    "evaluate_alerts",
    "raise_if_blocking",
]

_LOG = logging.getLogger(__name__)

AlertLevel = Literal["info", "warn", "block"]
"""``info`` records, ``warn`` flags, ``block`` refuses promotion (never serving)."""


class Alert(BaseModel):
    """One fired condition, with the number that fired it and the line it crossed.

    ``value`` and ``threshold`` are both required and both stored. An alert that says
    "drift detected" without the pair is unactionable — the reader cannot tell a share of
    0.42 against a 0.40 threshold (noise, most likely) from 0.92 against 0.40 (the world
    changed), and those call for opposite responses.
    """

    level: AlertLevel = Field(description="info | warn | block. See the module docstring.")
    code: str = Field(description="Stable machine key, e.g. 'drift.share.block'.")
    message: str = Field(description="One sentence a human can act on.")
    metric: str = Field(description="What was measured, e.g. 'drifted_share'.")
    value: float = Field(description="The MEASURED value.")
    threshold: float = Field(description="The line it was compared against.")


class AlertConfig(BaseModel):
    """Thresholds for :func:`evaluate_alerts`, defaulting to the process settings.

    Separate from :class:`~aegis_ml.settings.Settings` so a caller can evaluate the same
    report under different policies — a strict gate for promotion, a lax one for a
    dashboard — without mutating global state and without two sources of truth for
    "what is drifted".
    """

    drift_share_warn: float = Field(
        default_factory=lambda: settings.drift_share_warn,
        ge=0.0,
        le=1.0,
        description="Share of drifted features at which the report is flagged.",
    )
    drift_share_block: float = Field(
        default_factory=lambda: settings.drift_share_block,
        ge=0.0,
        le=1.0,
        description="Share at which promotion of anything calibrated on the drifted "
        "reference is refused. Serving is never affected.",
    )
    target_p_value: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Target-drift p-value below which the label distribution itself is "
        "flagged. Stricter than the per-feature 0.05 on purpose: the target is ONE test, "
        "so it needs no multiple-comparison headroom, and a shifted target invalidates "
        "the model's calibration rather than merely its inputs.",
    )
    prediction_p_value: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Prediction-drift p-value below which the model's own output "
        "distribution is flagged.",
    )
    min_current_rows: int = Field(
        default=200,
        ge=1,
        description="Below this, alerts are reported at 'info' regardless of the share: "
        "a drifted share computed over a handful of rows is sampling variation with a "
        "percentage sign on it.",
    )
    estimated_metric_floor: float | None = Field(
        default=None,
        description="Optional floor for DriftReport.estimated_metric_value (NannyML). "
        "None by default because the floor is metric-specific — an R² floor and an RMSE "
        "floor point in opposite directions, and guessing which one applies would produce "
        "an alert that fires backwards.",
    )


def evaluate_alerts(report: DriftReport, *, config: AlertConfig | None = None) -> list[Alert]:
    """Turn a measured drift report into a list of alerts, worst first.

    The headline check is the **share of drifted features**, never a single p-value. With
    a dozen features tested at p<0.05, one crossing the line is what happens when nothing
    at all has changed — a monitor that alerts on that is one a team mutes in a week, and a
    muted monitor is worse than none because it is still on the architecture diagram.

    A small current sample downgrades every share-based alert to ``info``: the same share
    means something entirely different over 80 rows and over 80,000, and the row count is
    the thing a reader forgets to check.

    Args:
        report: The measured report.
        config: Thresholds; defaults to the process settings.

    Returns:
        Alerts ordered ``block`` → ``warn`` → ``info``. Empty means nothing fired.
    """
    conf = config or AlertConfig()
    alerts: list[Alert] = []
    underpowered = report.n_current_rows < conf.min_current_rows

    if underpowered:
        alerts.append(
            Alert(
                level="info",
                code="drift.sample.small",
                message=(
                    f"Only {report.n_current_rows} current rows were compared against "
                    f"{report.n_reference_rows} reference rows (advisory minimum "
                    f"{conf.min_current_rows}). Share-based drift alerts are downgraded to "
                    f"info: at this size the share is dominated by sampling variation."
                ),
                metric="n_current_rows",
                value=float(report.n_current_rows),
                threshold=float(conf.min_current_rows),
            )
        )

    if report.drifted_share >= conf.drift_share_block:
        alerts.append(
            Alert(
                level="info" if underpowered else "block",
                code="drift.share.block",
                message=(
                    f"{report.drifted_share:.0%} of features drifted "
                    f"({len(report.drifted_features)} of them: "
                    f"{', '.join(report.drifted_features[:6]) or 'n/a'}), at or above the "
                    f"block threshold {conf.drift_share_block:.0%}. The serving model is "
                    f"NOT withdrawn — this blocks PROMOTION of anything calibrated on the "
                    f"drifted reference, because its gate metrics were measured against a "
                    f"distribution the live data has left."
                ),
                metric="drifted_share",
                value=report.drifted_share,
                threshold=conf.drift_share_block,
            )
        )
    elif report.drifted_share >= conf.drift_share_warn:
        alerts.append(
            Alert(
                level="info" if underpowered else "warn",
                code="drift.share.warn",
                message=(
                    f"{report.drifted_share:.0%} of features drifted "
                    f"({', '.join(report.drifted_features[:6]) or 'n/a'}), at or above the "
                    f"warn threshold {conf.drift_share_warn:.0%}. Serving and promotion "
                    f"both continue; the flag travels on the model card. Retraining on "
                    f"recent data is the usual response."
                ),
                metric="drifted_share",
                value=report.drifted_share,
                threshold=conf.drift_share_warn,
            )
        )

    if report.target_drift is not None and report.target_drift < conf.target_p_value:
        alerts.append(
            Alert(
                level="warn",
                code="drift.target",
                message=(
                    f"The target distribution itself moved (p={report.target_drift:.4g} < "
                    f"{conf.target_p_value:g}). This invalidates calibration, not just the "
                    f"inputs: the conformal interval was sized on the old label "
                    f"distribution, so its empirical coverage should be re-measured before "
                    f"the intervals are quoted to anyone."
                ),
                metric="target_drift_p_value",
                value=report.target_drift,
                threshold=conf.target_p_value,
            )
        )

    if report.prediction_drift is not None and report.prediction_drift < conf.prediction_p_value:
        alerts.append(
            Alert(
                level="warn",
                code="drift.prediction",
                message=(
                    f"The model's own output distribution moved "
                    f"(p={report.prediction_drift:.4g} < {conf.prediction_p_value:g}). With "
                    f"feature drift this is the expected consequence; WITHOUT it, suspect "
                    f"the serving pipeline — a changed encoder or a reordered feature "
                    f"vector shifts predictions while the inputs look identical."
                ),
                metric="prediction_drift_p_value",
                value=report.prediction_drift,
                threshold=conf.prediction_p_value,
            )
        )

    floor = conf.estimated_metric_floor
    estimated = report.estimated_metric_value
    if floor is not None and estimated is not None and estimated < floor:
        alerts.append(
            Alert(
                level="warn",
                code="perf.estimated.below_floor",
                message=(
                    f"{report.estimated_metric_name or 'estimated metric'} = "
                    f"{estimated:.4f}, below the configured floor {floor:.4f}. This is an "
                    f"ESTIMATE from NannyML, not a measurement — treat it as a reason to "
                    f"seek labels, not as a measured regression."
                ),
                metric=report.estimated_metric_name or "estimated_metric",
                value=estimated,
                threshold=floor,
            )
        )

    order = {"block": 0, "warn": 1, "info": 2}
    alerts.sort(key=lambda alert: order[alert.level])
    if alerts:
        _LOG.info(
            "alerts: run %s — %s",
            report.run_id,
            ", ".join(f"{a.level}:{a.code}" for a in alerts),
        )
    return alerts


def raise_if_blocking(report: DriftReport, *, config: AlertConfig | None = None) -> None:
    """Refuse promotion when drift is blocking — serving is never affected.

    Call this from the promotion path — :func:`aegis_ml.registry.promote.promote` installs
    a model whose gate metrics were measured on the reference frame, and when the live
    distribution has moved away from that frame those metrics describe a world that no
    longer exists. Calling it from a serving path would be a category error: it would take
    a working model offline because its *inputs* changed.

    Args:
        report: The measured drift report.
        config: Thresholds; defaults to the process settings.

    Raises:
        DriftThresholdExceededError: When a blocking alert fired. Its message states
            plainly that the serving model is not withdrawn.
    """
    for alert in evaluate_alerts(report, config=config):
        if alert.level == "block":
            raise DriftThresholdExceededError(alert.metric, alert.value, alert.threshold)
