"""Mirror a registry entry into MLflow — a read-only convenience, never the truth.

**The filesystem registry stays authoritative.** :mod:`aegis_ml.registry.store` writes the
run directory, :mod:`aegis_ml.registry.promote` replaces the serving artifact, and both
work with no server running, no database and no network. This module copies a *view* of
that into MLflow for its UI, its lineage graph and its comparison plots. Nothing in the
train → gate → promote → serve path reads it back.

That direction of dependency is the whole point. If the MLflow server is down at demo
time — or its backing Postgres is, or someone's tracking URI points at a laptop that went
to sleep — training still trains, promotion still promotes, and the model still serves.
The only thing lost is a browser tab. The inverse design (MLflow as the registry, the
filesystem as a cache) turns a flaky HTTP endpoint into a single point of failure for a
model that is already fitted and already on disk.

Mirroring is **off by default** (``AEGIS_ML_ENABLE_MLFLOW=0``). When off, :func:`mirror`
logs one line saying so and returns ``None``. That is a no-op with a receipt, not a silent
skip: a user who expected a mirror finds out from the log, not from an empty UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aegis_ml._require import require
from aegis_ml.contracts.protocols import RegistryEntry
from aegis_ml.settings import settings

__all__ = ["MirrorFailedError", "mirror", "mirror_enabled"]

_LOG = logging.getLogger(__name__)

_ARTIFACT_FILES: tuple[str, ...] = (
    "card.json",
    "card.md",
    "card.html",
    "shap.html",
    "recipe.json",
    "leaderboard.json",
    "metrics.json",
    "entry.json",
)
"""Files uploaded to the MLflow run when they exist in the run directory.

``model.joblib`` is deliberately **not** on this list. The artifact is large, it is
already durable in the run directory, and uploading it invites the belief that MLflow is
where the model lives. It is not: promotion copies from ``runs/<id>/model.joblib``.
"""


class MirrorFailedError(RuntimeError):
    """MLflow was reachable enough to try and failed part-way through.

    Raised rather than swallowed, per the house rule against silent fallbacks — but the
    message states plainly that the registry is unaffected, so a caller can decide to
    continue. A pipeline that wants mirroring to be strictly best-effort catches *this*
    type explicitly; nothing here catches it on the caller's behalf.
    """

    def __init__(self, run_id: str, cause: BaseException) -> None:
        """Name the run, the underlying failure, and what is (not) at risk."""
        super().__init__(
            f"MLflow mirroring of run {run_id!r} failed: {type(cause).__name__}: {cause}. "
            f"The filesystem registry is UNAFFECTED — the run directory, the model and "
            f"the promotion path are all intact, because the mirror is a copy and never "
            f"the source of truth. Fix the tracking server or set "
            f"AEGIS_ML_ENABLE_MLFLOW=0, then re-run `aegis-ml mirror`."
        )
        self.run_id = run_id
        self.cause = cause


def mirror_enabled() -> bool:
    """Return whether mirroring is switched on, without importing mlflow.

    Kept separate so a CLI can print "mirroring: off" in its ``doctor`` output on a
    machine where mlflow is not installed at all.
    """
    return bool(settings.enable_mlflow)


def _params(entry: RegistryEntry) -> dict[str, Any]:
    """Build the MLflow params — the *configuration* of the run, not its results."""
    result = entry.result
    params: dict[str, Any] = {
        "domain_id": entry.domain_id,
        "run_id": entry.run_id,
        "task": result.task,
        "target": result.target,
        "metric_name": result.metric_name,
        "requested_coverage": result.requested_coverage,
        "stage": entry.stage,
    }
    if result.recipe is not None:
        params["tier"] = result.recipe.tier
        params["ensemble_members"] = ",".join(m.name for m in result.recipe.members)
        params["search_seconds"] = result.recipe.search_seconds
    return params


def _metrics(entry: RegistryEntry) -> dict[str, float]:
    """Build the MLflow metrics — every measured scalar, with its name intact.

    ``requested_coverage`` is logged as a *param* and ``empirical_coverage`` as a *metric*
    precisely because they are different kinds of number: one was asked for, one was
    measured. MLflow's own UI then cannot present them as two samples of one series.
    """
    result = entry.result
    metrics: dict[str, float] = {
        result.metric_name: float(result.metric_value),
        "training_size": float(result.training_size),
        "calibration_size": float(result.calibration_size),
        "test_size": float(result.test_size),
    }
    if result.empirical_coverage is not None:
        metrics["empirical_coverage"] = float(result.empirical_coverage)
    if entry.gate is not None:
        for name, value in entry.gate.metrics.items():
            metrics[f"gate_{name}"] = float(value)
    for slice_metric in result.slices:
        key = f"slice_{slice_metric.feature}_{slice_metric.level}_{slice_metric.metric_name}"
        metrics[key.replace(" ", "_")[:250]] = float(slice_metric.metric_value)
    return metrics


def _tags(entry: RegistryEntry) -> dict[str, str]:
    """Build the MLflow tags — the lineage a human searches on."""
    tags = {
        "aegis_ml.run_id": entry.run_id,
        "aegis_ml.domain_id": entry.domain_id,
        "aegis_ml.stage": entry.stage,
        "aegis_ml.created_at": entry.created_at,
        "aegis_ml.source_of_truth": str(Path(settings.registry_dir) / "runs" / entry.run_id),
        "aegis_ml.serving_artifact": str(settings.artifact_path),
    }
    if entry.result.dataset_digest:
        tags["aegis_ml.dataset_digest"] = entry.result.dataset_digest
    if entry.gate is not None:
        tags["aegis_ml.gate_promoted"] = str(entry.gate.promoted)
        if entry.gate.champion_run_id:
            tags["aegis_ml.champion_run_id"] = entry.gate.champion_run_id
    return tags


def mirror(entry: RegistryEntry, *, tracking_uri: str | None = None) -> str | None:
    """Copy one registry entry into MLflow and return the MLflow run id.

    Args:
        entry: The registry row to mirror. Its run directory is read for the card and the
            other JSON/HTML artifacts listed in :data:`_ARTIFACT_FILES`.
        tracking_uri: Override the tracking URI. When ``None``, mlflow's own resolution
            applies (``MLFLOW_TRACKING_URI``, else a local ``mlruns/``), which is what
            makes the offline demo work with no server at all.

    Returns:
        The MLflow run id, or ``None`` when ``settings.enable_mlflow`` is False.

    Raises:
        ImportError: When mirroring is enabled but mlflow is not installed — carrying the
            exact install command, via :func:`aegis_ml._require.require`.
        MirrorFailedError: When the mirror itself fails. The registry is unaffected.
    """
    if not mirror_enabled():
        _LOG.info(
            "mlflow mirror: disabled (AEGIS_ML_ENABLE_MLFLOW=0) — run %s stays in the "
            "filesystem registry only, which is the source of truth either way",
            entry.run_id,
        )
        return None

    mlflow = require("aegis-ml[mlops]", "mlflow")
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(f"aegis-ml/{entry.domain_id}")
        with mlflow.start_run(run_name=entry.run_id) as active:
            mlflow.set_tags(_tags(entry))
            mlflow.log_params(_params(entry))
            mlflow.log_metrics(_metrics(entry))
            directory = Path(settings.registry_dir) / "runs" / entry.run_id
            for name in _ARTIFACT_FILES:
                candidate = directory / name
                if candidate.is_file():
                    mlflow.log_artifact(str(candidate))
            mlflow_run_id = str(active.info.run_id)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, explanatory failure
        raise MirrorFailedError(entry.run_id, exc) from exc

    _LOG.info(
        "mlflow mirror: run %s → mlflow run %s (mirror only; %s remains authoritative)",
        entry.run_id,
        mlflow_run_id,
        settings.registry_dir,
    )
    return mlflow_run_id
