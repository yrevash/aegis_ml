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
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.contracts.protocols import RegistryEntry
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "BackfillReport",
    "MirrorFailedError",
    "backfill",
    "mirror",
    "mirror_enabled",
]

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


# ────────────────────────────────────────────────────────────────────────── backfill ──
#
# `mirror` copies one entry as training finishes, and only when AEGIS_ML_ENABLE_MLFLOW is
# set. That gate is right for the training path — a flaky tracking server must never be
# able to slow down or fail a run that has already fitted a model. It is wrong for the
# dashboard, where the user has typed a command whose entire purpose is to populate the
# MLflow UI. So `backfill` does not consult the flag: the request IS the consent, and the
# function says so in its own log line rather than reading a setting the caller did not set.

_METRIC_KEY = re.compile(r"[^0-9a-zA-Z_\-./ :]")
"""Characters MLflow rejects in a metric key.

Slice metric keys are built from real categorical levels — ``route_class=last_mile_pool``,
but also levels carrying brackets, plus signs or non-ASCII. A rejected key aborts the whole
run's logging, so the key is rewritten and the *original* is preserved in the run's tags,
which keeps the mapping recoverable instead of merely surviving.
"""


@dataclass
class BackfillReport:
    """What one backfill pass did, run by run.

    Attributes:
        tracking_uri: Where the runs were written.
        logged: Run ids newly written into MLflow.
        skipped: Run ids already present, matched on the ``aegis_ml.run_id`` tag.
        failed: ``(run_id, reason)`` for runs that could not be written. A failure here is
            collected rather than raised so that one unreadable run directory cannot stop
            the other nineteen from reaching the UI — but every one of them is returned,
            and the caller prints them.
    """

    tracking_uri: str
    logged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        """How many entries were considered."""
        return len(self.logged) + len(self.skipped) + len(self.failed)

    def summary(self) -> str:
        """One line for the terminal, naming all three outcomes even when they are zero."""
        return (
            f"mlflow backfill: {len(self.logged)} logged, {len(self.skipped)} already "
            f"present, {len(self.failed)} failed, of {self.total} registry entries"
        )


def _artifact_files(directory: Path) -> list[tuple[Path, str | None]]:
    """Return ``(file, artifact_subdirectory)`` pairs to upload for one run.

    Args:
        directory: The run directory under ``registry_store/runs/``.

    Returns:
        The JSON/HTML artifacts named in :data:`_ARTIFACT_FILES` at the artifact root, plus
        every rendered PNG under ``visuals/`` in a ``visuals`` subdirectory. MLflow renders
        images inline, so uploading the nine figures turns each MLflow run page into the
        same evidence the hub shows — which is the point of mirroring at all.

    ``model.joblib`` stays excluded for the reason :data:`_ARTIFACT_FILES` gives: promotion
    copies from the run directory, and an MLflow copy invites the belief that it does not.
    """
    pairs: list[tuple[Path, str | None]] = []
    for name in _ARTIFACT_FILES:
        candidate = directory / name
        if candidate.is_file():
            pairs.append((candidate, None))
    visuals = directory / "visuals"
    if visuals.is_dir():
        for png in sorted(visuals.glob("*.png")):
            pairs.append((png, "visuals"))
    return pairs


def backfill(
    entries: Sequence[RegistryEntry],
    *,
    tracking_uri: str,
    registry_dir: Path | None = None,
) -> BackfillReport:
    """Write every registry entry into an MLflow store, skipping ones already there.

    Args:
        entries: Registry rows to mirror, in any order.
        tracking_uri: The MLflow tracking URI to write to — for the dashboard this is the
            SQLite store under ``registry_store/mlflow/``, which needs no server of its own
            to be written and no network to be read.
        registry_dir: Root the run directories are read from. Defaults to
            ``settings.registry_dir``.

    Returns:
        A :class:`BackfillReport`. Per-run failures are collected in it rather than raised.

    Raises:
        ImportError: When mlflow is not installed, carrying the exact install command.

    Idempotency is by search, not by bookkeeping: before writing, each entry's
    ``aegis_ml.run_id`` tag is looked up in its experiment, and a hit is skipped. That
    survives the cases a local ledger would not — a store rebuilt by hand, a second
    checkout writing into the same database, a backfill interrupted half way through.

    This does **not** consult ``settings.enable_mlflow``. That flag exists to keep a
    tracking server off the training path; a caller invoking this function has asked for
    the mirror explicitly, and quietly doing nothing in response would be the silent no-op
    this repository refuses everywhere else.
    """
    mlflow = require("aegis-ml[dashboard]", "mlflow")
    root = Path(registry_dir if registry_dir is not None else settings.registry_dir)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    report = BackfillReport(tracking_uri=tracking_uri)
    experiments: dict[str, str] = {}

    for entry in entries:
        try:
            name = f"aegis-ml/{entry.domain_id}"
            if name not in experiments:
                found = client.get_experiment_by_name(name)
                if found is not None:
                    experiments[name] = found.experiment_id
                else:
                    # An artifact root has to be named at creation time. Left to MLflow's
                    # own default this resolves to `./mlruns` relative to the working
                    # directory — so the figures would land beside the checkout rather
                    # than inside the registry, and a second invocation from a different
                    # directory would write a second, unrelated copy.
                    artifacts = Path(root) / "mlflow" / "artifacts" / entry.domain_id
                    artifacts.mkdir(parents=True, exist_ok=True)
                    experiments[name] = client.create_experiment(
                        name, artifact_location=artifacts.as_uri()
                    )
            experiment_id = experiments[name]

            already = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.\"aegis_ml.run_id\" = '{entry.run_id}'",
                max_results=1,
            )
            if already:
                report.skipped.append(entry.run_id)
                continue

            tags = dict(_tags(entry))
            tags["mlflow.runName"] = entry.run_id
            run = client.create_run(experiment_id=experiment_id, tags=tags)
            mlflow_run_id = run.info.run_id

            for key, value in _params(entry).items():
                client.log_param(mlflow_run_id, key, value)
            renamed: dict[str, str] = {}
            for key, value in _metrics(entry).items():
                if not math.isfinite(value):
                    continue
                clean = _METRIC_KEY.sub("_", key)
                if clean != key:
                    renamed[clean] = key
                client.log_metric(mlflow_run_id, clean, value)
            for clean, original in renamed.items():
                client.set_tag(mlflow_run_id, f"aegis_ml.metric_key.{clean}"[:250], original)

            for path, subdir in _artifact_files(root / "runs" / entry.run_id):
                client.log_artifact(mlflow_run_id, str(path), artifact_path=subdir)

            client.set_terminated(mlflow_run_id, "FINISHED")
            report.logged.append(entry.run_id)
        except Exception as exc:  # noqa: BLE001 - collected per run, returned, never hidden
            report.failed.append((entry.run_id, f"{type(exc).__name__}: {exc}"))

    _LOG.info(
        "%s (tracking_uri=%s; the filesystem registry at %s remains authoritative)",
        report.summary(),
        tracking_uri,
        root,
    )
    return report
