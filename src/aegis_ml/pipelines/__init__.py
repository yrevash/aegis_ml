"""Pipelines: plain Python functions that produce artifacts without an orchestrator.

Decision D4, restated because it governs everything in this package: **a trained artifact
must never depend on a server being up.** The flows here are ordinary functions returning
typed results; :mod:`aegis_ml.pipelines.prefect_shim` turns them into Prefect flows when
Prefect is installed and enabled, and does nothing otherwise. The artifacts are identical.

What each flow leaves behind is a :class:`~aegis_ml.contracts.protocols.RunManifest`: one
row per stage with its duration, row counts, measured metrics, cache key and — on failure —
the exception attributed to the stage that raised it. That is the difference between a log
and a lineage record, and it is what makes a crashed run resumable rather than merely
re-runnable.

Everything is imported lazily: the flows pull pandas, sklearn and the AutoML stack, and
``import aegis_ml.pipelines`` must not.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "REALISM_ACCURACY_BAND",
    "REALISM_R2_BAND",
    "DataBundle",
    "StageCache",
    "StageGraph",
    "StageSpec",
    "data_flow",
    "drift_flow",
    "eval_flow",
    "flow",
    "forecast_flow",
    "full_flow",
    "new_manifest",
    "prefect_active",
    "promote_flow",
    "render_summary",
    "stage",
    "task",
    "train_flow",
    "write_manifest",
]

_LAZY: dict[str, tuple[str, str]] = {
    "REALISM_ACCURACY_BAND": ("aegis_ml.pipelines.flows", "REALISM_ACCURACY_BAND"),
    "REALISM_R2_BAND": ("aegis_ml.pipelines.flows", "REALISM_R2_BAND"),
    "DataBundle": ("aegis_ml.pipelines.flows", "DataBundle"),
    "StageCache": ("aegis_ml.pipelines.manifest", "StageCache"),
    "StageGraph": ("aegis_ml.pipelines.manifest", "StageGraph"),
    "StageSpec": ("aegis_ml.pipelines.manifest", "StageSpec"),
    "data_flow": ("aegis_ml.pipelines.flows", "data_flow"),
    "drift_flow": ("aegis_ml.pipelines.flows", "drift_flow"),
    "eval_flow": ("aegis_ml.pipelines.flows", "eval_flow"),
    "flow": ("aegis_ml.pipelines.prefect_shim", "flow"),
    "forecast_flow": ("aegis_ml.pipelines.flows", "forecast_flow"),
    "full_flow": ("aegis_ml.pipelines.flows", "full_flow"),
    "new_manifest": ("aegis_ml.pipelines.manifest", "new_manifest"),
    "prefect_active": ("aegis_ml.pipelines.prefect_shim", "prefect_active"),
    "promote_flow": ("aegis_ml.pipelines.flows", "promote_flow"),
    "render_summary": ("aegis_ml.pipelines.manifest", "render_summary"),
    "stage": ("aegis_ml.pipelines.manifest", "stage"),
    "task": ("aegis_ml.pipelines.prefect_shim", "task"),
    "train_flow": ("aegis_ml.pipelines.flows", "train_flow"),
    "write_manifest": ("aegis_ml.pipelines.manifest", "write_manifest"),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a lazy re-export of many shapes
    """Resolve a pipeline symbol on first access, keeping import cost at pydantic-only.

    Args:
        name: Attribute being accessed on this package.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: If ``name`` is not one of this package's exports.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> list[str]:
    """List the lazy exports so tab-completion and ``help()`` see them."""
    return sorted(__all__)
