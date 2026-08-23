"""aegis_ml — SOTA ML/MLOps adapter factory for the Aegis agentic-AI platform.

Aegis already carries a serious ML spine: ``aegis.ml`` is an XGBoost + HistGradientBoosting
soft-voting ensemble with MAPIE split-conformal calibration on a disjoint split, SHAP
attribution averaged by member weight, SHA-256 dataset digests, and a ``ModelCard`` that
separates the coverage it *requested* from the coverage it *measured*. ``aegis.forecast``
is Nixtla StatsForecast with conformal intervals and rolling-origin backtests.

**This package extends that spine. It never replaces it.** What it adds is what Aegis has
no answer for today: AutoML model search, hyperparameter optimisation, data contracts,
a model registry with a promotion gate, drift and label-free performance estimation,
ONNX export, pipelines, and templates for all ten domain-adapter pieces.

The load-bearing design decision is the **two-venv split**. AutoGluon, TabPFN-2.5 and torch
will not resolve under the backend's ``pandas<2.4`` / ``numpy<2.5`` / ``numba==0.67.0``
caps, so the search runs in an isolated trainer venv and its answer crosses back as a
portable JSON :class:`~aegis_ml.contracts.protocols.Recipe`, which the Aegis spine then
fits — keeping conformal calibration, SHAP and the model card intact.

Heavy submodules are imported lazily through ``__getattr__``, mirroring
``aegis/ml/__init__.py``, so importing this package costs a pydantic import and nothing else.
"""

from __future__ import annotations

from typing import Any

from aegis_ml.contracts import (
    Candidate,
    DriftReport,
    FeatureSpec,
    GateDecision,
    Leaderboard,
    MLProblem,
    Recipe,
    RegistryEntry,
    TargetSpec,
    TrainResult,
)
from aegis_ml.settings import Settings, settings

__version__ = "0.1.0"

__all__ = [
    "Candidate",
    "DriftReport",
    "FeatureSpec",
    "GateDecision",
    "Leaderboard",
    "MLProblem",
    "Recipe",
    "RegistryEntry",
    "Settings",
    "TargetSpec",
    "TrainResult",
    "__version__",
    "settings",
]

_LAZY = {
    "train": ("aegis_ml.pipelines.flows", "train_flow"),
    "evaluate": ("aegis_ml.pipelines.flows", "eval_flow"),
    "promote": ("aegis_ml.registry.promote", "promote"),
    "drift": ("aegis_ml.monitor.drift", "drift_report"),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a lazy re-export of many shapes
    """Resolve a heavy symbol on first access, keeping import cost at pydantic-only."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])
