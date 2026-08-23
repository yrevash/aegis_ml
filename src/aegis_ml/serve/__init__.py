"""Serving surfaces: adapter tools for the agent loop, and an optional FastAPI router.

:mod:`aegis_ml.serve.tools` is the one that matters. Aegis's ML spine has been trustworthy
and unreachable — ``predict_explain`` has no consumers in the agent path — and five
``ToolSpec``s dropped into a domain's ``TOOL_REGISTRY`` fix that with no edits inside
``aegis/`` at all. All five are LOW risk, read-only and idempotent, because the platform's
rule is that ML informs and the human gate fires on a tool's risk tier: a prediction is
evidence for a decision, never the decision.

:mod:`aegis_ml.serve.router` is optional and additive — the MLOps console's data source,
mounted with one ``include_router`` call or not at all.

Both are imported lazily so this package costs nothing to import without FastAPI installed.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ML_TOOL_DEFINITIONS",
    "ML_TOOL_NAMES",
    "ML_TOOL_RISK",
    "MLToolResult",
    "build_router",
    "check_model_health",
    "explain_prediction",
    "forecast_series",
    "ml_tool_specs",
    "predict_outcome",
    "whatif_scenario",
]

_LAZY: dict[str, tuple[str, str]] = {
    "ML_TOOL_DEFINITIONS": ("aegis_ml.serve.tools", "ML_TOOL_DEFINITIONS"),
    "ML_TOOL_NAMES": ("aegis_ml.serve.tools", "ML_TOOL_NAMES"),
    "ML_TOOL_RISK": ("aegis_ml.serve.tools", "ML_TOOL_RISK"),
    "MLToolResult": ("aegis_ml.serve.tools", "MLToolResult"),
    "build_router": ("aegis_ml.serve.router", "build_router"),
    "check_model_health": ("aegis_ml.serve.tools", "check_model_health"),
    "explain_prediction": ("aegis_ml.serve.tools", "explain_prediction"),
    "forecast_series": ("aegis_ml.serve.tools", "forecast_series"),
    "ml_tool_specs": ("aegis_ml.serve.tools", "ml_tool_specs"),
    "predict_outcome": ("aegis_ml.serve.tools", "predict_outcome"),
    "whatif_scenario": ("aegis_ml.serve.tools", "whatif_scenario"),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a lazy re-export of many shapes
    """Resolve a serving symbol on first access, keeping import cost at pydantic-only.

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
