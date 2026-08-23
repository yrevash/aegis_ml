"""Portable export of the fitted point predictor — with its limits stated up front.

One module, :mod:`~aegis_ml.export.onnx`, and one thing worth repeating here because it is
the thing readers assume otherwise: **an ONNX export is the point predictor only.** MAPIE's
conformal interval and the SHAP attributions do not survive the conversion, so an
ONNX-served prediction has no coverage guarantee and cannot explain itself. It is a
portability and verification artefact — a file another runtime can score, plus a measured
round-trip difference proving it computes the same function — not an alternative serving
path for Aegis, which already runs ``onnxruntime`` transitively and already answers in
under a millisecond *with* the interval.
"""

from __future__ import annotations

from aegis_ml.export.onnx import (
    DEFAULT_TOLERANCE,
    register_converters,
    to_onnx,
    validate_roundtrip,
)

__all__ = [
    "DEFAULT_TOLERANCE",
    "register_converters",
    "to_onnx",
    "validate_roundtrip",
]
