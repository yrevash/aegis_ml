"""Feature handling: one preprocessing path that must not drift, one that may.

:mod:`~aegis_ml.features.pipeline` holds both.
:func:`~aegis_ml.features.pipeline.column_transformer`
is a deliberate mirror of ``aegis.ml.model._build_preprocessor`` — one-hot with
``handle_unknown="ignore"``, numeric passthrough, ``remainder="drop"`` — because the AutoML
recipe crosses a venv boundary and is re-fitted by the Aegis spine under *its* preprocessor,
and because ``_encoded_parents`` reconstructs SHAP attribution from exactly that column
layout. :func:`~aegis_ml.features.pipeline.skrub_pipeline` is the richer exploratory path,
whose scores are an accuracy ceiling rather than something to promote.

:mod:`~aegis_ml.features.leakage` finds the feature that already knows the answer. Aegis
catches only the perfect case — ``MLProblem`` refuses a target that is also a feature — and
every subtler form produces the best-looking numbers in the whole pipeline right up until
the model meets a row where the leaking column is not yet populated.

Both modules import pandas, scikit-learn and skrub inside functions through
:func:`aegis_ml._require.require`, so importing this package stays pydantic-only.
"""

from __future__ import annotations

from aegis_ml.features.leakage import LeakSignal, assert_no_leakage, detect_leakage
from aegis_ml.features.pipeline import column_transformer, encode_frame, skrub_pipeline

__all__ = [
    "LeakSignal",
    "assert_no_leakage",
    "column_transformer",
    "detect_leakage",
    "encode_frame",
    "skrub_pipeline",
]
