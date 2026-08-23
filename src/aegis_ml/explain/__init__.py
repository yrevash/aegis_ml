"""Explanation and reporting: SHAP, partial dependence, reason codes and the model card.

The Aegis spine already explains its own predictions — SHAP attribution averaged across the
soft-voting members by weight, returned on every ``MLExplainResponse``. This package is the
layer above that: it turns those attributions into **artifacts** (a stored global picture, a
self-contained HTML report, an extended model card) and into **sentences** the agent can act
on.

The through-line is that an explanation is only useful when it is stated in the domain's own
vocabulary. ``region = 1.0`` is the encoding; ``region = emea`` is the fact. Every renderer
here prefers ``ShapFeature.value_label`` over ``value`` for exactly that reason, and
:mod:`~aegis_ml.explain.reason_codes` generates the adapter's ``describe_prediction`` so the
same rule holds in the generated code that reaches the agent.

Two properties are shared by every report this package writes:

* **Self-contained.** Inline CSS, inline SVG, no CDN, no script, no font request. These pages
  are opened from a filesystem path and read offline; anything fetched is a section that
  renders blank exactly where the report matters.
* **Nothing plausible is invented.** An unmeasured quantity renders as "not measured" rather
  than as a reasonable-looking number, and a feature the model ignored keeps its near-zero
  row in the chart, because that row is the evidence the model is not chasing noise.
"""

from __future__ import annotations

from aegis_ml.explain.card import (
    TABPFN_LICENSE_NOTICE,
    CoverageBlock,
    ExtendedModelCard,
    build_card,
    measure_missingness,
    render_html,
    render_markdown,
)
from aegis_ml.explain.pdp import (
    PartialDependenceUnavailableError,
    PDPCurve,
    partial_dependence_curves,
)
from aegis_ml.explain.pdp import render_html as render_pdp_html
from aegis_ml.explain.reason_codes import (
    describe_prediction_text,
    emit_describe_prediction_source,
    reason_codes,
)
from aegis_ml.explain.shap_report import (
    ExplainerUnavailableError,
    global_importance,
    local_explanation,
)
from aegis_ml.explain.shap_report import render_html as render_shap_html

__all__ = [
    "TABPFN_LICENSE_NOTICE",
    "CoverageBlock",
    "ExplainerUnavailableError",
    "ExtendedModelCard",
    "PDPCurve",
    "PartialDependenceUnavailableError",
    "build_card",
    "describe_prediction_text",
    "emit_describe_prediction_source",
    "global_importance",
    "local_explanation",
    "measure_missingness",
    "partial_dependence_curves",
    "reason_codes",
    "render_html",
    "render_markdown",
    "render_pdp_html",
    "render_shap_html",
]
