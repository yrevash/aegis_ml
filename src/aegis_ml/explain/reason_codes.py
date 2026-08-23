"""SHAP into the domain's own sentence — and the generator that makes ``describe_prediction`` live.

``backend/src/app/adapter/ml_spec.py`` already contains a ``describe_prediction(resp, *,
top_k=3) -> str``. It is a good function: it names the target and its unit, states the
calibrated interval and its confidence, and lists the strongest signed SHAP drivers. Its
docstring says the core "injects the returned block into the planner and generate prompts so
the agent plans *with* the model".

**Nothing in the Aegis repository calls it.** It is dead code. The prediction, the conformal
interval and the SHAP attribution are all computed, and the sentence that would put them in
front of the agent is never invoked. The fix is not to edit the core — it is to route ML
through adapter **tools** (``aegis_ml.serve.tools`` registers ``predict_outcome``,
``explain_prediction``, ``whatif_scenario``), because a tool's return value *does* reach the
agent's context. That is what makes this function live, and this module is what generates it.

Three public functions, in increasing order of commitment:

* :func:`reason_codes` — one sentence per driver, in the domain's units.
* :func:`describe_prediction_text` — the whole decision-support block, including the
  conformal interval, its confidence, and the provenance signals the existing adapter
  version leaves out.
* :func:`emit_describe_prediction_source` — the Python source for the adapter's own
  ``describe_prediction``, to be embedded in a generated ``ml_spec.py``. It is parsed with
  :func:`ast.parse` before being returned, because generated code that does not compile
  takes down the adapter's import and, with it, ``resolve_spec``'s ability to find
  ``FEATURE_NAMES`` — which fails over to a four-column noise spec without raising.

The rendering rule that matters most: ``ShapFeature.value`` for a categorical feature is the
one-hot indicator ``1.0``, which names no level. ``value_label`` carries the level. Every
sentence built here prefers the label, so a driver reads ``region = emea`` and never
``region = 1.0`` — a sentence a human is asked to act on must name the thing, not its
encoding.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from aegis_ml.contracts.spec import MLProblem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "describe_prediction_text",
    "emit_describe_prediction_source",
    "reason_codes",
]


def _get(obj: object, name: str, default: Any = None) -> Any:  # noqa: ANN401 - duck-typed access
    """Read a field from either a pydantic response or a plain mapping.

    The same shape arrives from three places — ``aegis.ml.types.MLExplainResponse`` in the
    serving venv, a JSON dict across the venv boundary, and a test double built as a
    ``SimpleNamespace``. Accepting all three keeps this module free of a hard import of the
    heavy Aegis package, which is the same reason ``explain/card.py`` nests the Aegis model
    card as a dict.

    Args:
        obj: The response-like object.
        name: Field name.
        default: Value to return when the field is absent.

    Returns:
        The field value, or ``default``.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _format_number(value: float) -> str:
    """Render a feature value the way a domain reader writes it.

    Whole numbers lose their decimal point (``14``, not ``14.0``) and trailing zeros are
    trimmed (``3.2``, not ``3.20``), because the sentence is read aloud in a standup. Small
    magnitudes keep more significant digits so a value of 0.0032 does not render as ``0.00``,
    which would say the driver did nothing.

    Args:
        value: The numeric value.

    Returns:
        The formatted number.
    """
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def _render_value(feature: object, problem: MLProblem) -> str:
    """Render one driver's value, preferring the categorical level over the indicator.

    Args:
        feature: A ``ShapFeature``-shaped object or dict.
        problem: The declared problem, consulted for the feature's unit.

    Returns:
        The value as it should appear in a sentence — the level name for a categorical, the
        number plus its declared unit for a numeric.
    """
    label = _get(feature, "value_label")
    if label is not None and str(label) != "":
        return str(label)
    raw = _get(feature, "value")
    if raw is None:
        return "unknown"
    try:
        number = _format_number(float(raw))
    except (TypeError, ValueError):
        return str(raw)
    name = str(_get(feature, "feature", ""))
    spec = next((f for f in problem.features if f.name == name), None)
    if spec is not None and spec.unit:
        return f"{number} {spec.unit}"
    return number


def reason_codes(
    resp_like: object,
    problem: MLProblem,
    *,
    top_k: int = 3,
) -> list[str]:
    """Turn SHAP attributions into sentences a domain reader can act on.

    For a regression target, SHAP contributions are in the **target's own units** — that is a
    property of the attribution, not a convenience — so the sentence can honestly say
    "pushed the estimate up by 3.2 hours". For a classification target the contributions live
    in the model's output space (probability or log-odds depending on the estimator), so they
    are rendered as unitless weights and the sentence says "pushed the prediction toward"
    rather than attaching a fake quantity to them.

    Args:
        resp_like: A response carrying ``shap_attribution`` — an
            ``aegis.ml.types.MLExplainResponse``, a dict of the same shape, or anything with
            those attributes.
        problem: The declared problem; supplies the target's unit and the features' units.
        top_k: How many drivers to render, strongest absolute contribution first.

    Returns:
        One sentence per driver, strongest first. An empty list when the response carries no
        attribution — an empty list is checkable, whereas a sentence like "no strong drivers"
        would be a claim nobody measured.
    """
    attributions = list(_get(resp_like, "shap_attribution", []) or [])
    if not attributions:
        return []
    attributions.sort(
        key=lambda item: abs(float(_get(item, "contribution", 0.0) or 0.0)),
        reverse=True,
    )

    unit = problem.target.unit or ""
    suffix = f" {unit}" if unit else ""
    prediction = _get(resp_like, "prediction")
    out: list[str] = []
    for item in attributions[:top_k]:
        name = str(_get(item, "feature", "(unnamed feature)"))
        value = _render_value(item, problem)
        contribution = float(_get(item, "contribution", 0.0) or 0.0)
        if problem.target.task == "regression":
            direction = "up" if contribution >= 0 else "down"
            out.append(
                f"{name} = {value} pushed the estimate {direction} by "
                f"{_format_number(abs(contribution))}{suffix}"
            )
        else:
            toward = "toward" if contribution >= 0 else "away from"
            target_class = (
                str(prediction) if prediction is not None else problem.target.name
            )
            out.append(
                f"{name} = {value} pushed the prediction {toward} "
                f"'{target_class}' (weight {abs(contribution):.2f})"
            )
    return out


def describe_prediction_text(
    resp_like: object,
    problem: MLProblem,
    *,
    top_k: int = 3,
) -> str:
    """Render the full decision-support block: prediction, interval, drivers, provenance.

    The shape follows ``app.adapter.ml_spec.describe_prediction`` closely — a task header,
    the prediction with its unit, the calibrated interval or set, then the drivers — so the
    generated adapter reads like the reference one. Two things are added on purpose:

    * **The interval is stated in the target's unit**, with the confidence beside it. An
      interval without its confidence is a range with no meaning attached, and a confidence
      without its interval is a number with nothing to be confident about.
    * **Provenance is stated when it is bad news.** ``data_source == "synthetic"`` means the
      serving model carries no domain signal, and ``imputed_features`` names the inputs the
      caller did not supply that were filled from training medians. Both are already computed
      by the spine and both are dropped by the current adapter version, which is how a
      confident-looking sentence gets built on top of an imputed row.

    Args:
        resp_like: The spine response (or a dict of the same shape).
        problem: The declared problem; supplies the target's name, task and unit.
        top_k: How many SHAP drivers to name.

    Returns:
        A compact multi-line block safe to embed in a prompt or a tool result.
    """
    target = problem.target
    unit = f" {target.unit}" if target.unit else ""
    prediction = _get(resp_like, "prediction")

    if isinstance(prediction, int | float) and not isinstance(prediction, bool):
        head = f"Predicted {target.name}: {float(prediction):.1f}{unit}"
    elif prediction is None:
        head = f"Predicted {target.name}: unavailable"
    else:
        head = f"Predicted {target.name}: {prediction}"

    lines = [f"ML decision-support ({target.task}):", f"- {head}"]

    interval = _get(resp_like, "conformal_interval")
    confidence = _get(resp_like, "conformal_confidence")
    set_size = _get(resp_like, "prediction_set_size")
    width = _get(resp_like, "interval_width")
    if interval is not None and confidence is not None:
        low, high = float(interval[0]), float(interval[1])
        span = f" (width {float(width):.1f}{unit})" if width is not None else ""
        lines.append(
            f"- {float(confidence):.0%} calibrated interval [{low:.1f}, {high:.1f}]"
            f"{unit}{span} — the MEASURED coverage of this interval is on the model card; "
            f"the level quoted here is the one that was requested."
        )
    elif set_size is not None and confidence is not None:
        lines.append(
            f"- {float(confidence):.0%} conformal set size {int(set_size)} "
            f"(1 = the model excluded every other class at this confidence)"
        )

    codes = reason_codes(resp_like, problem, top_k=top_k)
    if codes:
        lines.append("- Why:")
        lines.extend(f"  - {code}" for code in codes)

    data_source = _get(resp_like, "data_source")
    if data_source == "synthetic":
        lines.append(
            "- PROVENANCE: this model was fitted on generated data and carries NO domain "
            "signal. Do not cite this prediction as evidence about the real world."
        )
    imputed = list(_get(resp_like, "imputed_features", []) or [])
    if imputed:
        lines.append(
            f"- Imputed (not supplied by the caller, filled from training medians/modes): "
            f"{', '.join(str(name) for name in imputed)}. The answer partly reflects the "
            f"training set's centre on those inputs, not this case."
        )
    unknown = list(_get(resp_like, "unknown_features", []) or [])
    if unknown:
        lines.append(
            f"- Ignored (not model features): {', '.join(str(name) for name in unknown)}."
        )

    lines.append(
        "Use this prediction to prioritise and choose the right action. It informs the "
        "decision; it does not authorise one — the human gate fires on the tool's risk tier."
    )
    return "\n".join(lines)


_TEMPLATE = '''

def describe_prediction(resp: MLExplainResponse, *, top_k: int = 3) -> str:
    """Render an ML prediction as decision-support text for the agent's reasoning.

    This is the **domain** framing of the spine's output: it names the target
    ({target_name!r}) and its unit, states the calibrated interval and the
    confidence that was requested for it, and lists the strongest signed SHAP
    drivers as sentences in this domain's own language.

    It is reached through the adapter's ML tools, whose return value enters the
    agent's context:
{tool_hint}
    That routing is what makes this function live: the prediction, the conformal
    interval and the SHAP attribution are all computed by the spine regardless,
    and without a caller the sentence that puts them in front of the agent is
    never built.

    A categorical driver renders from ``ShapFeature.value_label``
    ("{example_level}"), never from ``value``, which for a one-hot column is the
    indicator ``1.0`` and names nothing.

    Args:
        resp: The spine response (prediction, conformal fields, SHAP attribution).
        top_k: How many top SHAP drivers to surface.

    Returns:
        A compact, human-readable multi-line summary safe to embed in a prompt.
    """
    unit = "{unit_literal}"
    prediction = resp.prediction
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        head = f"Predicted {target_name}: {{float(prediction):.1f}}{{unit}}"
    else:
        head = f"Predicted {target_name}: {{prediction}}"

    lines = ["ML decision-support ({task}):", f"- {{head}}"]

    if resp.conformal_interval is not None and resp.conformal_confidence is not None:
        low, high = resp.conformal_interval
        lines.append(
            f"- {{resp.conformal_confidence:.0%}} calibrated interval "
            f"[{{low:.1f}}, {{high:.1f}}]{{unit}} — this is the level REQUESTED; the "
            f"measured coverage is on the model card."
        )
    elif resp.prediction_set_size is not None and resp.conformal_confidence is not None:
        lines.append(
            f"- {{resp.conformal_confidence:.0%}} conformal set size "
            f"{{resp.prediction_set_size}} (1 = every other class excluded)"
        )

    drivers = []
    for feature in sorted(
        resp.shap_attribution, key=lambda f: abs(f.contribution), reverse=True
    )[:top_k]:
        shown = feature.value_label if feature.value_label else _format_driver(feature.value)
        {driver_body}
    if drivers:
        lines.append("- Why:")
        lines.extend(f"  - {{driver}}" for driver in drivers)

    if resp.data_source == "synthetic":
        lines.append(
            "- PROVENANCE: fitted on generated data; carries NO domain signal. Do not cite "
            "this prediction as evidence about the real world."
        )
    if resp.imputed_features:
        lines.append(
            "- Imputed (filled from training medians/modes): "
            + ", ".join(resp.imputed_features)
            + ". The answer partly reflects the training set's centre on those inputs."
        )
    if resp.unknown_features:
        lines.append("- Ignored (not model features): " + ", ".join(resp.unknown_features))

    lines.append(
        "Use this prediction to prioritise and choose the right action. It informs the "
        "decision; it does not authorise one."
    )
    return "\\n".join(lines)
'''

_HELPER = '''

def _format_driver(value: float) -> str:
    """Render a driver value without a trailing '.0' or padded zeros (3.2, not 3.20)."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"
'''


def emit_describe_prediction_source(
    problem: MLProblem,
    *,
    tools: Sequence[str] = ("predict_outcome", "explain_prediction", "whatif_scenario"),
) -> str:
    """Return the Python source of the adapter's ``describe_prediction``, ready to embed.

    The generated function is written against ``MLExplainResponse`` and this domain's target
    only — no import from ``aegis_ml``, because the adapter must keep working when this
    package is not installed alongside it. That is the same reason
    ``app.adapter.ml_spec`` imports no ML libraries: it has to load, and be tested, without
    numpy or xgboost present.

    The source is parsed with :func:`ast.parse` before it is returned. Generated code that
    does not compile breaks the adapter module's import, and a failed import of
    ``ml_spec`` is precisely the condition under which ``aegis.ml.spec.resolve_spec`` falls
    back to its four-column ``FALLBACK_SPEC`` **without raising** — training the trustworthy
    spine on generated noise and serving the result as domain evidence.

    Args:
        problem: The declared problem; supplies the target's name, task and unit, and the
            categorical level used in the docstring's example.
        tools: Names of the adapter tools that will call this function, quoted in the
            docstring so a reader can see what makes it live.

    Returns:
        Python source: a ``_format_driver`` helper followed by ``describe_prediction``.
        Both are top-level definitions with no imports of their own beyond the
        ``MLExplainResponse`` name the generated module already carries.

    Raises:
        SyntaxError: When the generated source does not parse. Raised rather than returned,
            because a caller that writes unparsable source into ``ml_spec.py`` produces a
            silent fallback to a noise spec.
    """
    target = problem.target
    example_level = "emea"
    for spec in problem.features:
        if spec.levels:
            example_level = str(spec.levels[0])
            break

    if target.task == "regression":
        unit_suffix = f" {target.unit}" if target.unit else ""
        driver_body = (
            'direction = "up" if feature.contribution >= 0 else "down"\n'
            "        drivers.append(\n"
            '            f"{feature.feature} = {shown} pushed the estimate {direction} by "\n'
            f'            f"{{_format_driver(abs(feature.contribution))}}{unit_suffix}"\n'
            "        )"
        )
    else:
        driver_body = (
            'toward = "toward" if feature.contribution >= 0 else "away from"\n'
            "        drivers.append(\n"
            '            f"{feature.feature} = {shown} pushed the prediction {toward} "\n'
            "            f\"'{resp.prediction}' (weight {abs(feature.contribution):.2f})\"\n"
            "        )"
        )

    source = _HELPER + _TEMPLATE.format(
        target_name=target.name,
        task=target.task,
        unit_literal=f" {target.unit}" if target.unit else "",
        example_level=example_level,
        tool_hint="\n".join(f"        * ``{name}``" for name in tools),
        driver_body=driver_body,
    )
    ast.parse(source)
    return source
