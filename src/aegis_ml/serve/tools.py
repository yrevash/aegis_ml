"""Ready-made adapter tools — how ML finally reaches the agent loop, with zero core edits.

Aegis has a trustworthy ML spine and, until now, nothing calls it during reasoning.
``aegis.ml.predict_explain`` has no consumers in the agent path; ``describe_prediction``
has none at all; ``ml_predict`` appears in the README's request diagram but not in
``graph.py``'s ``NODE_LABELS``. The evidence exists and never reaches the model.

The fix is **not** a new graph node. Decision D2: ML enters through *tools*. An adapter's
``TOOL_REGISTRY`` is already the agent's whole vocabulary, already allowlisted per persona,
already risk-tiered, already audited. Dropping five specs into it wires prediction,
explanation, what-if, forecasting and model health into the loop without touching a line of
``aegis/`` — and the platform's own rule is preserved exactly:

    **ML informs, it never gates. The human gate fires on a tool's risk tier.**

Which is why all five are ``LOW`` risk, ``read_only=True``, ``destructive=False`` and
``idempotent=True``, asserted per tool rather than inferred. They read a fitted model and
return numbers; none of them changes anything. Routing a *prediction* to a human approval
dialog would put the gate in front of the step that merely tells the planner what the model
thinks, while the write it was supposed to guard sails past — the gate belongs on the
action, and these are not actions.

**This package must not hard-depend on any one adapter's ``ToolSpec`` class.** A domain's
spec class lives in that domain's ``tools.py``, is a frozen dataclass in the reference
implementation and may be something else in the next one. So :func:`ml_tool_specs` builds
the specs *using the caller's own class*, matching its constructor by inspection and
refusing loudly if a required field cannot be supplied. Alongside it,
:data:`ML_TOOL_DEFINITIONS` is the plain OpenAI/MCP function-schema list for a host that
wants the definitions without the class, and every handler is an ordinary async function
usable on its own.

Handlers reach the model through :func:`aegis_ml._require.require`-loaded ``aegis.ml`` and
render their answers through :mod:`aegis_ml.explain.reason_codes` when it is present,
falling back to a sentence composed here from the response's own measured fields — the
interval, the requested coverage and the top signed SHAP drivers. That is a *presentation*
difference and it is stated in the payload (``reason_codes`` says why it is missing); no
number changes either way.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aegis_ml._require import require
from aegis_ml.contracts.spec import MLProblem

__all__ = [
    "ML_TOOL_DEFINITIONS",
    "ML_TOOL_NAMES",
    "ML_TOOL_RISK",
    "CheckModelHealthArgs",
    "ExplainPredictionArgs",
    "ForecastSeriesArgs",
    "MLToolResult",
    "PredictOutcomeArgs",
    "SeriesPointArg",
    "WhatIfScenarioArgs",
    "check_model_health",
    "explain_prediction",
    "forecast_series",
    "ml_tool_specs",
    "predict_outcome",
    "whatif_scenario",
]

AEGIS_EXTRA = "aegis"
"""Install target for the host platform; named verbatim in every ImportError."""

ML_TOOL_RISK = "low"
"""The risk tier every tool here carries.

Asserted, not derived. LOW means "cheap to get wrong", and these tools are additionally
read-only — two separate claims, both made explicitly, because ``add_case_note`` in the
reference adapter is LOW **and** writes. A tool registered without thinking about it is
advertised as a writer by default, and none of these is.
"""

ML_TOOL_NAMES = (
    "predict_outcome",
    "explain_prediction",
    "whatif_scenario",
    "forecast_series",
    "check_model_health",
)
"""The five tool names, in the order a planner naturally reaches for them."""

#: The literal command that fixes "no model available", quoted verbatim wherever that
#: condition is reported. Aegis's convention: an error names the exact remedy, and this is
#: the remedy — training through the *library's* constant writes to a path nothing loads
#: from, and the endpoints keep answering 503.
FIX_COMMAND = "python -m app.ml"

#: Entry points accepted from ``aegis_ml.explain.reason_codes``, in preference order. The
#: module is authored independently against a shared brief; naming every spelling this
#: module will call — and refusing rather than guessing when none is present — is the
#: difference between an explicit seam and a silent one.
_REASON_CODE_ENTRY_POINTS = (
    "render_reason_codes",
    "reason_codes",
    "describe_prediction",
    "render",
)


# ────────────────────────────────────────────────────────────────── argument models ──


class PredictOutcomeArgs(BaseModel):
    """Arguments for :func:`predict_outcome`."""

    features: dict[str, Any] = Field(
        description="Feature name → value for one case. Features you omit are imputed from "
        "training medians/modes and are listed back to you in `imputed_features`; keys the "
        "model does not know are ignored and listed in `unknown_features`. Check both "
        "before quoting the answer — a prediction built mostly from medians is a prediction "
        "about the average case, not about yours."
    )


class ExplainPredictionArgs(BaseModel):
    """Arguments for :func:`explain_prediction`."""

    features: dict[str, Any] = Field(description="Feature name → value for one case.")
    top_k: int = Field(
        default=5, ge=1, le=25, description="How many signed feature contributions to return."
    )


class WhatIfScenarioArgs(BaseModel):
    """Arguments for :func:`whatif_scenario`."""

    features: dict[str, Any] = Field(description="The baseline case, feature name → value.")
    changes: dict[str, Any] = Field(
        description="Feature name → new value. Only these are altered; everything else is "
        "held at the baseline, so the difference is attributable to the change."
    )


class SeriesPointArg(BaseModel):
    """One observed point of history for :func:`forecast_series`."""

    ts: datetime = Field(description="Observation timestamp, ISO-8601.")
    value: float = Field(description="Observed value at that timestamp.")


class ForecastSeriesArgs(BaseModel):
    """Arguments for :func:`forecast_series`."""

    points: list[SeriesPointArg] = Field(
        min_length=2,
        description="Observed history, any order. Duplicate timestamps are summed.",
    )
    label: str = Field(description="Human label for the series, e.g. 'Shipments per day'.")
    horizon: int = Field(default=14, ge=1, le=365, description="Steps to forecast ahead.")
    freq: str | None = Field(
        default=None, description="Frequency alias 'h'|'D'|'W'|'MS'; inferred when omitted."
    )
    unit: str | None = Field(default=None, description="Unit of the values, e.g. 'USD'.")
    level: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Coverage level to REQUEST. What is ACHIEVED is measured on held-out "
        "windows and returned separately as `empirical_coverage`.",
    )


class CheckModelHealthArgs(BaseModel):
    """Arguments for :func:`check_model_health` — deliberately none.

    A health check that takes parameters invites a planner to check the health of something
    it named itself. There is one served artifact; this reports on that one.
    """


class MLToolResult(BaseModel):
    """This package's own tool outcome, shaped like an adapter's ``ToolActionResult``.

    ``changed`` is always ``False`` and that is a structural fact, not a default: every tool
    here reads. :func:`ml_tool_specs` adapts this into the host's own result class when one
    is supplied, so a domain keeps its native type and this package keeps no dependency on it.
    """

    ok: bool = Field(description="Whether the tool produced an answer.")
    changed: bool = Field(default=False, description="Always False — every ML tool reads.")
    summary: str = Field(description="The answer as a human (and a planner) should read it.")
    data: dict[str, Any] = Field(
        default_factory=dict, description="The machine-readable payload behind the summary."
    )
    previous_state: dict[str, Any] = Field(
        default_factory=dict, description="Always empty; nothing here has prior state."
    )


# ─────────────────────────────────────────────────────────────────────── rendering ──


def _call_first(
    fn: Callable[..., Any],
    attempts: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
) -> Any:  # noqa: ANN401 - the callee's own return type
    """Call ``fn`` with the first argument arrangement it accepts.

    Only :class:`TypeError` — raised by argument binding *before* the body runs — advances
    to the next arrangement. Every other exception propagates: a callee that was reached and
    failed is a real failure, and retrying it with different arguments until something
    sticks would turn a bug into a coin toss.

    Args:
        fn: The callable to invoke.
        attempts: Ordered ``(args, kwargs)`` arrangements.

    Returns:
        The callee's return value.

    Raises:
        TypeError: If every arrangement was rejected.
    """
    last: TypeError | None = None
    for args, kwargs in attempts:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last = exc
    assert last is not None  # noqa: S101 - `attempts` is never empty at any call site
    raise last


def _compose_summary(response: Any, problem: MLProblem | None, top_k: int = 3) -> str:  # noqa: ANN401
    """Compose a decision-support sentence from an ``MLExplainResponse``'s own fields.

    Args:
        response: An ``aegis.ml.types.MLExplainResponse``.
        problem: The supervised problem, for the target's unit and human description.
        top_k: How many drivers to name.

    Returns:
        One or two sentences carrying the prediction, its calibrated interval, the coverage
        that was *requested*, and the top signed drivers.

    Every clause here is read off a measured field. In particular the coverage is introduced
    as "the level requested" and never as "confidence": ``conformal_confidence`` on the
    response is the level the model was calibrated *for*, and the level it *achieved* lives
    on the model card as ``conformal_coverage_empirical``. Collapsing the two into one
    confident-sounding percentage is the overclaim this platform exists to refuse.
    """
    unit = f" {problem.target.unit}" if problem and problem.target.unit else ""
    what = problem.target.description if problem else "the target"
    prediction = response.prediction
    head = (
        f"{what}: {prediction:.4g}{unit}"
        if isinstance(prediction, int | float)
        else f"{what}: {prediction}"
    )

    interval = getattr(response, "conformal_interval", None)
    level = getattr(response, "conformal_confidence", None)
    if interval is not None and level is not None:
        head += (
            f", within [{interval[0]:.4g}, {interval[1]:.4g}]{unit} at the "
            f"{level:.0%} coverage level that was requested"
        )
    set_size = getattr(response, "prediction_set_size", None)
    if set_size is not None:
        head += (
            f"; the conformal prediction set holds {set_size} class(es)"
            f"{' — a confident call' if set_size == 1 else ' — ambiguous'}"
        )

    drivers = list(getattr(response, "shap_attribution", []) or [])[:top_k]
    if drivers:
        rendered = ", ".join(
            f"{d.feature} ({d.contribution:+.3g})" for d in drivers if hasattr(d, "feature")
        )
        if rendered:
            head += f". Top drivers: {rendered}"

    imputed = list(getattr(response, "imputed_features", []) or [])
    if imputed:
        head += f". NOT supplied and imputed from training data: {', '.join(imputed)}"
    if getattr(response, "data_source", None) == "synthetic":
        head += (
            ". WARNING: this model was fitted on the built-in synthesiser and carries no "
            "domain signal — do not cite it as evidence"
        )
    return head + "."


def _reason_codes(response: Any, problem: MLProblem | None, top_k: int) -> dict[str, Any]:  # noqa: ANN401
    """Render reason codes through :mod:`aegis_ml.explain.reason_codes`, or say why not.

    Args:
        response: An ``aegis.ml.types.MLExplainResponse``.
        problem: The supervised problem.
        top_k: How many codes to request.

    Returns:
        ``{"codes": <rendered>}`` on success, or ``{"unavailable": "<reason>"}``. The second
        form is deliberate and visible in the tool payload: a missing *rendering* changes no
        number, but a caller must be able to see that the prose it is reading was composed
        from the raw response rather than by the reason-code module.
    """
    try:
        from aegis_ml.explain import reason_codes as module
    except ImportError as exc:
        return {"unavailable": f"aegis_ml.explain.reason_codes is not importable: {exc}"}

    for name in _REASON_CODE_ENTRY_POINTS:
        renderer = getattr(module, name, None)
        if renderer is None or not callable(renderer):
            continue
        rendered = _call_first(
            renderer,
            [
                ((response, problem), {"top_k": top_k}),
                ((response, problem), {}),
                ((response,), {"top_k": top_k}),
                ((response,), {}),
            ],
        )
        if isinstance(rendered, str):
            return {"codes": [rendered], "entry_point": name}
        return {"codes": list(rendered), "entry_point": name}

    return {
        "unavailable": (
            "aegis_ml.explain.reason_codes exposes none of "
            f"{_REASON_CODE_ENTRY_POINTS}; the summary below was composed from the "
            "response's own measured fields instead."
        )
    }


def _predict(features: dict[str, Any]) -> Any:  # noqa: ANN401 - an MLExplainResponse
    """Call ``aegis.ml.predict_explain``, letting its refusal propagate untouched.

    Args:
        features: Feature name → value for one case.

    Returns:
        An ``aegis.ml.types.MLExplainResponse``.

    Raises:
        MLModelUnavailableError: When no model has been trained or persisted. Caught by the
            handlers and turned into ``ok=False`` carrying the literal fix command, because
            "no model" is an answer a planner can act on and a stack trace is not.
        ImportError: When ``aegis`` is not installed, naming the install.
    """
    ml = require(AEGIS_EXTRA, "aegis.ml")
    return ml.predict_explain(features)


def _unavailable_error() -> type[Exception]:
    """Return ``aegis.ml.MLModelUnavailableError`` for a narrow ``except`` clause."""
    ml = require(AEGIS_EXTRA, "aegis.ml")
    return ml.MLModelUnavailableError


def _no_model_result(exc: Exception) -> MLToolResult:
    """Render "there is no served model" as a usable tool answer.

    Args:
        exc: The underlying refusal.

    Returns:
        A failed :class:`MLToolResult` naming the exact command that fixes it. The platform
        deliberately does not fall back to a model fitted on synthetic noise, so the honest
        answer here is "no evidence available", and the agent's ML step is best-effort: an
        omitted opinion is strictly better than a confident number with no signal in it.
    """
    return MLToolResult(
        ok=False,
        summary=(
            f"No trained model is available, so there is no ML evidence to offer. "
            f"Train one with `{FIX_COMMAND}`. The platform will not fit a model on "
            f"synthetic noise and serve its interval as calibrated evidence, so this "
            f"question has no answer until a real artifact exists. ({exc})"
        ),
        data={"error": str(exc), "fix": FIX_COMMAND},
    )


# ──────────────────────────────────────────────────────────────────────── handlers ──


async def predict_outcome(
    args: dict[str, Any],
    ctx: Any = None,  # noqa: ANN401, ARG001 - the domain's tool context; unused, read-only tool
    *,
    problem: MLProblem | None = None,
) -> MLToolResult:
    """Predict the modelled outcome for one case, with its calibrated interval.

    Args:
        args: Matches :class:`PredictOutcomeArgs`.
        ctx: The domain's tool context. Unused — this tool touches no record store and
            writes no audit row, which is exactly what ``read_only=True`` asserts.
        problem: The supervised problem, supplying the target's unit and description.

    Returns:
        An :class:`MLToolResult` whose ``data`` carries the point prediction, the conformal
        interval, the level that interval was calibrated *for*, the imputed and unknown
        feature lists, and the training-data provenance.
    """
    parsed = PredictOutcomeArgs.model_validate(args)
    try:
        response = await asyncio.to_thread(_predict, parsed.features)
    except Exception as exc:  # noqa: BLE001 - narrowed immediately below
        if isinstance(exc, _unavailable_error()):
            return _no_model_result(exc)
        raise
    payload = response.model_dump(mode="json")
    payload["fields_supplied"] = sorted(parsed.features)
    return MLToolResult(ok=True, summary=_compose_summary(response, problem), data=payload)


async def explain_prediction(
    args: dict[str, Any],
    ctx: Any = None,  # noqa: ANN401, ARG001 - the domain's tool context; unused, read-only tool
    *,
    problem: MLProblem | None = None,
) -> MLToolResult:
    """Predict, and return the signed per-feature contributions behind the number.

    Args:
        args: Matches :class:`ExplainPredictionArgs`.
        ctx: The domain's tool context. Unused; this tool reads.
        problem: The supervised problem.

    Returns:
        An :class:`MLToolResult` carrying the prediction, the interval, the top-``k`` signed
        SHAP contributions sorted by absolute size, and the reason codes.

    The contributions are **signed** and the sign is the whole point: "delay risk is high
    *because* the route has three transfers (+0.14) *despite* the carrier's on-time record
    (−0.06)" is a defensible sentence; a ranked list of unsigned importances is not.
    """
    parsed = ExplainPredictionArgs.model_validate(args)
    try:
        response = await asyncio.to_thread(_predict, parsed.features)
    except Exception as exc:  # noqa: BLE001 - narrowed immediately below
        if isinstance(exc, _unavailable_error()):
            return _no_model_result(exc)
        raise

    codes = await asyncio.to_thread(_reason_codes, response, problem, parsed.top_k)
    payload = response.model_dump(mode="json")
    payload["shap_attribution"] = payload.get("shap_attribution", [])[: parsed.top_k]
    payload["reason_codes"] = codes
    summary = _compose_summary(response, problem, top_k=parsed.top_k)
    if isinstance(codes.get("codes"), list) and codes["codes"]:
        summary = " ".join(str(code) for code in codes["codes"])
    return MLToolResult(ok=True, summary=summary, data=payload)


async def whatif_scenario(
    args: dict[str, Any],
    ctx: Any = None,  # noqa: ANN401, ARG001 - the domain's tool context; unused, read-only tool
    *,
    problem: MLProblem | None = None,
) -> MLToolResult:
    """Compare the model's answer for a baseline case against one with fields changed.

    Args:
        args: Matches :class:`WhatIfScenarioArgs`.
        ctx: The domain's tool context. Unused; this tool reads.
        problem: The supervised problem.

    Returns:
        An :class:`MLToolResult` carrying both predictions, both intervals, and the delta.

    The delta is the answer, and it comes with a caveat the payload states explicitly: this
    is the model's *associational* response to changing an input, not a causal effect. If
    the training data never contained a case with the new value, the answer is an
    extrapolation dressed as a scenario. ``intervals_overlap`` is reported for the same
    reason — a delta smaller than the interval width is not a distinguishable difference,
    and a planner that acts on it is acting on noise.
    """
    parsed = WhatIfScenarioArgs.model_validate(args)
    scenario = {**parsed.features, **parsed.changes}
    try:
        base = await asyncio.to_thread(_predict, parsed.features)
        after = await asyncio.to_thread(_predict, scenario)
    except Exception as exc:  # noqa: BLE001 - narrowed immediately below
        if isinstance(exc, _unavailable_error()):
            return _no_model_result(exc)
        raise

    unit = f" {problem.target.unit}" if problem and problem.target.unit else ""
    changed = ", ".join(
        f"{k}: {parsed.features.get(k, '<unset>')} → {v}" for k, v in parsed.changes.items()
    )
    numeric = isinstance(base.prediction, int | float) and isinstance(after.prediction, int | float)
    delta = float(after.prediction) - float(base.prediction) if numeric else None

    overlap: bool | None = None
    if base.conformal_interval and after.conformal_interval:
        overlap = not (
            after.conformal_interval[0] > base.conformal_interval[1]
            or after.conformal_interval[1] < base.conformal_interval[0]
        )

    if numeric:
        summary = (
            f"Changing {changed} moves the prediction from {float(base.prediction):.4g}"
            f"{unit} to {float(after.prediction):.4g}{unit} ({delta:+.4g}{unit})."
        )
        if overlap:
            summary += (
                " The two conformal intervals OVERLAP, so at the requested coverage level "
                "this change is not a distinguishable difference."
            )
    else:
        summary = (
            f"Changing {changed} moves the prediction from {base.prediction} "
            f"to {after.prediction}."
        )
    summary += (
        " This is the model's associational response to the altered input, not a causal "
        "effect: it says what the model predicts for a case like this, not what would "
        "happen if you made the change."
    )

    return MLToolResult(
        ok=True,
        summary=summary,
        data={
            "baseline": base.model_dump(mode="json"),
            "scenario": after.model_dump(mode="json"),
            "changes": parsed.changes,
            "delta": delta,
            "intervals_overlap": overlap,
            "causal": False,
        },
    )


async def forecast_series(
    args: dict[str, Any],
    ctx: Any = None,  # noqa: ANN401, ARG001 - the domain's tool context; unused, read-only tool
    *,
    problem: MLProblem | None = None,  # noqa: ARG001 - accepted for a uniform handler signature
) -> MLToolResult:
    """Forecast a time series, and report the coverage the band actually achieved.

    Args:
        args: Matches :class:`ForecastSeriesArgs`.
        ctx: The domain's tool context. Unused; this tool reads.
        problem: Accepted so every handler has one signature; unused here because a series
            is not the supervised target and carries its own label and unit.

    Returns:
        An :class:`MLToolResult` carrying the horizon points with their bounds, the selected
        model, every scored candidate including the losers, and both coverage numbers.

    Three refusals are surfaced as ``ok=False`` with their real reason rather than raised at
    the planner: too little history, a perfectly flat series, and total fit failure. Each is
    an answer the agent can use ("we have two weeks of ledger" is a finding), and none of
    them is a forecast.
    """
    parsed = ForecastSeriesArgs.model_validate(args)
    from aegis_ml.forecast.engine import forecast as run_forecast

    def _run() -> Any:  # noqa: ANN401 - a ForecastRun
        return run_forecast(
            [(p.ts, p.value) for p in parsed.points],
            parsed.label,
            horizon=parsed.horizon,
            data_source="tool",
            freq=parsed.freq,
            unit=parsed.unit,
            level=parsed.level,
        )

    forecast_errors = require("aegis[forecast]", "aegis.forecast.types")
    try:
        run = await asyncio.to_thread(_run)
    except forecast_errors.ForecastError as exc:
        return MLToolResult(
            ok=False,
            summary=(
                f"No forecast for {parsed.label!r}: {type(exc).__name__}: {exc} "
                f"Nothing was substituted — a naive line through this history would look "
                f"like a forecast and carry none of the guarantees of one."
            ),
            data={"error": f"{type(exc).__name__}: {exc}", "label": parsed.label},
        )

    first, last = run.points[0], run.points[-1]
    summary = (
        f"{run.label}: {run.model} forecasts {first.point:.4g} at {first.ts:%Y-%m-%d} "
        f"through {last.point:.4g} at {last.ts:%Y-%m-%d} over {run.horizon} step(s). "
        f"The {run.requested_coverage:.0%} band was REQUESTED; on {run.backtest_windows} "
        f"held-out rolling-origin windows it ACHIEVED {run.empirical_coverage:.1%}"
        f"{'' if run.coverage_meets_request else ' — short of the request, which is the finding'}."
        f" Measured sMAPE {run.smape:.2f}%."
    )
    return MLToolResult(ok=True, summary=summary, data=run.model_dump(mode="json"))


def _health_snapshot(domain_id: str | None) -> dict[str, Any]:
    """Gather the served model's health, without raising on any single missing piece.

    Args:
        domain_id: The domain whose champion to report on, or ``None`` to skip the registry.

    Returns:
        A dict of independently-resolved facts. Each key that could not be established
        carries an explicit ``"unavailable: <reason>"`` string rather than being omitted:
        a missing key reads as "not applicable", and "we could not find out" is a different
        answer that an operator needs to see.
    """
    snapshot: dict[str, Any] = {"fix_command": FIX_COMMAND}

    try:
        ml = require(AEGIS_EXTRA, "aegis.ml")
        model = ml.get_model()
        card = model.model_card()
        snapshot["served_model"] = (
            card.model_dump(mode="json") if hasattr(card, "model_dump") else str(card)
        )
        snapshot["model_available"] = True
    except Exception as exc:  # noqa: BLE001 - every failure mode is reported, not raised
        snapshot["model_available"] = False
        snapshot["served_model"] = f"unavailable: {type(exc).__name__}: {exc}"

    if domain_id:
        try:
            from aegis_ml.registry import store

            champion = store.champion(domain_id)
            if champion is None:
                snapshot["champion"] = "unavailable: no promoted run for this domain"
            else:
                snapshot["champion"] = {
                    "run_id": champion.run_id,
                    "created_at": champion.created_at,
                    "metric_name": champion.result.metric_name,
                    "metric_value": champion.result.metric_value,
                    "requested_coverage": champion.result.requested_coverage,
                    "empirical_coverage": champion.result.empirical_coverage,
                    "dataset_digest": champion.result.dataset_digest,
                }
                drift_path = Path(champion.paths.get("manifest", "")).parent / "drift.json"
                if drift_path.exists():
                    import json

                    drift = json.loads(drift_path.read_text(encoding="utf-8"))
                    snapshot["drift"] = {
                        "verdict": drift.get("verdict"),
                        "drifted_share": drift.get("drifted_share"),
                        "drifted_features": drift.get("drifted_features"),
                        "estimated_metric_name": drift.get("estimated_metric_name"),
                        "estimated_metric_value": drift.get("estimated_metric_value"),
                    }
                else:
                    snapshot["drift"] = "unavailable: no drift report recorded for this run"
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            snapshot["champion"] = f"unavailable: {type(exc).__name__}: {exc}"

    try:
        from aegis_ml.registry import promote as promote_mod

        snapshot["artifact"] = promote_mod.current_artifact_info()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        snapshot["artifact"] = f"unavailable: {type(exc).__name__}: {exc}"

    return snapshot


async def check_model_health(
    args: dict[str, Any],
    ctx: Any = None,  # noqa: ANN401, ARG001 - the domain's tool context; unused, read-only tool
    *,
    problem: MLProblem | None = None,
) -> MLToolResult:
    """Report whether there is a served model, what it measured, and whether data has drifted.

    Args:
        args: Matches :class:`CheckModelHealthArgs` (no fields).
        ctx: The domain's tool context. Unused; this tool reads.
        problem: The supervised problem, supplying the domain id for the registry lookup.

    Returns:
        An :class:`MLToolResult` carrying the served model's card, the champion run's
        measured metric and coverage, the latest drift verdict, and the artifact's location
        and age.

    This is the tool a planner should reach for *before* citing a prediction, and the one an
    operator reaches for when the answers look odd. It reports the drift verdict without
    withdrawing anything: Aegis serves the model it has and flags it, because an outage in
    the evidence channel is worse than degraded evidence that says it is degraded.
    """
    CheckModelHealthArgs.model_validate(args)
    snapshot = await asyncio.to_thread(_health_snapshot, problem.domain_id if problem else None)

    if not snapshot.get("model_available"):
        return MLToolResult(
            ok=False,
            summary=(
                f"No model is currently served. Train one with `{FIX_COMMAND}`. Until then "
                f"every ML tool will decline rather than return a number with no signal in it."
            ),
            data=snapshot,
        )

    champion = snapshot.get("champion")
    parts = ["A trained model is served."]
    if isinstance(champion, dict):
        coverage = champion.get("empirical_coverage")
        parts.append(
            f"Champion run {champion['run_id']} measured {champion['metric_name']}="
            f"{champion['metric_value']:.4g} on its held-out split, requested "
            f"{champion['requested_coverage']:.0%} coverage and achieved "
            + (f"{coverage:.1%}." if isinstance(coverage, float) else "an unmeasured rate.")
        )
    drift = snapshot.get("drift")
    if isinstance(drift, dict) and drift.get("verdict"):
        parts.append(
            f"Latest drift verdict: {drift['verdict']} "
            f"({float(drift.get('drifted_share') or 0.0):.0%} of features moved against the "
            f"reference the model was calibrated on)."
        )
        if drift.get("estimated_metric_name"):
            parts.append(
                f"Label-free performance ESTIMATE: {drift['estimated_metric_name']} ≈ "
                f"{drift['estimated_metric_value']} — an estimate, not a measurement."
            )
    return MLToolResult(ok=True, summary=" ".join(parts), data=snapshot)


# ────────────────────────────────────────────────────────────── definitions & specs ──


def _definition(name: str, description: str, args_model: type[BaseModel]) -> dict[str, Any]:
    """Build one OpenAI/MCP ``{"type": "function", ...}`` schema from a pydantic model."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": args_model.model_json_schema(),
        },
    }


_DESCRIPTIONS: dict[str, str] = {
    "predict_outcome": (
        "Predict the modelled outcome for one case and return it with a calibrated "
        "conformal interval. Read-only: it changes nothing. The interval is the answer's "
        "uncertainty, not a decoration — quote it whenever you quote the number. Features "
        "you omit are imputed from training data and listed back to you; a prediction "
        "assembled mostly from imputed values describes the average case, not this one. "
        "This is evidence for a recommendation, never an authorisation to act."
    ),
    "explain_prediction": (
        "Predict, and return the signed per-feature contributions that produced the number, "
        "largest absolute effect first. Read-only. Use this when a human will be asked to "
        "act on the prediction: a signed driver list supports 'high risk BECAUSE x, DESPITE "
        "y', which an unexplained score cannot. The contributions are attributions of this "
        "model's behaviour, not statements about the world."
    ),
    "whatif_scenario": (
        "Predict twice — once for a baseline case, once with named fields changed — and "
        "return both answers, both intervals and the delta. Read-only; nothing is written. "
        "Use it to compare options before recommending one. The delta is associational, not "
        "causal: it is what the model predicts for a case like the altered one, not what "
        "would happen if you made the change. If the two intervals overlap, the difference "
        "is not distinguishable at the requested coverage level and should not be acted on."
    ),
    "forecast_series": (
        "Forecast a time series over a horizon and return each step with its interval "
        "bounds, plus the coverage the band ACHIEVED on held-out rolling-origin windows "
        "against the coverage that was REQUESTED. Read-only. It refuses rather than "
        "guessing: too little history, a perfectly flat series, or a total fit failure come "
        "back as an explained 'no forecast', which is a usable finding."
    ),
    "check_model_health": (
        "Report whether a trained model is being served, what it measured on its held-out "
        "split, whether live data has drifted away from the data it was calibrated on, and "
        "how old the served artifact is. Read-only, takes no arguments. Call this before "
        "citing any prediction as evidence in a decision a human will be asked to approve."
    ),
}

_ARG_MODELS: dict[str, type[BaseModel]] = {
    "predict_outcome": PredictOutcomeArgs,
    "explain_prediction": ExplainPredictionArgs,
    "whatif_scenario": WhatIfScenarioArgs,
    "forecast_series": ForecastSeriesArgs,
    "check_model_health": CheckModelHealthArgs,
}

_HANDLERS: dict[str, Callable[..., Any]] = {
    "predict_outcome": predict_outcome,
    "explain_prediction": explain_prediction,
    "whatif_scenario": whatif_scenario,
    "forecast_series": forecast_series,
    "check_model_health": check_model_health,
}

ML_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _definition(name, _DESCRIPTIONS[name], _ARG_MODELS[name]) for name in ML_TOOL_NAMES
]
"""Plain OpenAI/MCP function schemas for a host that wants the definitions without a class.

Usable directly as the ``tools=`` payload to a chat completion, or as the ``tools/list``
response of an MCP server. The handlers in :data:`ML_TOOL_NAMES` order are importable from
this module and take ``(args: dict, ctx)`` — the same signature the reference adapter's
``ToolHandler`` protocol declares.
"""


def _risk_low(explicit: Any = None) -> Any:  # noqa: ANN401 - the host's own risk enum member
    """Resolve the LOW risk value to stamp on every spec.

    Args:
        explicit: A value supplied by the caller, used as-is when given.

    Returns:
        ``aegis.core.types.RiskLevel.LOW`` when the host platform is importable, otherwise
        the plain string ``"low"`` — which is what that enum member's value is
        (:class:`RiskLevel` is a ``StrEnum``), so a host comparing against its own enum
        still matches.

    Raises:
        Nothing. Risk is the one field that must never be absent — a tool registered without
        one is treated as HIGH by the platform, which would route a *prediction* to a human
        approval dialog and put the gate in front of the wrong step entirely.
    """
    if explicit is not None:
        return explicit
    try:
        from aegis_ml._require import require as _require

        return _require(AEGIS_EXTRA, "aegis.core.types").RiskLevel.LOW
    except ImportError:
        return ML_TOOL_RISK


def ml_tool_specs(
    spec_cls: type,
    *,
    problem: MLProblem | None = None,
    risk_low: Any = None,  # noqa: ANN401 - the host's own risk enum member
    result_cls: type | None = None,
    names: Sequence[str] = ML_TOOL_NAMES,
) -> dict[str, Any]:
    """Build ML tool specs **using the caller's own spec class**, ready for ``TOOL_REGISTRY``.

    Usage in a domain adapter's ``tools.py``::

        from aegis_ml.serve.tools import ml_tool_specs
        from app.adapter.ml_spec import PROBLEM

        TOOL_REGISTRY: dict[str, ToolSpec] = {
            **{...the domain's own tools...},
            **ml_tool_specs(ToolSpec, problem=PROBLEM, result_cls=ToolActionResult),
        }
        ALLOWLIST = {"analyst": frozenset({..., *ml_tool_specs(ToolSpec, problem=PROBLEM)})}

    Args:
        spec_cls: The domain's own tool-spec class. Its constructor is inspected and only
            parameters it actually declares are passed, so a class with extra or
            differently-named fields still works.
        problem: The supervised problem. Bound into every handler so the summaries carry the
            target's unit and description rather than bare floats.
        risk_low: The host's LOW risk value. Resolved from ``aegis.core.types`` when omitted.
        result_cls: The host's tool-result class (``ToolActionResult`` in the reference
            adapter). When given, each handler's :class:`MLToolResult` is adapted into it by
            keyword, so the domain keeps its native type and this package keeps no
            dependency on it. When omitted the handlers return :class:`MLToolResult`.
        names: Which tools to build; defaults to all five.

    Returns:
        ``{tool_name: spec_cls instance}``, ready to merge into ``TOOL_REGISTRY``.

    Raises:
        ValueError: If ``spec_cls`` requires a constructor parameter this function cannot
            supply. Refusing beats constructing a half-populated spec: the field most likely
            to be missing is ``risk``, and a tool with no risk tier is an ungated action.
    """
    parameters = inspect.signature(spec_cls).parameters
    risk_value = _risk_low(risk_low)

    def build(name: str) -> Any:  # noqa: ANN401 - an instance of the caller's spec class
        args_model = _ARG_MODELS[name]
        handler = _bind_handler(_HANDLERS[name], problem=problem, result_cls=result_cls)
        available: dict[str, Any] = {
            "name": name,
            "description": _DESCRIPTIONS[name],
            "args_model": args_model,
            "parameters_model": args_model,
            "schema_model": args_model,
            "parameters": args_model.model_json_schema(),
            "definition": _definition(name, _DESCRIPTIONS[name], args_model),
            "handler": handler,
            "fn": handler,
            "func": handler,
            "risk": risk_value,
            "risk_level": risk_value,
            # Asserted per tool, never derived from the tier: every ML tool reads a fitted
            # model and returns numbers. read_only says it changes nothing; idempotent says
            # repeating it converges; destructive says it overwrites nothing a reader would
            # miss. All three are true here and all three are stated.
            "read_only": True,
            "readonly": True,
            "destructive": False,
            "idempotent": True,
        }
        kwargs = {key: value for key, value in available.items() if key in parameters}
        missing = [
            key
            for key, param in parameters.items()
            if key not in kwargs
            and param.default is inspect.Parameter.empty
            and param.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if missing:
            raise ValueError(
                f"cannot build an ML tool spec with {spec_cls.__name__}: it requires "
                f"{missing}, which aegis_ml.serve.tools cannot supply. Construct these "
                f"tools by hand rather than accepting a half-populated spec — the field "
                f"most often missing is `risk`, and a tool registered without a risk tier "
                f"is treated as HIGH by the platform, which routes a read-only prediction "
                f"to the human approval gate."
            )
        return spec_cls(**kwargs)

    return {name: build(name) for name in names}


def _bind_handler(
    handler: Callable[..., Any],
    *,
    problem: MLProblem | None,
    result_cls: type | None,
) -> Callable[..., Any]:
    """Bind ``problem`` into a handler and adapt its result into the host's result class.

    Args:
        handler: One of this module's async handlers.
        problem: The supervised problem to bind.
        result_cls: The host's result class, or ``None`` to return :class:`MLToolResult`.

    Returns:
        An ``async (args, ctx) -> result`` callable with the handler's name and docstring —
        the host reads ``__doc__`` and ``__name__`` off registered handlers.

    The adaptation is by keyword over the intersection of the two shapes, so a result class
    with extra fields keeps its own defaults for them and one with fewer does not receive
    arguments it cannot take. ``data`` is dropped when the target class has no field for it,
    which is why the summary carries every number a reader needs in prose as well.
    """
    import functools

    @functools.wraps(handler)
    async def bound(args: dict[str, Any], ctx: Any = None) -> Any:  # noqa: ANN401
        outcome = await handler(args, ctx, problem=problem)
        if result_cls is None:
            return outcome
        fields = getattr(result_cls, "model_fields", None)
        payload = outcome.model_dump()
        if fields is not None:
            payload = {key: value for key, value in payload.items() if key in fields}
        else:
            accepted = inspect.signature(result_cls).parameters
            payload = {key: value for key, value in payload.items() if key in accepted}
        return result_cls(**payload)

    return bound
