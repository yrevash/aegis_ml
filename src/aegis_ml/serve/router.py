"""An optional FastAPI router a host may include — the MLOps console's data source.

Nothing in this package requires it. The filesystem registry is the source of truth, the
CLI reads the same files, and a host that never mounts this router loses no capability. It
exists because the Aegis console already has an ML view and giving it real endpoints costs
one ``include_router`` call.

Two of Aegis's own conventions are followed exactly, because a host mounting this must not
have to learn a second dialect:

**503 with the literal fix command.** ``backend/src/app/ml/__init__.py`` answers a
model-less request with a 503 naming ``python -m app.ml``. So does every endpoint here.
Aegis's rule is that an error carries its own remedy — a 503 saying "model unavailable"
sends an operator to the logs, and a 503 saying "run ``python -m app.ml``" sends them to
the fix.

**CPU work goes off the event loop.** Loading a model, reading a parquet reference frame
and scoring a what-if are all blocking. Run on the loop they stall every other request in
the process, and the symptom is a slow *unrelated* endpoint, which is the hardest kind of
performance bug to attribute. Everything here goes through ``asyncio.to_thread``.

The router is read-only apart from ``POST /ml/whatif``, and that POST writes nothing — it
is a POST because a scenario body does not fit in a query string, not because it mutates.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from aegis_ml._require import require
from aegis_ml.settings import settings

__all__ = ["FIX_COMMAND", "build_router", "router"]

FASTAPI_EXTRA = "fastapi"
"""Install target for FastAPI; named verbatim in the ImportError if it is missing."""

FIX_COMMAND = "python -m app.ml"
"""The literal command every 503 in this module names. See the module docstring."""


def _no_model_detail(reason: str) -> dict[str, str]:
    """Build the 503 body: what is missing, and the exact command that fixes it."""
    return {
        "error": "ml_model_unavailable",
        "reason": reason,
        "fix": FIX_COMMAND,
        "detail": (
            f"No trained ML artifact is available at {settings.artifact_path}. Train one "
            f"with `{FIX_COMMAND}`. The platform will not fit a model on synthetic noise "
            f"and serve its interval as calibrated evidence, so this endpoint refuses "
            f"rather than answering."
        ),
    }


def _list_runs(domain_id: str | None, limit: int) -> list[dict[str, Any]]:
    """Read the registry, newest first. Blocking; always called through a thread."""
    from aegis_ml.registry import store

    entries = store.list_runs(domain_id) if domain_id else store.list_runs()
    rows: list[dict[str, Any]] = []
    for entry in list(entries)[:limit]:
        result = entry.result
        rows.append(
            {
                "run_id": entry.run_id,
                "domain_id": entry.domain_id,
                "created_at": entry.created_at,
                "stage": entry.stage,
                "task": result.task,
                "target": result.target,
                "metric_name": result.metric_name,
                "metric_value": result.metric_value,
                # Two fields, always. One "coverage" number would mean whichever the
                # console's author assumed, and the console is where a judge reads it.
                "requested_coverage": result.requested_coverage,
                "empirical_coverage": result.empirical_coverage,
                "training_size": result.training_size,
                "test_size": result.test_size,
                "dataset_digest": result.dataset_digest,
                "tier": result.recipe.tier if result.recipe else None,
                "promoted": entry.stage == "production",
            }
        )
    return rows


def _load_card(run_id: str, want_html: bool) -> tuple[str, str] | dict[str, Any]:
    """Return a run's model card, as ``(content_type, text)`` for HTML or a dict for JSON.

    Blocking (reads files, may render); always called through a thread.
    """
    from aegis_ml.registry import store

    entry = store.load_entry(run_id)
    key = "card_html" if want_html else "card_md"
    path = entry.paths.get(key)
    if want_html:
        if path and Path(path).exists():
            return ("text/html", Path(path).read_text(encoding="utf-8"))
        from aegis_ml.explain import card as card_mod

        return ("text/html", str(card_mod.render_html(card_mod.build_card(entry.result))))
    payload = entry.result.model_dump(mode="json")
    payload["stage"] = entry.stage
    payload["created_at"] = entry.created_at
    payload["paths"] = entry.paths
    if entry.gate is not None:
        payload["gate"] = entry.gate.model_dump(mode="json")
    if path and Path(path).exists():
        payload["card_markdown"] = Path(path).read_text(encoding="utf-8")
    return payload


def _leaderboard(run_id: str | None, domain_id: str | None) -> dict[str, Any]:
    """Return one run's leaderboard, or the champion's. Blocking; called through a thread."""
    from aegis_ml.registry import store

    if run_id:
        entry = store.load_entry(run_id)
    else:
        entry = store.champion(domain_id) if domain_id else None
        if entry is None:
            raise LookupError(
                "no promoted run to read a leaderboard from; pass run_id, or promote a run"
            )
    board = entry.result.leaderboard
    if board is None:
        raise LookupError(
            f"run {entry.run_id} recorded no leaderboard — it was resumed or fitted from a "
            f"supplied recipe rather than searched"
        )
    return {
        "run_id": entry.run_id,
        "domain_id": entry.domain_id,
        "leaderboard": board.model_dump(mode="json"),
        # Skipped tiers are part of the answer: an empty slot and an unavailable dependency
        # look identical on a chart, and one of them means "install it".
        "tiers_skipped": board.tiers_skipped,
    }


def _drift(run_id: str) -> dict[str, Any]:
    """Return the last recorded drift report for a run. Blocking; called through a thread."""
    from aegis_ml.registry import store

    entry = store.load_entry(run_id)
    path = Path(store.run_dir(run_id)) / "drift.json"
    if not path.exists():
        raise LookupError(
            f"run {run_id} has no recorded drift report. Run `aegis-ml drift --run-id "
            f"{run_id} --data <frame>` — drift is a comparison against the frozen reference "
            f"frame, and it is not computed until someone supplies the current data."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["domain_id"] = entry.domain_id
    payload["reference_frame"] = entry.paths.get("reference_frame")
    return payload


def _health() -> dict[str, Any]:
    """Assemble the health payload. Blocking; always called through a thread."""
    from aegis_ml.serve.tools import _health_snapshot

    return _health_snapshot(None)


def build_router(*, prefix: str = "/ml", tags: list[str] | None = None) -> Any:  # noqa: ANN401
    """Build the ML ``APIRouter``.

    Args:
        prefix: URL prefix; the host mounts it with ``app.include_router(build_router())``.
        tags: OpenAPI tags; defaults to ``["ml"]``.

    Returns:
        A ``fastapi.APIRouter`` exposing:

        * ``GET  /ml/registry`` — registered runs, newest first
        * ``GET  /ml/runs/{run_id}/card`` — the model card as JSON, or HTML with ``?format=html``
        * ``GET  /ml/leaderboard`` — a run's AutoML leaderboard, losers and skipped tiers included
        * ``GET  /ml/drift/{run_id}`` — the last recorded drift report
        * ``POST /ml/whatif`` — baseline vs altered prediction, with both intervals
        * ``GET  /ml/health`` — is a model served, what did it measure, has data drifted

    Raises:
        ImportError: If FastAPI is not installed, naming the install command.

    The router is built by a function rather than declared at module import so that
    importing :mod:`aegis_ml.serve.router` never requires FastAPI. A host that wants the
    module-level singleton can use :data:`router`, which is created on first attribute
    access for the same reason.
    """
    fastapi = require(FASTAPI_EXTRA, "fastapi")
    responses = require(FASTAPI_EXTRA, "fastapi.responses")
    api = fastapi.APIRouter(prefix=prefix, tags=tags or ["ml"])

    def _unavailable(exc: Exception) -> Any:  # noqa: ANN401 - an HTTPException
        return fastapi.HTTPException(status_code=503, detail=_no_model_detail(str(exc)))

    @api.get("/registry", summary="List registered training runs, newest first")
    async def registry(
        domain_id: str | None = None,
        limit: int = fastapi.Query(default=25, ge=1, le=200),
    ) -> dict[str, Any]:
        """Return the registry rows the MLOps console renders.

        Args:
            domain_id: Restrict to one domain.
            limit: Maximum rows.

        Returns:
            ``{"runs": [...], "count": n}``. Every row carries both coverage numbers.

        Raises:
            HTTPException: 503 when the registry cannot be read at all.
        """
        try:
            rows = await asyncio.to_thread(_list_runs, domain_id, limit)
        except Exception as exc:  # noqa: BLE001 - rendered as a 503 with the fix command
            raise _unavailable(exc) from exc
        return {"runs": rows, "count": len(rows), "registry_dir": str(settings.registry_dir)}

    @api.get("/runs/{run_id}/card", summary="One run's model card, as JSON or HTML")
    async def card(run_id: str, format: str = "json") -> Any:  # noqa: A002, ANN401
        """Return the model card for ``run_id``.

        Args:
            run_id: The registered run.
            format: ``"json"`` (default) or ``"html"``.

        Returns:
            The card payload, or an ``HTMLResponse``.

        Raises:
            HTTPException: 404 when the run is unknown, 503 when the card cannot be built.
        """
        try:
            loaded = await asyncio.to_thread(_load_card, run_id, format.lower() == "html")
        except (FileNotFoundError, KeyError, LookupError) as exc:
            raise fastapi.HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - rendered as a 503 with the fix command
            raise _unavailable(exc) from exc
        if isinstance(loaded, tuple):
            return responses.HTMLResponse(content=loaded[1])
        return loaded

    @api.get("/leaderboard", summary="The AutoML leaderboard, losers and skipped tiers included")
    async def leaderboard(run_id: str | None = None, domain_id: str | None = None) -> dict[str, Any]:
        """Return a run's leaderboard, or the champion's when no run is named.

        Args:
            run_id: A specific run.
            domain_id: Fall back to this domain's champion.

        Returns:
            The leaderboard with every candidate, winner and loser, plus skipped tiers.

        Raises:
            HTTPException: 404 when there is no leaderboard to return.
        """
        try:
            return await asyncio.to_thread(_leaderboard, run_id, domain_id)
        except (FileNotFoundError, KeyError, LookupError) as exc:
            raise fastapi.HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - rendered as a 503 with the fix command
            raise _unavailable(exc) from exc

    @api.get("/drift/{run_id}", summary="The last recorded drift report for a run")
    async def drift(run_id: str) -> dict[str, Any]:
        """Return the drift report recorded for ``run_id``.

        Args:
            run_id: The registered run.

        Returns:
            The drift payload, including the label-free performance ESTIMATE when one was
            computed — named ``estimated_*`` throughout so it is never read as a measurement.

        Raises:
            HTTPException: 404 when no drift report has been recorded for the run.
        """
        try:
            return await asyncio.to_thread(_drift, run_id)
        except (FileNotFoundError, KeyError, LookupError) as exc:
            raise fastapi.HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - rendered as a 503 with the fix command
            raise _unavailable(exc) from exc

    @api.post("/whatif", summary="Baseline vs altered prediction, with both intervals")
    async def whatif(payload: dict[str, Any]) -> dict[str, Any]:
        """Score a baseline case and an altered one, and return the difference.

        Args:
            payload: ``{"features": {...}, "changes": {...}}``.

        Returns:
            The tool result: both predictions, both conformal intervals, the delta, and
            whether the intervals overlap — because a delta smaller than the interval width
            is not a distinguishable difference.

        Raises:
            HTTPException: 422 on a malformed body, 503 when no model is served.

        A POST that writes nothing. The verb is dictated by the body, not by any mutation:
        this endpoint is as read-only as the GETs beside it.
        """
        from aegis_ml.serve.tools import whatif_scenario

        result = await whatif_scenario(payload)
        if not result.ok:
            raise fastapi.HTTPException(
                status_code=503, detail={**_no_model_detail(result.summary), "fix": FIX_COMMAND}
            )
        return result.model_dump(mode="json")

    @api.get("/health", summary="Is a model served, what did it measure, has data drifted")
    async def health() -> dict[str, Any]:
        """Return the served model's health snapshot.

        Returns:
            The served model's card, the champion's measured metric and coverage, the drift
            verdict and the artifact's location.

        Raises:
            HTTPException: 503 when no model is served, naming the exact training command.

        A drifted model still answers 200 here with its verdict attached. Aegis serves the
        model it has and flags it: withdrawing the evidence channel because the evidence got
        worse leaves a decision to be made with none at all.
        """
        snapshot = await asyncio.to_thread(_health)
        if not snapshot.get("model_available"):
            raise fastapi.HTTPException(
                status_code=503,
                detail={**_no_model_detail(str(snapshot.get("served_model"))), **snapshot},
            )
        return snapshot

    return api


def __getattr__(name: str) -> Any:  # noqa: ANN401 - the lazily-built router singleton
    """Build the module-level ``router`` on first access, so importing needs no FastAPI.

    Args:
        name: Attribute being accessed.

    Returns:
        The built ``APIRouter`` for ``router``.

    Raises:
        AttributeError: For any other name.
        ImportError: From :func:`build_router` when FastAPI is absent.
    """
    if name == "router":
        built = build_router()
        globals()["router"] = built
        return built
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
