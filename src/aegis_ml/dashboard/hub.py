"""Build the hub page: the landing view that reads the registry and says what happened.

Every figure on this page is read out of a file some pipeline stage already wrote. There is
no summarisation step, no derived score and no default value — where an artifact is absent
the page shows an em dash carrying the reason as its tooltip, or an empty panel naming the
command that produces the missing file. That rule is the reason this module is a *reader*
rather than a template: a number the hub could compute on its own is a number that can
disagree with the model card, and on a projector the hub is the one people believe.

The layout answers three questions in the order a viewer asks them:

1. **Should I trust the latest model?** — the verdict tiles, above the fold: the primary
   metric, the coverage that was *requested* next to the coverage that was *measured*, the
   gate's PASS/FAIL, the drift verdict, the winning tier, and which run is actually serving.
2. **What is the evidence?** — the gate's five criteria in its own words, then the nine
   figures the run rendered, in a gallery with a lightbox, then the leaderboard that shows
   what lost as well as what won.
3. **What else has been run?** — the run list, a two-run comparison, and the two premade
   UIs (MLflow, Optuna Dashboard) embedded next to it all.

The page is one file. The CSS is inlined from :mod:`aegis_ml.dashboard.theme`, the data is
inlined as a JSON island, and the interactive layer is inline vanilla JavaScript with a
hash router. No bundler, no framework, no CDN, no web font, no network request of any kind
— the machine this is demoed from may have its wifi off, and a dashboard that needs the
internet to describe an offline pipeline would be an odd thing to show.

Static run artifacts — the PNGs, ``card.html``, ``shap.html``, ``profile.html``,
``interactive.html``, the drift report — are *not* inlined. They total tens of megabytes
per run; :mod:`aegis_ml.dashboard.server` serves them from the run directory instead, which
also means the hub shows the same bytes the registry holds rather than a copy of them.
"""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml.dashboard import theme

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from aegis_ml.contracts.protocols import RegistryEntry

__all__ = ["DOC_SLOTS", "collect", "render", "run_view"]

DOC_SLOTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "bundle",
        "visuals/index.html",
        "Report bundle",
        "All nine figures with their captions, in one self-contained page.",
    ),
    (
        "interactive",
        "visuals/interactive.html",
        "Interactive",
        "Plotly panels — hover a point to read the row behind it.",
    ),
    (
        "card",
        "card.html",
        "Model card",
        "What was trained, on what, with which limits and which licence.",
    ),
    (
        "shap",
        "shap.html",
        "SHAP",
        "Global attribution, and the explainer that produced it.",
    ),
    (
        "profile",
        "profile.html",
        "Data profile",
        "Per-column distributions, missingness and cardinality of the reference frame.",
    ),
    (
        "card_md",
        "card.md",
        "Card (markdown)",
        "The same card as text, for pasting into a review.",
    ),
    (
        "leaderboard",
        "leaderboard.json",
        "Leaderboard JSON",
        "Every candidate, its score and why it was or was not portable.",
    ),
    (
        "entry",
        "entry.json",
        "Registry entry",
        "The registry row itself — the source every number here is read from.",
    ),
)
"""``(slot, path relative to the run directory, label, one-line note)``.

Ordered by how often a viewer opens them, because the tab strip is read left to right and
the model card is what a reviewer asks for first. The drift report is not in this list: it
lives under ``registry_store/reports/`` rather than the run directory, and is appended
separately by :func:`run_view` only when the run actually has one.
"""

_SLOT_TITLES: dict[str, str] = {
    "prediction_vs_actual": "Prediction vs measured",
    "residuals": "Residuals",
    "conformal_coverage": "Conformal coverage",
    "shap_global": "Global attribution",
    "slice_performance": "Slice performance",
    "leaderboard": "Leaderboard",
    "realism": "Realism band",
    "feature_distributions": "Feature distributions",
    "drift_features": "Drifted features",
    "forecast": "Forecast",
    "calibration": "Calibration",
    "confusion": "Confusion matrix",
}
"""Short titles for the gallery, used only when a run predates ``visuals/manifest.json``.

When the manifest is present its own ``title`` and ``caption`` win — they were written next
to the code that drew the figure and they describe what that specific figure shows.
"""


def _read_json(path: Path) -> Any:  # noqa: ANN401 - JSON is genuinely any shape
    """Return parsed JSON, or ``None`` when the file is absent or unreadable.

    Args:
        path: The file to read.

    Returns:
        The decoded document, or ``None``.

    A missing or corrupt artifact is a fact the page reports, not an error that stops it:
    one run with a truncated ``drift.json`` must not blank the whole dashboard. The absence
    surfaces in the view as a named empty state, so nothing is swallowed.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _title_from_file(name: str) -> str:
    """Turn ``05_slice_performance.png`` into a readable title.

    Args:
        name: The PNG file name.

    Returns:
        A title from :data:`_SLOT_TITLES` when the slot is known, otherwise the file stem
        with its ordering prefix and underscores removed.
    """
    stem = re.sub(r"^\d+_", "", Path(name).stem)
    if stem in _SLOT_TITLES:
        return _SLOT_TITLES[stem]
    return stem.replace("_", " ").capitalize()


def _worst_slice(entry: RegistryEntry, higher_is_better: bool) -> dict[str, Any] | None:
    """Return the worst-performing evaluated segment, or ``None`` if none were evaluated.

    Args:
        entry: The registry row, whose ``result.slices`` holds the per-segment scores.
        higher_is_better: Direction of the metric, taken from the leaderboard rather than
            assumed — ``rmse`` and ``r2`` disagree, and picking the wrong end silently
            reports the *best* slice as the worst.

    Returns:
        ``{"feature", "level", "n_rows", "metric_name", "metric_value"}`` for the extreme
        segment, or ``None``.

    This is the number the promotion gate holds a challenger to, and the one that catches a
    model with a fine headline score that has quietly given up on one carrier tier.
    """
    slices = list(entry.result.slices)
    if not slices:
        return None
    chosen = (min if higher_is_better else max)(slices, key=lambda s: s.metric_value)
    return {
        "feature": chosen.feature,
        "level": chosen.level,
        "n_rows": chosen.n_rows,
        "metric_name": chosen.metric_name,
        "metric_value": chosen.metric_value,
    }


def _figures(run_dir: Path, run_id: str) -> tuple[list[dict[str, Any]], str | None]:
    """Collect the run's rendered figures with their captions.

    Args:
        run_dir: The run directory.
        run_id: Used to build the URLs the server resolves back to ``run_dir``.

    Returns:
        ``(figures, reason)``. ``reason`` is non-``None`` only when the list is empty, and
        then it says which command renders them.

    The manifest is preferred because it records *why* each figure was or was not drawn.
    A slot the run deliberately omitted (no forecast payload for a tabular model, say) is
    carried through as an omission with its stated reason rather than dropped, so the
    gallery cannot imply a figure was never asked for.
    """
    visuals = run_dir / "visuals"
    base = f"/runs/{run_id}/files/visuals"
    manifest = _read_json(visuals / "manifest.json")
    figures: list[dict[str, Any]] = []
    if isinstance(manifest, dict) and isinstance(manifest.get("plots"), list):
        for plot in manifest["plots"]:
            if not isinstance(plot, dict):
                continue
            name = plot.get("file")
            rendered = plot.get("status") == "rendered" and bool(name)
            figures.append(
                {
                    "slot": plot.get("slot", ""),
                    "title": plot.get("title") or _title_from_file(str(name or "")),
                    "caption": plot.get("caption") or "",
                    "url": f"{base}/{name}" if rendered else None,
                    "rendered": rendered,
                    "reason": plot.get("reason") or "",
                    "numbers": plot.get("numbers") or {},
                }
            )
        if figures:
            return figures, None

    if not visuals.is_dir():
        return [], (
            "This run has no visuals directory. The training flows render one as their "
            "last stage; a run registered before that stage existed will not have it."
        )
    for png in sorted(visuals.glob("*.png")):
        figures.append(
            {
                "slot": re.sub(r"^\d+_", "", png.stem),
                "title": _title_from_file(png.name),
                "caption": "",
                "url": f"{base}/{png.name}",
                "rendered": True,
                "reason": "",
                "numbers": {},
            }
        )
    if figures:
        return figures, None
    return [], "The visuals directory exists but holds no PNG figures."


def _docs(run_dir: Path, run_id: str, reports_dir: Path) -> list[dict[str, Any]]:
    """List the standalone HTML/JSON artifacts for a run, marking each present or absent.

    Args:
        run_dir: The run directory.
        run_id: Used for the served URL prefix.
        reports_dir: ``registry_store/reports``, where the drift report is written.

    Returns:
        One dict per artifact with ``available`` and, when it is not, a ``reason``.

    An unavailable artifact stays in the list and is rendered disabled. Removing it would
    make a run that never produced a SHAP report look identical to one where SHAP was never
    part of the pipeline, and those need different follow-up.
    """
    docs: list[dict[str, Any]] = []
    for slot, relative, label, note in DOC_SLOTS:
        path = run_dir / relative
        docs.append(
            {
                "slot": slot,
                "label": label,
                "note": note,
                "url": f"/runs/{run_id}/files/{relative}" if path.is_file() else None,
                "available": path.is_file(),
                "reason": "" if path.is_file() else f"{relative} was not written for this run",
            }
        )
    drift_report = reports_dir / f"{run_id}_drift.html"
    if drift_report.is_file():
        docs.append(
            {
                "slot": "drift",
                "label": "Drift report",
                "note": "Evidently's full per-feature comparison of reference against current.",
                "url": f"/reports/{drift_report.name}",
                "available": True,
                "reason": "",
            }
        )
    return docs


def _drift(run_dir: Path) -> dict[str, Any] | None:
    """Return the run's drift summary, or ``None`` when drift was never evaluated.

    Args:
        run_dir: The run directory holding ``drift.json``.

    Returns:
        The drift block, or ``None``. ``None`` means "not checked", which the page renders
        as exactly that — never as "no drift", which is a claim nobody made.
    """
    payload = _read_json(run_dir / "drift.json")
    if not isinstance(payload, dict):
        return None
    return {
        "verdict": payload.get("verdict"),
        "dataset_drift": payload.get("dataset_drift"),
        "drifted_share": payload.get("drifted_share"),
        "drifted_features": payload.get("drifted_features") or [],
        "target_drift": payload.get("target_drift"),
        "estimated_metric_name": payload.get("estimated_metric_name"),
        "estimated_metric_value": payload.get("estimated_metric_value"),
        "n_reference_rows": payload.get("n_reference_rows"),
        "n_current_rows": payload.get("n_current_rows"),
    }


def run_view(entry: RegistryEntry, runs_root: Path, reports_dir: Path) -> dict[str, Any]:
    """Assemble everything the page shows about one run, from that run's own files.

    Args:
        entry: The registry row, as validated by :mod:`aegis_ml.registry.store`.
        runs_root: ``registry_store/runs``.
        reports_dir: ``registry_store/reports``.

    Returns:
        A JSON-safe dict. Optional fields are ``None`` rather than absent, so the page's
        rendering path is the same whether a value exists or not and a missing number
        cannot fall through to a stale one left in the DOM.
    """
    result = entry.result
    run_dir = runs_root / entry.run_id
    leaderboard = result.leaderboard
    higher_is_better = bool(leaderboard.higher_is_better) if leaderboard is not None else True

    manifest = _read_json(run_dir / "visuals" / "manifest.json")
    verdict = manifest.get("verdict") if isinstance(manifest, dict) else None
    verdict = verdict if isinstance(verdict, dict) else {}

    gate = entry.gate
    tolerance = None
    if gate is not None and "coverage_tolerance" in gate.metrics:
        tolerance = float(gate.metrics["coverage_tolerance"])
    elif isinstance(verdict.get("coverage_tolerance"), int | float):
        tolerance = float(verdict["coverage_tolerance"])

    figures, figures_reason = _figures(run_dir, entry.run_id)
    recipe = result.recipe
    return {
        "run_id": entry.run_id,
        "domain_id": entry.domain_id,
        "created_at": entry.created_at,
        "stage": entry.stage,
        "task": result.task,
        "target": result.target,
        "target_unit": verdict.get("target_unit"),
        "metric_name": result.metric_name,
        "metric_value": result.metric_value,
        "higher_is_better": higher_is_better,
        "requested_coverage": result.requested_coverage,
        "empirical_coverage": result.empirical_coverage,
        "coverage_tolerance": tolerance,
        "training_size": result.training_size,
        "calibration_size": result.calibration_size,
        "test_size": result.test_size,
        "dataset_digest": result.dataset_digest,
        "artifact_path": result.artifact_path,
        "tier": recipe.tier if recipe is not None else None,
        "recipe": None
        if recipe is None
        else {
            "tier": recipe.tier,
            "members": [
                {"name": m.name, "kind": m.kind, "weight": m.weight}
                for m in recipe.members
            ],
            "search_seconds": recipe.search_seconds,
            "numeric_features": list(recipe.numeric_features),
            "categorical_features": list(recipe.categorical_features),
            "notes": list(recipe.notes),
        },
        "gate": None
        if gate is None
        else {
            "promoted": gate.promoted,
            "checks": dict(gate.checks),
            "reasons": list(gate.reasons),
            "metrics": {k: float(v) for k, v in gate.metrics.items()},
            "champion_run_id": gate.champion_run_id,
        },
        "leaderboard": None
        if leaderboard is None
        else {
            "metric_name": leaderboard.metric_name,
            "higher_is_better": higher_is_better,
            "candidates": [
                {
                    "name": c.name,
                    "tier": c.tier,
                    "metric_value": c.metric_value,
                    "fit_seconds": c.fit_seconds,
                    "portable": c.portable,
                    "selected": c.selected,
                }
                for c in leaderboard.candidates
            ],
            "tiers_run": list(leaderboard.tiers_run),
            "tiers_skipped": dict(leaderboard.tiers_skipped),
        },
        "worst_slice": _worst_slice(entry, higher_is_better),
        "slice_count": len(result.slices),
        "drift": _drift(run_dir),
        "figures": figures,
        "figures_reason": figures_reason,
        "docs": _docs(run_dir, entry.run_id, reports_dir),
        "notes": list(result.notes),
        "tier_colour": theme.tier_colour(recipe.tier) if recipe is not None else None,
    }


def collect(
    entries: Sequence[RegistryEntry],
    *,
    registry_dir: Path,
    services: Mapping[str, Any],
    registry_error: str | None = None,
) -> dict[str, Any]:
    """Build the whole page payload from the registry rows and the live service states.

    Args:
        entries: Registry rows, newest first, as :func:`aegis_ml.registry.store.list_runs`
            returns them.
        registry_dir: The registry root, printed on the page so a viewer can find the files.
        services: ``{key: service state dict}`` from
            :class:`aegis_ml.dashboard.services.Supervisor`.
        registry_error: Why the registry could not be read, when it could not. Rendered as
            the empty state instead of an empty run list, because "no runs" and "the index
            is unreadable" require opposite responses.

    Returns:
        The JSON-safe payload embedded into the page and served at ``/api/state.json``.
    """
    runs_root = registry_dir / "runs"
    reports_dir = registry_dir / "reports"
    runs = [run_view(entry, runs_root, reports_dir) for entry in entries]
    champion = next((r for r in runs if r["stage"] == "production"), None)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "registry_dir": str(registry_dir),
        "runs": runs,
        "champion_run_id": champion["run_id"] if champion else None,
        "artifact_path": champion["artifact_path"] if champion else None,
        "registry_error": registry_error,
        "services": dict(services),
    }


_HTML_SHELL = """<!DOCTYPE html>
<html lang="en" data-theme="">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>__TITLE__</title>
<link rel="icon" href="data:image/svg+xml,__FAVICON__">
<style>
__CSS__
</style>
</head>
<body>
<div class="shell">
  <aside class="rail">
    <div class="rail-inner">
      <div class="brand">
        <div class="glyph">AM</div>
        <div><b>aegis-ml</b><span>model registry</span></div>
      </div>
      <nav class="nav" id="nav"></nav>
      <div class="rail-foot" id="rail-services"></div>
    </div>
  </aside>
  <main class="main"><div class="wrap" id="view"></div></main>
</div>
<div class="lb" id="lightbox" role="dialog" aria-modal="true" aria-label="Figure viewer">
  <div class="lb-bar">
    <div><div class="t" id="lb-title"></div><div class="n" id="lb-count"></div></div>
    <div class="sp">
      <button type="button" id="lb-prev">&#8592; Prev</button>
      <button type="button" id="lb-next">Next &#8594;</button>
      <button type="button" id="lb-close">Close &#215;</button>
    </div>
  </div>
  <div class="lb-img"><img id="lb-img" alt=""></div>
</div>
<script type="application/json" id="hub-data">__DATA__</script>
<script>
__JS__
</script>
</body>
</html>
"""

_FAVICON = (
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='8' fill='%23E2743B'/%3E"
    "%3Cpath d='M8 22 L16 9 L24 22 M11.5 18 H20.5' stroke='%23150D07' "
    "stroke-width='2.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E"
    "%3C/svg%3E"
)
"""An inline SVG favicon. A ``.ico`` file would be a second request; this is zero."""


_JS = r"""
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("hub-data").textContent);
  var RUNS = D.runs || [];
  var BY_ID = {};
  RUNS.forEach(function (r) { BY_ID[r.run_id] = r; });

  /* ---------- formatting -------------------------------------------------- */
  var ENT = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  function esc(v) {
    if (v === null || v === undefined) return "";
    return String(v).replace(/[&<>"']/g, function (c) { return ENT[c]; });
  }
  /* Every "value missing" path funnels through here so the page can never print a zero
     it did not read from a file. The tooltip says which artifact was absent. */
  function gap(why) { return '<span class="mono" title="' + esc(why || "not recorded") +
    '">&mdash;</span>'; }
  function sig(v, digits) {
    if (v === null || v === undefined || isNaN(v)) return null;
    var d = digits === undefined ? 4 : digits;
    var a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return Number(v).toExponential(2);
    return Number(v).toPrecision(d).replace(/\.?0+$/, "");
  }
  function pct(v, digits) {
    if (v === null || v === undefined || isNaN(v)) return null;
    return (v * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }
  function secs(v) {
    if (v === null || v === undefined || isNaN(v)) return null;
    if (v < 1) return (v * 1000).toFixed(0) + " ms";
    if (v < 90) return v.toFixed(1) + " s";
    return (v / 60).toFixed(1) + " min";
  }
  function ago(iso) {
    if (!iso) return "";
    var t = Date.parse(iso);
    if (isNaN(t)) return iso;
    var d = new Date(t);
    var day = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    var clock = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    return day + " " + clock;
  }
  function shortId(id) {
    var m = /^(.*)-(\d{8}T\d+)-([0-9a-f]+)$/.exec(id || "");
    return m ? m[3] : (id || "").slice(-6);
  }

  /* ---------- verdict logic (thresholds come from the run, never from here) --- */
  function coverageTone(r) {
    if (r.empirical_coverage === null || r.requested_coverage === null) return "";
    if (r.coverage_tolerance === null || r.coverage_tolerance === undefined) return "";
    var floor = r.requested_coverage - r.coverage_tolerance;
    if (r.empirical_coverage < floor) return "bad";
    if (r.empirical_coverage < r.requested_coverage) return "warn";
    return "good";
  }
  function driftTone(v) {
    if (v === "block") return "bad";
    if (v === "warn") return "warn";
    if (v === "ok" || v === "pass") return "good";
    return "flat";
  }
  function chip(text, tone, extra) {
    return '<span class="chip ' + tone + '"' + (extra || "") + ">" + esc(text) + "</span>";
  }
  function tierChip(r) {
    if (!r.tier) return "";
    return chip(r.tier, "flat tier", ' style="--tier:' + esc(r.tier_colour || "") + '"');
  }
  function gateChip(r) {
    if (!r.gate) return chip("gate not evaluated", "flat");
    var n = Object.keys(r.gate.checks || {}).length;
    var kept = Object.keys(r.gate.checks || {})
      .filter(function (k) { return r.gate.checks[k]; });
    var passed = kept.length;
    if (r.gate.promoted) return chip("GATE PASS " + passed + "/" + n, "good");
    return chip("GATE FAIL " + passed + "/" + n, "bad");
  }

  /* ---------- components --------------------------------------------------- */
  function tile(k, value, sub, tone, lead) {
    return '<div class="tile ' + (tone || "") + (lead ? " lead" : "") + '">' +
      '<div class="k">' + esc(k) + "</div>" +
      '<div class="v">' + value + "</div>" +
      (sub ? '<div class="s">' + sub + "</div>" : "") + "</div>";
  }
  function section(title, note, body, id) {
    return '<section class="section"' + (id ? ' id="' + id + '"' : "") + '>' +
      '<div class="bar"><h2>' + esc(title) + "</h2>" +
      (note ? '<span class="note">' + note + "</span>" : "") + "</div>" + body + "</section>";
  }
  function emptyState(title, why, remedy) {
    return '<div class="empty"><h3>' + esc(title) + "</h3>" +
      '<p class="why">' + esc(why) + "</p>" +
      (remedy ? '<pre class="cmd">' + esc(remedy) + "</pre>" : "") + "</div>";
  }

  /* ---------- verdict tiles ------------------------------------------------ */
  function verdictTiles(r) {
    var out = [];
    var metric = sig(r.metric_value);
    out.push(tile(
      r.metric_name,
      metric === null ? gap("metric_value absent from the registry row") : esc(metric),
      esc(r.target) + (r.test_size ? " &middot; " + r.test_size + " held-out rows" : ""),
      "", true));

    out.push(tile("coverage requested",
      r.requested_coverage === null
        ? gap("no interval was requested") : esc(pct(r.requested_coverage, 0)),
      "the guarantee that was asked for"));

    var tone = coverageTone(r);
    var floorNote = (r.coverage_tolerance !== null && r.coverage_tolerance !== undefined &&
                     r.requested_coverage !== null)
      ? "floor " + pct(r.requested_coverage - r.coverage_tolerance, 1)
      : "no tolerance recorded";
    out.push(tile("coverage measured",
      r.empirical_coverage === null
        ? gap("conformal coverage was not measured") : esc(pct(r.empirical_coverage)),
      esc(floorNote), tone));

    if (r.gate) {
      var checks = r.gate.checks || {};
      var names = Object.keys(checks);
      var passed = names.filter(function (k) { return checks[k]; }).length;
      out.push(tile("promotion gate", r.gate.promoted ? "PASS" : "FAIL",
        passed + " of " + names.length + " criteria", r.gate.promoted ? "good" : "bad"));
    } else {
      out.push(tile("promotion gate", gap("this run was never gated"),
        "no gate decision recorded"));
    }

    if (r.drift) {
      var share = r.drift.drifted_share;
      out.push(tile("drift", esc(String(r.drift.verdict || "unknown").toUpperCase()),
        share === null || share === undefined ? "share not recorded"
          : esc(pct(share, 0)) + " of features drifted",
        driftTone(r.drift.verdict) === "flat" ? "" : driftTone(r.drift.verdict)));
    } else {
      out.push(tile("drift", gap("drift.json absent for this run"), "no drift check has been run"));
    }

    out.push(tile("winning tier", r.tier ? esc(r.tier) : gap("no recipe recorded"),
      r.recipe && r.recipe.members.length
        ? esc(r.recipe.members.map(function (m) { return m.kind; }).join(" + "))
        : "no ensemble members recorded"));

    var champ = D.champion_run_id;
    out.push(tile("serving", champ ? esc(shortId(champ)) : gap("no run is in production"),
      champ ? (champ === r.run_id ? "this run is the champion" : "a different run is the champion")
            : "nothing has been promoted"));
    return '<div class="stats">' + out.join("") + "</div>";
  }

  /* ---------- gate criteria ------------------------------------------------ */
  function gateBlock(r) {
    if (!r.gate) {
      return emptyState("No gate decision for this run",
        "entry.json carries no gate block, so nothing here can say whether this " +
        "model was allowed to serve.",
        "aegis-ml eval --run-id " + r.run_id);
    }
    var rows = (r.gate.reasons || []).map(function (line) {
      var m = /^(PASS|FAIL|PROMOTED|REJECTED)\b/.exec(line);
      var kind = m ? m[1] : "";
      var mark = kind === "PASS" || kind === "PROMOTED" ? "ok" : (kind ? "no" : "");
      var glyph = mark === "ok" ? "&#10003;" : (mark === "no" ? "&#10007;" : "&middot;");
      var rest = kind ? line.slice(kind.length).replace(/^[:\s]+/, "") : line;
      return '<div class="crit-row"><div class="mark ' + mark + '">' + glyph + "</div>" +
        '<div class="why"><b>' + esc(kind || "note") + "</b> " + esc(rest) + "</div></div>";
    }).join("");
    return '<div class="card pad"><div class="crit">' + rows + "</div></div>";
  }

  /* ---------- drift block -------------------------------------------------- */
  function driftBlock(r) {
    if (!r.drift) {
      return emptyState("Drift has not been checked for this run",
        "There is no drift.json in the run directory, so no claim is made either way.",
        "aegis-ml drift --run-id " + r.run_id + " --current <frame.parquet>");
    }
    var d = r.drift;
    var feats = (d.drifted_features || []).map(function (f) { return chip(f, "warn"); }).join(" ");
    var est = d.estimated_metric_value === null || d.estimated_metric_value === undefined
      ? gap("no performance estimate was produced")
      : esc(sig(d.estimated_metric_value)) + ' <span class="cap">' +
        esc(d.estimated_metric_name || "") + "</span>";
    var report = (r.docs || [])
      .filter(function (x) { return x.slot === "drift" && x.available; })[0];
    return '<div class="card pad">' +
      '<div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;' +
        'margin-bottom:.85rem">' +
        chip(String(d.verdict || "unknown").toUpperCase(), driftTone(d.verdict)) +
        '<span class="sub" style="margin:0">' +
          (d.drifted_share === null || d.drifted_share === undefined ? "share not recorded"
            : esc(pct(d.drifted_share, 0)) + " of features drifted") +
          (d.n_reference_rows ? " &middot; " + d.n_reference_rows + " reference vs " +
             (d.n_current_rows || "?") + " current rows" : "") +
        "</span>" +
        (report ? '<a class="btn" style="margin-left:auto" target="_blank" rel="noopener" href="' +
           esc(report.url) + '">Open Evidently report &#8599;</a>' : "") +
      "</div>" +
      '<dl class="kv"><dt>estimated performance</dt><dd>' + est + "</dd>" +
      "<dt>target drift</dt><dd>" + (d.target_drift === null || d.target_drift === undefined
        ? gap("target drift not computed") : esc(sig(d.target_drift))) + "</dd></dl>" +
      (feats ? '<div style="margin-top:.9rem;display:flex;gap:.35rem;flex-wrap:wrap">' + feats +
        "</div>" : "") +
      "</div>";
  }

  /* ---------- run list ----------------------------------------------------- */
  function runRow(r) {
    var cov = r.empirical_coverage === null
      ? gap("coverage not measured") : esc(pct(r.empirical_coverage));
    var metric = sig(r.metric_value);
    return '<div class="run" data-run="' + esc(r.run_id) + '" tabindex="0" role="link">' +
      '<div class="stagebar ' + esc(r.stage) + '"></div>' +
      "<div><div class=\"id\">" + esc(r.run_id) + "</div>" +
        '<div class="meta">' + esc(r.domain_id) + " &middot; " + esc(r.target) +
        " &middot; " + esc(ago(r.created_at)) + "</div></div>" +
      '<div><div class="cap">stage</div>' + chip(r.stage, r.stage === "production" ? "good" :
        (r.stage === "staging" ? "info" : "flat")) + "</div>" +
      '<div><div class="cap">' + esc(r.metric_name) + '</div><div class="num">' +
        (metric === null ? gap("no metric recorded") : esc(metric)) + "</div></div>" +
      '<div><div class="cap">coverage</div><div class="num">' + cov + "</div></div>" +
      '<div class="go">&#8250;</div></div>';
  }

  /* ---------- leaderboard -------------------------------------------------- */
  function leaderboardBlock(r) {
    if (!r.leaderboard || !r.leaderboard.candidates.length) {
      return emptyState("No leaderboard for this run",
        "The registry row carries no candidate list, so there is nothing to say about what lost.",
        "aegis-ml train --adapter <module>");
    }
    var lb = r.leaderboard;
    var rows = lb.candidates.map(function (c) {
      return "<tr>" +
        "<td>" + (c.selected ? '<b style="color:var(--accent-ink)">' + esc(c.name) +
          "</b>" : esc(c.name)) + "</td>" +
        "<td>" + chip(c.tier, "flat tier", ' style="--tier:' + esc(D.tier_colours[c.tier] || "") +
          '"') + "</td>" +
        '<td class="n' + (c.selected ? " win" : "") + '">' +
          (c.metric_value === null
            ? gap("candidate produced no score") : esc(sig(c.metric_value))) +
            "</td>" +
        '<td class="n">' + (secs(c.fit_seconds) || gap("fit time not recorded")) + "</td>" +
        "<td>" + (c.portable ? chip("portable", "good") : chip("not portable", "warn")) + "</td>" +
        "</tr>";
    }).join("");
    var skipped = Object.keys(lb.tiers_skipped || {});
    var skipNote = skipped.length
      ? '<ul class="notes" style="margin-top:1rem">' + skipped.map(function (t) {
          return "<li><b>" + esc(t) + "</b> did not run &mdash; " + esc(lb.tiers_skipped[t]) +
            "</li>";
        }).join("") + "</ul>"
      : "";
    return '<div class="card pad"><div class="scroll-x"><table class="cmp"><thead><tr>' +
      "<th>candidate</th><th>tier</th><th>" + esc(lb.metric_name) +
        "</th><th>fit</th><th>portable</th>" +
      "</tr></thead><tbody>" + rows + "</tbody></table></div>" + skipNote + "</div>";
  }

  /* ---------- gallery ------------------------------------------------------ */
  var GALLERY = [];
  function galleryBlock(r) {
    GALLERY = (r.figures || []).filter(function (f) { return f.rendered && f.url; });
    if (!GALLERY.length) {
      return emptyState("No figures rendered for this run",
        r.figures_reason || "The visuals bundle was not produced.",
        "aegis-ml visuals --run-id " + r.run_id);
    }
    var cards = (r.figures || []).map(function (f) {
      if (!f.rendered || !f.url) {
        return '<div class="fig"><div class="cap"><h3>' + esc(f.title) + "</h3>" +
          '<p class="why" style="color:var(--warn)">not rendered &mdash; ' +
            esc(f.reason || "no reason recorded") +
          "</p></div></div>";
      }
      var i = GALLERY.indexOf(f);
      var cap = f.caption || "";
      var head = cap.length > 150 ? cap.slice(0, 150).replace(/\s+\S*$/, "") + "…" : cap;
      var tail = cap.length > 150 ? cap : "";
      return '<figure class="fig" style="margin:0">' +
        '<button class="plate" type="button" data-fig="' + i + '" aria-label="Enlarge ' +
          esc(f.title) + '"' +
          ' style="border:0;width:100%">' +
          '<img loading="lazy" src="' + esc(f.url) + '" alt="' + esc(f.title) + '"></button>' +
        '<figcaption class="cap"><h3>' + esc(f.title) + "</h3>" +
          (head ? "<p>" + esc(head) + "</p>" : "") +
          (tail ? '<p class="more">' + esc(tail) + "</p>" +
                  '<button class="toggle" type="button" data-more="1">' +
                  'Read the full caption</button>' : "") +
        "</figcaption></figure>";
    }).join("");
    return '<div class="gal">' + cards + "</div>";
  }

  /* ---------- documents / tabs --------------------------------------------- */
  function docsBlock(r) {
    var tabs = (r.docs || []).map(function (d) {
      if (!d.available) {
        return '<span class="tab off" title="' + esc(d.reason) + '">' + esc(d.label) + "</span>";
      }
      return '<a class="tab" target="_blank" rel="noopener" href="' + esc(d.url) + '" title="' +
        esc(d.note) + '">' + esc(d.label) + " &#8599;</a>";
    }).join("");
    return '<div class="tabs">' + tabs + "</div>";
  }

  /* ---------- views -------------------------------------------------------- */
  function noRuns() {
    if (D.registry_error) {
      return emptyState("The registry could not be read", D.registry_error,
        "ls " + D.registry_dir);
    }
    return emptyState("No runs are registered yet",
      "There is nothing under " + D.registry_dir + "/runs, so every number on this page " +
      "would have to be invented. It will not be. Train one run and this page fills in.",
      ".venv/bin/python scripts/run_demo.py");
  }

  function viewOverview() {
    if (!RUNS.length) return '<div class="head"><div class="grow">' +
      '<div class="eyebrow">overview</div>' +
      "<h1>Nothing to show yet</h1></div></div>" + noRuns();
    var r = RUNS[0];
    return '<div class="head"><div class="grow">' +
        '<div class="eyebrow">latest run &middot; ' + esc(r.domain_id) + "</div>" +
        "<h1>" + esc(r.target) + "</h1>" +
        '<div class="sub"><span class="mono">' + esc(r.run_id) + "</span> &middot; " +
          esc(r.task) + " &middot; " + esc(ago(r.created_at)) + "</div>" +
        '<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.7rem">' +
          chip(r.stage, r.stage === "production" ? "good" : "info") + gateChip(r) + tierChip(r) +
          (r.drift ? chip("drift " +
            String(r.drift.verdict || "?").toUpperCase(), driftTone(r.drift.verdict)) : "") +
        "</div></div>" +
        '<a class="btn primary" href="#/run/' + esc(r.run_id) + '">Open the evidence &#8594;</a>' +
      "</div>" +
      verdictTiles(r) +
      section("Why the gate said what it said",
        "read from entry.json, in the gate's own words", gateBlock(r)) +
      section("Drift", "reference frame against the current one", driftBlock(r)) +
      section("Recent runs", RUNS.length + " registered",
        '<div class="runs">' + RUNS.slice(0, 5).map(runRow).join("") + "</div>");
  }

  function viewRuns() {
    if (!RUNS.length) return '<div class="head"><div class="grow"><div class="eyebrow">runs</div>' +
      "<h1>Run history</h1></div></div>" + noRuns();
    return '<div class="head"><div class="grow"><div class="eyebrow">registry</div>' +
      "<h1>Run history</h1>" +
      '<div class="sub">' + RUNS.length + (RUNS.length === 1 ? " run" : " runs") +
      ' in <span class="mono">' + esc(D.registry_dir) +
      "</span>, newest first</div></div></div>" +
      '<div class="runs">' + RUNS.map(runRow).join("") + "</div>";
  }

  function viewRun(id) {
    var r = BY_ID[id];
    if (!r) return emptyState("No such run", "There is no registered run with id " + id + ".",
      "aegis-ml registry");
    var facts = [
      ["domain", r.domain_id], ["task", r.task], ["target", r.target],
      ["train / calib / test", r.training_size + " / " + r.calibration_size + " / " + r.test_size],
      ["slices evaluated", r.slice_count],
      ["search time", secs(r.recipe && r.recipe.search_seconds) || "not recorded"],
      ["dataset digest", r.dataset_digest || "not recorded"],
      ["serving artifact", r.artifact_path || "not recorded"],
      ["created", r.created_at]
    ].map(function (p) { return "<dt>" + esc(p[0]) + "</dt><dd>" + esc(p[1]) + "</dd>"; }).join("");

    var recipeNotes = r.recipe && r.recipe.notes.length
      ? section("What the search decided", "recipe notes, verbatim",
          '<div class="card pad"><ul class="notes">' +
          r.recipe.notes.map(function (n) { return "<li>" + esc(n) + "</li>"; }).join("") +
            "</ul></div>")
      : "";

    return '<div class="head"><div class="grow">' +
        '<div class="eyebrow"><a href="#/runs" style="color:var(--accent-ink)">' +
        '&#8592; all runs</a></div>' +
        "<h1>" + esc(r.target) + "</h1>" +
        '<div class="sub"><span class="mono">' + esc(r.run_id) + "</span></div>" +
        '<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.7rem">' +
          chip(r.stage, r.stage === "production" ? "good" : "info") + gateChip(r) + tierChip(r) +
          (r.drift ? chip("drift " +
            String(r.drift.verdict || "?").toUpperCase(), driftTone(r.drift.verdict)) : "") +
        "</div></div></div>" +
      verdictTiles(r) +
      section("Artifacts", "each opens in a new tab, served from the run directory", docsBlock(r)) +
      section("Figures", (r.figures || []).filter(function (f) { return f.rendered; }).length +
        " rendered &middot; click any to enlarge", galleryBlock(r)) +
      section("Promotion gate", "in the gate's own words", gateBlock(r)) +
      section("Leaderboard", "what won and what lost", leaderboardBlock(r)) +
      section("Drift", "", driftBlock(r)) +
      recipeNotes +
      section("Facts", "straight from entry.json", '<div class="card pad"><dl class="kv">' +
        facts + "</dl></div>");
  }

  function viewCompare() {
    if (RUNS.length < 2) {
      return '<div class="head"><div class="grow"><div class="eyebrow">compare</div>' +
        "<h1>Two runs, side by side</h1></div></div>" +
        emptyState("Comparison needs two runs",
          "The registry holds " + RUNS.length + " run" + (RUNS.length === 1 ? "" : "s") +
          ", so there is nothing to compare it against.",
          ".venv/bin/python scripts/run_demo.py");
    }
    var a = state.cmpA || RUNS[0].run_id;
    var b = state.cmpB || RUNS[1].run_id;
    function opts(sel) {
      return RUNS.map(function (r) {
        return '<option value="' + esc(r.run_id) + '"' + (r.run_id === sel ? " selected" : "") +
          ">" +
          esc(shortId(r.run_id)) + " · " + esc(ago(r.created_at)) + "</option>";
      }).join("");
    }
    var ra = BY_ID[a], rb = BY_ID[b];
    function cell(v, best) { return '<td class="n' + (best ? " win" : "") + '">' + v + "</td>"; }
    function cmpRow(label, va, vb, fa, fb, higher) {
      var wa = false, wb = false;
      if (fa !== null && fb !== null && fa !== undefined && fb !== undefined && fa !== fb) {
        wa = higher ? fa > fb : fa < fb; wb = !wa;
      }
      return '<tr><th class="row">' + esc(label) + "</th>" + cell(va, wa) + cell(vb, wb) + "</tr>";
    }
    var rows = [
      cmpRow(ra.metric_name + " (primary)", sig(ra.metric_value) || gap("no metric"),
        sig(rb.metric_value) || gap("no metric"),
        ra.metric_value, rb.metric_value, ra.higher_is_better),
      cmpRow("coverage requested", pct(ra.requested_coverage, 0) || gap("none"),
        pct(rb.requested_coverage, 0) || gap("none"), null, null, true),
      cmpRow("coverage measured", pct(ra.empirical_coverage) || gap("not measured"),
        pct(rb.empirical_coverage) || gap("not measured"), null, null, true),
      cmpRow("worst slice",
        ra.worst_slice ? esc(sig(ra.worst_slice.metric_value)) + ' <span class="cap">' +
          esc(ra.worst_slice.feature + "=" + ra.worst_slice.level) +
            "</span>" : gap("no slices evaluated"),
        rb.worst_slice ? esc(sig(rb.worst_slice.metric_value)) + ' <span class="cap">' +
          esc(rb.worst_slice.feature + "=" + rb.worst_slice.level) +
            "</span>" : gap("no slices evaluated"),
        ra.worst_slice ? ra.worst_slice.metric_value : null,
        rb.worst_slice ? rb.worst_slice.metric_value : null, ra.higher_is_better),
      cmpRow("tier", esc(ra.tier || "—"), esc(rb.tier || "—"), null, null, true),
      cmpRow("gate", ra.gate ? (ra.gate.promoted ? "PASS" : "FAIL") : gap("not gated"),
        rb.gate ? (rb.gate.promoted ? "PASS" : "FAIL") : gap("not gated"), null, null, true),
      cmpRow("drift", ra.drift ? String(ra.drift.verdict).toUpperCase() : gap("not checked"),
        rb.drift ? String(rb.drift.verdict).toUpperCase() : gap("not checked"), null, null, true),
      cmpRow("held-out rows", ra.test_size, rb.test_size, null, null, true),
      cmpRow("stage", esc(ra.stage), esc(rb.stage), null, null, true),
      cmpRow("created", esc(ago(ra.created_at)), esc(ago(rb.created_at)), null, null, true)
    ].join("");
    return '<div class="head"><div class="grow"><div class="eyebrow">compare</div>' +
      "<h1>Two runs, side by side</h1>" +
      '<div class="sub">A cell is highlighted only where the two values differ and the ' +
      "metric has a direction.</div></div></div>" +
      '<div class="card pad"><div style="display:flex;gap:.6rem;' +
        'margin-bottom:1rem;flex-wrap:wrap">' +
        '<select class="btn" id="cmp-a">' + opts(a) + "</select>" +
        '<select class="btn" id="cmp-b">' + opts(b) + "</select></div>" +
      '<div class="scroll-x"><table class="cmp"><thead><tr><th class="row"></th><th>' +
        esc(shortId(a)) + "</th><th>" + esc(shortId(b)) + "</th></tr></thead><tbody>" +
        rows + "</tbody></table></div></div>";
  }

  function viewService(key) {
    var s = (D.services || {})[key];
    if (!s) {
      return emptyState("Unknown service", "No service is registered under the key " + key +
        ".", "");
    }
    var entry = s.entry_url || s.url;
    var head = '<div class="head"><div class="grow"><div class="eyebrow">premade UI</div>' +
      "<h1>" + esc(s.label) + "</h1>" +
      '<div class="sub">' + esc(s.blurb) + "</div></div>" +
      (s.running ? '<a class="btn primary" target="_blank" rel="noopener" href="' + esc(entry) +
        '">Open in a new tab &#8599;</a>' : "") + "</div>";

    if (!s.running) {
      return head + emptyState(s.label + " is not running", s.reason ||
        "It is not answering on port " + s.port + ", and no reason was recorded.", s.remedy);
    }
    var bar = '<div class="frame-bar"><span class="dot up"></span><span>live</span>' +
      '<span class="url">' + esc(s.url) + "</span>" +
      '<span class="sp"><a class="btn" target="_blank" rel="noopener" href="' + esc(entry) +
      '">Open in a new tab &#8599;</a></span></div>';
    if (s.embeddable === false) {
      return head + '<div class="frame-wrap">' + bar + "</div>" +
        emptyState(s.label + " refuses to be embedded",
          "It is running and reachable, but its response carries " +
          (s.frame_reason || "a frame-ancestors restriction") +
          ", so a browser will not render it inside this page. The link above opens the real UI.",
          "open " + s.url);
    }
    return head + '<div class="frame-wrap">' + bar +
      '<iframe src="' + esc(entry) + '" title="' + esc(s.label) +
        '"></iframe></div>';
  }

  /* ---------- rail --------------------------------------------------------- */
  function renderRail(route) {
    var items = [
      ["#/overview", "◉", "Overview", ""],
      ["#/runs", "≡", "Runs", String(RUNS.length)],
      ["#/compare", "⇄", "Compare", ""],
      ["#/mlflow", "◴", "MLflow", ""],
      ["#/optuna", "⌘", "Optuna", ""]
    ];
    document.getElementById("nav").innerHTML =
      '<div class="nav-label">registry</div>' +
      items.slice(0, 3).map(function (i) { return navItem(i, route); }).join("") +
      '<div class="nav-label" style="margin-top:1rem">tooling</div>' +
      items.slice(3).map(function (i) { return navItem(i, route); }).join("");

    var svc = D.services || {};
    document.getElementById("rail-services").innerHTML = Object.keys(svc).map(function (k) {
      var s = svc[k];
      return '<a class="svc-row" href="#/' + esc(k) + '" title="' +
        esc(s.running ? (s.entry_url || s.url) : s.reason) + '">' +
        '<span class="dot ' + (s.running ? "up" : "down") + '"></span>' +
        '<span class="name">' + esc(s.label) + "</span>" +
        '<span class="port">:' + esc(s.port) + "</span></a>";
    }).join("") +
      '<button class="btn" type="button" id="theme-toggle" style="justify-content:center">' +
      "Switch theme</button>";
    var toggle = document.getElementById("theme-toggle");
    if (toggle) toggle.addEventListener("click", flipTheme);
  }
  function navItem(i, route) {
    var on = route === i[0].slice(2) || (i[0] === "#/runs" && route.indexOf("run/") === 0);
    return '<a href="' + i[0] + '" class="' + (on ? "on" : "") + '">' +
      '<span class="ic">' + i[1] + "</span>" + esc(i[2]) +
      (i[3] ? '<span class="tail">' + esc(i[3]) + "</span>" : "") + "</a>";
  }
  function flipTheme() {
    var root = document.documentElement;
    var now = root.getAttribute("data-theme");
    var next = now === "dark" ? "light" : (now === "light" ? "" : "light");
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("aegis-ml-theme", next);
    } catch (e) { /* private mode: the choice still holds for this session */ }
  }
  try {
    var saved = localStorage.getItem("aegis-ml-theme");
    if (saved !== null) document.documentElement.setAttribute("data-theme", saved);
  } catch (e) { /* storage unavailable: the system preference decides, which is the default */ }

  /* ---------- lightbox ----------------------------------------------------- */
  var lb = document.getElementById("lightbox"), lbIdx = 0;
  function openLb(i) {
    if (!GALLERY.length) return;
    lbIdx = (i + GALLERY.length) % GALLERY.length;
    var f = GALLERY[lbIdx];
    document.getElementById("lb-img").src = f.url;
    document.getElementById("lb-img").alt = f.title;
    document.getElementById("lb-title").textContent = f.title;
    document.getElementById("lb-count").textContent = (lbIdx + 1) + " of " + GALLERY.length;
    lb.classList.add("on");
  }
  function closeLb() { lb.classList.remove("on"); }
  document.getElementById("lb-close").addEventListener("click", closeLb);
  document.getElementById("lb-prev").addEventListener("click", function () { openLb(lbIdx - 1); });
  document.getElementById("lb-next").addEventListener("click", function () { openLb(lbIdx + 1); });
  lb.addEventListener("click", function (e) { if (e.target === lb) closeLb(); });
  document.addEventListener("keydown", function (e) {
    if (!lb.classList.contains("on")) return;
    if (e.key === "Escape") closeLb();
    if (e.key === "ArrowLeft") openLb(lbIdx - 1);
    if (e.key === "ArrowRight") openLb(lbIdx + 1);
  });

  /* ---------- router ------------------------------------------------------- */
  var state = { cmpA: null, cmpB: null };
  function route() { return (location.hash || "#/overview").slice(2); }
  function render() {
    var r = route();
    var view = document.getElementById("view");
    var body;
    if (r.indexOf("run/") === 0) body = viewRun(r.slice(4));
    else if (r === "runs") body = viewRuns();
    else if (r === "compare") body = viewCompare();
    else if (r === "mlflow" || r === "optuna") body = viewService(r);
    else body = viewOverview();
    view.innerHTML = body;
    renderRail(r);
    wire(view);
    if (r.indexOf("run/") !== 0) GALLERY = [];
    window.scrollTo(0, 0);
  }
  function wire(view) {
    view.querySelectorAll("[data-run]").forEach(function (el) {
      el.addEventListener("click", function () { location.hash = "#/run/" +
        el.getAttribute("data-run"); });
      el.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); }
      });
    });
    view.querySelectorAll("[data-fig]").forEach(function (el) {
      el.addEventListener("click", function () {
        openLb(parseInt(el.getAttribute("data-fig"), 10));
      });
    });
    view.querySelectorAll("[data-more]").forEach(function (el) {
      el.addEventListener("click", function () {
        var cap = el.parentNode;
        cap.classList.toggle("open");
        el.textContent = cap.classList.contains("open") ? "Show less" : "Read the full caption";
      });
    });
    var frame = view.querySelector("iframe");
    if (frame) {
      /* MLflow and Optuna both focus an element as they boot, which scrolls this page
         past its own header. Putting the scroll back after the frame loads keeps the
         panel's title and its "open in a new tab" link on screen. */
      frame.addEventListener("load", function () { window.scrollTo(0, 0); });
    }
    var a = view.querySelector("#cmp-a"), b = view.querySelector("#cmp-b");
    if (a) a.addEventListener("change", function () { state.cmpA = a.value; render(); });
    if (b) b.addEventListener("change", function () { state.cmpB = b.value; render(); });
  }
  window.addEventListener("hashchange", render);

  /* ---------- live service status ------------------------------------------ */
  function poll() {
    fetch("/api/services.json", { cache: "no-store" }).then(function (res) {
      return res.ok ? res.json() : null;
    }).then(function (next) {
      if (!next) return;
      var changed = JSON.stringify(next) !== JSON.stringify(D.services);
      D.services = next;
      if (!changed) return;
      var r = route();
      if (r === "mlflow" || r === "optuna") render(); else renderRail(r);
    }).catch(function () { /* the hub outliving a poll is not worth a visible error */ });
  }
  setInterval(poll, 5000);

  render();
})();
"""


def render(payload: Mapping[str, Any], *, title: str | None = None) -> str:
    """Render the complete hub page.

    Args:
        payload: The dict from :func:`collect`.
        title: Browser title. Defaults to the latest run's domain, so a viewer with three
            of these open on three checkouts can tell the tabs apart.

    Returns:
        A single self-contained HTML document. No external request of any kind — the only
        URLs in it are same-origin paths this package's own server resolves.
    """
    runs = payload.get("runs") or []
    domain = runs[0]["domain_id"] if runs else "no runs"
    document_title = title or f"aegis-ml · {domain}"
    data = dict(payload)
    data["tier_colours"] = dict(theme.TIER_COLOURS)
    # `</script>` inside the JSON island would end the element early; escaping the slash is
    # the standard, JSON-valid way to keep the parser inside the string.
    island = json.dumps(data, default=str).replace("</", "<\\/")
    return (
        _HTML_SHELL.replace("__TITLE__", html.escape(document_title))
        .replace("__FAVICON__", _FAVICON)
        .replace("__CSS__", theme.stylesheet())
        .replace("__DATA__", island)
        .replace("__JS__", _JS)
    )
