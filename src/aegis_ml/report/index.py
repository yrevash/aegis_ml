"""Render ``visuals/index.html`` — the one file a human opens.

Two decisions shape everything here.

**The page is self-contained.** Every PNG is inlined as a ``data:`` URI and the stylesheet
is inlined too, so the file can be attached to an email, dropped in a chat, copied onto a
USB stick or opened from a directory whose sibling files were not copied, and it still shows
the same evidence. A report that renders as a column of broken image icons in exactly the
room where someone needs it is worse than no report, because the failure looks like the
model rather than the transport.

**The captions carry the judgement, not the label.** "Figure 3: residuals" tells a reader
what they can already see. What they cannot reconstruct on their own is what a *good* one
looks like and which shape would be a finding — so every figure is captioned with both, and
the omitted ones are captioned with the reason they are missing. The omissions are rendered
on the same page as the figures, deliberately: a reader must be able to see that a run has
no SHAP attribution, and the only place they would ever look is here.

The page reads in the reader's own colour scheme, but each figure sits on a light plate
because the PNGs are drawn on a light surface. See :mod:`aegis_ml.report.theme`.
"""

from __future__ import annotations

import base64
import html
import json
from typing import TYPE_CHECKING, Any

from aegis_ml.report import theme

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping
    from pathlib import Path

__all__ = ["render_index", "write_index"]

_LAYOUT_CSS = """
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 1.25rem 4rem;
  background: var(--page); color: var(--ink);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial,
        sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; }
header.verdict {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 1.5rem 1.6rem; margin: 2rem 0 1.5rem;
}
header.verdict h1 { margin: 0 0 .35rem; font-size: 1.4rem; letter-spacing: -.01em; }
header.verdict .runid {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82rem;
  color: var(--muted); word-break: break-all;
}
.chips { display: flex; flex-wrap: wrap; gap: .45rem; margin: .9rem 0 1.1rem; }
.chip {
  font-size: .78rem; font-weight: 600; padding: .22rem .6rem; border-radius: 999px;
  border: 1px solid currentColor; white-space: nowrap;
}
.chip.good { color: var(--good); } .chip.warn { color: var(--warn); }
.chip.bad { color: var(--bad); } .chip.plain { color: var(--muted); }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1px;
         background: var(--line); border: 1px solid var(--line); border-radius: 10px;
         overflow: hidden; }
.stat { background: var(--card); padding: .8rem .9rem; }
.stat .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
           color: var(--muted); }
.stat .v { font-size: 1.15rem; font-weight: 650; margin-top: .18rem; word-break: break-word; }
.stat .s { font-size: .75rem; color: var(--muted); margin-top: .1rem; }
.note { font-size: .84rem; color: var(--muted); margin-top: 1rem; border-left: 3px solid
        var(--line); padding-left: .8rem; }
figure {
  margin: 0 0 1.6rem; background: var(--card); border: 1px solid var(--line);
  border-radius: 14px; overflow: hidden;
}
figure > .head { padding: 1.05rem 1.3rem .2rem; }
figure h2 { margin: 0; font-size: 1.03rem; letter-spacing: -.005em; }
figure .plate { background: var(--plate); padding: 1rem 1.1rem; }
figure img { display: block; width: 100%; height: auto; }
figcaption { padding: .95rem 1.3rem 1.1rem; font-size: .9rem; color: var(--ink); }
figcaption .lead { color: var(--muted); }
details { padding: 0 1.3rem 1.1rem; font-size: .82rem; color: var(--muted); }
details summary { cursor: pointer; }
details pre {
  background: var(--page); border: 1px solid var(--line); border-radius: 8px;
  padding: .7rem .85rem; overflow-x: auto; font-size: .76rem; line-height: 1.45;
}
.omitted { border-style: dashed; }
.omitted .why { padding: 0 1.3rem 1.15rem; font-size: .88rem; color: var(--bad); }
h3.section { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em;
             color: var(--muted); margin: 2.2rem 0 .9rem; }
a { color: var(--primary); }
footer { margin-top: 2.5rem; font-size: .78rem; color: var(--muted); }
footer code { word-break: break-all; }
table.src { width: 100%; border-collapse: collapse; font-size: .78rem; }
table.src td { padding: .22rem .5rem .22rem 0; vertical-align: top; color: var(--muted); }
table.src td:first-child { white-space: nowrap; color: var(--ink); font-weight: 600; }
"""


def _esc(value: object) -> str:
    """HTML-escape any value for text content."""
    return html.escape(str(value), quote=True)


def _data_uri(path: Path) -> str:
    """Return a PNG file as a base64 ``data:`` URI.

    Args:
        path: The image file.

    Returns:
        A URI usable directly as an ``<img src>``.
    """
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _chip(label: str, tone: str) -> str:
    """Render one status chip."""
    return f'<span class="chip {tone}">{_esc(label)}</span>'


def _stat(key: str, value: str, sub: str = "") -> str:
    """Render one headline statistic tile."""
    trailer = f'<div class="s">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div class="stat"><div class="k">{_esc(key)}</div>'
        f'<div class="v">{_esc(value)}</div>{trailer}</div>'
    )


def _coverage_tone(verdict: Mapping[str, Any]) -> tuple[str, str]:
    """Classify the coverage outcome for the header chip.

    Args:
        verdict: The manifest's verdict block.

    Returns:
        ``(label, tone)`` where tone is one of ``good``/``warn``/``bad``/``plain``.
    """
    requested = float(verdict["requested_coverage"])
    measured = verdict.get("empirical_coverage")
    if measured is None:
        return "coverage not measured", "plain"
    measured = float(measured)
    tolerance = float(verdict.get("coverage_tolerance") or 0.0)
    label = f"coverage {measured:.1%} vs {requested:.0%} requested"
    if measured >= requested:
        return label, "good"
    if measured >= requested - tolerance:
        return f"{label} — inside tolerance", "warn"
    return f"{label} — SHORTFALL", "bad"


def _header(verdict: Mapping[str, Any], recovery: Mapping[str, Any]) -> str:
    """Render the verdict header: the paragraph a reader gets before any figure.

    Everything a reader needs to decide whether the figures below are worth their attention
    lands here — what was trained, what it scored, whether the interval kept its promise,
    whether the gate let it through, and whether the world has moved since. The chips carry
    the outcomes; the tiles carry the numbers behind them.

    Args:
        verdict: The manifest's verdict block.
        recovery: The manifest's split-recovery block, quoted so the reader knows whether
            the held-out figures are drawn on rows that were proven to be held out.

    Returns:
        An HTML fragment.
    """
    chips = [_chip(f"stage: {verdict['stage']}", "plain")]
    promoted = verdict.get("gate_promoted")
    if promoted is True:
        chips.append(_chip("gate: PROMOTED", "good"))
    elif promoted is False:
        chips.append(_chip("gate: NOT promoted", "bad"))
    else:
        chips.append(_chip("gate: not evaluated", "plain"))
    label, tone = _coverage_tone(verdict)
    chips.append(_chip(label, tone))
    drift = verdict.get("drift_verdict")
    if drift:
        tones = {"pass": "good", "warn": "warn", "block": "bad"}
        share = verdict.get("drifted_share")
        suffix = f" ({float(share):.0%} of features)" if isinstance(share, int | float) else ""
        chips.append(_chip(f"drift: {drift}{suffix}", tones.get(str(drift), "plain")))
    else:
        chips.append(_chip("drift: not measured", "plain"))
    if verdict.get("tier"):
        chips.append(_chip(f"tier: {verdict['tier']}", "plain"))

    measured = verdict.get("empirical_coverage")
    tiles = [
        _stat(
            verdict["metric_name"],
            f"{float(verdict['metric_value']):.4g}",
            f"measured on {verdict['test_size']} held-out rows",
        ),
        _stat(
            "coverage",
            "not measured" if measured is None else f"{float(measured):.2%}",
            f"requested {float(verdict['requested_coverage']):.0%}",
        ),
        _stat("task", str(verdict["task"]), f"target: {verdict['target']}"),
        _stat(
            "rows",
            f"{verdict['training_size']} / {verdict['calibration_size']} / {verdict['test_size']}",
            "train / calibration / test",
        ),
    ]
    if verdict.get("estimated_metric_name"):
        value = verdict.get("estimated_metric_value")
        tiles.append(
            _stat(
                str(verdict["estimated_metric_name"]),
                "n/a" if value is None else f"{float(value):.4g}",
                "ESTIMATED without labels — not a measurement",
            )
        )

    provenance = (
        "Figures drawn on held-out rows are drawn on rows proven to be held out: "
        + _esc(recovery.get("reason", ""))
        if recovery.get("ok")
        else "Held-out figures were omitted. " + _esc(recovery.get("reason", ""))
    )
    digest = verdict.get("dataset_digest") or "not recorded"
    return f"""
<header class="verdict">
  <h1>{_esc(verdict["domain_id"])} — {_esc(verdict["target"])}</h1>
  <div class="runid">{_esc(verdict["run_id"])} · registered {_esc(verdict["created_at"])}</div>
  <div class="chips">{"".join(chips)}</div>
  <div class="stats">{"".join(tiles)}</div>
  <div class="note">{provenance}</div>
  <div class="note">dataset digest: <code>{_esc(digest)}</code></div>
</header>
"""


def _figure(row: Mapping[str, Any], directory: Path) -> str:
    """Render one figure card, or one omission card with its reason.

    Args:
        row: A manifest ``plots`` entry.
        directory: The visuals directory, for locating the PNG to inline.

    Returns:
        An HTML fragment.
    """
    title = _esc(row["title"])
    caption = _esc(row["caption"])
    if row["status"] != "rendered":
        return f"""
<figure class="omitted">
  <div class="head"><h2>{title}</h2></div>
  <div class="why"><strong>Omitted.</strong> {_esc(row.get("reason") or "no reason recorded")}
  </div>
  <figcaption class="lead">{caption}</figcaption>
</figure>
"""
    image = directory / str(row["file"])
    numbers = json.dumps(row.get("numbers") or {}, indent=2, default=str)
    return f"""
<figure>
  <div class="head"><h2>{title}</h2></div>
  <div class="plate"><img alt="{title}" src="{_data_uri(image)}"></div>
  <figcaption>{caption}</figcaption>
  <details>
    <summary>the numbers behind this figure — {_esc(row["file"])}, from
      {_esc(", ".join(row.get("inputs") or []))}</summary>
    <pre>{_esc(numbers)}</pre>
  </details>
</figure>
"""


def render_index(directory: Path, manifest: Mapping[str, Any]) -> str:
    """Render the whole page as one self-contained HTML string.

    Args:
        directory: The visuals directory holding the PNGs to inline.
        manifest: The bundle manifest, which is the only source of what to show. The page
            never scans the directory for images — a file the manifest does not describe is
            a file nobody can say where the numbers came from.

    Returns:
        The complete document.
    """
    verdict = dict(manifest.get("verdict") or {})
    recovery = dict(manifest.get("split_recovery") or {})
    rows = list(manifest.get("plots") or [])
    rendered = [row for row in rows if row["status"] == "rendered"]
    omitted = [row for row in rows if row["status"] != "rendered"]

    interactive = dict(manifest.get("interactive") or {})
    if interactive.get("file"):
        link = (
            f'<p><a href="{_esc(interactive["file"])}">Open the interactive version</a> — '
            f"same measurements, hover a point to read its row "
            f"({_esc(', '.join(interactive.get('panels') or []))})."
            f"</p>"
        )
    else:
        link = (
            f'<p class="note">No interactive page was written: '
            f"{_esc(interactive.get('reason') or 'no panel had data in it')}.</p>"
        )

    sources = "".join(
        f"<tr><td>{_esc(name)}</td><td>{_esc(path)}</td></tr>"
        for name, path in sorted((manifest.get("sources") or {}).items())
    )
    gate_reasons = "".join(
        f"<li>{_esc(reason)}</li>" for reason in (verdict.get("gate_reasons") or [])
    )
    notes = "".join(f"<li>{_esc(note)}</li>" for note in (verdict.get("notes") or []))

    body = [
        _header(verdict, recovery),
        link,
        f'<h3 class="section">{len(rendered)} figures, every number measured by this run</h3>',
        *(_figure(row, directory) for row in rendered),
    ]
    if omitted:
        body.append(
            f'<h3 class="section">{len(omitted)} figures omitted — and why</h3>'
            f'<p class="note">An input this run does not have is recorded rather than '
            f"drawn. A blank axis with a title reads as evidence and is not.</p>"
        )
        body.extend(_figure(row, directory) for row in omitted)

    if gate_reasons:
        body.append(f'<h3 class="section">Gate decision</h3><ul>{gate_reasons}</ul>')
    if notes:
        body.append(f'<h3 class="section">Run notes</h3><ul>{notes}</ul>')

    body.append(
        f"""
<footer>
  <h3 class="section">Where every number came from</h3>
  <table class="src">{sources}</table>
  <p>Bundle generated {_esc(manifest.get("generated_at", ""))} by
     <code>aegis_ml.report.bundle.build_bundle</code> ·
     manifest: <code>manifest.json</code> ·
     rebuild with <code>aegis-ml visuals {_esc(verdict.get("run_id", ""))}</code></p>
</footer>
"""
    )

    title = f"{verdict.get('domain_id', 'run')} — visual report"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{theme.stylesheet()}{_LAYOUT_CSS}</style>\n"
        "</head>\n<body>\n<div class=\"wrap\">\n"
        + "\n".join(body)
        + "\n</div>\n</body>\n</html>\n"
    )


def write_index(directory: Path, manifest: Mapping[str, Any]) -> Path:
    """Render the page and write it as ``index.html`` inside ``directory``.

    Args:
        directory: The visuals directory.
        manifest: The bundle manifest.

    Returns:
        The path written.
    """
    target = directory / "index.html"
    target.write_text(render_index(directory, manifest), encoding="utf-8")
    return target
