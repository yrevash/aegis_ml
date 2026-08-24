"""Design tokens and the hub's stylesheet — one visual system, no external requests.

The hub is projected onto a screen in front of people who have never seen this repository.
Whatever is on it has about ten seconds to say *"this number was measured, and here is the
evidence"*. That is a design problem as much as a data problem, so the rules are fixed here
rather than negotiated per component:

* **Dark first.** ``:root`` carries the dark palette because that is what a projector in a
  lit room renders best, and because the nine PNGs the run already produced are drawn on a
  *light* plate — a dark page around a light figure frames it like a print on a wall, where
  a light page around it makes the figure disappear into the background. The light palette
  is a full re-declaration, not an afterthought, and it is reachable three ways: the
  system preference, an explicit ``data-theme`` attribute, and the header toggle.
* **One accent.** :data:`ACCENT` is copper, inherited in spirit from
  :data:`aegis_ml.report.theme.PALETTE`'s ``accent``, and it means exactly what it means
  there: *the one thing the reader must not miss*. Verdicts (pass / warn / fail) use the
  semantic triple instead. An accent used for decoration stops being a signal, so it marks
  the active nav row, the focus ring, and the primary metric — nothing else.
* **System fonts only.** No CDN, no ``@font-face``, no network. The page must render
  identically on a laptop with the wifi switched off, because that is the machine it will
  be demoed from. Numbers are set in the platform monospace with ``tabular-nums`` so a
  column of coverage figures lines up on the decimal point.

Nothing here imports matplotlib. :mod:`aegis_ml.report.theme` is imported only for its
:data:`~aegis_ml.report.theme.PALETTE` and :data:`~aegis_ml.report.theme.TIER_COLOURS`
dictionaries, which are plain module-level data — so the hub and the PNGs it displays
cannot drift onto two different colour vocabularies for the same tier name.
"""

from __future__ import annotations

from aegis_ml.report.theme import TIER_COLOURS

__all__ = [
    "ACCENT",
    "DARK",
    "LIGHT",
    "MONO_STACK",
    "SANS_STACK",
    "TIER_COLOURS",
    "stylesheet",
    "tier_colour",
    "tokens_css",
]

ACCENT = "#E2743B"
"""Copper. The single accent, reserved for the element a reader must not miss."""

SANS_STACK = (
    'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
    '"Helvetica Neue", Arial, "Noto Sans", sans-serif'
)
"""Platform UI face. Resolves to SF Pro, Segoe UI or Roboto without a single byte fetched."""

MONO_STACK = (
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace'
)
"""Platform monospace, used for every figure so digits align in a column."""

DARK: dict[str, str] = {
    "bg": "#0A0E13",
    "bg-grad": "#0D131A",
    "surface": "#121A24",
    "surface-2": "#18222E",
    "surface-3": "#1E2A38",
    "line": "#22303E",
    "line-strong": "#31404F",
    "ink": "#E8EEF6",
    "ink-2": "#A5B3C3",
    "ink-3": "#6B7A8B",
    "accent": ACCENT,
    "accent-ink": "#F6A671",
    "accent-soft": "rgba(226, 116, 59, 0.16)",
    "good": "#4FB286",
    "good-soft": "rgba(79, 178, 134, 0.14)",
    "warn": "#D4B04A",
    "warn-soft": "rgba(212, 176, 74, 0.14)",
    "bad": "#E0645E",
    "bad-soft": "rgba(224, 100, 94, 0.14)",
    "info": "#6BA6DA",
    "info-soft": "rgba(107, 166, 218, 0.14)",
    "plate": "#FFFFFF",
    "shadow": "0 1px 0 rgba(255,255,255,0.03) inset, 0 12px 32px -20px rgba(0,0,0,0.9)",
    "ring": "rgba(226, 116, 59, 0.45)",
}
"""The dark palette, which is the default. ``plate`` stays white in both themes: the PNGs
carry their own light background and must not sit on a surface that fights it."""

LIGHT: dict[str, str] = {
    "bg": "#EEF1F5",
    "bg-grad": "#F5F7FA",
    "surface": "#FFFFFF",
    "surface-2": "#F7F9FB",
    "surface-3": "#EFF3F7",
    "line": "#E0E6EC",
    "line-strong": "#C6D0DA",
    "ink": "#111922",
    "ink-2": "#4F5C6B",
    "ink-3": "#7C8A99",
    "accent": "#BE5A22",
    "accent-ink": "#9C4718",
    "accent-soft": "rgba(190, 90, 34, 0.10)",
    "good": "#2E7A56",
    "good-soft": "rgba(46, 122, 86, 0.10)",
    "warn": "#8A6910",
    "warn-soft": "rgba(138, 105, 16, 0.12)",
    "bad": "#AE3B36",
    "bad-soft": "rgba(174, 59, 54, 0.10)",
    "info": "#2F6D8E",
    "info-soft": "rgba(47, 109, 142, 0.10)",
    "plate": "#FFFFFF",
    "shadow": "0 1px 2px rgba(16,24,32,0.04), 0 12px 28px -22px rgba(16,24,32,0.45)",
    "ring": "rgba(190, 90, 34, 0.35)",
}
"""The light palette. A complete re-declaration of every token in :data:`DARK` — a partial
override is how a theme ends up with one unreadable component nobody notices until a demo."""


def tier_colour(tier: str) -> str:
    """Return the swatch for an AutoML tier name, matching the leaderboard PNG.

    Args:
        tier: A tier name as written by :mod:`aegis_ml.automl.tiers` (``baseline``,
            ``flaml``, ``autogluon``, ``tabpfn``).

    Returns:
        A hex colour. An unrecognised tier gets the neutral ink rather than being
        recoloured onto some other tier's swatch, which would make the hub and the PNG
        disagree about which tier won.
    """
    return TIER_COLOURS.get(tier, DARK["ink-3"])


def _block(tokens: dict[str, str]) -> str:
    """Render one palette as CSS custom property declarations."""
    return "\n".join(f"  --{name}: {value};" for name, value in tokens.items())


def tokens_css() -> str:
    """Return the custom-property declarations for both themes.

    Dark is declared on bare ``:root`` so it is what renders when nothing else matches.
    Light arrives two ways — the system preference (guarded so an explicit dark choice
    still wins) and the ``data-theme="light"`` attribute the header toggle sets — and dark
    is re-declared under its own attribute so the toggle works in both directions.

    Returns:
        A CSS fragment, ready to concatenate into the stylesheet.
    """
    dark = _block(DARK)
    light = _block(LIGHT)
    return f""":root {{
  color-scheme: dark light;
{dark}
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    color-scheme: light;
{light}
  }}
}}
:root[data-theme="light"] {{
  color-scheme: light;
{light}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
{dark}
}}
"""


_LAYOUT = """
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(1200px 600px at 78% -12%, var(--accent-soft), transparent 62%),
    linear-gradient(180deg, var(--bg-grad), var(--bg) 42%);
  background-attachment: fixed;
  color: var(--ink);
  font-family: %(sans)s;
  font-size: 15px;
  line-height: 1.55;
  font-feature-settings: "cv05" 1, "ss01" 1;
  -webkit-font-smoothing: antialiased;
}
a { color: inherit; text-decoration: none; }
button { font: inherit; color: inherit; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 6px; }

/* ── shell ─────────────────────────────────────────────────────────────── */
.shell { display: grid; grid-template-columns: 250px minmax(0, 1fr); min-height: 100vh; }
/* The rail is the full-height grid cell so its rule reaches the bottom of a long page;
   the sticky viewport-height panel lives inside it. Making the rail itself sticky puts
   the border on a 100vh box, which stops a third of the way down a run detail page. */
.rail {
  border-right: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent 22rem);
}
.rail-inner {
  position: sticky; top: 0; height: 100vh;
  display: flex; flex-direction: column; gap: 1.5rem;
  padding: 1.5rem 1rem 1.25rem 1.5rem;
  overflow-y: auto;
}
.brand { display: flex; align-items: center; gap: .7rem; }
.brand .glyph {
  width: 30px; height: 30px; border-radius: 9px; flex: none;
  background: linear-gradient(145deg, var(--accent), #9c4718);
  display: grid; place-items: center;
  font: 700 13px/1 %(mono)s; color: #FFF7F1; letter-spacing: -.04em;
  box-shadow: 0 6px 18px -8px var(--accent);
}
.brand b { font-size: 15px; font-weight: 650; letter-spacing: -.015em; display: block; }
.brand span { font-size: 11px; color: var(--ink-3); letter-spacing: .04em; }
.nav { display: flex; flex-direction: column; gap: 2px; }
.nav-label {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .12em;
  color: var(--ink-3); font-weight: 600; padding: 0 .55rem; margin-bottom: .45rem;
}
.nav a {
  display: flex; align-items: center; gap: .6rem;
  padding: .48rem .55rem; border-radius: 8px;
  font-size: 13.5px; color: var(--ink-2); position: relative;
  transition: background .14s ease, color .14s ease;
}
.nav a:hover { background: var(--surface-2); color: var(--ink); }
.nav a.on { background: var(--accent-soft); color: var(--ink); font-weight: 550; }
.nav a.on::before {
  content: ""; position: absolute; left: -0.7rem; top: 50%; translate: 0 -50%;
  width: 3px; height: 17px; border-radius: 3px; background: var(--accent);
}
.nav .ic { width: 15px; text-align: center; opacity: .78; font-size: 13px; }
.nav .tail { margin-left: auto; font: 500 10.5px/1 %(mono)s; color: var(--ink-3); }
.rail-foot { margin-top: auto; display: flex; flex-direction: column; gap: .55rem; }
.svc-row {
  display: flex; align-items: center; gap: .55rem;
  font-size: 12px; color: var(--ink-2);
  padding: .34rem .5rem; border-radius: 7px; background: var(--surface);
  border: 1px solid var(--line);
}
.svc-row .name { font-weight: 550; color: var(--ink); }
.svc-row .port { margin-left: auto; font: 500 11px/1 %(mono)s; color: var(--ink-3); }
.dot { width: 7px; height: 7px; border-radius: 50%; flex: none; background: var(--ink-3); }
.dot.up { background: var(--good); box-shadow: 0 0 0 3px var(--good-soft); }
.dot.down { background: var(--bad); box-shadow: 0 0 0 3px var(--bad-soft); }

.main { min-width: 0; padding: 1.75rem 2.25rem 5rem; }
.wrap { max-width: 1320px; margin: 0 auto; }

/* ── page head ─────────────────────────────────────────────────────────── */
.head { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1.6rem; }
.head .grow { min-width: 0; flex: 1; }
.eyebrow {
  font-size: 11px; text-transform: uppercase; letter-spacing: .14em;
  color: var(--ink-3); font-weight: 600; margin-bottom: .45rem;
}
h1 {
  margin: 0; font-size: clamp(1.6rem, 2.4vw, 2.1rem); font-weight: 660;
  letter-spacing: -.028em; line-height: 1.15;
}
h2 {
  margin: 0 0 .9rem; font-size: 1.02rem; font-weight: 620; letter-spacing: -.012em;
}
h3 { margin: 0 0 .5rem; font-size: .86rem; font-weight: 620; letter-spacing: -.005em; }
.sub { color: var(--ink-2); font-size: 13.5px; margin-top: .4rem; }
.mono { font-family: %(mono)s; font-variant-numeric: tabular-nums; }
.section { margin: 2.25rem 0 0; }
.section > .bar {
  display: flex; align-items: baseline; gap: .75rem; margin-bottom: .9rem;
}
.section > .bar h2 { margin: 0; }
.section > .bar .note { font-size: 12px; color: var(--ink-3); }

/* ── controls ──────────────────────────────────────────────────────────── */
.btn {
  display: inline-flex; align-items: center; gap: .45rem;
  padding: .42rem .8rem; border-radius: 8px; cursor: pointer;
  border: 1px solid var(--line-strong); background: var(--surface);
  font-size: 12.5px; font-weight: 550; color: var(--ink-2);
  transition: background .14s ease, color .14s ease, border-color .14s ease;
}
.btn:hover { background: var(--surface-2); color: var(--ink); border-color: var(--ink-3); }
.btn.primary {
  background: var(--accent); border-color: var(--accent); color: #17100B; font-weight: 620;
}
.btn.primary:hover { filter: brightness(1.08); color: #17100B; }
select.btn { appearance: none; padding-right: 1.6rem;
  background-image: linear-gradient(45deg, transparent 50%, var(--ink-3) 50%),
                    linear-gradient(135deg, var(--ink-3) 50%, transparent 50%);
  background-position: right .72rem center, right .48rem center;
  background-size: 5px 5px, 5px 5px; background-repeat: no-repeat; }

/* ── cards ─────────────────────────────────────────────────────────────── */
.card {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 14px; box-shadow: var(--shadow);
}
.card.pad { padding: 1.15rem 1.25rem; }
.grid { display: grid; gap: 1rem; }
.cols-2 { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
.cols-3 { grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }

/* ── stat tiles ────────────────────────────────────────────────────────── */
/* Four fixed columns with the primary metric spanning two: seven tiles then fill two rows
   of four exactly. `auto-fit` was leaving the last two tiles orphaned on a second row at
   laptop width, which reads as an unfinished layout rather than a deliberate one. */
.stats { display: grid; gap: .85rem; grid-template-columns: repeat(4, minmax(0, 1fr)); }
.stats .tile.lead { grid-column: span 2; }
@media (max-width: 1240px) {
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 560px) {
  .stats { grid-template-columns: minmax(0, 1fr); }
  .stats .tile.lead { grid-column: span 1; }
}
.tile {
  position: relative; overflow: hidden;
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 14px; padding: 1rem 1.1rem 1.05rem; box-shadow: var(--shadow);
}
.tile .k {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .12em;
  color: var(--ink-3); font-weight: 600;
}
.tile .v {
  margin-top: .5rem; font-family: %(mono)s; font-variant-numeric: tabular-nums;
  font-size: clamp(1.7rem, 2.8vw, 2.3rem); font-weight: 600; letter-spacing: -.035em;
  line-height: 1.05; word-break: break-word;
}
.tile .s { margin-top: .35rem; font-size: 12px; color: var(--ink-2); }
.tile.lead::after {
  content: ""; position: absolute; inset: auto 0 0 0; height: 2px; background: var(--accent);
}
.tile.good .v { color: var(--good); }
.tile.warn .v { color: var(--warn); }
.tile.bad  .v { color: var(--bad); }
.tile .spark { display: block; margin-top: .55rem; }

/* ── chips ─────────────────────────────────────────────────────────────── */
.chip {
  display: inline-flex; align-items: center; gap: .35rem;
  padding: .16rem .5rem; border-radius: 999px;
  font-size: 11px; font-weight: 600; letter-spacing: .02em;
  border: 1px solid transparent; white-space: nowrap;
}
.chip.good { background: var(--good-soft); color: var(--good); border-color: var(--good); }
.chip.warn { background: var(--warn-soft); color: var(--warn); border-color: var(--warn); }
.chip.bad  { background: var(--bad-soft);  color: var(--bad);  border-color: var(--bad); }
.chip.info { background: var(--info-soft); color: var(--info); border-color: var(--info); }
.chip.flat { background: var(--surface-2); color: var(--ink-2); border-color: var(--line); }
.chip.tier::before {
  content: ""; width: 7px; height: 7px; border-radius: 2px; background: var(--tier, var(--ink-3));
}

/* ── run list ──────────────────────────────────────────────────────────── */
.runs { display: flex; flex-direction: column; gap: .5rem; }
.run {
  display: grid; align-items: center; gap: 1rem;
  grid-template-columns: 8px minmax(0, 2.1fr) 110px 130px 130px auto;
  padding: .85rem 1.1rem; border-radius: 12px;
  background: var(--surface); border: 1px solid var(--line);
  cursor: pointer; transition: border-color .14s ease, transform .14s ease, background .14s ease;
}
.run:hover {
  border-color: var(--line-strong); background: var(--surface-2);
  transform: translateY(-1px);
}
.run .stagebar { width: 3px; height: 30px; border-radius: 3px; background: var(--ink-3); }
.run .stagebar.production { background: var(--good); }
.run .stagebar.staging { background: var(--info); }
.run .stagebar.archived { background: var(--ink-3); }
.run .id { font-family: %(mono)s; font-size: 12.5px; font-weight: 550;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.run .meta { font-size: 11.5px; color: var(--ink-3); margin-top: .18rem; }
.run .num { font-family: %(mono)s; font-variant-numeric: tabular-nums; font-size: 13.5px; }
.run .cap { font-size: 10px; text-transform: uppercase; letter-spacing: .1em;
  color: var(--ink-3); font-weight: 600; }
.run .go { color: var(--ink-3); font-size: 15px; }
.run:hover .go { color: var(--accent); }

/* ── criteria / reasons ────────────────────────────────────────────────── */
.crit { display: flex; flex-direction: column; gap: .1rem; }
.crit-row {
  display: grid; grid-template-columns: 18px minmax(0, 1fr); gap: .7rem;
  align-items: start; padding: .55rem 0; border-top: 1px solid var(--line);
}
.crit-row:first-child { border-top: 0; }
.crit-row .mark { font-size: 12px; line-height: 1.5; font-weight: 700; }
.crit-row .mark.ok { color: var(--good); }
.crit-row .mark.no { color: var(--bad); }
.crit-row .why { font-size: 12.5px; color: var(--ink-2); line-height: 1.5; }
.crit-row .why b { color: var(--ink); font-weight: 600; }

/* ── key/value table ───────────────────────────────────────────────────── */
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: .35rem 1.2rem; }
.kv dt { font-size: 12px; color: var(--ink-3); }
.kv dd { margin: 0; font-size: 12.5px; font-family: %(mono)s;
  font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }

/* ── gallery ───────────────────────────────────────────────────────────── */
.gal { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }
.fig {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 14px; overflow: hidden; box-shadow: var(--shadow);
  display: flex; flex-direction: column;
}
.fig .plate {
  background: var(--plate); cursor: zoom-in; padding: .5rem; display: block;
  border-bottom: 1px solid var(--line);
}
.fig .plate img { display: block; width: 100%; height: auto; border-radius: 6px; }
.fig .cap { padding: .8rem 1rem 1rem; }
.fig .cap h3 { margin: 0 0 .35rem; font-size: 13px; }
.fig .cap p { margin: 0; font-size: 11.5px; color: var(--ink-3); line-height: 1.55; }
.fig .cap p.more { display: none; }
.fig .cap.open p.more { display: block; margin-top: .5rem; }
.fig .cap .toggle {
  margin-top: .5rem; font-size: 11px; color: var(--accent-ink);
  background: none; border: 0; padding: 0; cursor: pointer; font-weight: 600;
}

/* ── lightbox ──────────────────────────────────────────────────────────── */
.lb {
  position: fixed; inset: 0; z-index: 60; display: none;
  background: rgba(4, 7, 10, .93); backdrop-filter: blur(6px);
  padding: 2.5rem 3.5rem 4rem;
}
.lb.on { display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 1rem; }
.lb-bar { display: flex; align-items: center; gap: 1rem; color: #E8EEF6; }
.lb-bar .t { font-size: 13.5px; font-weight: 600; }
.lb-bar .n { font: 500 12px/1 %(mono)s; color: #8A98A8; }
.lb-bar .sp { margin-left: auto; display: flex; gap: .5rem; }
.lb-bar button {
  border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.06);
  color: #E8EEF6; border-radius: 8px; padding: .3rem .7rem; cursor: pointer; font-size: 12.5px;
}
.lb-bar button:hover { background: rgba(255,255,255,.14); }
/* Flex rather than grid centring: a percentage max-height on a `place-items: center` grid
   item resolves against a track the item is itself sizing, and the tall figures (the slice
   chart is twenty bars) then run off the bottom of the screen. */
.lb-img {
  display: flex; align-items: center; justify-content: center;
  min-height: 0; overflow: hidden;
}
.lb-img img {
  max-width: 100%; max-height: 100%; width: auto; height: auto;
  border-radius: 10px; background: #fff; padding: .5rem; object-fit: contain;
}

/* ── tabs / links ──────────────────────────────────────────────────────── */
.tabs { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: 1.1rem; }
.tab {
  display: inline-flex; align-items: center; gap: .45rem;
  padding: .45rem .85rem; border-radius: 9px; font-size: 12.5px; font-weight: 550;
  border: 1px solid var(--line); background: var(--surface); color: var(--ink-2);
  cursor: pointer; transition: all .14s ease;
}
.tab:hover { border-color: var(--line-strong); color: var(--ink); }
.tab.on { background: var(--accent-soft); border-color: var(--accent); color: var(--ink); }
.tab.off { opacity: .42; cursor: not-allowed; }

/* ── embedded service frames ───────────────────────────────────────────── */
.frame-wrap {
  border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
  background: var(--plate); box-shadow: var(--shadow);
}
/* Sized so the panel's own header, the status bar and the frame fit one viewport with no
   page scroll. An embedded UI focuses an element as it boots and drags the page down with
   it; leaving no room to scroll is what actually keeps the title and the "open in a new
   tab" link on screen, and the load handler in the page script is the second line. */
.frame-wrap iframe {
  display: block; width: 100%; border: 0;
  height: calc(100vh - 20rem); min-height: 360px;
}
.frame-bar {
  display: flex; align-items: center; gap: .7rem; padding: .6rem .9rem;
  background: var(--surface-2); border-bottom: 1px solid var(--line);
  font-size: 12px; color: var(--ink-2);
}
.frame-bar .url { font-family: %(mono)s; font-size: 11.5px; color: var(--ink-3); }
.frame-bar .sp { margin-left: auto; }

/* ── empty / degraded states ───────────────────────────────────────────── */
.empty {
  border: 1px dashed var(--line-strong); border-radius: 14px;
  padding: 2rem 1.75rem; background: var(--surface-2);
}
.empty h3 { font-size: 14.5px; margin-bottom: .5rem; }
.empty p { margin: 0 0 .9rem; font-size: 13px; color: var(--ink-2); max-width: 66ch;
  line-height: 1.6; }
.empty .why { color: var(--warn); font-weight: 550; }
code, pre.cmd {
  font-family: %(mono)s; font-size: 12px;
  background: var(--surface-3); border: 1px solid var(--line);
  border-radius: 7px; padding: .12rem .38rem; color: var(--ink);
}
pre.cmd {
  display: block; padding: .7rem .9rem; overflow-x: auto; margin: 0;
  white-space: pre; line-height: 1.6;
}

/* ── compare ───────────────────────────────────────────────────────────── */
.cmp { width: 100%; border-collapse: collapse; font-size: 13px; }
.cmp th, .cmp td { padding: .7rem .9rem; text-align: left; border-top: 1px solid var(--line); }
.cmp thead th { border-top: 0; font-size: 10.5px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--ink-3); font-weight: 600; }
.cmp td.n { font-family: %(mono)s; font-variant-numeric: tabular-nums; }
.cmp td.win { color: var(--good); font-weight: 600; }
.cmp th.row { color: var(--ink-3); font-weight: 500; font-size: 12px; width: 190px; }
.scroll-x { overflow-x: auto; }

/* ── notes ─────────────────────────────────────────────────────────────── */
.notes { display: flex; flex-direction: column; gap: .55rem; margin: 0; padding: 0;
  list-style: none; }
.notes li {
  font-size: 12.5px; color: var(--ink-2); line-height: 1.6;
  padding-left: .9rem; position: relative;
}
.notes li::before {
  content: ""; position: absolute; left: 0; top: .58em;
  width: 4px; height: 4px; border-radius: 50%; background: var(--ink-3);
}

/* ── responsive ────────────────────────────────────────────────────────── */
@media (max-width: 1060px) {
  .shell { grid-template-columns: 1fr; }
  .rail { border-right: 0; border-bottom: 1px solid var(--line); }
  .rail-inner {
    position: static; height: auto; flex-direction: row; align-items: center;
    flex-wrap: wrap; gap: .75rem; padding: 1rem 1.25rem; overflow: visible;
  }
  .nav { flex-direction: row; flex-wrap: wrap; }
  .nav-label { display: none; }
  .nav a.on::before { display: none; }
  .rail-foot { margin: 0 0 0 auto; flex-direction: row; }
  .main { padding: 1.25rem 1.25rem 4rem; }
}
@media (max-width: 720px) {
  /* Flex, not a narrower grid: the row has six cells and squeezing them into three
     columns leaves auto-placement to decide which number lands under which caption. */
  .run { display: flex; flex-wrap: wrap; align-items: center; gap: .55rem 1.6rem; }
  /* Explicit basis on every cell: flex items default to shrinking, and the metric and
     coverage columns collapse to zero width and print on top of each other without it. */
  .run > div { flex: 0 0 auto; }
  .run > div:nth-child(2) { flex: 1 1 100%; min-width: 0; }
  .run .stagebar { height: 20px; }
  .run .go { margin-left: auto; }
  .lb { padding: 1rem 1rem 2rem; }
  .head { flex-wrap: wrap; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
"""


def stylesheet() -> str:
    """Return the hub's complete stylesheet, tokens included.

    One string, inlined into a single ``<style>`` element. There is no build step and no
    second request: the page is served from a local process that may be the only thing
    running on a machine with no network, and a stylesheet that arrives separately is one
    more thing that can fail in front of an audience.

    Returns:
        CSS text, ready to place inside ``<style>``.
    """
    # Plain substitution rather than %-formatting or an f-string: this text is full of
    # literal `%` (every `width: 100%`) and `{` — both of which those two mechanisms would
    # try to interpret, and neither failure is obvious in a stylesheet.
    return tokens_css() + _LAYOUT.replace("%(sans)s", SANS_STACK).replace("%(mono)s", MONO_STACK)
