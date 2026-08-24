"""One palette and one set of rcParams, so a run's figures read as a single report.

Every plot in ``registry_store/runs/<run_id>/visuals/`` is looked at next to the others,
usually within a minute of each other, by someone deciding whether to trust the model. If
each figure picks its own colours, the reader spends that minute learning three different
encodings instead of reading the evidence. So the encoding is fixed here, once:

* :data:`PALETTE` — the semantic colours. ``good``/``warn``/``bad`` mean the same thing on
  the coverage chart as on the slice chart; ``accent`` **always** marks the one thing the
  reader must not miss (the worst slice, the promoted candidate, the drifted feature).
* :data:`TIER_COLOURS` — one colour per AutoML tier, so the leaderboard's grouping matches
  the vocabulary the model card and the manifest already use.
* :func:`stylesheet` — the *same* hex values as CSS custom properties, so ``index.html``
  cannot drift away from the PNGs it embeds.

The figures are rendered on a light surface even though the page around them adapts to the
reader's theme. That is deliberate: a PNG carries its own background, and an image drawn
for a dark page is unreadable when someone prints it or pastes it into a document. The page
puts every figure on a light plate instead, which costs a little contrast in dark mode and
buys a report that survives being screenshotted.

:func:`apply` forces the ``Agg`` backend before ``pyplot`` is imported. These figures are
rendered inside a pipeline stage on machines with no display; a backend that tries to open
a window there fails at ``savefig`` time, several minutes into a run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aegis_ml._require import require

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

__all__ = [
    "FIGURE_DPI",
    "PALETTE",
    "SEQUENCE",
    "TIER_COLOURS",
    "apply",
    "save",
    "stylesheet",
]

SERVE_EXTRA = "aegis-ml[serve]"
"""Install target named verbatim in every ImportError this package raises."""

FIGURE_DPI = 144
"""Rendering density. High enough that a 6-inch figure is legible when a reader zooms into
the embedded base64 image; low enough that ten of them stay well under a megabyte each."""

PALETTE: dict[str, str] = {
    "ink": "#1B2430",
    "muted": "#6B7785",
    "grid": "#DFE3E8",
    "surface": "#FFFFFF",
    "canvas": "#F4F6F8",
    "primary": "#2F6D8E",
    "primary_soft": "#A8C7D8",
    "accent": "#B4562C",
    "good": "#3D7A5A",
    "warn": "#B08422",
    "bad": "#A33A3A",
    "neutral": "#B3BBC3",
}
"""Semantic colours. ``accent`` is reserved for the single element a reader must not miss —
using it for decoration is what makes a highlight stop meaning anything."""

TIER_COLOURS: dict[str, str] = {
    "baseline": "#6B7785",
    "flaml": "#2F6D8E",
    "autogluon": "#3D7A5A",
    "tabpfn": "#7A4F9B",
}
"""One colour per AutoML tier, matching the four names in
:data:`aegis_ml.contracts.protocols.TierName`. An unknown tier falls back to ``neutral``
rather than being recoloured silently."""

SEQUENCE: tuple[str, ...] = (
    "#2F6D8E",
    "#B4562C",
    "#3D7A5A",
    "#7A4F9B",
    "#B08422",
    "#A33A3A",
    "#4C7A93",
    "#8A6F4E",
)
"""Categorical sequence for the rare chart that needs more than two series (drift overlays,
per-class probability histograms). Ordered so the first two are distinguishable in the most
common forms of colour-blindness and in greyscale."""


def apply() -> Any:  # noqa: ANN401 - the matplotlib.pyplot module object
    """Force the headless backend, install the palette, and return ``pyplot``.

    Called at the top of every plotting function rather than once at import time. Importing
    ``pyplot`` is expensive (roughly a third of a second, plus a font-cache build on a cold
    machine) and this package is imported by the CLI's module scan, which must stay fast.
    Applying the rcParams on every call is cheap and makes each function independent of
    whatever a caller did to the global state in between.

    Returns:
        The ``matplotlib.pyplot`` module, with the palette applied.

    Raises:
        ImportError: When matplotlib or seaborn is not installed, naming the install.
    """
    matplotlib = require(SERVE_EXTRA, "matplotlib")
    matplotlib.use("Agg", force=True)
    seaborn = require(SERVE_EXTRA, "seaborn")
    plt = require(SERVE_EXTRA, "matplotlib.pyplot")

    # seaborn.set_theme resets rcParams wholesale, so it goes first and the overrides that
    # matter to this report go after it. The reverse order silently loses every setting.
    seaborn.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "figure.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.autolayout": False,
            "lines.solid_capstyle": "round",
        }
    )
    for spine in ("top", "right"):
        plt.rcParams[f"axes.spines.{spine}"] = False
    return plt


def save(fig: Any, path: Path) -> Path:  # noqa: ANN401 - matplotlib.figure.Figure
    """Write a figure to ``path`` and close it.

    Closing is not tidiness. A training pipeline renders ten figures per run and a long
    session renders hundreds; matplotlib keeps every unclosed figure alive in a global
    registry, and the process grows until it is killed by the OOM killer partway through a
    demo. Closing here means no caller can forget.

    Args:
        fig: The figure to write.
        path: Destination file. Its parent directory is created.

    Returns:
        ``path``, for chaining into a manifest entry.
    """
    plt = require(SERVE_EXTRA, "matplotlib.pyplot")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)
    return path


def stylesheet() -> str:
    """Return the report's CSS, carrying the same hex values as :data:`PALETTE`.

    The page adapts to the reader's theme, but every figure plate stays light because the
    PNGs it holds are drawn on a light surface. Redefining only the page chrome keeps the
    contrast correct in both themes without re-rendering anything.

    Returns:
        A complete stylesheet, ready to inline into ``<style>``.
    """
    return f"""
:root {{
  --page: {PALETTE["canvas"]};
  --card: {PALETTE["surface"]};
  --ink: {PALETTE["ink"]};
  --muted: {PALETTE["muted"]};
  --line: {PALETTE["grid"]};
  --primary: {PALETTE["primary"]};
  --accent: {PALETTE["accent"]};
  --good: {PALETTE["good"]};
  --warn: {PALETTE["warn"]};
  --bad: {PALETTE["bad"]};
  --plate: {PALETTE["surface"]};
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --page: #12171D;
    --card: #1B222A;
    --ink: #E7ECF1;
    --muted: #9BA6B2;
    --line: #2B333C;
    --primary: #7FB3CC;
    --accent: #E08A5A;
    --good: #6FB58C;
    --warn: #D8B25E;
    --bad: #D97070;
  }}
}}
"""
