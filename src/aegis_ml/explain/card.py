"""The extended model card — every measured claim about one model, in one document.

Aegis already has a ``ModelCard`` (``aegis/src/aegis/ml/types.py``) and it is a good one:
it reads every field off the *actual* fitted spine and, critically, keeps
``conformal_coverage`` (the level **requested**) separate from
``conformal_coverage_empirical`` (the level **measured**). This module **extends** that card;
it does not replace it. The Aegis card's fields ride along as a nested dict
(:attr:`ExtendedModelCard.aegis_card`) rather than as a typed import, because importing
``aegis.ml.types`` here would pull the heavy package into a module the CLI loads at startup —
and because a card must still render for a run whose Aegis card is absent.

What this adds on top of the Aegis card is the MLOps context the spine has no view of: the
AutoML leaderboard (including the losers), the per-segment sweep with its worst entry, the
promotion-gate decision with the numbers behind it, drift against the stored reference, and
the pointers to the SHAP and partial-dependence reports.

**Every number here is passed in from a real measurement.** Nothing is defaulted to a
plausible value, nothing is invented at render time, and the limitations section is
*computed* from the measurements it was given — the coverage gap, the worst-slice gap, the
missingness share — not selected from a list of stock sentences. A limitations section that
says the same thing about every model is decoration; one that names this model's actual
weakest segment is evidence.

The card is written for a domain where held-out R² lands between roughly 0.45 and 0.80 and
accuracy between 0.65 and 0.88, on data with unobserved confounders and heteroscedastic
noise. The renderers therefore do not editorialise a 0.62 R² as poor. On genuinely noisy
data with real confounding, 0.62 is a good model and a card that implies otherwise teaches
its reader to distrust the honest number and prefer the overfitted one.
"""

from __future__ import annotations

import html as _html
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from aegis_ml.contracts.protocols import (
    DriftReport,
    GateDecision,
    Leaderboard,
    SliceMetric,
    TrainResult,
)
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    import pandas as pd

__all__ = [
    "PAGE_CSS",
    "TABPFN_LICENSE_NOTICE",
    "CoverageBlock",
    "ExtendedModelCard",
    "build_card",
    "escape",
    "html_page",
    "measure_missingness",
    "render_html",
    "render_markdown",
    "svg_bar_chart",
]


TABPFN_LICENSE_NOTICE = (
    "This run used TabPFN-2.5, whose weights are distributed under the Prior Labs "
    "License: research and evaluation use are permitted, commercial and production use "
    "are NOT. This card is therefore evidence for an evaluation, not a licence to deploy "
    "the TabPFN candidate. Set AEGIS_ML_ENABLE_TABPFN=0 to exclude the tier entirely; the "
    "AutoGluon tier reaches comparable accuracy given a longer search budget and carries "
    "no such restriction."
)
"""The notice printed on every card whose run touched TabPFN.

Stated on the card rather than in the README because the card is what travels: it is the
document a reader forwards, and a licence restriction that stayed behind in the repository
is a restriction nobody downstream ever sees.
"""


PAGE_CSS = """
:root{
  --bg:#ffffff; --fg:#16181d; --muted:#5b6270; --line:#e3e6ec; --card:#f7f8fa;
  --accent:#2f6feb; --good:#1a7f4b; --warn:#a86400; --bad:#b3261e; --chip:#eef1f6;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0f1115; --fg:#e8eaf0; --muted:#9aa3b2; --line:#242833; --card:#171a21;
    --accent:#6f9dff; --good:#54c48a; --warn:#e0a53a; --bad:#ef8079; --chip:#1c2029;
  }
}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
  font:15px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:60rem;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .25rem}
h2{font-size:1.05rem;margin:2.2rem 0 .6rem;padding-bottom:.35rem;
  border-bottom:1px solid var(--line);letter-spacing:.01em}
h3{font-size:.95rem;margin:1.2rem 0 .4rem}
p{margin:.5rem 0}
.sub{color:var(--muted);margin:0 0 1.5rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));gap:.75rem}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.75rem .9rem}
.tile .k{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.06em}
.tile .v{font-size:1.35rem;font-weight:600;margin-top:.15rem;font-variant-numeric:tabular-nums}
.tile .n{color:var(--muted);font-size:.78rem;margin-top:.2rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.5rem 0}
th,td{border-bottom:1px solid var(--line);padding:.42rem .55rem;text-align:left;
  vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:.76rem;text-transform:uppercase;
  letter-spacing:.05em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.worst td{background:color-mix(in srgb,var(--bad) 12%,transparent)}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.75rem;
  overflow-x:auto}
ul{margin:.4rem 0 .4rem 1.1rem;padding:0}
li{margin:.2rem 0}
.pill{display:inline-block;padding:.12rem .5rem;border-radius:999px;background:var(--chip);
  font-size:.75rem;color:var(--muted);margin-right:.3rem}
.pass{color:var(--good);font-weight:600}
.fail{color:var(--bad);font-weight:600}
.warnv{color:var(--warn);font-weight:600}
.notice{border:1px solid var(--line);border-left:3px solid var(--warn);background:var(--card);
  border-radius:8px;padding:.7rem .9rem;margin:.8rem 0}
.scroll{overflow-x:auto}
svg{max-width:100%;height:auto;display:block}
.bar-label{fill:var(--fg);font-size:11px}
.bar-value{fill:var(--muted);font-size:11px}
.axis{stroke:var(--line);stroke-width:1}
footer{color:var(--muted);font-size:.78rem;margin-top:2.5rem;border-top:1px solid var(--line);
  padding-top:.8rem}
"""
"""The one stylesheet every report in this package uses.

It lives with the model card deliberately: the card is the canonical document, and the SHAP
and partial-dependence pages are read alongside it. A second stylesheet would let them drift
into looking like two different products, which quietly undermines the claim that they
describe one model. Colours are declared as tokens on ``:root`` and re-declared under
``prefers-color-scheme: dark``, so both themes work without JavaScript and without a
network request — a report that needs a CDN is a report that renders blank in the room where
it matters.
"""


def escape(value: object) -> str:
    """HTML-escape any value for safe interpolation into a report.

    Feature names, categorical levels and note strings all originate outside this package
    (a domain spec, a data frame, an exception message). Rendering them raw would let a
    level called ``<script>`` break the page — and, more mundanely, a unit like ``m<sup>``
    corrupt the layout of an otherwise correct card.

    Args:
        value: Anything; stringified first.

    Returns:
        The escaped string.
    """
    return _html.escape(str(value), quote=True)


def html_page(title: str, body: str, *, subtitle: str = "", footer: str = "") -> str:
    """Wrap rendered sections in a complete, self-contained HTML document.

    Self-contained is a hard requirement, not a preference: these reports are opened from a
    filesystem path, attached to tickets and viewed offline. Every byte of CSS is inline and
    every chart is inline SVG, so there is no state in which the page renders as unstyled
    text because a CDN was unreachable.

    Args:
        title: Document title, used in ``<title>`` and as the ``<h1>``.
        body: Pre-rendered, already-escaped HTML for the document body.
        subtitle: Optional line under the title.
        footer: Optional footer note.

    Returns:
        A full ``<!doctype html>`` document as a string.
    """
    sub = f'<p class="sub">{subtitle}</p>' if subtitle else ""
    foot = f"<footer>{footer}</footer>" if footer else ""
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title>"
        f"<style>{PAGE_CSS}</style></head><body><main>"
        f"<h1>{escape(title)}</h1>{sub}{body}{foot}"
        "</main></body></html>\n"
    )


def svg_bar_chart(
    rows: Sequence[tuple[str, float]],
    *,
    width: int = 720,
    row_height: int = 22,
    label_width: int = 200,
    signed: bool = False,
    value_format: str = "{:+.4f}",
) -> str:
    """Render a horizontal bar chart as inline SVG, with no external dependency.

    Bars are drawn to scale against the largest absolute value, and **every row passed in is
    drawn**, including rows at or near zero. That matters for SHAP importance in particular:
    a feature the model correctly ignored should appear as a visible, empty row. Filtering
    it out would hide the single clearest piece of evidence that the model is not chasing
    noise.

    Args:
        rows: ``(label, value)`` pairs, in the order they should appear.
        width: Total SVG width in px.
        row_height: Height of one bar row in px.
        label_width: Width reserved for the labels in px.
        signed: When ``True``, negative values are drawn left of a centre axis in the "bad"
            colour and positives right of it — used for signed SHAP contributions, where the
            direction of a driver is the point.
        value_format: Format string applied to each value.

    Returns:
        An ``<svg>`` element as a string. Returns an empty string for an empty input rather
        than an empty chart frame, which reads as "measured nothing" instead of "nothing to
        measure".
    """
    items = list(rows)
    if not items:
        return ""
    height = row_height * len(items) + 16
    plot_left = label_width + 8
    plot_width = max(60, width - plot_left - 88)
    largest = max(abs(value) for _, value in items) or 1.0
    centre = plot_left + (plot_width / 2 if signed else 0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="bar chart">'
    ]
    if signed:
        parts.append(
            f'<line class="axis" x1="{centre:.1f}" y1="6" x2="{centre:.1f}" '
            f'y2="{height - 10}"/>'
        )
    for index, (label, value) in enumerate(items):
        top = 8 + index * row_height
        text_y = top + row_height * 0.62
        scale = (plot_width / 2 if signed else plot_width) / largest
        span = abs(value) * scale
        x_start = centre - span if signed and value < 0 else centre
        colour = "var(--bad)" if (signed and value < 0) else "var(--accent)"
        parts.append(
            f'<text class="bar-label" x="0" y="{text_y:.1f}">{escape(label)}</text>'
            f'<rect x="{x_start:.1f}" y="{top + 3}" width="{max(span, 0.6):.1f}" '
            f'height="{row_height - 9}" rx="2" fill="{colour}"/>'
            f'<text class="bar-value" x="{width - 82}" y="{text_y:.1f}">'
            f"{escape(value_format.format(value))}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def measure_missingness(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> dict[str, float]:
    """Measure the null share of each column — a real count, never an assumption.

    The card's limitations section quotes this. Missingness is not a footnote on data with
    missing-at-random gaps: every null was filled from a training median or mode before the
    model saw it, so a feature that is 30% missing contributed the *training set's* central
    value to 30% of the predictions, and any SHAP attribution to that feature on those rows
    describes the imputation rather than the caller's situation.

    Args:
        frame: The frame to measure; typically the training frame.
        columns: Restrict to these columns; defaults to every column present.

    Returns:
        Column name → measured share of nulls in ``[0, 1]``. Columns absent from the frame
        are omitted rather than reported as ``0.0``, since "not present" and "present and
        complete" are different facts.
    """
    names = list(columns) if columns is not None else list(frame.columns)
    n_rows = int(len(frame))
    if n_rows == 0:
        return {}
    return {
        name: float(frame[name].isna().sum()) / n_rows
        for name in names
        if name in frame.columns
    }


class CoverageBlock(BaseModel):
    """Requested versus measured conformal coverage, side by side and never collapsed.

    Two fields, always. The card renders them adjacent so a reader cannot mistake one for
    the other, which is the failure this shape exists to prevent: a single "coverage: 0.9"
    line is read as a measurement by everyone and is usually a request.
    """

    requested: float = Field(gt=0.0, lt=1.0, description="The level ASKED FOR.")
    empirical: float | None = Field(
        default=None, description="The level MEASURED on held-out rows; None if unmeasured."
    )
    tolerance: float = Field(ge=0.0, description="Allowed shortfall (sampling error).")
    n_rows: int = Field(default=0, ge=0, description="Held-out rows behind the measurement.")
    kind: Literal["interval", "set"] = "interval"
    mean_width: float | None = Field(
        default=None, description="Mean interval width, in the target's unit."
    )
    mean_set_size: float | None = Field(default=None, description="Mean prediction-set size.")

    @property
    def meets_request(self) -> bool | None:
        """Whether the measured rate cleared ``requested - tolerance``.

        Returns:
            ``None`` when nothing was measured — deliberately not ``False``, because
            "unmeasured" and "measured and failed" are different findings and the card
            prints them differently.
        """
        if self.empirical is None:
            return None
        return self.empirical >= self.requested - self.tolerance

    @property
    def gap(self) -> float | None:
        """Measured minus requested; negative means under-coverage.

        Returns:
            The signed gap, or ``None`` when nothing was measured.
        """
        if self.empirical is None:
            return None
        return self.empirical - self.requested


class ExtendedModelCard(BaseModel):
    """One model's identity, data, measured performance, coverage, slices and limitations.

    Built by :func:`build_card` from a real :class:`~aegis_ml.contracts.protocols.TrainResult`
    and rendered by :func:`render_markdown` / :func:`render_html`. Nothing in this type has a
    plausible default: an unmeasured quantity is ``None`` and renders as "not measured",
    because a card whose blanks are filled with reasonable-looking numbers is worse than no
    card at all.
    """

    # Identity.
    run_id: str
    domain_id: str
    created_at: str = Field(description="ISO-8601 UTC; stamped by the caller.")
    task: Literal["regression", "classification"]
    target: str
    target_unit: str | None = None
    target_description: str | None = None

    # Data.
    training_size: int = Field(default=0, ge=0)
    calibration_size: int = Field(default=0, ge=0)
    test_size: int = Field(default=0, ge=0)
    dataset_digest: str | None = None
    data_source: str | None = Field(
        default=None,
        description="'provided' | 'spec_provider' | 'synthetic'. A synthetic-source model "
        "carries no domain signal and the card says so in its limitations.",
    )
    missingness: dict[str, float] = Field(
        default_factory=dict, description="MEASURED null share per column."
    )

    # Model.
    recipe: dict[str, Any] | None = Field(
        default=None, description="The portable Recipe that produced this model, as JSON."
    )
    leaderboard: Leaderboard | None = Field(
        default=None, description="Every candidate, winners and losers, from every tier run."
    )
    tabpfn_used: bool = False

    # Performance.
    metric_name: str
    metric_value: float
    metrics: dict[str, float] = Field(
        default_factory=dict, description="The full measured metric set for this run."
    )
    cv_primary_mean: float | None = None
    cv_primary_std: float | None = Field(
        default=None,
        description="Fold-to-fold spread. Quoted next to the mean because a mean without a "
        "spread cannot say whether a promotion margin is real.",
    )

    # Conformal.
    coverage: CoverageBlock

    # Slices.
    slices: list[SliceMetric] = Field(default_factory=list)
    worst_slice: SliceMetric | None = None
    skipped_slices: list[dict[str, Any]] = Field(
        default_factory=list, description="Segments that were NOT measured, and why."
    )

    # Explanations, drift, governance.
    top_features: list[dict[str, Any]] = Field(
        default_factory=list, description="Global SHAP importance rows, strongest first."
    )
    shap_report_path: str | None = None
    pdp_report_path: str | None = None
    drift: DriftReport | None = None
    gate: GateDecision | None = None

    # Honesty.
    limitations: list[str] = Field(
        default_factory=list,
        description="COMPUTED from this run's measurements — the coverage gap, the worst "
        "segment, the missingness share — never a fixed list of caveats.",
    )
    notes: list[str] = Field(default_factory=list)
    aegis_card: dict[str, Any] | None = Field(
        default=None,
        description="The Aegis ModelCard's fields verbatim, as a dict. Nested rather than "
        "imported so this module keeps its dependency footprint and still renders when the "
        "Aegis package is not installed.",
    )


def _limitations(
    result: TrainResult,
    coverage: CoverageBlock,
    worst: SliceMetric | None,
    missingness: dict[str, float],
    skipped: list[dict[str, Any]],
    *,
    tabpfn_used: bool,
    top_features: list[dict[str, Any]],
    data_source: str | None,
) -> list[str]:
    """Derive the limitations section from what was actually measured.

    Every sentence produced here quotes a number from this run. A stock caveat ("model
    performance may vary") is true of every model ever trained and therefore tells a reader
    nothing about *this* one; a sentence naming the segment that scored 0.31 tells them
    exactly where not to trust it.

    Args:
        result: The training result being carded.
        coverage: The requested/measured coverage pair.
        worst: The worst measured slice, if any.
        missingness: MEASURED null share per column.
        skipped: Segments that were not measured.
        tabpfn_used: Whether the run used TabPFN.
        top_features: Global SHAP rows, strongest first.
        data_source: How the training frame was obtained.

    Returns:
        The limitation sentences, most consequential first.
    """
    out: list[str] = []

    if data_source == "synthetic":
        out.append(
            "The training frame came from the built-in synthesiser, so this model carries "
            "NO domain signal. Every number below is a measurement of a model fitted to "
            "generated data and must not be cited as domain evidence."
        )

    gap = coverage.gap
    if coverage.empirical is None:
        out.append(
            f"Conformal coverage was requested at {coverage.requested:.2f} and NEVER "
            f"MEASURED. The interval this model serves is unverified; treat its width as a "
            f"claim, not a guarantee."
        )
    elif gap is not None and gap < -coverage.tolerance:
        out.append(
            f"Conformal intervals UNDER-COVER: {coverage.empirical:.3f} measured against "
            f"{coverage.requested:.3f} requested, on {coverage.n_rows} held-out rows — a "
            f"shortfall of {abs(gap):.3f}. Split conformal calibrates one width for the "
            f"whole distribution, so under heteroscedastic noise the shortfall is "
            f"concentrated where the target is hardest to predict, not spread evenly."
        )
    elif gap is not None and gap < 0:
        out.append(
            f"Conformal coverage measured {coverage.empirical:.3f} against "
            f"{coverage.requested:.3f} requested ({gap:+.3f}). Within the "
            f"{coverage.tolerance:.3f} sampling tolerance on {coverage.n_rows} rows, but it "
            f"is on the low side of the request rather than above it."
        )

    if coverage.n_rows and coverage.n_rows < 100:
        out.append(
            f"Coverage was measured on only {coverage.n_rows} held-out rows; the standard "
            f"error at that size is roughly "
            f"{(coverage.requested * (1 - coverage.requested) / coverage.n_rows) ** 0.5:.3f}, "
            f"which is the resolution of this check."
        )

    if worst is not None:
        overall = result.metric_value
        direction_note = (
            f"{worst.metric_name} {worst.metric_value:.3f} against an overall "
            f"{overall:.3f}"
        )
        out.append(
            f"Weakest segment: {worst.feature}={worst.level} ({worst.n_rows} rows) scores "
            f"{direction_note}. The headline number is an average over segments that do not "
            f"experience this model equally; decisions taken inside this segment rest on "
            f"the weaker number, not the headline."
        )
    else:
        out.append(
            "No per-segment sweep is recorded for this run, so nothing here can say whether "
            "the headline metric holds evenly across the population. That is missing "
            "evidence, not a clean result."
        )

    if skipped:
        names = ", ".join(
            f"{entry.get('feature')}={entry.get('level')} (n={entry.get('n_rows')})"
            for entry in skipped[:5]
        )
        more = "" if len(skipped) <= 5 else f", and {len(skipped) - 5} more"
        out.append(
            f"{len(skipped)} segment(s) were too small to measure and remain unevaluated: "
            f"{names}{more}. The model was not shown to work there; it was simply not tested."
        )

    heavy = sorted(
        ((name, share) for name, share in missingness.items() if share > 0.05),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if heavy:
        rendered = ", ".join(f"{name} {share:.0%}" for name, share in heavy[:5])
        out.append(
            f"Measured missingness above 5%: {rendered}. Those nulls were imputed from "
            f"training medians/modes before fitting, so on the affected rows the model's "
            f"answer partly reflects the training set's centre rather than the caller's "
            f"input — and any SHAP attribution to those features describes the imputed "
            f"value."
        )

    ignored = [
        str(row.get("feature"))
        for row in top_features
        if float(row.get("mean_abs_shap", 0.0)) <= 1e-9
    ]
    if ignored:
        out.append(
            f"The model assigns effectively zero attribution to {', '.join(ignored)}. That "
            f"is reported rather than hidden: a feature the model ignored is evidence about "
            f"the feature, and removing it from the spec would simplify the contract the "
            f"adapter has to satisfy."
        )

    if result.test_size and result.test_size < 200:
        out.append(
            f"The held-out test split holds {result.test_size} rows. Every metric on this "
            f"card moves by several points across reseeds at that size, so treat "
            f"differences smaller than that as unresolved rather than as improvements."
        )

    if tabpfn_used:
        out.append(TABPFN_LICENSE_NOTICE)

    return out


def build_card(
    result: TrainResult,
    *,
    aegis_card: dict[str, Any] | None = None,
    leaderboard: Leaderboard | None = None,
    slices: Sequence[SliceMetric] | None = None,
    drift: DriftReport | None = None,
    tabpfn_used: bool = False,
    gate: GateDecision | None = None,
    metrics: dict[str, float] | None = None,
    missingness: dict[str, float] | None = None,
    skipped_slices: Sequence[dict[str, Any]] | None = None,
    top_features: Sequence[dict[str, Any]] | None = None,
    shap_report_path: str | None = None,
    pdp_report_path: str | None = None,
    cv_primary_mean: float | None = None,
    cv_primary_std: float | None = None,
    coverage_tolerance: float | None = None,
    mean_interval_width: float | None = None,
    mean_set_size: float | None = None,
    target_unit: str | None = None,
    target_description: str | None = None,
    data_source: str | None = None,
    created_at: str | None = None,
    notes: Sequence[str] | None = None,
) -> ExtendedModelCard:
    """Assemble the extended card from measurements that have already been made.

    This function computes nothing about model quality and invents nothing: it arranges
    measured inputs, derives the worst slice from the slice list, and derives the limitations
    from the measurements it was handed. If a caller has not measured something, the card
    says so — that is the whole design.

    Args:
        result: The training result; supplies identity, sizes, the primary metric and both
            coverage fields.
        aegis_card: The Aegis ``ModelCard``'s fields as a dict (``model_dump()``), when the
            spine produced one. Kept as a dict so this module never imports the heavy
            package. It is rendered verbatim, including its own
            ``conformal_coverage`` / ``conformal_coverage_empirical`` pair.
        leaderboard: Every AutoML candidate, losers included.
        slices: Measured per-segment metrics; defaults to ``result.slices``.
        drift: A drift report against the stored reference frame, if one has been run.
        tabpfn_used: Whether the run used TabPFN; adds the Prior Labs License notice.
        gate: The promotion-gate decision, with its numbers.
        metrics: The full measured metric set (``aegis_ml.evaluate.metrics.score``).
        missingness: MEASURED null share per column (:func:`measure_missingness`).
        skipped_slices: Segments that were not measured, as dicts.
        top_features: Global SHAP importance rows, strongest first.
        shap_report_path: Path to the rendered SHAP report.
        pdp_report_path: Path to the rendered partial-dependence report.
        cv_primary_mean: Cross-validated mean of the primary metric.
        cv_primary_std: Its fold-to-fold standard deviation.
        coverage_tolerance: Allowed coverage shortfall; defaults to the process setting.
        mean_interval_width: Mean conformal interval width, in the target's unit.
        mean_set_size: Mean conformal prediction-set size.
        target_unit: The target's unit, for rendering intervals in a human quantity.
        target_description: What the target means, in the domain's language.
        data_source: ``'provided'`` | ``'spec_provider'`` | ``'synthetic'``.
        created_at: ISO-8601 UTC timestamp; stamped by the caller, never invented here.
        notes: Free-text notes to append to the run's own.

    Returns:
        The assembled :class:`ExtendedModelCard`.
    """
    from aegis_ml.evaluate.slices import worst_slice as _worst_slice

    slice_list = list(slices) if slices is not None else list(result.slices)
    comparable = [s for s in slice_list if s.metric_name == result.metric_name]
    worst = _worst_slice(comparable) if comparable else None

    coverage = CoverageBlock(
        requested=result.requested_coverage,
        empirical=result.empirical_coverage,
        tolerance=(
            settings.coverage_tolerance if coverage_tolerance is None else coverage_tolerance
        ),
        n_rows=result.test_size,
        kind="interval" if result.task == "regression" else "set",
        mean_width=mean_interval_width,
        mean_set_size=mean_set_size,
    )
    missing = dict(missingness or {})
    skipped = [dict(entry) for entry in (skipped_slices or [])]
    features = [dict(row) for row in (top_features or [])]
    source = data_source or (aegis_card or {}).get("data_source")

    return ExtendedModelCard(
        run_id=result.run_id,
        domain_id=result.domain_id,
        created_at=created_at or "",
        task=result.task,
        target=result.target,
        target_unit=target_unit,
        target_description=target_description,
        training_size=result.training_size,
        calibration_size=result.calibration_size,
        test_size=result.test_size,
        dataset_digest=result.dataset_digest,
        data_source=source,
        missingness=missing,
        recipe=None if result.recipe is None else result.recipe.model_dump(),
        leaderboard=leaderboard if leaderboard is not None else result.leaderboard,
        tabpfn_used=tabpfn_used,
        metric_name=result.metric_name,
        metric_value=result.metric_value,
        metrics=dict(metrics or {}),
        cv_primary_mean=cv_primary_mean,
        cv_primary_std=cv_primary_std,
        coverage=coverage,
        slices=slice_list,
        worst_slice=worst,
        skipped_slices=skipped,
        top_features=features,
        shap_report_path=shap_report_path,
        pdp_report_path=pdp_report_path,
        drift=drift,
        gate=gate,
        limitations=_limitations(
            result,
            coverage,
            worst,
            missing,
            skipped,
            tabpfn_used=tabpfn_used,
            top_features=features,
            data_source=source,
        ),
        notes=[*result.notes, *(notes or [])],
        aegis_card=aegis_card,
    )


def _coverage_rows(card: ExtendedModelCard) -> list[tuple[str, str]]:
    """Build the requested/measured coverage rows both renderers share.

    Args:
        card: The card being rendered.

    Returns:
        ``(label, value)`` pairs. The requested and measured levels are separate rows by
        construction — the two renderers cannot accidentally merge them because neither of
        them builds this list.
    """
    coverage = card.coverage
    measured = (
        "not measured" if coverage.empirical is None else f"{coverage.empirical:.3f}"
    )
    verdict = coverage.meets_request
    verdict_text = (
        "unverified — nothing was measured"
        if verdict is None
        else ("met" if verdict else "NOT MET — the interval under-covers")
    )
    rows = [
        ("Coverage REQUESTED", f"{coverage.requested:.3f}"),
        ("Coverage MEASURED (empirical)", measured),
        ("Tolerance (sampling error)", f"{coverage.tolerance:.3f}"),
        ("Pass floor (requested − tolerance)", f"{coverage.requested - coverage.tolerance:.3f}"),
        ("Held-out rows behind the measurement", str(coverage.n_rows)),
        ("Verdict", verdict_text),
    ]
    if coverage.mean_width is not None:
        unit = f" {card.target_unit}" if card.target_unit else ""
        rows.append(("Mean interval width", f"{coverage.mean_width:.3f}{unit}"))
    if coverage.mean_set_size is not None:
        rows.append(("Mean prediction-set size", f"{coverage.mean_set_size:.3f}"))
    return rows


def render_markdown(card: ExtendedModelCard) -> str:
    """Render the card as Markdown, for a README, a PR body or a terminal.

    Args:
        card: The card to render.

    Returns:
        A Markdown document. Section order matches :func:`render_html` exactly so the two
        can be read against each other without hunting.
    """
    unit = f" {card.target_unit}" if card.target_unit else ""
    lines: list[str] = [
        f"# Model card — {card.domain_id} / {card.run_id}",
        "",
        f"*{card.target_description or card.target}*"
        if card.target_description
        else f"Target: `{card.target}`",
        "",
        "## Identity",
        "",
        f"- Run: `{card.run_id}`",
        f"- Domain: `{card.domain_id}`",
        f"- Created: {card.created_at or 'not stamped'}",
        f"- Task: {card.task}",
        f"- Target: `{card.target}`{unit}",
        "",
        "## Data",
        "",
        f"- Training rows: {card.training_size}",
        f"- Calibration rows (disjoint): {card.calibration_size}",
        f"- Held-out test rows: {card.test_size}",
        f"- Dataset digest: `{card.dataset_digest or 'none recorded'}`",
        f"- Data source: {card.data_source or 'not recorded'}",
    ]
    if card.missingness:
        worst_missing = sorted(card.missingness.items(), key=lambda p: p[1], reverse=True)
        rendered = ", ".join(f"`{n}` {s:.1%}" for n, s in worst_missing[:6] if s > 0)
        lines.append(f"- Measured missingness: {rendered or 'none in any column'}")

    lines += ["", "## Model", ""]
    if card.recipe:
        members = ", ".join(
            f"{m.get('name')} ({m.get('kind')}, w={m.get('weight')})"
            for m in card.recipe.get("members", [])
        )
        lines.append(f"- Recipe tier: `{card.recipe.get('tier')}`")
        lines.append(f"- Ensemble: {members}")
    else:
        lines.append("- No portable recipe recorded for this run.")
    if card.leaderboard and card.leaderboard.candidates:
        lines += [
            "",
            "### Leaderboard (losers included — the margin is the finding)",
            "",
            f"| Candidate | Tier | {card.leaderboard.metric_name} | Portable | Selected |",
            "| --- | --- | ---: | --- | --- |",
        ]
        for cand in card.leaderboard.candidates:
            lines.append(
                f"| {cand.name} | {cand.tier} | {cand.metric_value:.4f} | "
                f"{'yes' if cand.portable else 'NO'} | {'yes' if cand.selected else ''} |"
            )
        if card.leaderboard.tiers_skipped:
            skipped = ", ".join(
                f"{tier} ({why})" for tier, why in card.leaderboard.tiers_skipped.items()
            )
            lines.append("")
            lines.append(f"Tiers that did not run: {skipped}")

    lines += [
        "",
        "## Measured performance",
        "",
        f"- **{card.metric_name} = {card.metric_value:.4f}** on {card.test_size} held-out "
        f"rows the model was neither fitted nor calibrated on.",
    ]
    if card.cv_primary_mean is not None:
        std = "" if card.cv_primary_std is None else f" ± {card.cv_primary_std:.4f}"
        lines.append(
            f"- Cross-validated {card.metric_name}: {card.cv_primary_mean:.4f}{std} "
            f"(the spread, not the mean, says whether a promotion margin is real)."
        )
    for name, value in sorted(card.metrics.items()):
        if name != card.metric_name:
            lines.append(f"- {name}: {value:.4f}")

    lines += ["", "## Conformal coverage — requested vs measured", ""]
    lines += ["| Field | Value |", "| --- | --- |"]
    lines += [f"| {label} | {value} |" for label, value in _coverage_rows(card)]

    lines += ["", "## Slices", ""]
    if card.slices:
        lines += [
            f"| Feature | Level | Rows | {card.metric_name} | |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for entry in card.slices:
            flag = ""
            if card.worst_slice is not None and (
                entry.feature == card.worst_slice.feature
                and entry.level == card.worst_slice.level
            ):
                flag = "**WORST**"
            lines.append(
                f"| {entry.feature} | {entry.level} | {entry.n_rows} | "
                f"{entry.metric_value:.4f} | {flag} |"
            )
    else:
        lines.append("No per-segment sweep was recorded for this run.")
    if card.skipped_slices:
        lines += ["", "### Segments NOT measured", ""]
        lines += [
            f"- {entry.get('feature')}={entry.get('level')} "
            f"(n={entry.get('n_rows')}): {entry.get('reason')}"
            for entry in card.skipped_slices
        ]

    lines += ["", "## Explanations", ""]
    if card.top_features:
        lines += ["| Feature | mean \\|SHAP\\| | Share |", "| --- | ---: | ---: |"]
        total = sum(float(r.get("mean_abs_shap", 0.0)) for r in card.top_features) or 1.0
        for row in card.top_features:
            value = float(row.get("mean_abs_shap", 0.0))
            lines.append(f"| {row.get('feature')} | {value:.4f} | {value / total:.1%} |")
    else:
        lines.append("No SHAP importance recorded for this run.")
    if card.shap_report_path:
        lines.append(f"- SHAP report: `{card.shap_report_path}`")
    if card.pdp_report_path:
        lines.append(f"- Partial dependence report: `{card.pdp_report_path}`")

    lines += ["", "## Drift reference", ""]
    if card.drift is not None:
        lines += [
            f"- Verdict: **{card.drift.verdict}**",
            f"- Drifted features: {card.drift.drifted_share:.1%} "
            f"({', '.join(card.drift.drifted_features) or 'none'})",
            f"- Reference digest: `{card.drift.reference_digest or 'none'}`",
        ]
        if (
            card.drift.estimated_metric_name
            and card.drift.estimated_metric_value is not None
        ):
            lines.append(
                f"- ESTIMATED (label-free) {card.drift.estimated_metric_name}: "
                f"{card.drift.estimated_metric_value:.4f} — an estimate, not a measurement."
            )
    else:
        lines.append(
            f"No drift report yet. The reference frame and its digest "
            f"(`{card.dataset_digest or 'none recorded'}`) are what a later run compares "
            f"against."
        )

    if card.gate is not None:
        lines += ["", "## Promotion gate", ""]
        lines.append(f"- Decision: **{'PROMOTED' if card.gate.promoted else 'REJECTED'}**")
        lines += [f"- {reason}" for reason in card.gate.reasons]

    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in card.limitations] or ["- None recorded."]

    if card.aegis_card:
        lines += ["", "## Aegis spine card (verbatim)", ""]
        lines += [f"- `{k}`: {v}" for k, v in sorted(card.aegis_card.items())]

    if card.notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in card.notes]

    return "\n".join(lines) + "\n"


def _tile(key: str, value: str, note: str = "") -> str:
    """Render one headline statistic tile.

    Args:
        key: The label above the number.
        value: The number itself, pre-formatted.
        note: Optional qualifier under the number — where the honest caveat goes.

    Returns:
        The tile's HTML.
    """
    note_html = f'<div class="n">{escape(note)}</div>' if note else ""
    return (
        f'<div class="tile"><div class="k">{escape(key)}</div>'
        f'<div class="v">{escape(value)}</div>{note_html}</div>'
    )


def render_html(card: ExtendedModelCard) -> str:
    """Render the card as a self-contained HTML page that works in light and dark.

    No external stylesheet, no script, no font request, no image URL — the page is one file
    and renders identically offline. Sections mirror :func:`render_markdown`.

    The coverage section deliberately places requested and measured side by side in the same
    table, each labelled in full. Every other presentation of that pair invites the reader to
    remember only one number, and the one they remember is the one they were promised rather
    than the one that was delivered.

    Args:
        card: The card to render.

    Returns:
        A complete HTML document as a string.
    """
    unit = f" {card.target_unit}" if card.target_unit else ""
    coverage = card.coverage
    verdict = coverage.meets_request
    verdict_class = "warnv" if verdict is None else ("pass" if verdict else "fail")

    tiles = [
        _tile(
            card.metric_name,
            f"{card.metric_value:.4f}",
            f"measured on {card.test_size} held-out rows",
        ),
        _tile(
            "Coverage requested",
            f"{coverage.requested:.3f}",
            "the level asked for — not a measurement",
        ),
        _tile(
            "Coverage measured",
            "not measured" if coverage.empirical is None else f"{coverage.empirical:.3f}",
            f"tolerance {coverage.tolerance:.3f} on {coverage.n_rows} rows",
        ),
    ]
    if card.worst_slice is not None:
        tiles.append(
            _tile(
                "Worst segment",
                f"{card.worst_slice.metric_value:.3f}",
                f"{card.worst_slice.feature}={card.worst_slice.level} "
                f"({card.worst_slice.n_rows} rows)",
            )
        )
    if card.cv_primary_mean is not None:
        std = "" if card.cv_primary_std is None else f" ± {card.cv_primary_std:.3f}"
        tiles.append(
            _tile(
                f"CV {card.metric_name}",
                f"{card.cv_primary_mean:.3f}{std}",
                "mean ± fold-to-fold spread",
            )
        )

    body: list[str] = [f'<div class="grid">{"".join(tiles)}</div>']

    body.append("<h2>Identity</h2><table>")
    for label, value in [
        ("Run", card.run_id),
        ("Domain", card.domain_id),
        ("Created", card.created_at or "not stamped"),
        ("Task", card.task),
        ("Target", f"{card.target}{unit}"),
        ("Target meaning", card.target_description or "not declared"),
    ]:
        body.append(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>")
    body.append("</table>")

    body.append("<h2>Data</h2><table>")
    for label, value in [
        ("Training rows", str(card.training_size)),
        ("Calibration rows (disjoint)", str(card.calibration_size)),
        ("Held-out test rows", str(card.test_size)),
        ("Dataset digest", card.dataset_digest or "none recorded"),
        ("Data source", card.data_source or "not recorded"),
    ]:
        body.append(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>")
    body.append("</table>")
    if card.missingness:
        rows = sorted(card.missingness.items(), key=lambda p: p[1], reverse=True)
        present = [(name, share) for name, share in rows if share > 0]
        if present:
            body.append("<h3>Measured missingness</h3><div class='scroll'>")
            body.append(
                svg_bar_chart(present[:15], value_format="{:.1%}")
            )
            body.append(
                "</div><p class='sub'>Nulls were imputed from training medians/modes before "
                "fitting, so on the affected rows the answer partly reflects the training "
                "set's centre rather than the caller's input.</p>"
            )

    body.append("<h2>Model</h2>")
    if card.recipe:
        members = "".join(
            f"<tr><td>{escape(m.get('name'))}</td><td>{escape(m.get('kind'))}</td>"
            f"<td class='num'>{escape(m.get('weight'))}</td></tr>"
            for m in card.recipe.get("members", [])
        )
        body.append(
            f"<p><span class='pill'>tier {escape(card.recipe.get('tier'))}</span>"
            f"<span class='pill'>search {escape(card.recipe.get('search_seconds'))}s</span></p>"
            f"<table><tr><th>Member</th><th>Estimator</th><th class='num'>Weight</th></tr>"
            f"{members}</table>"
        )
    else:
        body.append("<p>No portable recipe recorded for this run.</p>")
    if card.leaderboard and card.leaderboard.candidates:
        rows = "".join(
            f"<tr><td>{escape(c.name)}</td><td>{escape(c.tier)}</td>"
            f"<td class='num'>{c.metric_value:.4f}</td>"
            f"<td>{'yes' if c.portable else 'NO'}</td>"
            f"<td>{'selected' if c.selected else ''}</td></tr>"
            for c in card.leaderboard.candidates
        )
        body.append(
            f"<h3>Leaderboard</h3><p class='sub'>Losers are listed on purpose: a winner that "
            f"won by a nose and a winner that won by a mile justify different amounts of "
            f"complexity, and only the margin says which happened.</p>"
            f"<div class='scroll'><table><tr><th>Candidate</th><th>Tier</th>"
            f"<th class='num'>{escape(card.leaderboard.metric_name)}</th>"
            f"<th>Portable</th><th></th></tr>{rows}</table></div>"
        )
        if card.leaderboard.tiers_skipped:
            skipped = "".join(
                f"<li><code>{escape(tier)}</code>: {escape(why)}</li>"
                for tier, why in card.leaderboard.tiers_skipped.items()
            )
            body.append(f"<h3>Tiers that did not run</h3><ul>{skipped}</ul>")

    body.append("<h2>Measured performance</h2><table>")
    body.append(
        f"<tr><th>{escape(card.metric_name)} (primary)</th>"
        f"<td class='num'>{card.metric_value:.4f}</td></tr>"
    )
    for name, value in sorted(card.metrics.items()):
        if name != card.metric_name:
            body.append(
                f"<tr><th>{escape(name)}</th><td class='num'>{value:.4f}</td></tr>"
            )
    body.append("</table>")

    body.append(
        "<h2>Conformal coverage — requested vs measured</h2>"
        "<p class='sub'>Two separate quantities, printed side by side. The first is a "
        "promise; the second is what was delivered on rows the calibration never saw.</p>"
        "<table>"
    )
    for label, value in _coverage_rows(card):
        css = f' class="{verdict_class}"' if label == "Verdict" else ""
        body.append(f"<tr><th>{escape(label)}</th><td{css}>{escape(value)}</td></tr>")
    body.append("</table>")

    body.append("<h2>Slices</h2>")
    if card.slices:
        rows = []
        for entry in card.slices:
            is_worst = card.worst_slice is not None and (
                entry.feature == card.worst_slice.feature
                and entry.level == card.worst_slice.level
            )
            rows.append(
                f"<tr class='{'worst' if is_worst else ''}'><td>{escape(entry.feature)}</td>"
                f"<td>{escape(entry.level)}</td><td class='num'>{entry.n_rows}</td>"
                f"<td class='num'>{entry.metric_value:.4f}</td>"
                f"<td>{'WORST — the gate compares this one' if is_worst else ''}</td></tr>"
            )
        body.append(
            "<p class='sub'>The gate reads the worst row, not the mean: a model that "
            "improves on average while collapsing on one segment is a regression for "
            "everyone in that segment, and an aggregate score is exactly the instrument "
            "that cannot see it.</p>"
            f"<div class='scroll'><table><tr><th>Feature</th><th>Level</th>"
            f"<th class='num'>Rows</th>"
            f"<th class='num'>{escape(card.metric_name)}</th><th></th></tr>"
            f"{''.join(rows)}</table></div>"
        )
    else:
        body.append("<p>No per-segment sweep was recorded for this run.</p>")
    if card.skipped_slices:
        items = "".join(
            f"<li>{escape(e.get('feature'))}={escape(e.get('level'))} "
            f"(n={escape(e.get('n_rows'))}): {escape(e.get('reason'))}</li>"
            for e in card.skipped_slices
        )
        body.append(
            f"<h3>Segments NOT measured</h3><p class='sub'>Listed rather than dropped — an "
            f"unmeasured segment is a gap in the evidence, not a pass.</p><ul>{items}</ul>"
        )

    body.append("<h2>Explanations</h2>")
    if card.top_features:
        body.append(
            "<div class='scroll'>"
            + svg_bar_chart(
                [
                    (str(row.get("feature")), float(row.get("mean_abs_shap", 0.0)))
                    for row in card.top_features
                ],
                value_format="{:.4f}",
            )
            + "</div><p class='sub'>Features at or near zero are kept in the chart on "
            "purpose: a feature the model correctly ignored is evidence that it is not "
            "chasing noise, and hiding it would remove that evidence.</p>"
        )
    else:
        body.append("<p>No SHAP importance recorded for this run.</p>")
    links = []
    if card.shap_report_path:
        links.append(f"<li>SHAP report: <code>{escape(card.shap_report_path)}</code></li>")
    if card.pdp_report_path:
        links.append(
            f"<li>Partial dependence: <code>{escape(card.pdp_report_path)}</code></li>"
        )
    if links:
        body.append(f"<ul>{''.join(links)}</ul>")

    body.append("<h2>Drift reference</h2>")
    if card.drift is not None:
        estimated = ""
        if (
            card.drift.estimated_metric_name
            and card.drift.estimated_metric_value is not None
        ):
            estimated = (
                f"<tr><th>ESTIMATED {escape(card.drift.estimated_metric_name)}</th>"
                f"<td class='num'>{card.drift.estimated_metric_value:.4f}</td></tr>"
            )
        body.append(
            f"<table><tr><th>Verdict</th><td>{escape(card.drift.verdict)}</td></tr>"
            f"<tr><th>Drifted share</th>"
            f"<td class='num'>{card.drift.drifted_share:.1%}</td></tr>"
            f"<tr><th>Drifted features</th>"
            f"<td>{escape(', '.join(card.drift.drifted_features) or 'none')}</td></tr>"
            f"<tr><th>Reference digest</th>"
            f"<td><code>{escape(card.drift.reference_digest or 'none')}</code></td></tr>"
            f"{estimated}</table>"
            "<p class='sub'>An ESTIMATED metric is NannyML's label-free estimate of live "
            "performance. It is named 'estimated' everywhere so it is never read as a "
            "measurement.</p>"
        )
    else:
        body.append(
            f"<p>No drift report yet. The stored reference frame and its digest "
            f"(<code>{escape(card.dataset_digest or 'none recorded')}</code>) are what a "
            f"later run compares against.</p>"
        )

    if card.gate is not None:
        verdict_html = (
            "<span class='pass'>PROMOTED</span>"
            if card.gate.promoted
            else "<span class='fail'>REJECTED</span>"
        )
        checks = "".join(
            f"<tr><td>{escape(name)}</td>"
            f"<td class='{'pass' if ok else 'fail'}'>{'PASS' if ok else 'FAIL'}</td></tr>"
            for name, ok in card.gate.checks.items()
        )
        reasons = "".join(f"<li>{escape(reason)}</li>" for reason in card.gate.reasons)
        body.append(
            f"<h2>Promotion gate</h2><p>Decision: {verdict_html}</p>"
            f"<table><tr><th>Criterion</th><th>Result</th></tr>{checks}</table>"
            f"<ul>{reasons}</ul>"
        )

    limitations = "".join(f"<li>{escape(item)}</li>" for item in card.limitations)
    body.append(
        "<h2>Limitations</h2><p class='sub'>Computed from this run's own measurements — the "
        "coverage gap, the weakest segment, the missingness share — not a fixed list of "
        "caveats.</p>"
        f"<ul>{limitations or '<li>None recorded.</li>'}</ul>"
    )

    if card.tabpfn_used:
        body.append(
            f"<div class='notice'><strong>TabPFN — Prior Labs License.</strong> "
            f"{escape(TABPFN_LICENSE_NOTICE)}</div>"
        )

    if card.aegis_card:
        rows = "".join(
            f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>"
            for k, v in sorted(card.aegis_card.items())
        )
        body.append(
            "<h2>Aegis spine card (verbatim)</h2><p class='sub'>The fitted spine's own "
            "ModelCard, unedited. This card extends it; it never restates it differently."
            f"</p><div class='scroll'><table>{rows}</table></div>"
        )

    if card.notes:
        notes = "".join(f"<li>{escape(note)}</li>" for note in card.notes)
        body.append(f"<h2>Notes</h2><ul>{notes}</ul>")

    subtitle = (
        f"{escape(card.domain_id)} · run <code>{escape(card.run_id)}</code> · "
        f"{escape(card.task)} on <code>{escape(card.target)}</code>"
    )
    return html_page(
        f"Model card — {card.domain_id}",
        "".join(body),
        subtitle=subtitle,
        footer=(
            "Generated by aegis-ml. Every figure on this page was measured on held-out data "
            "for this run; unmeasured quantities are printed as 'not measured' rather than "
            "filled in."
        ),
    )
