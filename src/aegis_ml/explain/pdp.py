"""Partial dependence and ICE curves — the model's *shape*, not just its rankings.

SHAP says which features mattered and in which direction for the rows it saw. Partial
dependence answers the question a domain expert actually asks next: *"what happens as this
feature increases?"* — and it is the cheapest available check that a model learned something
plausible rather than something that merely scores well. A queue-depth curve that slopes
downward, when every operator knows a deeper queue means slower resolution, is a finding no
held-out metric will ever report.

Two honesty notes are built into the output rather than left to the reader:

* **ICE curves are offered alongside the average.** Partial dependence averages over the
  other features, and an average can be flat while every individual row moves steeply in
  opposite directions. That is exactly what an interaction looks like, and only the
  individual curves show it. ``ice=True`` computes them.
* **Extrapolation is marked.** PD evaluates the model at feature values combined with other
  rows' values, including combinations that never occur — a 3-month-tenure agent handling an
  enterprise escalation, say. Every curve therefore carries the measured share of rows near
  each grid point, and the renderer greys the thinly populated ends.

Numeric features go through :func:`sklearn.inspection.partial_dependence`. Categorical
features are computed by explicit level substitution — set the column to the level for every
row, predict, average — which is the definition of partial dependence and avoids asking
sklearn's percentile grid to interpret a string column, which it cannot.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.explain.card import escape, html_page

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    import pandas as pd

__all__ = [
    "PDPCurve",
    "PartialDependenceUnavailableError",
    "partial_dependence_curves",
    "render_html",
]


class PartialDependenceUnavailableError(AegisMLError):
    """A partial-dependence curve cannot be computed for this model or feature.

    Raised rather than returning a flat line. A flat curve is a *finding* — "this feature
    does not move the model" — and a flat line emitted because the computation failed is
    indistinguishable from it.
    """

    def __init__(self, feature: str, reason: str) -> None:
        """Name the feature and why its curve could not be produced."""
        super().__init__(
            f"Partial dependence for {feature!r} could not be computed: {reason}. No flat "
            f"line is returned in its place — a flat curve means 'this feature does not "
            f"move the model', and a failed computation must never be readable as that."
        )
        self.feature = feature


class PDPCurve(BaseModel):
    """One feature's partial-dependence curve, with population support and optional ICE.

    ``support`` is carried alongside ``average`` because a partial-dependence value at a grid
    point the data barely visits is an extrapolation. Reporting the curve without it invites
    a reader to trust its ends exactly where they are least supported.
    """

    feature: str
    kind: Literal["numeric", "categorical"]
    grid_labels: list[str] = Field(description="Human-readable grid points, in curve order.")
    grid_values: list[float] | None = Field(
        default=None, description="Numeric grid points; None for categorical features."
    )
    average: list[float] = Field(description="Mean model output at each grid point.")
    support: list[float] = Field(
        default_factory=list,
        description="MEASURED share of rows near each grid point, in [0, 1]. Thin support "
        "means the curve is extrapolating there.",
    )
    ice: list[list[float]] = Field(
        default_factory=list,
        description="Individual conditional expectation curves, one per sampled row. An "
        "average can be flat while these fan out in opposite directions — that is an "
        "interaction, and the average is the one view that cannot show it.",
    )
    n_rows: int = Field(default=0, ge=0, description="Rows the curve was computed over.")
    response: str = Field(
        default="prediction",
        description="What the y-axis is: 'prediction' (regression) or the probability of "
        "the named class (classification).",
    )

    @property
    def swing(self) -> float:
        """Peak-to-trough movement of the averaged curve, in the response's units.

        Returns:
            ``max(average) - min(average)``, or ``0.0`` for an empty curve. This is the
            quantity a domain expert compares against their own sense of the effect size:
            a curve with the right *shape* and an implausible swing is still wrong.
        """
        if not self.average:
            return 0.0
        return float(max(self.average) - min(self.average))


def _response_function(model: object, problem: MLProblem, class_index: int) -> tuple[Any, str]:  # noqa: ANN401
    """Choose what the curve's y-axis measures, and say so in words.

    Args:
        model: The fitted estimator or pipeline.
        problem: The declared problem.
        class_index: Which class column to plot for a classifier.

    Returns:
        ``(callable, response_label)`` where the callable maps a frame to a 1-D array.

    Raises:
        PartialDependenceUnavailableError: When a classifier exposes no ``predict_proba``;
            a curve over hard class labels is a step function whose height means nothing.
    """
    if problem.target.task == "classification":
        if not hasattr(model, "predict_proba"):
            raise PartialDependenceUnavailableError(
                "(all features)",
                "the classifier exposes no predict_proba, so a partial-dependence curve "
                "would be a step function over class indices whose height carries no "
                "meaning",
            )
        levels = list(problem.target.levels)
        name = levels[class_index] if class_index < len(levels) else str(class_index)

        def respond(frame: pd.DataFrame) -> Any:  # noqa: ANN401 - ndarray without numpy import
            """Return the probability of the plotted class for each row."""
            return model.predict_proba(frame)[:, class_index]  # type: ignore[attr-defined]

        return respond, f"P({name})"

    def respond_regression(frame: pd.DataFrame) -> Any:  # noqa: ANN401
        """Return the predicted target value for each row."""
        return model.predict(frame)  # type: ignore[attr-defined]

    unit = f" ({problem.target.unit})" if problem.target.unit else ""
    return respond_regression, f"predicted {problem.target.name}{unit}"


def _numeric_curve(
    model: object,
    frame: pd.DataFrame,
    problem: MLProblem,
    feature: str,
    *,
    grid_resolution: int,
    ice: bool,
    ice_rows: int,
    class_index: int,
    response_label: str,
) -> PDPCurve:
    """Compute a numeric feature's curve with ``sklearn.inspection.partial_dependence``.

    Args:
        model: The fitted estimator or pipeline over the raw frame.
        frame: Rows to average over (feature columns only are used).
        problem: The declared problem.
        feature: The numeric feature to vary.
        grid_resolution: Grid points between the 5th and 95th percentile.
        ice: Whether to compute individual curves as well as the average.
        ice_rows: How many individual curves to keep.
        class_index: Class column for a classifier.
        response_label: Human name of the y-axis.

    Returns:
        The curve.

    Raises:
        PartialDependenceUnavailableError: When sklearn refuses the feature — most often a
            constant column, which has no grid to vary over.
    """
    import numpy as np
    from sklearn.inspection import partial_dependence

    features = frame[problem.feature_names]
    kind = "both" if ice else "average"
    try:
        result = partial_dependence(
            model,
            features,
            features=[feature],
            grid_resolution=grid_resolution,
            kind=kind,
            response_method="predict_proba" if problem.target.task == "classification" else "auto",
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed refusal, never swallowed
        raise PartialDependenceUnavailableError(feature, str(exc)) from exc

    grid = np.asarray(result["grid_values"][0], dtype=float)
    average = np.asarray(result["average"])
    average = average[class_index] if average.ndim == 2 and average.shape[0] > 1 else average[0]

    individual: list[list[float]] = []
    if ice and "individual" in result:
        raw = np.asarray(result["individual"])
        block = raw[class_index] if raw.shape[0] > 1 else raw[0]
        step = max(1, block.shape[0] // max(1, ice_rows))
        individual = [list(map(float, row)) for row in block[::step][:ice_rows]]

    column = features[feature].to_numpy(dtype=float, na_value=np.nan)
    finite = column[np.isfinite(column)]
    support: list[float] = []
    if finite.size and grid.size > 1:
        edges = np.concatenate(
            [[-np.inf], (grid[:-1] + grid[1:]) / 2.0, [np.inf]]
        )
        counts, _ = np.histogram(finite, bins=edges)
        support = [float(c) / float(finite.size) for c in counts]

    return PDPCurve(
        feature=feature,
        kind="numeric",
        grid_labels=[f"{value:.4g}" for value in grid],
        grid_values=[float(value) for value in grid],
        average=[float(value) for value in np.asarray(average).reshape(-1)],
        support=support,
        ice=individual,
        n_rows=int(len(features)),
        response=response_label,
    )


def _categorical_curve(
    model: object,
    frame: pd.DataFrame,
    problem: MLProblem,
    feature: str,
    *,
    respond: Any,  # noqa: ANN401 - the response callable from _response_function
    ice: bool,
    ice_rows: int,
    response_label: str,
) -> PDPCurve:
    """Compute a categorical feature's curve by explicit level substitution.

    This is the textbook definition of partial dependence — hold the level fixed for every
    row, predict, average — applied directly. sklearn's numeric grid builder cannot produce
    a grid for a string column (it takes percentiles), so routing categoricals through it
    would either raise or, worse, silently coerce the levels to codes and plot a curve whose
    x-axis ordering is an artefact of the encoder.

    Levels come from the spec first and the observed data second, and a declared level with
    no rows is still plotted: the model's answer for a level it never saw in this frame is
    exactly the case a reader wants to inspect.

    Args:
        model: The fitted estimator or pipeline.
        frame: Rows to average over.
        problem: The declared problem.
        feature: The categorical feature to vary.
        respond: The response callable from :func:`_response_function`.
        ice: Whether to keep individual curves.
        ice_rows: How many individual curves to keep.
        response_label: Human name of the y-axis.

    Returns:
        The curve.

    Raises:
        PartialDependenceUnavailableError: When the feature has no levels to vary over.
    """
    import numpy as np

    features = frame[problem.feature_names]
    spec = next((f for f in problem.features if f.name == feature), None)
    declared = list(spec.levels) if spec is not None else []
    observed = [str(v) for v in features[feature].dropna().unique()]
    levels = declared or sorted(set(observed))
    if not levels:
        raise PartialDependenceUnavailableError(
            feature, "the column declares no levels and contains no non-null values"
        )

    step = max(1, len(features) // max(1, ice_rows)) if ice else 1
    keep = list(range(0, len(features), step))[:ice_rows] if ice else []

    average: list[float] = []
    per_row: list[list[float]] = [[] for _ in keep]
    for level in levels:
        counterfactual = features.copy()
        counterfactual[feature] = level
        try:
            responses = np.asarray(respond(counterfactual), dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed refusal
            raise PartialDependenceUnavailableError(
                feature, f"the model rejected the counterfactual level {level!r}: {exc}"
            ) from exc
        average.append(float(responses.mean()))
        for position, row_index in enumerate(keep):
            per_row[position].append(float(responses[row_index]))

    counts = features[feature].astype("object").map(lambda v: None if v is None else str(v))
    total = int(counts.notna().sum()) or 1
    support = [float((counts == level).sum()) / total for level in levels]

    return PDPCurve(
        feature=feature,
        kind="categorical",
        grid_labels=[str(level) for level in levels],
        grid_values=None,
        average=average,
        support=support,
        ice=per_row,
        n_rows=int(len(features)),
        response=response_label,
    )


def partial_dependence_curves(
    model: object,
    frame: pd.DataFrame,
    problem: MLProblem,
    features: Sequence[str] | None = None,
    *,
    grid_resolution: int = 20,
    max_samples: int = 500,
    ice: bool = False,
    ice_rows: int = 30,
    class_index: int = -1,
    seed: int | None = None,
) -> dict[str, PDPCurve]:
    """Compute partial-dependence (and optional ICE) curves for the declared features.

    Args:
        model: The fitted estimator or pipeline, consuming the raw domain frame.
        frame: Rows to average over — normally the held-out split.
        problem: The declared problem; supplies the feature list and their dtypes.
        features: Restrict to these features; defaults to every declared feature.
        grid_resolution: Grid points for numeric features.
        max_samples: Subsample the frame to this many rows before computing. PD costs one
            model call per grid point per row, so this is the knob that keeps a 20-point
            curve over 50 000 rows from becoming a million predictions per feature.
        ice: Compute individual conditional expectation curves as well as the average.
        ice_rows: How many individual curves to keep when ``ice`` is set.
        class_index: For a classifier, which class column to plot. Defaults to ``-1``, the
            last class — the positive class in the usual binary ordering.
        seed: Subsampling seed, so the demo reproduces.

    Returns:
        Feature name → :class:`PDPCurve`, in the order requested.

    Raises:
        ValueError: When a requested feature is not declared or not present in the frame.
        PartialDependenceUnavailableError: When a curve cannot be computed; it is raised
            rather than returned as a flat line.
    """
    requested = list(features) if features is not None else list(problem.feature_names)
    unknown = [name for name in requested if name not in problem.feature_names]
    if unknown:
        raise ValueError(
            f"Features {unknown} are not declared in the spec. A curve for an undeclared "
            f"column describes a model the adapter does not serve."
        )
    absent = [name for name in problem.feature_names if name not in frame.columns]
    if absent:
        raise ValueError(f"Frame is missing declared feature columns {absent}.")

    sample = frame
    if max_samples and len(frame) > max_samples:
        sample = frame.sample(n=max_samples, random_state=seed)

    respond, response_label = _response_function(model, problem, class_index)
    categorical = set(problem.categorical_features)

    curves: dict[str, PDPCurve] = {}
    for name in requested:
        if name in categorical:
            curves[name] = _categorical_curve(
                model,
                sample,
                problem,
                name,
                respond=respond,
                ice=ice,
                ice_rows=ice_rows,
                response_label=response_label,
            )
        else:
            curves[name] = _numeric_curve(
                model,
                sample,
                problem,
                name,
                grid_resolution=grid_resolution,
                ice=ice,
                ice_rows=ice_rows,
                class_index=class_index,
                response_label=response_label,
            )
    return curves


def _svg_line(curve: PDPCurve, *, width: int = 640, height: int = 220) -> str:
    """Render one curve as inline SVG — average as a line, ICE as faint traces.

    Grid points whose measured support is below 2% of rows are drawn with a hollow marker.
    That is where partial dependence is evaluating the model on feature combinations the
    data barely contains, and a curve that is confidently wrong at its ends is the most
    common way a PD plot misleads.

    Args:
        curve: The curve to draw.
        width: SVG width in px.
        height: SVG height in px.

    Returns:
        An ``<svg>`` element, or an empty string when the curve has fewer than two points.
    """
    points = len(curve.average)
    if points < 2:
        return ""
    pad_left, pad_right, pad_top, pad_bottom = 52, 12, 12, 42
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_values = [*curve.average, *[v for row in curve.ice for v in row]]
    low, high = min(all_values), max(all_values)
    if high == low:
        high = low + 1e-9

    def x_of(index: int) -> float:
        """Map a grid index to an x coordinate."""
        return pad_left + (plot_w * index / (points - 1))

    def y_of(value: float) -> float:
        """Map a response value to a y coordinate."""
        return pad_top + plot_h * (1.0 - (value - low) / (high - low))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="partial dependence for {escape(curve.feature)}">',
        f'<line class="axis" x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
        f'y2="{pad_top + plot_h}"/>',
        f'<line class="axis" x1="{pad_left}" y1="{pad_top + plot_h}" '
        f'x2="{pad_left + plot_w}" y2="{pad_top + plot_h}"/>',
        f'<text class="bar-value" x="2" y="{pad_top + 8}">{high:.3g}</text>',
        f'<text class="bar-value" x="2" y="{pad_top + plot_h}">{low:.3g}</text>',
    ]
    for row in curve.ice:
        if len(row) == points:
            trace = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(row))
            parts.append(
                f'<polyline points="{trace}" fill="none" stroke="var(--muted)" '
                f'stroke-width="1" opacity="0.28"/>'
            )
    line = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(curve.average))
    parts.append(
        f'<polyline points="{line}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>'
    )
    for index, value in enumerate(curve.average):
        thin = bool(curve.support) and index < len(curve.support) and curve.support[index] < 0.02
        fill = "var(--bg)" if thin else "var(--accent)"
        parts.append(
            f'<circle cx="{x_of(index):.1f}" cy="{y_of(value):.1f}" r="3" fill="{fill}" '
            f'stroke="var(--accent)" stroke-width="1.4"/>'
        )
    every = max(1, points // 6)
    for index in range(0, points, every):
        parts.append(
            f'<text class="bar-value" x="{x_of(index):.1f}" y="{height - 22}" '
            f'text-anchor="middle">{escape(curve.grid_labels[index])}</text>'
        )
    parts.append(
        f'<text class="bar-value" x="{pad_left}" y="{height - 6}">'
        f"{escape(curve.feature)} → {escape(curve.response)}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def render_html(
    path: str | Path,
    curves: dict[str, PDPCurve],
    problem: MLProblem,
    *,
    title: str | None = None,
    notes: Sequence[str] | None = None,
) -> Path:
    """Write a self-contained partial-dependence report: inline SVG, no external assets.

    Curves are ordered by their swing, largest first, so the features that move the model
    most are read first — but every requested curve is rendered, including the flat ones. A
    flat curve is the answer to "does this feature matter?", and omitting it would leave the
    question unanswered while looking like a complete report.

    Args:
        path: Destination ``.html`` file; parent directories are created.
        curves: Output of :func:`partial_dependence_curves`.
        problem: The declared problem, for the target's name and unit.
        title: Page title; defaults to naming the domain and target.
        notes: Extra sentences printed under the charts.

    Returns:
        The path written.

    Raises:
        ValueError: When ``curves`` is empty — an empty page reads as a successful run.
    """
    if not curves:
        raise ValueError(
            "render_html was given no curves. An empty page is indistinguishable from a "
            "successful report; compute partial_dependence_curves first."
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(curves.values(), key=lambda c: c.swing, reverse=True)
    body = [
        "<p class='sub'>Partial dependence holds one feature at each grid point, averages "
        "the model's output over the other features as they actually occur, and plots the "
        "result. It answers what a metric cannot: whether the model's response has the "
        "shape the domain says it should.</p>",
    ]
    for curve in ordered:
        thin = sum(1 for share in curve.support if share < 0.02)
        caveats = []
        if thin:
            caveats.append(
                f"{thin} grid point(s) sit where under 2% of rows fall — hollow markers. "
                f"The curve is extrapolating there."
            )
        if curve.ice:
            caveats.append(
                f"{len(curve.ice)} individual (ICE) curves are drawn faintly behind the "
                f"average. If they fan out while the average stays flat, the feature "
                f"interacts with another and the average is hiding it."
            )
        caveat_html = (
            f"<p class='sub'>{' '.join(escape(text) for text in caveats)}</p>"
            if caveats
            else ""
        )
        body.append(
            f"<h2>{escape(curve.feature)}</h2>"
            f"<p><span class='pill'>{escape(curve.kind)}</span>"
            f"<span class='pill'>swing {curve.swing:.4g}</span>"
            f"<span class='pill'>{curve.n_rows} rows</span></p>"
            f"<div class='scroll'>{_svg_line(curve)}</div>{caveat_html}"
        )
    if notes:
        body.append("<h2>Notes</h2><ul>")
        body.extend(f"<li>{escape(note)}</li>" for note in notes)
        body.append("</ul>")

    heading = title or f"Partial dependence — {problem.domain_id} / {problem.target.name}"
    document = html_page(
        heading,
        "".join(body),
        subtitle=(
            f"{escape(problem.domain_id)} · {escape(problem.target.task)} on "
            f"<code>{escape(problem.target.name)}</code>"
        ),
        footer=(
            "Partial dependence evaluates the model on feature combinations that may not "
            "occur in the data. Read the shape, and read the marked low-support points as "
            "extrapolation rather than as measurement."
        ),
    )
    destination.write_text(document, encoding="utf-8")
    return destination
