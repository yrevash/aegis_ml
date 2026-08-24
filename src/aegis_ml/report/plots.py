"""The figures themselves — one function per plot, each returning what it drew.

Every function here takes **already-loaded, already-measured data** and returns the numbers
it put on the canvas. Neither half of that is incidental:

* Taking loaded data means nothing in this module decides what a plot is allowed to show.
  :mod:`aegis_ml.report.bundle` does the loading, and it is the only place that can conclude
  "this input is missing, so this figure is omitted". A plotting function that could reach
  for a file could also quietly reach for a *different* file than the one the caption names.
* Returning the numbers means the manifest records what was plotted, not merely that
  something was. A reader who suspects a figure can check the axis against
  ``manifest.json`` without re-deriving anything, and a reviewer can diff two runs' numbers
  without opening a single PNG.

No function here invents a series, pads a short array, or draws an empty axis. If a caller
hands it nothing to plot it raises ``ValueError``, and the bundle records the omission with
its reason — an empty axis with a title is the one outcome that looks like evidence and
is not.

Every figure is closed by :func:`aegis_ml.report.theme.save` before the function returns.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.report import theme

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence
    from datetime import datetime
    from pathlib import Path

    import pandas as pd

    from aegis_ml.contracts.protocols import Candidate, SliceMetric

__all__ = [
    "classification_overview",
    "conformal_coverage",
    "drift_features",
    "feature_distributions",
    "forecast_panel",
    "interactive_report",
    "leaderboard",
    "prediction_vs_actual",
    "realism_panel",
    "residuals",
    "shap_global",
    "slice_performance",
]

SERVE_EXTRA = "aegis-ml[serve]"
"""Install target named verbatim in every ImportError raised from this module."""

_MIN_ROLLING_WINDOW = 15
"""Smallest rolling window used for the residual trend line.

Below this the "trend" is mostly the noise it is supposed to summarise, and a reader sees a
jagged line where the honest answer is that the split is too small to say."""


def _np() -> Any:  # noqa: ANN401 - the numpy module object
    """Import numpy through :func:`~aegis_ml._require.require`."""
    return require(SERVE_EXTRA, "numpy")


def _label(text: str, limit: int = 34) -> str:
    """Shorten a tick label so a horizontal bar chart stays readable.

    Args:
        text: The full label.
        limit: Maximum characters before the middle is elided.

    Returns:
        ``text`` unchanged, or its head and tail joined by an ellipsis.
    """
    if len(text) <= limit:
        return text
    head = (limit - 1) // 2
    return f"{text[:head]}…{text[-(limit - head - 1):]}"


# ────────────────────────────────────────────────────── 01 prediction vs actual ──


def prediction_vs_actual(
    path: Path,
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    half_width: float,
    requested_coverage: float,
    metric_name: str,
    metric_value: float,
    target: str,
    unit: str | None = None,
) -> dict[str, Any]:
    """Scatter predictions against truth on the held-out split, inside the conformal band.

    The y=x line is where a perfect model would sit; the shaded band is the interval the
    model actually emits, so a point outside the band is a row the published interval got
    wrong. Plotting the band on the *same* axes as the scatter is the whole point — coverage
    quoted as a single percentage is easy to nod at, while a visible cloud of misses
    concentrated at one end of the range is a finding.

    Args:
        path: Destination PNG.
        y_true: Measured target values from the held-out split.
        y_pred: The model's point predictions, aligned with ``y_true``.
        half_width: Conformal interval half-width, from the calibration split's residuals.
        requested_coverage: The coverage level that half-width was calibrated for.
        metric_name: Primary metric name, for the annotation.
        metric_value: Primary metric value as measured by the run.
        target: Target column name, for the axis labels.
        unit: Target unit, appended to the axis labels when present.

    Returns:
        ``{"n_rows", "half_width", "covered", "covered_share", "min", "max"}`` — the numbers
        the figure asserts, so the manifest can carry them.

    Raises:
        ValueError: When the arrays are empty or of different lengths.
    """
    np = _np()
    plt = theme.apply()
    truth = np.asarray(y_true, dtype=float)
    point = np.asarray(y_pred, dtype=float)
    if truth.size == 0 or truth.size != point.size:
        raise ValueError(
            f"prediction_vs_actual needs equal, non-empty arrays; got {truth.size} truths "
            f"and {point.size} predictions."
        )

    inside = np.abs(truth - point) <= half_width
    axis_label = f"{target} ({unit})" if unit else target
    low = float(min(truth.min(), point.min()))
    high = float(max(truth.max(), point.max()))
    pad = 0.04 * (high - low or 1.0)
    line = np.linspace(low - pad, high + pad, 2)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.fill_between(
        line,
        line - half_width,
        line + half_width,
        color=theme.PALETTE["primary_soft"],
        alpha=0.35,
        linewidth=0,
        label=f"{requested_coverage:.0%} conformal band (±{half_width:.4g})",
    )
    ax.plot(line, line, color=theme.PALETTE["ink"], linewidth=1.2, linestyle="--", label="y = x")
    ax.scatter(
        truth[inside],
        point[inside],
        s=18,
        alpha=0.55,
        color=theme.PALETTE["primary"],
        edgecolors="none",
        label=f"inside the interval — {int(inside.sum())} rows",
    )
    if (~inside).any():
        ax.scatter(
            truth[~inside],
            point[~inside],
            s=26,
            alpha=0.9,
            color=theme.PALETTE["accent"],
            edgecolors="none",
            label=f"outside the interval — {int((~inside).sum())} rows",
        )
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel(f"measured {axis_label}")
    ax.set_ylabel(f"predicted {axis_label}")
    ax.set_title(
        f"Prediction vs measured on the held-out split\n"
        f"{metric_name} = {metric_value:.4g} over {truth.size} rows"
    )
    ax.legend(loc="upper left")
    theme.save(fig, path)
    return {
        "n_rows": int(truth.size),
        "half_width": float(half_width),
        "covered": int(inside.sum()),
        "covered_share": float(inside.mean()),
        "target_min": low,
        "target_max": high,
    }


def classification_overview(
    path: Path,
    *,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    classes: Sequence[Any],
    y_proba: Any | None = None,  # noqa: ANN401 - an (n, k) array or None
    target: str,
) -> dict[str, Any]:
    """Confusion matrix plus predicted-probability histogram by true class.

    The classification counterpart of :func:`prediction_vs_actual`, and it needs two panels
    because the two failure modes are different. The matrix shows *which* classes the model
    confuses; the probability histogram shows whether it is confidently wrong or merely
    uncertain — a model whose errors sit at p≈0.5 is calibrated and hesitant, and a model
    whose errors sit at p≈0.95 is calibrated at nothing and will mislead every downstream
    consumer that thresholds on confidence.

    Args:
        path: Destination PNG.
        y_true: Measured labels from the held-out split.
        y_pred: Predicted labels, aligned with ``y_true``.
        classes: Class ordering, matching the columns of ``y_proba``.
        y_proba: Predicted probabilities, shape ``(n, len(classes))``. The right-hand panel
            is omitted when this is ``None``, which is what happens for an estimator with no
            ``predict_proba`` — the matrix is still real and still worth showing.
        target: Target column name, for the titles.

    Returns:
        ``{"n_rows", "classes", "confusion", "accuracy", "has_probabilities"}``.

    Raises:
        ValueError: When the label arrays are empty or of different lengths.
    """
    np = _np()
    plt = theme.apply()
    seaborn = require(SERVE_EXTRA, "seaborn")
    truth = np.asarray(list(y_true))
    predicted = np.asarray(list(y_pred))
    if truth.size == 0 or truth.size != predicted.size:
        raise ValueError(
            f"classification_overview needs equal, non-empty label arrays; got "
            f"{truth.size} truths and {predicted.size} predictions."
        )
    order = [str(value) for value in classes]
    index = {name: i for i, name in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=int)
    for actual, guess in zip(truth, predicted, strict=True):
        row, column = index.get(str(actual)), index.get(str(guess))
        if row is not None and column is not None:
            matrix[row, column] += 1

    panels = 2 if y_proba is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.4 * panels, 5.6), squeeze=False)
    ax = axes[0][0]
    seaborn.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.6,
        linecolor=theme.PALETTE["surface"],
        xticklabels=order,
        yticklabels=order,
        ax=ax,
    )
    ax.set_xlabel("predicted class")
    ax.set_ylabel("measured class")
    ax.set_title(f"Confusion on the held-out split — {target}")

    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        confidence = proba.max(axis=1)
        ax2 = axes[0][1]
        for position, name in enumerate(order):
            mask = truth.astype(str) == name
            if not mask.any():
                continue
            ax2.hist(
                confidence[mask],
                bins=20,
                range=(0.0, 1.0),
                alpha=0.6,
                label=f"true = {name} ({int(mask.sum())})",
                color=theme.SEQUENCE[position % len(theme.SEQUENCE)],
            )
        ax2.set_xlabel("predicted probability of the model's chosen class")
        ax2.set_ylabel("rows")
        ax2.set_title("Confidence, split by the class that was actually true")
        ax2.legend()

    theme.save(fig, path)
    correct = int(np.trace(matrix))
    return {
        "n_rows": int(truth.size),
        "classes": order,
        "confusion": matrix.tolist(),
        "accuracy": correct / int(truth.size),
        "has_probabilities": y_proba is not None,
    }


# ───────────────────────────────────────────────────────────────── 02 residuals ──


def residuals(
    path: Path,
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    target: str,
    unit: str | None = None,
    heteroscedastic_feature: str | None = None,
) -> dict[str, Any]:
    """Residual against prediction, with a rolling mean and a rolling ±1σ envelope.

    This is where heteroscedasticity becomes visible. A residual cloud of constant height is
    a model whose one interval width is honest everywhere; a cloud that fans out towards one
    end is a model whose *single* conformal half-width is too wide for the quiet region and
    too narrow for the loud one, and the marginal coverage number hides both. The generated
    reference domain is built with a deliberately heteroscedastic noise term, so the fan is
    the expected shape here — its absence would mean the generator or the split is not doing
    what the realism report claims.

    The rolling mean is the second thing to read: it should sit on zero. A mean that drifts
    away from zero at one end is bias, not noise, and no interval width fixes bias.

    Args:
        path: Destination PNG.
        y_true: Measured target values from the held-out split.
        y_pred: Point predictions, aligned with ``y_true``.
        target: Target column name, for the axis labels.
        unit: Target unit, appended to the axis labels when present.
        heteroscedastic_feature: The feature the data generator's noise scales with, when the
            run recorded one. Named in the title so a reader knows what the fan is about.

    Returns:
        ``{"n_rows", "mean_residual", "std_residual", "spread_ratio", "window"}``, where
        ``spread_ratio`` is the residual standard deviation of the highest-prediction decile
        over the lowest — the number the fan shape is showing.

    Raises:
        ValueError: When the arrays are empty or of different lengths.
    """
    np = _np()
    pd = require(SERVE_EXTRA, "pandas")
    plt = theme.apply()
    truth = np.asarray(y_true, dtype=float)
    point = np.asarray(y_pred, dtype=float)
    if truth.size == 0 or truth.size != point.size:
        raise ValueError(
            f"residuals needs equal, non-empty arrays; got {truth.size} truths and "
            f"{point.size} predictions."
        )

    residual = truth - point
    order = np.argsort(point)
    ordered_pred = point[order]
    ordered_residual = residual[order]
    window = max(_MIN_ROLLING_WINDOW, truth.size // 20)
    series = pd.Series(ordered_residual)
    rolling_mean = series.rolling(window, center=True, min_periods=max(5, window // 3)).mean()
    rolling_std = series.rolling(window, center=True, min_periods=max(5, window // 3)).std()

    decile = max(1, truth.size // 10)
    low_spread = float(np.std(ordered_residual[:decile]))
    high_spread = float(np.std(ordered_residual[-decile:]))

    axis_label = f"{target} ({unit})" if unit else target
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.axhline(0.0, color=theme.PALETTE["ink"], linewidth=1.1, linestyle="--", label="zero error")
    ax.fill_between(
        ordered_pred,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color=theme.PALETTE["primary_soft"],
        alpha=0.45,
        linewidth=0,
        label=f"rolling ±1σ (window {window})",
    )
    ax.scatter(
        ordered_pred,
        ordered_residual,
        s=16,
        alpha=0.5,
        color=theme.PALETTE["primary"],
        edgecolors="none",
        label="residual (measured − predicted)",
    )
    ax.plot(
        ordered_pred,
        rolling_mean,
        color=theme.PALETTE["accent"],
        linewidth=2.0,
        label="rolling mean residual",
    )
    ax.set_xlabel(f"predicted {axis_label}")
    ax.set_ylabel(f"residual ({unit})" if unit else "residual")
    driver = f" — generator scales noise with {heteroscedastic_feature}" if (
        heteroscedastic_feature
    ) else ""
    ax.set_title(
        f"Residual spread across the prediction range{driver}\n"
        f"σ of the lowest decile {low_spread:.4g} vs the highest {high_spread:.4g}"
    )
    ax.legend(loc="upper left", ncols=2)
    theme.save(fig, path)
    return {
        "n_rows": int(truth.size),
        "mean_residual": float(residual.mean()),
        "std_residual": float(residual.std()),
        "low_decile_std": low_spread,
        "high_decile_std": high_spread,
        "spread_ratio": (high_spread / low_spread) if low_spread else None,
        "window": int(window),
    }


# ───────────────────────────────────────────────────────── 03 conformal coverage ──


def conformal_coverage(
    path: Path,
    *,
    requested: float,
    measured: float,
    tolerance: float,
    by_slice: Sequence[tuple[str, float, int]] = (),
) -> dict[str, Any]:
    """Requested against measured coverage, overall and per segment.

    The overall panel answers "did the interval keep its promise on average". The per-slice
    panel answers the question that actually matters, which is whether it kept it *for the
    segment you are about to make a decision in*. Marginal coverage is an average over
    segments, so a band can hit 90% overall while covering 97% of the easy majority and 71%
    of a tail segment — and every decision taken in that tail is made with an interval that
    is quietly a third too narrow.

    Args:
        path: Destination PNG.
        requested: The coverage level asked for.
        measured: The coverage rate achieved on the held-out split.
        tolerance: The shortfall the gate tolerates before calling it a finding.
        by_slice: ``(label, measured_coverage, n_rows)`` per segment, computed on the same
            split with the same interval. Empty means the split was not recoverable, and the
            figure honestly shows only the overall panel.

    Returns:
        ``{"requested", "measured", "gap", "n_slices", "worst_slice", "below_requested"}``.
    """
    plt = theme.apply()
    has_slices = len(by_slice) > 0
    summary_axis = None
    if has_slices:
        ordered = sorted(by_slice, key=lambda item: item[1])
        height = max(5.0, 0.32 * len(ordered) + 2.2)
        # The two panels carry very different row counts — two bars against twenty — so they
        # get their own grid cells rather than a shared row. A 2-row marginal panel stretched
        # to the height of a 20-row segment panel produces two slabs the eye reads as the
        # most important thing on the figure, which is the opposite of true here.
        fig = plt.figure(figsize=(13.8, height))
        grid = fig.add_gridspec(
            2, 2, width_ratios=[1.0, 1.55], height_ratios=[1.0, 2.6], wspace=0.5, hspace=0.32
        )
        ax = fig.add_subplot(grid[0, 0])
        summary_axis = fig.add_subplot(grid[1, 0])
        summary_axis.axis("off")
        ax2 = fig.add_subplot(grid[:, 1])
    else:
        ordered = []
        fig, ax = plt.subplots(figsize=(7.0, 3.4))

    gap = measured - requested
    overall_colour = theme.PALETTE["good"] if gap >= 0 else (
        theme.PALETTE["warn"] if gap >= -tolerance else theme.PALETTE["bad"]
    )
    ax.barh(
        ["measured", "requested"],
        [measured, requested],
        color=[overall_colour, theme.PALETTE["neutral"]],
        height=0.5,
    )
    ax.axvline(requested, color=theme.PALETTE["ink"], linewidth=1.2, linestyle="--")
    ax.axvline(
        requested - tolerance,
        color=theme.PALETTE["bad"],
        linewidth=1.0,
        linestyle=":",
        label=f"gate floor ({requested - tolerance:.0%})",
    )
    if summary_axis is None:
        ax.legend(loc="lower right")
    for position, value in enumerate([measured, requested]):
        ax.text(value + 0.014, position, f"{value:.2%}", va="center", fontsize=10)
    ax.set_xlim(0.0, 1.2)
    ax.set_xlabel("coverage")
    ax.set_title(f"Marginal coverage\nasked {requested:.0%}, achieved {measured:.2%}")

    worst: tuple[str, float, int] | None = None
    below = 0
    if has_slices:
        worst = ordered[0]
        below = sum(1 for _, value, _ in ordered if value < requested)
        labels = [f"{_label(name)}  (n={rows})" for name, _, rows in ordered]
        values = [value for _, value, _ in ordered]
        colours = [
            theme.PALETTE["bad"]
            if value < requested - tolerance
            else (theme.PALETTE["warn"] if value < requested else theme.PALETTE["primary"])
            for value in values
        ]
        ax2.barh(labels, values, color=colours, height=0.68)
        ax2.axvline(
            requested,
            color=theme.PALETTE["ink"],
            linewidth=1.3,
            linestyle="--",
            label=f"requested {requested:.0%}",
        )
        ax2.axvline(requested - tolerance, color=theme.PALETTE["bad"], linewidth=1.0, linestyle=":")
        for position, value in enumerate(values):
            ax2.text(value + 0.014, position, f"{value:.1%}", va="center", fontsize=8.5)
        ax2.set_xlim(0.0, 1.13)
        ax2.set_xlabel("measured coverage within the segment")
        ax2.set_title(
            f"Coverage by segment — {below} of {len(ordered)} below the requested level"
        )
        ax2.legend(loc="lower right")
        ax2.invert_yaxis()
        if summary_axis is not None:
            worst_label, worst_value, worst_rows = ordered[0]
            paragraphs = [
                f"{below} of {len(ordered)} segments fall below the requested "
                f"{requested:.0%}.",
                f"Worst: {worst_label} at {worst_value:.1%} over {worst_rows} rows.",
                "The marginal figure above is an average over these segments, so it can "
                "stay inside tolerance while one of them does not.",
                f"Dotted line: the {requested - tolerance:.0%} floor the promotion gate "
                f"refuses below.",
            ]
            summary_axis.text(
                0.0,
                1.0,
                "\n\n".join(textwrap.fill(text, width=46) for text in paragraphs),
                transform=summary_axis.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
                color=theme.PALETTE["muted"],
            )

    theme.save(fig, path)
    return {
        "requested": float(requested),
        "measured": float(measured),
        "gap": float(gap),
        "tolerance": float(tolerance),
        "n_slices": len(ordered),
        "slices_below_requested": below,
        "worst_slice": (
            {"label": worst[0], "coverage": float(worst[1]), "n_rows": int(worst[2])}
            if worst
            else None
        ),
    }


# ───────────────────────────────────────────────────────── 04 SHAP global impact ──


def shap_global(
    path: Path,
    *,
    importance: Sequence[Mapping[str, Any]],
    irrelevant: Sequence[str] = (),
) -> dict[str, Any]:
    """Mean absolute SHAP per feature — every feature, including the ones that scored zero.

    Near-zero bars are not clutter to be filtered out; on this data they are the evidence.
    The reference domain declares which columns were generated *independently* of the target,
    and a model that gives those columns a bar indistinguishable from zero has demonstrably
    not memorised a spurious correlation. Dropping them from the chart would remove the only
    on-screen proof of that, and would also hide the opposite finding — an "irrelevant"
    column with real attribution means either the generator leaked or the model latched onto
    a sampling artifact, and both are worth an investigation.

    Args:
        path: Destination PNG.
        importance: Rows from :func:`aegis_ml.explain.shap_report.global_importance` —
            ``feature``, ``mean_abs_shap``, ``mean_shap``, ``share``, ``n_samples``.
        irrelevant: Features the run's realism report declares are not drivers of the target.
            Drawn muted and annotated rather than removed.

    Returns:
        ``{"n_features", "n_samples", "top_feature", "irrelevant_attribution_share"}``.

    Raises:
        ValueError: When ``importance`` is empty.
    """
    plt = theme.apply()
    rows = list(importance)
    if not rows:
        raise ValueError("shap_global needs at least one attributed feature.")
    rows.sort(key=lambda row: float(row["mean_abs_shap"]))
    names = [str(row["feature"]) for row in rows]
    values = [float(row["mean_abs_shap"]) for row in rows]
    ignored = set(irrelevant)
    total = sum(values) or 1.0

    fig, ax = plt.subplots(figsize=(8.2, max(3.6, 0.42 * len(rows) + 1.6)))
    bars = ax.barh(
        names,
        values,
        height=0.68,
        color=[
            theme.PALETTE["neutral"] if name in ignored else theme.PALETTE["primary"]
            for name in names
        ],
        hatch=["//" if name in ignored else "" for name in names],
        edgecolor=theme.PALETTE["muted"],
        linewidth=0.6,
    )
    span = max(values) or 1.0
    for bar, name, value in zip(bars, names, values, strict=True):
        suffix = "  ← declared not a driver" if name in ignored else ""
        ax.text(
            value + span * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4g} ({value / total:.1%}){suffix}",
            va="center",
            fontsize=9,
            color=theme.PALETTE["muted"] if name in ignored else theme.PALETTE["ink"],
        )
    ax.set_xlim(0.0, span * 1.42)
    ax.set_xlabel("mean |SHAP| — average impact on one prediction, in target units")
    n_samples = int(rows[0].get("n_samples") or 0)
    ax.set_title(
        f"Global attribution over {n_samples} held-out rows — all "
        f"{len(rows)} declared features, unfiltered"
    )
    if ignored:
        ax.text(
            0.99,
            0.02,
            "Hatched grey: columns the data generator drew independently of the target.\n"
            "Near-zero bars there are the model correctly ignoring them.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=theme.PALETTE["muted"],
        )
    theme.save(fig, path)
    ignored_share = sum(v for n, v in zip(names, values, strict=True) if n in ignored) / total
    return {
        "n_features": len(rows),
        "n_samples": n_samples,
        "top_feature": names[-1],
        "top_mean_abs_shap": values[-1],
        "irrelevant_features": sorted(ignored),
        "irrelevant_attribution_share": float(ignored_share),
    }


# ───────────────────────────────────────────────────────── 05 slice performance ──


def slice_performance(
    path: Path,
    *,
    slices: Sequence[SliceMetric],
    metric_name: str,
    overall: float,
) -> dict[str, Any]:
    """The primary metric per segment, sorted, with the worst segment called out.

    The gate reads the worst slice rather than the mean, and this figure is that decision
    made visible. An aggregate score is precisely the instrument that cannot see a model
    which improves overall while collapsing on one region — everybody in that region
    experiences a regression, and the headline number goes up.

    The row count on the worst bar is not decoration. A segment of 30 rows scoring badly is
    a noisy estimate; the same score over 300 rows is a defect. Without the count the reader
    cannot tell those apart, and the gate's decision looks arbitrary.

    Args:
        path: Destination PNG.
        slices: Measured per-segment metrics from the run.
        metric_name: The metric name, for the axis label.
        overall: The same metric measured across the whole held-out split, drawn as a
            reference line so each segment is read as a distance from it.

    Returns:
        ``{"n_slices", "worst", "best", "overall", "spread"}``.

    Raises:
        ValueError: When ``slices`` is empty.
    """
    plt = theme.apply()
    rows = sorted(slices, key=lambda item: item.metric_value)
    if not rows:
        raise ValueError("slice_performance needs at least one measured slice.")
    labels = [f"{item.feature} = {_label(str(item.level), 22)}  (n={item.n_rows})" for item in rows]
    values = [float(item.metric_value) for item in rows]
    colours = [theme.PALETTE["primary"]] * len(rows)
    colours[0] = theme.PALETTE["accent"]

    fig, ax = plt.subplots(figsize=(9.0, max(3.8, 0.30 * len(rows) + 1.8)))
    ax.barh(labels, values, color=colours, height=0.7)
    ax.axvline(
        overall,
        color=theme.PALETTE["ink"],
        linewidth=1.3,
        linestyle="--",
        label=f"whole held-out split: {metric_name} = {overall:.4g}",
    )
    span = (max(values) - min(min(values), overall)) or 1.0
    for position, value in enumerate(values):
        ax.text(
            value + span * 0.01,
            position,
            f"{value:.4g}",
            va="center",
            fontsize=8.5,
            color=theme.PALETTE["accent"] if position == 0 else theme.PALETTE["muted"],
        )
    ax.set_xlabel(metric_name)
    ax.set_title(
        f"{metric_name} by segment — worst: {rows[0].feature} = {rows[0].level} at "
        f"{rows[0].metric_value:.4g} over {rows[0].n_rows} rows"
    )
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    theme.save(fig, path)
    return {
        "n_slices": len(rows),
        "metric_name": metric_name,
        "overall": float(overall),
        "worst": {
            "feature": rows[0].feature,
            "level": rows[0].level,
            "n_rows": rows[0].n_rows,
            "metric_value": float(rows[0].metric_value),
        },
        "best": {
            "feature": rows[-1].feature,
            "level": rows[-1].level,
            "n_rows": rows[-1].n_rows,
            "metric_value": float(rows[-1].metric_value),
        },
        "spread": float(values[-1] - values[0]),
    }


# ──────────────────────────────────────────────────────────────── 06 leaderboard ──


def leaderboard(
    path: Path,
    *,
    candidates: Sequence[Candidate],
    metric_name: str,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Every candidate the search scored, coloured by tier, losers included.

    A leaderboard showing only the winner cannot say whether the winner won by a nose or a
    mile, and the margin is exactly the part that says whether the extra complexity was
    worth carrying. Keeping the losers on the chart is what makes the selection auditable.

    Non-portable candidates are hatched and outlined. That distinction is load-bearing in
    this system: a candidate that cannot be re-fitted in the serving venv is reported as an
    **accuracy ceiling**, never promoted as the spine — so a hatched bar at the top of the
    chart means "there is this much headroom", not "this is what you are running".

    Args:
        path: Destination PNG.
        candidates: Every scored candidate, winners and losers.
        metric_name: The metric all candidates were scored on.
        higher_is_better: Ranking direction, so the sort matches the gate's.

    Returns:
        ``{"n_candidates", "selected", "best", "ceiling_gap", "tiers"}``.

    Raises:
        ValueError: When ``candidates`` is empty.
    """
    plt = theme.apply()
    rows = list(candidates)
    if not rows:
        raise ValueError("leaderboard needs at least one scored candidate.")
    rows.sort(key=lambda item: item.metric_value, reverse=not higher_is_better)
    labels = [
        f"{'* ' if item.selected else ''}{_label(item.name, 28)}"
        f"{'' if item.portable else '  (not portable)'}"
        for item in rows
    ]
    values = [float(item.metric_value) for item in rows]
    colours = [theme.TIER_COLOURS.get(item.tier, theme.PALETTE["neutral"]) for item in rows]

    fig, ax = plt.subplots(figsize=(9.4, max(3.6, 0.40 * len(rows) + 2.0)))
    bars = ax.barh(labels, values, height=0.7, color=colours)
    # Per-bar styling is applied here rather than through list-valued kwargs: matplotlib
    # accepts a sequence for `color` but silently keeps only the first `hatch`, which would
    # drop exactly the distinction this chart exists to make.
    for bar, item in zip(bars, rows, strict=True):
        if item.selected:
            bar.set_edgecolor(theme.PALETTE["accent"])
            bar.set_linewidth(2.0)
        elif not item.portable:
            # The hatch is drawn in the edge colour, and the baseline tier's fill is the
            # same grey as `muted` — hatching in it would be invisible on exactly the bars
            # the hatch exists to mark. White reads on every tier colour.
            bar.set_edgecolor(theme.PALETTE["surface"])
            bar.set_linewidth(1.1)
            bar.set_hatch("//")
        else:
            bar.set_edgecolor(theme.PALETTE["muted"])
            bar.set_linewidth(0.6)
    low = min(values)
    high = max(values)
    span = (high - low) or 1.0
    for bar, item in zip(bars, rows, strict=True):
        ax.text(
            item.metric_value + span * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{item.metric_value:.4g}"
            + (f"   ← promoted, fitted in {item.fit_seconds:.2f}s" if item.selected else ""),
            va="center",
            fontsize=9,
            color=theme.PALETTE["accent"] if item.selected else theme.PALETTE["muted"],
        )
    ax.set_xlim(low - span * 0.12, high + span * 0.38)
    ax.set_xlabel(metric_name)
    tiers = sorted({item.tier for item in rows})
    handles = [
        plt.Line2D([], [], color=theme.TIER_COLOURS.get(tier, theme.PALETTE["neutral"]),
                   linewidth=8, label=f"tier: {tier}")
        for tier in tiers
    ]
    handles.append(
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=theme.PALETTE["neutral"],
            edgecolor=theme.PALETTE["surface"],
            hatch="//",
            label="hatched = not portable → reported as an accuracy CEILING, never promoted",
        )
    )
    selected = next((item for item in rows if item.selected), None)
    best = rows[0] if higher_is_better else rows[-1]
    ax.set_title(
        f"Every candidate scored on {metric_name} — {len(rows)} kept, winners and losers"
    )
    ax.legend(handles=handles, loc="upper right", fontsize=8.5)
    ax.invert_yaxis()
    theme.save(fig, path)
    ceiling_gap = (
        float(best.metric_value - selected.metric_value)
        if selected is not None and best is not selected
        else None
    )
    return {
        "n_candidates": len(rows),
        "metric_name": metric_name,
        "higher_is_better": bool(higher_is_better),
        "selected": None if selected is None else {
            "name": selected.name,
            "tier": selected.tier,
            "metric_value": float(selected.metric_value),
            "portable": bool(selected.portable),
        },
        "best": {
            "name": best.name,
            "tier": best.tier,
            "metric_value": float(best.metric_value),
            "portable": bool(best.portable),
        },
        "ceiling_gap": ceiling_gap,
        "non_portable": [item.name for item in rows if not item.portable],
        "tiers": tiers,
    }


# ──────────────────────────────────────────────────────────────────── 07 realism ──


def realism_panel(
    path: Path,
    *,
    realism: Mapping[str, Any],
    band: tuple[float, float],
    achieved_metric_value: float,
) -> dict[str, Any]:
    """Is this data honestly hard? Achieved score against the band and the analytic ceiling.

    Two failures look identical on a metric alone. A score below the band means the target
    is closer to noise than signal and every interval downstream is honestly enormous. A
    score *above* the band is the more dangerous one: on generated data it means the latent
    function was sampled with almost no noise, no confounder and no missingness, so the
    gate margin, the coverage and the SHAP story all describe a world that does not exist —
    and the model collapses on the first frame that does.

    The analytic ceiling is the third number and the one that settles the argument. It is
    computed from the generator's own noise and confounder variances: no model can beat it,
    so a held-out score sitting just under it means the search found essentially all of the
    recoverable signal, and the remaining error is the world being unpredictable rather than
    the model being weak.

    Args:
        path: Destination PNG.
        realism: The run's ``realism`` block, as
            :func:`aegis_ml.data.latent.realism_report` produced it.
        band: ``(floor, ceiling)`` a realistic frame should land inside.
        achieved_metric_value: The metric measured on the held-out split by this run.

    Returns:
        The numbers drawn, including which optional sub-panels were available.

    Raises:
        ValueError: When ``realism`` carries no ``achieved`` block to plot.
    """
    plt = theme.apply()
    achieved = dict(realism.get("achieved") or {})
    if not achieved:
        raise ValueError("realism_panel needs the realism report's 'achieved' block.")
    noise = dict(realism.get("noise") or {})
    missingness = dict(realism.get("missingness") or {})
    latent = dict(realism.get("latent") or {})
    metric = str(achieved.get("metric", "metric"))
    floor, ceiling = band

    panels = ["band"]
    if noise.get("signal_variance") is not None:
        panels.append("variance")
    if missingness:
        panels.append("missingness")
    elif latent.get("driven_features") is not None:
        panels.append("drivers")

    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 4.4), squeeze=False)
    drawn = dict(zip(panels, axes[0], strict=True))

    ax = drawn["band"]
    oracle = noise.get("oracle_r2")
    implied = noise.get("implied_r2_ceiling")
    ax.axhspan(floor, ceiling, color=theme.PALETTE["primary_soft"], alpha=0.35,
               label=f"realistic band [{floor:.2f}, {ceiling:.2f}]")
    marks: list[tuple[str, float, str]] = [
        ("probe on the\nwhole frame", float(achieved.get("value", 0.0)), theme.PALETTE["muted"]),
        ("held-out\n(this model)", float(achieved_metric_value), theme.PALETTE["primary"]),
    ]
    if isinstance(oracle, int | float):
        marks.append(("oracle\n(latent signal)", float(oracle), theme.PALETTE["good"]))
    if isinstance(implied, int | float):
        marks.append(("analytic\nceiling", float(implied), theme.PALETTE["accent"]))
    ax.bar(
        [name for name, _, _ in marks],
        [value for _, value, _ in marks],
        color=[colour for _, _, colour in marks],
        width=0.55,
    )
    for position, (_, value, _) in enumerate(marks):
        ax.text(position, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.0, max(1.0, max(value for _, value, _ in marks) + 0.12))
    ax.set_ylabel(metric)
    ax.set_title("Achieved vs the band and the ceiling")
    ax.legend(loc="lower left", fontsize=8.5)

    if "variance" in drawn:
        ax2 = drawn["variance"]
        sigma = float(noise.get("sigma") or 0.0)
        parts = {
            "latent signal": float(noise.get("signal_variance") or 0.0),
            "confounders\n(unobserved)": float(noise.get("confounder_variance") or 0.0),
            "noise (σ²)": sigma**2,
        }
        total = sum(parts.values()) or 1.0
        ax2.bar(
            list(parts),
            list(parts.values()),
            color=[theme.PALETTE["primary"], theme.PALETTE["warn"], theme.PALETTE["neutral"]],
            width=0.55,
        )
        for position, value in enumerate(parts.values()):
            ax2.text(position, value + total * 0.015, f"{value / total:.1%}", ha="center",
                     fontsize=9)
        ax2.set_ylabel("variance in target units²")
        ratio = noise.get("noise_to_signal")
        ax2.set_title(
            "Where the target's variance comes from"
            + (f"\nnoise-to-signal {float(ratio):.3f}" if isinstance(ratio, int | float) else "")
        )

    if "missingness" in drawn:
        ax3 = drawn["missingness"]
        items = sorted(missingness.items(), key=lambda item: -float(item[1]))
        ax3.barh(
            [_label(name, 24) for name, _ in items],
            [float(share) for _, share in items],
            color=theme.PALETTE["warn"],
            height=0.42,
        )
        for position, (_, share) in enumerate(items):
            ax3.text(float(share) + 0.002, position, f"{float(share):.2%}", va="center",
                     fontsize=9)
        ax3.set_xlim(0.0, max(0.02, max(float(share) for _, share in items)) * 1.45)
        # Keep one bar from filling the whole panel: a single missing column is a fact, not
        # a wall, and a bar sized to the axis reads as a much bigger hole than 4% is.
        ax3.set_ylim(-0.6, max(3.5, len(items) - 0.5))
        ax3.set_xlabel("share of rows missing")
        ax3.set_title("Missing-at-random holes the generator built in")
        ax3.invert_yaxis()
    elif "drivers" in drawn:
        ax3 = drawn["drivers"]
        driven = list(latent.get("driven_features") or [])
        undriven = list(latent.get("undriven_features") or [])
        ax3.bar(
            ["drive the target", "generated\nindependently"],
            [len(driven), len(undriven)],
            color=[theme.PALETTE["primary"], theme.PALETTE["neutral"]],
            width=0.5,
        )
        ax3.set_ylabel("features")
        ax3.set_title(
            f"{len(driven)} real drivers, {len(undriven)} irrelevant columns\n"
            f"{len(latent.get('confounders') or [])} unobserved confounders"
        )

    fig.tight_layout()
    fig.suptitle("Is this data honestly hard?", fontsize=13, fontweight="bold", y=1.06)
    theme.save(fig, path)
    return {
        "metric": metric,
        "band": [float(floor), float(ceiling)],
        "probe_value": float(achieved.get("value", 0.0)),
        "held_out_value": float(achieved_metric_value),
        "oracle": float(oracle) if isinstance(oracle, int | float) else None,
        "analytic_ceiling": float(implied) if isinstance(implied, int | float) else None,
        "in_band": bool(floor <= float(achieved_metric_value) <= ceiling),
        "panels": panels,
        "noise": {key: value for key, value in noise.items() if isinstance(value, int | float)},
        "missingness": {name: float(share) for name, share in missingness.items()},
    }


# ──────────────────────────────────────────────────── 08 feature distributions ──


def feature_distributions(
    path: Path,
    *,
    frame: pd.DataFrame,
    features: Sequence[Mapping[str, Any]],
    ncols: int = 3,
) -> dict[str, Any]:
    """One panel per declared feature, with its missingness share on the panel.

    The distributions are what every other figure is conditioned on. A slice that scores
    badly over 30 rows and a slice that scores badly over 300 read very differently, and
    this is where a reader sees which one they are looking at. The missingness annotation
    sits on the same panel deliberately: a hole is a property of the column, and putting it
    in a separate table is how it gets forgotten.

    Args:
        path: Destination PNG.
        frame: The frozen reference frame — the exact rows this model was calibrated on.
        features: The declared feature specs, each carrying ``name`` and ``dtype``.
        ncols: Panels per row.

    Returns:
        ``{"n_rows", "n_features", "missingness"}``.

    Raises:
        ValueError: When no declared feature is present in the frame.
    """
    plt = theme.apply()
    seaborn = require(SERVE_EXTRA, "seaborn")
    present = [spec for spec in features if str(spec["name"]) in frame.columns]
    if not present:
        raise ValueError(
            "feature_distributions found none of the declared features in the frame; "
            "the frame and the problem describe different data."
        )
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    flat = [axis for row in axes for axis in row]
    missingness: dict[str, float] = {}

    for axis, spec in zip(flat, present, strict=False):
        name = str(spec["name"])
        column = frame[name]
        missing = float(column.isna().mean())
        missingness[name] = missing
        values = column.dropna()
        if str(spec.get("dtype")) == "categorical":
            counts = values.astype(str).value_counts()
            axis.bar(
                [_label(str(level), 14) for level in counts.index],
                counts.to_numpy(),
                color=theme.PALETTE["primary"],
                width=0.68,
            )
            axis.tick_params(axis="x", rotation=30)
            axis.set_ylabel("rows")
        else:
            seaborn.histplot(
                values.astype(float),
                bins=28,
                kde=True,
                color=theme.PALETTE["primary"],
                edgecolor="none",
                ax=axis,
                line_kws={"color": theme.PALETTE["accent"], "linewidth": 1.6},
            )
            axis.set_ylabel("rows")
        unit = spec.get("unit")
        axis.set_xlabel(str(unit) if unit else "")
        axis.set_title(name, fontsize=10.5)
        axis.text(
            0.98,
            0.94,
            "complete" if missing == 0 else f"{missing:.2%} missing",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=theme.PALETTE["muted"] if missing == 0 else theme.PALETTE["warn"],
        )

    for axis in flat[len(present):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Feature distributions over the {len(frame)} reference rows",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    theme.save(fig, path)
    return {
        "n_rows": int(len(frame)),
        "n_features": len(present),
        "missingness": missingness,
    }


# ────────────────────────────────────────────────────────────────────── 09 drift ──


def _ks_statistic(reference: Any, current: Any) -> float:  # noqa: ANN401 - numpy arrays
    """Two-sample Kolmogorov–Smirnov statistic: the largest gap between the two ECDFs.

    Computed here rather than imported so this module depends on nothing outside the
    declared ``serve`` extra. It is the same quantity Evidently reports for a numeric
    column, which keeps the ordering on this chart consistent with the verdict in
    ``drift.json`` instead of introducing a second, differently-scaled notion of "moved".

    Args:
        reference: Values from the frozen reference frame, NaNs already dropped.
        current: Values from the live frame, NaNs already dropped.

    Returns:
        The statistic in ``[0, 1]``; ``0.0`` when either side is empty.
    """
    np = _np()
    left = np.sort(np.asarray(reference, dtype=float))
    right = np.sort(np.asarray(current, dtype=float))
    if left.size == 0 or right.size == 0:
        return 0.0
    grid = np.concatenate([left, right])
    left_cdf = np.searchsorted(left, grid, side="right") / left.size
    right_cdf = np.searchsorted(right, grid, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _total_variation(reference: Any, current: Any) -> float:  # noqa: ANN401 - pandas Series
    """Total-variation distance between two categorical level distributions.

    Args:
        reference: Reference values.
        current: Live values.

    Returns:
        Half the summed absolute difference in level shares, in ``[0, 1]``.
    """
    left = reference.astype(str).value_counts(normalize=True)
    right = current.astype(str).value_counts(normalize=True)
    levels = set(left.index) | set(right.index)
    return float(
        0.5 * sum(abs(float(left.get(level, 0.0)) - float(right.get(level, 0.0)))
                  for level in levels)
    )


def drift_features(
    path: Path,
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Sequence[Mapping[str, Any]],
    drifted: Sequence[str],
    max_panels: int = 9,
) -> dict[str, Any]:
    """Reference against live distributions, ordered by how far each feature moved.

    Ordering by measured magnitude rather than by declaration order is what makes this
    figure answer the question a reader actually has, which is not "did something drift" —
    ``drift.json`` already says that — but "what moved, and by how much, and does the shape
    of the move look like an operational change or like a broken feed". A doubled mean is a
    hot season; a spike at a single value is a sensor writing a constant.

    Features the drift report did **not** flag are drawn muted rather than dropped, so the
    reader can see the stable columns as context for the moved ones.

    Args:
        path: Destination PNG.
        reference: The frozen frame the model was calibrated on.
        current: The live frame drift was measured against.
        features: The declared feature specs, each carrying ``name`` and ``dtype``.
        drifted: Feature names the drift report flagged.
        max_panels: How many features to draw, strongest movement first.

    Returns:
        ``{"n_reference_rows", "n_current_rows", "magnitudes", "drifted", "shown"}``.

    Raises:
        ValueError: When no declared feature is present in both frames.
    """
    plt = theme.apply()
    flagged = set(drifted)
    magnitudes: list[tuple[str, str, float]] = []
    for spec in features:
        name = str(spec["name"])
        if name not in reference.columns or name not in current.columns:
            continue
        kind = "categorical" if str(spec.get("dtype")) == "categorical" else "numeric"
        if kind == "categorical":
            magnitude = _total_variation(reference[name].dropna(), current[name].dropna())
        else:
            magnitude = _ks_statistic(
                reference[name].dropna().to_numpy(), current[name].dropna().to_numpy()
            )
        magnitudes.append((name, kind, magnitude))
    if not magnitudes:
        raise ValueError(
            "drift_features found no declared feature present in both the reference and the "
            "current frame; the two frames describe different data."
        )

    magnitudes.sort(key=lambda item: -item[2])
    shown = magnitudes[:max_panels]
    ncols = 3
    nrows = (len(shown) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.3 * nrows), squeeze=False)
    flat = [axis for row in axes for axis in row]

    for axis, (name, kind, magnitude) in zip(flat, shown, strict=False):
        moved = name in flagged
        live_colour = theme.PALETTE["accent"] if moved else theme.PALETTE["neutral"]
        if kind == "categorical":
            left = reference[name].astype(str).value_counts(normalize=True)
            right = current[name].astype(str).value_counts(normalize=True)
            levels = sorted(set(left.index) | set(right.index))
            positions = range(len(levels))
            axis.bar(
                [p - 0.2 for p in positions],
                [float(left.get(level, 0.0)) for level in levels],
                width=0.4,
                color=theme.PALETTE["primary"],
                label="reference",
            )
            axis.bar(
                [p + 0.2 for p in positions],
                [float(right.get(level, 0.0)) for level in levels],
                width=0.4,
                color=live_colour,
                label="current",
            )
            axis.set_xticks(list(positions))
            axis.set_xticklabels([_label(level, 12) for level in levels], rotation=30)
            axis.set_ylabel("share of rows")
            stat = "total-variation"
        else:
            bins = 30
            axis.hist(
                reference[name].dropna().astype(float),
                bins=bins,
                density=True,
                alpha=0.55,
                color=theme.PALETTE["primary"],
                label="reference",
            )
            axis.hist(
                current[name].dropna().astype(float),
                bins=bins,
                density=True,
                alpha=0.55,
                color=live_colour,
                label="current",
            )
            axis.set_ylabel("density")
            stat = "KS"
        axis.set_title(
            f"{name} — {stat} {magnitude:.3f}" + ("  ⚑ flagged" if moved else "  (stable)"),
            fontsize=10.5,
            color=theme.PALETTE["accent"] if moved else theme.PALETTE["muted"],
        )
        axis.legend(fontsize=8)

    for axis in flat[len(shown):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Reference ({len(reference)} rows) vs current ({len(current)} rows), "
        f"strongest movement first",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    theme.save(fig, path)
    return {
        "n_reference_rows": int(len(reference)),
        "n_current_rows": int(len(current)),
        "magnitudes": {name: float(value) for name, _, value in magnitudes},
        "drifted": sorted(flagged),
        "shown": [name for name, _, _ in shown],
    }


# ─────────────────────────────────────────────────────────────────── 10 forecast ──


def forecast_panel(
    path: Path,
    *,
    history: Sequence[tuple[datetime, float]],
    points: Sequence[tuple[datetime, float, float, float]],
    cutoffs: Sequence[datetime] = (),
    label: str,
    unit: str | None,
    model: str,
    requested_coverage: float,
    empirical_coverage: float,
    interval_method: str,
) -> dict[str, Any]:
    """Observed history, the forecast, its band, and the backtest origins that earned it.

    The vertical marks are the rolling-origin cutoffs the model was actually selected on.
    They matter because a forecast band with no backtest behind it is a model assumption
    drawn as a shaded region — the coverage number in the title is only evidence if it was
    measured on held-out windows, and the marks are where those windows began.

    Args:
        path: Destination PNG.
        history: Observed ``(timestamp, value)`` pairs.
        points: Forecast ``(timestamp, point, lo, hi)`` tuples.
        cutoffs: Timestamps of the rolling-origin backtest cutoffs.
        label: Human label of the series.
        unit: Unit of the values.
        model: The selected model's name.
        requested_coverage: The coverage level asked for.
        empirical_coverage: The rate achieved on the held-out windows.
        interval_method: ``conformal`` or ``parametric`` — read before quoting the band.

    Returns:
        ``{"history_points", "horizon", "requested_coverage", "empirical_coverage"}``.

    Raises:
        ValueError: When there is no history or no forecast to draw.
    """
    plt = theme.apply()
    if not history or not points:
        raise ValueError(
            f"forecast_panel needs both history and forecast points; got {len(history)} "
            f"observations and {len(points)} forecast steps."
        )
    plt_history_x = [item[0] for item in history]
    plt_history_y = [float(item[1]) for item in history]
    forecast_x = [item[0] for item in points]
    forecast_y = [float(item[1]) for item in points]
    low = [float(item[2]) for item in points]
    high = [float(item[3]) for item in points]

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.plot(
        plt_history_x, plt_history_y, color=theme.PALETTE["ink"], linewidth=1.4, label="observed"
    )
    ax.fill_between(
        forecast_x,
        low,
        high,
        color=theme.PALETTE["primary_soft"],
        alpha=0.45,
        linewidth=0,
        label=f"{requested_coverage:.0%} {interval_method} band",
    )
    ax.plot(
        forecast_x,
        forecast_y,
        color=theme.PALETTE["primary"],
        linewidth=2.0,
        label=f"forecast — {model}",
    )
    for position, cutoff in enumerate(cutoffs):
        ax.axvline(
            cutoff,
            color=theme.PALETTE["accent"],
            linewidth=0.9,
            linestyle=":",
            alpha=0.8,
            label="backtest origin" if position == 0 else None,
        )
    ax.set_xlabel("time")
    ax.set_ylabel(f"{label} ({unit})" if unit else label)
    ax.set_title(
        f"{label} — {model} over {len(points)} steps\n"
        f"asked {requested_coverage:.0%}, held-out windows achieved {empirical_coverage:.1%}"
    )
    ax.legend(loc="upper left", ncols=2)
    fig.autofmt_xdate()
    theme.save(fig, path)
    return {
        "history_points": len(history),
        "horizon": len(points),
        "backtest_origins": len(cutoffs),
        "model": model,
        "interval_method": interval_method,
        "requested_coverage": float(requested_coverage),
        "empirical_coverage": float(empirical_coverage),
    }


# ────────────────────────────────────────────────────────────── interactive HTML ──


def interactive_report(
    path: Path,
    *,
    title: str,
    scatter: Mapping[str, Any] | None = None,
    board: Mapping[str, Any] | None = None,
    slice_bars: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a self-contained Plotly page: hover the point, read the row.

    The PNGs are the record; this is the one surface where a reader can ask a question the
    static figure cannot answer — *which* row is that outlier, what did the model predict
    for it, how far outside the band did it fall. Plotly's JavaScript is inlined rather than
    pulled from a CDN, because a report that needs the network is a report that is blank in
    exactly the rooms where it matters.

    Args:
        path: Destination HTML file.
        title: Page title.
        scatter: ``{"y_true", "y_pred", "half_width", "hover"}`` for the prediction panel.
        board: ``{"names", "values", "colours", "metric_name"}`` for the leaderboard panel.
        slice_bars: ``{"labels", "values", "metric_name"}`` for the slice panel.

    Returns:
        ``{"panels": [...]}`` naming the panels that were drawn.

    Raises:
        ValueError: When every panel is empty — an interactive page with nothing in it is
            a broken link, not a degraded figure.
        ImportError: When plotly is not installed, naming the install.
    """
    graph_objects = require(SERVE_EXTRA, "plotly.graph_objects")
    subplots = require(SERVE_EXTRA, "plotly.subplots")
    panels = [name for name, value in
              (("prediction", scatter), ("leaderboard", board), ("slices", slice_bars))
              if value]
    if not panels:
        raise ValueError("interactive_report was given no panel with data in it.")

    titles = {
        "prediction": "Prediction vs measured (hover a point for its row)",
        "leaderboard": "Every candidate scored",
        "slices": "Metric by segment",
    }
    figure = subplots.make_subplots(
        rows=len(panels),
        cols=1,
        subplot_titles=[titles[name] for name in panels],
        vertical_spacing=0.10,
    )

    for row, name in enumerate(panels, start=1):
        if name == "prediction" and scatter is not None:
            truth = [float(value) for value in scatter["y_true"]]
            predicted = [float(value) for value in scatter["y_pred"]]
            width = float(scatter["half_width"])
            hover = list(scatter.get("hover") or ["" for _ in truth])
            low, high = min(min(truth), min(predicted)), max(max(truth), max(predicted))
            figure.add_trace(
                graph_objects.Scatter(
                    x=[low, high, high, low],
                    y=[low - width, high - width, high + width, low + width],
                    fill="toself",
                    fillcolor="rgba(47, 109, 142, 0.18)",
                    line={"width": 0},
                    hoverinfo="skip",
                    name="conformal band",
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                graph_objects.Scatter(
                    x=[low, high],
                    y=[low, high],
                    mode="lines",
                    line={"color": theme.PALETTE["ink"], "dash": "dash", "width": 1.2},
                    name="y = x",
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                graph_objects.Scatter(
                    x=truth,
                    y=predicted,
                    mode="markers",
                    marker={
                        "size": 7,
                        "color": [
                            theme.PALETTE["accent"] if abs(t - p) > width
                            else theme.PALETTE["primary"]
                            for t, p in zip(truth, predicted, strict=True)
                        ],
                        "opacity": 0.7,
                    },
                    text=hover,
                    hovertemplate="measured %{x:.4g}<br>predicted %{y:.4g}<br>%{text}"
                    "<extra></extra>",
                    name="held-out rows",
                ),
                row=row,
                col=1,
            )
        elif name == "leaderboard" and board is not None:
            figure.add_trace(
                graph_objects.Bar(
                    x=list(board["values"]),
                    y=list(board["names"]),
                    orientation="h",
                    marker={"color": list(board["colours"])},
                    hovertemplate=f"%{{y}}<br>{board['metric_name']} %{{x:.4g}}<extra></extra>",
                    name=str(board["metric_name"]),
                ),
                row=row,
                col=1,
            )
        elif name == "slices" and slice_bars is not None:
            figure.add_trace(
                graph_objects.Bar(
                    x=list(slice_bars["values"]),
                    y=list(slice_bars["labels"]),
                    orientation="h",
                    marker={"color": theme.PALETTE["primary"]},
                    hovertemplate=f"%{{y}}<br>{slice_bars['metric_name']} %{{x:.4g}}"
                    "<extra></extra>",
                    name=str(slice_bars["metric_name"]),
                ),
                row=row,
                col=1,
            )

    figure.update_layout(
        title=title,
        height=520 * len(panels),
        template="plotly_white",
        showlegend=False,
        margin={"l": 220, "r": 60, "t": 90, "b": 60},
        font={"color": theme.PALETTE["ink"]},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)
    return {"panels": panels}
