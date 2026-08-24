"""Build one run's visual bundle from that run's own artifacts, and record what it could not.

``registry_store/runs/<run_id>/visuals/`` is the directory a human opens. This module fills
it, and the contract it holds is narrower than "make some charts":

**Every number on every axis comes out of the run directory.** ``entry.json``,
``metrics.json``, ``leaderboard.json``, ``problem.json``, ``gate_inputs.json``,
``reference.parquet``, ``model.joblib``, ``drift.json``. Nothing here computes a metric from
a different frame than the one the run registered, and nothing here has a default value for
a missing input. When an input is absent the plot is **omitted** and ``manifest.json``
carries the reason in a sentence — because a blank axis with a title is the one output that
looks like evidence and is not, and a run whose SHAP artifact failed must not be
indistinguishable from a run whose features genuinely had no attribution.

**The held-out split is recovered, not assumed.** Three of the figures — prediction versus
measured, residuals, coverage by segment — are only worth drawing on the exact rows the
model never saw. Those rows are not stored, so :func:`recover_split` re-derives them and
then *proves* it got the right ones: it re-scores the persisted model on the reconstructed
test rows and requires the result to reproduce the metric the run registered, to floating
point. A reconstruction that does not reproduce it is rejected, and those three figures are
omitted with that stated as the reason. Reconstructing a split and hoping is how a chart
ends up quietly showing training rows.

**Re-runnable.** :func:`build_bundle` is a pure function of the run directory (plus two
optional frames a caller may supply), so ``aegis-ml visuals <run_id>`` regenerates the
bundle for any registered run at any time. It deletes the files it owns before it writes,
so a figure that was rendered by a previous build and is omitted by this one does not
survive to contradict the manifest.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.report import index as index_mod
from aegis_ml.report import plots, theme
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping, Sequence

    from aegis_ml.contracts.protocols import RegistryEntry

__all__ = [
    "CAPTIONS",
    "MANIFEST_NAME",
    "PLOT_FILES",
    "VISUALS_DIRNAME",
    "MissingInput",
    "PlotEntry",
    "RunAssets",
    "SplitRecovery",
    "build_bundle",
    "bundle_dir",
    "load_assets",
    "recover_split",
]

SERVE_EXTRA = "aegis-ml[serve]"
"""Install target named verbatim in every ImportError this module raises."""

VISUALS_DIRNAME = "visuals"
"""Subdirectory of the run directory this module owns, end to end."""

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.html"
INTERACTIVE_NAME = "interactive.html"
MANIFEST_SCHEMA = "aegis-ml/visuals-manifest/1"

PLOT_FILES: dict[str, str] = {
    "prediction_vs_actual": "01_prediction_vs_actual.png",
    "residuals": "02_residuals.png",
    "conformal_coverage": "03_conformal_coverage.png",
    "shap_global": "04_shap_global.png",
    "slice_performance": "05_slice_performance.png",
    "leaderboard": "06_leaderboard.png",
    "realism": "07_realism.png",
    "feature_distributions": "08_feature_distributions.png",
    "drift_features": "09_drift_features.png",
    "forecast": "10_forecast.png",
}
"""Slot name → file name. Numbered so ``ls`` shows them in reading order, which is also the
order the argument runs in: what the model predicts, how wrong it is, whether the interval
is honest, why it predicts that, where it is weakest, what it beat, whether the data was
hard, what the data looks like, whether the world has moved since."""

PLOT_TITLES: dict[str, str] = {
    "prediction_vs_actual": "Prediction vs measured, with the conformal band",
    "residuals": "Residuals across the prediction range",
    "conformal_coverage": "Conformal coverage — requested vs measured, overall and by segment",
    "shap_global": "Global SHAP attribution — every declared feature",
    "slice_performance": "Performance by segment",
    "leaderboard": "Leaderboard — every candidate the search scored",
    "realism": "Realism — is this data honestly hard?",
    "feature_distributions": "Feature distributions and missingness",
    "drift_features": "Drift — reference vs current distributions",
    "forecast": "Forecast with conformal band and backtest origins",
}

CAPTIONS: dict[str, str] = {
    "prediction_vs_actual": (
        "Each dot is one held-out row: measured value on the x axis, the model's prediction "
        "on the y. Look for the cloud hugging the dashed y=x line with roughly the requested "
        "share of dots inside the shaded conformal band. What would be wrong: a cloud that "
        "bends away from the line at one end (the model is biased in that range, and no "
        "interval width fixes bias), a flat cloud (the model is predicting the mean and has "
        "learned nothing), or misses concentrated at one end rather than scattered — that "
        "means the single interval width is wrong there even if the overall coverage number "
        "looks fine."
    ),
    "residuals": (
        "Residual (measured − predicted) against the prediction. Look for a band centred on "
        "zero whose height changes with the prediction — that fan is heteroscedasticity, and "
        "this domain's generator builds it in deliberately, so seeing it is the data behaving "
        "as documented. What would be wrong: a rolling mean that drifts off zero (bias, not "
        "noise), a visible curve or step (unmodelled structure the search missed), or a "
        "perfectly uniform band on data whose realism report claims heteroscedastic noise — "
        "that mismatch means the frame and the report describe different data."
    ),
    "conformal_coverage": (
        "Left: the coverage level asked for against the level measured on rows the fit and "
        "the calibration never saw. Right: the same measurement inside each segment, which "
        "is the panel that matters — marginal coverage is an average, so a band can hit its "
        "target overall while covering the easy majority generously and a tail segment far "
        "too thinly. Look for every bar at or above the dashed requested line. What would be "
        "wrong: any segment below the dotted gate floor, and especially a large segment "
        "there — every decision taken in that segment is being made with an interval that is "
        "narrower than its label claims."
    ),
    "shap_global": (
        "Mean absolute SHAP per feature: how much each column moves a single prediction, in "
        "target units. Every declared feature is shown, unfiltered. Look for the known "
        "drivers at the top and for the hatched grey columns — the ones the generator drew "
        "independently of the target — sitting at or near zero, which is the model "
        "demonstrating it did not memorise a spurious correlation. What would be wrong: a "
        "hatched column with real attribution (either the generator leaked or the search "
        "latched onto a sampling artifact), or one feature carrying almost all of the "
        "attribution, which usually means leakage rather than insight."
    ),
    "slice_performance": (
        "The primary metric recomputed inside each segment, sorted worst first, with the "
        "worst segment in the highlight colour and its row count on the label. The promotion "
        "gate reads this worst bar, not the mean. Look for a modest spread around the dashed "
        "whole-split line. What would be wrong: a long tail of collapsed segments, or a "
        "single very bad segment with a large row count — that is a real population the model "
        "fails for, and an aggregate score is exactly the instrument that cannot see it. A "
        "bad segment with 30-odd rows is a noisy estimate; the count is how you tell."
    ),
    "leaderboard": (
        "Every candidate the search scored, losers included, coloured by tier, with the "
        "promoted one outlined and starred. Look at the gap between the winner and the next "
        "bar: that margin is what says whether the extra machinery bought anything. Hatched "
        "bars are candidates that cannot be re-fitted in the serving venv — a hatched bar "
        "above the winner is an accuracy CEILING, reported as headroom, never promoted as the "
        "served model. What would be wrong: reading a hatched bar as this model's "
        "performance, or a winner separated from the baseline by less than the gate's minimum "
        "gain, which means the complexity is not earning its keep."
    ),
    "realism": (
        "The held-out score against the band a realistic generated frame should land in, and "
        "against the analytic ceiling implied by the generator's own noise and confounder "
        "variances. Look for the held-out bar inside the band and just under the ceiling — "
        "that combination means the search recovered nearly all of the signal that exists and "
        "the rest of the error is the world being unpredictable. What would be wrong: below "
        "the floor (the target is closer to noise than signal, and every interval downstream "
        "is honestly enormous) or above the ceiling, which is the more dangerous failure — it "
        "means the data was sampled with almost no noise and every downstream number "
        "describes a world that does not exist."
    ),
    "feature_distributions": (
        "One panel per declared feature over the frozen reference rows, with the share of "
        "missing values annotated. This is the context every other figure is conditioned on. "
        "Look for shapes that match the column's declared meaning and for missingness where "
        "the domain says a hole exists. What would be wrong: a spike at a single value (a "
        "constant feed rather than a measurement), a categorical level with a handful of rows "
        "(its slice metrics and its conformal sets carry no real guarantee), or missingness "
        "in a column declared non-nullable."
    ),
    "drift_features": (
        "The frozen reference distribution against the live one, ordered by measured movement "
        "— Kolmogorov–Smirnov distance for numeric columns, total-variation distance for "
        "categorical. Flagged features are in the highlight colour, stable ones are muted for "
        "context. Look at the shape of the move, not only its size: a shifted or widened "
        "distribution is an operational change, and the model is still serving. What would be "
        "wrong: a collapse onto one value or a vanished level, which is a broken feed rather "
        "than a changed world. A drifted model is not withdrawn — what drift blocks is "
        "promoting anything calibrated on a reference that no longer describes the world."
    ),
    "forecast": (
        "Observed history, the forecast, and its interval, with the rolling-origin backtest "
        "cutoffs marked. Look for a band whose width matches the historical volatility and "
        "for a measured coverage close to the requested level. What would be wrong: a band "
        "that narrows into the horizon (uncertainty grows with distance, it does not shrink), "
        "a forecast that flattens to a constant immediately, or a parametric interval quoted "
        "as though it were calibrated — only a conformal band was measured against held-out "
        "windows."
    ),
}
"""What to look for, and what would be wrong. A caption that names the figure teaches a
reader nothing they cannot see; the judgement is the part that is hard to reconstruct."""

_SPLIT_SEED_SEARCH = 128
"""How many split seeds :func:`recover_split` will try before giving up.

The flows take ``seed`` as an argument and do not record it, so the seed is recovered by
trying candidates and keeping only one that *reproduces the registered metric exactly*. The
bound exists so a run whose split genuinely cannot be reproduced fails in a second rather
than grinding; each candidate costs two ``train_test_split`` calls and one ``predict`` over
the test rows."""

_COVERAGE_SLICE_MIN_ROWS = 30
"""Smallest segment for which a coverage *rate* is reported.

Matches the floor the run's own slice sweep uses. Below it the measurement is one or two
rows wide and moves by whole percentage points per row, which reads as a finding and is
arithmetic."""

_EXACT = 1e-9
"""Relative tolerance for "this reconstruction reproduces the registered number".

Deliberately tight. The comparison is between two evaluations of the same deterministic
model on what should be the same rows, so anything beyond floating-point reassociation
means the rows differ — which is the thing being tested for."""


class MissingInput(AegisMLError):
    """A figure's input is absent, so the figure is omitted and the reason recorded.

    Carried as an exception rather than a return value because the check that discovers the
    absence is usually three calls deep inside a loader, and the alternative — threading an
    optional through every level — is how a ``None`` ends up being plotted as a zero.
    """

    def __init__(self, reason: str) -> None:
        """Record why the figure cannot be drawn, in a sentence a reader can act on."""
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class PlotEntry:
    """One row of the visual manifest: what was drawn, from what, with which numbers."""

    slot: str
    file: str
    title: str
    caption: str
    status: str
    inputs: list[str] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render as the plain JSON object ``manifest.json`` stores."""
        return {
            "slot": self.slot,
            "file": self.file,
            "title": self.title,
            "caption": self.caption,
            "status": self.status,
            "inputs": list(self.inputs),
            "numbers": self.numbers,
            "reason": self.reason,
        }


@dataclass(slots=True)
class SplitRecovery:
    """The outcome of re-deriving the held-out split, and the proof that it is the right one."""

    ok: bool
    reason: str
    source: str
    seed: int | None = None
    test_size: float | None = None
    calibration_size: float | None = None
    registered_metric: float | None = None
    recomputed_metric: float | None = None
    registered_coverage: float | None = None
    recomputed_coverage: float | None = None
    half_width: float | None = None
    train: Any = None
    calibration: Any = None
    test: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Render the evidence, without the frames, for ``manifest.json``."""
        return {
            "ok": self.ok,
            "reason": self.reason,
            "source": self.source,
            "seed": self.seed,
            "test_size": self.test_size,
            "calibration_size": self.calibration_size,
            "registered_metric": self.registered_metric,
            "recomputed_metric": self.recomputed_metric,
            "registered_coverage": self.registered_coverage,
            "recomputed_coverage": self.recomputed_coverage,
            "conformal_half_width": self.half_width,
        }


@dataclass(slots=True)
class RunAssets:
    """Everything readable from one run directory, with a reason for everything that is not."""

    run_id: str
    run_dir: Path
    entry: RegistryEntry
    problem: MLProblem | None = None
    reference: Any = None
    model: Any = None
    drift: dict[str, Any] | None = None
    gate_inputs: dict[str, Any] | None = None
    current: Any = None
    forecast: dict[str, Any] | None = None
    sources: dict[str, str] = field(default_factory=dict)
    absent: dict[str, str] = field(default_factory=dict)


def bundle_dir(run_id: str) -> Path:
    """Return ``<registry>/runs/<run_id>/visuals``, creating it if needed.

    Args:
        run_id: The registered run.

    Returns:
        The directory this module owns for that run.
    """
    from aegis_ml.registry import store

    directory = Path(store.run_dir(run_id)) / VISUALS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ──────────────────────────────────────────────────────────────────────── loading ──


def _read_json(path: Path) -> dict[str, Any]:
    """Parse a JSON object from ``path``."""
    return dict(json.loads(path.read_text(encoding="utf-8")))


def load_assets(
    run_id: str,
    *,
    current_frame: Any | None = None,  # noqa: ANN401 - a pandas.DataFrame
    forecast_payload: Mapping[str, Any] | None = None,
) -> RunAssets:
    """Read every artifact a figure could need, recording a reason for each absence.

    Nothing is fatal except the registry entry itself: a run with no SHAP artifact still has
    a leaderboard worth plotting, and a run with no drift report still has a model worth
    inspecting. Every absence lands in :attr:`RunAssets.absent` keyed by artifact, and those
    strings become the omission reasons in the manifest — which is the only reason a reader
    can tell "this was not measured" from "this measured zero".

    Args:
        run_id: The registered run.
        current_frame: The live frame drift was measured against. Falls back to
            ``current.parquet`` in the run directory, which ``drift_flow`` writes.
        forecast_payload: A ``forecast_flow`` payload. Falls back to ``forecast.json`` in the
            run directory, then to ``<reports>/forecasts/<run_id>.json``.

    Returns:
        The populated :class:`RunAssets`.

    Raises:
        FileNotFoundError: When the run is not registered — there is no run to report on.
    """
    from aegis_ml.registry import store

    entry = store.load_entry(run_id)
    directory = Path(store.run_dir(run_id))
    assets = RunAssets(run_id=run_id, run_dir=directory, entry=entry)
    assets.sources["entry.json"] = str(store.artifact(run_id, "entry.json"))

    problem_path = Path(entry.paths.get("problem", store.artifact(run_id, "problem.json")))
    if problem_path.exists():
        assets.problem = MLProblem.model_validate_json(problem_path.read_text(encoding="utf-8"))
        assets.sources["problem.json"] = str(problem_path)
    else:
        assets.absent["problem"] = (
            f"the run wrote no problem.json at {problem_path} — without the declared feature "
            f"list and target there is no way to say which columns a figure should show"
        )

    reference_path = Path(
        entry.paths.get("reference_frame", store.artifact(run_id, "reference.parquet"))
    )
    if reference_path.exists():
        assets.reference = require(SERVE_EXTRA, "pandas").read_parquet(reference_path)
        assets.sources["reference.parquet"] = str(reference_path)
    else:
        assets.absent["reference"] = (
            f"the run froze no reference frame at {reference_path}, so there are no rows to "
            f"re-derive the held-out split from"
        )

    model_path = Path(entry.paths.get("model", store.artifact(run_id, "model.joblib")))
    if model_path.exists():
        assets.model = require(SERVE_EXTRA, "joblib").load(model_path)
        assets.sources["model.joblib"] = str(model_path)
    else:
        assets.absent["model"] = (
            f"the run persisted no model at {model_path}; predictions, attributions and the "
            f"conformal band are all functions of the model and none can be recovered "
            f"without it"
        )

    drift_path = store.artifact(run_id, "drift.json")
    if drift_path.exists():
        assets.drift = _read_json(drift_path)
        assets.sources["drift.json"] = str(drift_path)
    else:
        assets.absent["drift"] = (
            "no drift.json in the run directory — drift_flow has not been run against this "
            "run, so there is no measured comparison to draw"
        )

    gate_path = Path(entry.paths.get("gate_inputs", store.artifact(run_id, "gate_inputs.json")))
    if gate_path.exists():
        assets.gate_inputs = _read_json(gate_path)
        assets.sources["gate_inputs.json"] = str(gate_path)
    else:
        assets.absent["gate_inputs"] = (
            f"no gate_inputs.json at {gate_path}; the realism report is measured at training "
            f"time and re-deriving it now would describe a different frame"
        )

    if current_frame is not None:
        assets.current = current_frame
        assets.sources["current_frame"] = "supplied by the caller"
    else:
        current_path = store.artifact(run_id, "current.parquet")
        if current_path.exists():
            assets.current = require(SERVE_EXTRA, "pandas").read_parquet(current_path)
            assets.sources["current.parquet"] = str(current_path)
        else:
            assets.absent["current"] = (
                "the live frame drift was measured against was not persisted with this run, "
                "so the reference/current overlay cannot be drawn from measured rows"
            )

    if forecast_payload is not None:
        assets.forecast = dict(forecast_payload)
        assets.sources["forecast"] = "supplied by the caller"
    else:
        candidates = [
            store.artifact(run_id, "forecast.json"),
            settings.reports_dir / "forecasts" / f"{run_id}.json",
        ]
        found = next((path for path in candidates if path.exists()), None)
        if found is not None:
            assets.forecast = _read_json(found)
            assets.sources["forecast"] = str(found)
        else:
            assets.absent["forecast"] = (
                "no forecast payload for this run — forecast_flow writes one per series and "
                "this run registered a tabular model, not a series"
            )

    return assets


# ─────────────────────────────────────────────────────────── held-out split recovery ──


def _conformal_quantile_level(n: int, coverage: float) -> float:
    """Return the finite-sample-corrected quantile level for a split-conformal band.

    ``ceil((n + 1) · coverage) / n``. The ``(n + 1)`` is the correction that makes the
    guarantee hold at finite ``n``; the plain empirical quantile under-covers by roughly
    ``1/n``. This is the same formula the measurement stage used, reproduced here rather
    than imported so this module does not reach into a flow's private helpers — and the
    result is checked against the coverage the run registered before anything is drawn.

    Args:
        n: Number of calibration residuals.
        coverage: The coverage level being requested.

    Returns:
        The quantile level, capped at 1.0.

    Raises:
        ValueError: When there are no residuals to take a quantile of.
    """
    if n < 1:
        raise ValueError("a split-conformal band needs at least one calibration residual")
    return min(1.0, math.ceil((n + 1) * coverage) / n)


def _covered_mask(
    model: Any,  # noqa: ANN401 - the fitted estimator
    calibration: Any,  # noqa: ANN401 - a pandas.DataFrame
    test: Any,  # noqa: ANN401 - a pandas.DataFrame
    problem: MLProblem,
    coverage: float,
) -> tuple[Any, float | None]:
    """Return a per-test-row "the interval contained the truth" mask, and the half-width.

    Regression calibrates an absolute-residual band and the half-width is meaningful.
    Classification calibrates a threshold on ``1 − p(true class)`` and produces prediction
    *sets*, for which no half-width exists — hence the ``None``. Both answer the same
    question, which is how often the honest answer contained the truth, so both feed the
    same coverage figures.

    Args:
        model: The fitted estimator.
        calibration: The disjoint calibration split.
        test: The disjoint test split.
        problem: The declared problem.
        coverage: The coverage level requested.

    Returns:
        ``(mask, half_width)`` — a boolean numpy array over the test rows, and the band's
        half-width for regression or ``None`` for classification.

    Raises:
        MissingInput: When a classifier exposes no ``predict_proba``, so no conformal set
            can be rebuilt from what the run persisted.
    """
    np = require(SERVE_EXTRA, "numpy")
    features = problem.feature_names
    target = problem.target.name

    if problem.target.task == "regression":
        residuals = np.abs(
            np.asarray(calibration[target], dtype=float)
            - np.asarray(model.predict(calibration[features]), dtype=float)
        )
        residuals = residuals[np.isfinite(residuals)]
        level = _conformal_quantile_level(int(residuals.size), coverage)
        width = float(np.quantile(residuals, level, method="higher"))
        truth = np.asarray(test[target], dtype=float)
        point = np.asarray(model.predict(test[features]), dtype=float)
        return np.abs(truth - point) <= width, width

    proba = getattr(model, "predict_proba", None)
    if proba is None:
        raise MissingInput(
            "the persisted classifier exposes no predict_proba, so the conformal prediction "
            "sets this run measured cannot be rebuilt from the artifacts"
        )
    classes = list(model.classes_)
    position = {value: i for i, value in enumerate(classes)}
    cal_proba = np.asarray(proba(calibration[features]), dtype=float)
    scores = np.array(
        [
            1.0 - cal_proba[i, position[value]]
            for i, value in enumerate(calibration[target])
            if value in position
        ],
        dtype=float,
    )
    level = _conformal_quantile_level(int(scores.size), coverage)
    threshold = float(np.quantile(scores, level, method="higher"))
    test_proba = np.asarray(proba(test[features]), dtype=float)
    mask = np.array(
        [
            (1.0 - test_proba[i, position[value]]) <= threshold if value in position else False
            for i, value in enumerate(test[target])
        ],
        dtype=bool,
    )
    return mask, None


def recover_split(
    assets: RunAssets,
    *,
    seed_search: int = _SPLIT_SEED_SEARCH,
) -> SplitRecovery:
    """Re-derive the run's held-out split, and prove it is the right one before returning it.

    The split is a deterministic function of the frozen reference frame, the two split
    fractions and the seed. The frame is stored; the fractions and the seed are flow
    arguments and are not. So this walks the candidates and keeps only one that **reproduces
    the metric the run registered**, to floating point, when the persisted model is re-scored
    on the reconstructed test rows. A candidate that matches on row counts but not on the
    metric is a different split that happens to be the same size, and it is rejected.

    ``split.json`` short-circuits the search when a newer run recorded its own split
    parameters — the verification still runs, because a recorded parameter that has been
    edited is exactly as wrong as a guessed one.

    Args:
        assets: The loaded run artifacts.
        seed_search: How many seeds to try. Each candidate is two splits and one predict.

    Returns:
        A :class:`SplitRecovery`. When :attr:`SplitRecovery.ok` is false, the reason is a
        sentence naming what stopped it, and the figures that need held-out rows are omitted
        rather than drawn on rows the model was fitted on.
    """
    from aegis_ml.data import splits
    from aegis_ml.evaluate import metrics as metrics_mod

    result = assets.entry.result
    if assets.problem is None or assets.reference is None or assets.model is None:
        blocked = [
            name for name in ("problem", "reference", "model") if getattr(assets, name) is None
        ]
        return SplitRecovery(
            ok=False,
            source="unavailable",
            reason=(
                f"cannot re-derive the held-out split: this run is missing {', '.join(blocked)}. "
                + " ".join(assets.absent.get(name, "") for name in blocked).strip()
            ),
        )

    problem = assets.problem
    frame = assets.reference
    candidates: list[tuple[int, float, float]] = []
    recorded = assets.run_dir / "split.json"
    source = "verified reconstruction"
    if recorded.exists():
        stored = _read_json(recorded)
        candidates.append(
            (
                int(stored.get("seed", settings.random_seed)),
                float(stored.get("test_size", 0.2)),
                float(stored.get("calibration_size", 0.2)),
            )
        )
        source = "split.json, verified"
    # The flows' own defaults, then a bounded seed sweep. Only a candidate that reproduces
    # the registered metric is ever accepted, so a wrong guess cannot become a wrong chart.
    candidates.extend(
        (seed, 0.2, 0.2)
        for seed in [settings.random_seed, *range(seed_search)]
    )

    tried: set[tuple[int, float, float]] = set()
    for seed, test_size, calibration_size in candidates:
        key = (seed, test_size, calibration_size)
        if key in tried:
            continue
        tried.add(key)
        try:
            parts = splits.three_way_split(
                frame,
                problem,
                test_size=test_size,
                calibration_size=calibration_size,
                seed=seed,
            )
        except AegisMLError:
            continue
        if (
            len(parts.train) != result.training_size
            or len(parts.calibration) != result.calibration_size
            or len(parts.test) != result.test_size
        ):
            continue
        predictions = assets.model.predict(parts.test[problem.feature_names])
        scored = metrics_mod.score(
            problem, list(parts.test[problem.target.name]), list(predictions)
        )
        name, value = metrics_mod.primary(problem, scored)
        if name != result.metric_name or not math.isclose(
            value, result.metric_value, rel_tol=_EXACT, abs_tol=_EXACT
        ):
            continue

        recovery = SplitRecovery(
            ok=True,
            source=source if key == candidates[0] else "verified reconstruction",
            reason=(
                f"recovered at seed={seed}, test_size={test_size}, "
                f"calibration_size={calibration_size}: re-scoring the persisted model on "
                f"these {len(parts.test)} rows reproduces the registered "
                f"{result.metric_name}={result.metric_value:.12g}, so they are the rows the "
                f"run measured on"
            ),
            seed=seed,
            test_size=test_size,
            calibration_size=calibration_size,
            registered_metric=float(result.metric_value),
            recomputed_metric=float(value),
            registered_coverage=result.empirical_coverage,
            train=parts.train,
            calibration=parts.calibration,
            test=parts.test,
        )
        try:
            mask, width = _covered_mask(
                assets.model, parts.calibration, parts.test, problem, result.requested_coverage
            )
            recovery.half_width = width
            recovery.recomputed_coverage = float(mask.mean())
        except (MissingInput, ValueError) as exc:
            recovery.reason += f". The conformal band could not be rebuilt: {exc}"
        return recovery

    return SplitRecovery(
        ok=False,
        source="unavailable",
        reason=(
            f"no split among {len(tried)} candidates reproduced the registered "
            f"{result.metric_name}={result.metric_value:.6g} on {result.test_size} test rows. "
            f"The run was trained with split parameters outside the searched set, or on a "
            f"frame that differs from the frozen reference. Figures that need held-out rows "
            f"are omitted rather than drawn on rows the model was fitted on."
        ),
        registered_metric=float(result.metric_value),
    )


def _coverage_by_segment(
    test: Any,  # noqa: ANN401 - a pandas.DataFrame
    mask: Any,  # noqa: ANN401 - a boolean numpy array
    problem: MLProblem,
) -> list[tuple[str, float, int]]:
    """Measure interval coverage inside each categorical level of the held-out split.

    Segments are the declared categorical levels carrying at least
    :data:`_COVERAGE_SLICE_MIN_ROWS` rows — the same floor the run's own slice sweep uses,
    so a segment that appears on the performance chart appears here too and the two can be
    read against each other.

    Args:
        test: The held-out split.
        mask: Per-row "the interval contained the truth", aligned with ``test``.
        problem: The declared problem, read for its categorical features.

    Returns:
        ``(label, coverage, n_rows)`` per segment, unsorted.
    """
    np = require(SERVE_EXTRA, "numpy")
    covered = np.asarray(mask, dtype=bool)
    rows: list[tuple[str, float, int]] = []
    for name in problem.categorical_features:
        if name not in test.columns:
            continue
        values = test[name].astype(str).to_numpy()
        for level in sorted(set(values)):
            selector = values == level
            count = int(selector.sum())
            if count < _COVERAGE_SLICE_MIN_ROWS:
                continue
            rows.append((f"{name} = {level}", float(covered[selector].mean()), count))
    return rows


# ───────────────────────────────────────────────────────────────────── the bundle ──


def _attempt(
    entries: list[PlotEntry],
    slot: str,
    inputs: Sequence[str],
    render: Callable[[Path], dict[str, Any]],
    directory: Path,
) -> None:
    """Render one figure, or record why it was omitted. Never leaves a half-written file.

    The exception is caught rather than propagated because one unavailable figure must not
    cost a reader the other nine — but it is *recorded*, with its type and message, in the
    manifest and on the page. That is the opposite of a silent fallback: the omission is
    louder in the report than the figure would have been.

    Args:
        entries: The manifest rows being accumulated.
        slot: Which figure this is.
        inputs: Artifact names this figure reads, for the lineage record.
        render: Callable that draws the figure at the given path and returns its numbers.
        directory: The visuals directory.
    """
    target = directory / PLOT_FILES[slot]
    row = PlotEntry(
        slot=slot,
        file=PLOT_FILES[slot],
        title=PLOT_TITLES[slot],
        caption=CAPTIONS[slot],
        status="omitted",
        inputs=list(inputs),
    )
    try:
        row.numbers = render(target)
        row.status = "rendered"
    except Exception as exc:
        row.reason = (
            exc.reason if isinstance(exc, MissingInput) else f"{type(exc).__name__}: {exc}"
        )
        if target.exists():
            target.unlink()
    entries.append(row)


def _forecast_series(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the plottable pieces out of a ``forecast_flow`` payload.

    Args:
        payload: ``{"forecast": ..., "backtest": ...}`` as the flow writes it.

    Returns:
        Keyword arguments for :func:`aegis_ml.report.plots.forecast_panel`.

    Raises:
        MissingInput: When the payload carries no history or no forecast points.
    """
    from aegis_ml.forecast.backtest import rolling_origin_cutoffs

    run = dict(payload.get("forecast") or {})
    history = [
        (datetime.fromisoformat(str(item["ts"])), float(item["value"]))
        for item in run.get("history") or []
    ]
    points = [
        (
            datetime.fromisoformat(str(item["ts"])),
            float(item["point"]),
            float(item["lo"]),
            float(item["hi"]),
        )
        for item in run.get("points") or []
    ]
    if not history or not points:
        raise MissingInput(
            "the forecast payload carries no history or no forecast points, so there is "
            "nothing measured to draw"
        )
    cutoffs: list[datetime] = []
    windows = int(run.get("backtest_windows") or 0)
    horizon = int(run.get("horizon") or 0)
    if windows and horizon:
        indices = rolling_origin_cutoffs(len(history), horizon=horizon, windows=windows)
        cutoffs = [history[i - 1][0] for i in indices if 0 < i <= len(history)]
    return {
        "history": history,
        "points": points,
        "cutoffs": cutoffs,
        "label": str(run.get("label") or run.get("series_id") or "series"),
        "unit": run.get("unit"),
        "model": str(run.get("model") or "unknown"),
        "requested_coverage": float(run.get("requested_coverage") or 0.0),
        "empirical_coverage": float(run.get("empirical_coverage") or 0.0),
        "interval_method": str(run.get("interval_method") or "unstated"),
    }


def _verdict(assets: RunAssets, recovery: SplitRecovery) -> dict[str, Any]:
    """Assemble the header a reader sees before any figure.

    Args:
        assets: The loaded artifacts.
        recovery: The split-recovery outcome, so the header can say whether the held-out
            figures below it are drawn on proven rows.

    Returns:
        A JSON-safe dict of the run's headline facts.
    """
    entry = assets.entry
    result = entry.result
    gate = entry.gate
    drift = assets.drift or {}
    return {
        "run_id": entry.run_id,
        "domain_id": entry.domain_id,
        "created_at": entry.created_at,
        "stage": entry.stage,
        "task": result.task,
        "target": result.target,
        "target_unit": assets.problem.target.unit if assets.problem else None,
        "metric_name": result.metric_name,
        "metric_value": float(result.metric_value),
        "requested_coverage": float(result.requested_coverage),
        "empirical_coverage": result.empirical_coverage,
        "coverage_tolerance": float(settings.coverage_tolerance),
        "training_size": result.training_size,
        "calibration_size": result.calibration_size,
        "test_size": result.test_size,
        "dataset_digest": result.dataset_digest,
        "tier": result.recipe.tier if result.recipe else None,
        "gate_promoted": None if gate is None else bool(gate.promoted),
        "gate_reasons": [] if gate is None else list(gate.reasons),
        "gate_checks": {} if gate is None else dict(gate.checks),
        "drift_verdict": drift.get("verdict"),
        "drifted_share": drift.get("drifted_share"),
        "drifted_features": list(drift.get("drifted_features") or []),
        "estimated_metric_name": drift.get("estimated_metric_name"),
        "estimated_metric_value": drift.get("estimated_metric_value"),
        "held_out_rows_verified": recovery.ok,
        "notes": list(result.notes),
    }


def build_bundle(
    run_id: str,
    *,
    current_frame: Any | None = None,  # noqa: ANN401 - a pandas.DataFrame
    forecast_payload: Mapping[str, Any] | None = None,
    shap_max_samples: int = 300,
    seed_search: int = _SPLIT_SEED_SEARCH,
) -> Path:
    """Render every figure this run's artifacts support, and write the page that shows them.

    Idempotent by construction: the files this module owns are deleted before anything is
    written, so a figure that a previous build rendered and this one cannot no longer sits in
    the directory contradicting the manifest. Running it twice on an unchanged run produces
    the same bundle; running it after ``drift_flow`` produces one more figure.

    Args:
        run_id: A registered run.
        current_frame: The live frame drift was measured against, when the caller has it in
            hand. Otherwise ``current.parquet`` from the run directory is used.
        forecast_payload: A ``forecast_flow`` payload to draw, when the caller has one.
        shap_max_samples: Rows to explain when recomputing global attribution. SHAP's cost is
            linear in this and the attribution is an average, so a few hundred rows is a
            stable estimate; the count used is recorded on the figure and in the manifest.
        seed_search: Split-seed candidates :func:`recover_split` may try.

    Returns:
        The visuals directory.

    Raises:
        FileNotFoundError: When ``run_id`` is not registered.
        ImportError: When matplotlib/seaborn are missing — the whole bundle is figures, so
            there is nothing to degrade to.
    """
    directory = bundle_dir(run_id)
    for name in (*PLOT_FILES.values(), MANIFEST_NAME, INDEX_NAME, INTERACTIVE_NAME):
        stale = directory / name
        if stale.exists():
            stale.unlink()

    assets = load_assets(
        run_id, current_frame=current_frame, forecast_payload=forecast_payload
    )
    recovery = recover_split(assets, seed_search=seed_search)
    result = assets.entry.result
    problem = assets.problem
    entries: list[PlotEntry] = []
    target_unit = problem.target.unit if problem else None

    def _need(condition: bool, reason: str) -> None:
        """Raise :class:`MissingInput` when a precondition does not hold."""
        if not condition:
            raise MissingInput(reason)

    # ── 01 prediction vs actual ────────────────────────────────────────────────
    def render_prediction(path: Path) -> dict[str, Any]:
        _need(recovery.ok and problem is not None, recovery.reason)
        test = recovery.test
        truth = list(test[problem.target.name])
        predicted = list(assets.model.predict(test[problem.feature_names]))
        if problem.target.task == "regression":
            _need(
                recovery.half_width is not None,
                "the conformal half-width could not be rebuilt from the calibration split, "
                "and a prediction chart without its interval omits the thing this run "
                "measured",
            )
            return plots.prediction_vs_actual(
                path,
                y_true=[float(value) for value in truth],
                y_pred=[float(value) for value in predicted],
                half_width=float(recovery.half_width or 0.0),
                requested_coverage=float(result.requested_coverage),
                metric_name=result.metric_name,
                metric_value=float(result.metric_value),
                target=result.target,
                unit=target_unit,
            )
        proba = getattr(assets.model, "predict_proba", None)
        classes = list(getattr(assets.model, "classes_", problem.target.levels))
        return plots.classification_overview(
            path,
            y_true=truth,
            y_pred=predicted,
            classes=classes,
            y_proba=None if proba is None else proba(test[problem.feature_names]),
            target=result.target,
        )

    _attempt(
        entries,
        "prediction_vs_actual",
        ["entry.json", "problem.json", "reference.parquet", "model.joblib"],
        render_prediction,
        directory,
    )

    # ── 02 residuals ───────────────────────────────────────────────────────────
    def render_residuals(path: Path) -> dict[str, Any]:
        _need(recovery.ok and problem is not None, recovery.reason)
        _need(
            problem.target.task == "regression",
            "residual-versus-prediction is a regression figure; this run is a "
            "classification run and its error structure is on the confusion matrix instead",
        )
        test = recovery.test
        noise = dict((assets.gate_inputs or {}).get("realism", {}).get("noise") or {})
        return plots.residuals(
            path,
            y_true=[float(value) for value in test[problem.target.name]],
            y_pred=[
                float(value) for value in assets.model.predict(test[problem.feature_names])
            ],
            target=result.target,
            unit=target_unit,
            heteroscedastic_feature=noise.get("heteroscedastic_feature"),
        )

    _attempt(
        entries,
        "residuals",
        ["entry.json", "problem.json", "reference.parquet", "model.joblib"],
        render_residuals,
        directory,
    )

    # ── 03 conformal coverage ──────────────────────────────────────────────────
    def render_coverage(path: Path) -> dict[str, Any]:
        _need(
            result.empirical_coverage is not None,
            "this run recorded no empirical coverage — the fitted estimator exposed no "
            "interval or probability interface, and the run reported null rather than "
            "defaulting to the level it asked for",
        )
        by_slice: list[tuple[str, float, int]] = []
        if recovery.ok and problem is not None and recovery.recomputed_coverage is not None:
            mask, _ = _covered_mask(
                assets.model,
                recovery.calibration,
                recovery.test,
                problem,
                float(result.requested_coverage),
            )
            by_slice = _coverage_by_segment(recovery.test, mask, problem)
        return plots.conformal_coverage(
            path,
            requested=float(result.requested_coverage),
            measured=float(result.empirical_coverage or 0.0),
            tolerance=float(settings.coverage_tolerance),
            by_slice=by_slice,
        )

    _attempt(
        entries,
        "conformal_coverage",
        ["entry.json", "metrics.json", "reference.parquet", "model.joblib"],
        render_coverage,
        directory,
    )

    # ── 04 SHAP global ─────────────────────────────────────────────────────────
    def render_shap(path: Path) -> dict[str, Any]:
        from aegis_ml.explain import shap_report

        _need(recovery.ok and problem is not None, recovery.reason)
        latent = dict((assets.gate_inputs or {}).get("realism", {}).get("latent") or {})
        importance = shap_report.global_importance(
            assets.model,
            recovery.test,
            problem,
            max_samples=shap_max_samples,
            seed=recovery.seed,
        )
        return plots.shap_global(
            path,
            importance=importance,
            irrelevant=list(latent.get("undriven_features") or []),
        )

    _attempt(
        entries,
        "shap_global",
        ["model.joblib", "reference.parquet", "problem.json", "gate_inputs.json"],
        render_shap,
        directory,
    )

    # ── 05 slice performance ───────────────────────────────────────────────────
    def render_slices(path: Path) -> dict[str, Any]:
        _need(
            bool(result.slices),
            "this run registered no per-slice metrics; the slice sweep is an optional stage "
            "and its absence is recorded in the run manifest",
        )
        return plots.slice_performance(
            path,
            slices=result.slices,
            metric_name=result.metric_name,
            overall=float(result.metric_value),
        )

    _attempt(entries, "slice_performance", ["entry.json"], render_slices, directory)

    # ── 06 leaderboard ─────────────────────────────────────────────────────────
    def render_leaderboard(path: Path) -> dict[str, Any]:
        board = result.leaderboard
        _need(
            board is not None and bool(board.candidates),
            "this run registered no leaderboard, so there are no scored candidates to rank",
        )
        return plots.leaderboard(
            path,
            candidates=board.candidates,
            metric_name=board.metric_name,
            higher_is_better=board.higher_is_better,
        )

    _attempt(
        entries, "leaderboard", ["entry.json", "leaderboard.json"], render_leaderboard, directory
    )

    # ── 07 realism ─────────────────────────────────────────────────────────────
    def render_realism(path: Path) -> dict[str, Any]:
        from aegis_ml.pipelines.flows import realism_band_for

        _need(
            assets.gate_inputs is not None,
            assets.absent.get(
                "gate_inputs", "this run recorded no realism report at training time"
            ),
        )
        realism = dict((assets.gate_inputs or {}).get("realism") or {})
        _need(
            bool(realism),
            "gate_inputs.json carries no realism block for this run, so there is no measured "
            "band or ceiling to plot the achieved score against",
        )
        _need(problem is not None, assets.absent.get("problem", "no problem.json"))
        return plots.realism_panel(
            path,
            realism=realism,
            band=realism_band_for(problem),
            achieved_metric_value=float(result.metric_value),
        )

    _attempt(
        entries, "realism", ["gate_inputs.json", "problem.json", "entry.json"], render_realism,
        directory,
    )

    # ── 08 feature distributions ───────────────────────────────────────────────
    def render_distributions(path: Path) -> dict[str, Any]:
        _need(assets.reference is not None, assets.absent.get("reference", "no reference frame"))
        _need(problem is not None, assets.absent.get("problem", "no problem.json"))
        return plots.feature_distributions(
            path,
            frame=assets.reference,
            features=[spec.model_dump(mode="json") for spec in problem.features],
        )

    _attempt(
        entries,
        "feature_distributions",
        ["reference.parquet", "problem.json"],
        render_distributions,
        directory,
    )

    # ── 09 drift ───────────────────────────────────────────────────────────────
    def render_drift(path: Path) -> dict[str, Any]:
        _need(assets.drift is not None, assets.absent.get("drift", "no drift.json"))
        _need(assets.current is not None, assets.absent.get("current", "no current frame"))
        _need(assets.reference is not None, assets.absent.get("reference", "no reference frame"))
        _need(problem is not None, assets.absent.get("problem", "no problem.json"))
        return plots.drift_features(
            path,
            reference=assets.reference,
            current=assets.current,
            features=[spec.model_dump(mode="json") for spec in problem.features],
            drifted=list((assets.drift or {}).get("drifted_features") or []),
        )

    _attempt(
        entries,
        "drift_features",
        ["drift.json", "reference.parquet", "current.parquet", "problem.json"],
        render_drift,
        directory,
    )

    # ── 10 forecast ────────────────────────────────────────────────────────────
    def render_forecast(path: Path) -> dict[str, Any]:
        _need(assets.forecast is not None, assets.absent.get("forecast", "no forecast payload"))
        return plots.forecast_panel(path, **_forecast_series(assets.forecast or {}))

    _attempt(entries, "forecast", ["forecast.json"], render_forecast, directory)

    # ── interactive ────────────────────────────────────────────────────────────
    interactive_reason: str | None = None
    interactive_panels: list[str] = []
    try:
        scatter = None
        if recovery.ok and problem is not None and problem.target.task == "regression":
            test = recovery.test
            truth = [float(value) for value in test[problem.target.name]]
            predicted = [
                float(value) for value in assets.model.predict(test[problem.feature_names])
            ]
            hover = [
                "<br>".join(
                    f"{name} = {test.iloc[i][name]}" for name in problem.feature_names[:6]
                )
                for i in range(len(test))
            ]
            scatter = {
                "y_true": truth,
                "y_pred": predicted,
                "half_width": float(recovery.half_width or 0.0),
                "hover": hover,
            }
        board = None
        if result.leaderboard is not None and result.leaderboard.candidates:
            ranked = sorted(
                result.leaderboard.candidates,
                key=lambda item: item.metric_value,
                reverse=not result.leaderboard.higher_is_better,
            )
            board = {
                "names": [item.name for item in ranked],
                "values": [float(item.metric_value) for item in ranked],
                "colours": [
                    theme.PALETTE["accent"]
                    if item.selected
                    else theme.TIER_COLOURS.get(item.tier, theme.PALETTE["neutral"])
                    for item in ranked
                ],
                "metric_name": result.leaderboard.metric_name,
            }
        slice_bars = None
        if result.slices:
            ordered = sorted(result.slices, key=lambda item: item.metric_value)
            slice_bars = {
                "labels": [f"{item.feature} = {item.level} (n={item.n_rows})" for item in ordered],
                "values": [float(item.metric_value) for item in ordered],
                "metric_name": result.metric_name,
            }
        interactive_panels = plots.interactive_report(
            directory / INTERACTIVE_NAME,
            title=f"{assets.entry.domain_id} — {run_id}",
            scatter=scatter,
            board=board,
            slice_bars=slice_bars,
        )["panels"]
    except Exception as exc:
        interactive_reason = f"{type(exc).__name__}: {exc}"

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "visuals_dir": str(directory),
        "verdict": _verdict(assets, recovery),
        "split_recovery": recovery.to_dict(),
        "shap_max_samples": int(shap_max_samples),
        "sources": dict(assets.sources),
        "absent_inputs": dict(assets.absent),
        "plots": [row.to_dict() for row in entries],
        "interactive": {
            "file": INTERACTIVE_NAME if interactive_reason is None else None,
            "panels": interactive_panels,
            "reason": interactive_reason,
        },
        "rendered": sum(1 for row in entries if row.status == "rendered"),
        "omitted": sum(1 for row in entries if row.status == "omitted"),
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    index_mod.write_index(directory, manifest)
    return directory
