"""The pipelines: ordinary Python functions that fit real models and measure real numbers.

Six flows, in the order a domain moves through them:

``data_flow``
    Ingest → schema contract → profile → learnability → realism → leakage → chronology-safe
    three-way split → frozen reference frame. Everything expensive downstream depends on
    this having passed, which is the point: an AutoML search over a target that carries no
    signal costs five minutes to discover what a learnability probe finds in two seconds.
``train_flow``
    AutoML search → Optuna tune → fit → measure on a held-out split → slices → SHAP →
    model card → registry. Returns a populated :class:`~aegis_ml.contracts.protocols.TrainResult`.
``eval_flow``
    Re-score a registered run on data it has never seen.
``promote_flow``
    Build the gate decision against the current champion, and promote only if it passes.
``drift_flow``
    Evidently drift against the frozen reference plus a NannyML label-free performance
    estimate — the half of monitoring that works before ground truth arrives.
``forecast_flow`` / ``full_flow``
    A series forecast, and the end-to-end bundle a demo is given.

Four properties hold across all of them.

**Every number is measured.** There is no branch anywhere in this module that produces a
metric without a fit. A stage that cannot measure records ``None`` and says why —
:attr:`TrainResult.empirical_coverage` is deliberately ``float | None`` for exactly that
case — because a plausible default is worse than a gap: nobody investigates a gap they
cannot see.

**Stages are content-addressed and resumable.** Each stage declares the context keys it
reads and writes, and the expensive one (the AutoML search) is keyed on the frame digest
plus its own configuration. A re-run after a crash reuses the search; a re-run over changed
data does not, and cannot, because the digest is part of the key. ``resume_from=run_id``
adopts a previous run's recipe and **refuses** if that run's dataset digest differs —
resuming onto different data would silently attribute one dataset's search to another's.

**Failure is attributed, not swallowed.** The manifest names the stage, the exception type
and the message. Only *reporting* stages (data profile, SHAP HTML, model-card render) are
marked optional, and an optional failure is recorded as ``degraded`` in the manifest and
appended to ``TrainResult.notes`` — loud, just not fatal.

**The data is assumed to be hard.** Real generated frames carry unobserved confounders,
heteroscedastic noise, MAR missingness, class imbalance and irrelevant columns, and a
held-out R² in the 0.45–0.80 band is the *target*, not a disappointment. Nothing here drops
NaNs behind the caller's back or quietly reindexes; the realism report is surfaced in the
run summary as evidence that the data is honest rather than toy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.contracts.protocols import (
    DriftReport,
    GateDecision,
    Leaderboard,
    Recipe,
    RegistryEntry,
    RunManifest,
    SliceMetric,
    TrainResult,
)
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.pipelines.manifest import (
    CacheSpec,
    SkipStage,
    StageCache,
    StageGraph,
    StageRecord,
    StageSpec,
    finish_manifest,
    new_manifest,
    render_summary,
    write_manifest,
)
from aegis_ml.pipelines.prefect_shim import flow
from aegis_ml.settings import settings

__all__ = [
    "REALISM_ACCURACY_BAND",
    "REALISM_R2_BAND",
    "DataBundle",
    "FrameSourceMissingError",
    "InSampleEvaluationError",
    "ResumeMismatchError",
    "data_flow",
    "drift_flow",
    "eval_flow",
    "forecast_flow",
    "frame_digest",
    "full_flow",
    "promote_flow",
    "realism_band_for",
    "train_flow",
]

SERVE_EXTRA = "aegis-ml[serve]"
"""Install target for pandas/numpy/sklearn/joblib — named verbatim in every ImportError."""

REALISM_R2_BAND: tuple[float, float] = (0.45, 0.80)
"""Held-out R² a *realistic* regression frame should land in.

Below the floor the target is closer to noise than to signal and the conformal interval
will be honestly enormous. **Above the ceiling is equally a defect**: an R² of 0.97 on
generated data means the latent function was sampled with almost no noise, no confounder
and no missingness, so every downstream number — the gate margin, the coverage, the SHAP
story — describes a world that does not exist. A model that looks perfect in the demo and
collapses on the first real frame is the failure this band exists to catch.
"""

REALISM_ACCURACY_BAND: tuple[float, float] = (0.62, 0.92)
"""The same idea for classification: better than a majority-class guess, short of perfect."""


class FrameSourceMissingError(AegisMLError):
    """A flow needs a training frame and was given no way to obtain one."""

    def __init__(self, domain_id: str) -> None:
        """Name every source the flow tried and the flag that supplies one."""
        super().__init__(
            f"No training frame for domain {domain_id!r}. A flow will not invent one: "
            f"fitting on a built-in noise synthesiser and reporting its interval as "
            f"calibrated evidence is the exact failure this package exists to prevent. "
            f"Supply one of: frame=<DataFrame>, source=<path to .csv/.parquet>, "
            f"source=<callable returning a DataFrame>, or register a run whose frozen "
            f"reference frame can be reused (`aegis-ml registry`)."
        )
        self.domain_id = domain_id


class InSampleEvaluationError(AegisMLError):
    """``eval_flow`` was asked to re-score with no fresh data and no explicit opt-in.

    Labelling a misleading number after the fact still leaves it the default, and the
    default is what gets read. Re-scoring on the run's own frozen reference frame measures
    the model on the rows it was fitted on; the result is optimistic by construction and is
    not evidence about unseen data. It is still a useful *integrity* check — the artifact
    loads, deserialises and predicts — which is why it stays reachable behind an explicit
    flag rather than being removed.
    """

    def __init__(self, run_id: str) -> None:
        """Say what the default would have measured, and how to ask for it on purpose."""
        super().__init__(
            f"Re-scoring run {run_id!r} with no fresh frame would measure it on its own "
            f"frozen reference frame — the WHOLE dataset, training rows included. That "
            f"number is optimistic by construction and is not evidence about unseen data, "
            f"so it is not the default. Supply fresh labelled data (frame=<DataFrame>, "
            f"source=<path>, or `--data`), or pass allow_in_sample=True to ask for the "
            f"artifact-loads-and-predicts integrity check on purpose."
        )
        self.run_id = run_id


class ResumeMismatchError(AegisMLError):
    """``resume_from`` pointed at a run fitted on a different dataset."""

    def __init__(self, run_id: str, previous: str | None, current: str) -> None:
        """Report both digests, because the whole point is that they differ."""
        super().__init__(
            f"Cannot resume from run {run_id!r}: it was searched on dataset "
            f"{previous or '<undigested>'} and this run's frame digests to {current}. "
            f"Adopting its recipe would attribute one dataset's model search to another's "
            f"data, and the model card would name a digest the search never saw. Re-run "
            f"the search (drop resume_from) or point resume_from at the matching run."
        )
        self.run_id = run_id


# ───────────────────────────────────────────────────────── cross-module helpers ──


def _pd() -> Any:  # noqa: ANN401 - the pandas module
    """Import pandas, or raise naming the install command."""
    return require(SERVE_EXTRA, "pandas")


def _np() -> Any:  # noqa: ANN401 - the numpy module
    """Import numpy, or raise naming the install command."""
    return require(SERVE_EXTRA, "numpy")


def frame_digest(frame: Any, columns: Sequence[str] | None = None) -> str:  # noqa: ANN401
    """Return a stable ``sha256:`` fingerprint of a dataframe's contents.

    Args:
        frame: The dataframe to fingerprint.
        columns: Restrict to these columns, in this order. Passing the problem's features
            plus its target — which every caller here does — makes the digest independent
            of incidental id or timestamp columns that do not enter the model.

    Returns:
        ``"sha256:<hex>"``.

    This is tamper-**evidence**, matching ``aegis.ml.frame_digest``'s contract exactly:
    a mismatch proves a model was not fitted on the frame you believe it was. It is not
    tamper-prevention — nothing screens a poisoned frame, it is simply fingerprinted on the
    way in, and the column names are folded into the hash so a rename cannot pass unnoticed.
    """
    import hashlib

    pd = _pd()
    subset = frame[list(columns)] if columns else frame
    hashed = pd.util.hash_pandas_object(subset, index=False).to_numpy()
    header = "|".join(map(str, subset.columns)).encode("utf-8")
    return "sha256:" + hashlib.sha256(header + hashed.tobytes()).hexdigest()


def realism_band_for(problem: MLProblem) -> tuple[float, float]:
    """Return the held-out-score band a realistic frame for ``problem`` should land in.

    Args:
        problem: The supervised problem.

    Returns:
        ``(floor, ceiling)`` — :data:`REALISM_R2_BAND` for regression,
        :data:`REALISM_ACCURACY_BAND` for classification.
    """
    # Read through settings, not off the module constants, so `config/contracts.toml` and
    # `AEGIS_ML_REALISM_*` actually reach the band every flow, the doctor line and the
    # realism chart are judged against. The constants above remain the defaults those
    # settings fields fall back to, so behaviour is unchanged when nothing overrides them —
    # but there is now exactly one source of truth instead of two that could drift apart.
    band = (
        settings.realism_r2_band
        if problem.target.task == "regression"
        else settings.realism_accuracy_band
    )
    return (float(band[0]), float(band[1]))


def _resolve_frame(
    problem: MLProblem,
    frame: Any | None,  # noqa: ANN401
    source: str | Path | Callable[[], Any] | None,
) -> tuple[Any, str]:
    """Obtain the training frame, and say where it came from.

    Resolution order, and there is deliberately no fourth step: the caller's frame, an
    explicit ``source`` (path or callable), then the champion run's frozen reference frame.
    When all three are absent the flow raises :class:`FrameSourceMissingError` rather than
    synthesising anything.

    Args:
        problem: The supervised problem (its ``domain_id`` locates the champion).
        frame: A dataframe supplied directly by the caller.
        source: A ``.csv``/``.parquet`` path, or a zero-argument callable returning a frame
            (an adapter's ``training_frame`` bound with its record count, for instance).

    Returns:
        ``(frame, provenance)`` where provenance is recorded on the manifest so a frame read
        off a champion's reference parquet is never mistaken for fresh data.

    Raises:
        FrameSourceMissingError: When no source is available.
        ValueError: For a path whose suffix is not ``.csv``, ``.parquet`` or ``.pq``.
    """
    if frame is not None:
        return frame, "caller"
    if callable(source):
        return source(), f"callable:{getattr(source, '__name__', 'anonymous')}"
    if source is not None:
        path = Path(source)
        pd = _pd()
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path), f"csv:{path}"
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path), f"parquet:{path}"
        raise ValueError(f"unsupported frame source {path.suffix!r}; use .csv or .parquet")

    from aegis_ml.registry import store

    champion = store.champion(problem.domain_id)
    if champion is not None:
        reference = champion.paths.get("reference_frame")
        if reference and Path(reference).exists():
            return _pd().read_parquet(reference), f"champion_reference:{champion.run_id}"
    raise FrameSourceMissingError(problem.domain_id)


# ────────────────────────────────────────────────────────────── the data bundle ──


@dataclass
class DataBundle:
    """Everything :func:`data_flow` established about one dataset.

    Held as a dataclass rather than a pydantic model because three of its fields are
    dataframes. What *is* JSON — the digest, the measured scores, the leakage list — is
    mirrored onto :class:`~aegis_ml.contracts.protocols.TrainResult` and the run manifest,
    so nothing a reader needs is trapped in a live Python object.
    """

    problem: MLProblem
    frame: Any
    train: Any
    calibration: Any
    test: Any
    digest: str
    provenance: str
    contract_ok: bool
    contract_report: Any = None
    leakage: list[Any] = field(default_factory=list)
    learnability: float | None = None
    realism: dict[str, Any] = field(default_factory=dict)
    reference_path: str | None = None
    profile_path: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def sizes(self) -> tuple[int, int, int]:
        """``(training, calibration, test)`` row counts of the three disjoint splits."""
        return len(self.train), len(self.calibration), len(self.test)

    def summary(self) -> dict[str, Any]:
        """Render the JSON-safe half of the bundle, for the manifest and the run summary."""
        train_rows, calib_rows, test_rows = self.sizes
        return {
            "digest": self.digest,
            "provenance": self.provenance,
            "rows": len(self.frame),
            "training_size": train_rows,
            "calibration_size": calib_rows,
            "test_size": test_rows,
            "contract_ok": self.contract_ok,
            "leakage": [str(item) for item in self.leakage],
            "learnability": self.learnability,
            "realism": self.realism,
            "reference_frame": self.reference_path,
            "profile": self.profile_path,
            "notes": list(self.notes),
        }


# ────────────────────────────────────────────────────────────────── data stages ──


def _stage_ingest(
    problem: MLProblem,
    frame: Any | None,  # noqa: ANN401
    source: Any | None,  # noqa: ANN401
    provenance_out: dict[str, str],
) -> Callable[[StageRecord], Any]:
    """Build the ingest stage body, which also validates the declared columns are present.

    Args:
        problem: The supervised problem, whose declared columns must all be present.
        frame: A caller-supplied dataframe, or ``None``.
        source: A path or callable to resolve the frame from.
        provenance_out: A dict the stage writes ``{"provenance": ...}`` into, so the bundle
            can report where the frame came from without re-parsing the manifest's notes.

    Returns:
        The stage body.
    """

    def run(record: StageRecord) -> Any:  # noqa: ANN401
        resolved, provenance = _resolve_frame(problem, frame, source)
        provenance_out["provenance"] = provenance
        record.note(f"frame source: {provenance}")
        record.rows_out = int(len(resolved))
        missing = [c for c in [*problem.feature_names, problem.target.name] if c not in resolved]
        if missing:
            raise ValueError(
                f"frame from {provenance} is missing declared columns {missing}. The spec "
                f"and the data disagree; fix one before anything expensive runs — a "
                f"missing FEATURE_NAMES entry is how aegis.ml.resolve_spec ends up serving "
                f"a four-column noise fallback without raising."
            )
        nulls = {
            col: int(resolved[col].isna().sum())
            for col in problem.feature_names
            if int(resolved[col].isna().sum()) > 0
        }
        if nulls:
            record.note(
                "nulls present (expected — MAR missingness is part of a realistic frame): "
                + ", ".join(f"{k}={v}" for k, v in sorted(nulls.items()))
            )
            record.metric("null_cells", float(sum(nulls.values())))
        record.metric("provenance_is_fresh", 0.0 if "champion_reference" in provenance else 1.0)
        return resolved

    return run


def _run_data_stages(
    graph: StageGraph,
    problem: MLProblem,
    *,
    frame: Any | None,  # noqa: ANN401
    source: Any | None,  # noqa: ANN401
    latent: Any | None,  # noqa: ANN401
    seed: int,
    test_size: float,
    calibration_size: float,
    reference_path: Path | None,
    profile_path: Path | None,
) -> DataBundle:
    """Run ingest → contract → profile → learnability → realism → leakage → split → freeze.

    Args:
        graph: The stage graph recording into the run's manifest.
        problem: The supervised problem.
        frame: A caller-supplied dataframe, or ``None`` to resolve from ``source``.
        source: Path or callable producing the frame.
        latent: The generator's latent-function object, when the caller has one. Passed to
            ``data.latent.realism_report``; without it the realism stage reports what it can
            measure from the frame alone and says so.
        seed: Split seed, so the three-way split reproduces exactly.
        test_size: Fraction held out for measurement.
        calibration_size: Fraction held out for conformal calibration — **disjoint from
            both** the training and test splits, which is what makes the coverage number a
            measurement rather than a re-reading of the training residuals.
        reference_path: Where to freeze the reference frame for later drift comparison.
        profile_path: Where to write the skrub data profile, when that module is present.

    Returns:
        The populated :class:`DataBundle`.
    """
    ctx = graph.context
    provenance: dict[str, str] = {}

    graph.run(
        StageSpec(
            name="ingest",
            description="resolve and admit the training frame",
            outputs=("frame",),
        ),
        _stage_ingest(problem, frame, source, provenance),
    )

    def contract(record: StageRecord) -> Any:  # noqa: ANN401
        from aegis_ml.data import contract_check

        # include_leakage=False: the leakage scan is its own stage below, so it is timed,
        # recorded and re-runnable on its own rather than folded into this one's verdict.
        report = contract_check.check(ctx["frame"], problem, include_leakage=False, seed=seed)
        record.rows_in = int(len(ctx["frame"]))
        record.metric("contract_ok", 1.0 if report.ok else 0.0)
        record.metric("schema_ok", 1.0 if report.schema_ok else 0.0)
        if report.metric_value is not None:
            record.metric(report.metric_name or "learnability", float(report.metric_value))
        for issue in report.issues:
            record.note(f"issue: {issue}")
        for warning in report.warnings:
            record.note(f"warning: {warning}")
        if not report.ok:
            record.note(
                "CONTRACT FAILED — training continues for diagnosis, but the promotion "
                "gate takes contract_ok as an input and will refuse this run."
            )
        return report

    graph.run(
        StageSpec(
            name="contract",
            description="pandera schema, ranges, null policy, categorical levels",
            inputs=("frame",),
            outputs=("contract_report",),
        ),
        contract,
    )

    def profile(record: StageRecord) -> Any:  # noqa: ANN401
        # Imported from the full module path: ``aegis_ml.data.__init__`` re-exports the
        # function ``profile``, which shadows the submodule of the same name on the package.
        from aegis_ml.data.profile import profile_frame

        target = profile_path or (settings.reports_dir / f"{problem.domain_id}_profile.html")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        summary = profile_frame(
            ctx["frame"], out_html=target, title=f"{problem.domain_id} training frame"
        )
        record.rows_in = int(len(ctx["frame"]))
        for key in ("n_rows", "n_columns"):
            if isinstance(summary.get(key), int | float):
                record.metric(key, float(summary[key]))
        record.artifact("profile", target)
        return str(target)

    graph.run(
        StageSpec(
            name="profile",
            description="skrub TableReport — distributions, cardinality, missingness",
            inputs=("frame",),
            outputs=("profile_path",),
            optional=True,
        ),
        profile,
    )

    def learnability(record: StageRecord) -> float:
        from aegis_ml.data import latent as latent_mod

        score = float(latent_mod.assert_learnable(ctx["frame"], problem, seed=seed))
        floor, ceiling = realism_band_for(problem)
        record.metric("held_out_score", score)
        record.metric("realism_floor", floor)
        record.metric("realism_ceiling", ceiling)
        if score > ceiling:
            record.note(
                f"held-out score {score:.3f} is ABOVE the realism ceiling {ceiling:.2f}: the "
                f"generator is sampling the label with too little noise, too few "
                f"confounders or no missingness. Everything downstream will look better "
                f"than it will on real data."
            )
        return score

    graph.run(
        StageSpec(
            name="learnability",
            description="fit a fast probe and assert the label carries signal",
            inputs=("frame",),
            outputs=("learnability",),
        ),
        learnability,
    )

    def realism(record: StageRecord) -> dict[str, Any]:
        from aegis_ml.data import latent as latent_mod

        data = dict(latent_mod.realism_report(ctx["frame"], problem, latent, seed=seed))
        for key, value in data.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                record.metric(key, float(value))
        if latent is None:
            record.note(
                "no latent function supplied — noise-to-signal is measured from the frame "
                "alone, not compared against the coefficients that generated it"
            )
        return data

    graph.run(
        StageSpec(
            name="realism",
            description="achieved score, noise-to-signal, missingness, class balance",
            inputs=("frame",),
            outputs=("realism",),
            optional=True,
        ),
        realism,
    )

    def leakage(record: StageRecord) -> list[Any]:
        from aegis_ml.features import leakage as leakage_mod

        found = list(leakage_mod.detect_leakage(ctx["frame"], problem, seed=seed))
        record.metric("leaking_features", float(len(found)))
        if found:
            record.note(
                "LEAKAGE FLAGGED: "
                + ", ".join(f"{s.feature} ({s.kind}, score {s.score:.3f})" for s in found)
                + " — a single feature that predicts the target this well is not signal, "
                "and the gate refuses a run carrying one."
            )
        return found

    graph.run(
        StageSpec(
            name="leakage",
            description="single-feature target-leakage scan",
            inputs=("frame",),
            outputs=("leakage",),
        ),
        leakage,
    )

    def split(record: StageRecord) -> tuple[Any, Any, Any]:
        from aegis_ml.data import splits

        parts = splits.three_way_split(
            ctx["frame"],
            problem,
            test_size=test_size,
            calibration_size=calibration_size,
            seed=seed,
            confidence_level=problem.requested_coverage,
        )
        train, calibration, test = parts.train, parts.calibration, parts.test
        record.rows_in = int(len(ctx["frame"]))
        record.rows_out = int(len(train))
        record.metric("training_size", float(len(train)))
        record.metric("calibration_size", float(len(calibration)))
        record.metric("test_size", float(len(test)))
        record.note(
            "three disjoint splits: the calibration rows are seen by neither the fit nor "
            "the measurement, which is what makes the coverage number evidence"
        )
        return train, calibration, test

    graph.run(
        StageSpec(
            name="split",
            description="disjoint train / calibration / test split",
            inputs=("frame",),
            outputs=("train", "calibration", "test"),
        ),
        split,
    )

    def digest(record: StageRecord) -> str:
        value = frame_digest(ctx["frame"], [*problem.feature_names, problem.target.name])
        record.note(f"dataset digest {value}")
        return value

    graph.run(
        StageSpec(
            name="digest",
            description="SHA-256 fingerprint of the exact frame this run saw",
            inputs=("frame",),
            outputs=("digest",),
        ),
        digest,
    )

    def freeze(record: StageRecord) -> str:
        if reference_path is None:
            raise SkipStage("no reference path supplied; the frame is not frozen for drift")
        Path(reference_path).parent.mkdir(parents=True, exist_ok=True)
        ctx["frame"].to_parquet(reference_path, index=False)
        record.rows_out = int(len(ctx["frame"]))
        record.artifact("reference_frame", reference_path)
        record.note(
            "the reference frame is stored WITH the run: drift is only meaningful against "
            "the exact distribution this model was calibrated on"
        )
        return str(reference_path)

    graph.run(
        StageSpec(
            name="freeze_reference",
            description="persist the reference frame drift will be measured against",
            inputs=("frame",),
            outputs=("reference_path",),
        ),
        freeze,
    )

    contract_report = ctx.get("contract_report")
    return DataBundle(
        problem=problem,
        frame=ctx["frame"],
        train=ctx["train"],
        calibration=ctx["calibration"],
        test=ctx["test"],
        digest=ctx["digest"],
        provenance=provenance.get("provenance", "unknown"),
        contract_ok=bool(contract_report.ok) if contract_report is not None else False,
        contract_report=contract_report,
        leakage=list(ctx.get("leakage") or []),
        learnability=ctx.get("learnability"),
        realism=dict(ctx.get("realism") or {}),
        reference_path=ctx.get("reference_path"),
        profile_path=ctx.get("profile_path"),
        notes=list(graph.degraded),
    )


# ─────────────────────────────────────────────────────────── measurement helpers ──


def _xy(frame: Any, problem: MLProblem) -> tuple[Any, Any]:  # noqa: ANN401
    """Split a frame into the declared feature columns and the target column."""
    return frame[problem.feature_names], frame[problem.target.name]


def _conformal_quantile_level(n: int, coverage: float) -> float:
    """Return the finite-sample-corrected quantile level for split conformal prediction.

    Args:
        n: Number of calibration residuals.
        coverage: Requested marginal coverage, e.g. ``0.9``.

    Returns:
        ``ceil((n + 1) * coverage) / n``, capped at 1.0.

    The ``(n + 1)`` is not a rounding nicety — it is the correction that makes the coverage
    guarantee hold at *finite* n. Using the plain empirical quantile under-covers by roughly
    ``1/n``, which on a 200-row calibration split is half a percentage point of silently
    missing coverage on every interval the model ever emits.
    """
    if n < 1:
        raise ValueError("split conformal calibration needs at least one residual")
    return min(1.0, math.ceil((n + 1) * coverage) / n)


def _regression_coverage(
    model: Any,  # noqa: ANN401
    calibration: Any,  # noqa: ANN401
    test: Any,  # noqa: ANN401
    problem: MLProblem,
    coverage_level: float,
) -> tuple[float, str]:
    """Measure empirical coverage of a split-conformal band on the held-out test split.

    Fit on train, calibrate on the disjoint calibration split, count on the disjoint test
    split. Every one of those three words is doing work: calibrating on training residuals
    produces a band that is too narrow, and measuring on the calibration split produces a
    coverage figure that is guaranteed by construction and therefore evidence of nothing.

    Args:
        model: The fitted estimator (only ``predict`` is required).
        calibration: The calibration split, never seen in fitting.
        test: The test split, never seen in fitting or calibration.
        problem: The supervised problem.
        coverage_level: The coverage being REQUESTED.

    Returns:
        ``(empirical_coverage, method_note)``.
    """
    from aegis_ml.evaluate import calibration as calib_mod

    np = _np()
    x_cal, y_cal = _xy(calibration, problem)
    residuals = np.abs(
        np.asarray(y_cal, dtype=float) - np.asarray(model.predict(x_cal), dtype=float)
    )
    residuals = residuals[np.isfinite(residuals)]
    level = _conformal_quantile_level(int(residuals.size), coverage_level)
    width = float(np.quantile(residuals, level, method="higher"))

    x_test, y_test = _xy(test, problem)
    point = np.asarray(model.predict(x_test), dtype=float)
    intervals = [(float(p - width), float(p + width)) for p in point]
    measured = float(calib_mod.coverage(list(np.asarray(y_test, dtype=float)), intervals))
    note = (
        f"split conformal on {residuals.size} disjoint calibration residuals; "
        f"half-width {width:.4g} at the {level:.4f} quantile"
    )
    return measured, note


def _classification_coverage(
    model: Any,  # noqa: ANN401
    calibration: Any,  # noqa: ANN401
    test: Any,  # noqa: ANN401
    problem: MLProblem,
    coverage_level: float,
) -> tuple[float, str]:
    """Measure empirical coverage of a conformal prediction *set* for a classifier.

    The score is ``1 - p(true class)`` on the calibration split; the threshold is its
    corrected quantile; the set for a test row is every class scoring at or below it.
    Coverage is the fraction of test rows whose true label is in that set.

    A classification interval does not exist, so the coverage claim attaches to the set. It
    is reported under the same name as the regression one because it answers the same
    question — *how often does the honest answer contain the truth* — and a reader comparing
    a classifier and a regressor across the registry must not have to normalise two
    vocabularies.

    Args:
        model: The fitted classifier; must expose ``predict_proba`` and ``classes_``.
        calibration: The disjoint calibration split.
        test: The disjoint test split.
        problem: The supervised problem.
        coverage_level: The coverage being REQUESTED.

    Returns:
        ``(empirical_coverage, method_note)``.

    Raises:
        AttributeError: If the estimator exposes no ``predict_proba`` — recorded by the
            caller as "coverage not measurable for this estimator" rather than defaulted.
    """
    np = _np()
    classes = list(model.classes_)
    index = {value: i for i, value in enumerate(classes)}

    x_cal, y_cal = _xy(calibration, problem)
    proba_cal = np.asarray(model.predict_proba(x_cal), dtype=float)
    rows = [index.get(value) for value in y_cal]
    scores = np.array(
        [1.0 - proba_cal[i, j] for i, j in enumerate(rows) if j is not None], dtype=float
    )
    level = _conformal_quantile_level(int(scores.size), coverage_level)
    threshold = float(np.quantile(scores, level, method="higher"))

    x_test, y_test = _xy(test, problem)
    proba_test = np.asarray(model.predict_proba(x_test), dtype=float)
    hits = 0
    total = 0
    set_sizes = 0
    for i, truth in enumerate(y_test):
        j = index.get(truth)
        member = {classes[k] for k in range(len(classes)) if 1.0 - proba_test[i, k] <= threshold}
        set_sizes += len(member)
        if j is None:
            continue
        total += 1
        hits += 1 if truth in member else 0
    measured = hits / total if total else 0.0
    note = (
        f"conformal prediction sets from {scores.size} disjoint calibration scores; "
        f"threshold {threshold:.4g}, mean set size {set_sizes / max(1, len(y_test)):.2f}"
    )
    return measured, note


def _with_predictions(frame: Any, model: Any, problem: MLProblem) -> Any:  # noqa: ANN401
    """Return a copy of ``frame`` carrying the model's output in a ``y_pred`` column.

    Args:
        frame: A frame holding at least the declared feature columns.
        model: The fitted estimator.
        problem: The supervised problem.

    Returns:
        A copy with ``y_pred`` added, plus ``y_pred_proba`` for a binary classifier.

    NannyML's CBPE and DLE estimate performance *from the model's own output* on unlabelled
    rows — that is the entire mechanism, and it is why the prediction column is not optional.
    Scoring the frames here rather than assuming a prediction column already exists keeps the
    estimate attributable to the exact artifact this run registered, instead of to whatever
    wrote a column called ``y_pred`` at some earlier point.
    """
    scored = frame.copy()
    features = frame[problem.feature_names]
    scored["y_pred"] = model.predict(features)
    if problem.target.task == "classification" and len(problem.target.levels) == 2:
        proba = getattr(model, "predict_proba", None)
        if proba is not None:
            classes = list(getattr(model, "classes_", problem.target.levels))
            positive = problem.target.levels[-1]
            if positive in classes:
                scored["y_pred_proba"] = proba(features)[:, classes.index(positive)]
    return scored


# ──────────────────────────────────────────────────────────────────── the flows ──


@flow(name="aegis-ml-data")
def data_flow(
    problem: MLProblem,
    frame: Any | None = None,  # noqa: ANN401
    *,
    source: str | Path | Callable[[], Any] | None = None,
    latent: Any | None = None,  # noqa: ANN401
    seed: int | None = None,
    test_size: float = 0.2,
    calibration_size: float = 0.2,
    run_id: str | None = None,
    reference_path: str | Path | None = None,
    profile_path: str | Path | None = None,
    manifest: RunManifest | None = None,
    quiet: bool = False,
) -> DataBundle:
    """Establish that a dataset is fit to train on, and freeze what training will need.

    Runs *before* anything expensive on purpose. An AutoML search over a target that
    carries no signal spends its whole budget discovering that, and reports it as a
    leaderboard of models that all failed equally — which reads like a hard problem rather
    than a broken generator. The learnability probe answers the same question in seconds,
    and the realism report says whether the frame is *too easy*, which is the failure mode
    nobody looks for.

    Args:
        problem: The supervised problem.
        frame: The training frame, or ``None`` to resolve from ``source``.
        source: A ``.csv``/``.parquet`` path or a zero-argument callable returning a frame.
        latent: The generator's latent-function object, if available, for the realism report.
        seed: Split seed; defaults to ``settings.random_seed``.
        test_size: Fraction held out for measurement.
        calibration_size: Fraction held out for conformal calibration.
        run_id: Run id to record on the manifest; one is minted when omitted.
        reference_path: Where to freeze the reference frame. Defaults inside the run dir.
        profile_path: Where to write the data profile.
        manifest: An open manifest to append to. When given, this flow does **not** close or
            write it — the caller owns it, which is how ``full_flow`` produces one manifest
            for the whole bundle.
        quiet: Suppress the console summary table.

    Returns:
        A populated :class:`DataBundle`.

    Raises:
        FrameSourceMissingError: No frame and no way to obtain one.
        LabelNotLearnableError: The target carries no recoverable signal.
        ValueError: The frame is missing columns the spec declares.
    """
    from aegis_ml.registry import store

    owns = manifest is None
    resolved_run_id = run_id or store.new_run_id(problem.domain_id)
    active = manifest or new_manifest(resolved_run_id, "data_flow")
    run_dir = Path(store.run_dir(resolved_run_id))
    graph = StageGraph(active, context={"problem": problem})

    try:
        bundle = _run_data_stages(
            graph,
            problem,
            frame=frame,
            source=source,
            latent=latent,
            seed=settings.random_seed if seed is None else seed,
            test_size=test_size,
            calibration_size=calibration_size,
            reference_path=(
                Path(reference_path) if reference_path else run_dir / "reference.parquet"
            ),
            profile_path=Path(profile_path) if profile_path else None,
        )
    except BaseException as exc:
        if owns:
            finish_manifest(active, error=exc)
            write_manifest(run_dir / "manifest.json", active)
            if not quiet:
                print(render_summary(active))
        raise

    if owns:
        finish_manifest(active)
        write_manifest(run_dir / "manifest.json", active)
        if not quiet:
            print(render_summary(active))
    return bundle


@flow(name="aegis-ml-train")
def train_flow(  # noqa: PLR0915 - one linear pipeline; splitting it would hide the order
    problem: MLProblem,
    frame: Any | None = None,  # noqa: ANN401
    *,
    tiers: Sequence[str] | None = None,
    time_budget: int | None = None,
    seed: int | None = None,
    use_trainer_venv: bool = False,
    do_hpo: bool = True,
    source: str | Path | Callable[[], Any] | None = None,
    latent: Any | None = None,  # noqa: ANN401
    test_size: float = 0.2,
    calibration_size: float = 0.2,
    force: bool = False,
    resume_from: str | None = None,
    run_id: str | None = None,
    manifest: RunManifest | None = None,
    quiet: bool = False,
) -> TrainResult:
    """Search, tune, fit, measure and register one model — every number from a real fit.

    The order is not negotiable. The data stages run first because they are cheap and they
    are what makes the expensive stages meaningful. The split happens before the fit because
    a calibration split carved out afterwards has already been seen. The measurement happens
    on rows neither the fit nor the calibration touched, because that is the only reason to
    believe it.

    Args:
        problem: The supervised problem.
        frame: The training frame, or ``None`` to resolve from ``source``.
        tiers: AutoML tiers to run, e.g. ``("baseline", "flaml")``. ``None`` lets
            :func:`aegis_ml.automl.search.search` decide from what is installed and enabled,
            and every tier it skips is recorded on the leaderboard with the reason.
        time_budget: Search budget in seconds; defaults to ``settings.automl_time_budget``.
        seed: Random state for split, search and fit; defaults to ``settings.random_seed``.
        use_trainer_venv: Run the search in the isolated trainer venv through the subprocess
            bridge. This is the **only** stage with retries, because it is the only one whose
            failure can be transient (a cold venv, a subprocess killed under memory
            pressure). Retrying a deterministic stage just re-derives the same exception.
        do_hpo: Run the Optuna study over the winning recipe.
        source: Frame source when ``frame`` is ``None``.
        latent: Latent-function object for the realism report.
        test_size: Fraction held out for measurement.
        calibration_size: Fraction held out for conformal calibration.
        force: Bypass the stage cache — re-run the search even on an unchanged frame.
        resume_from: Adopt a previous run's recipe and leaderboard instead of searching.
            Refuses if that run's dataset digest differs from this frame's.
        run_id: Run id to register under; one is minted when omitted.
        manifest: An open manifest to append to (the caller then owns closing it).
        quiet: Suppress the console summary table.

    Returns:
        A populated :class:`~aegis_ml.contracts.protocols.TrainResult`: the metric measured
        on the test split, the coverage requested, the coverage achieved, the recipe, the
        full leaderboard including losers, the per-slice metrics and the artifact path.

    Raises:
        LabelNotLearnableError: The target carries no recoverable signal.
        ResumeMismatchError: ``resume_from`` names a run fitted on different data.
        TrainerVenvMissingError: ``use_trainer_venv`` with no venv at ``settings.trainer_venv``.
        FrameSourceMissingError: No frame and no way to obtain one.
    """
    from aegis_ml.registry import store

    owns = manifest is None
    resolved_seed = settings.random_seed if seed is None else seed
    resolved_budget = settings.automl_time_budget if time_budget is None else time_budget
    resolved_run_id = run_id or store.new_run_id(problem.domain_id)
    active = manifest or new_manifest(resolved_run_id, "train_flow")
    run_dir = Path(store.run_dir(resolved_run_id))
    run_dir.mkdir(parents=True, exist_ok=True)

    cache = StageCache(settings.registry_dir / "_cache", flow="train_flow", enabled=not force)
    graph = StageGraph(active, cache=cache, context={"problem": problem})
    ctx = graph.context

    try:
        bundle = _run_data_stages(
            graph,
            problem,
            frame=frame,
            source=source,
            latent=latent,
            seed=resolved_seed,
            test_size=test_size,
            calibration_size=calibration_size,
            reference_path=run_dir / "reference.parquet",
            profile_path=run_dir / "profile.html",
        )

        if resume_from:
            previous = store.load_entry(resume_from)
            if previous.result.dataset_digest != bundle.digest:
                raise ResumeMismatchError(
                    resume_from, previous.result.dataset_digest, bundle.digest
                )
            ctx["recipe"] = previous.result.recipe
            ctx["leaderboard"] = previous.result.leaderboard
            ctx["resume_from"] = resume_from

        search_cache = CacheSpec(
            key=lambda c: [
                c["digest"],
                sorted(tiers) if tiers else None,
                resolved_budget,
                resolved_seed,
                bool(use_trainer_venv),
            ],
            dumps=lambda value: {
                "recipe": value[0].model_dump(mode="json"),
                "leaderboard": value[1].model_dump(mode="json"),
            },
            loads=lambda blob: (
                Recipe.model_validate(blob["recipe"]),
                Leaderboard.model_validate(blob["leaderboard"]),
            ),
        )

        def search(record: StageRecord) -> tuple[Recipe, Leaderboard]:
            record.rows_in = int(len(bundle.train))
            if use_trainer_venv:
                from aegis_ml.automl import runner

                found = runner.run_in_trainer_venv(
                    bundle.train,
                    problem,
                    tiers=tiers,
                    time_budget=resolved_budget,
                    seed=resolved_seed,
                )
                record.note(f"search ran in the isolated trainer venv: {settings.trainer_venv}")
            else:
                from aegis_ml.automl.search import run_search

                found = run_search(
                    bundle.train,
                    problem,
                    tiers=tiers,
                    time_budget=resolved_budget,
                    seed=resolved_seed,
                )
            recipe, board = found[0], found[1]
            record.metric("candidates", float(len(board.candidates)))
            record.note(f"winning tier: {recipe.tier}")
            for tier, why in board.tiers_skipped.items():
                record.note(f"tier {tier} skipped: {why}")
            return recipe, board

        graph.run(
            StageSpec(
                name="search",
                description="AutoML tier ladder; every candidate kept, winners and losers",
                inputs=("train", "digest"),
                outputs=("recipe", "leaderboard"),
                skip_if=lambda c: (
                    f"recipe adopted from run {c['resume_from']}" if c.get("recipe") else None
                ),
                cache=search_cache,
                retries=2 if use_trainer_venv else 0,
                backoff_seconds=3.0,
            ),
            search,
        )

        def hpo(record: StageRecord) -> Recipe:
            from aegis_ml.automl import hpo as hpo_mod

            tuned = hpo_mod.tune(
                bundle.train,
                problem,
                ctx["recipe"],
                n_trials=settings.hpo_trials,
                timeout=settings.hpo_timeout,
                seed=resolved_seed,
            )
            record.metric("n_trials", float(settings.hpo_trials))
            record.note(f"Optuna study over the {ctx['recipe'].tier} recipe")
            return tuned

        graph.run(
            StageSpec(
                name="hpo",
                description="Optuna TPE study over the winning recipe",
                inputs=("recipe", "train"),
                outputs=("recipe",),
                skip_if=lambda c: (
                    None
                    if do_hpo and not c.get("resume_from")
                    else (
                        "resumed recipe is already tuned"
                        if c.get("resume_from")
                        else "do_hpo=False"
                    )
                ),
            ),
            hpo,
        )

        def fit(record: StageRecord) -> Any:  # noqa: ANN401
            from aegis_ml.automl import recipe as recipe_mod

            model = recipe_mod.fit_recipe(
                ctx["recipe"], bundle.train, problem, random_state=resolved_seed
            )
            record.rows_in = int(len(bundle.train))
            record.note(
                "fitted on the TRAINING split only — the calibration and test rows are "
                "held back so the interval and the metric are independent of the fit"
            )
            return model

        graph.run(
            StageSpec(
                name="fit",
                description="fit the portable recipe on the training split",
                inputs=("recipe", "train"),
                outputs=("model",),
            ),
            fit,
        )

        def measure(record: StageRecord) -> dict[str, Any]:
            from aegis_ml.evaluate import metrics as metrics_mod

            model = ctx["model"]
            x_test, y_test = _xy(bundle.test, problem)
            predictions = model.predict(x_test)
            scored = dict(metrics_mod.score(problem, list(y_test), list(predictions)))
            metric_name, metric_value = metrics_mod.primary(problem, scored)
            for key, value in scored.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    record.metric(key, float(value))

            requested = problem.requested_coverage
            empirical: float | None = None
            note: str
            try:
                if problem.target.task == "regression":
                    empirical, note = _regression_coverage(
                        model, bundle.calibration, bundle.test, problem, requested
                    )
                else:
                    empirical, note = _classification_coverage(
                        model, bundle.calibration, bundle.test, problem, requested
                    )
            except AttributeError as exc:
                note = (
                    f"empirical coverage NOT measured: the fitted estimator exposes no "
                    f"probability or interval interface ({exc}). Reported as null rather "
                    f"than defaulted to the requested level."
                )
            record.note(note)
            if empirical is not None:
                record.metric("requested_coverage", requested)
                record.metric("empirical_coverage", empirical)
                if empirical < requested - settings.coverage_tolerance:
                    record.note(
                        f"COVERAGE SHORTFALL: asked for {requested:.2%}, achieved "
                        f"{empirical:.2%} — outside the {settings.coverage_tolerance:.2%} "
                        f"tolerance. This is a finding, not a rounding error."
                    )
            record.rows_in = int(len(bundle.test))
            return {
                "metrics": scored,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "empirical_coverage": empirical,
                "predictions": list(predictions),
                "coverage_note": note,
            }

        graph.run(
            StageSpec(
                name="measure",
                description="score and count coverage on the untouched test split",
                inputs=("model", "test", "calibration"),
                outputs=("measurement",),
            ),
            measure,
        )

        def slices(record: StageRecord) -> list[SliceMetric]:
            from aegis_ml.evaluate import slices as slices_mod

            _, y_test = _xy(bundle.test, problem)
            found = list(
                slices_mod.slice_metrics(
                    bundle.test,
                    list(y_test),
                    ctx["measurement"]["predictions"],
                    problem,
                    min_rows=30,
                )
            )
            record.metric("slices", float(len(found)))
            if found:
                worst = min(found, key=lambda s: s.metric_value)
                record.metric("worst_slice", worst.metric_value)
                record.note(
                    f"worst slice {worst.feature}={worst.level} at {worst.metric_name}="
                    f"{worst.metric_value:.4g} over {worst.n_rows} rows — the gate reads "
                    f"this, not the mean: a model that improves on average while "
                    f"collapsing on one segment is a regression for everyone in it"
                )
            return found

        graph.run(
            StageSpec(
                name="slices",
                description="the primary metric per data segment; the gate reads the worst",
                inputs=("model", "test"),
                outputs=("slices",),
                optional=True,
            ),
            slices,
        )

        def shap(record: StageRecord) -> dict[str, Any]:
            from aegis_ml.explain import shap_report

            importance = shap_report.global_importance(
                ctx["model"], bundle.test, problem, seed=resolved_seed
            )
            target = run_dir / "shap.html"
            shap_report.render_html(
                target,
                importance=importance,
                problem=problem,
                title=f"{problem.domain_id} — global SHAP attribution",
                notes=[
                    "Attributions describe THIS model's behaviour on the held-out split. "
                    "They are not statements about the world, and a driver's sign is not a "
                    "causal direction.",
                ],
            )
            record.metric("features_attributed", float(len(importance)))
            record.artifact("shap", target)
            return {"importance": importance, "path": str(target)}

        graph.run(
            StageSpec(
                name="shap",
                description="global SHAP attribution over the test split",
                inputs=("model", "test"),
                outputs=("shap",),
                optional=True,
            ),
            shap,
        )

        measurement = ctx["measurement"]
        train_rows, calib_rows, test_rows = bundle.sizes
        notes = list(bundle.notes)
        notes.append(measurement["coverage_note"])
        if not bundle.contract_ok:
            notes.append("data contract FAILED — this run cannot be promoted")
        if bundle.leakage:
            notes.append(
                "leakage flagged on: " + ", ".join(str(item) for item in bundle.leakage)
            )
        if bundle.learnability is not None:
            floor, ceiling = realism_band_for(problem)
            notes.append(
                f"learnability probe scored {bundle.learnability:.3f} against the "
                f"realism band [{floor:.2f}, {ceiling:.2f}]"
            )
        notes.extend(graph.degraded)

        result = TrainResult(
            run_id=resolved_run_id,
            domain_id=problem.domain_id,
            task=problem.target.task,
            target=problem.target.name,
            metric_name=str(measurement["metric_name"]),
            metric_value=float(measurement["metric_value"]),
            requested_coverage=problem.requested_coverage,
            empirical_coverage=measurement["empirical_coverage"],
            training_size=train_rows,
            calibration_size=calib_rows,
            test_size=test_rows,
            dataset_digest=bundle.digest,
            recipe=ctx.get("recipe"),
            leaderboard=ctx.get("leaderboard"),
            slices=list(ctx.get("slices") or []),
            artifact_path=str(run_dir / "model.joblib"),
            notes=notes,
        )

        def card(record: StageRecord) -> dict[str, str]:
            from aegis_ml.explain import card as card_mod

            shap_bundle = ctx.get("shap") or {}
            built = card_mod.build_card(
                result,
                leaderboard=ctx.get("leaderboard"),
                slices=list(ctx.get("slices") or []),
                metrics=dict(measurement["metrics"]),
                top_features=list(shap_bundle.get("importance") or []),
                shap_report_path=shap_bundle.get("path"),
                coverage_tolerance=settings.coverage_tolerance,
                target_unit=problem.target.unit,
                target_description=problem.target.description,
                data_source=bundle.provenance,
                created_at=datetime.now(UTC).isoformat(),
                tabpfn_used=bool(
                    ctx.get("recipe") is not None and ctx["recipe"].tier == "tabpfn"
                ),
                notes=result.notes,
            )
            written: dict[str, str] = {}
            for name, renderer, suffix in (
                ("card_md", card_mod.render_markdown, ".md"),
                ("card_html", card_mod.render_html, ".html"),
            ):
                text = renderer(built)
                path = run_dir / f"card{suffix}"
                path.write_text(str(text), encoding="utf-8")
                record.artifact(name, path)
                written[name] = str(path)
            return written

        graph.run(
            StageSpec(
                name="card",
                description="render the model card in Markdown and HTML",
                inputs=("measurement",),
                outputs=("card_paths",),
                optional=True,
            ),
            card,
        )

        def register(record: StageRecord) -> RegistryEntry:
            paths: dict[str, str] = {"manifest": str(run_dir / "manifest.json")}
            if bundle.reference_path:
                paths["reference_frame"] = bundle.reference_path
            if bundle.profile_path:
                paths["profile"] = bundle.profile_path
            if ctx.get("shap"):
                paths["shap"] = ctx["shap"]["path"]
            paths.update(ctx.get("card_paths") or {})

            (run_dir / "problem.json").write_text(
                problem.model_dump_json(indent=2), encoding="utf-8"
            )
            paths["problem"] = str(run_dir / "problem.json")
            # The gate's non-metric inputs are recorded here, at the moment they were
            # measured. Re-deriving them at promotion time would measure a different frame.
            (run_dir / "gate_inputs.json").write_text(
                json.dumps(
                    {
                        "contract_ok": bundle.contract_ok,
                        "leakage": [str(item) for item in bundle.leakage],
                        "learnability": bundle.learnability,
                        "realism": bundle.realism,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            paths["gate_inputs"] = str(run_dir / "gate_inputs.json")
            # recipe.json, leaderboard.json and metrics.json are written by save_run from
            # the entry itself, so there is exactly one writer for each and no way for a
            # hand-written copy to drift from the registered result.
            entry = RegistryEntry(
                run_id=resolved_run_id,
                domain_id=problem.domain_id,
                created_at=datetime.now(UTC).isoformat(),
                stage="staging",
                result=result,
                paths=paths,
            )
            # `artifacts=` is deliberately not passed: every file above already lives in
            # the run directory, and handing save_run their paths would copy each onto
            # itself. `entry.paths` is carried through verbatim instead.
            directory = store.save_run(entry, model=ctx["model"])
            stored = store.load_entry(resolved_run_id)
            record.artifact("run_dir", directory)
            record.metric("artifacts", float(len(stored.paths)))
            record.note(f"registered as staging run {resolved_run_id}")
            return stored

        graph.run(
            StageSpec(
                name="register",
                description="persist model, card, recipe, leaderboard and reference frame",
                inputs=("model", "measurement"),
                outputs=("entry",),
            ),
            register,
        )

        def visuals(record: StageRecord) -> str:
            from aegis_ml.report.bundle import build_bundle

            # The split is a function of these three arguments and of nothing else, and the
            # bundle re-derives it from the frozen reference frame. Recording them means a
            # later `aegis-ml visuals` rebuild reads the split parameters instead of
            # searching for them, and can prove it has the same held-out rows this stage did.
            (run_dir / "split.json").write_text(
                json.dumps(
                    {
                        "seed": resolved_seed,
                        "test_size": test_size,
                        "calibration_size": calibration_size,
                        "training_size": train_rows,
                        "calibration_size_rows": calib_rows,
                        "test_size_rows": test_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            directory = build_bundle(resolved_run_id)
            manifest_path = directory / "manifest.json"
            summary = json.loads(manifest_path.read_text(encoding="utf-8"))
            record.metric("figures_rendered", float(summary["rendered"]))
            record.metric("figures_omitted", float(summary["omitted"]))
            record.artifact("visuals", directory / "index.html")
            for figure in summary["plots"]:
                if figure["status"] != "rendered":
                    record.note(f"{figure['file']} omitted: {figure['reason']}")
            record.note(
                f"{summary['rendered']} figures written to {directory} — open index.html; "
                f"every number on them comes from this run's own artifacts"
            )
            return str(directory)

        graph.run(
            StageSpec(
                name="visuals",
                description="per-run figures and the self-contained index.html",
                inputs=("entry",),
                outputs=("visuals_dir",),
                # Optional for the same reason the SHAP and card stages are: losing a chart
                # must never lose a trained model. The failure is written into the manifest
                # and into the run's notes, so it is loud without being fatal.
                optional=True,
            ),
            visuals,
        )

        entry = ctx.get("entry")
        if entry is not None and entry.result.artifact_path:
            result.artifact_path = entry.result.artifact_path

    except BaseException as exc:
        if owns:
            finish_manifest(active, error=exc)
            write_manifest(run_dir / "manifest.json", active)
            if not quiet:
                print(render_summary(active))
        raise

    if owns:
        finish_manifest(active)
        write_manifest(run_dir / "manifest.json", active)
        if not quiet:
            print(render_summary(active))
    return result


@flow(name="aegis-ml-eval")
def eval_flow(
    run_id: str,
    frame: Any | None = None,  # noqa: ANN401
    *,
    source: str | Path | Callable[[], Any] | None = None,
    allow_in_sample: bool = False,
    manifest: RunManifest | None = None,
    quiet: bool = False,
) -> TrainResult:
    """Re-score a registered run on data it has never seen.

    The registered ``TrainResult`` is a measurement on *one* held-out split. This flow
    answers the different question a week later: does that number survive on the data that
    has arrived since? It loads the persisted model and re-measures — it does not refit, so
    a change in the metric is a change in the world rather than in the seed.

    Args:
        run_id: The registered run to re-score.
        frame: Fresh labelled data. ``None`` re-scores on the run's own frozen reference
            frame — an integrity check that the artifact loads and predicts, and nothing
            more. That frame is the **whole** dataset, including the rows the model was
            fitted on, so the number it produces is expected to be *better* than the
            registered one and is recorded as in-sample rather than presented as evidence.
        source: A path or callable to load the frame from, when ``frame`` is ``None``.
        allow_in_sample: Permit the no-fresh-data path. That frame is the **whole**
            dataset, including the rows the model was fitted on, so the number it produces
            is expected to be *better* than the registered one. It is an integrity check
            that the artifact loads and predicts, and nothing more. Off by default because
            labelling a misleading number after the fact still leaves it the default.
        manifest: An open manifest to append to.
        quiet: Suppress the console summary table.

    Returns:
        A :class:`~aegis_ml.contracts.protocols.TrainResult` carrying the *re-measured*
        metric and coverage, with ``notes`` naming the frame it was measured on. The recipe
        and leaderboard are carried over from the registered run unchanged — they describe a
        search that happened once.

    Raises:
        FileNotFoundError: The run has no persisted model or no stored problem spec.
        InSampleEvaluationError: No frame, no source, and ``allow_in_sample`` is False.
        InsufficientLabelsError: Propagated when the fresh frame carries too few labels.
    """
    from aegis_ml.registry import store

    entry = store.load_entry(run_id)
    run_dir = Path(store.run_dir(run_id))
    owns = manifest is None
    active = manifest or new_manifest(run_id, "eval_flow")
    graph = StageGraph(active)
    ctx = graph.context

    try:
        def load(record: StageRecord) -> tuple[MLProblem, Any]:
            joblib = require(SERVE_EXTRA, "joblib")
            problem_path = Path(entry.paths.get("problem", run_dir / "problem.json"))
            if not problem_path.exists():
                raise FileNotFoundError(
                    f"run {run_id} has no stored problem spec at {problem_path}; it cannot "
                    f"be re-scored without knowing which columns are features"
                )
            problem = MLProblem.model_validate_json(problem_path.read_text(encoding="utf-8"))
            model_path = Path(entry.paths.get("model", run_dir / "model.joblib"))
            if not model_path.exists():
                raise FileNotFoundError(
                    f"run {run_id} has no persisted model at {model_path}; re-scoring "
                    f"requires the exact fitted artifact, not a refit"
                )
            record.note(f"loaded {model_path.name} and problem spec for {problem.domain_id}")
            return problem, joblib.load(model_path)

        graph.run(
            StageSpec(
                name="load_run",
                description="load the persisted model and the problem it was fitted for",
                outputs=("problem", "model"),
            ),
            load,
        )
        problem: MLProblem = ctx["problem"]

        in_sample = False

        def ingest(record: StageRecord) -> Any:  # noqa: ANN401
            nonlocal in_sample
            if frame is not None:
                record.note("re-scoring on caller-supplied fresh data")
                fresh = frame
            elif source is not None:
                fresh, provenance = _resolve_frame(problem, None, source)
                record.note(f"re-scoring on {provenance}")
            else:
                if not allow_in_sample:
                    raise InSampleEvaluationError(run_id)
                reference = entry.paths.get("reference_frame")
                if not reference or not Path(reference).exists():
                    raise FrameSourceMissingError(problem.domain_id)
                record.note(
                    "IN-SAMPLE: re-scoring on the run's own frozen reference frame, which "
                    "contains the rows the model was fitted on. This checks that the "
                    "artifact loads and predicts; the score it produces is optimistic by "
                    "construction and is not evidence about unseen data. Pass a fresh "
                    "labelled frame for that."
                )
                in_sample = True
                fresh = _pd().read_parquet(reference)
            record.rows_out = int(len(fresh))
            return fresh

        graph.run(
            StageSpec(name="ingest", description="load the evaluation frame", outputs=("frame",)),
            ingest,
        )

        def rescore(record: StageRecord) -> dict[str, Any]:
            from aegis_ml.evaluate import metrics as metrics_mod

            fresh = ctx["frame"]
            x, y = _xy(fresh, problem)
            predictions = ctx["model"].predict(x)
            scored = dict(metrics_mod.score(problem, list(y), list(predictions)))
            name, value = metrics_mod.primary(problem, scored)
            for key, metric_value in scored.items():
                if isinstance(metric_value, int | float) and not isinstance(metric_value, bool):
                    record.metric(key, float(metric_value))
            record.rows_in = int(len(fresh))
            delta = float(value) - entry.result.metric_value
            record.metric("delta_vs_registered", delta)
            record.note(
                f"{name}: registered {entry.result.metric_value:.4g} → re-measured "
                f"{float(value):.4g} (delta {delta:+.4g})"
                + (
                    " — IN-SAMPLE, and therefore expected to be higher; this is not a "
                    "generalisation result"
                    if in_sample
                    else ""
                )
            )
            return {
                "metric_name": name,
                "metric_value": float(value),
                "predictions": list(predictions),
                "delta": delta,
            }

        graph.run(
            StageSpec(
                name="rescore",
                description="re-measure the persisted model on the evaluation frame",
                inputs=("model", "frame"),
                outputs=("rescored",),
            ),
            rescore,
        )

        def slices(record: StageRecord) -> list[SliceMetric]:
            from aegis_ml.evaluate import slices as slices_mod

            _, y = _xy(ctx["frame"], problem)
            found = list(
                slices_mod.slice_metrics(
                    ctx["frame"], list(y), ctx["rescored"]["predictions"], problem, min_rows=30
                )
            )
            record.metric("slices", float(len(found)))
            return found

        graph.run(
            StageSpec(
                name="slices",
                description="per-segment metrics on the evaluation frame",
                inputs=("frame",),
                outputs=("slices",),
                optional=True,
            ),
            slices,
        )

        rescored = ctx["rescored"]
        digest = frame_digest(ctx["frame"], [*problem.feature_names, problem.target.name])
        result = entry.result.model_copy(
            update={
                "metric_name": str(rescored["metric_name"]),
                "metric_value": float(rescored["metric_value"]),
                "empirical_coverage": None,
                "dataset_digest": digest,
                "test_size": int(len(ctx["frame"])),
                "slices": list(ctx.get("slices") or []),
                "notes": [
                    *entry.result.notes,
                    f"re-scored on {digest} at {datetime.now(UTC).isoformat()}"
                    + (" (IN-SAMPLE: the run's own reference frame)" if in_sample else ""),
                    f"delta vs registered: {rescored['delta']:+.4g}",
                    "empirical_coverage is null here on purpose: coverage is a property of "
                    "a calibration split, and a re-score has none. Use drift_flow for the "
                    "label-free performance estimate.",
                    *graph.degraded,
                ],
            }
        )
    except BaseException as exc:
        if owns:
            finish_manifest(active, error=exc)
            write_manifest(run_dir / "manifest_eval.json", active)
            if not quiet:
                print(render_summary(active))
        raise

    if owns:
        finish_manifest(active)
        write_manifest(run_dir / "manifest_eval.json", active)
        if not quiet:
            print(render_summary(active))
    return result


@flow(name="aegis-ml-promote")
def promote_flow(
    run_id: str,
    *,
    force: bool = False,
    manifest: RunManifest | None = None,
    quiet: bool = False,
) -> GateDecision:
    """Judge a challenger against the reigning champion and promote only if it wins.

    Promotion is the one irreversible-looking act in the pipeline — it replaces the artifact
    the platform serves — so the decision is a typed object carrying every number behind it,
    populated on a pass as well as on a failure. "Promoted" with no figures is exactly as
    opaque as "rejected" with no figures, and the model card quotes both.

    ``force`` promotes despite a failed gate. It does not fabricate the decision: the
    :class:`~aegis_ml.contracts.protocols.GateDecision` still records ``promoted`` as the
    gate computed it, every failed check keeps its reason, and the override is written into
    the reasons list. An operator reading the registry afterwards can see that a human
    overrode a refusal, which is the only thing that makes an override acceptable.

    Args:
        run_id: The challenger run.
        force: Promote even though the gate refused, recording the override.
        manifest: An open manifest to append to.
        quiet: Suppress the console summary table.

    Returns:
        The :class:`~aegis_ml.contracts.protocols.GateDecision`.

    Raises:
        PromotionRejectedError: Never raised here — the decision is returned so the caller
            can render it. The CLI turns a refusal into a non-zero exit code.
    """
    from aegis_ml.evaluate import gate as gate_mod
    from aegis_ml.registry import promote as promote_mod
    from aegis_ml.registry import store

    entry = store.load_entry(run_id)
    run_dir = Path(store.run_dir(run_id))
    owns = manifest is None
    active = manifest or new_manifest(run_id, "promote_flow")
    graph = StageGraph(active)
    ctx = graph.context

    try:
        def judge(record: StageRecord) -> GateDecision:
            champion = store.champion(entry.domain_id)
            gate_inputs: dict[str, Any] = {}
            gate_path = Path(entry.paths.get("gate_inputs", run_dir / "gate_inputs.json"))
            if gate_path.exists():
                gate_inputs = json.loads(gate_path.read_text(encoding="utf-8"))
            else:
                record.note(
                    f"no gate_inputs.json for run {run_id}: contract and leakage status are "
                    f"unknown, so both are treated as UNPROVEN (contract_ok=False). A gate "
                    f"input that was never recorded is not a passing one."
                )
            decision = gate_mod.evaluate_gate(
                entry.result,
                champion.result if champion is not None else None,
                contract_ok=bool(gate_inputs.get("contract_ok", False)),
                leakage=list(gate_inputs.get("leakage", [])),
            )
            record.metric("promoted", 1.0 if decision.promoted else 0.0)
            for key, value in decision.metrics.items():
                record.metric(key, float(value))
            for reason in decision.reasons:
                record.note(reason)
            if champion is None:
                record.note("no reigning champion: this run is judged on its absolute floors")
            return decision

        graph.run(
            StageSpec(
                name="gate",
                description="challenger vs champion on metric, coverage, contract, slices, leakage",
                outputs=("decision",),
            ),
            judge,
        )

        def apply(record: StageRecord) -> Any:  # noqa: ANN401
            decision: GateDecision = ctx["decision"]
            if not decision.promoted and not force:
                raise SkipStage(
                    "gate refused: " + "; ".join(decision.reasons[:3] or ["no reason recorded"])
                )
            if not decision.promoted and force:
                record.note(
                    "FORCED: a human overrode a failed gate. Every failed check above still "
                    "stands and is recorded on the decision."
                )
            outcome = promote_mod.promote(run_id, decision=decision, force=force)
            record.note(f"artifact replaced at {settings.artifact_path}")
            record.artifact("artifact", settings.artifact_path)
            return outcome

        graph.run(
            StageSpec(
                name="apply",
                description="atomically replace the served artifact and archive the previous",
                inputs=("decision",),
                outputs=("promotion",),
            ),
            apply,
        )

        decision = ctx["decision"]
        if force and not decision.promoted:
            decision = decision.model_copy(
                update={
                    "reasons": [
                        *decision.reasons,
                        "OVERRIDE: promoted with force=True despite the failures above.",
                    ]
                }
            )
    except BaseException as exc:
        if owns:
            finish_manifest(active, error=exc)
            write_manifest(run_dir / "manifest_promote.json", active)
            if not quiet:
                print(render_summary(active))
        raise

    if owns:
        finish_manifest(active)
        write_manifest(run_dir / "manifest_promote.json", active)
        if not quiet:
            print(render_summary(active))
    return decision


@flow(name="aegis-ml-drift")
def drift_flow(
    run_id: str,
    current_frame: Any,  # noqa: ANN401
    *,
    manifest: RunManifest | None = None,
    quiet: bool = False,
) -> DriftReport:
    """Measure distribution drift against the frozen reference, and estimate performance.

    Two measurements that answer different questions, which is why both run:

    * **Evidently** compares the live frame against the exact frame the model was calibrated
      on, and says which features moved. It needs no labels, but it also cannot tell you
      whether the movement *hurt*.
    * **NannyML** (CBPE for classification, DLE for regression) estimates the metric from
      the model's own confidence under the observed covariate shift — performance *before*
      ground truth arrives. Every field carrying it is named ``estimated_*`` so it is never
      read as a measurement.

    A drifted model is **not** withdrawn. Aegis serves the model it has and flags it; what
    drift blocks is the *promotion* of anything calibrated on a reference that no longer
    describes the world.

    Args:
        run_id: The registered run whose reference frame is the comparison baseline.
        current_frame: The live frame to compare.
        manifest: An open manifest to append to.
        quiet: Suppress the console summary table.

    Returns:
        A populated :class:`~aegis_ml.contracts.protocols.DriftReport`.

    Raises:
        FileNotFoundError: The run has no frozen reference frame to compare against.
    """
    from aegis_ml.registry import store

    entry = store.load_entry(run_id)
    run_dir = Path(store.run_dir(run_id))
    owns = manifest is None
    active = manifest or new_manifest(run_id, "drift_flow")
    graph = StageGraph(active)
    ctx = graph.context

    try:
        def load(record: StageRecord) -> tuple[MLProblem, Any, Any]:
            reference_path = entry.paths.get("reference_frame")
            if not reference_path or not Path(reference_path).exists():
                raise FileNotFoundError(
                    f"run {run_id} froze no reference frame, so there is nothing to measure "
                    f"drift AGAINST. Drift is a comparison; without the baseline the only "
                    f"honest answer is that it cannot be computed."
                )
            problem_path = Path(entry.paths.get("problem", run_dir / "problem.json"))
            problem = MLProblem.model_validate_json(problem_path.read_text(encoding="utf-8"))
            reference = _pd().read_parquet(reference_path)
            model = None
            model_path = Path(entry.paths.get("model", run_dir / "model.joblib"))
            if model_path.exists():
                model = require(SERVE_EXTRA, "joblib").load(model_path)
            else:
                record.note(
                    f"no persisted model at {model_path}: distribution drift is still "
                    f"measurable, but the label-free performance estimate is not — it is a "
                    f"function of the model's output and there is no model to ask."
                )
            record.rows_out = int(len(reference))
            record.note(f"reference frame: {reference_path} ({len(reference)} rows)")
            return problem, reference, model

        graph.run(
            StageSpec(
                name="load_reference",
                description="load the frame this model was calibrated on, and the model",
                outputs=("problem", "reference", "model"),
            ),
            load,
        )
        problem: MLProblem = ctx["problem"]

        def measure(record: StageRecord) -> DriftReport:
            from aegis_ml.monitor import drift as drift_mod

            html_out = settings.reports_dir / f"{run_id}_drift.html"
            html_out.parent.mkdir(parents=True, exist_ok=True)
            report = drift_mod.drift_report(
                ctx["reference"], current_frame, problem, run_id=run_id, html_out=html_out
            )
            record.rows_in = int(len(current_frame))
            record.metric("drifted_share", report.drifted_share)
            record.metric("dataset_drift", 1.0 if report.dataset_drift else 0.0)
            if report.drifted_features:
                record.note("drifted features: " + ", ".join(report.drifted_features))
            return report

        graph.run(
            StageSpec(
                name="drift",
                description="Evidently data / target / prediction drift vs the reference",
                inputs=("reference",),
                outputs=("report",),
            ),
            measure,
        )

        def estimate(record: StageRecord) -> dict[str, Any]:
            from aegis_ml.monitor import perf

            model = ctx.get("model")
            if model is None:
                raise SkipStage(
                    "no persisted model: CBPE/DLE estimate performance FROM the model's "
                    "own output, so there is nothing to estimate from"
                )
            estimated = dict(
                perf.estimate_performance(
                    _with_predictions(ctx["reference"], model, problem),
                    _with_predictions(current_frame, model, problem),
                    problem,
                    run_id=run_id,
                )
            )
            for key, value in estimated.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    record.metric(key, float(value))
            record.note(
                "NannyML estimates performance WITHOUT ground truth. Every number here is "
                "an ESTIMATE and is named as one; it is not a measurement and must not be "
                "compared against a measured metric as though it were."
            )
            return estimated

        graph.run(
            StageSpec(
                name="estimate_performance",
                description="NannyML CBPE/DLE label-free performance estimate",
                inputs=("reference",),
                outputs=("estimated",),
                optional=True,
            ),
            estimate,
        )

        report: DriftReport = ctx["report"]
        estimated = ctx.get("estimated") or {}
        updates: dict[str, Any] = {}
        if report.estimated_metric_name is None and estimated:
            name = str(estimated.get("estimated_metric_name") or "")
            value = estimated.get("estimated_metric_value")
            if name and isinstance(value, int | float):
                updates["estimated_metric_name"] = name
                updates["estimated_metric_value"] = float(value)
        if report.verdict == "pass":
            if report.drifted_share >= settings.drift_share_block:
                updates["verdict"] = "block"
            elif report.drifted_share >= settings.drift_share_warn:
                updates["verdict"] = "warn"
        if updates:
            report = report.model_copy(update=updates)

        def alert(record: StageRecord) -> Any:  # noqa: ANN401
            from aegis_ml.monitor import alerts as alerts_mod

            raised = list(alerts_mod.evaluate_alerts(report))
            record.metric("alerts", float(len(raised)))
            for item in raised:
                record.note(f"[{item.level}] {item.code}: {item.message}")
            record.note(
                f"verdict {report.verdict} — recorded, not acted on: the served model is "
                f"NOT withdrawn on drift. What this blocks is the promotion of anything "
                f"calibrated on a reference that no longer describes the world."
            )
            return [item.model_dump(mode="json") for item in raised]

        graph.run(
            StageSpec(
                name="alerts",
                description="dispatch the drift verdict to whatever is listening",
                inputs=("report",),
                outputs=("alerts",),
                optional=True,
            ),
            alert,
        )

        (run_dir / "drift.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

        def visuals(record: StageRecord) -> str:
            from aegis_ml.report.bundle import build_bundle

            # The live frame is persisted next to the reference it was compared against.
            # Without it the drift figure could only be redrawn from the verdict — which
            # says *that* seven features moved and not *how*, and the shape of the move is
            # what separates a hot season from a broken feed.
            current_path = run_dir / "current.parquet"
            _pd()  # fail here, naming the install, rather than inside to_parquet
            current_frame.to_parquet(current_path, index=False)
            record.artifact("current_frame", current_path)
            directory = build_bundle(run_id, current_frame=current_frame)
            summary = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            record.metric("figures_rendered", float(summary["rendered"]))
            record.metric("figures_omitted", float(summary["omitted"]))
            record.artifact("visuals", directory / "index.html")
            record.note(
                f"{summary['rendered']} figures rebuilt at {directory}, now including the "
                f"reference-vs-current overlay for the features this run flagged"
            )
            return str(directory)

        graph.run(
            StageSpec(
                name="visuals",
                description="rebuild the run's figures, now with the drift overlay",
                inputs=("report",),
                outputs=("visuals_dir",),
                # Optional: a drift verdict that was measured is not lost because a chart
                # of it could not be drawn. The failure is recorded in this manifest.
                optional=True,
            ),
            visuals,
        )
    except BaseException as exc:
        if owns:
            finish_manifest(active, error=exc)
            write_manifest(run_dir / "manifest_drift.json", active)
            if not quiet:
                print(render_summary(active))
        raise

    if owns:
        finish_manifest(active)
        write_manifest(run_dir / "manifest_drift.json", active)
        if not quiet:
            print(render_summary(active))
    return report


@flow(name="aegis-ml-forecast")
def forecast_flow(
    points: Sequence[tuple[datetime, float]],
    *,
    label: str,
    unit: str | None = None,
    horizon: int = 14,
    freq: str | None = None,
    level: float | None = None,
    data_source: str = "adapter",
    include_ml_candidates: bool = False,
    run_id: str | None = None,
    manifest: RunManifest | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Forecast one series and record the measurement alongside it.

    Args:
        points: Observed history as ``(timestamp, value)`` pairs, in any order.
        label: Human label for the series, e.g. ``"Shipments dispatched per day"``.
        unit: Unit of the values.
        horizon: Steps to forecast beyond the last observation.
        freq: Frequency alias; inferred from the observed spacing when ``None``.
        level: Coverage level to REQUEST; defaults to ``settings.requested_coverage``.
        data_source: Provenance tag recorded on the result.
        include_ml_candidates: Also score the ``mlforecast`` roster on the same windows.
        run_id: Run id for the manifest; derived from the label when omitted.
        manifest: An open manifest to append to.
        quiet: Suppress the console summary table.

    Returns:
        ``{"forecast": ForecastRun-as-dict, "backtest": BacktestSummary-as-dict}`` — JSON
        throughout so the CLI, the registry and the serving router all consume one shape.

    Raises:
        InsufficientHistoryError: Too little history to fit, calibrate and backtest.
        DegenerateSeriesError: A flat series; 100% coverage from a zero-width band is not a
            measurement, and saying "no variation recorded" is the honest answer.
        ForecastFitError: Every candidate failed; there is no naive-line fallback.
    """
    from aegis_ml.forecast.backtest import summarise
    from aegis_ml.forecast.engine import forecast as forecast_series

    slug = "".join(ch if ch.isalnum() else "-" for ch in label.lower()).strip("-")
    resolved_run_id = run_id or f"forecast-{slug}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    owns = manifest is None
    active = manifest or new_manifest(resolved_run_id, "forecast_flow")
    graph = StageGraph(active)
    ctx = graph.context
    out_dir = settings.reports_dir / "forecasts"

    try:
        def run(record: StageRecord) -> Any:  # noqa: ANN401
            result = forecast_series(
                points,
                label,
                horizon=horizon,
                data_source=data_source,
                freq=freq,
                unit=unit,
                level=level,
                include_ml_candidates=include_ml_candidates,
            )
            record.rows_in = result.history_points
            record.rows_out = len(result.points)
            record.metric("smape", result.smape)
            record.metric("requested_coverage", result.requested_coverage)
            record.metric("empirical_coverage", result.empirical_coverage)
            record.note(
                f"selected {result.model} on measured sMAPE; "
                f"{result.interval_method_detail}"
            )
            if not result.coverage_meets_request:
                record.note(
                    f"COVERAGE SHORTFALL: asked for {result.requested_coverage:.0%}, the "
                    f"band achieved {result.empirical_coverage:.1%} on held-out windows"
                )
            return result

        graph.run(
            StageSpec(
                name="forecast",
                description="rolling-origin selection, conformal band, refit on full history",
                outputs=("run",),
            ),
            run,
        )

        def measure(record: StageRecord) -> Any:  # noqa: ANN401
            summary = summarise(ctx["run"])
            margin = summary.baseline_margin
            if margin is not None:
                record.metric("baseline_margin_smape", margin)
                record.note(
                    f"winner's sMAPE margin over the next-best candidate: {margin:+.3f} "
                    f"(negative is better) — the number behind 'it beat the baseline'"
                )
            return summary

        graph.run(
            StageSpec(
                name="rank",
                description="rank every candidate and quantify the winner's margin",
                inputs=("run",),
                outputs=("summary",),
            ),
            measure,
        )

        payload = {
            "forecast": ctx["run"].model_dump(mode="json"),
            "backtest": ctx["summary"].model_dump(mode="json"),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{resolved_run_id}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    except BaseException as exc:
        if owns:
            finish_manifest(active, error=exc)
            write_manifest(out_dir / f"{resolved_run_id}_manifest.json", active)
            if not quiet:
                print(render_summary(active))
        raise

    if owns:
        finish_manifest(active)
        write_manifest(out_dir / f"{resolved_run_id}_manifest.json", active)
        if not quiet:
            print(render_summary(active))
    return payload


def _stress_shift(frame: Any, problem: MLProblem, *, magnitude: float, seed: int) -> Any:  # noqa: ANN401
    """Return a deliberately shifted copy of ``frame``, for demonstrating drift detection.

    Three real, documented distortions — nothing is fabricated, the rows are the caller's
    own data pushed around:

    1. **Covariate shift**: every numeric feature is scaled by ``1 + magnitude`` and given a
       ``magnitude`` standard-deviation offset, so the marginal distributions genuinely move.
    2. **Concept-adjacent shift**: the frame is resampled with a probability that increases
       with the first numeric feature's rank, so the *joint* distribution moves too, not just
       the margins. A drift detector that only sees rescaled margins is easy to satisfy.
    3. **Categorical re-weighting**: the rarest level of each categorical feature is
       upsampled, which is what an operational change actually looks like.

    Args:
        frame: The reference frame to distort.
        problem: The supervised problem, naming which columns are numeric or categorical.
        magnitude: Shift size; ``0.25`` is a clearly-detectable but not absurd move.
        seed: Random state, so the demonstration reproduces exactly.

    Returns:
        A shifted copy.

    This exists so ``full_flow`` can show the drift path firing on data whose shift is
    *known*. It is labelled ``synthetic_stress_shift`` everywhere it appears, and it is
    never used as, or compared against, live data.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    shifted = frame.copy()

    numeric = [
        column
        for column in problem.numeric_features
        if column in shifted and str(shifted[column].dtype).startswith(("int", "float"))
    ]
    for column in numeric:
        series = shifted[column].astype(float)
        spread = float(series.std(skipna=True) or 0.0)
        shifted[column] = series * (1.0 + magnitude) + magnitude * spread

    if numeric:
        ranks = shifted[numeric[0]].astype(float).rank(pct=True).fillna(0.5).to_numpy()
        weights = 0.25 + 1.5 * ranks
        keep = rng.random(len(shifted)) < (weights / weights.max())
        if keep.sum() >= max(20, len(shifted) // 5):
            shifted = shifted[keep]

    for column in problem.categorical_features:
        if column not in shifted:
            continue
        counts = shifted[column].value_counts()
        if len(counts) < 2:
            continue
        rarest = counts.index[-1]
        extra = shifted[shifted[column] == rarest]
        if len(extra) and len(extra) < len(shifted) // 2:
            shifted = _pd().concat([shifted, extra, extra], ignore_index=True)

    return shifted.reset_index(drop=True)


def _render_run_summary(
    *,
    problem: MLProblem,
    result: TrainResult,
    decision: GateDecision,
    drift: DriftReport | None,
    bundle_paths: Mapping[str, str],
    manifest: RunManifest,
) -> str:
    """Render the one-page ``RUN_SUMMARY.md`` a demo is actually read from.

    Every figure on the page is copied from a measured field; nothing is recomputed here,
    so the summary and the registry can never disagree. Where a number is missing it says
    "not measured" and why, because a blank cell reads as a zero.

    Args:
        problem: The supervised problem.
        result: The training result.
        decision: The gate decision.
        drift: The drift report, when the drift stage ran.
        bundle_paths: Role → path for every artifact in the bundle.
        manifest: The run's manifest, rendered as the stage table.

    Returns:
        Markdown.
    """
    floor, ceiling = realism_band_for(problem)
    coverage = (
        f"{result.empirical_coverage:.2%}"
        if result.empirical_coverage is not None
        else "not measured (see notes)"
    )
    gap = (
        f"{result.requested_coverage - result.empirical_coverage:+.2%}"
        if result.empirical_coverage is not None
        else "—"
    )
    lines = [
        f"# {problem.domain_id} — run `{result.run_id}`",
        "",
        f"Generated {datetime.now(UTC).isoformat()} by `full_flow`.",
        "",
        "## Verdict",
        "",
        f"- **Promoted:** {'yes' if decision.promoted else 'NO'}",
        *[f"  - {reason}" for reason in decision.reasons],
        "",
        "## What was measured",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Task | {result.task} on `{result.target}` |",
        f"| Primary metric | **{result.metric_name} = {result.metric_value:.4g}** "
        f"(held-out test split) |",
        f"| Requested coverage | {result.requested_coverage:.0%} |",
        f"| Empirical coverage | {coverage} |",
        f"| Coverage gap (requested − achieved) | {gap} |",
        f"| Split sizes (train / calib / test) | {result.training_size} / "
        f"{result.calibration_size} / {result.test_size} |",
        f"| Dataset digest | `{result.dataset_digest}` |",
        f"| Winning tier | {result.recipe.tier if result.recipe else 'n/a'} |",
        f"| Realism band for this task | [{floor:.2f}, {ceiling:.2f}] |",
        "",
    ]

    if result.leaderboard and result.leaderboard.candidates:
        lines += [
            "## Leaderboard (losers kept — the margin is the point)",
            "",
            f"| Model | Tier | {result.leaderboard.metric_name} | Portable | Selected |",
            "| --- | --- | --- | --- | --- |",
        ]
        for candidate in result.leaderboard.candidates[:12]:
            lines.append(
                f"| {candidate.name} | {candidate.tier} | {candidate.metric_value:.4g} | "
                f"{'yes' if candidate.portable else 'NO'} | "
                f"{'**yes**' if candidate.selected else ''} |"
            )
        for tier, why in result.leaderboard.tiers_skipped.items():
            lines.append(f"| _{tier}_ | skipped | — | — | {why} |")
        lines.append("")

    if result.slices:
        worst = min(result.slices, key=lambda s: s.metric_value)
        lines += [
            "## Worst slice",
            "",
            f"`{worst.feature} = {worst.level}` scored {worst.metric_name}="
            f"{worst.metric_value:.4g} over {worst.n_rows} rows. The gate reads this, not "
            f"the mean: a model that improves on average while collapsing on one segment "
            f"is a regression for everyone in that segment.",
            "",
        ]

    if drift is not None:
        estimated = (
            f"{drift.estimated_metric_name} ≈ {drift.estimated_metric_value:.4g} "
            f"(ESTIMATE, no ground truth)"
            if drift.estimated_metric_name and drift.estimated_metric_value is not None
            else "not estimated"
        )
        lines += [
            "## Drift",
            "",
            f"- Verdict: **{drift.verdict}**",
            f"- Drifted share: {drift.drifted_share:.1%} "
            f"({len(drift.drifted_features)} feature(s): "
            f"{', '.join(drift.drifted_features) or 'none'})",
            f"- Label-free performance: {estimated}",
            "",
            "> A drifted model is **not** withdrawn. Aegis serves the model it has and "
            "flags it; drift blocks the *promotion* of anything calibrated on a reference "
            "that no longer describes the world.",
            "",
        ]

    if result.notes:
        lines += ["## Notes", "", *[f"- {note}" for note in result.notes], ""]

    lines += ["## Artifacts", ""]
    lines += [f"- `{role}`: `{path}`" for role, path in sorted(bundle_paths.items())]
    lines += ["", "## Stages", "", "```", render_summary(manifest), "```", ""]
    return "\n".join(lines)


@flow(name="aegis-ml-full")
def full_flow(
    problem: MLProblem,
    frame: Any | None = None,  # noqa: ANN401
    *,
    current_frame: Any | None = None,  # noqa: ANN401
    stress_drift: bool = True,
    stress_magnitude: float = 0.25,
    promote: bool = True,
    force: bool = False,
    quiet: bool = False,
    **train_kwargs: Any,  # noqa: ANN401 - forwarded verbatim to train_flow
) -> dict[str, Any]:
    """Train → promote → drift, under **one** manifest, producing the demo bundle.

    The bundle is what gets shown on the day: a trained and (if it earns it) promoted model,
    the model card in Markdown and HTML, the SHAP report, the leaderboard with its losers,
    a drift report, and ``RUN_SUMMARY.md`` — one page that a person can read in a minute and
    that contains no number the pipeline did not measure.

    Args:
        problem: The supervised problem.
        frame: The training frame, or ``None`` to resolve from ``source`` in ``train_kwargs``.
        current_frame: Live data to measure drift against. When ``None`` and ``stress_drift``
            is set, a deliberately shifted copy of the reference frame is used instead —
            labelled ``synthetic_stress_shift`` in the manifest and in the summary, because
            a drift number computed against data we distorted ourselves is a demonstration
            of the detector, not evidence about the world.
        stress_drift: Whether to synthesise that shifted frame when none is supplied.
        stress_magnitude: How hard to push the synthetic shift.
        promote: Run the promotion gate. ``False`` trains and reports without touching the
            served artifact.
        force: Promote despite a failed gate, recording the override on the decision.
        quiet: Suppress the console summary table.
        **train_kwargs: Forwarded to :func:`train_flow` (``tiers``, ``time_budget``,
            ``seed``, ``use_trainer_venv``, ``do_hpo``, ``source``, ``latent``, ``force``,
            ``resume_from``, ...).

    Returns:
        ``{"run_id", "result", "decision", "drift", "paths", "summary_path", "manifest"}``
        — every value JSON-safe.

    Raises:
        Exception: Anything the underlying flows raise. The manifest is closed and written
            first, so a failed bundle still leaves a readable lineage record naming the
            stage that failed.
    """
    from aegis_ml.registry import store

    run_id = train_kwargs.pop("run_id", None) or store.new_run_id(problem.domain_id)
    manifest = new_manifest(run_id, "full_flow")
    run_dir = Path(store.run_dir(run_id))
    run_dir.mkdir(parents=True, exist_ok=True)

    result: TrainResult | None = None
    decision: GateDecision | None = None
    drift: DriftReport | None = None

    try:
        result = train_flow(
            problem,
            frame,
            run_id=run_id,
            manifest=manifest,
            quiet=True,
            **train_kwargs,
        )

        if promote:
            decision = promote_flow(run_id, force=force, manifest=manifest, quiet=True)
        else:
            decision = GateDecision(
                promoted=False,
                challenger_run_id=run_id,
                reasons=["promotion not attempted: full_flow(promote=False)"],
            )

        entry = store.load_entry(run_id)
        reference_path = entry.paths.get("reference_frame")
        comparison = current_frame
        if comparison is None and stress_drift and reference_path:
            reference = _pd().read_parquet(reference_path)
            comparison = _stress_shift(
                reference,
                problem,
                magnitude=stress_magnitude,
                seed=int(train_kwargs.get("seed") or settings.random_seed),
            )
            manifest.stages.append(
                {
                    "stage": "synthetic_stress_shift",
                    "status": "ok",
                    "started_at": datetime.now(UTC).isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": 0.0,
                    "rows_in": int(len(reference)),
                    "rows_out": int(len(comparison)),
                    "notes": [
                        f"reference frame distorted by magnitude={stress_magnitude} to "
                        f"exercise the drift detector. This is a DEMONSTRATION frame: it "
                        f"is data we shifted ourselves, never live evidence.",
                    ],
                }
            )
        if comparison is not None:
            drift = drift_flow(run_id, comparison, manifest=manifest, quiet=True)

        entry = store.load_entry(run_id)
        paths = dict(entry.paths)
        paths["manifest"] = str(run_dir / "manifest.json")
        summary = _render_run_summary(
            problem=problem,
            result=result,
            decision=decision,
            drift=drift,
            bundle_paths=paths,
            manifest=manifest,
        )
        summary_path = run_dir / "RUN_SUMMARY.md"
        summary_path.write_text(summary, encoding="utf-8")
        paths["run_summary"] = str(summary_path)
    except BaseException as exc:
        finish_manifest(manifest, error=exc)
        write_manifest(run_dir / "manifest.json", manifest)
        if not quiet:
            print(render_summary(manifest))
        raise

    finish_manifest(manifest)
    write_manifest(run_dir / "manifest.json", manifest)
    if not quiet:
        print(render_summary(manifest))
        print(f"\nbundle: {run_dir}\nsummary: {summary_path}")

    return {
        "run_id": run_id,
        "result": result.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "drift": drift.model_dump(mode="json") if drift is not None else None,
        "paths": paths,
        "summary_path": str(summary_path),
        "manifest": manifest.model_dump(mode="json"),
    }
