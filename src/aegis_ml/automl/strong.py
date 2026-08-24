"""Keep the strong models and serve them, instead of reporting a number nobody can call.

Until this module existed, an AutoGluon stack or a TabPFN fit was trained, scored on the
shared holdout, written onto the leaderboard as an *accuracy ceiling* — and then deleted
with its temporary directory. The ceiling was an assertion: "AutoGluon scored 0.79", with
no artifact left behind that anyone could re-run to check. That is the flaw this module
fixes. The strong model is persisted into the run directory, and it can be called again
through the same process boundary that fitted it.

**Why a process boundary and not an import.** The two venvs are pinned differently on
purpose (decision D1): AutoGluon 1.6 + TabPFN 8.4 + torch 2.10 will not resolve under the
Aegis backend's ``pandas<2.4`` / ``numpy<2.5`` / ``numba==0.67.0`` caps. So the model is
loaded and called by ``settings.trainer_python``, never by this interpreter. What crosses
is a parquet frame in and a parquet prediction column out — never a pickle, in either
direction, for exactly the reason :mod:`aegis_ml.automl.runner` gives.

**Why that is sound rather than hopeful.** ``pandas``, ``numpy`` and ``scikit-learn``
resolve to the *same versions* in both venvs (2.3.3 / 2.4.6 / 1.9.0 — see
``RESOLUTION.md``), so a frame written here is read byte-identically there. The mismatch
between the two venvs is a *package* mismatch, not a data-format one, and a process
boundary is the correct and complete answer to a package mismatch. The manifest records
the exact library versions the model was fitted with, and
:func:`predict_strong` refuses when the trainer venv no longer matches them. That version
record is the point: it turns "pin it and use it from there" into something verifiable.

**Latency, honestly.** One :func:`predict_strong` call is a fresh interpreter, a torch
import, a model load off disk, and a parquet round-trip. Measured on the reference domain
that is on the order of ten seconds for an AutoGluon stack, of which the prediction itself
is milliseconds; a warm page cache does not remove the interpreter start or the torch
import. This is a **batch** path: evaluation, the model card, the ceiling verification,
offline scoring, a demo. It is emphatically **not** an in-request path for the Aegis
agent — a tool call cannot spend ten seconds forking a second Python.

The in-request path is unchanged and is the portable
:class:`~aegis_ml.contracts.protocols.Recipe`: constructor kwargs that cross as JSON and
are re-fitted into the Aegis spine, which serves them in-process with its own MAPIE
conformal intervals and SHAP attribution. This module does not replace that. It makes the
ceiling the recipe is measured against an artifact instead of a claim.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.automl import runner
from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.contracts.protocols import TierName
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import light
    from collections.abc import Mapping, Sequence

    import numpy as np
    import pandas as pd

__all__ = [
    "AUTOGLUON_DIRNAME",
    "COMMON_VERSION_KEYS",
    "DEFAULT_VERIFY_TOLERANCE",
    "FRAME_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "PREDICTIONS_FILENAME",
    "PREPROCESSOR_FILENAME",
    "REQUEST_FILENAME",
    "RESULT_FILENAME",
    "STRONG_DIRNAME",
    "TABPFN_FILENAME",
    "TIER_VERSION_KEYS",
    "UNRECORDED_VERSION",
    "StrongModelFeatureMismatchError",
    "StrongModelMissingError",
    "StrongModelVersionDriftError",
    "StrongModelWeightsUnavailableError",
    "has_strong_model",
    "library_versions",
    "predict_proba_strong",
    "predict_strong",
    "save_strong_model",
    "strong_dir",
    "strong_manifest",
    "verify_strong",
    "version_drift",
]

STRONG_DIRNAME = "strong"
"""Sub-directory of a run directory that holds the non-portable winners.

A sub-directory rather than a flat artifact because an AutoGluon predictor *is* a
directory tree — ``aegis_ml.registry.store.artifact`` deliberately refuses anything but a
bare file name, so the strong models get their own namespace beside it rather than
flattening a tree into the run root.
"""

MANIFEST_FILENAME = "manifest.json"
"""Names the tier, the artifact, the feature order, the score and the fitted-with versions."""

AUTOGLUON_DIRNAME = "autogluon"
"""``strong/autogluon/`` — a cloned ``TabularPredictor`` directory, loadable in place."""

TABPFN_FILENAME = "tabpfn.joblib"
"""``strong/tabpfn.joblib`` — the fitted estimator, pickled and unpickled ONLY inside the
trainer venv. It never crosses to the serving venv, which is what makes the pickle safe:
both ends of the joblib round-trip are the same interpreter with the same wheels."""

PREPROCESSOR_FILENAME = "preprocessor.joblib"
"""The fitted ``ColumnTransformer`` a TabPFN fit was encoded through.

TabPFN is fitted on the encoded numeric matrix, not on the raw frame, so the encoder is
part of the model. Persisting the estimator without it would leave an artifact that can
only be called with an array nobody can reconstruct — which is a different way of throwing
the model away. AutoGluon needs no counterpart: it is fitted on the raw frame and does its
own encoding.
"""

MANIFEST_VERSION = 1
"""Schema version of ``strong/manifest.json``, so a future reader can refuse an old one."""

FRAME_FILENAME = "predict_frame.parquet"
"""The rows to score, written into the exchange directory for the predict worker."""

REQUEST_FILENAME = "predict_request.json"
"""Where to find the model, and what to compute. Read by ``_predict_worker``."""

PREDICTIONS_FILENAME = "predictions.parquet"
"""The worker's answer: a ``prediction`` column, plus one ``proba::<label>`` per class."""

RESULT_FILENAME = "predict_result.json"
"""Timings, the class label order, and the versions the trainer venv actually had."""

UNRECORDED_VERSION = "unrecorded"
"""Written when an interpreter cannot report a package's version.

Recorded as a literal string rather than omitted, because "we did not record this" and
"this package was absent" are different facts and a missing key cannot tell them apart.
"""

COMMON_VERSION_KEYS: tuple[str, ...] = ("pandas", "numpy", "scikit-learn", "joblib")
"""Distributions whose versions are recorded for every tier.

These four are the ones that decide whether a frame written on one side is read the same
way on the other. They are also the four that currently resolve identically in both venvs,
which is the fact the whole bridge rests on — so a drift in any of them is exactly what
the reader needs to be told about.
"""

TIER_VERSION_KEYS: dict[str, tuple[str, ...]] = {
    "autogluon": ("autogluon.tabular", "autogluon.core", "torch"),
    "tabpfn": ("tabpfn", "tabpfn_extensions", "torch"),
    "baseline": (),
    "flaml": ("flaml",),
}
"""Tier → the additional distributions whose versions are recorded at fit time.

``tabpfn_extensions`` is listed for the tabpfn tier even though the tier runs without it:
its presence or absence decides whether the fitted model is AutoTabPFN or plain TabPFN,
so a manifest that omitted it could not tell those two artifacts apart.
"""

_PROBA_PREFIX = "proba::"
"""Column prefix the worker uses for per-class probabilities in ``predictions.parquet``."""

DEFAULT_VERIFY_TOLERANCE = 1e-6
"""How far a reproduced score may drift from the recorded one before it is a failure.

Effectively exact. Both sides run the same model on the same rows in the same interpreter
family, so a real difference here does not mean "floating point" — it means the frame
handed to :func:`verify_strong` is not the frame the score was recorded on, which is a
finding, not a rounding error.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Refusals
# ─────────────────────────────────────────────────────────────────────────────
class StrongModelMissingError(AegisMLError):
    """No strong model is registered for this run.

    Raised instead of returning ``None`` or falling back to the portable recipe: a caller
    asking for the ceiling model and silently receiving the promoted spine would publish
    the spine's numbers under the ceiling's name.
    """

    def __init__(self, run_id: str, directory: Path, detail: str = "") -> None:
        """Name the run, where the model was looked for, and how one gets written."""
        suffix = f" {detail}" if detail else ""
        super().__init__(
            f"No strong model is registered for run {run_id!r}: nothing usable at "
            f"{str(directory)!r}.{suffix} A strong model is written only when the "
            f"autogluon or tabpfn tier actually fits one AND the search was given a run "
            f"to write into — re-run the search through "
            f"`aegis_ml.automl.runner.run_in_trainer_venv(..., run_id=<run_id>)`. "
            f"There is no fallback here: the portable recipe is a different model and "
            f"reporting its predictions as this one's would be a lie."
        )
        self.run_id = run_id
        self.directory = directory


class StrongModelFeatureMismatchError(AegisMLError):
    """The frame's feature columns are not the ones the model was fitted on.

    Column *order* is checked, not only membership. AutoGluon reads a named frame and
    would tolerate a reorder, but a TabPFN fit is called through a ``ColumnTransformer``
    positioned by declaration order, and a silently reordered frame there produces
    plausible numbers computed from the wrong columns.
    """

    def __init__(self, run_id: str, expected: list[str], received: list[str]) -> None:
        """Report the recorded order against the received one, naming the difference."""
        missing = [c for c in expected if c not in received]
        extra = [c for c in received if c not in expected]
        if missing or extra:
            difference = f"missing {missing or 'nothing'}; unexpected {extra or 'nothing'}"
        else:
            difference = "same columns, different order"
        super().__init__(
            f"Frame does not match the feature order run {run_id!r}'s strong model was "
            f"fitted on ({difference}).\n"
            f"  fitted on: {expected}\n"
            f"  received : {received}\n"
            f"Feeding it anyway would produce numbers computed from the wrong columns, "
            f"which is worse than an error because they look like predictions."
        )
        self.run_id = run_id
        self.expected = expected
        self.received = received


class StrongModelVersionDriftError(AegisMLError):
    """The trainer venv no longer has the libraries this model was fitted with.

    This is the refusal the whole manifest exists to make possible. Predicting through a
    different AutoGluon or torch than the one that fitted the artifact is the failure mode
    the two-venv split was built to avoid — silently, it produces either a load error or a
    subtly different model, and the second is indistinguishable from a real result.
    """

    def __init__(self, run_id: str, drift: Mapping[str, Sequence[str]]) -> None:
        """List every package whose version moved, with both numbers."""
        lines = "\n  - ".join(
            f"{package}: fitted with {pair[0]}, trainer venv now has {pair[1]}"
            for package, pair in sorted(drift.items())
        )
        super().__init__(
            f"Strong model for run {run_id!r} was fitted with library versions the "
            f"trainer venv no longer has:\n  - {lines}\n"
            f"Re-pin the trainer venv from requirements-strong.lock.txt "
            f"(`uv pip install --python <trainer-venv> -r requirements-strong.lock.txt`) "
            f"or re-fit the strong model against the current pins. Pass "
            f"allow_version_drift=True only when you have decided the drift is harmless — "
            f"the drift is then reported in the result rather than hidden."
        )
        self.run_id = run_id
        self.drift = {str(k): [str(v[0]), str(v[1])] for k, v in drift.items()}


class StrongModelWeightsUnavailableError(AegisMLError):
    """A TabPFN model cannot be called because its pretrained weights are not reachable.

    ``tabpfn`` imports cleanly with no weights on disk and then raises inside ``.fit()``
    or at load time. :func:`aegis_ml.automl.tiers.unavailable_reason` already probes for
    this before a search starts; this raises the same actionable refusal *before* the
    model is loaded, so the failure never lands in the middle of a prediction.
    """

    def __init__(self, run_id: str, reason: str) -> None:
        """Carry the tier probe's reason verbatim — it already names the remedy."""
        super().__init__(
            f"Strong model for run {run_id!r} is a TabPFN fit and its weights are not "
            f"available in the trainer venv: {reason}"
        )
        self.run_id = run_id
        self.reason = reason


# ─────────────────────────────────────────────────────────────────────────────
# Locations and manifests
# ─────────────────────────────────────────────────────────────────────────────
def strong_dir(run_id: str) -> Path:
    """Return ``<registry>/runs/<run_id>/strong``, creating the run directory.

    Args:
        run_id: The run identifier, validated by
            :func:`aegis_ml.registry.store.run_dir` — it reaches this function from JSON
            that crossed a venv boundary, so it is treated as untrusted input.

    Returns:
        The directory the strong artifacts live in. It is not created here; only the run
        directory is, because an empty ``strong/`` would make
        :func:`has_strong_model` look at a directory that promises a model it has not got.
    """
    from aegis_ml.registry import store  # noqa: PLC0415 - avoids a package-level import cycle

    return store.run_dir(run_id) / STRONG_DIRNAME


def strong_manifest(run_id: str) -> dict[str, Any] | None:
    """Return the strong model's manifest for ``run_id``, or ``None`` when there is none.

    Args:
        run_id: The run to look up.

    Returns:
        The parsed ``strong/manifest.json``, or ``None`` when the run has no strong model.
        ``None`` is a legitimate answer here — this is the question "is there one?", so a
        negative answer is information rather than a degradation.
    """
    path = strong_dir(run_id) / MANIFEST_FILENAME
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def has_strong_model(run_id: str) -> bool:
    """Return whether ``run_id`` has a strong model that is actually on disk.

    Checks the artifact the manifest names, not merely the manifest: a run that was
    interrupted between writing the two would otherwise report a model that cannot be
    loaded, and the report is read by the model card.

    Args:
        run_id: The run to check.

    Returns:
        ``True`` when both the manifest and the artifact it names exist.
    """
    manifest = strong_manifest(run_id)
    if manifest is None:
        return False
    artifact = strong_dir(run_id) / str(manifest.get("artifact", ""))
    return artifact.exists()


def library_versions(tier: str) -> dict[str, str]:
    """Return the installed versions of every distribution this tier's artifact depends on.

    Read from installed distribution metadata rather than by importing the packages: the
    caller may be ``aegis-ml doctor`` in the serving venv, and importing ``torch`` to read
    ``torch.__version__`` costs seconds and megabytes to answer a question about a string.

    Args:
        tier: The search tier the model came from. Unknown tiers get the common keys only,
            which is a smaller record rather than a wrong one.

    Returns:
        Distribution name → version, or :data:`UNRECORDED_VERSION` when this interpreter
        cannot report it (the distribution is absent, or installed without metadata).
    """
    from importlib import metadata  # noqa: PLC0415 - stdlib, imported where it is used

    keys = (*COMMON_VERSION_KEYS, *TIER_VERSION_KEYS.get(tier, ()))
    found: dict[str, str] = {}
    for name in keys:
        try:
            found[name] = metadata.version(name)
        # audit-ok: the absent-package case IS the recorded answer, never a silent skip.
        except metadata.PackageNotFoundError:
            found[name] = UNRECORDED_VERSION
    return found


def version_drift(
    recorded: Mapping[str, str], current: Mapping[str, str]
) -> dict[str, list[str]]:
    """Return every package whose version differs between a manifest and an interpreter.

    Only keys the manifest recorded are compared. A key the manifest does not carry is not
    drift — it is a manifest written by an older version of this module, and treating that
    as drift would refuse every model saved before a key was added.

    Args:
        recorded: ``library_versions`` as captured at fit time.
        current: ``library_versions`` as measured in the interpreter about to predict.

    Returns:
        Package → ``[fitted_with, now]`` for every difference. Empty when they agree.
    """
    drift: dict[str, list[str]] = {}
    for package, was in recorded.items():
        now = current.get(package, UNRECORDED_VERSION)
        if str(was) != str(now):
            drift[package] = [str(was), str(now)]
    return drift


# ─────────────────────────────────────────────────────────────────────────────
# Persisting
# ─────────────────────────────────────────────────────────────────────────────
def save_strong_model(  # noqa: PLR0913 - each argument is a separate recorded fact
    run_id: str,
    tier: TierName,
    model: Any,  # noqa: ANN401 - a TabularPredictor or a fitted sklearn-style estimator
    *,
    problem: MLProblem,
    score: float,
    feature_order: Sequence[str],
    metric: str | None = None,
    preprocessor: Any = None,  # noqa: ANN401 - a fitted ColumnTransformer, or None
    model_name: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a fitted strong model into ``runs/<run_id>/strong/`` with its manifest.

    Call this **from the interpreter that fitted the model** — the trainer venv. The
    versions written into the manifest are read here, at fit time, from this interpreter,
    which is the only place they are true. Reading them later, from a venv that has since
    been re-resolved, would record a pin the artifact was never built against.

    Args:
        run_id: The run to write into.
        tier: ``"autogluon"`` or ``"tabpfn"``. Other tiers are portable and belong in a
            :class:`~aegis_ml.contracts.protocols.Recipe`, not here.
        model: The fitted artifact. An ``autogluon`` tier gets a ``TabularPredictor``,
            which is cloned to the destination (a predictor is a directory tree, not an
            object); anything else is written with joblib.
        problem: The spec, stored whole so the manifest is self-describing — a verifier
            does not have to be handed the problem again to know the task, the metric or
            the class levels.
        score: The value this model scored on the search's held-out split. This is the
            number :func:`verify_strong` reproduces.
        feature_order: The raw frame columns, in the order the model consumes them.
        metric: Metric name for ``score``; defaults to the problem's primary metric.
        preprocessor: The fitted encoder a non-frame model was fitted through. Required
            for TabPFN, which is fitted on an encoded matrix — see
            :data:`PREPROCESSOR_FILENAME`.
        model_name: For AutoGluon, the specific fitted model the score belongs to (e.g.
            ``"WeightedEnsemble_L2"``). Recorded so prediction uses the same one that was
            scored, rather than whichever the predictor considers best today.
        detail: Anything else worth carrying — licence notices, subsampling notes.

    Returns:
        The ``strong/`` directory that now holds the artifact and its manifest.

    Raises:
        ValueError: If the tabpfn tier is saved without the encoder it was fitted through,
            which would produce an artifact nobody can call.
        ImportError: If joblib is missing.
    """
    destination = strong_dir(run_id)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if tier == "autogluon":
        artifact_name = AUTOGLUON_DIRNAME
        artifact_kind = "autogluon_predictor"
        model.clone(path=str(destination / AUTOGLUON_DIRNAME), return_clone=False)
    else:
        artifact_name = TABPFN_FILENAME if tier == "tabpfn" else f"{tier}.joblib"
        artifact_kind = "joblib"
        joblib = require("aegis-ml[strong]", "joblib")
        joblib.dump(model, destination / artifact_name)

    preprocessor_name: str | None = None
    if preprocessor is not None:
        joblib = require("aegis-ml[strong]", "joblib")
        joblib.dump(preprocessor, destination / PREPROCESSOR_FILENAME)
        preprocessor_name = PREPROCESSOR_FILENAME
    elif tier == "tabpfn":
        raise ValueError(
            "save_strong_model(tier='tabpfn') needs the fitted preprocessor: TabPFN is "
            "fitted on the encoded matrix, so an estimator saved without its encoder can "
            "only be called with an array no later caller can reconstruct."
        )

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "tier": tier,
        "estimator_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_name": model_name,
        "artifact": artifact_name,
        "artifact_kind": artifact_kind,
        "preprocessor": preprocessor_name,
        "feature_order": [str(c) for c in feature_order],
        "task": problem.target.task,
        "target": problem.target.name,
        "labels": [str(level) for level in problem.target.levels],
        "metric": metric or problem.metric,
        "score": float(score),
        "library_versions": library_versions(tier),
        "fitted_by": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": sys.platform,
        },
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "problem": problem.model_dump(mode="json"),
        "detail": dict(detail or {}),
        "serving_note": (
            "Batch path only. Calling this model runs a subprocess in the trainer venv: "
            "interpreter start, torch import and model load dominate, so a single call "
            "costs seconds. The in-request path is the portable recipe fitted into the "
            "Aegis spine; this artifact exists so the accuracy ceiling can be verified "
            "and scored offline rather than only asserted."
        ),
    }
    (destination / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )
    return destination


# ─────────────────────────────────────────────────────────────────────────────
# Predicting across the venv boundary
# ─────────────────────────────────────────────────────────────────────────────
def _load_manifest_or_refuse(run_id: str) -> dict[str, Any]:
    """Return the manifest, raising :class:`StrongModelMissingError` when it is unusable."""
    directory = strong_dir(run_id)
    manifest = strong_manifest(run_id)
    if manifest is None:
        raise StrongModelMissingError(run_id, directory, "no manifest.json.")
    artifact = directory / str(manifest.get("artifact", ""))
    if not artifact.exists():
        raise StrongModelMissingError(
            run_id,
            directory,
            f"manifest names artifact {manifest.get('artifact')!r}, which is not there.",
        )
    return manifest


def _check_feature_order(run_id: str, manifest: Mapping[str, Any], frame: pd.DataFrame) -> None:
    """Refuse a frame whose feature columns differ, in content or in order, from the fit.

    Non-feature columns are allowed through and ignored — a holdout frame legitimately
    still carries its target column, and demanding it be dropped first would make the
    honest call the awkward one. What is checked is that the *feature* columns present,
    read in the frame's own order, are exactly the recorded order.
    """
    expected = [str(c) for c in manifest["feature_order"]]
    wanted = set(expected)
    present = [str(c) for c in frame.columns if str(c) in wanted]
    if present != expected:
        raise StrongModelFeatureMismatchError(run_id, expected, present)


def _exchange(
    run_id: str,
    manifest: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    want_proba: bool,
    allow_version_drift: bool,
    timeout: int | None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Run one prediction in the trainer venv and read the answer back.

    Returns:
        ``(prediction, proba_or_None, result)`` where ``result`` carries the worker's
        timings, the class label order and the versions the trainer venv actually had.

    Raises:
        TrainerVenvMissingError: If the trainer interpreter is absent.
        StrongModelVersionDriftError: If the trainer venv's libraries moved since the fit.
        StrongModelWeightsUnavailableError: If a TabPFN artifact has no reachable weights.
        AegisMLError: For any other child failure, carrying the child's traceback.
    """
    interpreter = runner.trainer_python()
    require("aegis-ml[serve]", "pyarrow")
    pd_mod = require("aegis-ml[serve]", "pandas")
    np_mod = require("aegis-ml[serve]", "numpy")

    with tempfile.TemporaryDirectory(prefix="aegis_ml_predict_") as tmp:
        directory = Path(tmp)
        frame.to_parquet(directory / FRAME_FILENAME, index=False)
        request = {
            "run_id": run_id,
            "strong_dir": str(strong_dir(run_id)),
            "want_proba": bool(want_proba),
            "allow_version_drift": bool(allow_version_drift),
        }
        (directory / REQUEST_FILENAME).write_text(
            json.dumps(request, indent=2), encoding="utf-8"
        )

        started = time.perf_counter()
        try:
            runner.invoke_trainer_module(
                "aegis_ml.automl._predict_worker",
                directory,
                interpreter=interpreter,
                timeout=timeout if timeout is not None else _default_predict_timeout(),
                label="Strong-model prediction",
                banner=(
                    f"[aegis-ml] strong-model prediction for run {run_id} "
                    f"(tier {manifest.get('tier')}) in the trainer venv"
                ),
                remedy=(
                    "The strong model was NOT called, so there is no partial prediction "
                    "to fall back on. The portable recipe in this run's recipe.json is a "
                    "different model and must not be substituted for it."
                ),
            )
        except AegisMLError as exc:
            raise _typed_child_error(run_id, directory, exc) from exc
        round_trip = time.perf_counter() - started

        predictions = pd_mod.read_parquet(directory / PREDICTIONS_FILENAME)
        result = dict(json.loads((directory / RESULT_FILENAME).read_text(encoding="utf-8")))

    result["round_trip_seconds"] = round_trip
    prediction = np_mod.asarray(predictions["prediction"].to_numpy())
    proba_columns = [c for c in predictions.columns if str(c).startswith(_PROBA_PREFIX)]
    proba = None
    if proba_columns:
        order = [f"{_PROBA_PREFIX}{label}" for label in result.get("labels", [])]
        columns = order if set(order) <= set(map(str, proba_columns)) else proba_columns
        proba = np_mod.asarray(predictions[columns].to_numpy(dtype=float))
    return prediction, proba, result


def _typed_child_error(run_id: str, directory: Path, failure: AegisMLError) -> AegisMLError:
    """Re-raise a child failure as the specific refusal it was, when it named one.

    The worker writes ``error.json`` carrying its own exception type. Mapping the two
    version-and-weights cases back to typed parent exceptions is what lets a caller
    distinguish "your trainer venv drifted" from "AutoGluon crashed" without parsing an
    error message — and drift is precisely the condition that must be reported rather than
    absorbed.
    """
    payload = runner.read_child_error(directory)
    if payload is None:
        return failure
    kind = str(payload.get("type", ""))
    detail = payload.get("detail") or {}
    if kind == StrongModelVersionDriftError.__name__:
        return StrongModelVersionDriftError(run_id, dict(detail.get("drift", {})))
    if kind == StrongModelWeightsUnavailableError.__name__:
        return StrongModelWeightsUnavailableError(run_id, str(payload.get("message", "")))
    if kind == StrongModelMissingError.__name__:
        return StrongModelMissingError(run_id, strong_dir(run_id), str(payload.get("message", "")))
    return failure


def _default_predict_timeout() -> int:
    """Return the wall-clock ceiling for one prediction subprocess.

    Ten minutes. Generous on purpose: the cost is a cold torch import and an AutoGluon
    stack load, both of which can take a minute on a cold filesystem, and a ceiling that
    fires during a legitimate load would look exactly like the model being broken. It is
    still a ceiling — a hung child is killed and reported, never waited on forever.
    """
    return int(os.environ.get("AEGIS_ML_STRONG_PREDICT_TIMEOUT", "600"))


def predict_strong(
    run_id: str,
    frame: pd.DataFrame,
    *,
    allow_version_drift: bool = False,
    timeout: int | None = None,
) -> np.ndarray:
    """Predict with ``run_id``'s persisted strong model, running it in the trainer venv.

    This is the call that makes the accuracy ceiling usable instead of merely reported.
    The model is loaded and called by ``settings.trainer_python``; this interpreter only
    writes a parquet frame and reads a parquet column back.

    **Cost.** A subprocess round-trip: a fresh interpreter, a torch import, a model load,
    and two parquet writes. On the reference domain that is seconds per call, dominated by
    load rather than by the prediction. Use it for batch scoring, evaluation, the model
    card and :func:`verify_strong`. Do **not** put it on an Aegis tool-call path — the
    in-request model is the portable recipe fitted into the Aegis spine, which answers
    in-process.

    Args:
        run_id: The run whose strong model to call.
        frame: Rows to score. Must carry every recorded feature column, in the recorded
            order; other columns (a target, an id) are ignored.
        allow_version_drift: When ``False`` (the default) a trainer venv whose libraries
            no longer match the manifest is refused. Setting it ``True`` proceeds *and
            records the drift* in the worker result — it never hides it.
        timeout: Wall-clock ceiling for the child, in seconds.

    Returns:
        A 1-d array of predictions, aligned row-for-row with ``frame``.

    Raises:
        TrainerVenvMissingError: If the trainer venv is not built.
        StrongModelMissingError: If this run has no strong model on disk.
        StrongModelFeatureMismatchError: If the frame's feature columns differ.
        StrongModelVersionDriftError: If the trainer venv's libraries moved since the fit.
        StrongModelWeightsUnavailableError: If a TabPFN artifact's weights are unreachable.
        AegisMLError: If the child fails for any other reason, carrying its traceback.
    """
    manifest = _load_manifest_or_refuse(run_id)
    _check_feature_order(run_id, manifest, frame)
    prediction, _proba, _result = _exchange(
        run_id,
        manifest,
        frame,
        want_proba=False,
        allow_version_drift=allow_version_drift,
        timeout=timeout,
    )
    return prediction


def predict_proba_strong(
    run_id: str,
    frame: pd.DataFrame,
    *,
    allow_version_drift: bool = False,
    timeout: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Predict class probabilities with the persisted strong model, in the trainer venv.

    Separate from :func:`predict_strong` rather than a flag on it, because the two return
    different shapes and a caller that got a matrix where it expected a vector finds out
    several lines later. Same subprocess cost; see :func:`predict_strong` on latency.

    Args:
        run_id: The run whose strong model to call.
        frame: Rows to score, matching the recorded feature order.
        allow_version_drift: See :func:`predict_strong`.
        timeout: Wall-clock ceiling for the child, in seconds.

    Returns:
        ``(proba, labels)`` — an ``(n_rows, n_classes)`` matrix and the class labels naming
        its columns, in column order.

    Raises:
        AegisMLError: If the model is a regressor, or for the refusals
            :func:`predict_strong` documents.
    """
    manifest = _load_manifest_or_refuse(run_id)
    if manifest.get("task") != "classification":
        raise AegisMLError(
            f"Strong model for run {run_id!r} is a {manifest.get('task')} model; it has "
            f"no class probabilities. Call predict_strong() instead."
        )
    _check_feature_order(run_id, manifest, frame)
    _prediction, proba, result = _exchange(
        run_id,
        manifest,
        frame,
        want_proba=True,
        allow_version_drift=allow_version_drift,
        timeout=timeout,
    )
    if proba is None:
        raise AegisMLError(
            f"Strong model for run {run_id!r} returned no probabilities: its estimator "
            f"{manifest.get('estimator_class')!r} exposes no predict_proba. Score it on a "
            f"label metric (accuracy, f1_macro) rather than roc_auc or log_loss."
        )
    return proba, [str(label) for label in result.get("labels", [])]


def verify_strong(
    run_id: str,
    frame: pd.DataFrame,
    expected: Any,  # noqa: ANN401 - a pandas Series or 1-d array of ground truth
    *,
    tolerance: float = DEFAULT_VERIFY_TOLERANCE,
    allow_version_drift: bool = False,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Re-run the persisted strong model and check its recorded score reproduces.

    This is what turns the leaderboard's accuracy-ceiling row from an assertion into a
    measurement. The leaderboard says "AutoGluon scored 0.7912"; this loads the model that
    produced that number, runs it again on the same held-out rows through the venv bridge,
    scores it with the same function the search used, and reports the difference.

    A non-zero delta is a real finding, not noise: the same model on the same rows in the
    same interpreter is deterministic. It means either the rows are not the recorded
    holdout, or the artifact is not the model that was scored.

    Args:
        run_id: The run to verify.
        frame: The rows the recorded score was measured on — the search's holdout split.
        expected: Ground-truth target values, aligned with ``frame``.
        tolerance: Absolute difference allowed before ``reproduces`` is ``False``.
        allow_version_drift: See :func:`predict_strong`. The drift is reported in the
            result either way.
        timeout: Wall-clock ceiling for the child, in seconds.

    Returns:
        A dictionary carrying ``recorded_score``, ``reproduced_score``, ``delta``,
        ``reproduces``, the metric name, the row count, both version records, any
        ``version_drift``, and the measured ``round_trip_seconds`` /
        ``worker_load_seconds`` / ``worker_predict_seconds`` — the latency this path
        actually costs, measured rather than estimated.

    Raises:
        AegisMLError: For every refusal :func:`predict_strong` documents.
    """
    from aegis_ml.automl.search import score_predictions  # noqa: PLC0415 - avoids a cycle

    manifest = _load_manifest_or_refuse(run_id)
    _check_feature_order(run_id, manifest, frame)
    problem = MLProblem.model_validate(manifest["problem"])
    metric = str(manifest["metric"])
    want_proba = problem.target.task == "classification"

    prediction, proba, result = _exchange(
        run_id,
        manifest,
        frame,
        want_proba=want_proba,
        allow_version_drift=allow_version_drift,
        timeout=timeout,
    )
    labels = [str(label) for label in result.get("labels", [])] or None
    reproduced = score_predictions(metric, expected, prediction, y_proba=proba, labels=labels)
    recorded = float(manifest["score"])
    delta = reproduced - recorded
    recorded_versions = dict(manifest.get("library_versions", {}))
    current_versions = dict(result.get("library_versions", {}))

    return {
        "run_id": run_id,
        "tier": manifest.get("tier"),
        "estimator_class": manifest.get("estimator_class"),
        "model_name": manifest.get("model_name"),
        "metric": metric,
        "n_rows": int(len(frame)),
        "recorded_score": recorded,
        "reproduced_score": float(reproduced),
        "delta": float(delta),
        "reproduces": bool(abs(delta) <= tolerance),
        "tolerance": float(tolerance),
        "library_versions_at_fit": recorded_versions,
        "library_versions_now": current_versions,
        "version_drift": version_drift(recorded_versions, current_versions),
        "round_trip_seconds": float(result.get("round_trip_seconds", 0.0)),
        "worker_load_seconds": float(result.get("load_seconds", 0.0)),
        "worker_predict_seconds": float(result.get("predict_seconds", 0.0)),
        "trainer_python": str(settings.trainer_python),
    }
