"""Load a persisted strong model and predict with it, inside the trainer venv.

Run as ``python -m aegis_ml.automl._predict_worker <exchange-dir>`` by
:func:`aegis_ml.automl.strong.predict_strong`. This is the half of the bridge that has
AutoGluon, TabPFN and torch; the parent has none of them and never will, which is the
whole design (decision D1).

It is a module and not a generated script for the same reason
:mod:`aegis_ml.automl._worker` is: a bridge whose remote half is a string of Python built
at call time is a bridge nobody can review, lint or type-check.

Its contract is small and its order is deliberate:

1. read ``predict_request.json`` and the manifest it points at;
2. **compare library versions before loading anything.** The recorded versions are checked
   against this interpreter's from distribution metadata — no imports, so a drift is
   reported in milliseconds rather than after a sixty-second AutoGluon load. This is the
   check the whole manifest exists for: predicting through a different torch than the one
   that fitted the artifact is exactly the silent-wrongness the two-venv split was built
   to prevent;
3. for a TabPFN artifact, confirm the pretrained weights are reachable **before** the
   load, so a missing ``TABPFN_TOKEN`` surfaces as the same actionable refusal
   :func:`aegis_ml.automl.tiers.unavailable_reason` gives, never as a crash mid-predict;
4. load, predict, write ``predictions.parquet`` and ``predict_result.json``;
5. on any failure write ``error.json`` — carrying the exception *type*, so the parent can
   re-raise the specific refusal rather than pattern-matching a string — and exit non-zero.

**Why the frame crosses as parquet and the model never crosses at all.** ``pandas``,
``numpy`` and ``scikit-learn`` resolve to identical versions in both venvs, so a frame
written by the serving venv is read byte-identically here. The fitted model, by contrast,
is unpickled only here — the same interpreter and the same wheels that pickled it. Nothing
built against AutoGluon or torch is ever handed to the serving venv, in either direction.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from aegis_ml._require import require
from aegis_ml.automl import strong
from aegis_ml.automl.runner import ERROR_FILENAME

__all__ = ["main"]


def _log(message: str) -> None:
    """Write one line to stderr, which the parent streams live and retains a tail of."""
    print(message, file=sys.stderr, flush=True)


def _check_versions(manifest: dict[str, Any], *, allow_drift: bool) -> dict[str, str]:
    """Compare the manifest's recorded versions against this interpreter's.

    Args:
        manifest: The strong model's manifest.
        allow_drift: When ``True`` the drift is logged and the prediction proceeds. It is
            still recorded in the result the parent reads, so an allowed drift is a
            reported drift — never an invisible one.

    Returns:
        The versions this interpreter actually has.

    Raises:
        StrongModelVersionDriftError: When a version moved and ``allow_drift`` is ``False``.
    """
    tier = str(manifest.get("tier", ""))
    recorded = dict(manifest.get("library_versions", {}))
    current = strong.library_versions(tier)
    drift = strong.version_drift(recorded, current)
    if not drift:
        _log(f"aegis-ml predict: library versions match the manifest ({len(recorded)} pinned)")
        return current
    if not allow_drift:
        raise strong.StrongModelVersionDriftError(str(manifest.get("run_id", "?")), drift)
    for package, pair in sorted(drift.items()):
        _log(
            f"aegis-ml predict: VERSION DRIFT ALLOWED for {package}: fitted with "
            f"{pair[0]}, predicting with {pair[1]} — this is recorded in the result"
        )
    return current


def _check_weights(manifest: dict[str, Any]) -> None:
    """Refuse a TabPFN artifact whose pretrained weights are not reachable here.

    Checked before the load rather than caught during it. ``tabpfn`` imports cleanly with
    no weights on disk and only raises when it reaches for them, which without this check
    would be a traceback in the middle of a prediction rather than a sentence naming the
    one-time setup that fixes it.

    Raises:
        StrongModelWeightsUnavailableError: When the weights cannot be reached.
    """
    if str(manifest.get("tier")) != "tabpfn":
        return
    from aegis_ml.automl import tiers  # noqa: PLC0415 - only this branch needs the probe

    reason = tiers.unavailable_reason("tabpfn")
    if reason is not None:
        raise strong.StrongModelWeightsUnavailableError(str(manifest.get("run_id", "?")), reason)


def _predict_autogluon(
    directory: Path, manifest: dict[str, Any], frame: Any  # noqa: ANN401 - a pandas DataFrame
) -> tuple[Any, Any, list[str]]:
    """Load the cloned ``TabularPredictor`` and predict with the model that was scored.

    Returns:
        ``(prediction, proba_or_None, labels)``.
    """
    tabular = require("aegis-ml[strong]", "autogluon.tabular")
    predictor = tabular.TabularPredictor.load(str(directory / str(manifest["artifact"])))
    model_name = manifest.get("model_name")
    features = frame[[str(c) for c in manifest["feature_order"]]]

    prediction = predictor.predict(features, model=model_name)
    if str(manifest.get("task")) != "classification":
        return prediction, None, []
    proba_frame = predictor.predict_proba(features, model=model_name)
    labels = [str(c) for c in proba_frame.columns]
    return prediction, proba_frame, labels


def _predict_joblib(
    directory: Path, manifest: dict[str, Any], frame: Any  # noqa: ANN401 - a pandas DataFrame
) -> tuple[Any, Any, list[str]]:
    """Load a joblib'd estimator plus its encoder and predict on the encoded matrix.

    The encoder is not optional for these tiers and its absence is an error rather than a
    guess: TabPFN was fitted on the transformed matrix, so predicting on the raw frame
    would either raise or — with the right number of numeric columns — silently score the
    wrong quantities.

    Returns:
        ``(prediction, proba_or_None, labels)``.

    Raises:
        StrongModelMissingError: If the manifest names no encoder.
    """
    joblib = require("aegis-ml[strong]", "joblib")
    np_mod = require("aegis-ml[serve]", "numpy")
    preprocessor_name = manifest.get("preprocessor")
    if not preprocessor_name:
        raise strong.StrongModelMissingError(
            str(manifest.get("run_id", "?")),
            directory,
            "the manifest names no preprocessor, and this artifact was fitted on an "
            "encoded matrix that cannot be reconstructed without one.",
        )
    estimator = joblib.load(directory / str(manifest["artifact"]))
    preprocessor = joblib.load(directory / str(preprocessor_name))

    features = frame[[str(c) for c in manifest["feature_order"]]]
    encoded = np_mod.asarray(preprocessor.transform(features), dtype=float)
    prediction = estimator.predict(encoded)
    if str(manifest.get("task")) != "classification":
        return prediction, None, []
    proba_fn = getattr(estimator, "predict_proba", None)
    if not callable(proba_fn):
        return prediction, None, []
    proba = proba_fn(encoded)
    labels = [str(c) for c in getattr(estimator, "classes_", [])]
    return prediction, proba, labels


def _run(directory: Path) -> int:
    """Execute one prediction in this interpreter and write its two result files.

    Args:
        directory: The exchange directory prepared by
            :func:`aegis_ml.automl.strong.predict_strong`.

    Returns:
        Process exit code: 0 on success.
    """
    pd_mod = require("aegis-ml[serve]", "pandas")

    request = json.loads((directory / strong.REQUEST_FILENAME).read_text(encoding="utf-8"))
    model_dir = Path(str(request["strong_dir"]))
    manifest_path = model_dir / strong.MANIFEST_FILENAME
    if not manifest_path.exists():
        raise strong.StrongModelMissingError(
            str(request.get("run_id", "?")), model_dir, "no manifest.json in the trainer venv."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _log(f"aegis-ml predict: python {sys.version.split()[0]} at {sys.executable}")
    _log(
        f"aegis-ml predict: run {manifest.get('run_id')} tier {manifest.get('tier')} "
        f"model {manifest.get('model_name')} recorded "
        f"{manifest.get('metric')}={manifest.get('score')}"
    )
    current = _check_versions(manifest, allow_drift=bool(request.get("allow_version_drift")))
    _check_weights(manifest)

    frame = pd_mod.read_parquet(directory / strong.FRAME_FILENAME)
    _log(f"aegis-ml predict: scoring {len(frame)} rows")

    load_started = time.perf_counter()
    kind = str(manifest.get("artifact_kind"))
    loader = _predict_autogluon if kind == "autogluon_predictor" else _predict_joblib
    prediction, proba, labels = loader(model_dir, manifest, frame)
    elapsed = time.perf_counter() - load_started

    out = pd_mod.DataFrame({"prediction": list(prediction)})
    if proba is not None and request.get("want_proba"):
        matrix = proba.to_numpy() if hasattr(proba, "to_numpy") else proba
        for index, label in enumerate(labels):
            out[f"proba::{label}"] = matrix[:, index]
    out.to_parquet(directory / strong.PREDICTIONS_FILENAME, index=False)

    result = {
        "run_id": manifest.get("run_id"),
        "tier": manifest.get("tier"),
        "model_name": manifest.get("model_name"),
        "labels": labels,
        "n_rows": int(len(frame)),
        # Load and predict are reported as one number because they are one cost to the
        # caller and separating them would invite quoting the smaller half: the model is
        # loaded from disk on every call, since every call is a new interpreter.
        "load_seconds": float(elapsed),
        "predict_seconds": float(elapsed),
        "library_versions": current,
        "predicted_by": {"python": sys.version.split()[0], "executable": sys.executable},
    }
    (directory / strong.RESULT_FILENAME).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _log(f"aegis-ml predict: wrote {len(out)} predictions in {elapsed:.2f}s (load + predict)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m aegis_ml.automl._predict_worker <exchange-dir>``.

    Args:
        argv: Command-line arguments after the module name; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a usage error, 1 if the prediction raised (with ``error.json``
        written next to the inputs, carrying the exception type the parent maps back onto
        a typed refusal).
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        _log("usage: python -m aegis_ml.automl._predict_worker <exchange-dir>")
        return 2
    directory = Path(args[0])
    if not directory.is_dir():
        _log(f"exchange directory {directory} does not exist")
        return 2

    try:
        return _run(directory)
    except Exception as exc:  # audit-ok: the traceback is written out, never swallowed
        payload: dict[str, Any] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "detail": {"drift": getattr(exc, "drift", {})},
        }
        (directory / ERROR_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess, not by import
    raise SystemExit(main())
