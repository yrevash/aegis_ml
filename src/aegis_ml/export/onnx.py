"""Export the fitted **point predictor** to ONNX, and prove the export by round-tripping it.

Read this before using it, because the honest scope is narrow:

* **MAPIE's conformal intervals do not export.** The calibrated interval is not a tensor
  operation inside the model — it is a quantile of held-out residuals applied around the
  point prediction at request time. An ONNX graph produced here returns the point estimate
  and nothing else, so an ONNX-served prediction carries no coverage guarantee at all. The
  single most valuable property of the Aegis spine is the one that does not survive this
  file.
* **SHAP attributions do not export either.** ``TreeExplainer`` walks the fitted tree
  structure; there is no ONNX op for it. An ONNX deployment cannot answer "why".
* **This is not a faster serving path for Aegis.** ``onnxruntime`` (CPU) is already present
  in the backend venv transitively via ``fastembed``, and the spine's in-process predict is
  already sub-millisecond. Swapping it for ONNX would trade the interval and the
  explanation for a speedup nobody is waiting on.

So what is it for? Two real things. First, **portability**: a single ``.onnx`` file that a
Java, C#, Go or browser runtime can score, for the integration conversation that always
comes up and never has a good answer. Second, **verification**: :func:`validate_roundtrip`
measures the largest disagreement between the fitted sklearn model and the exported graph
on real rows, and refuses above tolerance. An export nobody round-tripped is a file whose
predictions have never been compared to anything.

Both converters are registered explicitly rather than relied upon: skl2onnx ships shape
calculators for sklearn's own estimators, and XGBoost/LightGBM members need
``onnxmltools``' converters wired in by :func:`register_converters` before
``convert_sklearn`` will accept a pipeline containing them.

**Two limits found by running this, not by reading about it.** Both are the reason
:func:`validate_roundtrip` is not optional:

* **Missing values do not round-trip reliably.** scikit-learn's tree learners route NaN
  down a direction they *learned* during fitting. The ONNX ``TreeEnsemble`` op does not
  reproduce that routing for every member type, so a model fitted on frames with MAR
  missingness converts without an error and then disagrees by whole units — on exactly the
  rows that are missing a value, which are the interesting ones. Measured here at ~17 on a
  target whose range is tens, against ~2e-6 for the same forest on complete rows. If the
  export has to be faithful, impute before the model rather than relying on the learner's
  native NaN path.
* **Conversion is version-coupled.** skl2onnx, onnx and scikit-learn have to agree; when
  they do not, the failure is a ``TypeError`` deep inside the TreeEnsemble attribute
  encoding rather than an "unsupported model" message. :func:`to_onnx` catches that and
  re-raises it with the member list and both likely causes attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml._require import is_available, require
from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.contracts.spec import MLProblem

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import light
    import pandas as pd

__all__ = [
    "DEFAULT_TOLERANCE",
    "register_converters",
    "to_onnx",
    "validate_roundtrip",
]

DEFAULT_TOLERANCE: float = 1e-4
"""Largest tolerated disagreement between the sklearn model and its ONNX graph.

Not zero, and the reason is arithmetic rather than sloppiness: ONNX tree ensembles score in
float32 while sklearn scores in float64, and the error compounds across a few hundred
boosted trees. Empirically that lands around 1e-6 and stays well inside 1e-4; a difference
*larger* than this is not rounding, it is a converter mapping the model wrongly — which is
exactly what this function exists to catch.
"""


def register_converters() -> list[str]:
    """Register the third-party converters ``convert_sklearn`` needs, and say which.

    skl2onnx knows sklearn. It does not know XGBoost or LightGBM, and a pipeline containing
    an ``XGBRegressor`` fails conversion with a shape-calculator error that reads like a
    bug in the model rather than a missing registration. Registering is idempotent, so this
    is safe to call before every export.

    Returns:
        The names of the estimator classes wired up, in registration order. A caller can
        assert on this: an empty list where XGBoost was expected means the export would
        silently have covered a different set of members than the model contains.
    """
    skl2onnx = require("aegis-ml[mlops]", "skl2onnx")
    shape_calc = require("aegis-ml[mlops]", "skl2onnx.common.shape_calculator")
    registered: list[str] = []

    if is_available("xgboost"):
        xgboost = require("aegis-ml[serve]", "xgboost")
        converters = require(
            "aegis-ml[mlops]", "onnxmltools.convert.xgboost.operator_converters.XGBoost"
        )
        skl2onnx.update_registered_converter(
            xgboost.XGBRegressor,
            "XGBoostXGBRegressor",
            shape_calc.calculate_linear_regressor_output_shapes,
            converters.convert_xgboost,
        )
        skl2onnx.update_registered_converter(
            xgboost.XGBClassifier,
            "XGBoostXGBClassifier",
            shape_calc.calculate_linear_classifier_output_shapes,
            converters.convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        registered += ["XGBRegressor", "XGBClassifier"]

    if is_available("lightgbm"):
        lightgbm = require("aegis-ml[strong]", "lightgbm")
        converters = require(
            "aegis-ml[mlops]", "onnxmltools.convert.lightgbm.operator_converters.LightGbm"
        )
        skl2onnx.update_registered_converter(
            lightgbm.LGBMRegressor,
            "LightGbmLGBMRegressor",
            shape_calc.calculate_linear_regressor_output_shapes,
            converters.convert_lightgbm,
        )
        skl2onnx.update_registered_converter(
            lightgbm.LGBMClassifier,
            "LightGbmLGBMClassifier",
            shape_calc.calculate_linear_classifier_output_shapes,
            converters.convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        registered += ["LGBMRegressor", "LGBMClassifier"]

    return registered


def _initial_types(problem: MLProblem) -> list[tuple[str, Any]]:
    """Describe the graph's inputs: one named tensor per declared feature column.

    Per-column inputs rather than one wide matrix, because the fitted model starts with a
    ``ColumnTransformer`` that addresses its columns *by name*. Handing the converter a
    single ``FloatTensorType([None, n])`` would type the categorical columns as floats and
    convert a graph whose one-hot block encodes numbers that were never in the data.
    """
    types_mod = require("aegis-ml[mlops]", "skl2onnx.common.data_types")
    categorical = set(problem.categorical_features)
    initial: list[tuple[str, Any]] = []
    for name in problem.feature_names:
        if name in categorical:
            initial.append((name, types_mod.StringTensorType([None, 1])))
        else:
            initial.append((name, types_mod.FloatTensorType([None, 1])))
    return initial


def _sklearn_core(model: Any) -> Any:  # noqa: ANN401 - accepts a fitted estimator or wrapper
    """Return the sklearn estimator to convert, unwrapping one documented layer.

    ``aegis.ml.model.TrustworthyModel`` is not itself an sklearn estimator: it *holds* the
    fitted pipeline alongside the MAPIE wrapper and the SHAP explainers. Unwrapping exactly
    one named attribute makes the common call site work while keeping the refusal loud for
    anything else — guessing deeper would eventually pick the MAPIE wrapper and export a
    graph the caller believes carries intervals.
    """
    if hasattr(model, "predict"):
        return model
    for attribute in ("pipeline", "estimator", "model"):
        inner = getattr(model, attribute, None)
        if inner is not None and hasattr(inner, "predict"):
            return inner
    raise TypeError(
        f"{type(model).__name__} is not an sklearn estimator and exposes no .pipeline / "
        f".estimator / .model that is. Pass the fitted point predictor — for example the "
        f"Pipeline returned by aegis_ml.automl.recipe.fit_recipe."
    )


def _member_kinds(estimator: Any) -> list[str]:  # noqa: ANN401
    """Return the class names inside a fitted pipeline's voting ensemble, best effort.

    Used only to make a conversion failure legible: "skl2onnx refused this model" is a
    dead end, while "…and its members are XGBRegressor, LGBMRegressor" points straight at
    the missing converter registration.
    """
    kinds: list[str] = []
    steps = getattr(estimator, "named_steps", {})
    inner = steps.get("estimator", estimator) if steps else estimator
    for _name, member in getattr(inner, "estimators", []) or []:
        kinds.append(type(member).__name__)
    return kinds or [type(inner).__name__]


def to_onnx(model: Any, problem: MLProblem, path: str | Path) -> Path:  # noqa: ANN401
    """Convert a fitted point predictor to ONNX and write it to ``path``.

    Args:
        model: The fitted sklearn pipeline — typically
            :func:`aegis_ml.automl.recipe.fit_recipe`'s return value. A
            ``TrustworthyModel``-style wrapper holding one is unwrapped; see
            :func:`_sklearn_core`.
        problem: The spec, which supplies the input column names and which of them are
            strings. It must be the same spec the model was fitted against.
        path: Destination ``.onnx`` file; parent directories are created.

    Returns:
        The path written.

    Raises:
        ImportError: If skl2onnx/onnxmltools are absent (message names the install).
        TypeError: If ``model`` is not, and does not hold, an sklearn estimator.
        AegisMLError: If the converter produced no graph, or refused one of the members —
            in which case the message names the members and the two known causes.
    """
    skl2onnx = require("aegis-ml[mlops]", "skl2onnx")
    registered = register_converters()
    estimator = _sklearn_core(model)

    options: dict[Any, Any] = {}
    if problem.target.task == "classification":
        # zipmap=False makes the probability output a plain (n, n_classes) tensor instead of
        # a sequence of dicts. Runtimes outside Python handle the tensor; several cannot
        # represent the ZipMap output at all, and the round-trip check cannot compare it.
        options[id(estimator)] = {"zipmap": False}

    try:
        onnx_model = skl2onnx.convert_sklearn(
            estimator,
            initial_types=_initial_types(problem),
            options=options or None,
        )
    except Exception as exc:  # audit-ok: re-raised with context, chained, never swallowed
        raise AegisMLError(
            f"skl2onnx could not convert this model ({type(exc).__name__}: "
            f"{str(exc)[:400]}).\nMembers: {_member_kinds(estimator)}. Third-party "
            f"converters registered: {registered or 'none'}.\nThe two causes worth checking "
            f"first: (1) a member with no registered converter — XGBoost and LightGBM need "
            f"onnxmltools, which `aegis-ml[mlops]` provides; (2) a version skew between "
            f"skl2onnx, onnx and scikit-learn, which shows up as a TypeError deep inside "
            f"the TreeEnsemble attributes rather than as an unsupported-model message. "
            f"The fitted model is unaffected — only the export failed."
        ) from exc
    if onnx_model is None:
        raise AegisMLError(
            "skl2onnx returned no model. Every ensemble member must have a registered "
            f"converter; register_converters() wired up {registered or 'none'} and this "
            f"model's members are {_member_kinds(estimator)}."
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(onnx_model.SerializeToString())
    return target


def _feed_from_frame(session: Any, frame: pd.DataFrame) -> dict[str, Any]:  # noqa: ANN401
    """Build the runtime feed dict, typed from the graph's own declared inputs.

    Types are read off the session rather than re-derived from the spec: if the two ever
    disagree, the graph is the thing being tested and re-deriving would hide the
    disagreement behind a matching cast.
    """
    np_mod = require("aegis-ml[serve]", "numpy")
    feed: dict[str, Any] = {}
    for spec in session.get_inputs():
        column = frame[spec.name]
        if "string" in spec.type:
            values = column.astype(str).to_numpy().reshape(-1, 1)
        else:
            values = column.to_numpy(dtype=np_mod.float32).reshape(-1, 1)
        feed[spec.name] = values
    return feed


def validate_roundtrip(
    model: Any,  # noqa: ANN401
    onnx_path: str | Path,
    sample_frame: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Score ``sample_frame`` both ways and return the largest disagreement.

    For a regression model this compares predicted values. For a classification model it
    compares predicted *probabilities* when the model exposes them — a label-only
    comparison passes trivially on an easy frame and would miss a converter that is
    systematically shifting the probabilities, which is what an ONNX consumer thresholds.

    Args:
        model: The fitted sklearn model the ONNX file was exported from.
        onnx_path: The exported graph.
        sample_frame: Real rows to compare on — use held-out rows, not the training head:
            a converter bug that only shows up on unseen categorical levels is exactly the
            bug ``handle_unknown="ignore"`` makes silent.
        tolerance: Largest tolerated difference; see :data:`DEFAULT_TOLERANCE`.

    Returns:
        The maximum absolute difference observed.

    Raises:
        AegisMLError: If the difference exceeds ``tolerance``. The export is left on disk
            so it can be inspected, but the caller must not publish it as equivalent.
    """
    ort = require("aegis-ml[mlops]", "onnxruntime")
    np_mod = require("aegis-ml[serve]", "numpy")
    estimator = _sklearn_core(model)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, _feed_from_frame(session, sample_frame))

    proba_fn = getattr(estimator, "predict_proba", None)
    if callable(proba_fn) and len(outputs) > 1:
        expected = np_mod.asarray(proba_fn(sample_frame), dtype=float)
        actual = np_mod.asarray(outputs[1], dtype=float)
    else:
        expected = np_mod.asarray(estimator.predict(sample_frame), dtype=float).ravel()
        actual = np_mod.asarray(outputs[0], dtype=float).ravel()

    if expected.shape != actual.shape:
        raise AegisMLError(
            f"ONNX round-trip shape mismatch: sklearn produced {expected.shape}, the graph "
            f"produced {actual.shape}. The export does not compute the same function and "
            f"must not be published as equivalent."
        )

    difference = float(np_mod.max(np_mod.abs(expected - actual)))
    if difference > tolerance:
        nan_columns = sorted(
            column for column in sample_frame.columns if bool(sample_frame[column].isna().any())
        )
        nan_hint = (
            f"\nFIRST SUSPECT: {nan_columns} contain missing values in this sample. "
            f"scikit-learn's tree learners route NaN down a direction they LEARNED; the "
            f"ONNX TreeEnsemble op does not reproduce that routing faithfully for every "
            f"member type, so a model fitted on frames with MAR missingness can convert "
            f"without error and then disagree by whole units on exactly the rows that are "
            f"missing a value. Impute before the model if the export must be faithful."
            if nan_columns
            else ""
        )
        raise AegisMLError(
            f"ONNX round-trip differs by {difference:.6g}, above the {tolerance:.6g} "
            f"tolerance. Float32 rounding accounts for ~1e-6; a gap this size means the "
            f"graph does not compute the same function.{nan_hint}\nThe file at "
            f"{onnx_path} must NOT be published as equivalent to the fitted model."
        )
    return difference
