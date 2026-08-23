"""Two preprocessing paths, and a hard rule about which one may not change.

## The portable one

:func:`column_transformer` returns, byte for byte in behaviour, what
``aegis.ml.model.TrustworthyModel._build_preprocessor`` returns: ``OneHotEncoder(
handle_unknown="ignore", sparse_output=False)`` over the declared categoricals, passthrough
over everything else, ``remainder="drop"``, ``verbose_feature_names_out=False``.

That identity is load-bearing, in two distinct ways.

**The recipe crosses a venv boundary.** The AutoML search runs in the trainer venv and
returns JSON; the Aegis spine re-fits it in the serving venv *using its own preprocessor*.
If the search ranked candidates under a richer encoding than the one the spine will apply,
the winning candidate is the winner of a different competition, and the leaderboard number
quoted on the model card stops describing the model actually being served.

**SHAP aggregation assumes this exact layout.** ``aegis.ml.model._encoded_parents``
reconstructs which encoded column belongs to which original feature by walking the
categorical blocks in declared order and then the numeric passthroughs — and it raises if
its reconstruction does not match the emitted column count. Change the transformer order,
add a scaler, or set ``verbose_feature_names_out=True`` and either the attribution silently
maps to the wrong parent or the whole fit fails. So this function is a mirror, and it stays
a mirror.

## The richer one

:func:`skrub_pipeline` uses skrub's ``TableVectorizer``, which is genuinely better at the
things a hand-declared spec is bad at: high-cardinality strings (its default encoder is now
``StringEncoder``, replacing ``GapEncoder``), datetime expansion into calendar components,
and dtype inference on columns nobody declared. That is the right tool for the AutoML-side
exploration — *what could this data support?* — and the wrong tool for the promoted spine,
because none of it survives the crossing into the serving venv.

Both are exposed on purpose, and every result that came from the second one must say so.
A score obtained under ``TableVectorizer`` is an **accuracy ceiling**, in exactly the sense
``Candidate.portable=False`` already means elsewhere in this package — never a number to
promote as the spine's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    from aegis_ml.contracts.spec import MLProblem

__all__ = [
    "cast_declared_categoricals",
    "column_transformer",
    "encode_frame",
    "skrub_pipeline",
]

_SERVE_EXTRA = "aegis-ml[serve]"
"""Install target quoted when scikit-learn, pandas or skrub are missing."""


def _sklearn(submodule: str) -> ModuleType:
    """Import a scikit-learn submodule through :func:`~aegis_ml._require.require`."""
    return require(_SERVE_EXTRA, f"sklearn.{submodule}")


def _pandas() -> ModuleType:
    """Import pandas through :func:`~aegis_ml._require.require`."""
    return require(_SERVE_EXTRA, "pandas")


def column_transformer(problem: MLProblem) -> ColumnTransformer:
    """Build the **portable** preprocessor — the one the Aegis spine will actually apply.

    Every argument below is copied from ``aegis.ml.model._build_preprocessor`` and must not
    drift from it; see this module's docstring for the two failures that drift causes.

    ``handle_unknown="ignore"`` deserves its own note, because it is the reason
    :mod:`aegis_ml.contracts.frames` exists. An unseen categorical level does not raise
    here — it encodes to an all-zero block, and the row is scored as though the feature were
    absent. That is correct at *inference* time (a live request naming a new carrier should
    still get an answer) and silently wrong at *training* time, which is why the pandera
    contract checks ``Check.isin(levels)`` at the boundary instead.

    Args:
        problem: The declarative problem; its ``categorical_features`` and
            ``numeric_features`` properties define the two blocks.

    Returns:
        An unfitted ``ColumnTransformer`` producing a dense array whose columns are the
        one-hot blocks in declared order followed by the numeric passthroughs.

    Raises:
        ImportError: When scikit-learn is not installed, naming the install command.
    """
    compose = _sklearn("compose")
    preprocessing = _sklearn("preprocessing")
    return compose.ColumnTransformer(
        transformers=[
            (
                "cat",
                preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(problem.categorical_features),
            ),
            ("num", "passthrough", list(problem.numeric_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encode_frame(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    transformer: ColumnTransformer | None = None,
) -> pd.DataFrame:
    """Fit-and-apply the portable preprocessor, returning a named DataFrame.

    This is the encoding every *measurement* in this package runs through — the learnability
    probe, the leakage detector, the slice metrics — so that all of them are measuring the
    representation the spine will actually see rather than a more generous one of their own.

    One deliberate departure, stated loudly because it is the kind of thing that becomes a
    silent bug six weeks later: **datetime columns are converted to epoch seconds here, and
    the Aegis spine does not do that.** ``MLProblem.numeric_features`` is everything that is
    not categorical, mirroring ``ResolvedSpec``, so a declared ``datetime`` feature is
    handed to ``"passthrough"`` and arrives at the estimator as ``datetime64[ns]``, which
    tree learners reject outright. Converting it here means a measurement can still be
    taken; it does **not** mean the spine will train. A declared datetime feature is a
    spec-level bug, and :func:`aegis_ml.data.contract_check.check` flags it as one.

    Args:
        frame: The raw frame. Extra columns are dropped by the transformer.
        problem: The problem defining the two column blocks.
        transformer: An already-fitted transformer to reuse. Pass one when encoding a test
            split, so both splits share a column layout; leave ``None`` to fit fresh.

    Returns:
        A dense DataFrame indexed like ``frame``, with the encoder's own column names.

    Raises:
        AegisMLError: When a declared feature column is missing from the frame.
    """
    pd = _pandas()
    missing = [name for name in problem.feature_names if name not in frame.columns]
    if missing:
        raise AegisMLError(
            f"Frame is missing declared feature columns {missing}. Encoding would proceed "
            f"with those features silently absent, and every score measured afterwards "
            f"would describe a different model from the one the spec declares."
        )
    prepared = frame.copy()
    for feature in problem.features:
        if feature.dtype == "datetime":
            prepared[feature.name] = (
                pd.to_datetime(prepared[feature.name], errors="coerce").astype("int64") / 1e9
            )
        elif feature.dtype == "boolean":
            prepared[feature.name] = pd.to_numeric(prepared[feature.name], errors="coerce")
    if transformer is None:
        transformer = column_transformer(problem)
        matrix = transformer.fit_transform(prepared)
    else:
        matrix = transformer.transform(prepared)
    names = list(transformer.get_feature_names_out())
    return pd.DataFrame(matrix, columns=names, index=frame.index).astype("float64")


def skrub_pipeline(
    problem: MLProblem,
    *,
    estimator: Any = None,  # noqa: ANN401 - any sklearn-compatible estimator
    high_cardinality_threshold: int = 40,
) -> Pipeline:
    """Build the **exploratory** preprocessor: skrub's ``TableVectorizer``.

    What this buys over :func:`column_transformer`: string columns too high-cardinality to
    one-hot are embedded rather than exploded, datetimes are expanded into calendar
    components instead of being passed through as an unusable dtype, and columns absent from
    the spec are still handled sensibly. On a real client CSV that is often several points
    of accuracy.

    What it costs: none of it crosses the venv boundary. The Aegis spine re-fits a recipe
    with *its own* ``ColumnTransformer``, so a candidate tuned against a ``TableVectorizer``
    representation cannot be promoted as the spine and its score must be reported the way
    this package reports every non-portable result — as an accuracy ceiling, with
    ``Candidate.portable=False``, next to the portable runner-up that will actually serve.

    ``problem`` is still consumed rather than ignored: the categorical columns it declares
    are handed to skrub explicitly instead of being inferred, so a coded categorical stored
    as an integer is not silently treated as a quantity.

    Args:
        problem: The declarative problem.
        estimator: Optional final step. Without one the pipeline is a transformer, which is
            what a feature-importance or clustering pass wants.
        high_cardinality_threshold: Level count above which skrub switches from one-hot to
            its string encoder.

    Returns:
        An unfitted sklearn ``Pipeline``.

    Raises:
        ImportError: When skrub or scikit-learn are not installed, naming the install.
    """
    skrub = require(_SERVE_EXTRA, "skrub")
    pipeline_module = _sklearn("pipeline")
    preprocessing = _sklearn("preprocessing")
    declare = preprocessing.FunctionTransformer(
        cast_declared_categoricals,
        kw_args={"categorical": tuple(problem.categorical_features)},
        validate=False,
    )
    vectorizer = skrub.TableVectorizer(cardinality_threshold=high_cardinality_threshold)
    steps: list[tuple[str, Any]] = [("declare", declare), ("vectorizer", vectorizer)]
    if estimator is not None:
        steps.append(("estimator", estimator))
    return pipeline_module.Pipeline(steps)


def cast_declared_categoricals(
    frame: pd.DataFrame, *, categorical: tuple[str, ...]
) -> pd.DataFrame:
    """Cast the spec's categorical columns to strings before skrub infers dtypes.

    Defined at module level rather than as a closure so the pipeline stays picklable — a
    ``TableVectorizer`` pipeline that cannot be joblib-dumped cannot be handed back across
    the trainer/serving venv boundary at all, and discovering that at save time is late.

    The cast itself prevents skrub's inference from making the one mistake a declared spec
    exists to rule out: a categorical stored as ``1``/``2``/``3`` looks numeric to any
    inference pass, and treating a coded category as a quantity imposes an ordering on it
    that the domain never claimed.

    Args:
        frame: The raw frame.
        categorical: Column names the spec declares categorical.

    Returns:
        A copy with those columns as nullable strings; nulls stay null so skrub's own
        missing-value handling still applies.
    """
    result = frame.copy()
    for column in categorical:
        if column in result.columns:
            mask = result[column].isna()
            result[column] = result[column].astype("object").astype(str)
            result.loc[mask, column] = None
    return result
