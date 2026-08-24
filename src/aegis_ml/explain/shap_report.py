"""SHAP reporting — global importance, local attribution, and a self-contained HTML page.

The Aegis spine already computes SHAP: ``aegis.ml`` attributes each prediction across its
soft-voting members and averages the per-member attributions **by member weight**, returning
them as ``list[ShapFeature]`` on every ``MLExplainResponse``. That is the serving path and
this module does not duplicate or replace it.

What this module adds is the *reporting* layer over the same idea, for two consumers the
serving path does not have: the **registry** (a global picture stored next to the model, so a
later run can be compared against it) and the **demo** (a page a human opens). The shapes are
kept deliberately compatible — :func:`local_explanation` returns exactly the fields of
``aegis.ml.types.ShapFeature`` (``feature``, ``value``, ``value_label``, ``contribution``),
including the ``value``/``value_label`` split, so a categorical driver renders as
``region = emea`` and never as ``region = 1.0``.

Two decisions worth stating, because both are easy to get wrong in the direction that looks
better:

* **Nothing is filtered out of the global importance.** A feature the model ignored appears
  with an importance of essentially zero, and it stays in the table and in the chart. On a
  dataset that deliberately contains irrelevant features, a near-zero bar is the clearest
  available evidence that the model is not chasing noise — dropping those rows would remove
  the finding and leave a report that looks equally good for an honest model and an
  overfitted one.
* **The explainer is model-agnostic on purpose, and that is now a choice rather than a
  limit.** Algorithm selection lives in :mod:`aegis_ml.explain.explainers`, which routes tree
  models to ``TreeExplainer``, linear models to ``LinearExplainer`` and everything else to
  ``PermutationExplainer`` — so *any* fitted estimator can be reported on here, not only the
  tree learners. This module hands that dispatch a wrapped ``predict`` over the raw declared
  columns rather than the bare estimator, which pins it to the permutation branch by
  construction. That is deliberate: ``TreeExplainer`` is far faster, but it cannot see
  through the encoding pipeline the estimator is wrapped in, and attributions computed on
  one-hot columns cannot be summed back to the original feature without an assumption. Every
  attribution here is stated in terms of the columns the domain spec actually declares.

``shap`` is an optional dependency, imported through
:func:`aegis_ml._require.require` so an absent install raises with the exact command that
fixes it. There is no fallback path: a report produced without SHAP and a report produced
with it must never be indistinguishable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml.contracts.spec import MLProblem
from aegis_ml.explain.card import escape, html_page, svg_bar_chart
from aegis_ml.explain.explainers import ExplainerUnavailableError, build_explainer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    import pandas as pd

__all__ = [
    "ExplainerUnavailableError",
    "global_importance",
    "local_explanation",
    "render_html",
]


class _FrameCodec:
    """Round-trip a domain frame through the numeric matrix SHAP's masker requires.

    This exists because of a hard constraint, not a preference. ``shap.maskers.Independent``
    decides which cells a masked row actually changed with ``numpy.isclose``, which is
    arithmetic — hand it a column of strings and it raises ``unsupported operand type(s) for
    -: 'str' and 'str'``. So the frame is encoded to floats before it reaches the masker, and
    decoded back to real levels before it reaches the model.

    The encoding is a per-column codebook, **not** a one-hot expansion, and that is the
    load-bearing choice: masking a code substitutes another row's level wholesale, exactly as
    masking a real categorical should. One-hot columns would let the masker produce rows that
    are two levels at once, or none — combinations the model was never fitted on — and the
    attributions would then have to be summed back to the original feature under an
    assumption nobody stated.

    Attributes:
        columns: Feature columns in declaration order.
        codebooks: Column → ordered level list, for the columns that are coded.
    """

    def __init__(self, template: pd.DataFrame, problem: MLProblem) -> None:
        """Build the codebooks from the template frame and the declared spec.

        Args:
            template: A frame carrying the columns, dtypes and observed levels.
            problem: The declared problem; its categorical features are always coded, and so
                is any column whose data is non-numeric regardless of what the spec says —
                a mismatch there is a real defect, and crashing inside numpy would report it
                far from its cause.
        """
        import pandas as pd

        self.columns = list(problem.feature_names)
        self._dtypes = {name: template[name].dtype for name in self.columns}
        declared = set(problem.categorical_features)
        self.codebooks: dict[str, list[str]] = {}
        for name in self.columns:
            column = template[name]
            numeric = pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(
                column
            )
            if name in declared or not numeric:
                spec = next((f for f in problem.features if f.name == name), None)
                levels = [str(level) for level in (spec.levels if spec else [])]
                observed = [str(value) for value in column.dropna().unique()]
                self.codebooks[name] = list(dict.fromkeys([*levels, *observed]))

    def encode(self, frame: pd.DataFrame) -> Any:  # noqa: ANN401 - ndarray, numpy imported inside
        """Encode a domain frame to the float matrix the masker can mask.

        Args:
            frame: A frame carrying every declared feature column.

        Returns:
            A ``(n_rows, n_features)`` float array. Unknown levels and nulls encode to NaN,
            so an unseen level stays visibly missing rather than colliding with level 0.
        """
        import numpy as np

        out = np.empty((len(frame), len(self.columns)), dtype=float)
        for index, name in enumerate(self.columns):
            column = frame[name]
            if name in self.codebooks:
                lookup = {level: float(i) for i, level in enumerate(self.codebooks[name])}
                out[:, index] = [
                    lookup.get(str(value), float("nan")) if value == value else float("nan")
                    for value in column.to_numpy()
                ]
            else:
                out[:, index] = column.to_numpy(dtype=float, na_value=float("nan"))
        return out

    def decode(self, array: Any) -> pd.DataFrame:  # noqa: ANN401 - ndarray from shap
        """Decode a float matrix back into the frame the fitted model expects.

        Args:
            array: A ``(n_rows, n_features)`` float array from the masker.

        Returns:
            A frame whose coded columns carry real level strings again and whose integer and
            boolean columns get their dtype back when no null crept in — a ``ColumnTransformer``
            that selects columns by dtype would otherwise route them to the wrong branch.
        """
        import numpy as np
        import pandas as pd

        values = np.asarray(array, dtype=float).reshape(-1, len(self.columns))
        data: dict[str, Any] = {}
        for index, name in enumerate(self.columns):
            column = values[:, index]
            if name in self.codebooks:
                levels = self.codebooks[name]
                data[name] = [
                    levels[int(code)]
                    if np.isfinite(code) and 0 <= int(code) < len(levels)
                    else None
                    for code in column
                ]
            else:
                data[name] = column
        frame = pd.DataFrame(data, columns=self.columns)
        for name, dtype in self._dtypes.items():
            if name in self.codebooks:
                continue
            if dtype.kind in {"b", "i", "u"} and not frame[name].isna().any():
                frame[name] = frame[name].round().astype(dtype)
        return frame


def _predict_function(model: object, problem: MLProblem, codec: _FrameCodec) -> Any:  # noqa: ANN401
    """Wrap the fitted model so SHAP can call it with the encoded matrix.

    Args:
        model: The fitted estimator or pipeline.
        problem: The declared problem; decides whether probabilities or values are explained.
        codec: The codec that turns the masker's float rows back into a domain frame.

    Returns:
        A callable ``(array) -> ndarray`` suitable for ``shap.Explainer``.

    Raises:
        ExplainerUnavailableError: When a classifier exposes no ``predict_proba``. A hard
            0/1 output has no gradient for SHAP to attribute, and explaining the encoded
            class index would produce numbers whose sign means nothing.
    """
    classification = problem.target.task == "classification"
    if classification and not hasattr(model, "predict_proba"):
        raise ExplainerUnavailableError(
            "the classifier exposes no predict_proba, so there is no continuous output to "
            "attribute; explaining the hard class index would produce signed numbers whose "
            "direction is arbitrary. Wrap the estimator in CalibratedClassifierCV, or use "
            "an estimator that reports probabilities"
        )

    def predict(array: object) -> object:
        """Decode SHAP's masked rows into a domain frame and run the model on them."""
        frame = codec.decode(array)
        if classification:
            return model.predict_proba(frame)  # type: ignore[attr-defined]
        return model.predict(frame)  # type: ignore[attr-defined]

    return predict


def _explanation(
    model: object,
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    max_samples: int,
    background_samples: int,
    seed: int | None,
    n_permutations: int = 4,
    progress: bool = False,
) -> Any:  # noqa: ANN401 - a shap.Explanation, typed only when shap is installed
    """Compute a SHAP ``Explanation`` for a sample of rows.

    Cost is the reason every size here is a parameter. One permutation round costs
    ``2 * n_features + 1`` model evaluations per explained row, each expanded across the whole
    background — so the total is ``rows × rounds × (2f+1) × background`` model rows.
    shap's own default of ``max_evals=500`` silently buys ~45 rounds on a five-feature
    problem, which is roughly an order of magnitude more precision than a report needs and
    turns a 500-row explanation into several minutes. ``n_permutations`` makes that trade
    explicit instead of expensive by default.

    Args:
        model: The fitted estimator or pipeline.
        frame: Rows to draw the sample and the background from; feature columns only are used.
        problem: The declared problem.
        max_samples: How many rows to explain.
        background_samples: How many rows form the masking distribution. The background is
            what "absent" means to SHAP — attributions are stated *relative to* it, so it
            must come from the same population as the explained rows.
        seed: Sampling seed, so a re-run of the demo produces the same figures.
        n_permutations: Permutation rounds per row. More rounds reduce the Monte-Carlo noise
            on each attribution at a proportional cost; the ranking is stable well before the
            individual values stop moving.
        progress: Show shap's progress bar. Off by default because these reports are written
            from inside a pipeline whose logs are read afterwards, where a redrawn bar is
            thousands of lines of noise around the one line that matters.

    Returns:
        The ``shap.Explanation``.

    Raises:
        ExplainerUnavailableError: When the frame is empty or lacks the declared features.
    """
    missing = [name for name in problem.feature_names if name not in frame.columns]
    if missing:
        raise ExplainerUnavailableError(
            f"the frame is missing declared feature columns {missing}, so any attribution "
            f"would describe a different model than the spec declares"
        )
    features = frame[problem.feature_names]
    if len(features) == 0:
        raise ExplainerUnavailableError("the frame has no rows to explain")

    n_background = max(1, min(background_samples, len(features)))
    n_explain = max(1, min(max_samples, len(features)))
    background = features.sample(n=n_background, random_state=seed)
    sample = features.sample(n=n_explain, random_state=seed)

    codec = _FrameCodec(features, problem)
    predict = _predict_function(model, problem, codec)
    explainer, _kind = build_explainer(
        predict, codec.encode(background), feature_names=list(problem.feature_names)
    )
    per_round = 2 * len(problem.feature_names) + 1
    return explainer(
        codec.encode(sample),
        max_evals=max(per_round, n_permutations * per_round),
        silent=not progress,
    )


def global_importance(
    model: object,
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    max_samples: int = 500,
    background_samples: int = 50,
    seed: int | None = None,
    n_permutations: int = 4,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Mean absolute SHAP value per declared feature, strongest first — all of them.

    ``mean_abs_shap`` measures how much the feature moved predictions in either direction;
    ``mean_shap`` keeps the sign, which says whether the feature pushed the model's output up
    or down *on average across the sample*. Both are reported because they answer different
    questions and one without the other misleads: a feature that pushes half the rows up and
    half down has a large ``mean_abs_shap`` and a ``mean_shap`` near zero, and that is a real
    interaction rather than a weak feature.

    Every declared feature appears in the output, including features whose importance rounds
    to zero. On a dataset that deliberately contains irrelevant columns, those rows are the
    evidence that the model ignored them.

    Args:
        model: The fitted estimator or pipeline, accepting the raw domain frame.
        frame: Rows to explain — normally the held-out split, so the report describes
            behaviour on data the model was not fitted on.
        problem: The declared problem.
        max_samples: How many rows to explain. See :func:`_explanation` for the cost.
        background_samples: Rows forming SHAP's masking distribution.
        seed: Sampling seed for reproducibility.
        n_permutations: Permutation rounds per row; the accuracy/cost dial.
        progress: Show shap's progress bar (off by default — see :func:`_explanation`).

    Returns:
        One dict per feature: ``feature``, ``mean_abs_shap``, ``mean_shap``, ``share``
        (of total absolute attribution) and ``n_samples``. Sorted by ``mean_abs_shap``
        descending.

    Raises:
        ImportError: When ``shap`` is not installed, naming the install command.
        ExplainerUnavailableError: When the model or frame cannot be explained.
    """
    import numpy as np

    explanation = _explanation(
        model,
        frame,
        problem,
        max_samples=max_samples,
        background_samples=background_samples,
        seed=seed,
        n_permutations=n_permutations,
        progress=progress,
    )
    values = np.asarray(explanation.values)
    # Multiclass explanations carry a trailing class axis: average the magnitude over
    # classes so one row per feature is reported, and keep the signed mean over the
    # positive-class column only where a signed direction is meaningful (binary).
    if values.ndim == 3:
        abs_per_feature = np.abs(values).mean(axis=(0, 2))
        signed_per_feature = values[:, :, -1].mean(axis=0)
    else:
        abs_per_feature = np.abs(values).mean(axis=0)
        signed_per_feature = values.mean(axis=0)

    names = list(problem.feature_names)
    total = float(abs_per_feature.sum()) or 1.0
    rows = [
        {
            "feature": name,
            "mean_abs_shap": float(abs_per_feature[index]),
            "mean_shap": float(signed_per_feature[index]),
            "share": float(abs_per_feature[index]) / total,
            "n_samples": int(values.shape[0]),
        }
        for index, name in enumerate(names)
    ]
    rows.sort(key=lambda row: row["mean_abs_shap"], reverse=True)
    return rows


def local_explanation(
    model: object,
    row: Mapping[str, Any] | pd.Series | pd.DataFrame,
    problem: MLProblem,
    *,
    background: pd.DataFrame | None = None,
    background_samples: int = 50,
    seed: int | None = None,
    n_permutations: int = 8,
) -> list[dict[str, Any]]:
    """Explain one prediction, in the field shape Aegis's ``ShapFeature`` uses.

    The ``value`` / ``value_label`` split is carried over from
    ``aegis.ml.types.ShapFeature`` and is not cosmetic. For a categorical feature the number
    the model saw is the one-hot indicator ``1.0``, which names no level on its own; the
    level goes in ``value_label`` so a renderer can write ``region = emea``. Collapsing the
    two produces ``region = 1.0`` in a sentence a human is supposed to act on.

    Args:
        model: The fitted estimator or pipeline.
        row: The single row to explain, as a mapping, Series or 1-row frame.
        problem: The declared problem.
        background: Rows forming the masking distribution. **Required in practice**: SHAP
            attributions are stated relative to a background, and a background drawn from a
            different population changes every number. Pass the training frame.
        background_samples: How many background rows to sample.
        seed: Sampling seed.
        n_permutations: Permutation rounds. Higher than the global default because this is a
            single row: the whole cost is one row's worth, and the number is quoted verbatim
            in a sentence a human acts on, so its Monte-Carlo noise is worth paying to remove.

    Returns:
        One dict per feature with ``feature``, ``value``, ``value_label`` and
        ``contribution``, sorted by absolute contribution descending — the same ordering
        ``describe_prediction`` walks when it names the top drivers.

    Raises:
        ImportError: When ``shap`` is not installed.
        ExplainerUnavailableError: When no background frame is supplied, or the model cannot
            be explained.
    """
    import numpy as np
    import pandas as pd

    if background is None:
        raise ExplainerUnavailableError(
            "no background frame was supplied. A SHAP contribution is always stated "
            "relative to a reference distribution — 'compared to what' is part of the "
            "answer — so there is no default that would not silently change every number"
        )
    if isinstance(row, pd.DataFrame):
        single = row.iloc[[0]]
    elif isinstance(row, pd.Series):
        single = row.to_frame().T
    else:
        single = pd.DataFrame([dict(row)])

    missing = [name for name in problem.feature_names if name not in single.columns]
    if missing:
        raise ExplainerUnavailableError(
            f"the row is missing declared features {missing}; an attribution computed "
            f"without them describes a different model"
        )
    reference = background[problem.feature_names]
    single = single[problem.feature_names].astype(reference.dtypes.to_dict())

    # The row is explained directly rather than through `_explanation`, which samples: the
    # caller asked about THIS row, and an attribution computed for a row that merely
    # resembles it would be presented as if it were the same answer.
    sampled = reference.sample(
        n=max(1, min(background_samples, len(reference))), random_state=seed
    )
    codec = _FrameCodec(reference, problem)
    predict = _predict_function(model, problem, codec)
    explainer, _kind = build_explainer(
        predict, codec.encode(sampled), feature_names=list(problem.feature_names)
    )
    per_round = 2 * len(problem.feature_names) + 1
    explanation = explainer(
        codec.encode(single),
        max_evals=max(per_round, n_permutations * per_round),
        silent=True,
    )

    values = np.asarray(explanation.values)
    # A 3-D explanation is (rows, features, classes); the last class column is the positive
    # class in sklearn's usual ordering, and it is the one the prediction is read against.
    contributions = values[0, :, -1] if values.ndim == 3 else values[0]

    categorical = set(problem.categorical_features)
    out: list[dict[str, Any]] = []
    for index, name in enumerate(problem.feature_names):
        raw = single.iloc[0][name]
        if name in categorical:
            numeric_value = 1.0
            label: str | None = None if raw is None else str(raw)
        else:
            try:
                numeric_value = float(raw)
                label = None
            except (TypeError, ValueError):
                # A non-numeric value in a column the spec calls numeric is a real finding:
                # report the level rather than inventing a number for it.
                numeric_value = 0.0
                label = str(raw)
        out.append(
            {
                "feature": name,
                "value": numeric_value,
                "value_label": label,
                "contribution": float(contributions[index]),
            }
        )
    out.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return out


def render_html(
    path: str | Path,
    *,
    importance: Sequence[dict[str, Any]],
    problem: MLProblem,
    local: Sequence[dict[str, Any]] | None = None,
    local_prediction: float | str | None = None,
    title: str | None = None,
    notes: Sequence[str] | None = None,
) -> Path:
    """Write a self-contained SHAP report — inline CSS and inline SVG, no network.

    The page is opened from a filesystem path, attached to a ticket, read offline. Anything
    fetched from a CDN is a section that renders blank in exactly the room where the report
    matters, so there is nothing to fetch.

    Args:
        path: Destination ``.html`` file. Parent directories are created.
        importance: Rows from :func:`global_importance`. Passed in rather than recomputed so
            the page shows the same numbers the model card was built from — a page that
            recomputed its own would eventually disagree with the card beside it.
        problem: The declared problem, for the target's name and unit.
        local: Optional rows from :func:`local_explanation`, rendered as a signed chart.
        local_prediction: The prediction the local explanation belongs to, if any.
        title: Page title; defaults to naming the domain and target.
        notes: Extra sentences to print under the charts.

    Returns:
        The path written.

    Raises:
        ValueError: When ``importance`` is empty — an importance report with no rows is not
            a report, and writing an empty page would look like a successful run.
    """
    rows = list(importance)
    if not rows:
        raise ValueError(
            "render_html was given no importance rows. An empty report is indistinguishable "
            "from a successful one at a glance; compute global_importance first, or do not "
            "write the file."
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    unit = f" {problem.target.unit}" if problem.target.unit else ""
    heading = title or f"SHAP report — {problem.domain_id} / {problem.target.name}"

    body = [
        "<h2>Global importance</h2>",
        "<p class='sub'>Mean absolute SHAP value per feature across the explained sample. "
        "Every declared feature is listed, including those the model effectively ignored: a "
        "near-zero bar is evidence that the model is not chasing an irrelevant column, and "
        "removing those rows would hide it.</p>",
        "<div class='scroll'>",
        svg_bar_chart(
            [(str(row["feature"]), float(row["mean_abs_shap"])) for row in rows],
            value_format="{:.4f}",
        ),
        "</div>",
        "<div class='scroll'><table><tr><th>Feature</th><th class='num'>mean |SHAP|</th>"
        "<th class='num'>mean SHAP (signed)</th><th class='num'>Share</th></tr>",
    ]
    for row in rows:
        body.append(
            f"<tr><td>{escape(row['feature'])}</td>"
            f"<td class='num'>{float(row['mean_abs_shap']):.4f}</td>"
            f"<td class='num'>{float(row.get('mean_shap', 0.0)):+.4f}</td>"
            f"<td class='num'>{float(row.get('share', 0.0)):.1%}</td></tr>"
        )
    body.append("</table></div>")
    body.append(
        "<p class='sub'>A large mean |SHAP| with a mean signed SHAP near zero is not a weak "
        "feature — it is a feature that pushes some rows up and others down, which is an "
        "interaction the single-number view cannot show.</p>"
    )

    if local:
        predicted = ""
        if local_prediction is not None:
            rendered = (
                f"{local_prediction:.3f}{unit}"
                if isinstance(local_prediction, int | float)
                else str(local_prediction)
            )
            predicted = f"<p>Prediction: <strong>{escape(rendered)}</strong></p>"
        body.extend(
            [
                "<h2>One prediction</h2>",
                predicted,
                "<p class='sub'>Signed contributions for a single row, relative to the "
                "background distribution. Positive pushes the model's output up, negative "
                "pushes it down.</p>",
                "<div class='scroll'>",
                svg_bar_chart(
                    [
                        (
                            f"{item['feature']} = "
                            f"{item.get('value_label') or item.get('value')}",
                            float(item["contribution"]),
                        )
                        for item in local
                    ],
                    signed=True,
                ),
                "</div>",
            ]
        )

    if notes:
        body.append("<h2>Notes</h2><ul>")
        body.extend(f"<li>{escape(note)}</li>" for note in notes)
        body.append("</ul>")

    document = html_page(
        heading,
        "".join(body),
        subtitle=(
            f"{escape(problem.domain_id)} · {escape(problem.target.task)} on "
            f"<code>{escape(problem.target.name)}</code>"
        ),
        footer=(
            "Attributions are computed against the fitted pipeline's own predict on the raw "
            "declared columns, so every driver is named in the domain's own vocabulary "
            "rather than in encoded columns. The serving path (aegis.ml) computes the same "
            "quantity per ensemble member, averaged by member weight."
        ),
    )
    destination.write_text(document, encoding="utf-8")
    return destination
