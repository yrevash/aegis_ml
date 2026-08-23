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
* **The explainer is model-agnostic on purpose.** ``TreeExplainer`` is faster, but it cannot
  see through the encoding pipeline the estimator is wrapped in, and attributions computed on
  one-hot columns cannot be summed back to the original feature without an assumption. The
  permutation explainer runs against the fitted pipeline's own ``predict`` on the raw frame,
  so every attribution is stated in terms of the columns the domain spec actually declares.

``shap`` is an optional dependency, imported through
:func:`aegis_ml._require.require` so an absent install raises with the exact command that
fixes it. There is no fallback path: a report produced without SHAP and a report produced
with it must never be indistinguishable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.explain.card import escape, html_page, svg_bar_chart

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    import pandas as pd

__all__ = [
    "ExplainerUnavailableError",
    "global_importance",
    "local_explanation",
    "render_html",
]


class ExplainerUnavailableError(AegisMLError):
    """The model cannot be explained in the form this module requires.

    Raised instead of falling back to a cruder attribution (feature permutation on the
    metric, say, or coefficient magnitudes). Those answer a different question, and a report
    that silently swapped one for the other would attribute a prediction to the wrong cause
    while looking identical.
    """

    def __init__(self, reason: str) -> None:
        """Name what was missing and what the caller can do about it."""
        super().__init__(
            f"Cannot compute SHAP attributions: {reason}. Nothing here substitutes a "
            f"different notion of importance on your behalf — permutation importance and "
            f"SHAP answer different questions, and a report that quietly swapped them would "
            f"attribute a prediction to the wrong cause with no visible symptom."
        )


def _predict_function(model: object, problem: MLProblem, template: pd.DataFrame) -> Any:  # noqa: ANN401
    """Wrap the fitted model so SHAP can call it with plain arrays.

    SHAP's model-agnostic explainers pass masked rows as a numpy array, but a fitted
    sklearn ``Pipeline`` over a raw domain frame needs the column names and the original
    dtypes — a string column arriving as ``object`` and a numeric column arriving as
    ``object`` are treated very differently by a ``ColumnTransformer``. This wrapper
    rebuilds the frame with the template's own dtypes on every call, so the model sees
    exactly the shape it was fitted on.

    Args:
        model: The fitted estimator or pipeline.
        problem: The declared problem; decides whether probabilities or values are explained.
        template: A frame carrying the correct column order and dtypes.

    Returns:
        A callable ``(array) -> ndarray`` suitable for ``shap.Explainer``.

    Raises:
        ExplainerUnavailableError: When a classifier exposes no ``predict_proba``. A hard
            0/1 output has no gradient for SHAP to attribute, and explaining the encoded
            class index would produce numbers whose sign means nothing.
    """
    import pandas as pd

    columns = list(template.columns)
    dtypes = template.dtypes.to_dict()
    classification = problem.target.task == "classification"
    if classification and not hasattr(model, "predict_proba"):
        raise ExplainerUnavailableError(
            "the classifier exposes no predict_proba, so there is no continuous output to "
            "attribute; explaining the hard class index would produce signed numbers whose "
            "direction is arbitrary. Wrap the estimator in CalibratedClassifierCV, or use "
            "an estimator that reports probabilities"
        )

    def predict(array: object) -> object:
        """Rebuild a domain frame from SHAP's array and run the model on it."""
        frame = array if isinstance(array, pd.DataFrame) else pd.DataFrame(array, columns=columns)
        frame = frame.astype(dtypes)
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
) -> Any:  # noqa: ANN401 - a shap.Explanation, typed only when shap is installed
    """Compute a SHAP ``Explanation`` for a sample of rows.

    Cost is the reason both sample sizes are parameters: the permutation explainer evaluates
    the model roughly ``2 * n_features + 1`` times per explained row, against every
    background row. Explaining 500 rows against 50 background rows on a 10-feature problem is
    around half a million model evaluations — seconds for a gradient-boosted tree, minutes if
    the numbers are raised carelessly. The defaults are chosen to be honest and finishable.

    Args:
        model: The fitted estimator or pipeline.
        frame: Rows to draw the sample and the background from; feature columns only are used.
        problem: The declared problem.
        max_samples: How many rows to explain.
        background_samples: How many rows form the masking distribution. The background is
            what "absent" means to SHAP — attributions are stated *relative to* it, so it
            must come from the same population as the explained rows.
        seed: Sampling seed, so a re-run of the demo produces the same figures.

    Returns:
        The ``shap.Explanation``.

    Raises:
        ExplainerUnavailableError: When the frame is empty or lacks the declared features.
    """
    shap = require("aegis-ml[serve]", "shap")

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

    predict = _predict_function(model, problem, features)
    masker = shap.maskers.Independent(background, max_samples=n_background)
    explainer = shap.Explainer(predict, masker, algorithm="permutation")
    return explainer(sample)


def global_importance(
    model: object,
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    max_samples: int = 500,
    background_samples: int = 50,
    seed: int | None = None,
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
    shap = require("aegis-ml[serve]", "shap")
    sampled = reference.sample(
        n=max(1, min(background_samples, len(reference))), random_state=seed
    )
    predict = _predict_function(model, problem, reference)
    masker = shap.maskers.Independent(sampled, max_samples=len(sampled))
    explainer = shap.Explainer(predict, masker, algorithm="permutation")
    explanation = explainer(single)

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
