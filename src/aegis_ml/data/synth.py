"""SDV wrapper for the "we have a real CSV, make ten times more of it" path.

This is the **secondary** generator. The primary one — the procedural + LLM hybrid in
``templates/adapter/generator.py``, mirroring ``app.adapter.generator`` — stays primary for
a specific reason: it draws its labels from a declared latent function
(:mod:`aegis_ml.data.latent`), so the target is learnable by construction and its
irreducible noise is a number somebody chose. SDV learns a joint distribution from data
that already exists. When there is real data that is exactly what you want, and when there
is not, SDV has nothing to learn from.

Two failure modes make the ordering matter, and both are things this module measures rather
than assumes.

**A synthesizer can memorise.** A copula or GAN fitted on 400 rows can emit rows that are
those 400 rows with the noise rounded off. Train on that, evaluate against the real data,
and the score is a lookup, not a prediction. :func:`quality_report` therefore carries
``new_row_share`` from SDMetrics' ``NewRowSynthesis`` alongside the headline quality number,
because a high fidelity score and a low novelty share together mean *copying*, which is the
one combination that reads as success.

**A synthesizer reproduces the leakage it was shown.** If the real CSV contains a column
that is a restatement of the label, the synthetic copy contains it too, faithfully. Run
:func:`aegis_ml.features.leakage.detect_leakage` on the output, not just the input.

SDV lives in the ``strong`` extra and therefore in the isolated trainer venv: it pulls
torch through CTGAN and will not resolve under the backend's ``pandas<2.4`` /
``numpy<2.5`` / ``numba==0.67.0`` caps. Sampling happens there; the parquet it writes is
what crosses back.

Verified against **SDV 1.38.1 / SDMetrics 0.29.0** in the trainer venv, on a 300-row frame:
``gaussian_copula`` fit → 300 sampled rows → quality 0.9306 (column shapes 0.9587, column
pair trends 0.9025) with ``new_row_share`` 1.0. The only surface that moved between 1.37 and
1.38 is the evaluation entry point — see :func:`quality_report`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError, InsufficientLabelsError
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

    import pandas as pd

    from aegis_ml.contracts.spec import MLProblem

__all__ = [
    "MIN_FIT_ROWS",
    "SDV_EXTRA",
    "FittedSynthesizer",
    "SynthModel",
    "fit_synthesizer",
    "metadata_for",
    "quality_report",
    "sample",
    "synthesize",
]

logger = logging.getLogger(__name__)

SDV_EXTRA = "aegis-ml[strong]"
"""Install target quoted when SDV is missing. Deliberately the trainer-venv extra."""

MIN_FIT_ROWS = 50
"""Fewest real rows worth fitting a synthesizer on.

Below this the learned joint distribution is a description of the sample rather than of the
population, and every synthetic row is a near-copy of a real one. Refusing is honest; the
alternative is ten thousand rows of laundered training data.
"""

SynthModel = Literal["gaussian_copula", "ctgan", "tvae"]
"""Which synthesizer to fit.

``gaussian_copula`` is the default and usually the right answer: it fits in seconds, has no
epochs to tune, and captures the marginal distributions plus the correlation structure,
which is most of what a tabular demo needs. ``ctgan`` and ``tvae`` are neural and model
multi-modal conditionals better, at the cost of minutes of fitting and a much greater
appetite for memorising a small table.
"""

_SYNTHESIZER_CLASSES = {
    "gaussian_copula": "GaussianCopulaSynthesizer",
    "ctgan": "CTGANSynthesizer",
    "tvae": "TVAESynthesizer",
}
"""Model key → the class name to pull from ``sdv.single_table``."""

_SDTYPE_FOR_DTYPE = {
    "numeric": "numerical",
    "categorical": "categorical",
    "boolean": "boolean",
    "datetime": "datetime",
}
"""Spec dtype → SDV sdtype, so a declared spec overrides SDV's own inference."""


def _sdv(submodule: str) -> ModuleType:
    """Import an SDV submodule through :func:`~aegis_ml._require.require`."""
    return require(SDV_EXTRA, f"sdv.{submodule}")


@dataclass(frozen=True)
class FittedSynthesizer:
    """A fitted SDV synthesizer plus everything needed to evaluate what it produces.

    A dataclass rather than a pydantic model because it holds a live SDV object, which is
    neither serialisable nor validatable — and pretending otherwise by declaring it ``Any``
    on a ``BaseModel`` would put an unvalidatable field in a package whose contract layer is
    strictly pydantic. It is frozen so the metadata cannot drift away from the model that
    was fitted under it, which would make :func:`quality_report` score against the wrong
    schema and report a fidelity number for a comparison that never happened.

    Attributes:
        synthesizer: The fitted SDV synthesizer.
        metadata: The SDV ``Metadata`` it was fitted under.
        model: Which :data:`SynthModel` was used.
        n_train_rows: How many real rows it learned from. Carried because it is the single
            most important caveat on any sample drawn from it.
        table_name: The table key inside ``metadata``.
    """

    synthesizer: Any
    metadata: Any
    model: SynthModel
    n_train_rows: int
    table_name: str

    def sample(self, n: int) -> pd.DataFrame:
        """Draw ``n`` synthetic rows.

        Args:
            n: How many rows to generate.

        Returns:
            A frame with the same columns as the frame this was fitted on.

        Raises:
            ValueError: When ``n`` is not positive.
        """
        if n <= 0:
            raise ValueError(f"n must be positive; got {n!r}")
        if n > self.n_train_rows * 100:
            logger.warning(
                "Sampling %d rows from a synthesizer fitted on %d: beyond roughly 100× the "
                "training size the sample stops adding information and only adds "
                "confidence in it.",
                n,
                self.n_train_rows,
            )
        return self.synthesizer.sample(num_rows=n)


def metadata_for(
    frame: pd.DataFrame,
    *,
    problem: MLProblem | None = None,
    table_name: str = "table",
) -> Any:  # noqa: ANN401 - an sdv.metadata.Metadata, imported lazily
    """Detect SDV metadata from a frame, overriding it with the spec where one exists.

    Detection alone gets one thing reliably wrong: a categorical stored as an integer code
    is inferred as numerical, and the synthesizer then happily emits ``2.7`` for a column
    whose only legal values are ``1``, ``2`` and ``3``. The declared spec already knows
    better, so where it speaks it wins.

    Args:
        frame: The real data.
        problem: The declarative problem, when one exists.
        table_name: Key for the single table inside the metadata object.

    Returns:
        An SDV ``Metadata`` instance.

    Raises:
        ImportError: When SDV is not installed, naming the exact install command.
    """
    metadata_module = _sdv("metadata")
    metadata = metadata_module.Metadata.detect_from_dataframe(data=frame, table_name=table_name)
    if problem is None:
        return metadata
    declared = {f.name: _SDTYPE_FOR_DTYPE[f.dtype] for f in problem.features}
    if problem.target.task == "classification":
        declared[problem.target.name] = "categorical"
    else:
        declared[problem.target.name] = "numerical"
    for column, sdtype in declared.items():
        if column in frame.columns:
            metadata.update_column(column_name=column, sdtype=sdtype, table_name=table_name)
    return metadata


def fit_synthesizer(
    frame: pd.DataFrame,
    *,
    model: SynthModel = "gaussian_copula",
    problem: MLProblem | None = None,
    epochs: int | None = None,
    table_name: str = "table",
) -> FittedSynthesizer:
    """Fit a synthesizer to real data.

    Args:
        frame: The real rows. Every column present is modelled, including the target — a
            synthesizer fitted on the features alone would produce feature rows with no
            label, and re-labelling them from a latent function would make the synthesizer
            pointless.
        model: Which :data:`SynthModel` to use.
        problem: Used to override SDV's dtype inference; see :func:`metadata_for`.
        epochs: Training epochs for the neural models. Ignored by ``gaussian_copula``,
            which has none — passing it there is refused rather than silently dropped.
        table_name: Key for the single table inside the metadata.

    Returns:
        A :class:`FittedSynthesizer`.

    Raises:
        InsufficientLabelsError: When the frame has fewer than :data:`MIN_FIT_ROWS` rows.
        AegisMLError: When ``model`` is unknown, or ``epochs`` is passed to a model that
            has no epochs.
        ImportError: When SDV is not installed, naming the exact install command.
    """
    if model not in _SYNTHESIZER_CLASSES:
        raise AegisMLError(
            f"Unknown synthesizer {model!r}. Available: {sorted(_SYNTHESIZER_CLASSES)}."
        )
    if len(frame) < MIN_FIT_ROWS:
        raise InsufficientLabelsError(
            have=int(len(frame)),
            need=MIN_FIT_ROWS,
            what=f"Fitting the {model!r} synthesizer",
        )
    if epochs is not None and model == "gaussian_copula":
        raise AegisMLError(
            "gaussian_copula has no training epochs — it is fitted in closed form. Passing "
            "`epochs` here would be silently ignored, so it is refused instead. Use "
            "model='ctgan' or model='tvae' if you meant to train a neural synthesizer."
        )

    single_table = _sdv("single_table")
    metadata = metadata_for(frame, problem=problem, table_name=table_name)
    constructor = getattr(single_table, _SYNTHESIZER_CLASSES[model])
    kwargs: dict[str, Any] = {"metadata": metadata}
    if epochs is not None:
        kwargs["epochs"] = epochs
    synthesizer = constructor(**kwargs)
    synthesizer.fit(frame)
    logger.info("Fitted %s on %d rows × %d columns.", model, len(frame), len(frame.columns))
    return FittedSynthesizer(
        synthesizer=synthesizer,
        metadata=metadata,
        model=model,
        n_train_rows=int(len(frame)),
        table_name=table_name,
    )


def sample(fitted: FittedSynthesizer, n: int) -> pd.DataFrame:
    """Draw ``n`` synthetic rows from a fitted synthesizer.

    Args:
        fitted: The result of :func:`fit_synthesizer`.
        n: How many rows to generate.

    Returns:
        A frame with the fitted schema's columns.
    """
    return fitted.sample(n)


def quality_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    metadata: Any = None,  # noqa: ANN401 - an sdv.metadata.Metadata
    problem: MLProblem | None = None,
    table_name: str = "table",
) -> dict[str, Any]:
    """Score synthetic data against the real data it was learned from.

    Read the two headline numbers **together**, never separately:

    * ``overall_score`` (SDMetrics quality) says how closely the synthetic marginals and
      pairwise structure match the real ones. High is good.
    * ``new_row_share`` (``NewRowSynthesis``) says what fraction of synthetic rows are not
      copies of real ones. High is good.

    A quality score of 0.97 with a new-row share of 0.20 is not a good synthesizer, it is a
    photocopier — and a model trained on its output and evaluated against the real data is
    performing a lookup while reporting a prediction. That combination is the reason this
    function returns both instead of the headline number alone.

    Args:
        real: The original rows.
        synthetic: The generated rows.
        metadata: The SDV ``Metadata`` both frames conform to. Detected from ``real`` when
            omitted; pass the one from :class:`FittedSynthesizer` when you have it, so the
            evaluation uses the same schema the fit used.
        problem: Used for dtype overrides when metadata must be detected.
        table_name: Key for the single table inside the metadata.

    Returns:
        ``{"overall_score", "properties", "new_row_share", "n_real", "n_synthetic",
        "notes"}`` — all JSON-safe.

    Raises:
        AegisMLError: When the two frames do not share a column set, which would make every
            comparison below it meaningless.
        ImportError: When SDV or SDMetrics are not installed, naming the install command.
    """
    real_columns, synthetic_columns = set(real.columns), set(synthetic.columns)
    if real_columns != synthetic_columns:
        raise AegisMLError(
            f"Real and synthetic frames have different columns; comparison is meaningless. "
            f"Only in real: {sorted(real_columns - synthetic_columns)}. "
            f"Only in synthetic: {sorted(synthetic_columns - real_columns)}."
        )
    # ``sdv.evaluation``, not ``sdv.evaluation.single_table``. SDV 1.38 moved the evaluation
    # entry points up one level and listed the old ones in
    # ``sdv.evaluation.single_table.DEPRECATED_EVALUATION_FUNCTIONS``; calling them still
    # works but emits a FutureWarning on every report, which is exactly the kind of noise
    # that trains a reader to ignore warnings. Both spellings take the same arguments.
    evaluation = _sdv("evaluation")
    resolved = metadata or metadata_for(real, problem=problem, table_name=table_name)
    report = evaluation.evaluate_quality(
        real_data=real, synthetic_data=synthetic, metadata=resolved, verbose=False
    )
    properties = report.get_properties()
    scores = {
        str(row.Property): float(row.Score)
        for row in properties.itertuples(index=False)
        if row.Score == row.Score  # drop NaN scores rather than serialising them
    }
    result: dict[str, Any] = {
        "overall_score": float(report.get_score()),
        "properties": scores,
        "new_row_share": _new_row_share(real, synthetic, resolved),
        "n_real": int(len(real)),
        "n_synthetic": int(len(synthetic)),
        "notes": [],
    }
    _annotate(result)
    return result


def _new_row_share(real: pd.DataFrame, synthetic: pd.DataFrame, metadata: Any) -> float | None:  # noqa: ANN401
    """Fraction of synthetic rows that are not copies of real ones, via SDMetrics.

    Returns ``None`` rather than a substitute number when the metric cannot run — a
    fabricated novelty score is precisely the kind of plausible-looking figure this package
    refuses to emit, and ``None`` propagates into a note the reader can see.
    """
    single_table = require(SDV_EXTRA, "sdmetrics.single_table")
    metric = getattr(single_table, "NewRowSynthesis", None)
    if metric is None:
        logger.warning(
            "sdmetrics.single_table has no NewRowSynthesis in this version; the novelty "
            "share cannot be measured and is reported as null rather than assumed."
        )
        return None
    as_dict = metadata.to_dict() if hasattr(metadata, "to_dict") else metadata
    tables = as_dict.get("tables") if isinstance(as_dict, dict) else None
    single = next(iter(tables.values())) if tables else as_dict
    return float(metric.compute(real_data=real, synthetic_data=synthetic, metadata=single))


def _annotate(result: dict[str, Any]) -> None:
    """Attach the readings a human would otherwise have to derive from the two numbers."""
    novelty = result["new_row_share"]
    quality = result["overall_score"]
    if novelty is None:
        result["notes"].append(
            "novelty could not be measured; treat the quality score as unverified — a "
            "memorising synthesizer scores near-perfect fidelity by definition"
        )
        return
    if novelty < 0.9 and quality > 0.85:
        result["notes"].append(
            f"quality {quality:.3f} with only {novelty:.1%} novel rows: the synthesizer is "
            f"largely copying its training data. A model trained on this and evaluated "
            f"against the real frame is performing a lookup, not a prediction."
        )
    if novelty >= 0.9 and quality < 0.6:
        result["notes"].append(
            f"quality {quality:.3f} is low with {novelty:.1%} novel rows: the synthesizer "
            f"is inventing freely but not reproducing the real structure. Fit on more rows, "
            f"or use model='ctgan' for multi-modal conditionals."
        )


def synthesize(
    frame: pd.DataFrame,
    *,
    n: int,
    model: SynthModel = "gaussian_copula",
    problem: MLProblem | None = None,
    epochs: int | None = None,
    evaluate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit, sample and score in one call — the whole "make 10× more" path.

    Args:
        frame: The real rows.
        n: How many synthetic rows to draw.
        model: Which :data:`SynthModel` to fit.
        problem: Used for dtype overrides.
        epochs: Training epochs for the neural models.
        evaluate: Whether to run :func:`quality_report`. Turning it off is legitimate for a
            very large sample where the metrics dominate the runtime, but the returned
            report then carries a note saying the sample is unvalidated rather than an
            empty dictionary that reads like a pass.

    Returns:
        ``(synthetic_frame, report)``.
    """
    fitted = fit_synthesizer(frame, model=model, problem=problem, epochs=epochs)
    synthetic = fitted.sample(n)
    if not evaluate:
        return synthetic, {
            "overall_score": None,
            "properties": {},
            "new_row_share": None,
            "n_real": int(len(frame)),
            "n_synthetic": int(len(synthetic)),
            "notes": [
                "evaluate=False: this sample has NOT been scored for fidelity or novelty. "
                "Do not report it as validated synthetic data."
            ],
        }
    report = quality_report(
        frame, synthetic, metadata=fitted.metadata, table_name=fitted.table_name
    )
    report["model"] = fitted.model
    report["n_train_rows"] = fitted.n_train_rows
    report["seed_note"] = (
        f"SDV sampling is seeded by the synthesizer's own RNG, not by "
        f"settings.random_seed ({settings.random_seed}); a repeat call draws different "
        f"rows unless the fitted object is reused."
    )
    return synthetic, report
