"""The one spec three consumers read — and the generator that makes it un-misspellable.

``aegis.ml.spec.resolve_spec`` reads an adapter's ML spec *leniently*, and that leniency
is a trap with a demo-sized blast radius (``aegis/src/aegis/ml/spec.py``)::

    if not features or not target:
        return FALLBACK_SPEC          # four columns of generated noise

A misspelled ``FEATURE_NAMES`` does not raise. It trains the trustworthy spine on noise
and serves the result as domain evidence, and the only native symptom is ``distinct=False``
on the last line of ``python -m app.ml``.

So this module does not *document* the five names an adapter must expose — it
**generates** the module that exposes them, from one declarative spec that also drives the
pandera data contract (``aegis_ml.contracts.frames``) and the AutoML feature handling
(``aegis_ml.features.pipeline``). One source, three consumers, no place to typo.

Imports pydantic and nothing else, deliberately: ``tests/test_types_is_dep_free.py``
asserts it, mirroring ``aegis/tests/ml/test_types_is_dep_free.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "DType",
    "FeatureSpec",
    "MLProblem",
    "TargetSpec",
    "TaskType",
]

DType = Literal["numeric", "categorical", "boolean", "datetime"]
"""How a feature is encoded. ``categorical`` maps to the spine's one-hot subset."""

TaskType = Literal["regression", "classification"]
"""The supervised task. Matches ``aegis.ml.spec.TaskType`` exactly — do not widen it
here: ``resolve_spec._coerce_task`` silently coerces anything unrecognised to
``"regression"``, so a third task type would train the wrong estimator in silence."""


class FeatureSpec(BaseModel):
    """One input column, in the shape the reference adapter's ``FEATURES`` list uses.

    ``dtype`` is load-bearing beyond documentation: ``resolve_spec._resolve_categorical``
    derives the one-hot subset from ``.dtype == "categorical"`` when the adapter declares
    no explicit ``CATEGORICAL_FEATURES``. A numeric column mislabelled categorical becomes
    hundreds of one-hot columns; a categorical mislabelled numeric is fed to the estimator
    as a meaningless integer ordering.
    """

    name: str = Field(description="Column name; must be a valid Python identifier.")
    dtype: DType = Field(description="Encoding class; drives the one-hot subset.")
    description: str = Field(description="What this measures, in the domain's language.")
    unit: str | None = Field(default=None, description="Unit for numeric features.")
    levels: list[str] = Field(
        default_factory=list,
        description="Allowed values for a categorical feature. Enforced by the pandera "
        "contract, so an unseen level is caught at the boundary rather than silently "
        "one-hot-encoded to all-zeros by OneHotEncoder(handle_unknown='ignore').",
    )
    minimum: float | None = Field(default=None, description="Inclusive lower bound.")
    maximum: float | None = Field(default=None, description="Inclusive upper bound.")
    nullable: bool = Field(default=False, description="Whether nulls are contract-legal.")

    @field_validator("name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        """Reject a name that cannot survive code generation into a module."""
        if not value.isidentifier():
            raise ValueError(f"feature name {value!r} is not a valid Python identifier")
        return value

    @model_validator(mode="after")
    def _levels_only_for_categorical(self) -> FeatureSpec:
        """Keep ``levels`` and ``dtype`` from disagreeing about what this column is."""
        if self.levels and self.dtype != "categorical":
            raise ValueError(f"{self.name!r} declares levels but dtype is {self.dtype!r}")
        if self.dtype == "categorical" and not self.levels:
            raise ValueError(
                f"{self.name!r} is categorical but declares no levels — the data contract "
                f"cannot check an open set, and an unseen level encodes to all-zeros "
                f"without raising."
            )
        return self


class TargetSpec(BaseModel):
    """The predicted column: its name, its task, and the unit a human reads it in.

    ``unit`` is not decoration — ``backend/src/app/ml/__main__.py`` prints the sanity
    probe with ``TARGET.unit``, and ``describe_prediction`` renders the conformal interval
    with it. A target without one prints bare floats into a decision-support sentence.
    """

    name: str = Field(description="Column name of the label.")
    task: TaskType = Field(description="'regression' or 'classification'.")
    description: str = Field(description="What the model predicts, in the client's words.")
    unit: str | None = Field(default=None, description="Unit, for regression targets.")
    levels: list[str] = Field(
        default_factory=list, description="Class labels, for classification targets."
    )
    minimum: float | None = Field(default=None, description="Inclusive lower bound.")
    maximum: float | None = Field(default=None, description="Inclusive upper bound.")

    @field_validator("name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        """Reject a name that cannot survive code generation into a module."""
        if not value.isidentifier():
            raise ValueError(f"target name {value!r} is not a valid Python identifier")
        return value

    @model_validator(mode="after")
    def _task_matches_shape(self) -> TargetSpec:
        """A classification target needs levels; a regression target needs a unit."""
        if self.task == "classification" and len(self.levels) < 2:
            raise ValueError(f"{self.name!r} is classification but declares <2 levels")
        if self.task == "regression" and self.levels:
            raise ValueError(f"{self.name!r} is regression but declares class levels")
        return self


class MLProblem(BaseModel):
    """The whole supervised problem, from which the adapter's piece 2 is generated.

    Every consumer reads this one object:

    * ``aegis_ml.contracts.frames`` derives the pandera ``DataFrameModel``
    * ``aegis_ml.features.pipeline`` derives the skrub/ColumnTransformer split
    * ``emit_ml_spec_module`` writes ``ml_spec.py`` with the exact five names
      ``aegis.adapter.MLSpecModule`` requires and ``resolve_spec`` reads

    Attributes:
        domain_id: Stable machine id; must match the adapter's ``DOMAIN_ID``.
        features: Ordered input columns. Order is preserved into ``FEATURE_NAMES``.
        target: The predicted column.
        primary_metric: The metric the promotion gate ranks challengers on.
        requested_coverage: The conformal level asked for. Named "requested" because the
            measured one is a separate field everywhere in Aegis — never one field.
    """

    domain_id: str = Field(description="Matches the adapter's DOMAIN_ID.")
    features: list[FeatureSpec] = Field(min_length=1)
    target: TargetSpec
    primary_metric: str = Field(default="", description="Blank ⇒ derived from the task.")
    requested_coverage: float = Field(default=0.9, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _coherent(self) -> MLProblem:
        """Reject duplicate columns, and a target that is also a feature."""
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate feature names: {dupes}")
        if self.target.name in names:
            raise ValueError(
                f"target {self.target.name!r} is also a feature — that is perfect leakage"
            )
        return self

    @property
    def feature_names(self) -> list[str]:
        """Ordered feature-column names — becomes the adapter's ``FEATURE_NAMES``."""
        return [f.name for f in self.features]

    @property
    def categorical_features(self) -> list[str]:
        """The one-hot subset, in declaration order."""
        return [f.name for f in self.features if f.dtype == "categorical"]

    @property
    def numeric_features(self) -> list[str]:
        """Everything the spine passes through to the estimator unencoded."""
        return [f.name for f in self.features if f.dtype != "categorical"]

    @property
    def metric(self) -> str:
        """The gate's ranking metric, defaulting to the task's Aegis-native one.

        ``r2`` and ``accuracy`` are the two ``ModelCard.metric_name`` ever carries
        (``aegis/src/aegis/ml/types.py``), so defaulting to them keeps the gate and the
        card talking about the same number.
        """
        if self.primary_metric:
            return self.primary_metric
        return "r2" if self.target.task == "regression" else "accuracy"
