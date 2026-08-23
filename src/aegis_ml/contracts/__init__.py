"""Dep-free contract layer: one spec, typed results, typed refusals.

Imports pydantic and nothing else. ``tests/test_types_is_dep_free.py`` asserts it in a
subprocess, mirroring ``aegis/tests/ml/test_types_is_dep_free.py`` — so the backend's
light API-schema layer can name these shapes without pulling pandas, sklearn or torch.
"""

from __future__ import annotations

from aegis_ml.contracts.errors import (
    AegisMLError,
    AutoMLTierUnavailableError,
    DriftThresholdExceededError,
    InsufficientLabelsError,
    LabelNotLearnableError,
    PromotionRejectedError,
    RecipeNotPortableError,
    TargetLeakageError,
    TrainerVenvMissingError,
)
from aegis_ml.contracts.protocols import (
    Candidate,
    DriftReport,
    GateDecision,
    Leaderboard,
    Recipe,
    RecipeMember,
    RegistryEntry,
    RunManifest,
    SliceMetric,
    TierName,
    TrainResult,
)
from aegis_ml.contracts.spec import DType, FeatureSpec, MLProblem, TargetSpec, TaskType

__all__ = [
    "AegisMLError",
    "AutoMLTierUnavailableError",
    "Candidate",
    "DType",
    "DriftReport",
    "DriftThresholdExceededError",
    "FeatureSpec",
    "GateDecision",
    "InsufficientLabelsError",
    "LabelNotLearnableError",
    "Leaderboard",
    "MLProblem",
    "PromotionRejectedError",
    "Recipe",
    "RecipeMember",
    "RecipeNotPortableError",
    "RegistryEntry",
    "RunManifest",
    "SliceMetric",
    "TargetLeakageError",
    "TargetSpec",
    "TaskType",
    "TierName",
    "TrainResult",
    "TrainerVenvMissingError",
]
