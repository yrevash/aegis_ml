"""Result shapes that cross module and process boundaries — pydantic only.

Two boundaries make these types load-bearing rather than decorative:

1. **The venv boundary.** The AutoML search runs in an isolated trainer venv holding
   AutoGluon/TabPFN/torch, and its answer crosses back into the serving venv as JSON.
   :class:`Recipe` is that answer. If it does not round-trip through ``model_dump_json``,
   the two-venv split does not work.
2. **The light/heavy boundary.** ``aegis/ml/types.py`` imports nothing but pydantic so the
   backend's API-schema layer can name ML shapes without pulling xgboost transitively, and
   it has a test enforcing that. This module holds the same line.

Naming rule inherited from Aegis and applied without exception: a **requested** level and
a **measured** one are always two fields. ``ModelCard.conformal_coverage`` vs
``conformal_coverage_empirical``; ``BacktestReport.requested_coverage`` vs
``empirical_coverage``. Never one field that means whichever the reader assumes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "Candidate",
    "DriftReport",
    "GateDecision",
    "Leaderboard",
    "Recipe",
    "RecipeMember",
    "RegistryEntry",
    "RunManifest",
    "SliceMetric",
    "TierName",
    "TrainResult",
]

TierName = Literal["baseline", "flaml", "autogluon", "tabpfn"]
"""The four AutoML search tiers, weakest-but-always-present to strongest."""


class RecipeMember(BaseModel):
    """One estimator in a portable ensemble recipe.

    ``kind`` is a class name the *serving* venv must be able to construct — that is what
    makes the recipe portable. ``aegis_ml.automl.recipe.to_aegis_members`` maps it through
    an explicit allowlist and raises :class:`~aegis_ml.contracts.errors.RecipeNotPortableError`
    on anything unknown, rather than importing a name a search result asked for.
    """

    name: str = Field(description="Member key in the ensemble, e.g. 'xgboost'.")
    kind: str = Field(description="Estimator class name, e.g. 'XGBRegressor'.")
    params: dict[str, Any] = Field(default_factory=dict, description="Constructor kwargs.")
    weight: float = Field(default=1.0, ge=0.0, description="Soft-voting weight.")


class Recipe(BaseModel):
    """The portable result of an AutoML search — the keystone of the two-venv split.

    This is JSON that crosses from the trainer venv into the serving venv, where the Aegis
    spine fits it and keeps its own MAPIE conformal calibration, SHAP attribution, dataset
    digest and ModelCard. Full AutoML benefit, zero changes to ``aegis/``.
    """

    task: Literal["regression", "classification"]
    members: list[RecipeMember] = Field(min_length=1)
    categorical_features: list[str] = Field(default_factory=list)
    numeric_features: list[str] = Field(default_factory=list)
    tier: TierName = Field(description="Which search tier produced this.")
    search_seconds: float = Field(default=0.0, ge=0.0)
    notes: list[str] = Field(
        default_factory=list,
        description="Anything a reader must know — e.g. that the winning model was NOT "
        "portable and this recipe is the best portable runner-up.",
    )


class Candidate(BaseModel):
    """One scored entry on the leaderboard, winner or loser.

    Losers are kept deliberately, mirroring ``aegis.forecast.ForecastResult.candidates``,
    which reports SeasonalNaive's score even when it loses. A leaderboard showing only the
    winner cannot tell you whether the winner won by a nose or a mile — and the margin is
    what says whether the extra complexity was worth it.
    """

    name: str = Field(description="Model or configuration name.")
    tier: TierName
    metric_name: str
    metric_value: float
    fit_seconds: float = Field(default=0.0, ge=0.0)
    portable: bool = Field(
        default=True,
        description="Whether this can be re-fitted in the serving venv. A non-portable "
        "winner is reported as the accuracy ceiling, never promoted as the spine.",
    )
    selected: bool = Field(default=False)
    detail: dict[str, Any] = Field(default_factory=dict)


class Leaderboard(BaseModel):
    """Every candidate from every tier that ran, ranked."""

    metric_name: str
    higher_is_better: bool = True
    candidates: list[Candidate] = Field(default_factory=list)
    tiers_run: list[TierName] = Field(default_factory=list)
    tiers_skipped: dict[str, str] = Field(
        default_factory=dict,
        description="Tier → why it did not run. A skipped tier is never silent: an empty "
        "leaderboard slot and an unavailable dependency look identical otherwise.",
    )


class SliceMetric(BaseModel):
    """The primary metric restricted to one segment of the data.

    The gate reads the *worst* slice, not the mean. A model that improves on average while
    collapsing on one region is a regression for everyone in that region, and an aggregate
    score is exactly the instrument that cannot see it.
    """

    feature: str
    level: str
    n_rows: int
    metric_name: str
    metric_value: float


class TrainResult(BaseModel):
    """What one training run produced and measured."""

    run_id: str
    domain_id: str
    task: Literal["regression", "classification"]
    target: str
    metric_name: str
    metric_value: float = Field(description="MEASURED on the held-out test split.")
    requested_coverage: float = Field(description="The conformal level ASKED FOR.")
    empirical_coverage: float | None = Field(
        default=None, description="The rate ACTUALLY ACHIEVED on the held-out split."
    )
    training_size: int = 0
    calibration_size: int = 0
    test_size: int = 0
    dataset_digest: str | None = None
    recipe: Recipe | None = None
    leaderboard: Leaderboard | None = None
    slices: list[SliceMetric] = Field(default_factory=list)
    artifact_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class GateDecision(BaseModel):
    """Whether a challenger may replace the champion, and every number behind it.

    ``reasons`` is populated on a pass as well as a failure. "Promoted" with no figures is
    as opaque as "rejected" with no figures, and the model card quotes both.
    """

    promoted: bool
    challenger_run_id: str
    champion_run_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


class DriftReport(BaseModel):
    """Measured drift of live data against the reference frame stored at training time."""

    run_id: str
    reference_digest: str | None = None
    n_reference_rows: int = 0
    n_current_rows: int = 0
    dataset_drift: bool = False
    drifted_share: float = Field(default=0.0, ge=0.0, le=1.0)
    drifted_features: list[str] = Field(default_factory=list)
    target_drift: float | None = None
    prediction_drift: float | None = None
    estimated_metric_name: str | None = Field(
        default=None,
        description="NannyML CBPE/DLE estimate — performance WITHOUT ground truth. "
        "Named 'estimated' throughout so it is never read as a measurement.",
    )
    estimated_metric_value: float | None = None
    verdict: Literal["pass", "warn", "block"] = "pass"
    html_report_path: str | None = None


class RegistryEntry(BaseModel):
    """One immutable row in the filesystem model registry."""

    run_id: str
    domain_id: str
    created_at: str = Field(description="ISO-8601 UTC; stamped by the caller, not here.")
    stage: Literal["staging", "production", "archived"] = "staging"
    result: TrainResult
    gate: GateDecision | None = None
    paths: dict[str, str] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """What one pipeline execution did, stage by stage."""

    run_id: str
    flow: str
    started_at: str
    finished_at: str | None = None
    stages: list[dict[str, Any]] = Field(default_factory=list)
    ok: bool = True
    error: str | None = None
