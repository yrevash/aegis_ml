"""AutoML: four tiers, one leaderboard, one portable recipe.

The package's shape follows decision D1 in ``finalplan.md``. Read it in this order:

* :mod:`~aegis_ml.automl.tiers` — what can run here, and why the rest cannot.
* :mod:`~aegis_ml.automl.search` — :func:`~aegis_ml.automl.search.run_search` runs every
  available tier, scores every candidate on one held-out split, keeps the losers, and
  selects the best **portable** one.
* :mod:`~aegis_ml.automl.recipe` — the keystone: the allowlist that decides what may cross
  a venv boundary, and the code that rebuilds it into the exact ``[(name, estimator)]``
  shape ``aegis.ml.model._regression_members()`` returns.
* :mod:`~aegis_ml.automl.hpo` — Optuna refinement of the chosen recipe, resumable.
* :mod:`~aegis_ml.automl.runner` / ``_worker`` — the same search, executed in the isolated
  trainer venv when the strong tiers are not installable next to the serving stack.

Every name below is re-exported from a module that imports nothing heavier than pydantic at
module scope: sklearn, xgboost, FLAML, AutoGluon, TabPFN and Optuna are all imported inside
functions through :func:`aegis_ml._require.require`. Importing ``aegis_ml.automl`` therefore
stays cheap enough for the CLI and the light contracts layer, which is what keeps
``aegis-ml doctor`` able to *report* on tiers it cannot run.
"""

from __future__ import annotations

from aegis_ml.automl.hpo import tune
from aegis_ml.automl.recipe import (
    PORTABLE_KINDS,
    assert_portable,
    baseline_recipe,
    fit_recipe,
    load_recipe,
    save_recipe,
    to_aegis_members,
)
from aegis_ml.automl.runner import run_in_trainer_venv, trainer_available
from aegis_ml.automl.search import run_search, score_predictions
from aegis_ml.automl.tiers import (
    TABPFN_LICENSE_NOTICE,
    TIER_ORDER,
    available_tiers,
    require_tier,
    resolve_tiers,
    tier_status,
)

__all__ = [
    "PORTABLE_KINDS",
    "TABPFN_LICENSE_NOTICE",
    "TIER_ORDER",
    "assert_portable",
    "available_tiers",
    "baseline_recipe",
    "fit_recipe",
    "load_recipe",
    "require_tier",
    "resolve_tiers",
    "run_in_trainer_venv",
    "run_search",
    "save_recipe",
    "score_predictions",
    "tier_status",
    "to_aegis_members",
    "trainer_available",
    "tune",
]
