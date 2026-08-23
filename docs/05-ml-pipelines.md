# 05 · ML pipelines

How data becomes a promoted, monitored model. Module map: `aegis_ml.pipelines`, `aegis_ml.automl`, `aegis_ml.features`, `aegis_ml.evaluate`, `aegis_ml.explain`, `aegis_ml.forecast`.

---

## 1. The flows

`aegis_ml/src/aegis_ml/pipelines/flows.py`. Ordinary Python functions over a `Stage` protocol. Prefect is a **decorator**, never a requirement — `prefect_shim.py` applies `@flow` / `@task` when Prefect imports and identity decorators otherwise. **A trained artifact never depends on a server being up.**

| Flow | Stages | Produces |
|---|---|---|
| `data_flow` | profile → contract → learnable → leakage → split → reference | validated frame, `drift_ref.parquet`, `profile.html` |
| `train_flow` | `data_flow` → search → hpo → recipe → fit → calibrate | `model.joblib`, `recipe.json`, `leaderboard.json` |
| `eval_flow` | metrics → coverage → slices → shap → card | `card.json`, `card.html`, `shap.html`, `slices.json` |
| `promote_flow` | gate → register → publish | `GateDecision`, the registry entry, `backend/.artifacts/ml_spine.joblib` |
| `drift_flow` | reference-load → evidently → nannyml → verdict | `DriftReport`, `drift.html` |
| `full_flow` | `data` → `train` → `eval` → `promote` → `drift` | everything above, one `RunManifest` |

Every flow returns a `RunManifest` (`run_id`, `flow`, `started_at`, `finished_at`, `stages`, `ok`, `error`) so a partial run says which stage stopped it.

### CLI

```bash
cd /Users/yrevash/aegis_ml

uv run aegis-ml doctor                        # environment, tiers, paths, learnability
uv run aegis-ml init --domain <domain_id>     # scaffold templates/adapter/ into a new dir
uv run aegis-ml contract                      # pandera + assert_learnable + leakage
uv run aegis-ml synth --rows 5000             # SDV path (needs .venv-ml)
uv run aegis-ml train --tier all              # the AutoML search + fit
uv run aegis-ml eval                          # metrics, coverage, slices, card
uv run aegis-ml promote                       # the gate; writes ml_spine.joblib on pass
uv run aegis-ml drift                         # Evidently + NannyML
uv run aegis-ml forecast                      # the domain demand series
uv run aegis-ml card --run <run_id>           # render a model card
uv run aegis-ml export --onnx                 # portable point-predictor
```

PowerShell is identical apart from `Set-Location` instead of `cd`; see `docs/08-windows.md`.

---

## 2. Stage by stage

### 2.1 `profile` — `aegis_ml.data.profile`

skrub `TableReport` over the training frame plus `realism_report` (see `docs/04-synthetic-data.md` §7.3). Writes `profile.html`. Free, fast, and it is the tab you open when a judge asks "what does your data look like?".

### 2.2 `contract` — `aegis_ml.contracts.frames`

Derives a pandera `DataFrameSchema` from the **same `MLProblem`** the adapter's `ml_spec.py` was generated from, so the contract cannot drift from the spec that trains the model.

Enforced per column: dtype (`numeric`→`float64`, `boolean`→`bool`, `datetime`→`datetime64[ns]`, `categorical`→`str`), `minimum`/`maximum` as inclusive range checks, `nullable`, and — for categoricals — **the declared level set**.

> The level check is the point. `aegis.ml.model.train` one-hot-encodes with `handle_unknown="ignore"`: an unseen level **does not raise**, it encodes to an all-zero block and the row is scored as if the feature were absent. `numeric` maps to `float64` rather than a permissive union deliberately, so a column of stringified numbers (the classic CSV round-trip failure) is coerced and range-checked instead of being one-hot-encoded as a categorical by accident.

Failure reports the first `_MAX_REPORTED_FAILURES = 20` violations with column, check and offending value.

### 2.3 `learnable` — `aegis_ml.data.latent`

`assert_learnable`. Raises `LabelNotLearnableError`. **This is the stage that saves the demo.** See `docs/04-synthetic-data.md`.

### 2.4 `leakage` — `aegis_ml.features.leakage`

Every feature scored alone against the target. Anything above `settings.leakage_threshold` (0.98) raises `TargetLeakageError` naming the feature and its score, with the remedy: drop it from `FEATURES`, or declare it intentional via config if it is genuinely available at prediction time.

Also flags near-duplicate feature pairs (|r| > 0.99) and features that are constant.

### 2.5 `split` — `aegis_ml.data.splits`

Three disjoint splits, because the Aegis spine needs all three:

| Split | Used for | Default |
|---|---|---|
| train | fitting the ensemble members | 60% |
| **calibration** | **MAPIE conformal calibration — must be disjoint** | 20% |
| test | held-out metrics, empirical coverage, slice metrics | 20% |

`aegis.ml.model.TrustworthyModel.train` takes `calibration_size` as a fraction and does the train/calibration split itself; `aegis_ml` holds out the test split *before* handing the rest over, so the reported metric and the measured coverage come from rows the model has never seen in any capacity.

Stratified for classification. `_min_calibration_rows(confidence_level)` in `aegis/ml/model.py` enforces a floor — a 90% interval calibrated on 12 rows is not a guarantee.

### 2.6 `reference` — the drift baseline

The exact validated training frame is written to `drift_ref.parquet` alongside its SHA-256 digest. `drift_flow` compares live data against **this** frame, not against a fresh generation.

### 2.7 `search` — `aegis_ml.automl.search`

See §3.

### 2.8 `hpo` — `aegis_ml.automl.hpo`

An Optuna study over the winning recipe. **TPE sampler + HyperbandPruner, SQLite storage so it resumes.** Defaults: `settings.hpo_trials = 60`, `settings.hpo_timeout = 600`.

TPE overtakes random search from roughly the 30th trial at 3–5 important hyperparameters, so 60 trials is a sensible floor, not a round number. Under time pressure, skip HPO entirely: `--no-hpo`. The search tiers already give you most of the gain.

The study database lives at `registry_store/hpo/<domain_id>.db`. Re-running resumes rather than restarting — which is exactly what you want when the first attempt was interrupted.

### 2.9 `recipe` → `fit` → `calibrate`

The search result crosses back as a `Recipe`; `to_aegis_members(recipe, random_state=...)` turns it into the `list[tuple[str, Estimator]]` shape `aegis.ml.model._regression_members()` returns; the **Aegis spine** fits it and does its own MAPIE calibration, SHAP explainer construction, dataset digest and `ModelCard`. See §4.

### 2.10 `metrics`, `coverage`, `slices` — `aegis_ml.evaluate`

`metrics.py` computes:

| Task | Metrics |
|---|---|
| regression | `r2`, `mae`, `rmse`, `mape`, `median_ae`, `max_error` |
| classification | `accuracy`, `balanced_accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `roc_auc` (binary), `log_loss`, `brier` |

`HIGHER_IS_BETTER` is an explicit table, and `higher_is_better(name)` raises `UnknownMetricError` for an unknown metric **rather than defaulting to "higher wins"** — a gate that silently assumed the wrong direction would promote every regression.

`MLProblem.metric` defaults to `"r2"` for regression and `"accuracy"` for classification, because those are the only two values `ModelCard.metric_name` ever carries, which keeps the gate and the card talking about the same number.

**Coverage** is the honesty check: measure how often the conformal interval actually contained the truth on the held-out split, and report it as `empirical_coverage` next to `requested_coverage`. Never one field.

**Slices** (`evaluate/slices.py`) recompute the primary metric restricted to each level of each categorical feature, returning `SliceMetric(feature, level, n_rows, metric_name, metric_value)`. The gate reads the **worst** slice, not the mean: a model that improves on average while collapsing on one region is a regression for everyone in that region, and an aggregate score is exactly the instrument that cannot see it.

### 2.11 `shap`, `card` — `aegis_ml.explain`

- `shap_report.py` — beeswarm, bar and dependence plots rendered to a self-contained `shap.html`. Reads the spine's own `shap_attribution`, so what you show is what the agent was told.
- `pdp.py` — partial-dependence curves for the top drivers. This is where an irrelevant feature visibly flatlines.
- `reason_codes.py` — top-k signed drivers per prediction as human sentences; the same content `describe_prediction` renders.
- `card.py` — `card.json` (the `ModelCard` plus `TrainResult`, `Leaderboard`, `GateDecision`, slice table and licence notices) and `card.html`.

**Anything TabPFN touched carries `TABPFN_LICENSE_NOTICE` in the card.** It is a module constant precisely so it can be *copied into data* — `Recipe.notes`, `Candidate.detail`, the card — not merely read by a developer.

### 2.12 Forecasting — `aegis_ml.forecast`

`engine.py` **wraps** `aegis.forecast` rather than replacing it: Nixtla StatsForecast (AutoARIMA / AutoETS / SeasonalNaive) with `ConformalIntervals` and rolling-origin backtests, fed by your adapter's `domain_series_events()`.

`ml_forecast.py` adds Nixtla **`mlforecast`** candidates — global ML forecasting with automated lag/rolling feature engineering, and conformal prediction on all models. `foundation.py` adds time-series foundation-model candidates via `autogluon.timeseries` (trainer venv only). `backtest.py` runs the rolling-origin evaluation across every candidate.

New candidates are **added to the existing engine's candidate list**, keeping `BacktestReport.requested_coverage` / `empirical_coverage` and the losers-reported-too behaviour.

---

## 3. The AutoML tiers

`aegis_ml/src/aegis_ml/automl/tiers.py`. `TIER_ORDER = ("baseline", "flaml", "autogluon", "tabpfn")` — weakest-but-always-present to strongest. **Order is load-bearing twice**: a time-budgeted search spends its first seconds on the tier guaranteed to produce something portable, and leaderboard ties break towards the earlier, cheaper tier.

| Tier | Needs | Venv | Portable? | What it is |
|---|---|---|---|---|
| `baseline` | `sklearn` | either | **Yes** | sklearn + xgboost soft-voting — *the same members `aegis.ml.model` builds* — plus a linear reference floor. The explicit floor the other tiers must beat to justify themselves. |
| `flaml` | `flaml` | either | **Yes** | FLAML cost-frugal search under a wall-clock budget. Pure Python, so its winner is re-fittable in the serving venv with no subprocess. |
| `autogluon` | `autogluon.tabular` | **`.venv-ml`** | No | `TabularPredictor(presets="best_quality")` — multi-layer stacked ensembles. The stack cannot be re-fitted in the serving venv, so it is reported as an **accuracy ceiling**, never promoted as the spine. |
| `tabpfn` | `tabpfn` | **`.venv-ml`** | No | TabPFN-2.5 foundation model (plus AutoTabPFN when `tabpfn_extensions` is present). Strongest at the 1k–10k row scale this factory generates. Not portable: **the prediction *is* the pretrained transformer.** |

`TIER_REQUIREMENTS["baseline"]` names `sklearn` and **not** `xgboost`, deliberately. XGBoost is a backend-venv dependency but a bare `aegis-ml` dev venv may have sklearn without it, and requiring it would make the *only always-portable tier* unavailable there, leaving a search with no recipe to return. XGBoost is instead checked per-member by `recipe.is_portable_kind`, and its absence removes candidates **with a recorded reason** rather than removing the tier.

### Available, disabled, unavailable — three different states

```python
resolve_tiers(requested) -> (to_run: list[TierName], skipped: dict[str, str])
```

The **single** place that decides which tiers run is also the single place that produces the `tiers_skipped` map, so it is impossible to drop a tier without writing down why. An unknown tier name (a typo in `--tier`) is skipped *with a reason*, not silently dropped.

| State | Reason string | Switch |
|---|---|---|
| available | — | — |
| **disabled by policy** | `"disabled via AEGIS_ML_ENABLE_TABPFN=0"` | `settings.enable_flaml` / `enable_autogluon` / `enable_tabpfn` |
| **not installed** | names the missing module and `TIER_EXTRAS[tier]` | `uv pip install 'aegis-ml[strong]'` |
| not requested | `"not requested by the caller"` | `--tier` |

`baseline` has **no switch** on purpose: something portable must always be able to run, or there is no recipe to hand back.

Asking for a tier that is not importable raises `AutoMLTierUnavailableError` naming the tier, the failed import and the install command — **never a silent fall-through to a weaker tier.** A caller who asked for AutoGluon and got plain XGBoost must be able to tell that apart from a caller who got what they asked for, because the leaderboard they publish says which one ran.

### The leaderboard keeps the losers

`Leaderboard.candidates` holds **every** scored candidate from every tier, winner and loser, mirroring `aegis.forecast.ForecastResult.candidates`. A leaderboard showing only the winner cannot tell you whether the winner won by a nose or a mile — and the margin is what says whether the extra complexity was worth it.

Each `Candidate` carries `name`, `tier`, `metric_name`, `metric_value`, `fit_seconds`, **`portable`**, `selected`, `detail`.

### TabPFN-2.5 licence

On by default, because at 1k–10k rows this is the band where TabPFN-2.5 has a ~100% win rate against default XGBoost, and a hackathon demo is evaluation use.

```
TabPFN-2.5 weights are distributed under the Prior Labs License: research and evaluation
use are permitted, commercial and production use are NOT. This tier's score is reported
as an accuracy ceiling for evaluation purposes only. Set AEGIS_ML_ENABLE_TABPFN=0 to
switch the tier off entirely.
```

Printed by `aegis-ml doctor`, carried in `Recipe.notes`, `Candidate.detail` and every model card. Disable with:

```bash
export AEGIS_ML_ENABLE_TABPFN=0        # bash
$env:AEGIS_ML_ENABLE_TABPFN = "0"      # PowerShell
```

> Note the exact spelling: **`AEGIS_ML_ENABLE_TABPFN`**, not `AEGIS_ML_TABPFN`. `Settings` uses `env_prefix="AEGIS_ML_"` over the field `enable_tabpfn`.

---

## 4. The portable-recipe mechanism

**This is the keystone of the whole design.** See `docs/10-architecture-decisions.md` D1.

### Why it exists

`backend/pyproject.toml` carries hard caps: `pandas>=2.2,<2.4` (nemoguardrails), `numpy>=1.26,<2.5` (presidio-analyzer, and numba/llvmlite via shap), and `[tool.uv] constraint-dependencies = ["litellm==1.96.0", "presidio-analyzer==2.2.364", "numba==0.67.0"]`. AutoGluon 1.6 + TabPFN-2.5 + torch will not resolve cleanly inside that. **Installing them there is the single most likely way to lose the morning.**

### The bridge

```
.venv-ml (trainer)                                backend/.venv (serving)
──────────────────                                ───────────────────────
AutoGluon · TabPFN · torch · SDV · mlforecast     pandera · skrub · optuna · flaml
                                                  evidently · nannyml · xgboost
                                                  mapie · shap · pandas<2.4
        │                                                    ▲
        │  search runs here                                  │
        ▼                                                    │
   Recipe (JSON) ─────────────────────────────────────────────
   {"task": "regression",
    "members": [{"name": "xgboost", "kind": "XGBRegressor",
                 "params": {...}, "weight": 1.0}, ...],
    "categorical_features": [...], "numeric_features": [...],
    "tier": "flaml", "search_seconds": 118.4,
    "notes": ["autogluon scored 0.71 but is not portable; ..."]}
```

`aegis_ml.automl.runner` shells to `settings.trainer_python` with **parquet in, JSON + joblib out**. If the trainer venv is missing, `TrainerVenvMissingError` names the exact commands that create it.

Then, in the serving venv:

```python
from aegis_ml.automl.recipe import to_aegis_members, load_recipe

recipe  = load_recipe("registry_store/runs/<run_id>/recipe.json")
members = to_aegis_members(recipe, random_state=7)   # list[tuple[str, Estimator]]
```

…and the **Aegis spine** fits those members, keeping MAPIE conformal calibration, SHAP attribution, the `dataset_digest` and the `ModelCard`. **Full AutoML benefit, zero changes inside `aegis/`.**

### The portability allowlist

`recipe.PORTABLE_KINDS` maps estimator class name → the module it may be imported from. It is an **explicit allowlist**, not a dynamic import of whatever a search result asked for:

| Kind | Module |
|---|---|
| `XGBRegressor` / `XGBClassifier` | `xgboost` |
| `HistGradientBoostingRegressor` / `Classifier` | `sklearn.ensemble` |
| `RandomForestRegressor` / `Classifier` | `sklearn.ensemble` |
| `ExtraTreesRegressor` / `Classifier` | `sklearn.ensemble` |
| `LGBMRegressor` / `LGBMClassifier` | `lightgbm` |

**Every entry is a tree learner `shap.TreeExplainer` supports**, because the Aegis spine explains its ensemble member-by-member with exactly that explainer. Adding a non-tree member here would produce a model that trains, scores, promotes — and then raises inside `explain()` on the first request that asks why.

Anything else raises `RecipeNotPortableError`, whose message says what to do instead: *report its leaderboard score as the accuracy ceiling and export it to ONNX for a side-by-side predictor — do not promote it as the spine.*

`coerce_params(kind, params)` drops constructor kwargs the serving-venv version of an estimator does not accept, **returning the dropped names** so the recipe's `notes` can say what was dropped rather than silently changing the model.

`baseline_recipe(problem)` reproduces `aegis.ml.model`'s own defaults exactly — `XGB_PARAMS` (`n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9, n_jobs=1, tree_method="hist"`) and `HGB_PARAMS` (`max_iter=200, max_depth=4, learning_rate=0.1`) — so a "baseline" leaderboard row is an honest floor: **it is the model Aegis would have trained anyway.** Those constants are duplicated deliberately; `aegis` is a sibling checkout, not a dependency of this package, and importing them would couple installation to a filesystem path.

### The non-portable winner

Two things happen, both honest:

1. Its leaderboard score is reported in the model card as the **accuracy ceiling** — "AutoGluon's stacked ensemble reached R² 0.71; the promoted portable ensemble reaches 0.66, so we are giving up 0.05 to keep conformal intervals and SHAP."
2. The fitted model is exported to **ONNX** for an optional side-by-side predictor (`aegis_ml.export.onnx`).

That is a better demo slide than pretending the ceiling does not exist.

---

## 5. Feature handling — `aegis_ml.features.pipeline`

skrub `TableVectorizer` / `tabular_pipeline()`, derived from the same `MLProblem`:

- categoricals → one-hot for low cardinality, `StringEncoder` for high cardinality (skrub's current default; `GapEncoder` was the old one, and `tabular_learner` is deprecated — use `tabular_pipeline`);
- numerics → passthrough with median imputation, matching what the spine does;
- datetimes → calendar features.

> **Alignment matters more than sophistication.** The Aegis spine does its own one-hot encoding of the declared categorical subset. Any pipeline used to *search* must produce a model whose input contract the spine can reproduce, or the recipe is not portable. When in doubt, keep the search pipeline identical to the spine's encoding and let the tiers compete on the estimator, not the preprocessing.

Full skrub features want sklearn ≥ 1.8; the repo carries ≥ 1.5, so pin accordingly and prefer the stable surface.

---

## 6. Caching and resume

Nothing here should ever have to be redone because a later stage failed.

| Artefact | Path | Reused when |
|---|---|---|
| Generated frame | `registry_store/cache/frame-<digest>.parquet` | the `MLProblem` and seed are unchanged |
| Search result | `registry_store/runs/<run_id>/recipe.json` | `--reuse-recipe` |
| Optuna study | `registry_store/hpo/<domain_id>.db` | always — TPE resumes from trial *n* |
| Fitted model | `registry_store/runs/<run_id>/model.joblib` | `eval`/`promote` load it rather than refit |
| Drift reference | `registry_store/runs/<run_id>/drift_ref.parquet` | every `drift_flow` |

Keying on the **dataset digest**, not on a timestamp, is what makes the cache correct: change a generator coefficient and the digest changes and the cache misses, which is exactly right.

```bash
uv run aegis-ml train --reuse-recipe --no-hpo    # refit an existing recipe, fast
uv run aegis-ml eval --run <run_id>              # re-score without refitting
```

---

## 7. Reading the outputs

A completed `full_flow` leaves:

```
registry_store/runs/<run_id>/
├── manifest.json        RunManifest — every stage, its duration, ok/error
├── model.joblib         the fitted TrustworthyModel (spine format)
├── recipe.json          the portable Recipe
├── leaderboard.json     every candidate, winner and loser, with `portable`
├── card.json            ModelCard + TrainResult + GateDecision + slices + licences
├── card.html            the same, rendered
├── shap.html            beeswarm + bar + dependence
├── profile.html         skrub TableReport
├── slices.json          per-level metrics
├── drift_ref.parquet    the exact validated training frame
└── drift.html           Evidently report (after drift_flow)
```

**The five numbers to read, in this order:**

| # | Where | What it must say |
|---|---|---|
| 1 | `card.json` → `metric_name` / `metric_value` | R² in **0.45–0.80**, accuracy in **0.65–0.88**. Above 0.90 is a bug report (`docs/04-synthetic-data.md` §3). |
| 2 | `card.json` → `conformal_coverage` vs `conformal_coverage_empirical` | Requested 0.90, measured within `settings.coverage_tolerance` (0.05). **If measured coverage is far above requested, the intervals are too wide and the model is under-confident.** |
| 3 | `slices.json` → the worst row | No slice collapsed. A model that is great on average and useless for one region is not shippable. |
| 4 | `leaderboard.json` → the margin between rank 1 and `baseline` | Says whether the AutoML search earned its time. A 0.01 margin means ship the baseline. |
| 5 | `card.json` → `dataset_digest` | Present, and matching `drift_ref.parquet`'s digest. Provenance. |

Plus the two flags: `data_source` must be `"provided"` or `"spec_provider"` — **`"synthetic"` means the spine fell back to its own noise synthesiser** and the model carries no domain signal. And `imputed_features` / `unknown_features` on any served `MLExplainResponse` say how much of an answer came from training medians rather than the caller's input.

---

## 8. Next

`docs/06-mlops-registry-drift.md`.
