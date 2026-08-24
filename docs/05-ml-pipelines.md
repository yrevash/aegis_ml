# 05 · ML pipelines

How data becomes a promoted, monitored model. Module map: `aegis_ml.pipelines`, `aegis_ml.automl`, `aegis_ml.features`, `aegis_ml.evaluate`, `aegis_ml.explain`, `aegis_ml.forecast`.

---

## 1. The flows

`aegis_ml/src/aegis_ml/pipelines/flows.py`. Ordinary Python functions over a `Stage` protocol. Prefect is a **decorator**, never a requirement — `prefect_shim.py` applies `@flow` / `@task` when Prefect imports and identity decorators otherwise. **A trained artifact never depends on a server being up.**

Seven flow functions are exported from `flows.py`: `data_flow`, `train_flow`, `eval_flow`, `promote_flow`, `drift_flow`, `forecast_flow`, `full_flow`. `data_flow` is not a separate CLI step — `train_flow` runs it as its own first half.

| Flow | Stages (the `StageSpec` names in the manifest) | Produces |
|---|---|---|
| `data_flow` | ingest → contract → profile → learnability → realism → leakage → split → digest → freeze_reference | a `DataBundle`: validated frame, three splits, `reference.parquet`, `profile.html` |
| `train_flow` | `data_flow` → search → hpo → fit → measure → slices → shap → card → register | `model.joblib`, `recipe.json`, `leaderboard.json`, `metrics.json`, `card.md`, `card.html`, `shap.html`, `entry.json` |
| `eval_flow` | load_run → ingest → rescore → slices | a re-measured `TrainResult` and `manifest_eval.json` |
| `promote_flow` | gate → apply | `GateDecision`, the updated registry entry, `backend/.artifacts/ml_spine.joblib` |
| `drift_flow` | load_reference → drift → estimate_performance → alerts | `DriftReport`, `<run_id>_drift.html` and `<run_id>_drift.json` under `reports_dir` |
| `forecast_flow` | forecast → rank | a `ForecastRun` plus its JSON under `reports_dir/forecasts/` |
| `full_flow` | `train_flow` → `promote_flow` → `drift_flow` | everything above under **one** manifest, plus `RUN_SUMMARY.md` |

Each flow **writes** a `RunManifest` (`run_id`, `flow`, `started_at`, `finished_at`, `stages`, `ok`, `error`) into the run directory — `manifest.json` for `train_flow`/`full_flow`, `manifest_eval.json`, `manifest_promote.json`, `manifest_drift.json` for the others — so a partial run says which stage stopped it. What a flow *returns* is its own typed result: `DataBundle`, `TrainResult`, `GateDecision`, `DriftReport`, or (for `full_flow`) a JSON-safe dict carrying all of them.

### CLI

```bash
cd /Users/yrevash/aegis_ml

uv run aegis-ml doctor                              # environment, tiers, paths, learnability
uv run aegis-ml init --domain-id <domain_id> \
    --out problem.json --templates ./adapter        # problem scaffold + adapter templates
uv run aegis-ml contract --data frame.parquet       # pandera + assert_learnable + leakage
uv run aegis-ml synth --data real.csv \
    --out synthetic.parquet --rows 5000             # SDV path (needs .venv-ml)
uv run aegis-ml train --tier baseline --tier flaml  # the AutoML search + fit
uv run aegis-ml eval --run-id <run_id>              # re-score on fresh data
uv run aegis-ml promote --run-id <run_id>           # the gate; writes ml_spine.joblib on pass
uv run aegis-ml rollback --domain-id <domain_id>    # restore the previous champion
uv run aegis-ml drift --run-id <run_id> \
    --data live.parquet                             # Evidently + NannyML
uv run aegis-ml forecast --data series.csv \
    --label "Shipments dispatched per day"          # the domain demand series
uv run aegis-ml card --run-id <run_id>              # render a model card
uv run aegis-ml export --run-id <run_id>            # portable ONNX point-predictor
uv run aegis-ml registry                            # every run, newest first
```

`--tier` is **repeatable and takes tier names** (`baseline`, `flaml`, `autogluon`, `tabpfn`). There is no `--tier all`: omitting the flag already runs every tier that is installed and enabled, and records a reason for each one it skips. Add `--full` to `train` to promote and drift-check in the same run (`full_flow`).

PowerShell is identical apart from `Set-Location` instead of `cd`; see `docs/08-windows.md`.

---

## 2. Stage by stage

### 2.1 `profile` — `aegis_ml.data.profile.profile_frame`

`profile_frame(frame, *, out_html=..., title=...)` returns a JSON-safe summary (`n_rows`, `n_columns`, `duplicate_rows`, `memory_bytes`, per-column `columns`, `findings`, `html_path`) and, when `out_html` is given, writes skrub's `TableReport` there. Free, fast, and it is the tab you open when a judge asks "what does your data look like?".

> The function is `profile_frame`, not `profile`: `aegis_ml.data.__init__` re-exports it, and a function re-exported under its own module's name would shadow `aegis_ml.data.profile` on the package. The **module** path is unchanged.

The realism band is a separate measurement: `realism_report` lives in `aegis_ml.data.latent`, not here (see `docs/04-synthetic-data.md` §7.3). The `realism` stage in `data_flow` is the one that calls it.

### 2.2 `contract` — `aegis_ml.contracts.frames`

Derives a pandera `DataFrameSchema` from the **same `MLProblem`** the adapter's `ml_spec.py` was generated from, so the contract cannot drift from the spec that trains the model.

Enforced per column: dtype (`numeric`→`float64`, `boolean`→`bool`, `datetime`→`datetime64[ns]`, `categorical`→`str`), `minimum`/`maximum` as inclusive range checks, `nullable`, and — for categoricals — **the declared level set**.

> The level check is the point. `aegis.ml.model.train` one-hot-encodes with `handle_unknown="ignore"`: an unseen level **does not raise**, it encodes to an all-zero block and the row is scored as if the feature were absent. `numeric` maps to `float64` rather than a permissive union deliberately, so a column of stringified numbers (the classic CSV round-trip failure) is coerced and range-checked instead of being one-hot-encoded as a categorical by accident.

Two functions: `schema_for(problem, *, include_target=True, coerce=True, strict=False)` builds the pandera `DataFrameSchema`, and `validate(frame, problem, *, include_target=True, strict=False)` applies it and returns the coerced frame. `strict` defaults to `False` because a *training* frame legitimately carries columns the model does not use; pass `strict=True` for a **serving** payload, where an extra column is a caller bug worth naming.

Failure reports the first `_MAX_REPORTED_FAILURES = 20` violations with column, check and offending value.

### 2.3 `learnable` — `aegis_ml.data.latent`

`assert_learnable`. Raises `LabelNotLearnableError`. **This is the stage that saves the demo.** See `docs/04-synthetic-data.md`.

### 2.4 `leakage` — `aegis_ml.features.leakage`

Three passes per feature, all of them **against the target**, each producing a `LeakSignal(feature, kind, score, threshold, detail)`:

| `kind` | What it measures |
|---|---|
| `duplicate` | share of rows where feature and target hold the same value — the label stored twice, usually a bad join |
| `correlation` | \|Pearson r\| against a regression target (numeric features only) |
| `single_feature` | held-out score of a model fitted on that feature alone |

Anything at or above `settings.leakage_threshold` (0.98) is a signal; `assert_no_leakage` turns the strongest into `TargetLeakageError` naming the feature and its score, with the remedy: drop it from `FEATURES`, or declare it intentional if it is genuinely available at prediction time. `detect_leakage` needs at least `MIN_LEAKAGE_ROWS` (40) labelled rows and refuses rather than emitting a finding a small sample cannot support.

Constant and identifier-like columns are reported separately, as advisory `findings` from `profile_frame` (§2.1) — they are not leakage and do not raise.

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

The `digest` and `freeze_reference` stages write the exact validated training frame to `registry_store/runs/<run_id>/reference.parquet` alongside its SHA-256 digest. `drift_flow` compares live data against **this** frame, not against a fresh generation.

### 2.7 `search` — `aegis_ml.automl.search.run_search`

`run_search(frame, problem, *, tiers=None, time_budget=None, seed=None) -> (Recipe, Leaderboard)`. See §3.

> The function is `run_search`, not `search`, for the same shadowing reason as `profile_frame` (§2.1). The module path `aegis_ml.automl.search` is unchanged.

### 2.8 `hpo` — `aegis_ml.automl.hpo`

An Optuna study over the winning recipe. **TPE sampler + HyperbandPruner, SQLite storage so it resumes.** Defaults: `settings.hpo_trials = 60`, `settings.hpo_timeout = 600`.

TPE overtakes random search from roughly the 30th trial at 3–5 important hyperparameters, so 60 trials is a sensible floor, not a round number. Under time pressure, skip HPO entirely: `--no-hpo`. The search tiers already give you most of the gain.

The study database lives at `registry_store/optuna/studies.db` — **one SQLite file, one Optuna study per domain inside it** (`hpo.storage_url()` and `hpo.study_name_for()` are the two functions that decide this). Re-running resumes rather than restarting, which is exactly what you want when the first attempt was interrupted.

`tune` never silently ships a regression: if no trial beats the incoming recipe, it returns that recipe **unchanged** with a note saying so, rather than handing back "the best trial".

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

**Slices** (`evaluate/slices.py`) recompute the primary metric restricted to each level of each categorical feature, returning `SliceMetric(feature, level, n_rows, metric_name, metric_value)`. `slice_report()` wraps them with the `SkippedSlice`s that were too small to score, and `worst_slice()` picks the one the gate reads — not the mean: a model that improves on average while collapsing on one region is a regression for everyone in that region, and an aggregate score is exactly the instrument that cannot see it.

### 2.11 `shap`, `card` — `aegis_ml.explain`

- `shap_report.py` — `global_importance` / `local_explanation`, rendered to a self-contained `shap.html` by its `render_html`. Reads the spine's own `shap_attribution`, so what you show is what the agent was told.
- `pdp.py` — `partial_dependence_curves` for the top drivers. This is where an irrelevant feature visibly flatlines.
- `reason_codes.py` — `build_reason_codes` gives top-k signed drivers per prediction as human sentences; `describe_prediction_text` renders the whole decision-support block, and `emit_describe_prediction_source` generates the adapter's own `describe_prediction`.
- `card.py` — `build_card` returns an `ExtendedModelCard` (`TrainResult` fields plus `metrics`, `coverage`, `slices`, `worst_slice`, `top_features`, `recipe`, `leaderboard`, `gate`, `drift`, `limitations`, `notes`, and the host's `aegis_card` nested whole). `render_markdown` and `render_html` write `card.md` and `card.html` into the run directory. **There is no `card.json` on disk** — `aegis-ml card --run-id <id> --format json` prints the JSON on demand, and `metrics.json` carries the headline numbers for a machine.

**Anything TabPFN touched carries `TABPFN_LICENSE_NOTICE` in the card.** It is a module constant precisely so it can be *copied into data* — `Recipe.notes`, `Candidate.detail`, the card — not merely read by a developer.

### 2.12 Forecasting — `aegis_ml.forecast`

`engine.py` **wraps** `aegis.forecast` rather than replacing it: Nixtla StatsForecast (AutoARIMA / AutoETS / SeasonalNaive) with `ConformalIntervals` and rolling-origin backtests, fed by your adapter's `domain_series_events()`.

`ml_forecast.py` adds Nixtla **`mlforecast`** candidates through `ml_candidates()` — global ML forecasting with automated lag/rolling feature engineering, scored on the *same* windows as the statistical models so "would gradient boosting on lags have beaten AutoETS?" gets a measured answer. `backtest.py` runs the rolling-origin evaluation across every candidate and is the module that **refuses to random-split a series at all** (`RandomSplitRefusedError`).

The package is three modules — `engine.py`, `ml_forecast.py`, `backtest.py` — and its public names are `forecast`, `ml_candidates`, `backtest`, `rank_candidates`, `summarise`, `to_forecast_run`, plus the `ForecastRun` / `ForecastCandidate` / `ForecastPoint` / `SeriesObservation` / `BacktestSummary` shapes. All of them are lazily imported, so `import aegis_ml.forecast` costs nothing without the forecasting stack installed.

New candidates are **added to the existing engine's candidate list**, keeping `BacktestSummary.requested_coverage` / `empirical_coverage` (plus its `coverage_meets_request` flag) and the losers-reported-too behaviour: `BacktestSummary.candidates` holds every model scored, and `excluded_models` says why anything was left out.

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

### Available, disabled, unavailable — different states, different reasons

```python
resolve_tiers(requested) -> (to_run: list[TierName], skipped: dict[str, str])
```

The **single** place that decides which tiers run is also the single place that produces the `tiers_skipped` map, so it is impossible to drop a tier without writing down why. An unknown tier name (a typo in `--tier`) is skipped *with a reason*, not silently dropped.

| State | Reason string (verbatim from `tiers.unavailable_reason` / `resolve_tiers`) | Switch |
|---|---|---|
| available | — | — |
| **disabled by policy** | `"disabled by settings.enable_tabpfn (AEGIS_ML_ENABLE_TABPFN=0) — this is a policy choice, not a missing dependency"` | `settings.enable_flaml` / `enable_autogluon` / `enable_tabpfn` |
| **not installed** | `"not importable in this interpreter: <module>. Install with `uv pip install '<TIER_EXTRAS[tier]>'`, or run the search through aegis_ml.automl.runner…"` | `uv pip install 'aegis-ml[strong]'` |
| **installed but unusable** | TabPFN only: importable, but no weights and `TABPFN_TOKEN` unset — `.fit()` would raise `TabPFNLicenseError` mid-search | see §11 of `ISSUES.md` |
| not requested | `"not requested by the caller"` | `--tier` |
| unknown name | `"unknown tier 'xyz'; known tiers are ['baseline', 'flaml', 'autogluon', 'tabpfn']"` | fix the `--tier` typo |

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

`aegis_ml.automl.runner.run_in_trainer_venv` shells to `settings.trainer_python`, running `python -m aegis_ml.automl._worker <dir>`: **`frame.parquet` + `request.json` in, `recipe.json` + `leaderboard.json` out** — no pickle and no joblib crosses the boundary, which is the entire point. On failure the child writes `error.json` with its full traceback and exits non-zero, so a search that dies twelve minutes in surfaces the child's exception rather than a truncated stderr tail. If the trainer venv is missing, `TrainerVenvMissingError` names the exact commands that create it.

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

**Entries are tree *and* linear learners.** The spine explains member-by-member, dispatching
per family — `TreeExplainer` for trees (exact, no background), `LinearExplainer` for linear
models, `PermutationExplainer` for anything else — so "can this be explained?" no longer
decides "can this be promoted?".

This was not always true, and the old behaviour is worth knowing: the allowlist was
tree-only, so a ridge regression that scored **0.7460 — the best score in a real run** — was
refused promotion and a model scoring 0.7379 was promoted instead. A tooling limitation was
choosing the winner. Linear members carry `SimpleImputer(median) → StandardScaler` in front
of the estimator, because unlike the tree learners they have no native NaN path and the data
deliberately carries ~4% missingness.

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

**Two preprocessing paths, and only one of them may change.** The module's four public names are `column_transformer`, `skrub_pipeline`, `encode_frame` and `cast_declared_categoricals`.

`column_transformer(problem)` is the **portable** one, and it is a deliberate mirror of what `aegis.ml.model.TrustworthyModel._build_preprocessor` returns: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` over the declared categoricals, passthrough over everything else, `remainder="drop"`, `verbose_feature_names_out=False`. That identity is load-bearing twice — the recipe crosses a venv boundary and is re-fitted by the spine's own preprocessor, and `aegis.ml.model._encoded_parents` reconstructs SHAP parentage by walking the categorical blocks in declared order and then the numeric passthroughs. Change the order, add a scaler, or flip `verbose_feature_names_out` and either the attribution maps to the wrong parent or the fit fails outright.

`skrub_pipeline(problem, *, high_cardinality_threshold=...)` is the **richer, exploratory** one: skrub's `TableVectorizer` — one-hot below the cardinality threshold, `StringEncoder` above it (skrub's current default, replacing `GapEncoder`), datetime expansion into calendar components, and dtype inference on undeclared columns. `cast_declared_categoricals` runs first so a coded categorical is not inferred as numeric.

> **Alignment matters more than sophistication.** A score obtained under `TableVectorizer` is an **accuracy ceiling** in exactly the sense `Candidate.portable=False` means elsewhere in this package — never a number to promote as the spine's. When in doubt, keep the search pipeline identical to the spine's encoding and let the tiers compete on the estimator, not the preprocessing.

Full skrub features want sklearn ≥ 1.8; the repo carries ≥ 1.5, so pin accordingly and prefer the stable surface.

---

## 6. Caching and resume

Nothing here should ever have to be redone because a later stage failed.

| Artefact | Path | Reused when |
|---|---|---|
| Search result (`StageCache`) | `registry_store/_cache/` | the frame digest **and** the search configuration are unchanged; `--force` bypasses it |
| Search result (explicit) | `registry_store/runs/<run_id>/recipe.json` | `--resume-from <run_id>` adopts that run's recipe and leaderboard |
| Optuna study | `registry_store/optuna/studies.db` | always — TPE resumes from trial *n*, one study per domain |
| Fitted model | `registry_store/runs/<run_id>/model.joblib` | `eval`/`promote` load it rather than refit |
| Drift reference | `registry_store/runs/<run_id>/reference.parquet` | every `drift_flow` |

Keying on the **dataset digest**, not on a timestamp, is what makes the cache correct: change a generator coefficient and the digest changes and the cache misses, which is exactly right. `--resume-from` goes further and **refuses** when the named run's dataset digest differs from this run's (`ResumeMismatchError`) — adopting its recipe would attribute one dataset's search to another's data, and the model card would then name a digest the search never saw.

```bash
uv run aegis-ml train --resume-from <run_id> --no-hpo   # refit an existing recipe, fast
uv run aegis-ml eval --run-id <run_id> --data fresh.parquet
```

---

## 7. Reading the outputs

A completed `full_flow` leaves:

```
registry_store/runs/<run_id>/
├── manifest.json        RunManifest — every stage, its duration, ok/error
├── manifest_promote.json / manifest_eval.json / manifest_drift.json   (per later flow)
├── entry.json           the authoritative RegistryEntry: TrainResult, gate, every path
├── model.joblib         the fitted TrustworthyModel (spine format)
├── recipe.json          the portable Recipe
├── leaderboard.json     every candidate, winner and loser, with `portable`
├── metrics.json         the headline numbers, flat and machine-readable
├── card.md              the model card
├── card.html            the same, rendered
├── shap.html            global importance + local explanations
├── profile.html         skrub TableReport
├── problem.json         the MLProblem this run was fitted against
├── gate_inputs.json     contract + leakage status, recorded for the gate
└── reference.parquet    the exact validated training frame
```

`profile.html` and the drift report also land under `registry_store/reports/` (`<domain_id>_profile.html`, `<run_id>_drift.html`, `<run_id>_drift.json`); `RUN_SUMMARY.md` is written by `full_flow` into the run directory.

**The five numbers to read, in this order:**

| # | Where | What it must say |
|---|---|---|
| 1 | `metrics.json` → `metric_name` / `metric_value` | R² in **0.45–0.80**, accuracy in **0.62–0.92** (`flows.REALISM_R2_BAND` / `REALISM_ACCURACY_BAND`). Above the band is a bug report (`docs/04-synthetic-data.md` §3). |
| 2 | `metrics.json` → `requested_coverage` vs `empirical_coverage` | Requested 0.90, measured within `settings.coverage_tolerance` (0.05). **If measured coverage is far above requested, the intervals are too wide and the model is under-confident.** |
| 3 | `entry.json` → `result.slices`, or the card's worst-slice row | No slice collapsed. A model that is great on average and useless for one region is not shippable. |
| 4 | `leaderboard.json` → the margin between rank 1 and `baseline` | Says whether the AutoML search earned its time. A 0.01 margin means ship the baseline. |
| 5 | `metrics.json` → `dataset_digest` | Present, and matching `reference.parquet`'s digest. Provenance. |

Plus the provenance string. `ExtendedModelCard.data_source` carries `DataBundle.provenance` verbatim, and `_resolve_frame` mints exactly four shapes: `"caller"`, `"callable:<name>"`, `"csv:<path>"` / `"parquet:<path>"`, and `"champion_reference:<run_id>"`. The last one means the frame was read off a previous champion's frozen reference parquet — **not fresh data**, and the manifest records `provenance_is_fresh = 0.0` for exactly that case. There is no fifth branch: with no frame, no source and no champion, `data_flow` raises `FrameSourceMissingError` rather than synthesising anything.

On the serving side, `imputed_features` / `unknown_features` on any `MLExplainResponse` say how much of an answer came from training medians rather than the caller's input, and `data_source == "synthetic"` there means the **spine** fell back to its own noise synthesiser — a model carrying no domain signal at all.

---

## 8. Next

`docs/06-mlops-registry-drift.md`.
