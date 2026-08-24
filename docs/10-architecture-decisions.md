# 10 · Architecture decisions

Six ADRs, then the tool-by-tool evidence behind each choice, then the risk register. Everything here is defensible to a judge; the sources are at the end.

Format: **Context → Decision → Consequences**.

---

## ADR-D1 · Two virtualenvs, one portable "recipe"

**Status:** accepted. **The keystone of the whole package.**

### Context

`/Users/yrevash/aegis/backend/pyproject.toml` carries hard version caps that the entire backend depends on:

| Cap | Imposed by |
|---|---|
| `pandas>=2.2,<2.4` | nemoguardrails |
| `numpy>=1.26,<2.5` | presidio-analyzer 2.2.364 declares `numpy<2.5`; numba/llvmlite (a shap dependency) have no numpy-2.5 release, and without the cap the resolver drags numba back to an ancient version that cannot build against numpy 2.4 |
| `numba==0.67.0`, `litellm==1.96.0`, `presidio-analyzer==2.2.364` | `[tool.uv] constraint-dependencies` |

AutoGluon 1.6 + TabPFN-2.5 + torch will not resolve cleanly inside that. Installing them there is the single most likely way to lose a hackathon morning.

But the alternative — not using them — gives up the strongest models available for exactly the data size this factory generates.

### Decision

**Install everything; isolate the heavy half. Bridge them with a portable JSON `Recipe`.**

| Venv | Contents | Purpose |
|---|---|---|
| `backend/.venv` (existing) | `+ aegis-ml[serve]`: pandas, numpy, sklearn, joblib, xgboost, mapie, shap, pandera, skrub, optuna, flaml, evidently, nannyml, pyarrow — every one pure-Python or already present, chosen for that reason | Serving, promotion, drift, the FastAPI router, the adapter tools |
| `aegis_ml/.venv-ml` (new) | `aegis-ml[strong]`: autogluon.tabular, autogluon.timeseries, tabpfn, tabpfn-extensions, torch (CPU), sdv, mlforecast, lightgbm, catboost — unconstrained resolve | AutoML search, foundation models, synthetic-data fitting |

The bridge:

```json
{"task": "regression",
 "members": [{"name": "xgboost", "kind": "XGBRegressor", "params": {...}, "weight": 1.0}],
 "categorical_features": [...], "numeric_features": [...],
 "tier": "flaml", "search_seconds": 118.4, "notes": [...]}
```

`aegis_ml.automl.recipe.to_aegis_members(recipe, random_state=...)` turns that into exactly the `list[tuple[str, Estimator]]` shape `aegis.ml.model._regression_members()` returns, and the **Aegis spine** fits it — keeping MAPIE conformal calibration, SHAP attribution, the `ModelCard` and the `dataset_digest`.

`PORTABLE_KINDS` is an **explicit allowlist**, never a dynamic import of whatever a search asked for. Entries are tree and linear learners. The spine explains member-by-member with a per-family dispatch, so explainability no longer gates promotion. It once did, and it cost the best model in a real run: a ridge at 0.7460 was refused while a 0.7379 tree was promoted.

### Consequences

**Good.** Full AutoML benefit with **zero changes inside `aegis/`**. The serving venv keeps its caps. A model that cannot be re-fitted portably is refused (`RecipeNotPortableError`) rather than half-applied. The recipe is inspectable JSON, which is itself a demo artefact.

**Costs.** Two venvs to create. A subprocess boundary with parquet-in / JSON-out. Non-portable winners (TabPFN, AutoGluon's stacked ensemble) cannot be promoted as the spine.

**Mitigation for the last one, and it is honest:** their leaderboard scores are reported in the model card as the **accuracy ceiling**, and the fitted model is exported to ONNX for an optional side-by-side predictor. *"AutoGluon's stack reached 0.71; we promote the portable ensemble at 0.66 because it comes with a calibrated interval and a SHAP explanation."* That is a better slide than pretending the ceiling does not exist.

**Also:** `baseline_recipe()` duplicates `aegis.ml.model`'s `XGB_PARAMS` / `HGB_PARAMS` rather than importing them, because `aegis` is a *sibling checkout*, not a dependency — importing them would couple installation to a filesystem path.

---

## ADR-D2 · ML enters the agent loop through tools, not the graph

**Status:** accepted.

### Context

The Aegis README's request path names an `ml_predict` node. Verified against source: **`aegis/src/aegis/agent/graph.py` declares no such node** — not in `NODE_LABELS`, not in the builder. And `describe_prediction`, the adapter member whose job is to render a prediction into the plan, has **zero consumers** across `backend/src/`, `aegis/src/` and `web/src/`.

The prose describes an intention that was never wired. Wiring it properly means adding a node to `graph.py` — a core edit, in the file that carries `SPECIALIST_NODES`, on the morning of a demo.

### Decision

**Ship ML as adapter *tools*.** `aegis_ml.serve.tools` provides ready-made specs that drop into the domain's own `TOOL_REGISTRY`:

| Tool | Risk | read_only | idempotent |
|---|---|---|---|
| `predict_outcome` | LOW | ✔ | ✔ |
| `explain_prediction` | LOW | ✔ | ✔ |
| `whatif_scenario` | LOW | ✔ | ✔ |
| `forecast_series` | LOW | ✔ | ✔ |
| `check_model_health` | LOW | ✔ | ✔ |

The answers carry the conformal interval and the top signed SHAP drivers, rendered by `describe_prediction`.

`ml_tool_specs(ToolSpec)` is handed the *domain's own* `ToolSpec` class, so the specs are constructed in the domain's shape and this package never imports from `app.*` — honouring invariant 1.

### Consequences

**Good.** Needs **no core edit**. Reuses the tool path, which is already gated, already audited, already streamed to the console, already exposed over MCP. Finally gives `describe_prediction` a consumer. Respects the platform's stated rule — **ML informs, it never gates; the human gate fires on a tool's risk tier.** All five tools are LOW and read-only, so ML never triggers an approval, and the domain's own HIGH-risk writes still do.

**Costs.** The model is consulted when the agent *chooses* to call the tool, rather than on every turn. In practice that is a feature: a turn that does not need a prediction does not pay for one.

**Rejected alternative:** add an `ml_predict` node to `graph.py`. Correct in the long run, wrong on the day — it is a core edit in the most sensitive file in the repository.

---

## ADR-D3 · The registry is filesystem-first; promotion writes the path Aegis already loads

**Status:** accepted.

### Context

Nothing about ML is persisted relationally in Aegis today. `aegis.ml` loads exactly one file: `backend/.artifacts/ml_spine.joblib`. A registry needs to be inspectable, diffable, survivable across a laptop reboot, and independent of any server being up.

### Decision

**The filesystem is the source of truth.** `registry_store/` holds immutable run directories plus a derived, rebuildable `index.json`. There is no champion pointer file: the champion is the run whose `RegistryEntry.stage` is `"production"`, and displaced champions become `"archived"`. Promotion = the five-criterion gate passes → **atomic replace** of `backend/.artifacts/ml_spine.joblib`, with the outgoing artifact's bytes copied into its own run directory first, so rollback restores exactly what was serving.

**MLflow 3 is an optional mirror** (`AEGIS_ML_ENABLE_MLFLOW=1`), for the demo UI and lineage, never the source of truth.

**Three optional SQLAlchemy tables** — `ml_runs`, `ml_predictions`, `ml_drift_reports` — fill the relational gap, off by default. They register on `aegis.data.AegisBase` when it is importable and on a local `MLBase` otherwise; which one happened is recorded on `MLTables.base_origin` rather than guessed.

### Consequences

**Good.** No server dependency in the critical path. A run directory is a self-contained artefact you can zip and hand to a judge. Rollback is a pointer move that never deletes a run. **And critically: this package needs no changes inside `aegis/` to serve a promoted model** — Aegis loads the file it always loaded.

**The trap this avoids, which cost real time:** there are two `DEFAULT_ARTIFACT_PATH` constants. `app.ml`'s resolves to `backend/.artifacts/`, which `get_model()` loads. `aegis.ml`'s resolves *inside the installed library*. Training through the library constant writes where nothing loads from — training appears to succeed and the endpoints keep answering 503, with the two paths differing by a directory nobody looks at. `settings.artifact_path` computes the host path, deliberately.

**Costs.** No multi-user concurrency story. Correct for a hackathon; would need Postgres for a team. If the tables are enabled, invariant 3 applies — they must be added to the RLS plan in the same change.

---

## ADR-D4 · Pipelines are plain Python; Prefect is a decorator

**Status:** accepted.

### Context

Four cron flows buy nothing in a 24-hour demo, and a pipeline framework that must be running for a model to train is a single point of failure at the worst moment. (Note: Prefect acquired Dagster Labs in July 2026, so the two are converging; Prefect is the lighter Python-native option either way.)

### Decision

`aegis_ml.pipelines.flows` are **ordinary functions** over a `Stage` protocol. `prefect_shim.py` applies `@flow` / `@task` when Prefect imports and **identity decorators otherwise**.

### Consequences

**Good.** **A trained artifact never depends on a server being up.** Every flow is debuggable with a plain stack trace and a breakpoint. Real Prefect orchestration, retries and a UI appear for free when the server *is* up.

**Costs.** No distributed execution, no scheduler, no built-in retry when the shim is inactive. All fine at this scale.

**Every flow returns a `RunManifest`** (`run_id`, `flow`, `started_at`, `finished_at`, `stages`, `ok`, `error`), so a partial run says which stage stopped it whether or not Prefect was involved.

---

## ADR-D5 · Follow Aegis's own discipline, verbatim

**Status:** accepted.

### Context

`aegis_ml` is read alongside Aegis by the same agent on the same day. Two dialects means two mental models and twice the friction.

### Decision

| Rule | How it shows up here |
|---|---|
| **Light types, heavy impl.** | `aegis_ml.contracts` imports **pydantic and nothing else**, with `tests/test_types_is_dep_free.py` mirroring `aegis/tests/ml/test_types_is_dep_free.py`. pandera is imported *inside functions* in `contracts/frames.py` so `import aegis_ml.contracts` stays pydantic-only. |
| **Requested vs measured is a naming rule.** | Never one field. `ModelCard.conformal_coverage` / `conformal_coverage_empirical`; `TrainResult.requested_coverage` / `empirical_coverage`; `BacktestReport.requested_coverage` / `empirical_coverage`. NannyML output is spelled `estimated_*` throughout so it is never read as a measurement. |
| **Refuse rather than degrade.** | Nine typed errors in `contracts/errors.py`, each carrying its remedy. `AutoMLTierUnavailableError` exists so that "AutoGluon is not installed" and "AutoGluon found nothing better than the baseline" are never indistinguishable on the leaderboard. |
| **Optional deps via `require()` naming the exact pip command.** | `aegis_ml._require.require(extra, module)`. Never `except ImportError: pass`. `is_available()` exists **only for capability reporting** — never to silently choose a different code path. |
| **Tooling.** | Python 3.11; ruff `E,F,I,UP,B,SIM,ANN,D`; line-length 100; Google docstrings that carry the reasoning. `pyproject.toml` mirrors `aegis/pyproject.toml` exactly, so one `ruff check` across both trees gives one verdict instead of two dialects. |

### Consequences

**Good.** An agent that has read `AGENTS.md` can read this package without a second orientation. Every failure mode in this package fails the same way Aegis's do.

**Costs.** More boilerplate than a scratch script. Deliberate — this is the difference between "the model said 42" and "the model said 42 ± 6 with 91% measured coverage, and here is why."

---

## ADR-D6 · TabPFN-2.5 — enabled by default, licence-flagged

**Status:** accepted.

### Context

TabPFN-2.5 (arXiv 2511.08667) is the leading method on TabArena. It handles 50k rows × 2k features (20× TabPFNv2), has a **100% win rate against default XGBoost at ≤10k rows / 500 features** and 87% at ≤100k, and matches AutoGluon 1.4's four-hour tuned ensemble **out of the box**. Hackathon synthetic data is 1k–10k rows — precisely that 100%-win band.

Its weights are under the **Prior Labs License**: research and evaluation permitted, commercial and production use **not**.

### Decision

**On by default.** A hackathon demo is evaluation use. Every artefact it touches prints the notice, which lives as a module *constant* (`tiers.TABPFN_LICENSE_NOTICE`) rather than a docstring, because it has to be **copied into data** — `Recipe.notes`, `Candidate.detail`, the model card — and a licence condition that only exists in a docstring travels nowhere.

Off-switch: `AEGIS_ML_ENABLE_TABPFN=0`.

Its score is reported as an **accuracy ceiling**, never promoted as the spine — it is not portable, because **the prediction *is* the pretrained transformer.**

### Consequences

**Good.** The single strongest number on the leaderboard, honestly labelled. A one-variable answer if a judge questions the licence. The AutoGluon tier is close given budget, so nothing collapses if it is switched off.

**Costs.** Lives in the trainer venv. Cannot be the promoted model.

> Note the exact env-var spelling: **`AEGIS_ML_ENABLE_TABPFN`**. `Settings` uses `env_prefix="AEGIS_ML_"` over the field name `enable_tabpfn`.

---

# The tool choices

Each with what the current evidence actually says.

## AutoML for tabular

| Library | Status in 2026 | Verdict |
|---|---|---|
| **AutoGluon 1.6** | Benchmark leader. Multi-layer stacked ensembles + model stacking rather than HPO alone. `autogluon.tabular` and `autogluon.timeseries` install on Windows; **`autogluon.multimodal` does not**. Python 3.10–3.13. | **Tier 3** — best raw accuracy, pulls torch |
| **TabPFN-2.5** | Leading method on TabArena. 50k × 2k. 100% win rate vs default XGBoost at ≤10k rows / 500 features; 87% at ≤100k. Matches AutoGluon 1.4's four-hour tuned ensemble out of the box. Ships a distillation engine → compact MLP/tree for low-latency serving. | **Tier 4, on by default.** Licence: research/eval only, notice printed |
| **FLAML** | Lightweight, resource-aware, cost-frugal search. Pure Python. | **Tier 2** — the reliable fast path, and portable |
| **auto-sklearn** | Linux-only (Unix `resource`, SWIG). | **Excluded** |
| **TPOT** | Genetic search, slow; a poor fit for a 300-second budget. | **Dropped** |

**Why four tiers and not one.** `TIER_ORDER = ("baseline", "flaml", "autogluon", "tabpfn")` is load-bearing twice: a time-budgeted search spends its first seconds on the tier guaranteed to produce something *portable*, and leaderboard ties break towards the earlier, cheaper tier. `baseline` reproduces the Aegis spine's own defaults exactly, so a "baseline" row is an honest floor — it is the model Aegis would have trained anyway.

## Hyperparameter optimisation — Optuna 4.x over Ray Tune

> *"Choose Ray for workloads that no longer fit on one machine. Choose Optuna for squeezing the last few points out of a model."*

TPE overtakes random search from roughly the **30th trial** at 3–5 important hyperparameters — which is why the default is 60 trials, not a round number. Pure Python, SQLite-resumable studies (`HyperbandPruner` + TPE), so an interrupted search resumes from trial *n* rather than restarting.

## Feature handling — skrub

`TableVectorizer` / `tabular_pipeline()` — sklearn-compatible dataframe wrangling. Two currency notes that matter: the default high-cardinality encoder is now **`StringEncoder`** (it was `GapEncoder`), and **`tabular_learner` is deprecated** in favour of `tabular_pipeline`. Full features want sklearn ≥ 1.8; the repo carries ≥ 1.5, so pin accordingly and prefer the stable surface. `TableReport` gives free data profiling.

## Data contracts — pandera over Great Expectations

**12 transitive dependencies against 107.** And on idiom: *"closer in spirit to pydantic — run-time enforced type-annotations for your dataframes"* — which is precisely how this codebase already uses pydantic. pandera 0.29 (Jan 2026) supports pandas / polars / dask / pyspark / ibis.

The concrete thing it buys: `aegis.ml.model.train` one-hot-encodes with `handle_unknown="ignore"`, so an **unseen categorical level does not raise** — it encodes to an all-zero block and the row is scored as if the feature were absent. A generator emitting `"REFRIGERATED "` (trailing space) for 3% of rows produces a silently degraded model and no error anywhere in the stack. The contract catches it at the boundary. That is also why `FeatureSpec` **refuses a categorical with no declared `levels`**.

## Monitoring — Evidently **and** NannyML, not either alone

**Evidently** is the broadest drift / quality / report surface, but it **requires ground-truth labels for all performance metrics**. Use the modern `Report` / `Preset` API of 0.7+; prose targeting `>=0.4` describes a removed surface.

**NannyML's CBPE / DLE estimates performance *without labels*** — the one capability that makes monitoring real in a demo where labels arrive days later. It is the strongest single differentiator in this stack for an enterprise-trust story, and it fits Aegis's thesis exactly.

Naming discipline: everything NannyML produces is spelled `estimated_*`, so it is never read as a measurement.

## Conformal prediction — MAPIE 1.x, extended not replaced

Already an Aegis dependency (`mapie>=1.4`) and the most popular Python conformal library. v1 changed the API significantly from 0.9; 2026 added adaptive CP methods and exchangeability tests. **Extend it, do not re-implement it** — the Aegis spine's split-conformal calibration on a disjoint split is already correct, and `ModelCard` already separates requested from measured coverage.

## Forecasting — Nixtla

`statsforecast` is already in Aegis, with `ConformalIntervals` and rolling-origin backtests. **`mlforecast`** is added: it now supports conformal prediction on *all* models and brings global ML forecasting with automated feature engineering. New candidates are **added to the existing engine's candidate list** rather than replacing it, keeping the losers-reported-too behaviour that lets you see the winner's margin.

## Synthetic data from real data — SDV 1.37

GaussianCopula → CTGAN ladder, single / multi-table / sequential, with SDMetrics quality reports. The *"we got a real CSV, make 10× more"* path.

**Not the primary route.** The procedural + latent-function generator is, because it gives you a *declared* ground truth you can check the model recovered. A synthesizer reproduces the relationships in its source — if the source has no signal, neither does the copy, and now you have no ground truth to compare against. Run `assert_learnable` on the real CSV **before** fitting a synthesizer to it.

## Experiment tracking — MLflow 3, as an optional mirror

Best-in-class local registry and UI, one pip install, works against the Postgres you already have. Kept as a mirror so nothing breaks when the server is down — the filesystem registry stays the source of truth. This is the one place in the package where a failure is *tolerated* rather than raised, and it is tolerated because the mirror is by definition redundant.

## Orchestration — Prefect 3 as a decorator

See D4. Prefect acquired Dagster Labs in July 2026; Prefect is the lighter Python-native option. Four cron flows buy nothing in a 24-hour demo, hence the shim.

## Export — ONNX Runtime, for portability only

XGBoost→ONNX inference measures ~**0.029 ms/request** on an Apple M1, and `onnxruntime` (CPU) is already in the backend venv transitively via `fastembed`.

**The caveat, carried honestly in the docstring and the model card: MAPIE intervals and SHAP attributions do not export.** The value is a portable point-predictor and the round-trip validation, **not a new serving path** — swapping serving onto ONNX would silently drop the two things Aegis exists to provide.

---

# Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | Dependency resolution fails on the morning (`pandas<2.4` / `numpy<2.5` / `numba==0.67.0` vs torch/AutoGluon/TabPFN) | **High** | D1's two-venv split, **plus both lockfiles resolved, committed and Windows-verified in advance.** This is pre-work item #1 and it is the difference between a five-minute setup and a lost morning. |
| 2 | Generator label is noise → the model learns nothing, the conformal interval is honestly enormous, the demo dies | **High** — *nothing in Aegis catches it* | `aegis_ml.data.latent.assert_learnable` fails in seconds. `aegis-ml doctor` and `aegis-ml contract` both run it. Wire it into the rewritten `tests/adapter/test_ml_spec.py`. |
| 3 | `resolve_spec()` silently returns `FALLBACK_SPEC` on a misspelled attribute | Medium | `contracts/spec.py` **generates** `ml_spec.py`, so the five names cannot be misspelled. Conformance check #12 is the backstop, not the plan. |
| 4 | The console still shows the old domain's words after a perfect Python retarget | **High** | Step 7 of the procedure names all four `web/` files explicitly. They are outside the Python-only vocabulary scan. |
| 5 | Old corpus documents and skill playbooks survive a `cp -r` and get served by retrieval | Medium | `rsync -a --delete` (`robocopy /MIR` on Windows), plus an explicit `ls` check afterwards. |
| 6 | TabPFN licence questioned by judges | Low | Off-switch `AEGIS_ML_ENABLE_TABPFN=0`; notice printed on every card; the AutoGluon tier matches it given budget. |
| 7 | Prefect or MLflow unavailable at demo time | Medium | Both are optional shims. Plain-Python flows and the filesystem registry always work. |
| 8 | Adding a genuinely new specialist requires a core `graph.py` edit | Low | The roster falls back to `qa` with a warning, never raises. Re-voice `qa` and `memory` instead. A third specialist is the one *other* sanctioned core edit and must be reported. |
| 9 | Empirical coverage far below requested at demo time | Medium | The promotion gate refuses it (criterion 2) rather than shipping it. |
| 10 | A judge asks why the promoted model is not the best one on the leaderboard | Medium | Answer it before they ask: `Recipe.notes` and the card carry the ceiling explicitly. This is a strength, not a gap. |

---

# Sources

- **TabPFN-2.5** — [arXiv 2511.08667](https://arxiv.org/abs/2511.08667) · [Prior-Labs/tabpfn_2_5 on Hugging Face](https://huggingface.co/Prior-Labs/tabpfn_2_5) · [The state of Tabular Foundation Models (2026)](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models)
- **AutoML** — [Open-source AutoML projects in 2026 (mljar)](https://mljar.com/blog/open-source-automl-projects-in-2026/) · [AutoML frameworks — OpenML benchmark](https://openml.github.io/automlbenchmark/frameworks.html) · [Installing AutoGluon 1.6](https://auto.gluon.ai/stable/install.html)
- **HPO** — [Ray Tune vs Optuna in 2026](https://www.swfte.com/blog/ray-tune-hyperparameter-tuning-guide) · [Optuna vs Ray Tune — ML Journey](https://mljourney.com/hyperparameter-tuning-with-optuna-vs-ray-tune/)
- **skrub / sklearn** — [skrub TableVectorizer](https://skrub-data.org/stable/modules/default_wrangling/table_vectorizer.html) · [skrub release history](https://skrub-data.org/stable/CHANGES.html) · [scikit-learn 1.8 release notes](https://scikit-learn.org/stable/whats_new/v1.8.html)
- **Data contracts** — [Pandera vs Great Expectations (endjin)](https://endjin.com/blog/a-look-into-pandera-and-great-expectations-for-data-validation) · [Pandera guide 2026](https://pythondatabench.com/article/data-validation-python-pandera-practical-guide)
- **Monitoring** — [Evidently vs whylogs vs NannyML — self-hosted monitoring guide 2026](https://www.pistack.xyz/posts/2026-04-29-evidently-vs-whylogs-vs-nannyml-self-hosted-model-monitoring-guide-2026/) · [Open-Source Drift Detection Tools in Action (arXiv 2404.18673)](https://arxiv.org/abs/2404.18673)
- **Conformal prediction** — [MAPIE documentation](https://mapie.readthedocs.io/en/latest/) · [scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE)
- **Forecasting** — [Nixtla mlforecast](https://github.com/Nixtla/mlforecast) · [mlforecast prediction intervals](https://nixtlaverse.nixtla.io/mlforecast/docs/tutorials/prediction_intervals_in_forecasting_models.html) · [Conformal Prediction Algorithms for Time Series Forecasting (arXiv 2601.18509)](https://arxiv.org/pdf/2601.18509)
- **Synthetic data** — [SDV on GitHub](https://github.com/sdv-dev/sdv) · [SDV docs](https://docs.sdv.dev/sdv)
- **Orchestration / tracking** — [Dagster vs Prefect vs Airflow 2026 (Orchestra)](https://www.getorchestra.io/blog/dagster-vs-prefect-vs-airflow-complete-data-orchestration-comparison-2026) · [We Tested 9 MLflow Alternatives (ZenML)](https://www.zenml.io/blog/mlflow-alternatives)
- **Export** — [sklearn-onnx introduction](https://onnx.ai/sklearn-onnx/introduction.html) · [Converting a LightGBM pipeline to ONNX](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_lightgbm.html)

---

## In-repo authorities

| Question | File |
|---|---|
| What is the adapter contract? | `/Users/yrevash/aegis/aegis/src/aegis/adapter.py` |
| How do I retarget? | `/Users/yrevash/aegis/SKILL.md` — **the** procedure; nothing else supersedes it |
| What are the invariants? | `/Users/yrevash/aegis/AGENTS.md` |
| What does the ML spine actually do? | `/Users/yrevash/aegis/aegis/src/aegis/ml/model.py`, `types.py`, `spec.py` |
| What do the fourteen checks check? | `/Users/yrevash/aegis/aegis/src/aegis/conformance/test_conformance.py` |
| What may the core not know? | `/Users/yrevash/aegis/aegis/src/aegis/conformance/_vocabulary.py` |
| What are the real caps? | `/Users/yrevash/aegis/backend/pyproject.toml` |
| What does a filled adapter look like? | `/Users/yrevash/aegis/backend/src/app/adapter/` |
