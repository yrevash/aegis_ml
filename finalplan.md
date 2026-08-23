# aegis_ml — SOTA ML/MLOps adapter factory for Aegis

## Context

**What Aegis is.** `/Users/yrevash/aegis` is a domain-agnostic enterprise agentic-AI platform: an importable library (`aegis/src/aegis/`, 30 packages) plus a FastAPI host (`backend/src/app/`) plus a Next.js console (`web/`). Retargeting it to a new problem means writing exactly one thing — a **domain adapter** satisfying `aegis.adapter.DomainAdapter` (`aegis/src/aegis/adapter.py:492`): 11 members across 10 pieces, verified by `missing_members()` + 14 conformance checks.

**Why this package.** On hackathon day you hand your agent a problem statement. Without a base, it spends the morning re-deriving the adapter contract and hand-rolling ML. `aegis_ml` is the base: a standalone, pip-installable package holding (a) the ML/MLOps machinery Aegis lacks, (b) templates + authoring prompts for all 10 adapter pieces, and (c) a **fully worked reference domain** proving the whole pipeline green before the day.

**Critical constraint discovered during research — Aegis already has a serious ML spine.** `aegis.ml` (`model.py`, 966 lines) is XGBoost + HistGradientBoosting soft-voting, MAPIE split-conformal calibrated on a disjoint split, SHAP TreeExplainer averaged by member weight, `ModelCard` separating *requested* from *empirical* coverage, SHA-256 dataset digests, and `MLModelUnavailableError` instead of silent fallback. `aegis.forecast` is Nixtla StatsForecast (AutoARIMA/AutoETS/SeasonalNaive) with `ConformalIntervals` and rolling-origin backtests. **aegis_ml extends this; it never replaces it.**

**Three corrections to the existing `/Users/yrevash/aegis_ml/PLAN.md` (DeepSeek's), verified against source:**

| Claim | Reality |
|---|---|
| "Conformance checks generator↔latent coupling" | **False.** Zero references to the generator in `test_conformance.py`. A noise target passes all 14 checks. Only `distinct=False` from `python -m app.ml` catches it. We must build this check. |
| "`aegis/src/aegis/` is NEVER touched" | Contradicts itself — §6 orders editing `aegis/src/aegis/conformance/_vocabulary.py`. That edit **is** required and **is** sanctioned by `SKILL.md`. Say it once. |
| ML flows into the agent's reasoning | **Not wired.** `describe_prediction` has zero consumers; `ml_predict` appears in the README's request path but not in `graph.py`'s `NODE_LABELS`. |

Also missing from that plan: the four `web/` console files carrying shipped-domain literals (outside the Python-only vocabulary scan — a demo-killer), the `backend/tests/adapter/*` rewrite, `rsync --delete` vs `cp -r` leaving the old corpus/skills behind, OpenAPI+TS client regeneration, and the ~15 host-bound adapter symbol names beyond the Protocol (`ToolActionResult`, `InMemoryRecordStore`, `GeneratorConfig`, `ToolContext`, `Persona`, `ScopeKind`, `TARGET`, `training_frame`).

---

## Architectural decisions

### D1. Two virtualenvs, one portable "recipe" — the keystone

The backend venv carries hard caps documented in `backend/pyproject.toml`: `pandas>=2.2,<2.4` (nemoguardrails), `numpy>=1.26,<2.5` (presidio + numba/llvmlite via shap), and `[tool.uv] constraint-dependencies = ["numba==0.67.0", "litellm==1.96.0", "presidio-analyzer==2.2.364"]`. AutoGluon 1.6 + TabPFN-2.5 + torch will not resolve cleanly inside that. Installing them there is the single most likely way to lose the morning.

So: **install everything, isolate the heavy half.**

| Venv | Contents | Purpose |
|---|---|---|
| `backend/.venv` (existing) | + `aegis-ml` base only: `pandera`, `skrub`, `optuna`, `flaml`, `evidently`, `nannyml`, `typer` — all pure-Python, all resolvable under the caps | Serving, promotion, drift, the FastAPI router, the adapter tools |
| `aegis_ml/.venv-ml` (new) | `autogluon.tabular[all]`, `autogluon.timeseries`, `tabpfn`, `tabpfn-extensions`, `torch` (CPU), `sdv`, `mlflow`, `mlforecast` — unconstrained resolve | AutoML search, foundation models, synthetic-data fitting |

The bridge is a **portable recipe**: the trainer venv runs the search and returns JSON — `{"members": [{"name": "xgboost", "kind": "XGBRegressor", "params": {...}}, ...], "preprocess": {...}, "leaderboard": [...]}`. `aegis_ml.automl.recipe.to_aegis_members(recipe)` turns that into exactly the `list[tuple[str, Estimator]]` shape `aegis.ml.model._regression_members()` returns, and the Aegis spine fits it — keeping MAPIE conformal, SHAP, the ModelCard, and the dataset digest. **Zero core changes, full AutoML benefit.**

Models that cannot be re-fit in the serving venv (TabPFN, AutoGluon's stacked ensemble) go two ways: their leaderboard scores are reported in the model card as the accuracy ceiling, and the fitted model is exported to ONNX for an optional side-by-side predictor. Honest, and it makes a great demo slide.

### D2. Where ML enters the agent loop — through tools, not the graph

`aegis_ml.serve.tools` ships ready-made `ToolSpec`s that drop into the adapter's `TOOL_REGISTRY`: `predict_outcome` (LOW, read-only, idempotent), `explain_prediction` (LOW), `whatif_scenario` (LOW), `forecast_series` (LOW), `check_model_health` (LOW). The agent calls them; the answers carry the conformal interval and top SHAP drivers rendered by `describe_prediction`. This finally wires the dead code, needs no `graph.py` edit, and respects the platform's stated rule — *ML informs, it never gates; the human gate fires on a tool's risk tier.*

### D3. Registry is filesystem-first; promotion writes the path Aegis already loads

`aegis.ml` loads `backend/.artifacts/ml_spine.joblib`. Promotion = gate passes → atomic replace of that file, previous version retained for rollback. MLflow 3 is an **optional mirror** for the demo UI and lineage, never the source of truth. Optional SQLAlchemy tables (`ml_runs`, `ml_predictions`, `ml_drift_reports`) on `AegisBase` fill the gap the exploration confirmed — nothing ML is persisted relationally today.

### D4. Pipelines are plain Python; Prefect is a decorator

`aegis_ml.pipelines.flows` are ordinary functions over a `Stage` protocol. `prefect_shim.py` applies `@flow`/`@task` when Prefect imports, and identity decorators otherwise. A trained artifact never depends on a server being up.

### D5. Follow Aegis's own discipline, verbatim

- **Light types, heavy impl.** `contracts/` imports pydantic only, with a `test_types_is_dep_free.py` mirroring `aegis/tests/ml/test_types_is_dep_free.py`.
- **Requested vs measured is a naming rule.** Never one field: `requested_coverage` / `empirical_coverage`.
- **Refuse rather than degrade.** Typed errors, never a plausible-looking number.
- **Optional deps via a `require()` that names the exact pip command.** Never `except ImportError: pass`.
- Python 3.11, ruff `E,F,I,UP,B,SIM,ANN,D`, line-length 100, Google docstrings that carry the reasoning.

### D6. TabPFN-2.5 — enabled, license-flagged

Weights are Prior Labs License: research/evaluation permitted, commercial/production not. A hackathon demo is evaluation use. It is on by default; the CLI and every model card it touches print the notice. `AEGIS_ML_TABPFN=0` disables it.

---

## Package layout

```
/Users/yrevash/aegis_ml/
├── README.md                     # human entry point
├── AGENT_CONTEXT.md              # REWRITE — verified facts, corrections above folded in
├── HACKATHON_RUNBOOK.md          # minute-by-minute, verbatim tested commands (bash + PowerShell)
├── pyproject.toml                # aegis-ml; extras: [serve] [strong] [mlops] [dev]
├── uv.lock                       # RESOLVED AND COMMITTED BEFORE THE DAY  ← top risk mitigation
├── Makefile / tasks.ps1
│
├── docs/  00-architecture · 01-domain-authoring · 02-pipelines · 03-data-contracts
│          04-mlops · 05-integration · 06-windows · 07-troubleshooting
│
├── prompts/                      # authoring packs the agent reads on the day
│   ├── 00-intake.md              # problem statement → Domain Brief
│   ├── 01..10-<piece>.md         # one per adapter piece: contract, trap, verify command
│   ├── 11-ml-pipeline.md  12-integration.md  13-console.md  14-final-gate.md
│   └── CHECKLIST.md
│
├── src/aegis_ml/
│   ├── settings.py               # pydantic-settings; AEGIS_ML_* env
│   ├── cli.py                    # typer: doctor|init|contract|synth|train|eval|promote|drift|forecast|card|export
│   ├── contracts/    spec.py · frames.py · protocols.py · errors.py     # pydantic-only
│   ├── data/         synth.py (SDV) · latent.py · splits.py · profile.py (skrub TableReport) · contract_check.py
│   ├── features/     pipeline.py (skrub TableVectorizer) · leakage.py
│   ├── automl/       tiers.py · search.py · hpo.py (Optuna) · recipe.py · runner.py (subprocess bridge)
│   ├── forecast/     engine.py (wraps aegis.forecast) · ml_forecast.py · foundation.py · backtest.py
│   ├── evaluate/     metrics.py · cv.py · calibration.py · slices.py · gate.py
│   ├── explain/      shap_report.py · pdp.py · reason_codes.py · card.py
│   ├── registry/     store.py · promote.py · mlflow_mirror.py · db.py
│   ├── monitor/      log.py · drift.py (Evidently 0.7+) · perf.py (NannyML CBPE) · alerts.py
│   ├── export/       onnx.py
│   ├── serve/        router.py (FastAPI) · tools.py (adapter ToolSpecs)
│   └── pipelines/    flows.py · prefect_shim.py · manifest.py
│
├── templates/adapter/            # the 10 pieces as annotated skeletons + _CHECKLIST.md
├── reference/                    # FULL worked domain — cold-chain logistics
├── config/  automl.toml · forecast.toml · monitoring.toml · pipeline.toml · contracts.toml
└── tests/
```

---

## Component detail

### `contracts/` — one spec, three consumers
`spec.py` defines `FeatureSpec(name, dtype, unit, description)` / `TargetSpec(name, task, unit)` matching the reference adapter's shapes exactly, and `emit_ml_spec_module()` which **generates** `ml_spec.py` with `FEATURES`, `FEATURE_NAMES`, `CATEGORICAL_FEATURES`, `TARGET`, `training_frame`, `describe_prediction` — the exact five members `MLSpecModule` requires and the exact names `resolve_spec()` reads. This kills the highest-cost silent failure in the platform: `aegis/ml/spec.py:137` returns the 4-feature-noise `FALLBACK_SPEC` when `FEATURE_NAMES` or `TARGET.name` is missing, and nothing raises.

`frames.py` derives a **pandera** `DataFrameModel` from the same specs — dtypes, ranges, null policy, categorical level sets — so the training frame is validated at the boundary and drift references are schema-checked.

### `data/latent.py` + the check DeepSeek imagined
`SKILL.md`'s named trap: *"the generator must sample labels around your latent function. If it does not, the target is noise, the model finds nothing, and the conformal interval is honestly enormous."* Nothing enforces it.

`latent.py` builds monotone-driver latent functions with declared coefficients and seeded Gaussian noise, and `tests/test_label_is_learnable.py` fits a fast model on `training_frame(num_records=1200)` and **asserts held-out R² (or accuracy) clears a floor**. Run in CI and by `aegis-ml doctor`. This is the real version of the fabricated check — and it fails in seconds instead of at demo time.

`synth.py` wraps SDV (GaussianCopula default, CTGAN opt-in) for the "we have a real CSV, make 10× more" path, with an SDMetrics quality report. The procedural+LLM hybrid in `templates/adapter/generator.py` stays the primary route, matching `backend/src/app/adapter/generator.py`'s proven pattern.

### `automl/` — four tiers, one leaderboard
`tiers.py` probes availability and runs: `baseline` (sklearn/xgboost, always present) → `flaml` (time-budgeted) → `autogluon` (`best_quality`) → `tabpfn` (2.5, plus `tabpfn-extensions` AutoTabPFN). `search.py` returns **every candidate including losers**, mirroring `aegis.forecast.ForecastResult.candidates`. `hpo.py` runs an Optuna study (TPE + HyperbandPruner, SQLite storage so it resumes) over the winning recipe. `runner.py` shells to `.venv-ml` with parquet in / JSON+joblib out.

### `evaluate/gate.py` — the MLOps heart
A challenger is promoted only if **all** hold, each reported with its number:
1. beats champion on the primary metric by ≥ ε (config)
2. `empirical_coverage ≥ requested_coverage − δ` on the held-out split
3. every pandera contract passes
4. worst slice is no worse than champion's worst slice
5. no target leakage flagged by `features/leakage.py`

Returns a typed `GateDecision(promoted: bool, reasons: list[str], metrics: dict)`. Refuses loudly; never promotes silently.

### `monitor/` — including the part without labels
Evidently 0.7+ (the **modern** `Report`/`Preset` API — DeepSeek's `>=0.4` prose targets the removed 0.4.x surface) for data/target/prediction drift against the registry's stored reference frame. **NannyML CBPE/DLE** estimates live performance *before ground truth arrives* — the strongest single differentiator in the stack for an enterprise-trust demo, and it fits Aegis's thesis exactly.

### `reference/` — cold-chain logistics
A complete non-Aegis domain (lexically far from `service_request_management`, so the vocabulary quarantine is exercised for real): `Shipment`, `Carrier`, `Facility`, `SensorReading`, `Document`. Regression target `spoilage_risk_pct`; secondary classification `excursion_flag`; series *"Shipments dispatched per day"*. Ships all 10 pieces filled, both config sets, and a `make demo` that runs contract → synth → train → eval → promote → drift → card → export and produces real artifacts. Proven green before the day, so the agent pattern-matches working code instead of empty templates.

---

## Files to create

| Group | Count | Notes |
|---|---|---|
| Root docs + packaging | 6 | `README`, `AGENT_CONTEXT` (rewrite), `HACKATHON_RUNBOOK`, `pyproject.toml`, `uv.lock`, `Makefile`/`tasks.ps1` |
| `docs/` | 8 | |
| `prompts/` | 16 | intake + 10 pieces + 4 procedure + checklist |
| `src/aegis_ml/` | ~42 | modules above |
| `templates/adapter/` | 12 | 8 modules + 2 dirs + `__init__.py` + `_CHECKLIST.md` |
| `reference/` | ~16 | full worked domain + corpus/skills content |
| `config/` | 5 | |
| `tests/` | ~18 | |

The `pyproject.toml` extras — `[serve]` (backend-venv-safe, pure Python), `[strong]` (AutoGluon, TabPFN, torch, SDV), `[mlops]` (MLflow, Prefect), `[dev]`.

---

## Integration on hackathon day

Nine steps, all commands tested and living verbatim in `HACKATHON_RUNBOOK.md` (bash + PowerShell):

1. `aegis-ml doctor` — resolved versions, available AutoML tiers, artifact path writable, Postgres reachable.
2. Problem statement → `prompts/00-intake.md` → **Domain Brief** (entities, target, features+dtypes, latent drivers, personas, tools+risk tiers, roster, series label/unit).
3. Fill `templates/adapter/` from the Brief, one piece at a time, in the order `SKILL.md` prescribes — schema → ml_spec → generator → tools → personas → prompts → memory_spec → roster → corpus → skills. *The whole suite is red at import from piece 1 until piece 8 lands. That is expected. Do not chase it.*
4. `aegis-ml contract` — pandera + `test_label_is_learnable` before anything expensive runs.
5. **Sync, don't copy:** `rsync -a --delete` into `backend/src/app/adapter/` (a plain `cp -r` leaves the reference domain's 3 corpus docs and 2 skills behind, and retrieval will serve them).
6. Rewrite `backend/tests/adapter/*` — leaving `test_piece_manifest.py`, `test_domain_adapter_protocol.py`, `test_conformance_suite.py` and `broken_adapter/` untouched.
7. Edit `aegis/src/aegis/conformance/_vocabulary.py` — **the one sanctioned core edit**, required by the quarantine check, and report it.
8. Re-voice the four console files the Python-only scan cannot see: `web/src/config/personas.ts`, `web/src/components/ops/opsShared.ts`, `web/src/components/sim/SimulationView.tsx`, `web/src/components/ml/MLOpsView.tsx`.
9. Train and promote: `python -m app.ml` (spine) → `aegis-ml train --tier all` (AutoML in `.venv-ml`) → `aegis-ml promote` (gate) → `aegis-ml drift`.

---

## Pre-hackathon work, in priority order

1. **Resolve and commit both lockfiles.** The 12-dependency addition has never been resolved against `pandas<2.4` / `numpy<2.5` / `numba==0.67.0`. Do this first; it is the difference between a 5-minute setup and a lost morning.
2. Build `src/aegis_ml/` and the reference domain; get `make demo` green on macOS.
3. Validate the whole path on the Windows box (torch CPU wheel, AutoGluon tabular, no Docker/WSL).
4. Write `HACKATHON_RUNBOOK.md` from commands that actually ran, with real output pasted in.
5. Write `AGENT_CONTEXT.md` last, from verified facts — no pinned test counts (`AGENTS.md` and `README.md` already disagree: 2247 vs 2268, 1121 vs 1174).

---

## Verification

```bash
# 0. environment
cd /Users/yrevash/aegis_ml && uv sync --extra dev && uv run aegis-ml doctor

# 1. package tests, including dep-free types and recipe portability
uv run pytest tests -q

# 2. the reference domain satisfies the real contract
uv run python -c "import reference.adapter as a; from aegis.adapter import DomainAdapter, missing_members; \
print('missing:', missing_members(a)); print('satisfies:', isinstance(a, DomainAdapter))"

cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src:../../aegis_ml/reference \
  .venv/bin/python -m pytest --pyargs aegis.conformance --aegis-adapter reference.adapter -q

# 3. full pipeline end-to-end on the reference domain
cd /Users/yrevash/aegis_ml && make demo
#   → registry/runs/<id>/{model.joblib,card.json,card.html,recipe.json,leaderboard.json,shap.html,drift_ref.parquet}
#   → asserts empirical_coverage >= requested - delta, and prints the gate decision

# 4. the spine still trains and the label is learnable  (distinct=True is the pass signal)
cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml | tail -1

# 5. ML tools reach the agent — call one through the adapter's registry
uv run pytest tests/test_ml_tools_roundtrip.py -q

# 6. lint, matching Aegis's own config
uv run ruff check src reference tests
```

**Definition of done:** `make demo` produces a promoted artifact with measured conformal coverage inside tolerance, the reference adapter passes all 14 conformance checks, `python -m app.ml` prints `distinct=True`, and both lockfiles are committed and reproduce on Windows.

---

## Appendix A — SOTA research findings (August 2026)

Every tooling choice above, with what the current evidence actually says.

### AutoML for tabular

| Library | Status in 2026 | Verdict |
|---|---|---|
| **AutoGluon 1.6** | Benchmark leader; multi-layer stacked ensembles + model stacking rather than HPO alone. `autogluon.tabular` and `autogluon.timeseries` install on Windows (`autogluon.multimodal` does not). Python 3.10–3.13. | **Tier 3** — best raw accuracy, pulls torch |
| **TabPFN-2.5** (arXiv 2511.08667) | Now the leading method on **TabArena**. Handles 50k rows × 2k features (20× TabPFNv2). 100% win rate vs default XGBoost at ≤10k rows / 500 features; 87% at ≤100k. Matches AutoGluon 1.4's four-hour tuned ensemble *out of the box*. Ships a distillation engine → compact MLP/tree for low-latency serving. | **Tier 4, on by default** — this is the "wow". Licence: research/eval only, notice printed |
| **FLAML** | Lightweight, resource-aware, cost-frugal search. Pure Python. | **Tier 2** — the reliable fast path |
| **auto-sklearn** | Linux-only (Unix `resource`, SWIG). | **Excluded** — correct call in the DeepSeek plan |
| **TPOT** | Genetic search, slow, poor fit for a 300 s budget; DeepSeek listed it as a dep with no module and no flow. | **Dropped** |

Hackathon synthetic data is 1k–10k rows — precisely TabPFN-2.5's 100%-win-rate band. That is why it is enabled by default.

### Everything else

- **Optuna 4.x** over Ray Tune. *"Choose Ray for workloads that no longer fit on one machine. Choose Optuna for squeezing the last few points out of a model."* TPE overtakes random search from roughly the 30th trial at 3–5 important hyperparameters. Pure Python, SQLite-resumable studies.
- **skrub** `TableVectorizer` / `tabular_pipeline()` — sklearn-compatible dataframe wrangling; default high-cardinality encoder is now `StringEncoder` (was `GapEncoder`); `tabular_learner` is deprecated. Also gives `TableReport` for free data profiling. Requires sklearn ≥ 1.8 for full features; the repo has ≥ 1.5, so pin accordingly.
- **pandera** over Great Expectations: **12 dependencies vs 107**, and *"closer in spirit to pydantic — run-time enforced type-annotations for your dataframes."* Exactly Aegis's idiom. 0.29 (Jan 2026) supports pandas/polars/dask/pyspark/ibis.
- **Evidently + NannyML**, not either alone. Evidently is the broadest drift/quality/report surface but **requires ground-truth labels for all metrics**. NannyML's CBPE/DLE **estimates performance without labels** — the one capability that makes monitoring real in a demo where labels arrive days later.
- **MAPIE 1.x** is already an Aegis dependency (`mapie>=1.4`) and is the most popular Python conformal library. v1 changed the API significantly vs 0.9; 2026 added adaptive CP methods and exchangeability tests. Extend it, do not re-implement it.
- **Nixtla**: `statsforecast` (already in Aegis, with `ConformalIntervals`) plus **`mlforecast`**, which now supports conformal prediction on *all* models and brings global ML forecasting with automated feature engineering. Adds candidates to the existing engine rather than replacing it.
- **SDV 1.37** (Jul 2026) — GaussianCopula → CTGAN ladder, single/multi-table/sequential, with SDMetrics quality reports. The "we got a real CSV, make 10× more" path.
- **MLflow 3** — best-in-class local registry + UI, one pip install, works against the Postgres you already have. Kept as an **optional mirror**: the filesystem registry stays the source of truth so nothing breaks when the server is down.
- **Prefect 3 / Dagster** — note Prefect acquired Dagster Labs in July 2026. Prefect is the lighter Python-native option, but four cron flows buy nothing in a 24-hour demo. Hence the decorator-shim design: real flows if the server is up, plain functions otherwise.
- **ONNX Runtime** — XGBoost→ONNX inference measured at ~0.029 ms/request on Apple M1. Caveat carried honestly in the docstring: **MAPIE intervals and SHAP attributions do not export**, and `onnxruntime` (CPU) is already in the venv transitively via `fastembed`. Value is the portable point-predictor and the round-trip validation, not a new serving path.

### Sources

- [TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models (arXiv 2511.08667)](https://arxiv.org/abs/2511.08667) · [Prior-Labs/tabpfn_2_5 on Hugging Face](https://huggingface.co/Prior-Labs/tabpfn_2_5) · [The state of Tabular Foundation Models (2026)](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models)
- [Open-source AutoML projects in 2026 (mljar)](https://mljar.com/blog/open-source-automl-projects-in-2026/) · [AutoML frameworks — OpenML benchmark](https://openml.github.io/automlbenchmark/frameworks.html) · [Installing AutoGluon 1.6](https://auto.gluon.ai/stable/install.html)
- [Ray Tune vs Optuna in 2026](https://www.swfte.com/blog/ray-tune-hyperparameter-tuning-guide) · [Optuna vs Ray Tune — ML Journey](https://mljourney.com/hyperparameter-tuning-with-optuna-vs-ray-tune/)
- [skrub TableVectorizer](https://skrub-data.org/stable/modules/default_wrangling/table_vectorizer.html) · [skrub release history](https://skrub-data.org/stable/CHANGES.html) · [scikit-learn 1.8 release notes](https://scikit-learn.org/stable/whats_new/v1.8.html)
- [Pandera vs Great Expectations (endjin)](https://endjin.com/blog/a-look-into-pandera-and-great-expectations-for-data-validation) · [Pandera guide 2026](https://pythondatabench.com/article/data-validation-python-pandera-practical-guide)
- [Evidently vs whylogs vs NannyML — self-hosted monitoring guide 2026](https://www.pistack.xyz/posts/2026-04-29-evidently-vs-whylogs-vs-nannyml-self-hosted-model-monitoring-guide-2026/) · [Open-Source Drift Detection Tools in Action (arXiv 2404.18673)](https://arxiv.org/abs/2404.18673)
- [MAPIE documentation](https://mapie.readthedocs.io/en/latest/) · [scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE)
- [Nixtla mlforecast](https://github.com/Nixtla/mlforecast) · [mlforecast prediction intervals](https://nixtlaverse.nixtla.io/mlforecast/docs/tutorials/prediction_intervals_in_forecasting_models.html) · [Conformal Prediction Algorithms for Time Series Forecasting (arXiv 2601.18509)](https://arxiv.org/pdf/2601.18509)
- [SDV on GitHub](https://github.com/sdv-dev/sdv) · [SDV docs](https://docs.sdv.dev/sdv)
- [Dagster vs Prefect vs Airflow 2026 (Orchestra)](https://www.getorchestra.io/blog/dagster-vs-prefect-vs-airflow-complete-data-orchestration-comparison-2026) · [We Tested 9 MLflow Alternatives (ZenML)](https://www.zenml.io/blog/mlflow-alternatives)
- [sklearn-onnx introduction](https://onnx.ai/sklearn-onnx/introduction.html) · [Converting a LightGBM pipeline to ONNX](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_lightgbm.html)

---

## Appendix B — Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | Dependency resolution fails on the morning (`pandas<2.4`, `numpy<2.5`, `numba==0.67.0` vs torch/AutoGluon/TabPFN) | **High** | Two-venv split (D1) + both lockfiles resolved, committed and Windows-verified in advance. Pre-work item #1. |
| 2 | Generator label is noise → model learns nothing, conformal interval is honestly enormous, demo dies | **High** (nothing catches it) | `data/latent.py` + `test_label_is_learnable` fail in seconds. `aegis-ml doctor` runs it. |
| 3 | `resolve_spec()` silently returns `FALLBACK_SPEC` (4-feature noise) on a misspelled attribute | Medium | `contracts/spec.py` **generates** `ml_spec.py`, so the five names cannot be misspelled. Conformance check #12 is the backstop. |
| 4 | Console still shows the old domain's words after a perfect Python retarget | **High** | Step 8 of the runbook names all four `web/` files explicitly. Outside the Python-only vocabulary scan. |
| 5 | Old corpus/skills survive a `cp -r` and get served by retrieval | Medium | `rsync -a --delete` in step 5. |
| 6 | TabPFN licence questioned by judges | Low | Off-switch (`AEGIS_ML_TABPFN=0`); notice printed on every card; AutoGluon tier matches it given budget. |
| 7 | Prefect/MLflow server unavailable at demo time | Medium | Both are optional shims. Plain-Python flows and the filesystem registry always work. |
| 8 | Adding a genuinely new specialist requires a core `graph.py` edit | Low | Roster falls back to `qa` with a warning, never raises. `SPECIALIST_NODES` has `qa`/`memory`/`team`; a third specialist is the one other sanctioned core edit and must be reported. |
