# aegis_ml — SOTA ML Adapter Factory for Aegis Enterprise Agentic Platform

## Purpose

On hackathon day, you receive a problem statement. You hand it to your coding agent
(Claude Opus 5) along with this directory. The agent reads `AGENT_CONTEXT.md` to
understand what Aegis is and how to integrate, fills in the 10 domain template files
under `domain/`, configures the AutoML system via `config/*.toml`, and produces a
fully retargeted Aegis adapter with a production-grade ML pipeline — schema, tools,
personas, prompts, memory, roster, synthetic data generator, knowledge corpus, skills,
AutoML training, hyperparameter optimization, conformal prediction, SHAP explainability,
time-series forecasting, ONNX export, model registry, and drift monitoring.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │     PROBLEM STATEMENT            │
                    │  (given on hackathon day)        │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       CODING AGENT               │
                    │    (Claude Opus 5)               │
                    │                                   │
                    │  Reads AGENT_CONTEXT.md           │
                    │  Fills in domain/ templates       │
                    │  Sets config/*.toml               │
                    │  Writes tests/                    │
                    └──────────────┬──────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
┌─────────▼─────────┐  ┌───────────▼──────────┐  ┌────────▼──────────┐
│   domain/          │  │   ml/                 │  │   config/          │
│   (10 pieces)      │  │   (AutoML engine)     │  │   (declarative)    │
│                    │  │                       │  │                    │
│ schema.py          │  │ automl.py (FLAML)     │  │ automl.toml        │
│ ml_spec.py         │  │ hpo.py (Optuna)       │  │ forecast.toml      │
│ generator.py       │  │ forecast.py           │  │ monitoring.toml    │
│ tools.py           │  │ evaluate.py           │  │ pipeline.toml      │
│ personas.py        │  │ explain.py (SHAP)     │  │                    │
│ prompts.py         │  │ export.py (ONNX)      │  └────────────────────┘
│ memory_spec.py     │  │ monitor.py (Evidently)│
│ roster.py          │  │ card.py (Model Card)  │
│ corpus/*.md        │  │ registry.py           │
│ skills/*.md        │  │ pipelines.py (Prefect)│
│                    │  │ train.py              │
└────────┬───────────┘  └───────────┬──────────┘
         │                          │
         │      COPY ON DAY         │      MERGE ON DAY
         │                          │
┌────────▼──────────────────────────▼──────────────────────────┐
│           backend/src/app/adapter/   (Aegis Reference Host)   │
│                                                               │
│  Domain logic lives here. The core never learns the domain.   │
│  ML pipeline runs from this directory.                        │
└───────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │          AEGIS CORE (aegis/src/aegis/)    │
                    │                                            │
                    │  aegis.ml         XGBoost + MAPIE + SHAP  │
                    │  aegis.forecast   Nixtla ARIMA+Conformal  │
                    │  aegis.evals      RAGAS-style metrics      │
                    │  aegis.agent      LangGraph orchestration  │
                    │  aegis.gateway    LiteLLM chokepoint       │
                    │  aegis.guardrails NeMo + Presidio          │
                    │  aegis.retrieval  Qdrant + Neo4j + LightRAG│
                    │  aegis.memory     Postgres episodic/semantic│
                    │  aegis.governance RLS + JWT + audit        │
                    │                                            │
                    │  UNTOUCHED on hackathon day               │
                    └────────────────────────────────────────────┘
```

## Directory Layout

```
aegis_ml/
├── PLAN.md                        # THIS FILE — the full plan
├── AGENT_CONTEXT.md               # The bridge: everything the agent needs to know
├── pyproject.toml                 # Optional deps for backend/ (agent copies relevant lines)
├── prefect.yaml                   # Prefect 3.x ephemeral deployment config
│
├── domain/                        # THE 10 DOMAIN BLANKS — agent fills these
│   ├── __init__.py                # Registry re-exports (STABLE — agent edits __all__ only)
│   ├── schema.py                  # Piece 1: entities + enums + SyntheticDataset
│   ├── ml_spec.py                 # Piece 2: FEATURES + TARGET + latent_* function
│   ├── generator.py               # Piece 3: procedural + LLM + fallback generator
│   ├── tools.py                   # Piece 4: action tools + risk tiers + ALLOWLIST
│   ├── personas.py                # Piece 5: personas + data scopes + PERSONA_BY_ROLE
│   ├── prompts.py                 # Piece 6: system prompts + PLATFORM_FLOOR
│   ├── memory_spec.py             # Piece 7: FACT_TYPES + PROFILE_FIELDS + skills selector
│   ├── roster.py                  # Piece 8: agent_roster + sub_agent_roster
│   ├── corpus/                    # Piece 9: seed knowledge .md files
│   │   └── .gitkeep
│   └── skills/                    # Piece 10: procedural playbook .md files
│       └── .gitkeep
│
├── ml/                            # AUTO-ML ENGINE (agent NEVER edits these)
│   ├── __init__.py                # Public API: train(), evaluate(), register(), forecast()
│   ├── automl.py                  # FLAML AutoML orchestrator
│   ├── hpo.py                     # Optuna hyperparameter optimization
│   ├── forecast.py                # Time-series: Nixtla model selection + conformal
│   ├── evaluate.py                # Cross-validation, holdout, backtest scoring
│   ├── explain.py                 # SHAP + feature importance reports
│   ├── export.py                  # ONNX model export + ONNX Runtime inference
│   ├── monitor.py                 # Evidently data/drift monitoring
│   ├── card.py                    # Model card generation (ML model factsheet)
│   ├── registry.py                # Model versioning, metadata store, promotion
│   ├── train.py                   # Full train→evaluate→register pipeline
│   └── pipelines.py               # Prefect flows: train_flow, eval_flow, forecast_flow, drift_flow
│
├── config/                        # DECLARATIVE CONFIG — agent edits these
│   ├── automl.toml                # AutoML: search space, time budget, metric, ensemble config
│   ├── forecast.toml              # Forecast: horizon, seasonality, candidate models
│   ├── monitoring.toml            # Monitoring: drift thresholds, reference window, alert rules
│   └── pipeline.toml              # Pipeline: schedule, retry, artifact paths
│
├── tests/                         # TESTS — agent fills domain-specific test cases
│   ├── __init__.py
│   ├── test_schema.py             # Schema validation, enum coverage
│   ├── test_ml_spec.py            # Feature contracts, training_frame shape, latent function
│   ├── test_generator.py          # Synthetic data quality gate, label consistency
│   ├── test_tools.py              # Tool idempotency, reversibility, risk tiers
│   ├── test_domain_adapter.py     # STRUCTURAL: DomainAdapter Protocol conformance
│   └── test_ml_pipeline.py        # INTEGRATION: train → evaluate → explain full cycle
│
├── data/                          # OPTIONAL: real CSV data the agent may place here
│   └── .gitkeep
│
└── notebooks/                     # EXPLORATION: Jupyter notebook for hackathon demo
    └── explore.ipynb
```

## The AutoML Pipeline (Prefect Flows)

### Flow 1: `train_pipeline`
```
config/automl.toml
       │
       ▼
┌──────────────────┐
│  FLAML AutoML     │  ← finds best 3 model families (XGBoost, LightGBM, CatBoost, RF, etc.)
│  time_budget: 300s│
│  metric: r2/acc   │
└────────┬─────────┘
         │ top 3 candidates
         ▼
┌──────────────────┐
│  Optuna HPO       │  ← fine-tunes hyperparams for each candidate
│  n_trials: 100    │
│  pruner: Median   │
└────────┬─────────┘
         │ best model
         ▼
┌──────────────────┐
│  Evaluate         │  ← 5-fold CV + holdout + SHAP + conformal calibration
│  + Calibrate      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Register         │  ← model registry JSON, model card, ONNX export
│  + Card + Export  │
└──────────────────┘
```

### Flow 2: `forecast_pipeline`
```
config/forecast.toml
       │
       ▼
┌──────────────────┐
│  Nixtla Stats     │  ← AutoARIMA, AutoETS, SeasonalNaive
│  Forecast         │     Chronological train/test split (NO random split)
│  + Conformal      │     ConformalIntervals for calibrated bands
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Backtest         │  ← Rolling-origin backtest, empirical coverage measurement
│  + Model Select   │     Rank by empirical coverage, select best
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Register         │  ← forecast model registry + forecast card
└──────────────────┘
```

### Flow 3: `eval_pipeline` (scheduled)
```
model registry
       │
       ▼
┌──────────────────┐
│  Holdout Scoring  │  ← r2, RMSE, MAE (regression) / accuracy, F1 (classification)
│  + Metrics        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Evidently Drift  │  ← Data drift, target drift, prediction drift
│  Report           │     vs. reference window from training
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Promote / Alert  │  ← Within threshold → promote to production
│                    │     Breached threshold → alert, keep current model
└──────────────────┘
```

### Flow 4: `drift_pipeline` (scheduled)
```
reference data (from training)
       │
       ▼
┌──────────────────┐
│  Evidently        │  ← DataDriftTable, DataQualityPreset, RegressionPerformance
│  Monitor          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Threshold Check  │  ← config/monitoring.toml thresholds
│  + Alert          │     Pass → green, Fail → alert + report HTML
└──────────────────┘
```

## The 10 Domain Pieces — What the Agent Fills In

These are template files in `domain/` that follow the EXACT Protocol contracts
defined by `aegis.adapter.DomainAdapter` (in `aegis/src/aegis/adapter.py`).
The agent replaces placeholder content with domain-specific content.

### Piece 1: `domain/schema.py` — Entity Models + Enums

**Protocol contract:** `SchemaModule` — must export `SCHEMA_VERSION: str`

**Agent writes:**
- Pydantic v2 `BaseModel` entity classes (the domain's "records")
- `StrEnum` categorical vocabularies
- `RESOLVED_STATUSES` (which statuses count as "done" for ML labeling)
- `SCHEMA_VERSION` (bump to "1.0.0" for new domain)
- `SyntheticDataset` container class (MUST keep `customers`, `agents`, `requests`, `documents`, `metadata`, `labelled_requests()` — the ML spine reads these by name)

**Template provides:**
- Import skeleton (pydantic, datetime, enum)
- Docstring structure with "piece 1 of 10" convention
- `SyntheticDataset` skeleton with `customer_by_id`, `agent_by_id`, `labelled_requests` methods
- `DatasetMetadata` model

### Piece 2: `domain/ml_spec.py` — ML Features + Target

**Protocol contract:** `MLSpecModule` — must export `FEATURES`, `FEATURE_NAMES`, `TARGET`, `training_frame(num_records, seed)`, `describe_prediction(resp, top_k)`

**Agent writes:**
- `FEATURES: list[FeatureSpec]` — ordered feature contract (name, dtype categorical/numeric/boolean, description, levels for categoricals)
- `TARGET: TargetSpec` — name, task (regression/classification), unit, description
- `latent_<domain_signal_name>(features)` — the deterministic ground-truth function the generator samples labels around. MUST be monotone in every driver so the tree model can learn it and conformal intervals are meaningful.
- `features_for_request(record, ...)` — record → flat feature dict
- `feature_matrix(dataset)` — (X dicts, y list) from a SyntheticDataset
- `training_frame(num_records, seed)` — return pd.DataFrame with FEATURE_NAMES + TARGET.name columns
- `describe_prediction(resp, top_k)` — domain-framed decision-support text for agent prompts

**Template provides:**
- `FeatureSpec` and `TargetSpec` Pydantic models
- `FeatureDType = Literal["categorical", "numeric", "boolean"]`
- `_INTERCEPT` + dict-pattern for monotone latent function
- Full `training_frame` pipeline: calls generator sync, builds DataFrame
- `describe_prediction` skeleton using TARGET.name/TARGET.unit

### Piece 3: `domain/generator.py` — Synthetic Data Generator

**Protocol contract:** `GeneratorModule` — must export `DOMAIN_SERIES_LABEL`, `DOMAIN_SERIES_UNIT`, `domain_series_events(num_records, seed)`, `generate_synthetic(config)`, `generate_synthetic_sync(config)`

**Agent writes:**
- `GeneratorConfig` — num_entities, resolved_fraction, noise_scale, use_llm, seed
- Procedural draw functions for new entity types
- LLM-fabrication prompt text (requested by `ModelRole`, not hardcoded model id)
- `DOMAIN_SERIES_LABEL` — client-facing chart title (a SENTENCE a jury reads)
- `DOMAIN_SERIES_UNIT` — y-axis unit for demand series
- `domain_series_events(num_records, seed)` — (timestamp, value) arrival events

**Template provides:**
- `_LLMResultLike` and `CompleteFn` Protocols (dependency injection)
- `_EPOCH` constant for deterministic timestamps
- `_assemble()` shared sync/async core
- `_resolve_complete()` — lazily imports `app.core.llm.complete`
- `_template_*` fallback functions for offline determinism
- `DatasetQualityReport` model (quality gate)
- Full public entry points: `generate_synthetic`, `generate_synthetic_sync`

### Piece 4: `domain/tools.py` — Action Tools + Risk Registry

**Protocol contract:** `ToolsModule` — must export `TOOL_REGISTRY`, `ALLOWLIST`, `is_allowed(persona_id, tool_name)`, `tools_for(persona_id)`, `tool_definitions_for(persona_id)`, `run_tool(persona_id, tool_name, args, ctx)`

**Agent writes:**
- Argument models (Pydantic, one per action tool)
- Tool handler functions (async, typed, idempotent, reversible, audited)
- `TOOL_REGISTRY: dict[str, ToolSpec]` — name → spec with `risk: RiskLevel` (LOW/MEDIUM/HIGH)
- `ALLOWLIST: dict[str, frozenset[str]]` — persona → allowed tool names
- Risk tier assignments (the ONLY signal that drives the human gate)

**Template provides:**
- `RecordStore` Protocol, `AuditFn` Protocol
- `InMemoryRecordStore` class
- `ToolContext` dataclass (injected store, actor, audit sink)
- `ToolHandler` Protocol, `ToolSpec` dataclass (frozen, with `definition()` method)
- `_emit_audit()` helper
- `run_tool()` authorization gate (checks allowlist BEFORE side effect)
- `is_allowed()`, `tools_for()`, `tool_definitions_for()` helpers
- `_ID_RULE` constant for tool descriptions
- `UnknownToolError`, `ToolNotAllowedError` exceptions

### Piece 5: `domain/personas.py` — Who Is Served

**Protocol contract:** `PersonasModule` — must export `PERSONAS`, `DEFAULT_PERSONA_ID`, `PERSONA_BY_ROLE`, `get_persona(persona_id)`, `persona_for_role(role)`

**Agent writes:**
- Persona definitions (id, role, display_name, description, data_scope, prompt_key)
- `PERSONA_BY_ROLE` mapping (EVERY RBAC role → a persona id — missing one = KeyError on login)
- Data scopes (ALL vs OWN with subject_field)

**Template provides:**
- `ScopeKind` enum (ALL, OWN)
- `DataScope` Pydantic model
- `Persona` Pydantic model with `tool_names` property (reads from ALLOWLIST)
- `PERSONAS` dict assembly pattern
- `DEFAULT_PERSONA_ID` fallback
- `PERSONA_BY_ROLE` template (Role.ADMIN → ..., Role.AI_TEAM → ..., etc.)
- `persona_for_role(role)` with loud KeyError on missing mapping
- `get_persona(persona_id)` with default fallback

### Piece 6: `domain/prompts.py` — System Prompts

**Protocol contract:** `PromptsModule` — must export `SYSTEM_PROMPTS`, `PLATFORM_FLOOR`, `render_system_prompt(persona, extra_context)`, `render_platform_floor(persona)`

**Agent writes:**
- `SYSTEM_PROMPTS: dict[str, str]` — persona prompt_key → base system prompt (domain-specific)
- Each prompt names the domain, the tools, the rules

**Template provides:**
- `_scope_clause(persona)` — derives data scope text from persona
- `_tools_clause(persona)` — lists allowed tools with descriptions and risk tiers
- `PLATFORM_FLOOR` — the non-negotiable platform preamble (DO NOT CHANGE — this is Aegis's floor, not the domain's)
- `render_platform_floor(persona)` — composes floor + scope + tools
- `render_system_prompt(persona, extra_context)` — composes base + floor + extra

### Piece 7: `domain/memory_spec.py` — What Is Worth Remembering

**Protocol contract:** `MemorySpecModule` — must export `FACT_TYPES`, `PROFILE_FIELDS`, `FACT_EXTRACTION_PROMPT`, `IMPORTANCE_HINTS`, `SKILLS_DIR`, `FactSchema`, `FactExtraction`, `memory_subject_for(user_id, persona_id)`, `render_profile(profile)`, `select_skills(query, persona_id, available)`.

Note: `memory_spec` is NOT re-exported through `adapter/__init__.py`'s `__all__` — it is imported directly as the module object by `backend/src/app/memory/__init__.py:33`. So its PATH and SYMBOL NAMES must stay stable.

**Agent writes:**
- `FACT_TYPES` — what kinds of facts this domain distils
- `PROFILE_FIELDS` — structured profile fields (the "human block")
- `PROFILE_ALIASES` (optional) — predicate spellings → PROFILE_FIELDS mapping
- `FACT_EXTRACTION_PROMPT` — the cheap-model extractor's system prompt
- `IMPORTANCE_HINTS` — domain guidance for 1-10 importance (poignancy) rating
- `SKILLS_DIR` — path to `skills/` directory
- `select_skills` hints dict — keyword → playbook filename (CRITICAL: adding a playbook without updating this means it's never selected, and nothing warns you)

**Template provides:**
- `FactSchema` Pydantic model
- `FactExtraction` Pydantic container
- `memory_subject_for()` — scopes memory to user/entity (the app-level isolation key)
- `render_profile()` — formats stored profile as prompt text
- `select_skills()` — deterministic keyword match on query

### Piece 8: `domain/roster.py` — Agent Specialists + Fan-Out Team

**Protocol contract:** `RosterModule` — must export `agent_roster()`, `sub_agent_roster()`

**Agent writes:**
- Specialist definitions: `role` (must be "qa" or "memory" — the graph has nodes for these two only), `description`, `keywords` (lowercase phrase hints), `is_default` (exactly one)
- Sub-agent team: `agent_id`, `role`, `label`, `system_prompt`, `tool_allowlist` (MUST be names from TOOL_REGISTRY — stale names are silently dropped)

**Template provides:**
- `RosterSpecialist` frozen dataclass
- `AgentRoster` dataclass with `default_role`, `roles()`, `named()` properties
- `SubAgentSpec` import from `aegis.agent`
- `agent_roster()` function pattern (returns frozen constant)
- `sub_agent_roster()` function pattern

### Piece 9: `domain/corpus/*.md` — Seed Knowledge

**Protocol contract:** `CorpusModule` — must export `load_seed_corpus()`

**Agent writes:**
- Seed Markdown documents with YAML frontmatter: `id`, `kind`, `category`, `tags`, `title`
- Domain knowledge the retrieval system uses before real documents are ingested

**Template provides:**
- `__init__.py` with `load_seed_corpus()` that parses `*.md` files
- `_parse_frontmatter()`, `_parse_tags()`, `_to_document()` helpers
- Frontmatter format documentation

### Piece 10: `domain/skills/*.md` — Procedural Playbooks

**Protocol contract:** No member of its own — discovered from `memory_spec.SKILLS_DIR`

**Agent writes:**
- How-to-act playbooks in Markdown
- Selected by `memory_spec.select_skills()` via keyword match

**Template provides:**
- `.gitkeep` placeholder
- Documentation in `memory_spec.py` about the hints dict trap

## The AutoML Engine (`ml/`)

### `ml/automl.py` — FLAML AutoML Orchestrator
- Reads `config/automl.toml`
- Runs FLAML's `AutoML.fit()` with the domain's training frame
- Returns top 3 model candidates with scores
- Supports regression and classification tasks
- Time budget configurable per-run

### `ml/hpo.py` — Optuna Hyperparameter Optimization
- Define-by-run search space (XGBoost, LightGBM, CatBoost, RF params)
- MedianPruner for early stopping
- Integration with FLAML's best candidates
- Study persistence for resumability

### `ml/forecast.py` — Time-Series Forecasting
- Wraps Aegis's existing `aegis.forecast` (Nixtla StatsForecast)
- AutoARIMA, AutoETS, SeasonalNaive candidates
- Chronological split (NO random split — would leak future into training)
- ConformalIntervals for calibrated prediction bands
- Rolling-origin backtest with empirical coverage measurement
- Best model selection by empirical coverage

### `ml/evaluate.py` — Evaluation & Scoring
- Regression metrics: r2, RMSE, MAE, MAPE
- Classification metrics: accuracy, precision, recall, F1, ROC-AUC
- 5-fold cross-validation
- Holdout evaluation
- Conformal coverage measurement
- Feature importance from SHAP values

### `ml/explain.py` — SHAP Explainability
- Uses Aegis's existing `aegis.ml` SHAP TreeExplainer (per ensemble member, then averaged)
- Global feature importance (mean |SHAP|)
- Per-prediction explanation
- Waterfall, beeswarm, dependence plot generation
- Explanation report HTML generation

### `ml/export.py` — ONNX Export
- Convert trained sklearn/XGBoost pipeline to ONNX
- `onnxruntime` inference for serving without Python ML deps
- Input/output signature preservation
- Validation round-trip (predict same for ONNX and sklearn)

### `ml/monitor.py` — Evidently Drift Detection
- Data drift: `DataDriftTable` on feature distributions
- Target drift: comparison of target distributions
- Prediction drift: comparison of prediction distributions
- Regression performance: `RegressionPerformanceReport`
- Classification performance: `ClassificationPerformanceReport`
- Configurable reference window from training data
- Alert thresholds from `config/monitoring.toml`

### `ml/card.py` — Model Card Generation
- Extends Aegis's `aegis.ml.types.ModelCard`
- Adds: AutoML search space summary, FLAML leaderboard, Optuna best trial
- Training data provenance (synthetic vs. real)
- Fairness metrics (if demographic features present)
- Limitations and intended use sections
- Markdown + HTML output

### `ml/registry.py` — Model Registry
- Filesystem-backed: `~/.aegis_ml/models/`
- Versioned: `model-v1.joblib`, `model-v2.joblib`, etc.
- Metadata JSON per version: timestamp, metrics, data source, model card ref
- Promotion: `staging` → `production` → `archived`
- Rollback: keep N previous versions
- Query: `registry.latest("production")`, `registry.list("staging")`

### `ml/train.py` — Full Training Pipeline
- Single entry point: `python -m aegis_ml.ml.train`
- Orchestrates: FLAML → Optuna → Evaluate → Register → Card → Export
- Returns: trained model path, metrics dict, model card path

### `ml/pipelines.py` — Prefect Flows
- `train_pipeline`: the full AutoML training flow
- `forecast_pipeline`: time-series model selection flow
- `eval_pipeline`: scheduled evaluation + drift check
- `drift_pipeline`: continuous monitoring
- All flows use Prefect 3.x ephemeral server (SQLite, no infra)

## Configuration (`config/`)

### `config/automl.toml`
```toml
[automl]
task = "regression"            # "regression" or "classification"
time_budget = 300              # seconds
metric = "r2"                  # "r2", "rmse", "accuracy", "roc_auc", etc.
eval_method = "cv"             # "cv" or "holdout"
n_splits = 5

[estimators]
include = ["xgboost", "lgbm", "rf", "catboost", "extra_tree"]
exclude = []

[hpo]
n_trials = 100
timeout = 600
pruner = "median"              # "median", "hyperband", "none"

[ensemble]
method = "voting"              # "voting", "stacking", "none"
voting = "soft"
```

### `config/forecast.toml`
```toml
[forecast]
horizon = 30                   # periods ahead
season_length = 7              # for seasonal models
frequency = "D"                # "D" daily, "H" hourly, "W" weekly, "M" monthly

[models]
candidates = ["AutoARIMA", "AutoETS", "SeasonalNaive"]

[conformal]
confidence_level = 0.9
calibration_ratio = 0.2

[backtest]
n_windows = 5
step_size = 7
```

### `config/monitoring.toml`
```toml
[drift]
method = "evidently"
reference_window_days = 30

[thresholds]
data_drift_score = 0.2
target_drift_score = 0.1
prediction_drift_score = 0.1
performance_drop_r2 = 0.05        # r2 drop threshold
performance_drop_accuracy = 0.03  # accuracy drop threshold

[alerts]
on_drift = "warn"                  # "warn", "block", "none"
on_performance_drop = "block"      # "warn", "block", "none"
report_format = "html"             # "html", "json", "both"
```

### `config/pipeline.toml`
```toml
[prefect]
ephemeral = true                   # Use ephemeral server (no infra)
log_level = "INFO"

[schedule]
train = null                       # manual trigger only
evaluate = "0 6 * * 1"            # weekly Monday 6am
drift = "0 */6 * * *"             # every 6 hours

[retry]
max_retries = 3
retry_delay_seconds = 60

[artifacts]
registry_path = "~/.aegis_ml/models"
report_path = "~/.aegis_ml/reports"
card_path = "~/.aegis_ml/cards"
```

## Dependencies (to add to `backend/pyproject.toml`)

```toml
# AutoML + HPO
"flaml[automl]>=2.0",       # Microsoft FLAML — fast AutoML, pure Python, Windows-native
"optuna>=4.0",               # Hyperparameter optimization — pure Python, define-by-run
"tpot>=0.12",                # Genetic pipeline optimization — pure Python, Windows wheel

# Model export
"onnx>=1.17",                # Open Neural Network Exchange — Windows wheel
"onnxruntime>=1.20",         # ONNX Runtime inference — Microsoft, Windows wheel

# Monitoring
"evidently>=0.4",            # ML monitoring + drift detection — pure Python, OS-independent

# Pipeline orchestration
"prefect>=3.0",              # Workflow orchestration — Windows-compatible (ephemeral server)

# Additional ML libraries (FLAML may pull these, but explicit is safe)
"lightgbm>=4.0",             # Gradient boosting — Windows wheel, Microsoft
"catboost>=1.2",             # Gradient boosting — Windows wheel
```

**All are Windows `pip install` compatible:**
- FLAML: pure Python (`py3-none-any` wheel)
- Optuna: pure Python (`py3-none-any` wheel)
- TPOT: pure Python (`py3-none-any` wheel)
- ONNX: prebuilt Windows wheel (`win_amd64`)
- ONNX Runtime: prebuilt Windows wheel (`win_amd64`) — Microsoft-maintained
- Evidently: pure Python (`py3-none-any` wheel)
- Prefect 3.x: works on Windows natively (all critical bugs fixed in 3.0.3+)
- LightGBM: prebuilt Windows wheel (`win_amd64`) — Microsoft
- CatBoost: prebuilt Windows wheel (`win_amd64`)

**NOT included:**
- auto-sklearn: Linux-only (requires Unix `resource` module, SWIG, GCC)
- MLflow: requires a tracking server — adds infra dependency with no hackathon-day benefit
- Metaflow: requires AWS or local services — overkill for synthetic data training

## Hackathon Day Workflow

### Step 1: Receive problem statement
Agent reads `AGENT_CONTEXT.md` to understand Aegis architecture and the DomainAdapter Protocol.

### Step 2: Fill in domain templates
Agent edits the 10 `domain/` files. Each file's docstring says exactly what to write.

### Step 3: Set config
Agent edits `config/*.toml` with domain-specific model parameters.

### Step 4: Write tests
Agent writes domain-specific test cases in `tests/`.

### Step 5: Copy domain/ into Aegis
```bash
cp -r domain/* ../backend/src/app/adapter/
```

### Step 6: Conformance check
```bash
(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)
```
Must pass all 14 checks. Each check descends from a real wiring defect this repo shipped.

### Step 7: Train ML spine
```bash
# Aegis core trainer (XGBoost + MAPIE + SHAP)
(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml)

# AutoML pipeline (FLAML + Optuna + Prefect)
(cd backend && PYTHONPATH=src:.venv/bin/python -m aegis_ml.ml.train)
```

### Step 8: Run full test suite
```bash
(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest -q)
(cd aegis && PYTHONPATH=src ../backend/.venv/bin/python -m pytest -q)
(cd web && npm run build && npm test)
backend/.venv/bin/python -m ruff check aegis backend
```

### Step 9: Demo
- Open `notebooks/explore.ipynb` for interactive exploration
- Check model registry for trained artifacts
- View SHAP explanations and model card
- Check Prefect dashboard for pipeline status

## What We Are NOT Building

- NOT a new Aegis core module — everything uses existing `aegis.ml`, `aegis.forecast`, `aegis.evals`, `aegis.adapter` Protocols
- NOT a replacement for `backend/src/app/adapter/` — this is a template FACTORY that PRODUCES the adapter contents
- NOT auto-sklearn — does not work on Windows
- NOT MLflow — adds server dependency with no hackathon-day benefit
- NOT Docker or WSL dependencies — everything is pip-installable native Windows
- NOT GPU-requiring — all models run on CPU (XGBoost `tree_method="hist"`, no CUDA)