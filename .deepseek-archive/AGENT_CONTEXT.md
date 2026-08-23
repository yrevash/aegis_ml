# AGENT_CONTEXT.md — Aegis Platform Context for the Hackathon Agent

> **Your task:** Receive a problem statement on hackathon day. Fill in the 10 domain
> template files under `domain/`. Configure `config/*.toml`. Write tests under `tests/`.
> Copy `domain/` into `backend/src/app/adapter/`. Run conformance. Train the ML spine.
> Run the full suite. Demo.
>
> **Your ML capabilities:** Regression, classification, and time-series forecasting with
> AutoML (FLAML + Optuna), hyperparameter optimization (Optuna), conformal prediction
> (MAPIE), SHAP explainability, ONNX export, model registry with versioning and promotion,
> drift monitoring (Evidently), ML model cards, and Prefect pipeline orchestration.
>
> **Your environment:** Windows native (no WSL, no Docker). Python 3.11. Every dependency
> is pip-installable with prebuilt Windows wheels (`win_amd64`) or pure Python
> (`py3-none-any`). No GPU — all models run on CPU.

---

## 1. What Is Aegis?

Aegis is a **domain-agnostic, multi-tenant enterprise agentic-AI platform**. It is
NOT a framework you fork — it is a library you import (`pip install aegis[extra]`).
The core (`aegis/`) knows nothing about any specific domain. The domain lives
entirely in ONE directory: `backend/src/app/adapter/`. Your job is to produce the
contents of that directory.

The platform's thesis: **every autonomous action is uncertainty-bounded,
explainable, guarded, human-approved and fully traced.**

Three layers:

| Layer | What | Where | Size |
|---|---|---|---|
| **Core library** | 29 importable packages, 50 Stable public names, 2268 tests | `aegis/src/aegis/` | ~30 subpackages |
| **Backend** | FastAPI composition root, 121 endpoints, 1174 tests | `backend/src/app/` | ~27 modules |
| **Console** | Next.js 15 frontend, 4 role-scoped portals, 158 tests | `web/src/` | React 19 + TypeScript |
| **Domain adapter** | The 10 pieces you will write | `backend/src/app/adapter/` | 8 modules + 2 dirs |

## 2. The Architecture You're Plugging Into

### The Agent Loop (LangGraph plan-and-execute)

```
guard_input → route → recall_memory → retrieve → plan → gate → act → reflect → generate → guard_output → persist_memory
```

Every step is bracketed by the AG-UI event stream (SSE) — what the console renders
live is simultaneously what OpenTelemetry exports as a trace.

### The 12 Platform Modules (Customer-Facing Capabilities)

| Module | Tech Underneath | What It Does |
|---|---|---|
| **Aegis Gateway** | LiteLLM | Single model chokepoint — role-based routing, per-tenant budgets enforced BEFORE spend, timeout, retry, append-only usage ledger |
| **Aegis Router** | LangGraph | Multi-agent supervisor — deterministic keyword classifier + cheap-LLM tiebreak for routing a turn to the right specialist |
| **Aegis Memory** | Postgres + Qdrant | Episodic, semantic, procedural memory — bitemporal (world-time + record-time), consolidated nightly |
| **Aegis Cache** | Redis (Memurai on Windows) | Semantic response cache (RediSearch vector index via redisvl) |
| **Aegis Retrieval** | Neo4j/LightRAG + Qdrant | Hybrid RAG: vector + graph + BM25 → Reciprocal Rank Fusion → local ONNX cross-encoder rerank |
| **Aegis Signal** | XGBoost + MAPIE + SHAP | Soft-voting ensemble (XGBoost + HistGradientBoosting) + calibrated split-conformal intervals + exact SHAP TreeExplainer attributions |
| **Aegis Guardrails** | NeMo Colang + Presidio | 6-layer input/output rail stack: injection classification, PII detection (Presidio + spaCy, regex fallback), schema validation, topical scope, content safety, output grounding |
| **Aegis Evals** | RAGAS-style proxies + LLM judge | Deterministic retrieval metrics (no LLM call), LLM-judge answer evaluation, CI regression gate |
| **Aegis Loop** | Native | LLM-Ops self-improvement: trace → eval → diagnose → tiered prompt-version release |
| **Aegis Governance** | Postgres RLS + JWT | Multi-tenant RBAC (4 roles + 2 privilege tiers), per-tenant budgets, row-level security, append-only audit log (enforced by Postgres privileges, not convention) |
| **Aegis Trace** | OpenTelemetry → Arize Phoenix | End-to-end glass-box tracing with OpenInference GenAI semantic conventions |
| **Aegis Tools / MCP** | Native + MCP SDK | Risk-tiered tool registry (LOW/MEDIUM/HIGH) + human approval gate, exposed over MCP Streamable HTTP |

### Data Stores

| Store | Role | Windows Note |
|---|---|---|
| **Postgres** | Relational data, RBAC, budgets, append-only audit, LangGraph checkpoints. `pgvector` REMOVED — vector search is entirely Qdrant. | Native Windows install via `winget` |
| **Qdrant** | THE ONE vector store. Both `aegis.retrieval` and LightRAG's `QdrantVectorDBStorage` write to a single Qdrant node. | `qdrant-x86_64-pc-windows-msvc.zip` — Apache-2.0 zip with a binary, no Docker, no installer |
| **Neo4j** | Knowledge graph (entities + relationships extracted at ingestion) | Neo4j Desktop via `winget` |
| **Redis** | Semantic cache, rate limiter slot leases, notification pub/sub | **Memurai** — Windows-native Redis-compatible, same wire protocol, same port 6379 |
| **Temporal** | Durable ingestion workflows (optional for ML) | `temporal server start-dev` — ephemeral, no persistent server needed |

### The Human Gate (Critical Design Feature)

Tools carry a `risk: RiskLevel` (LOW / MEDIUM / HIGH). A tool at or above
`AgentConfig.gate_min_risk` (platform default HIGH) pauses for human approval via
LangGraph's `interrupt()` checkpointed to Postgres. Tool risk is the **ONLY**
gating signal — not model confidence, not the ML prediction. Mark a consequential
write HIGH and the approval gate appears with no engine change at all.

A parked run checkpoints durably and resumes on any worker from a persisted
approvals-inbox row. Kill the process, approve on the new one, and the run finishes
from the interrupted checkpoint without re-running a single pre-gate node.

## 3. The DomainAdapter Protocol — Your Contract

The core reaches the domain EXCLUSIVELY through `aegis.adapter.DomainAdapter` —
a `runtime_checkable` Protocol with 11 members (9 pieces + DOMAIN_ID + DOMAIN_DESCRIPTION).

**Every member must exist.** `isinstance(adapter, DomainAdapter)` checks presence.
A type checker checks shape. Both are needed. `missing_members(adapter)` returns
what's absent.

```
DomainAdapter
├── DOMAIN_ID: str                    # Stable machine id of the loaded domain
├── DOMAIN_DESCRIPTION: str           # One paragraph (ALSO = guardrails allowed_topics)
├── schema: SchemaModule              # Piece 1 — record types + SCHEMA_VERSION
├── ml_spec: MLSpecModule             # Piece 2 — features, target, training_frame
├── generator: GeneratorModule        # Piece 3 — synthetic data + demand series
├── tools: ToolsModule                # Piece 4 — action tools + risk registry + ALLOWLIST
├── personas: PersonasModule          # Piece 5 — who is served + data scope + role mapping
├── prompts: PromptsModule            # Piece 6 — system prompts + platform floor (half no tenant may edit)
├── memory_spec: MemorySpecModule     # Piece 7 — durable facts + structured profile + skills directory
├── roster: RosterModule              # Piece 8 — specialists the supervisor routes to + fan-out team
└── corpus: CorpusModule              # Piece 9 — seed corpus loader
```

**Piece 10 (skills/) has NO member** — it is a directory discovered from
`memory_spec.SKILLS_DIR`. Giving it a separate top-level spelling would create
exactly the ambiguity ("is it 5 or 6?") this Protocol exists to end.

### Key Protocol Members — What Each Piece MUST Export

#### SchemaModule (Piece 1)
- `SCHEMA_VERSION: str`

#### MLSpecModule (Piece 2)
- `FEATURES: list[FeatureSpec]` — each spec has `.name`, `.dtype` (categorical/numeric/boolean), `.description`, `.levels` (optional, for categoricals)
- `FEATURE_NAMES: list[str]` — `[f.name for f in FEATURES]`
- `TARGET: TargetSpec` — `.name`, `.task` ("regression" or "classification"), `.unit`, `.description`
- `training_frame(self, *, num_records: int = ..., seed: int = ...) -> pd.DataFrame` — note: `num_records`, NOT `num_requests`. A core Protocol may not force every domain to call its rows "requests".
- `describe_prediction(self, resp, *, top_k: int = 3) -> str` — domain-framed decision-support text injected into the agent's reasoning

#### GeneratorModule (Piece 3)
- `DOMAIN_SERIES_LABEL: str` — chart title, a sentence a client reads
- `DOMAIN_SERIES_UNIT: str` — y-axis unit
- `domain_series_events(self, *, num_records, seed) -> Sequence[tuple[timestamp, float]]` — (timestamp, value) arrival events
- `generate_synthetic_sync(self, config = None) -> SyntheticDataset` — no LLM, no `await`
- `generate_synthetic(self, config = None) -> Awaitable[SyntheticDataset]` — optionally with LLM

#### ToolsModule (Piece 4)
- `TOOL_REGISTRY: Mapping[str, ToolSpecLike]` — each spec has `.name: str`, `.description: str`, `.risk: RiskLevel`, `.definition() -> dict`
- `ALLOWLIST: Mapping[str, frozenset[str]]` — persona id → allowed tool names
- `is_allowed(self, persona_id: str, tool_name: str) -> bool`
- `tools_for(self, persona_id: str) -> list[ToolSpecLike]`
- `tool_definitions_for(self, persona_id: str) -> list[dict]` — LLM `tools=` payload
- `run_tool(self, persona_id, tool_name, args, ctx) -> Awaitable[ToolOutcome]` — authorizes BEFORE side effect

#### PersonasModule (Piece 5)
- `PERSONAS: Mapping[str, Persona]` — persona id → persona (each has `.id`, `.role`, `.data_scope`, `.prompt_key`, `.tool_names`)
- `DEFAULT_PERSONA_ID: str`
- `PERSONA_BY_ROLE: Mapping[Role, str]` — EVERY Role value MUST map to a persona id that EXISTS in PERSONAS. Missing entry = `KeyError` on login. This table used to live in the core's login path and was the most expensive retarget defect.
- `get_persona(self, persona_id: str | None) -> Persona`
- `persona_for_role(self, role: Role | str) -> str`

#### PromptsModule (Piece 6)
- `SYSTEM_PROMPTS: dict[str, str]` — persona prompt_key → base system prompt
- `PLATFORM_FLOOR: str` — DO NOT CHANGE THIS. This is Aegis's non-negotiable platform preamble, composed UNDER every prompt version. A tenant can edit the task half; the platform floor is the half they can never remove. It is NOT a row in prompt_versions — it is composed at render time.
- `render_system_prompt(self, persona, *, extra_context = None) -> str`
- `render_platform_floor(self, persona) -> str`

#### MemorySpecModule (Piece 7)
- `FACT_TYPES: list[str]` — kinds of durable facts this domain distils
- `PROFILE_FIELDS: list[str]` — structured profile fields (the always-injected "human block")
- `FACT_EXTRACTION_PROMPT: str` — cheap-model extractor's system prompt
- `IMPORTANCE_HINTS: str` — domain guidance for 1-10 importance/poignancy rating
- `SKILLS_DIR: str` — path to skills/ directory
- `FactSchema: type[BaseModel]` — Pydantic model for one extracted fact
- `FactExtraction: type[BaseModel]` — Pydantic container with `facts` list
- `memory_subject_for(self, user_id, persona_id = None) -> str | None` — the app-level memory isolation key. Returns `None` = no memory for this run. This function is load-bearing: every memory consumer goes through it rather than composing the key itself.
- `render_profile(self, profile: dict) -> str` — formats stored profile as prompt "human block"
- `select_skills(self, query, persona_id, available) -> list[str] | None` — deterministic keyword match

**Note:** `memory_spec` is NOT listed in `adapter/__init__.py`'s `__all__`. It is imported directly as the module OBJECT by `backend/src/app/memory/__init__.py:33` via `set_default_spec(app.adapter.memory_spec)`. The consumer binds to the MODULE OBJECT, not its individual names, so the module path and every member name must stay stable.

#### RosterModule (Piece 8)
- `agent_roster(self) -> AgentRosterLike` — the specialist set. Each specialist has `.role: str`, `.description: str`, `.keywords: tuple[str,...]`, `.is_default: bool`. The roster itself has `.default_role: str`, `.roles() -> list[str]`, `.named() -> list[RosterSpecialist]`.
- `sub_agent_roster(self) -> Sequence[SubAgentSpec]` — the fan-out team. Each spec has `agent_id`, `role`, `label`, `system_prompt`, `tool_allowlist: frozenset[str]` (optional).

#### CorpusModule (Piece 9)
- `load_seed_corpus(self) -> list[Document]` — possibly empty (that's honest). A missing loader = seed knowledge silently failed to load.

## 4. The Existing ML Infrastructure You're Extending

### `aegis.ml` — Trustworthy-ML Spine (Aegis Core)
Located at `aegis/src/aegis/ml/`. ALREADY BUILT. You use it, you don't change it.

**What it does (from `aegis/src/aegis/ml/model.py`, 966 lines):**
- **Soft-voting ensemble:** XGBoost (`tree_method="hist"`, CPU-only) + sklearn HistGradientBoosting — two complementary boosting implementations averaged to reduce variance. Swap/add members by editing `_regression_members()` / `_classification_members()`.
- **MAPIE split conformal prediction:** Wraps the already-fitted ensemble. Uses a held-out calibration split (separate from training data) to produce prediction intervals (regression) or prediction sets (classification) with guaranteed marginal coverage equal to `confidence_level`. The calibrated interval is the honest uncertainty the agent surfaces as supporting evidence.
- **SHAP TreeExplainer (per member):** Exact per-feature attributions computed for each tree member and averaged with ensemble member weights. Explains the ensemble's actual output.
- **Model card (`ModelCard`):** Honest, measured metadata: task, target, features, ensemble members + weights, conformal method, requested coverage, calibration/training/test split sizes, data source, dataset digest (SHA-256 fingerprint for tamper evidence), empirical coverage on holdout, held-out metric.
- **One-hot encoding:** sklearn `ColumnTransformer` with `OneHotEncoder` on categorical features, `handle_unknown="ignore"`.
- **Joblib persistence:** `train(spec, frame, path)` writes to disk, `load(path)` reads back, `get_model()` auto-loads on first use. **No silent fallback:** if no model is trained or persisted, `MLModelUnavailableError` is raised rather than training on noise and serving it as evidence.

**Key types (from `aegis/src/aegis/ml/types.py`):**
- `ResolvedSpec` — features, target, task, categorical_features, frame_provider
- `TrustworthyModel` — the fitted ensemble + conformal predictor + preprocessor
- `MLExplainResponse` — prediction, conformal_interval (tuple or None), conformal_confidence, interval_width, prediction_set_size, shap_attribution (list[ShapFeature]), data_source ("provided"/"spec_provider"/"synthetic"), imputed_features, unknown_features
- `ModelCard` — task, target, features, n_features, categorical_features, numeric_features, encoded_feature_count, ensemble_members, conformal_method, conformal_predictor, conformal_coverage (REQUESTED), calibration_size, training_size, test_size, conformal_coverage_empirical (MEASURED), metric_name, metric_value, data_source, dataset_digest
- `ShapFeature` — feature name, value, value_label (for categorical levels), contribution (signed SHAP)
- `EnsembleMember` — name, kind (estimator class), weight

**Key functions:**
- `train(spec, frame, confidence_level=0.9, calibration_size=0.25, test_size=0.2, random_state=0, path)` — train, calibrate, evaluate, persist
- `predict_explain(features: dict) -> MLExplainResponse` — one prediction with conformal + SHAP
- `load(path)` — load persisted artifact
- `get_model()` — load from default path or raise MLModelUnavailableError
- `resolve_spec(spec) -> ResolvedSpec` — leniently reads adapter's ml_spec (see section below)
- `frame_digest(frame, columns)` — SHA-256 content fingerprint

**Entry point:** `python -m app.ml` trains on the adapter's `ml_spec.training_frame(num_records, seed)`.

### `aegis.forecast` — Time-Series Forecasting (Aegis Core)
Located at `aegis/src/aegis/forecast/`. ALSO ALREADY BUILT.

**What it does (from `aegis/src/aegis/forecast/engine.py`, 537 lines):**
- **Nixtla StatsForecast:** AutoARIMA, AutoETS, SeasonalNaive candidates.
- **ConformalIntervals:** Calibrated prediction bands — the default, never the model's own parametric intervals (which are a model assumption, not a calibration).
- **Chronological split only:** NEVER random train_test_split. On a time series, random splitting is a leak — calibration rows drawn from after training rows make the residual distribution optimistic and void the coverage guarantee. Everything is split by time.
- **Rolling-origin backtest:** Empirical coverage measured by counting held-out actuals that landed inside the band. Routinely below requested level on real data — that gap is the finding.
- **Explicit failures:** `InsufficientHistoryError` (not enough periods), `ForecastFitError` (no model converged), `DegenerateSeriesError` (constant/variance-only series). No naive-line fallback.

### `aegis.evals` — Evaluation Metrics (Aegis Core)
Located at `aegis/src/aegis/evals/`. ALREADY BUILT.

**What it does:**
- RAGAS-style deterministic proxies (no LLM call, no `ragas` dependency): faithfulness, answer relevancy, context precision, context recall
- LLM judge harness for answer evaluation
- Regression testing for retrieval quality: `ablation.py` (drop arm → measure recall), `regression.py` (CI gate)
- Gold set management: `goldset.py`

### How `resolve_spec` Reads Your Adapter's ML Spec

From `aegis/src/aegis/ml/spec.py`, the function `resolve_spec` reads your `ml_spec.py` leniently using multiple fallback paths:

```python
# What it looks for:
features = spec.FEATURE_NAMES        # list[str] — ordered feature column names
    # Falls back to spec.features (lowercase) if FEATURE_NAMES is absent

target_obj = spec.TARGET             # object with .name and .task attributes
target = target_obj.name             # str — the target column name
    # Falls back to spec.target (lowercase)

task = target_obj.task               # "regression" or "classification"
    # Falls back to spec.task (lowercase), defaults to "regression"

categorical = spec.CATEGORICAL_FEATURES  # optional list[str]
    # Falls back to reading FEATURES list — any spec with dtype=="categorical"
    # has its .name added to categorical features

provider = spec.training_frame       # callable(*, num_records, seed) -> pd.DataFrame
    # Falls back to None
```

**CRITICAL:** If `resolve_spec` cannot find both FEATURE_NAMES and TARGET.name, it silently falls back to `FALLBACK_SPEC` — a generic 4-feature regression spec with synthetic noise. The model trains on noise and serves its predictions as domain evidence with no warning. The only indication is `distinct=False` at the end of `python -m app.ml`. This is the single most common silent failure in Aegis retargeting and the most expensive to debug.

## 5. The Conformance Suite — Your First Gate

```bash
(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)
```

14 checks in one file (`aegis/src/aegis/conformance/test_conformance.py`, 1036 lines).
No database, no model call, under a second. Every check descends from a real wiring
defect this repository actually shipped. Every failure prints the exact fix, the
consequence of leaving it, and the defect it came from.

### Critical checks for ML:

1. **Check: ml_spec wiring** — Your adapter's `ml_spec` must have the attributes `resolve_spec` reads. Missing = silently trains on noise.

2. **Check: generator coupling** — `generate_synthetic_sync()` must produce records whose labels come from `ml_spec`'s latent function. If the generator invents its own labels independently, the target column contains noise, the model fits noise, and the conformal interval is honestly enormous.

3. **Check: persona → role mapping** — Every RBAC role in `PERSONA_BY_ROLE` must map to a persona id that exists in `PERSONAS`. Missing entry = `KeyError` on login, but no test goes through the login path so nothing goes red.

4. **Check: roster → graph dispatch** — Every `role` in `agent_roster().specialists` must be "qa" or "memory" (the two specialist handler nodes in the graph). Unknown role = silently falls back to qa pipeline with a log warning.

5. **Check: sub-agent tool allowlist** — Every tool name in every `sub_agent_roster()` entry's `tool_allowlist` must be a key in TOOL_REGISTRY. Stale name = silently dropped, sub-agent runs with fewer tools than you think — possibly none.

6. **Check: skill reachability** — `memory_spec.select_skills()` must be able to return at least one playbook name for some input. A hints dict with no matching playbook = skills never selected, nothing warns.

7. **Check: core vocabulary quarantine** — Scans EVERY Python file outside `backend/src/app/adapter/` for the reference domain's vocabulary (from `aegis/src/aegis/conformance/_vocabulary.py`). Fails with file, line number, and term. This is the check that makes "only the adapter changes" a fact rather than a promise.

**The vocabulary file MUST be updated.** `_vocabulary.py` contains the shipped domain's words (persona ids, record types, feature keys, tool names, playbook names, the domain id). When you retarget the adapter, update it with YOUR domain's vocabulary in the same commit. The check fails when a listed word no longer appears in the adapter.

Anti-vacuity: the check also fails if the word list is empty, too few files are scanned, the scanner matches nothing, or a listed word the reference adapter no longer uses. A quiet pass is treated as a failure.

## 6. The Reference Domain (What You're Replacing)

The current reference adapter in `backend/src/app/adapter/` is a **service-request /
case-management** domain. It is illustrative only. Here's what it looks like so you
know what to replace:

### Entities
- `Customer` — id, name, email, region, tier, created_at
- `SupportAgent` — id, name, team, tenure_months, region, specialties
- `ServiceRequest` — id, title, description, category, priority, channel, region, status, customer_id, assigned_agent_id, created_at, updated_at, resolved_at, queue_depth_at_open, reopened_count, first_response_minutes, sla_hours, satisfaction_score, resolution_hours (ML TARGET), tags, notes (list[CaseNote])
- `CaseNote` — id (deterministic, for idempotency), author, body, created_at
- `Document` — id, kind, title, body, category, tags, source

### Enums
- `Priority`: low, medium, high, urgent
- `Category`: billing, technical, account, shipping, general
- `Channel`: email, chat, phone, portal
- `Region`: na, eu, apac, latam
- `CustomerTier`: standard, premium, enterprise
- `RequestStatus`: new, triaged, in_progress, waiting_customer, resolved, closed, reopened
- `DocumentKind`: kb_article, policy, faq, runbook

### ML Spec
- **Task:** Regression
- **Target:** `resolution_hours` (wall-clock hours from open to resolved)
- **Features (9):** priority (cat), category (cat), channel (cat), region (cat), customer_tier (cat), agent_tenure_months (num), queue_depth_at_open (num), reopened_count (num), description_length (num)
- **Latent function:** `latent_resolution_hours(features) -> float` — deterministic, monotone in every driver. Intercept 12h + category base (8-30h) + channel delay (1-8h) + region delay (2-7h) - priority speedup (0-12h) - tier speedup (0-6h) + 0.8×queue_depth + 6.0×reopened - 0.5×tenure + 0.01×description_length, floored at 0.5h. The generator calls this + Gaussian noise (σ=4h) to set each resolved request's label.

### Tools (TOOL_REGISTRY)
| Tool | Risk | Idempotent | Destructive | Read-only |
|---|---|---|---|---|
| `find_requests` | LOW | Yes | No | Yes |
| `add_case_note` | LOW | No | No | No |
| `assign_request` | MEDIUM | Yes | No | No |
| `update_request_status` | HIGH | Yes | Yes | No |

### Personas
| Persona | Role | Scope | Tools |
|---|---|---|---|
| `operations_lead` | ADMIN | ALL | all four |
| `client` | CLIENT | OWN (customer_id) | add_case_note only |

### Specialists (roster)
- `qa` (default) — full recall→retrieve→plan→gate→act→reflect→generate pipeline, no keywords
- `memory` — answers from long-term memory, keyword-triggered ("what do you know about me", "what do you remember", etc.)

### Sub-Agent Team (sub_agent_roster)
- `research` — external evidence (no tools)
- `knowledge` — internal corpus (no tools)
- `data` — record access + writes (tool_allowlist: update_request_status, add_case_note)
- `policy` — rules + recommendations (no tools)

### Conformance Vocabulary (from `_vocabulary.py`)
These words are quarantined — they may NOT appear in any core module outside the adapter:
`operations_lead`, `dataset.requests`, `Service requests opened per day`, `num_requests`,
`queue_depth_at_open`, `agent_tenure_months`, `reopened_count`, `customer_tier`,
`description_length`, `resolution_hours`, `ServiceRequest`, `SupportAgent`,
`update_request_status`, `assign_request`, `add_case_note`, `closing_requests`,
`de_escalation`, `service_request_management`

**When you replace the adapter, replace this list.** Edit
`aegis/src/aegis/conformance/_vocabulary.py` — swap in YOUR domain's persona ids,
record type names, feature keys, tool names, playbook names, and domain id.

## 7. The 10 Pieces — Edit Order and Traps

Edit in THIS ORDER. The order is not arbitrary — each piece consumes the vocabulary
the previous one defined.

### Piece 1: `domain/schema.py` — Entity Models + Enums

**You write:** Pydantic v2 BaseModel entity classes, StrEnum categorical vocabularies, RESOLVED_STATUSES (which statuses count as "done" for ML labeling), SCHEMA_VERSION (bump to "1.0.0").

**TRAP:** Keep the `SyntheticDataset` container. The ML spine, the generator, and the adapter registry ALL bind to it by attribute name. The container MUST have: `metadata: DatasetMetadata`, `customers`, `agents`, `requests`, `documents`, `labelled_requests()`, `customer_by_id(id)`, `agent_by_id(id)`. Rename the container and the whole vertical slice breaks at import. Change every field inside it freely — just keep the container name.

**TRAP:** After you edit schema.py, EVERY test in the repository fails at import. The conftest imports through `app.adapter`, and the old entity names no longer exist. A wall of ImportError — hundreds of lines — nothing to do with whether your edit was correct. It stays red until piece 8 (roster.py) is finished and the registry's re-exports all resolve. This is expected; do not chase it. Verify one file at a time with the per-step test commands.

### Piece 2: `domain/ml_spec.py` — ML Features + Target

**You write:** FEATURES list (use `FeatureSpec` Pydantic model — name, dtype: categorical/numeric/boolean, description, levels for categoricals), TARGET TargetSpec (name, task: regression or classification, unit, description), `latent_<signal_name>(features: dict) -> float` (deterministic, monotone in every driver), `features_for_<record>(record, agent, customer) -> dict`, `feature_matrix(dataset) -> (list[dict], list[float])`, `training_frame(num_records, seed) -> pd.DataFrame`, `describe_prediction(resp, top_k) -> str`.

**TRAP:** The `training_frame` keyword is `num_records`, NOT `num_requests`. The core Protocol (`aegis.adapter.MLSpecModule`) spells it `num_records` — a core Protocol may not force every future domain to call its rows "requests".

**TRAP (THIS COSTS A DEMO):** The generator MUST compute labels by calling YOUR latent function + noise. If the generator invents its own labels independently, the target is noise, the model finds nothing (r² ≈ 0, accuracy ≈ chance), and the conformal interval balloons to cover the whole range. `distinct=False` at the end of training means this happened. Check the generator (step 3) before debugging the model.

### Piece 3: `domain/generator.py` — Synthetic Data Generator

**You write:** `GeneratorConfig` (num_entities, resolved_fraction, noise_scale, use_llm, seed), procedural draw functions for YOUR entity types, LLM-fabrication prompt text (requested by `ModelRole.CHEAP` for bulk text, `ModelRole.GENERATION` for richer content — never a hardcoded model id), `DOMAIN_SERIES_LABEL` (a SENTENCE a client reads — the chart title on the forecast dashboard), `DOMAIN_SERIES_UNIT`, `domain_series_events(num_records, seed)`.

**TRAP:** `generate_synthetic_sync(config)` MUST return schema-valid records with `complete=None` (no LLM at all). This is what makes the system demonstrable while the model key is still being sorted out. The hybrid pattern is: seeded procedural structure → label from latent function + noise → LLM text enrichment (optional) → templated text fallback.

**TRAP:** Labels must come from `ml_spec.latent_*()` + Gaussian noise. Do not invent labels in the generator. The coupling point is where the latent function's name appears — the generator imports it and calls it.

### Piece 4: `domain/tools.py` — Action Tools + Risk Registry

**You write:** Argument Pydantic models (one per tool, with `extra="forbid"`), tool handler functions (async, idempotent — re-running with same args converges, reversible — returns inverse action, audited — calls `_emit_audit`), TOOL_REGISTRY with honest RiskLevel for each tool, ALLOWLIST persona → allowed tool names.

**TRAP:** An UNREGISTERED tool name resolves to HIGH risk (the safe default — a hallucinated name cannot slip under the gate). Forgetting to register means it requires approval rather than runs unguarded, which looks like an over-cautious gate rather than a bug. Do not rely on that.

**TRAP:** `read_only`, `destructive`, and `idempotent` are per-tool assertions, never derived from risk tier. Risk does not imply idempotency — appending a note is LOW risk and NOT idempotent, while a gated status transition is HIGH risk and IS idempotent. The defaults (all False) are the cautious reading — a tool registered without thinking is never advertised as safer than it is.

### Piece 5: `domain/personas.py` — Who Is Served

**You write:** Persona definitions (id, role → maps to RBAC role enum, display_name, description, data_scope → ALL or OWN with subject_field, prompt_key → key in SYSTEM_PROMPTS), PERSONA_BY_ROLE mapping EVERY role → persona id, DEFAULT_PERSONA_ID.

**TRAP:** A role with no persona = `KeyError` on login. Every authenticated principal resolves through `persona_for_role(role)`. Missing entry = cannot sign in. No test in the repo goes through the login path — the adapter suite, agent suite, conformance suite and ruff all stay green. This was the single most expensive defect from a real retarget rehearsal, and it cost the most to find because nothing failed until a human typed a password.

### Piece 6: `domain/prompts.py` — System Prompts

**You write:** `SYSTEM_PROMPTS` (one per persona prompt_key), domain-specific instructions naming your entities, tools, and rules.

**DO NOT CHANGE `PLATFORM_FLOOR`.** This is Aegis's non-negotiable platform preamble — the half no tenant may edit. It is composed UNDER every prompt version at render time, so a tenant can write whatever they want in their version and the floor is always underneath. DO NOT CHANGE IT. The floor text contains: platform rules, scope confinement, id-fabrication prohibition, tool-only gating, untrusted data rule.

### Piece 7: `domain/memory_spec.py` — What Is Worth Remembering

**You write:** `FACT_TYPES` (kinds of durable facts), `PROFILE_FIELDS` (structured profile), `PROFILE_ALIASES` (optional — predicate spellings → field names), `FACT_EXTRACTION_PROMPT` (cheap-model extractor system prompt), `IMPORTANCE_HINTS` (domain 1-10 guidance), `SKILLS_DIR` (path to skills/), `select_skills` hints dict (keyword → playbook filename).

**TRAP:** `memory_spec` is imported as a MODULE OBJECT by `backend/src/app/memory/__init__.py:33`. The path and every member name must stay stable. It is NOT in `adapter/__init__.py`'s `__all__`.

**TRAP (HOUR-LONG DEBUG):** Adding a playbook to `skills/` without updating the `hints` dict in `select_skills` = playbook is NEVER selected. Nothing warns you. The hints dict is a literal mapping of keyword → filename (without .md). If a keyword doesn't point to an existing file, skills are never injected, and the agent acts without procedural guidance. You will read that as a prompt problem and spend an hour in the wrong file.

### Piece 8: `domain/roster.py` — Agent Specialists + Fan-Out Team

**You write:** Specialist definitions with roles ("qa" or "memory" ONLY — the graph has handler nodes for these two roles), descriptions, keywords (lowercase phrase hints, first-match-wins), is_default (exactly one). Sub-agent team with tool_allowlists.

**TRAP:** A role that is NOT "qa" or "memory" silently falls back to qa pipeline with a log warning. It does NOT raise. You won't know your specialist doesn't work unless you read the `routing` stream event.

**TRAP:** Every tool name in every `sub_agent_roster()` entry's `tool_allowlist` MUST exist in TOOL_REGISTRY. Stale names are silently intersected-out. The sub-agent runs with fewer tools than you think — or zero tools — and nothing warns.

### Piece 9: `domain/corpus/*.md` — Seed Knowledge Documents

**You write:** Markdown files with YAML frontmatter: `id`, `kind` (must be a valid DocumentKind value from your schema), `category`, `tags` (comma-separated in brackets), `title`. The body is free Markdown.

The loader (`__init__.py`) parses `*.md` files automatically. Drop a new `.md` file — no code change needed.

### Piece 10: `domain/skills/*.md` — Procedural Playbooks

**You write:** How-to-act playbooks in Markdown. Must update `memory_spec.select_skills` hints dict to map keywords to filenames (without `.md`).

## 8. What NOT to Touch

- `aegis/src/aegis/` — the entire core library. Your edits go in `backend/src/app/adapter/` only.
- `backend/src/app/agent/` — agent graph and orchestration
- `backend/src/app/memory/` — memory subsystem
- `backend/src/app/retrieval/` — hybrid retrieval
- `backend/src/app/api/` — HTTP routes
- `backend/src/app/ml/` — ML spine host integration
- `backend/src/app/guardrails/` — rail stack
- `backend/src/app/governance/` — RBAC/RLS
- `backend/src/app/data/` — data layer
- `backend/src/app/forecast/` — forecast host
- `backend/src/app/observability/` — OTel setup
- `backend/src/app/seed.py` — seeder

**The conformance core check enforces this.** It scans every `.py` file outside `backend/src/app/adapter/` for the reference domain's vocabulary and fails with file, line, and word if any appears.

## 9. Windows Environment Notes

- **No Docker, no WSL, no GPU.** Everything runs natively on Windows as local processes.
- **Redis** → Memurai. Windows-native Redis-compatible service. Same wire protocol, same port 6379. `redis-py` drives it unchanged. Install: `winget install Memurai.Memurai`.
- **Qdrant** → `qdrant-x86_64-pc-windows-msvc.zip`. Unzip, run `qdrant.exe`. Listens on `:6333`. Both retrieval and LightRAG write to this one node. In-process mode available for tests.
- **PostgreSQL** → Native Windows install via `winget`. Database `taif`, serving role `aegis_app` (NOSUPERUSER NOBYPASSRLS).
- **Neo4j** → Neo4j Desktop via `winget`. Manual: create instance, set password, put same value in `NEO4J_PASSWORD` env var.
- **All pip packages** ship prebuilt Windows wheels (`win_amd64`) or are pure Python (`py3-none-any`).
- **Model inference** is API-only against the Azure/GenAI Lab fleet — no local GPU. The models run on CPU (`XGBoost tree_method="hist"`, `onnxruntime CPUExecutionProvider`).
- **Prefect 3.x** runs with an ephemeral SQLite-backed server — `pip install prefect`, no separate server needed. All critical Windows bugs fixed in 3.0.3+.
- **SHAP / numba** JIT-compiles on first run (~30-60s), cached after.

## 10. The Final Gate — Run After Every Edit

```bash
# 1. Structural check — seconds, catches a whole missing piece
(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import app.adapter
from aegis.adapter import DomainAdapter, missing_members
assert not missing_members(app.adapter), missing_members(app.adapter)
assert isinstance(app.adapter, DomainAdapter)
print('adapter contract: satisfied')
")

# 2. Conformance — 14 checks, under a second, no infra
(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)

# 3. Adapter + agent suite — on injected fakes, no database, no key
(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter tests/agent -q)

# 4. Full backend suite — ~1174 tests
(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest -q)

# 5. Core package suite — ~2268 tests, must stay exactly as green as baseline
(cd aegis && PYTHONPATH=src ../backend/.venv/bin/python -m pytest -q)

# 6. Lint — must be clean
backend/.venv/bin/python -m ruff check aegis backend

# 7. Train ML spine — reads your adapter's ml_spec
(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml)
# Read the last line. distinct=True = model learned signal.
# distinct=False = target is noise → check generator's latent function coupling.

# 8. AutoML pipeline — Prefect, FLAML, Optuna
(cd backend && PYTHONPATH=src:.venv/bin/python -m aegis_ml.ml.train)
```

## 11. Quick Reference: Key File Paths in the Aegis Repository

| What | Where |
|---|---|
| DomainAdapter Protocol | `aegis/src/aegis/adapter.py` (609 lines) |
| ML spine public API | `aegis/src/aegis/ml/__init__.py` (245 lines) |
| ML spine engine | `aegis/src/aegis/ml/model.py` (966 lines) |
| ML spec resolver | `aegis/src/aegis/ml/spec.py` (175 lines) |
| ML response types | `aegis/src/aegis/ml/types.py` (153 lines) |
| ML dataset/synthesis | `aegis/src/aegis/ml/dataset.py` (127 lines) |
| ML provenance (digest) | `aegis/src/aegis/ml/provenance.py` |
| Forecast engine | `aegis/src/aegis/forecast/engine.py` (537 lines) |
| Forecast types | `aegis/src/aegis/forecast/types.py` |
| Evals metrics | `aegis/src/aegis/evals/metrics.py` |
| Conformance checks | `aegis/src/aegis/conformance/test_conformance.py` (1036 lines) |
| Conformance vocabulary | `aegis/src/aegis/conformance/_vocabulary.py` (149 lines) |
| Conformance report helper | `aegis/src/aegis/conformance/_report.py` |
| Agent graph (plan-gate-act) | `aegis/src/aegis/agent/graph.py` (1954 lines) |
| Agent deps (AgentDeps seam) | `aegis/src/aegis/agent/deps.py` |
| Agent sub-agent spec | `aegis/src/aegis/agent/subagent.py` |
| Agent team fan-out | `aegis/src/aegis/agent/team.py` |
| Agent router | `aegis/src/aegis/agent/router.py` |
| Retrieval types (RetrievalScope) | `aegis/src/aegis/retrieval/types.py` |
| Core types (RiskLevel, GuardVerdict) | `aegis/src/aegis/core/types.py` |
| Core lazy imports (require) | `aegis/src/aegis/core/lazy.py` |
| Reference adapter __init__ | `backend/src/app/adapter/__init__.py` (172 lines) |
| Reference schema | `backend/src/app/adapter/schema.py` (274 lines) |
| Reference ml_spec | `backend/src/app/adapter/ml_spec.py` (357 lines) |
| Reference generator | `backend/src/app/adapter/generator.py` (791 lines) |
| Reference tools | `backend/src/app/adapter/tools.py` (823 lines) |
| Reference personas | `backend/src/app/adapter/personas.py` (168 lines) |
| Reference prompts | `backend/src/app/adapter/prompts.py` (133 lines) |
| Reference memory_spec | `backend/src/app/adapter/memory_spec.py` (228 lines) |
| Reference roster | `backend/src/app/adapter/roster.py` (213 lines) |
| Reference corpus loader | `backend/src/app/adapter/corpus/__init__.py` (78 lines) |
| Backend pyproject.toml | `backend/pyproject.toml` (226 lines) |
| Aegis pyproject.toml | `aegis/pyproject.toml` (199 lines) |
| Aegis PUBLIC.md (Stable API) | `aegis/PUBLIC.md` (266 lines) |
| System architecture docs | `docs/architecture/system-architecture.md` (383 lines) |
| SKILL.md (retarget procedure) | `SKILL.md` (repo root, 556 lines) |
| AGENTS.md (coding agent guide) | `AGENTS.md` (repo root, 180 lines) |
| CHANGELOG.md | `CHANGELOG.md` (repo root, 313 lines) |

## 12. Original User Prompt (Your Full Context)

The user's exact request that created this `aegis_ml` directory:

> "Ok so this is my Aegis enterprise application. Right now based on the problem
> statement on the hackathon day, we can have an ML-based need also. I have made
> `/Users/yrevash/aegis_ml` this folder and my agent setup. I want to plan the most
> SOTA implementation with modules and pipelines and MDs made so that on the
> hackathon day I just give my problem statement and my requirement to my agent and
> it has a good base to start from as the data is going to be synthetic. I want all
> best auto sklearn AutoML ML Ops full thing, plus everything like data schema,
> everything. I want that aegis_ml should that I can give to my main Aegis agent
> and it can integrate gracefully without any issue. That is my main."

Additional constraints from follow-up Q&A:
- **Integration model:** Standalone adapter package — `cp -r domain/* ../backend/src/app/adapter/` on hackathon day
- **ML scope:** Regression + Classification + Time Series Forecasting
- **Generator strategy:** Procedural structure + LLM text enrichment + templated fallback (the Aegis hybrid pattern)
- **Pipeline engine:** Prefect 3.x ephemeral (Windows-compatible, SQLite-backed, no infra)
- **MLOps storage:** Local filesystem + joblib + metadata JSON (matches Aegis's `ml/artifacts/` convention)
- **Environment:** Windows native — no WSL, no Docker, no GPU, pip-installable only
- **Excluded by design:** auto-sklearn (Linux-only) · MLflow (requires server) · Metaflow (requires AWS/local services) · Docker/WSL · GPU

## 13. What's Included vs. What's NOT

**Included in aegis_ml:**
- 10 domain template files (the adapter pieces), each with docstrings saying exactly what to write
- Full reference domain documentation (what you're replacing, with entity/enum/tool/persona tables)
- DomainAdapter Protocol contract (every member name, every attribute, every trap documented)
- AutoML engine: FLAML (fast model selection) + Optuna (hyperparameter optimization) + TPOT (genetic pipeline search)
- Time-series forecasting: wrapping Aegis's Nixtla StatsForecast (AutoARIMA/ETS/SeasonalNaive + ConformalIntervals)
- SHAP explainability: global importance, per-prediction explanation, waterfall/beeswarm plots
- ONNX model export + ONNX Runtime inference (CPU)
- Evidently: data drift, target drift, prediction drift, regression/classification performance monitoring
- Model registry: local filesystem, versioned, staging→production→archived promotion
- Model card generation: extending Aegis's existing ModelCard with AutoML search summary
- Prefect 3.x pipeline flows: train, forecast, evaluate, drift monitor
- Declarative config files: automl.toml, forecast.toml, monitoring.toml, pipeline.toml
- Test templates: schema, ml_spec, generator, tools, adapter protocol, ML pipeline integration
- Demo Jupyter notebook skeleton
- All dependencies documented with Windows wheel availability

**NOT included (by design):**
- auto-sklearn: Linux-only. Requires Unix `resource` module, SWIG, GCC toolchain. Official docs say "cannot run on Windows."
- MLflow: requires a tracking server (local or remote). Adds infrastructure dependency with no hackathon-day benefit.
- Metaflow: requires AWS services or local services. Overkill for synthetic data training.
- Docker/WSL dependencies: every component is native Windows, pip-installable.
- GPU support: all models run on CPU (XGBoost `tree_method="hist"`, `onnxruntime CPUExecutionProvider`, no CUDA).
- Changes to Aegis core: `aegis/src/aegis/` is NEVER touched. The conformance core check enforces this.
