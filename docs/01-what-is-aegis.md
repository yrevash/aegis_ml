# 01 · What Aegis is

Repo: `/Users/yrevash/aegis/`. Read-only except for the paths named in `docs/07-integration-with-aegis.md`.

---

## 1. The thesis, in three sentences

> **"Autonomy you can audit."**

1. **Import, not fork.** `aegis/src/aegis/` is a library you install. The domain lives entirely in one adapter directory. *"If you are making the core learn something about the domain, you have gone the wrong way."* (`AGENTS.md`)
2. **The instrumentation is the product.** Most agent stacks are a framework plus glue: they can call tools but cannot tell you *why* they acted, *what* they read, *who* approved it, or *what it cost*. Aegis is built the other way around.
3. **Every autonomous action is uncertainty-bounded, explainable, guarded, human-approved and fully traced.**

---

## 2. The four layers

```
Browser
  │  HTTPS · JWT · SSE
  ▼
1 · Console            web/                     Next.js 15 · React 19 · TypeScript
  │                                             landing page + role-scoped portals
  ▼  fetch + SSE
2 · Composition root   backend/src/app/         FastAPI · app factory · routes · JWT
  │                                             RBAC · tenant scoping · sweepers
  ▼  imports · injected deps
3 · Importable core    aegis/src/aegis/         agent · gateway · guardrails · retrieval
  │                                             memory · ml · governance · ops · evals
  │                                             observability · redteam · data · core
  ▼  async drivers
4 · Stores and sinks   Postgres · Qdrant · Neo4j · Redis · Arize Phoenix
```

And, cutting across layer 2:

```
Domain adapter   backend/src/app/adapter/       schema · ml_spec · generator · tools
                                                personas · prompts · memory_spec
                                                roster · corpus/ · skills/
                 ── the only seam that changes per domain ──
```

---

## 3. The request path

```
POST /query
  → guard_input        input rails: injection, PII, schema, topical scope (fail-closed)
  → route              supervisor picks a specialist (qa | memory | team)
  → recall_memory      episodic + semantic + procedural
  → retrieve           vector + graph + BM25 → RRF → rerank; every claim cited
  → ml_predict         [see the note below]
  → plan               the model proposes actions
  → gate               human approval, decided by the TOOL'S RISK TIER
  → (approval interrupt — the run checkpoints durably and resumes on any worker)
  → act                the tool runs; the result is audited
  → reflect
  → generate
  → guard_output       output rails
  → persist_memory
```

> **Verified correction.** `ml_predict` appears in the Aegis README's request path but **there is no `ml_predict` node in `aegis/src/aegis/agent/graph.py`**. `NODE_LABELS` and the graph builder do not declare one, and `describe_prediction` — the adapter member that renders a prediction into the plan — has **zero consumers** anywhere in `backend/src/`, `aegis/src/` or `web/src/`. The prose describes an intention that is not wired.
>
> **`aegis_ml`'s answer is `aegis_ml.serve.tools`**: ready-made `ToolSpec`s you drop into your adapter's `TOOL_REGISTRY`, so the agent reaches the model through the *tool* path that already exists and is already gated. **This needs no core edit.** See `docs/07-integration-with-aegis.md` §7.

Two rules the whole design hangs on:

1. **ML informs, it never gates.** The prediction and its conformal interval are *evidence* injected into the plan. The human gate fires on a **tool's risk tier** — never on model confidence. Honour this: do not build a "the model said no, so we blocked it" flow.
2. **A gated run checkpoints durably** and resumes on any worker from a persisted approvals-inbox row.

---

## 4. The trust stack

Six checkpoints stand between the model and a real action.

| # | Checkpoint | Mechanism |
|---|---|---|
| 01 | Input rails | injection classification · PII · schema · topical scope, fail-closed |
| 02 | Retrieval | vector + graph + BM25 → RRF → rerank, every claim cited |
| 03 | Signal | conformal interval + SHAP drivers |
| 04 | Human gate | by tool risk tier |
| 05 | Governance | budget enforced before spend · row-level security |
| 06 | Audit | OpenTelemetry trace + append-only audit row |

Your adapter feeds checkpoints 01 (`DOMAIN_DESCRIPTION` → `allowed_topics`), 02 (`corpus/`), 03 (`ml_spec`), 04 (`tools.TOOL_REGISTRY` risk tiers) and 05 (`personas.PERSONAS[*].data_scope`).

---

## 5. The six invariants

From `/Users/yrevash/aegis/AGENTS.md` §Boundaries. These are invariants, not preferences. Most have a test.

| # | Invariant | Why, and how it is enforced |
|---|---|---|
| 1 | **Never import `app.*` from `aegis.*`.** | The core has zero knowledge of the host. Verify: `grep -rn "^from app\.\|^import app\." /Users/yrevash/aegis/aegis/src/` returns nothing. |
| 2 | **Never import one leaf `aegis` module from another leaf.** | Hoist the shared type into `aegis.core`, which must stay dependency-free. `aegis/tests/core/test_core_is_dep_free.py` imports it in a subprocess and fails if litellm, torch, langgraph, xgboost, fastapi, redis, nemoguardrails, sqlalchemy, jwt, argon2 or opentelemetry appears in `sys.modules`. |
| 3 | **Every tenant-scoped table must be covered by `aegis.governance.rls`.** | `audit_rls_enforcement` reports any live unprotected table. Five cross-tenant leaks have shipped and been closed. |
| 4 | **No silent fallbacks.** | *"A control that cannot run fails closed and says so — it never degrades quietly into something that looks like it worked. … a silent downgrade is worse than an outage because nobody goes looking."* **This is the single most important rule in the codebase**, and `aegis_ml` mirrors it: every refusal in `aegis_ml.contracts.errors` exists because the alternative was a number a human would have believed. |
| 5 | **Domain logic never leaks into the core.** | Enforced, not asserted: conformance check #14 scans every module outside the adapter for the shipped domain's vocabulary and fails naming the file and line. The quarantined word list is `aegis/src/aegis/conformance/_vocabulary.py`; **if you change what the adapter calls things, update it in the same commit.** |
| 6 | **Optional dependencies go through `aegis.core.require(extra, module)`**, which raises naming the exact `pip install`. | Never `except ImportError: pass`. `aegis_ml._require.require` is the mirror. |

**Conventions that apply to code you write inside the Aegis checkout:** Python 3.11; ruff with `E,F,I,UP,B,SIM,ANN,D`; Google-style docstrings; line length 100; annotations required. Docstrings carry the *reasoning*, often naming the defect that caused the design. A docstring that restates the parameter names is noise. `aegis_ml/pyproject.toml` mirrors that ruff config exactly, so one `ruff check` across both trees gives one verdict.

---

## 6. The three run modes

A demo never depends on infrastructure being healthy.

| Mode | What runs | Needs |
|---|---|---|
| `safe` | Console only, in-browser mock transport | nothing |
| `lite` | Real agent, no databases (SQLite audit) | a model API key |
| `full` | Everything, all four server stores | key + Postgres, Neo4j, Redis, Qdrant |

No Docker, no GPU, no WSL anywhere. Every store is a native local install. On Windows, Redis is **Memurai** — same wire protocol, same port, no config change.

Start: `./scripts/bootstrap.sh && ./scripts/dev-native.sh` then `cd web && npm run dev`.
Windows: `.\scripts\install-windows.ps1` then `.\scripts\start.ps1 -Mode full`.

Seed accounts first — `cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.seed`. There is no fallback login table; an unseeded backend answers 503 and says so. Logins: `admin` / `ai` / `devops` / `client`, plus `northwind.admin` / `vertex.admin`, password `demo`.

---

## 7. **What Aegis already gives you for free — do not rebuild any of this**

This is the most important section in this document. Aegis is not a thin shell. Building any of the following again wastes hours and produces something weaker.

### 7.1 The trustworthy-ML spine — `aegis.ml`

`aegis/src/aegis/ml/model.py` (966 lines) is a complete, calibrated, explainable supervised-learning stack:

| Capability | Implementation |
|---|---|
| Ensemble | XGBoost + HistGradientBoosting **soft-voting**; `_regression_members()` / `_classification_members()` |
| Uncertainty | **MAPIE split-conformal**, calibrated on a **disjoint** calibration split |
| Explanation | **SHAP**, dispatched per member family (`TreeExplainer` for trees, `LinearExplainer` for linear, `PermutationExplainer` otherwise), averaged across members by voting weight, with `value_label` carrying the categorical level name so a UI can render `region = emea` and not `region = 1.0` |
| Honesty | `ModelCard` separates **requested** coverage (`conformal_coverage`) from **measured** coverage (`conformal_coverage_empirical`). Never one field. |
| Provenance | SHA-256 `dataset_digest` (`'sha256:<hex>'`), invariant to column order and index, sensitive to any cell value, row order, dtype or added/removed column |
| Refusal | `MLModelUnavailableError` instead of silently training on the noise synthesiser and serving it as domain evidence |
| Encoding | One-hot for the declared categorical subset; numerics passed through |

Public surface (`aegis.ml.__all__`): `TrustworthyModel`, `train`, `load`, `get_model`, `predict_explain`, `resolve_spec`, `frame_digest`, `ResolvedSpec`, `MLSpec`, `FALLBACK_SPEC`, `ModelCard`, `MLExplainResponse`, `ShapFeature`, `EnsembleMember`, `MLModelUnavailableError`, `TaskType`, `DEFAULT_ARTIFACT_PATH`.

Host wrapper (`backend/src/app/ml/__init__.py`): `train`, `load`, `get_model`, `predict_explain`, `DEFAULT_ARTIFACT_PATH` — which resolves to **`backend/.artifacts/ml_spine.joblib`**, deliberately *not* the library path under `aegis/src/aegis/ml/artifacts/`. Training through the library constant writes where nothing loads from and the endpoints keep answering 503.

`aegis_ml` **extends this and never replaces it.** The AutoML search finds a better configuration and hands it back as a portable `Recipe`; the Aegis spine then fits that recipe, keeping conformal calibration, SHAP, the card and the digest. See `docs/10-architecture-decisions.md` D1.

### 7.2 The forecaster — `aegis.forecast`

Nixtla StatsForecast (AutoARIMA / AutoETS / SeasonalNaive) with `ConformalIntervals` and rolling-origin backtests. `ForecastResult.candidates` reports **losers as well as the winner**, so you can see whether the winner won by a nose or a mile.

Public surface includes `forecast_series`, `bucket_events`, `infer_freq`, `minimum_history`, `season_length_for`, `project_burndown`, `BacktestReport`, `ForecastResult`, `CandidateScore`, `DegenerateSeriesError`, `InsufficientHistoryError`.

Your adapter feeds it through three names in `generator.py`: `DOMAIN_SERIES_LABEL`, `DOMAIN_SERIES_UNIT`, `domain_series_events()`.

### 7.3 Guardrails

Six rail layers, programmatic-always-on plus NeMo Colang: prompt-injection classification, Presidio PII detection and masking (input, output *and* tool results), schema validation, topical scope. **The topical rail's `allowed_topics` is your adapter's `DOMAIN_DESCRIPTION`, wired straight through** (`backend/src/app/guardrails/__init__.py` imports it). A vague description is a loose rail.

### 7.4 Governance — RBAC and row-level security

Multi-tenant RBAC with Postgres `FORCE ROW LEVEL SECURITY` and a `NOSUPERUSER NOBYPASSRLS` serving role. Four platform roles: `admin`, `ai_team`, `devops`, `client`. Your adapter's `personas.PERSONA_BY_ROLE` maps each of those roles onto one of *your* personas. Budgets are enforced **before** spend.

### 7.5 The human gate

Risk-tiered approval on tools. `AgentConfig.gate_min_risk` (platform default `HIGH`) is compared against `ToolSpec.risk` and **nothing else** decides whether a proposed action stops for a human. An unregistered tool name resolves to `HIGH`, so a hallucinated tool can never slip under the gate. The gated run checkpoints to Postgres and resumes on a fresh worker.

### 7.6 Audit and tracing

Append-only audit rows (`app.data.record_audit`, which every tool handler calls) plus OpenTelemetry + OpenInference GenAI-semantic-convention spans exported to Arize Phoenix.

### 7.7 Evals

RAGAS-style deterministic proxies with no LLM call and no `ragas` dependency (`aegis/src/aegis/evals/metrics.py`), plus an LLM judge, an offline evaluation gate and a CI regression gate on retrieval quality.

### 7.8 Retrieval

Hybrid RAG: Qdrant vector + LightRAG/Neo4j graph + BM25 → Reciprocal Rank Fusion → local ONNX cross-encoder rerank. Each passage keeps its origin arm. Your `corpus/*.md` files seed it.

### 7.9 The console

Four role-scoped portals (`admin`, `ai_team`, `devops`, `client`) plus a public landing page. There is already an ML-Ops view, a simulation view, an approvals inbox, a knowledge-graph view and a guardrails view. You re-voice four files (§4.3 of `docs/00-START-HERE.md`); you do not build screens.

---

## 8. The twelve modules, and their honest tech

Branding, never hiding. Mirrors `backend/src/app/capabilities.py`, served at `GET /platform/capabilities`.

| Module | Tech underneath |
|---|---|
| Aegis Gateway | LiteLLM |
| Aegis Router | LangGraph |
| Aegis Memory | Postgres + Qdrant |
| Aegis Cache | Redis / Memurai |
| Aegis Retrieval | Neo4j/LightRAG + Qdrant |
| **Aegis Signal** | **XGBoost + MAPIE + SHAP** ← the spine `aegis_ml` extends |
| Aegis Guardrails | programmatic + NeMo Colang |
| Aegis Evals | RAGAS-style proxies + LLM judge |
| Aegis Loop | native (trace → eval → diagnose → tiered release) |
| Aegis Governance | Postgres RLS + JWT |
| Aegis Trace | OpenTelemetry → Phoenix |
| Aegis Tools / MCP | native + MCP SDK |

---

## 9. Where to go next

`docs/02-domain-adapter-contract.md` — the contract you must satisfy, in full.
