# CHECKLIST — the whole day on one page

Print this. Tick as you go. Every box maps to a prompt-pack or a doc section.

---

## The four things that lose the demo

| # | Trap | Guard |
|---|---|---|
| 1 | **A noise target passes all 14 conformance checks.** Only `distinct=False` from `python -m app.ml` catches it natively — minutes before the demo. | `aegis-ml contract` → `assert_learnable`, in seconds |
| 2 | **`resolve_spec` silently returns `FALLBACK_SPEC`** (4 columns of noise) on a misspelled `FEATURE_NAMES` or `TARGET.name`. Nothing raises. | Generate `ml_spec.py` from `MLProblem`; check #12 is the backstop |
| 3 | **Four `web/` files show the old domain's words.** The vocabulary scan is Python-only. | `prompts/13-console.md` |
| 4 | **`cp -r` leaves 3 corpus docs + 2 playbooks behind** and retrieval serves them. | `rsync -a --delete` / `robocopy /MIR` |

**And the one required core edit:** `aegis/src/aegis/conformance/_vocabulary.py`. Required, sanctioned, and it must be reported. Nothing else under `aegis/src/aegis/` changes.

**And: never quote a test count from a document.** `AGENTS.md` and `README.md` disagree (2247 vs 2268; 1121 vs 1174). Record what you actually get.

---

## Phase 0 — Environment

- [ ] `cd /Users/yrevash/aegis_ml && uv sync --extra dev && uv run aegis-ml doctor`
- [ ] `cd /Users/yrevash/aegis && ./scripts/bootstrap.sh` (idempotent)
- [ ] Trainer venv: `uv venv .venv-ml --python 3.11 && uv pip install --python .venv-ml -e '.[strong,serve]'` *(skippable — `baseline,flaml` run without it)*
- [ ] `uv pip install --python /Users/yrevash/aegis/backend/.venv -e '/Users/yrevash/aegis_ml[serve]'`
- [ ] Caps intact: `pandas 2.2–2.3`, `numpy <2.5`, `numba 0.67.0`
- [ ] **Baseline recorded:** `(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter tests/agent -q)` → ______
- [ ] `missing_members(app.adapter) == []` on the *unmodified* checkout

## Phase 1 — Domain Brief · `prompts/00-intake.md`

- [ ] `/Users/yrevash/aegis_ml/DOMAIN_BRIEF.md` exists, all 15 sections
- [ ] 8–12 features; every categorical has a full level set
- [ ] Target is not a feature; no feature is knowable only after the target
- [ ] §5 gives a **sign and magnitude for every driver**
- [ ] Exactly one interaction term
- [ ] 1–2 deliberately irrelevant features named
- [ ] One unobserved confounder named, with a magnitude
- [ ] `target_r2` in 0.50–0.70 (or accuracy 0.70–0.85)
- [ ] `PERSONA_BY_ROLE` covers `admin`, `ai_team`, `devops`, `client`
- [ ] At least one HIGH-risk tool
- [ ] Roster roles are exactly `qa` and `memory`; `team` absent
- [ ] §14 quarantine list is specific identifiers only

## Phase 2 — The ten pieces, in order

### 1 · `schema.py` — `prompts/01-schema.md`
- [ ] `SCHEMA_VERSION: str`; `StrEnum` (not `Enum`) for every vocabulary
- [ ] **`SyntheticDataset` keeps that exact name**, with `metadata`, entity lists, `*_by_id`, `labelled_*()`
- [ ] Target field is `| None` on its entity; a `Document` model exists

### 2 · `ml_spec.py` — `prompts/02-ml-spec.md`
- [ ] `FEATURE_NAMES`, `FEATURES`, `TARGET` at module scope, non-empty
- [ ] **`resolve_spec(ml_spec) is not FALLBACK_SPEC`**
- [ ] `TARGET.task` exactly `"regression"` / `"classification"`; `.unit` set for regression
- [ ] `CATEGORICAL_FEATURES` declared; levels derived from `StrEnum`s
- [ ] `latent_*` is pure Python, monotone, floored, with **one interaction term**
- [ ] `latent_*` does **not** read the irrelevant features
- [ ] `training_frame(*, num_records, seed)`; pandas imported inside the body
- [ ] `describe_prediction` names **your** target and unit

### 3 · `generator.py` — `prompts/03-generator.md` ⚠ **the expensive one**
- [ ] Label is `ml_spec.latent_*(ml_spec.features_for_*(...)) + noise` — both **called**
- [ ] **One** `random.Random(cfg.seed)` threaded everywhere
- [ ] `noise_scale` derived from `target_r2`, formula in the docstring
- [ ] Heteroscedastic σ; unobserved confounder; MAR missingness
- [ ] `generate_synthetic_sync` does not `await` and needs no LLM
- [ ] Both entry points give the same structure and labels
- [ ] Every count knob is `num_*: int, ge=1`
- [ ] `DOMAIN_SERIES_LABEL` is a client sentence; events are **arrivals**
- [ ] Fixed seed → byte-identical dataset
- [ ] **`assert_learnable` passes; held-out score in 0.45–0.80 (not >0.90)**

### 4 · `tools.py` — `prompts/04-tools.md`
- [ ] Every tool has a real `RiskLevel`; **at least one HIGH**
- [ ] `destructive` / `idempotent` asserted per tool, independent of risk
- [ ] Handlers: typed args, audited, `ToolActionResult` with `previous_state` + `inverse`
- [ ] `run_tool` raises `UnknownToolError` then `ToolNotAllowedError` **before** side effects
- [ ] `ALLOWLIST` keys ⊆ persona ids; values ⊆ `TOOL_REGISTRY`
- [ ] The 5 `aegis_ml.serve.tools` specs registered — LOW, read-only
- [ ] `ALLOWLIST`, `AuditFn`, `RecordStore`, `ToolActionResult`, `ToolContext`, `ToolNotAllowedError`, `UnknownToolError`, `is_allowed`, `InMemoryRecordStore` all present

### 5 · `personas.py` — `prompts/05-personas.md` ⚠ **the login-killer**
- [ ] `DEFAULT_PERSONA_ID` ∈ `PERSONAS`
- [ ] **`PERSONA_BY_ROLE` has all four roles, every value ∈ `PERSONAS`**
- [ ] `Persona.tool_names` reads `ALLOWLIST` — no second copy
- [ ] One persona is row-scoped (`ScopeKind.OWN` + `subject_field`)
- [ ] `ScopeKind` and `Persona` keep those names (piece 6 imports both)

### 6 · `prompts.py` — `prompts/06-prompts.md`
- [ ] `SYSTEM_PROMPTS` has an entry for **every** `prompt_key` (no silent `.get` default)
- [ ] `PLATFORM_FLOOR` appears **verbatim** in every rendered prompt
- [ ] Scope and tool clauses **derived** from `data_scope` / `TOOL_REGISTRY`, not literals
- [ ] Order: task → floor → extra context

### 7 · `memory_spec.py` — `prompts/07-memory-spec.md`
- [ ] All seven constants present, exact names; `FactSchema` / `FactExtraction`
- [ ] `SKILLS_DIR` is a **`str`** from `__file__`
- [ ] `FACT_EXTRACTION_PROMPT` embeds `IMPORTANCE_HINTS` and says what **not** to extract
- [ ] `memory_subject_for(None)` → `None`; the subject is the right one (leak risk)
- [ ] `render_profile` iterates `PROFILE_FIELDS`
- [ ] Module path and symbol names unchanged (`set_default_spec` binds the module object)

### 8 · `roster.py` — `prompts/08-roster.md`
- [ ] Roles are **exactly** `"qa"` and `"memory"`; `"team"` absent
- [ ] Exactly one `is_default=True`; the default has no keywords
- [ ] Every `tool_allowlist` name ∈ `TOOL_REGISTRY`
- [ ] Every `SubAgentSpec` `label` and `system_prompt` re-voiced
- [ ] **No `SPECIALIST_NODES` edit** (or it is in the report)

### 9 · `corpus/` — `prompts/09-corpus.md`
- [ ] `load_seed_corpus()` via `importlib.resources`, sorted by id, `source="seed"`
- [ ] 3–6 docs, each >400 chars with real `##` headings, unique stable ids
- [ ] ≥2 different `kind`s; one explains why a HIGH-risk tool needs approval; one names the ML target

### 10 · `skills/` + the `hints` table — `prompts/10-skills.md` ⚠ **two files**
- [ ] 2–4 playbooks: when-it-applies, ordered steps, a **"Never"** section
- [ ] Each names your tools; one names the HIGH-risk tool and its gate; one uses the ML tools and insists on the interval
- [ ] **The `hints` table in `memory_spec.py` names every file on disk and nothing else**
- [ ] `select_skills` filters on `available` and returns `None` when nothing matches

### Gate for phase 2
- [ ] `missing_members(reference.adapter) == []` and `isinstance(...) is True`
- [ ] `uv run aegis-ml contract` passes

## Phase 3 — Integration · `prompts/12-integration.md`

- [ ] **`rsync -a --delete`** (or `robocopy /MIR`), trailing slash on the source
- [ ] `__pycache__` cleared under the adapter directory
- [ ] Imports rewritten `reference.adapter.*` → `app.adapter.*`
- [ ] **The 3 old corpus docs and 2 old playbooks are gone**
- [ ] `missing_members(app.adapter) == []`; every host-bound symbol resolves
- [ ] `backend/tests/adapter/*` rewritten — **leaving** `test_piece_manifest.py`, `test_domain_adapter_protocol.py`, `test_conformance_suite.py`, `broken_adapter/`
- [ ] `test_ml_spec.py` carries the learnability + no-fallback assertions
- [ ] **`_vocabulary.py` updated** — `SHIPPED_DOMAIN_ID` + every term, specific identifiers only
- [ ] Python leak grep clean outside `conformance/`
- [ ] OpenAPI + TS client regenerated if a route or model moved

## Phase 4 — Console · `prompts/13-console.md`

- [ ] `web/src/config/personas.ts` — ids match the adapter **exactly**; query 0 single-clause; query 1 four clauses; query 2 matches `memory` keywords; no literal record ids
- [ ] `web/src/components/ops/opsShared.ts` — `PROMPT_KEY` ∈ `SYSTEM_PROMPTS`; your HIGH-risk tool named with the right tier
- [ ] `web/src/components/sim/SimulationView.tsx` — both persona ids, labels, `SIM_QUERY` that differs by persona
- [ ] `web/src/components/ml/MLOpsView.tsx` — `EXAMPLE.features` keys **are** `FEATURE_NAMES`; declared levels only; ~85th-percentile row
- [ ] TS + prose sweeps clean
- [ ] `npx tsc --noEmit && npm test && npx next build`

## Phase 5 — ML · `prompts/11-ml-pipeline.md`

- [ ] `python -m app.ml` — your target/task, and **`distinct=True`**
- [ ] `aegis-ml train --tier all` — leaderboard has losers on it
- [ ] `aegis-ml eval` — metric **0.45–0.80** (not >0.90); coverage within ±0.05 **both ways**; no collapsed slice; `data_source` ≠ `"synthetic"`; digest present
- [ ] `shap.html` — irrelevant features flat; real drivers with the right signs
- [ ] `aegis-ml promote` — a `GateDecision` with numbers, pass or fail; backend restarted
- [ ] `aegis-ml drift` — a verdict plus a NannyML `estimated_*` figure
- [ ] `aegis-ml forecast` — chart title is your `DOMAIN_SERIES_LABEL`
- [ ] 5 ML tools registered, allowlisted, round-trip tested
- [ ] TabPFN-touched artefacts carry the Prior Labs notice

## Phase 6 — Final gate · `prompts/14-final-gate.md`

- [ ] Structural check: `adapter contract: satisfied`
- [ ] **Conformance: 14 passed**
- [ ] `tests/adapter tests/agent` ≥ your baseline
- [ ] Full backend suite ≥ your baseline
- [ ] Core package suite unchanged
- [ ] `ruff check aegis backend` clean
- [ ] Console tsc + test + build
- [ ] `aegis_ml` tests + ruff + `contract` + `doctor`
- [ ] All three leak sweeps return nothing
- [ ] **Signed in as `admin` AND as `client`** — the `KeyError` path no test covers
- [ ] The human gate fired on a real HIGH-risk action; an approval completed the run
- [ ] ML evidence in a transcript, with interval and drivers
- [ ] No screen shows the reference domain's words
- [ ] Report written: core edit named, four console files named, **measured** numbers only

---

## The five commands to keep in your shell history

```bash
# 1 — the contract, seconds, no infrastructure
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import app.adapter; from aegis.adapter import DomainAdapter, missing_members
print('missing:', missing_members(app.adapter)); print('satisfies:', isinstance(app.adapter, DomainAdapter))")

# 2 — the fourteen checks, under a second
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)

# 3 — the cheap ML gate: pandera + learnability + leakage
cd /Users/yrevash/aegis_ml && uv run aegis-ml contract

# 4 — the fast loop
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter tests/agent -q)

# 5 — the native backstop; distinct=True is the pass signal
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml | tail -1)
```

PowerShell: use `Push-Location` / `Pop-Location` and `$env:PYTHONPATH = "src;..\aegis\src"` (semicolon, not colon). Full translations in `docs/08-windows.md` §3.

---

## When stuck

1. Conformance suite — every failure names the fix, the consequence and the defect it came from.
2. `missing_members` — names a whole piece you forgot.
3. `aegis-ml contract` — pandera + learnability + leakage, in seconds.
4. Last line of `python -m app.ml`.
5. Grep the console for the shipped vocabulary.
6. `docs/09-troubleshooting.md` — symptom → cause → fix.
7. **If the failure did not raise, it is in `docs/09-troubleshooting.md`.**
