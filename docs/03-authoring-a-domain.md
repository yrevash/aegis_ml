# 03 · Authoring a domain

From a problem statement to ten filled pieces. Read `docs/02-domain-adapter-contract.md` first.

---

## 1. The workflow

```
problem statement
    │  prompts/00-intake.md
    ▼
Domain Brief                    ← everything downstream reads this
    │  aegis_ml.contracts.spec.MLProblem  (the ML half of the Brief, as code)
    ▼
templates/adapter/*  →  ten filled pieces  (in /Users/yrevash/aegis_ml/reference/adapter/ or a new dir)
    │  aegis-ml contract   (pandera + assert_learnable)
    ▼
rsync -a --delete  →  /Users/yrevash/aegis/backend/src/app/adapter/
```

**Author the adapter inside `aegis_ml`, verify it there, then sync it into Aegis.** Do not author directly in the Aegis checkout: you lose the ability to run `aegis-ml contract` against it cheaply, and a half-written adapter in the Aegis tree means every backend test fails at import while you work.

---

## 2. Get the Domain Brief first

**Do not write a line of adapter code before the Brief exists.** Every piece consumes the vocabulary the previous one defined; a piece written without the Brief will disagree with the pieces written after it, and you will discover that at integration time.

Run `prompts/00-intake.md` against the problem statement. It produces a structured Markdown document with these sections:

| Section | Feeds |
|---|---|
| Domain identity (`domain_id`, one-paragraph description) | `__init__.py`, guardrails `allowed_topics` |
| Entities and enums | piece 1 `schema.py` |
| Target: name, task, unit, levels, bounds | piece 2 `ml_spec.py` |
| Features: name, dtype, unit, levels, bounds, nullable | piece 2, the pandera contract, the feature pipeline |
| Latent drivers: feature → sign → magnitude, plus one interaction | piece 2's latent function, piece 3's label |
| Realism targets: `target_r2`, missingness rate, class balance, confounders | piece 3 `generator.py` |
| Series: label sentence, unit, arrival shape | piece 3's `DOMAIN_SERIES_*` |
| Personas: id, display name, RBAC role mapping, data scope | pieces 5 and 6 |
| Tools: name, args, risk tier, destructive, idempotent, per-persona allowlist | piece 4 |
| Roster: `qa` and `memory` re-voicing, sub-agent team | piece 8 |
| Memory: fact types, profile fields, subject scoping | piece 7 |
| Corpus: 3–6 seed document titles and topics | piece 9 |
| Skills: 2–4 playbook filenames and their trigger keywords | piece 10 |

Save it to `/Users/yrevash/aegis_ml/DOMAIN_BRIEF.md`. Every prompt-pack from `01-schema.md` onward opens it.

---

## 3. The editing order, and why

**Edit in this order. It is not arbitrary — each piece consumes the vocabulary the previous one defined.**

| # | File | You define | Depends on |
|---|---|---|---|
| 1 | `schema.py` | Entities and enums — the shared vocabulary | the Brief |
| 2 | `ml_spec.py` | What gets predicted, from which features, and the **latent ground truth** | 1 (enum `.value` strings become categorical levels) |
| 3 | `generator.py` | Synthetic records consistent with 1 and 2 | 1, 2 (**calls the latent function**) |
| 4 | `tools.py` | The real actions, each with a risk tier | 1 (tools read and write your records) |
| 5 | `personas.py` | Who is served, and what each may see and call | 4 (`Persona.tool_names` reads `ALLOWLIST`) |
| 6 | `prompts.py` | Who the agent is, per persona | 4, 5 (the floor renders live scope + tools) |
| 7 | `memory_spec.py` | What counts as a durable fact | 1, 5 |
| 8 | `roster.py` | Which specialists the supervisor may route to | 4 (`tool_allowlist` names must exist) |
| 9 | `corpus/*.md` | Seed knowledge documents | 1 (frontmatter `category` is a schema enum) |
| 10 | `skills/*.md` + the `hints` table in 7 | Procedural playbooks | 7 |

Backwards dependencies are what make out-of-order editing expensive. `personas.py` imports `ALLOWLIST` from `tools.py`. `prompts.py` imports `TOOL_REGISTRY` and `ScopeKind`. `generator.py` imports `ml_spec`. `ml_spec.py` imports the schema enums.

---

## 4. **The suite is red from piece 1 until piece 8. That is expected.**

`backend/tests/conftest.py` imports through `app.adapter`. The moment you replace the entity models, **every test in the repository fails at import** — a wall of `ImportError`, hundreds of lines, none of it about whether your edit was right. It stays that way until piece 8 lands and the registry's re-exports all resolve again.

> **Do not chase the wall. Do not "fix" it by loosening a conftest.**

Two things are meaningful mid-flight:

1. **The conformance suite** — green from your first edit to your last, needs no infrastructure, runs in under a second. **Lean on it.**
2. **The per-piece verify commands** below — but *only once you have rewritten the test file they run*. `backend/tests/adapter/*` is not domain-neutral scaffolding; it carries between 3 and 26 shipped-domain literals per file (`test_tools.py` 26, `test_allowlist.py` 19, `test_ml_spec.py` 13, `test_schema.py` 9, `test_generator.py` 7, `test_registry.py` 3). **Rewriting those tests is part of each step, not a follow-up.**

If you are authoring inside `aegis_ml/` and syncing at the end (§1), the wall does not appear until you sync. That is the main reason to work that way.

---

## 5. Piece by piece

Each entry: **what it is**, **what to write**, **the trap**, **verify**. The corresponding prompt-pack in `prompts/` has a worked example fragment.

Verify commands assume the adapter has already been synced into Aegis. While authoring in `aegis_ml/`, substitute `--aegis-adapter reference.adapter` and `PYTHONPATH=...:/Users/yrevash/aegis_ml`.

---

### Piece 1 — `schema.py`

**What it is.** The domain's record types and their version. The platform passes them around opaquely; only `SCHEMA_VERSION` is a Protocol member.

**What to write.**
- Pydantic v2 `BaseModel` per entity. Field constraints (`ge`, `le`, `gt`, `min_length`) are free validation — use them.
- `StrEnum` for every closed vocabulary. Their `.value` strings become your categorical feature *levels*, so choose them once and never re-spell them.
- `SCHEMA_VERSION: str` — bump it.
- **`SyntheticDataset`**: the container the generator returns and the ML spine reads. Give it a `metadata` field, one list per entity type, and lookup helpers (`*_by_id`) plus a `labelled_*()` method returning only the rows that carry a target value.
- A `DatasetMetadata` model carrying `schema_version`, `seed`, `llm_used`, and a count per entity type.

**Trap.** Keep the *container* names the registry re-exports even while you change every field inside them. `__all__` names `SyntheticDataset` specifically, and both the ML spine and the generator bind to it.

**Verify.**
```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter/test_schema.py -q)
```

---

### Piece 2 — `ml_spec.py`

**What it is.** The single source of truth for what is predictable. Conformance check #12 reads it; `resolve_spec` reads it leniently and falls back to noise.

**What to write.**

```python
FEATURES: list[FeatureSpec]           # name, dtype, description, levels for categoricals
FEATURE_NAMES: list[str]              # [f.name for f in FEATURES]
CATEGORICAL_FEATURES: list[str]       # not a Protocol member; declare it anyway
NUMERIC_FEATURES: list[str]
TARGET: TargetSpec                    # name, task, unit, description

def latent_<target>(features: dict) -> float:   # THE GROUND TRUTH — rename for your domain
def features_for_<record>(record, *, ...) -> dict
def feature_matrix(dataset) -> tuple[list[dict], list[float]]
def training_frame(*, num_records: int = 1200, seed: int = 7) -> pd.DataFrame
def describe_prediction(resp, *, top_k: int = 3) -> str
```

- **No heavyweight imports at module scope.** Keep this module pure Python + pydantic so it loads and tests without numpy/pandas/xgboost present. `training_frame` imports pandas *inside the function*.
- **The keyword is `num_records`.** Not `num_rows`, not your domain's noun. `aegis.adapter.MLSpecModule` names it.
- `describe_prediction` output is **injected into the plan as evidence**. Re-voice it or it will name the old target and unit out loud in front of a jury.
- Prefer generating this file: `aegis_ml.contracts.spec.emit_ml_spec_module(problem)` writes exactly the five names `MLSpecModule` requires and `resolve_spec` reads.

**Trap — this one costs a demo.** The generator must sample labels **around your latent function**. If it does not, the target is noise, the model finds nothing, and the conformal interval is honestly enormous. The coupling is kept in piece 3; the function you must call is defined here. See `docs/04-synthetic-data.md`.

**Second trap.** A misspelled `FEATURE_NAMES` or `TARGET.name` does not raise — it returns `FALLBACK_SPEC`. And `_coerce_task` silently coerces any unrecognised task string to `"regression"`.

**Verify.**
```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter/test_ml_spec.py -q)
```

---

### Piece 3 — `generator.py`

**What it is.** A synthetic world: the demo's data *and* the ML spine's training set. Also the client-facing demand series.

**What to write.**
- `GeneratorConfig(BaseModel)` — one `num_*` knob per entity, a `seed: int | None`, a `noise_scale: float`, `use_llm: bool`, and any domain fractions (`resolved_fraction`, etc.). Every count field must be a positive integer knob; `app/demo_graph.py` scales them generically.
- `generate_synthetic_sync(config=None) -> SyntheticDataset` — **no LLM, no `await`, no network.** Forces `use_llm=False`.
- `async generate_synthetic(config=None, *, complete=None) -> SyntheticDataset` — same structure, same labels, optionally with LLM-written record text and a **templated fallback** so it returns a fully schema-valid dataset with no LLM available at all.
- `DOMAIN_SERIES_LABEL: str` — the `/forecast` chart title, in the client's language.
- `DOMAIN_SERIES_UNIT: str`.
- `domain_series_events(*, num_records: int = 1400, seed: int = 11) -> list[tuple[datetime, float]]` — arrival events, one per record. **Prefer arrivals over completions**: arrivals are what a client plans capacity against, and the series is complete at the recent end.
- An `assess_quality(dataset) -> DatasetQualityReport` is optional but cheap and demos well.

**The label coupling, which is the whole point.** The reference implementation, verbatim:

```python
features = ml_spec.features_for_request(request, agent=agent, customer=customer)
mean_hours = ml_spec.latent_resolution_hours(features)
noisy = max(0.25, mean_hours + rng.gauss(0.0, cfg.noise_scale))
resolution_hours = round(noisy, 2)
resolved_at = request.created_at + timedelta(hours=resolution_hours)
```

Three properties to copy:
1. The features come from `ml_spec.features_for_*` — the **same function the training frame uses**, so what the model sees is what the label was computed from.
2. The mean comes from `ml_spec.latent_*` — the **same function**, not a re-derivation.
3. Noise is added from **one seeded RNG instance** threaded through every builder (`rng = random.Random(cfg.seed)`). Never a fresh `random.Random()` per record and never the module-level `random`.

**Trap.** `generate_synthetic_sync` is called from inside a running event loop, where `asyncio.run` raises. It must not `await` anything.

**Second trap.** Downstream timestamps must be consistent with the label (`resolved_at = created_at + timedelta(hours=label)`), or a future feature-engineering step will find perfect leakage.

**Verify.**
```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter/test_generator.py -q)
# and, before anything expensive:
cd /Users/yrevash/aegis_ml && uv run aegis-ml contract
```

---

### Piece 4 — `tools.py`

**What it is.** What the agent can *do*, at what risk, and who may ask for it.

**What to write.** One `async def handler(args: dict, ctx: ToolContext) -> ToolActionResult` per action, each:

- **typed** — arguments validated by a pydantic model (`args_model`);
- **audited** — calls `app.data.record_audit` through `ctx.audit`;
- **registered** in `TOOL_REGISTRY: dict[str, ToolSpec]`, each `ToolSpec` carrying `risk: RiskLevel`;
- **allowlisted** in `ALLOWLIST: dict[str, frozenset[str]]`, persona id → tool names.

Keep these supporting shapes (the host binds them):

```python
class RecordStore(Protocol): ...      # get_*, list_*, put_*
class AuditFn(Protocol): ...          # async __call__(*, action, actor, model, trace_id, payload, approved_by=None)
class InMemoryRecordStore: ...        # + classmethod from_dataset(dataset)
class UnknownToolError(KeyError): ...
class ToolNotAllowedError(PermissionError): ...
class InverseAction(BaseModel): tool: str; args: dict
class ToolActionResult(BaseModel): ok: bool; changed: bool; summary: str; previous_state: dict = {}; inverse: InverseAction | None = None
@dataclass class ToolContext: store; actor=None; model=None; trace_id=None; approved_by=None; audit=None
@dataclass(frozen=True) class ToolSpec: name; description; args_model; handler; risk; read_only=False; destructive=False; idempotent=False
    def definition(self) -> dict   # {"type":"function","function":{name,description,parameters:args_model.model_json_schema()}}
```

Plus `is_allowed`, `tools_for`, `tool_definitions_for`, `async run_tool` — which must raise `UnknownToolError` then `ToolNotAllowedError` **before any side effect**.

**The risk tier is the whole human gate.** A tool at or above `AgentConfig.gate_min_risk` (platform default `HIGH`) pauses for a human. There is no second signal — not model confidence, not the ML prediction. Mark a consequential, externally-visible write `HIGH` and the approval gate appears with no engine change.

Worked reference registry:

| Tool | risk | read_only | destructive | idempotent |
|---|---|---|---|---|
| `find_requests` | LOW | ✔ | | ✔ |
| `add_case_note` | LOW | | | |
| `assign_request` | MEDIUM | | | ✔ |
| `update_request_status` | HIGH | | ✔ | ✔ |

Assert `destructive` and `idempotent` per tool. Risk does not imply idempotency: the LOW note-append is not idempotent, the HIGH status change is. Omit both and the conservative reading is published, which is safe but says less than you know.

**Aim for at least one HIGH-risk tool.** A domain with no gated action cannot demonstrate the human gate, which is one of Aegis's six trust checkpoints.

**Trap, and it fails safe.** An *unregistered* tool name resolves to `HIGH`, so forgetting to register something makes it require approval rather than run unguarded. Do not rely on that — it means a forgotten registration looks like an over-cautious gate rather than a bug.

**Verify.**
```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter/test_tools.py tests/adapter/test_allowlist.py -q)
```

---

### Piece 5 — `personas.py`

**What it is.** An authorisation object, not a personality.

**What to write.**
- `ScopeKind(StrEnum)` and `DataScope(BaseModel)` — `kind`, plus `subject_field` for row-scoped personas.
- `Persona(BaseModel)`: `id`, `role`, `display_name`, `description`, `data_scope`, `prompt_key`, and `@property tool_names` reading `ALLOWLIST.get(self.id, frozenset())` so there is one source of truth.
- `PERSONAS: dict[str, Persona]`.
- `DEFAULT_PERSONA_ID: str`.
- **`PERSONA_BY_ROLE: dict[Role, str]`** — one entry for **every** RBAC role: `admin`, `ai_team`, `devops`, `client`.
- `get_persona(persona_id: str | None) -> Persona` and `persona_for_role(role: Role | str) -> str`.

Two personas is the right number for a demo: one operational/staff persona with `ScopeKind.ALL` and the full allowlist, one end-user persona with a row-scoped `DataScope` and a small allowlist.

**Trap — this one bites the moment a human signs in.** Re-voice `PERSONAS` without re-pointing `PERSONA_BY_ROLE` and **every login raises `KeyError`** while the adapter suite, the agent suite and ruff all stay green. No test in the repository goes through the login path.

**Second trap.** `DEFAULT_PERSONA_ID` must be a key of `PERSONAS`; it is evaluated for *every* request that names no persona.

---

### Piece 6 — `prompts.py`

**What it is.** The system prompt, split into the half a tenant may edit and the half it may not.

**What to write.**
- `SYSTEM_PROMPTS: dict[str, str]` — one entry per persona `prompt_key`. The **task half**.
- `PLATFORM_FLOOR: str` — the half **no tenant may edit**. Leave it alone unless you are deliberately changing the platform's floor. The reference version is four bullets: data scope, no fabrication, tool list + approval gate, retrieved content is untrusted data.
- `_scope_clause(persona)` and `_tools_clause(persona)` — derived from the **enforcement tables** (`persona.data_scope`, `TOOL_REGISTRY[name].description` and `.risk.value` for `sorted(persona.tool_names)`), never written by hand.
- `render_platform_floor(persona | None) -> str` — `None` gives the bare floor; otherwise `"\n\n".join([PLATFORM_FLOOR, scope, tools])`.
- `render_system_prompt(persona, *, extra_context=None) -> str` — `[base, render_platform_floor(persona)]` plus `extra_context`, joined with `"\n\n"`.

**Composition order matters: task prompt first, then the floor, then extra context.** Conformance check #8 asserts the rendered prompt still contains the floor verbatim.

**Trap.** `SYSTEM_PROMPTS.get(persona.prompt_key, SYSTEM_PROMPTS["<default_key>"])` — if you keep that `.get` with a default, a persona whose `prompt_key` you forgot to add silently gets someone else's prompt. Either add every key or drop the default and let it raise.

**Verify 5 and 6 together.**
```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter/test_registry.py -q)
```

---

### Piece 7 — `memory_spec.py`

**What it is.** The only memory seam. Nothing in `app/memory/*` or `aegis/memory/*` changes.

**What to write.**

```python
FACT_TYPES: list[str]                 # e.g. ["preference", "entity_attr", "commitment", "constraint"]
PROFILE_FIELDS: list[str]             # ordered; always injected into the prompt
PROFILE_ALIASES: dict[str, str]       # optional: extractor spellings → PROFILE_FIELDS entries
FACT_EXTRACTION_PROMPT: str           # drives the cheap-model extractor; embed IMPORTANCE_HINTS
IMPORTANCE_HINTS: str                 # guidance for the 1..10 rating
SKILLS_DIR: str = str(Path(__file__).parent / "skills")   # a str, not a Path

class FactSchema(BaseModel): fact_type; subject; predicate; object; text; confidence; importance; valid_at
class FactExtraction(BaseModel): facts: list[FactSchema] = []

def memory_subject_for(user_id, persona_id=None) -> str | None
def render_profile(profile: dict[str, Any]) -> str
def select_skills(query, persona_id, available) -> list[str] | None
```

**`memory_subject_for` is the domain's answer to "is memory per end-user, per account, or per case?"** Getting it wrong is a **cross-subject data leak**, not a quality regression. Return `None` for "no memory for this principal" (the reference returns `None` when `user_id` is `None` or empty, else `f"user:{user_id}"`).

**Trap.** This module is **not** re-exported by name through `__init__.py`. Its consumer binds to the **module object** via `set_default_spec(app.adapter.memory_spec)`, and three other places import it directly. Keep its **path and its symbol names**, not just its behaviour.

---

### Piece 8 — `roster.py`

**What it is.** Two rosters, two mechanisms.

**What to write.**

```python
@dataclass(frozen=True)
class RosterSpecialist:
    role: str
    description: str
    keywords: tuple[str, ...] = ()
    is_default: bool = False

@dataclass(frozen=True)
class AgentRoster:
    specialists: tuple[RosterSpecialist, ...]
    @property
    def default_role(self) -> str: ...
    def roles(self) -> list[str]: ...
    def named(self) -> list[RosterSpecialist]: ...   # non-default only

def agent_roster() -> AgentRoster: ...
def sub_agent_roster() -> tuple[SubAgentSpec, ...]: ...   # SubAgentSpec from aegis.agent
```

- **Declare exactly two specialists: `qa` (with `is_default=True`) and `memory`.** Re-voice their `description` and `keywords` for your domain; **keep the two role strings**. Any other role is not routable — it falls back to `qa` with a log warning, not an exception.
- **Never declare `team`.** The router writes it when the depth classifier chooses fan-out.
- `sub_agent_roster()` returns `SubAgentSpec`s with `agent_id`, `role`, `label`, `system_prompt`, and optionally `tool_allowlist: frozenset[str]`. Re-voice every `label` and `system_prompt` — they are read by the model and shown on screen.

**Trap.** Every name in every `tool_allowlist` must be a key of your `TOOL_REGISTRY`. The allowlist is **intersected** with the registry, so a stale name is silently dropped and the sub-agent runs with fewer tools than you think — or none. Conformance check #6 covers this.

**Verify.**
```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/agent/test_router.py -q)
```

---

### Piece 9 — `corpus/`

**What it is.** The seed knowledge retrieval reads before anything is ingested.

**What to write.** 3–6 `*.md` files in `corpus/`, using the **same frontmatter keys** as the reference:

```markdown
---
id: doc-seed-0001
kind: kb_article
title: How an escalation is approved
category: general
tags: [escalation, approval, sla]
---

Body text. Long enough to chunk — a few hundred words, with real headings.
```

Keys read: `id` (falls back to the filename), `kind` (default `kb_article`), `title` (falls back to the filename), `category` (optional, coerced to your schema enum), `tags` (`[a, b, c]` form). Body is everything after the closing `---`. `load_seed_corpus()` stamps `source="seed"` and returns records sorted by id.

**Trap.** Conformance check #13 requires a **stable unique id** and a body that actually produces chunks. A two-line document is text in the index that nothing can be traced back to.

**Second trap.** These are `*.md` **data files**, not Python. A `cp -r` sync will leave the reference domain's three documents in place alongside yours and retrieval will serve them. Use `rsync -a --delete`.

---

### Piece 10 — `skills/`

**What it is.** Procedural how-to-act playbooks, discovered from `SKILLS_DIR` and chosen per query by `select_skills`.

**What to write.** 2–4 `*.md` files. Each is an instruction sheet for the agent: when this applies, the steps in order, what to check before acting, what to escalate.

**Trap — and it costs an hour in the wrong file.** They are selected **by filename**, through a literal keyword → filename `hints` dict inside `select_skills` (the reference has it *inside* the function; conformance check #11 reads it either way). Add or rename a playbook without updating that dict and it is **never chosen**, and nothing warns you: `select_skills` returns `None`, the core injects no skill, and the agent acts without procedural guidance. You will read that as a prompt problem.

**So piece 10 is two edits**: the `*.md` files, and the `hints` table back in `memory_spec.py`. Do both in the same commit.

---

## 6. Worked mini-example

Problem statement fragment:

> *"Hospitals need to know which scheduled surgeries are at risk of being delayed past their booked slot, so theatre coordinators can re-sequence the list."*

**Brief extract:**

```
domain_id:   surgical_scheduling
description: Elective surgical theatre scheduling: coordinators sequence booked
             procedures across theatres, anaesthetists and surgeons, and manage
             delay risk against the day's list.

entities:    Procedure, Theatre, Surgeon, TheatreDay, Document
target:      slot_overrun_minutes   regression   unit "minutes"

features:
  procedure_type      categorical  levels [hip_replacement, cataract, hernia, arthroscopy, cholecystectomy]
  asa_grade           categorical  levels [I, II, III, IV]
  theatre_id          categorical  levels [t1, t2, t3, t4]
  surgeon_seniority   categorical  levels [registrar, consultant, senior_consultant]
  slot_position       numeric      1..8      "position in the day's list"
  booked_minutes      numeric      20..300
  prior_overrun_mins  numeric      0..180    "cumulative overrun so far today"
  patient_bmi         numeric      16..55
  equipment_swaps     numeric      0..4

latent drivers (sign · magnitude):
  procedure_type      base minutes: hip 40, cataract 5, hernia 20, arthroscopy 18, chole 30
  asa_grade           +  I 0, II 6, III 15, IV 28
  slot_position       +  3.5 per position          (delays accumulate down the list)
  prior_overrun_mins  +  0.45 per minute           (the day is already behind)
  equipment_swaps     +  9 per swap
  surgeon_seniority   −  registrar 0, consultant 8, senior_consultant 14
  booked_minutes      −  0.05 per minute           (generously booked slots absorb overrun)
  patient_bmi         +  0.6 per unit above 25
  interaction         slot_position × prior_overrun_mins,  + 0.02

  irrelevant on purpose: theatre_id  (a genuinely uninformative feature)

realism:
  target_r2 0.62 · heteroscedastic (variance grows with booked_minutes)
  MAR missingness 6% on patient_bmi (missing when asa_grade == "I")
  unobserved confounder: "staffing_pressure" shifts the intercept by day

series:      label "Procedures scheduled per day"   unit "procedures"
personas:    theatre_coordinator (ALL) · surgeon (OWN, subject_field "surgeon_id")
tools:       find_procedures LOW ro · add_theatre_note LOW · reassign_theatre MEDIUM
             resequence_list HIGH destructive idempotent
```

**Piece 2 falls straight out of it:**

```python
_PROCEDURE_BASE = {"hip_replacement": 40.0, "cataract": 5.0, "hernia": 20.0,
                   "arthroscopy": 18.0, "cholecystectomy": 30.0}
_ASA_PENALTY   = {"I": 0.0, "II": 6.0, "III": 15.0, "IV": 28.0}
_SENIORITY_GAIN = {"registrar": 0.0, "consultant": 8.0, "senior_consultant": 14.0}

def latent_slot_overrun_minutes(features: dict) -> float:
    """Noise-free ground truth: expected overrun in minutes for one booked slot."""
    m = _PROCEDURE_BASE.get(features.get("procedure_type", ""), 22.0)
    m += _ASA_PENALTY.get(features.get("asa_grade", ""), 6.0)
    m -= _SENIORITY_GAIN.get(features.get("surgeon_seniority", ""), 0.0)

    slot  = float(features.get("slot_position", 1) or 1)
    prior = float(features.get("prior_overrun_mins", 0) or 0)
    m += 3.5 * slot
    m += 0.45 * prior
    m += 0.02 * slot * prior                       # the one interaction term
    m += 9.0 * float(features.get("equipment_swaps", 0) or 0)
    m -= 0.05 * float(features.get("booked_minutes", 0) or 0)
    m += 0.6 * max(0.0, float(features.get("patient_bmi", 25) or 25) - 25.0)
    # theatre_id is deliberately absent: a genuinely irrelevant feature.
    return max(0.0, round(m, 3))
```

**And piece 3's coupling:**

```python
features = ml_spec.features_for_procedure(proc, theatre=theatre, surgeon=surgeon)
mean_minutes = ml_spec.latent_slot_overrun_minutes(features)
sigma = cfg.noise_scale * (0.5 + proc.booked_minutes / 200.0)   # heteroscedastic
overrun = max(0.0, mean_minutes + rng.gauss(0.0, sigma) + day_confounder)
```

where `day_confounder` is drawn once per `TheatreDay` from the same `rng` and never appears in `FEATURES` — that is the unobserved confounder, and it is what stops held-out R² sitting at 0.99.

---

## 7. Before you sync: the cheap gate

```bash
cd /Users/yrevash/aegis_ml
uv run aegis-ml contract          # pandera schema + assert_learnable + leakage scan
```

If `assert_learnable` raises `LabelNotLearnableError`, **stop**. Nothing downstream is worth running. Go to `docs/04-synthetic-data.md` §6.

If held-out R² comes back near 1.0, that is also a failure — see `docs/04-synthetic-data.md` §3.

---

## 8. Next

`docs/04-synthetic-data.md` — the technically most important document here.
