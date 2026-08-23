# 02 · The domain-adapter contract

**Source of truth: `/Users/yrevash/aegis/aegis/src/aegis/adapter.py` (609 lines).** This document reproduces it. When the two disagree, the source is right — go read it.

This is the reference you return to constantly. Keep it open.

---

## 1. The shape of the thing

Retargeting Aegis means writing **one object** — in practice a Python *package* — that structurally satisfies `aegis.adapter.DomainAdapter`.

- **Ten pieces**: eight Python modules plus two content directories.
- **Eleven members**: nine of the pieces are members, plus `DOMAIN_ID` and `DOMAIN_DESCRIPTION`.
- `skills/` is the one piece with **no member of its own**, deliberately: it is a directory of Markdown playbooks discovered at call time and it is already named by `memory_spec.SKILLS_DIR`.
- `__init__.py` is **not** one of the ten. It is the *registry* — the interface the core imports.

No inheritance. No registration. No base class. A package satisfies it structurally:

```python
import myapp.adapter
from aegis.adapter import DomainAdapter, missing_members

assert not missing_members(myapp.adapter)        # every member present
assert isinstance(myapp.adapter, DomainAdapter)  # runtime_checkable
def wire(adapter: DomainAdapter) -> None: ...    # mypy checks every signature
```

> **Members are attributes of the package, so they must be imported.** A submodule becomes an attribute of its parent package only once *something* imports it. An adapter whose `__init__.py` never touches `memory_spec` does **not** have a `memory_spec` member, however present the file is on disk. This bit the reference adapter itself — `missing_members(app.adapter)` returned `['memory_spec']` with the file sitting on disk and named in the manifest. That is conformance check #1's scar.

---

## 2. The piece table

| Piece | File / dir | Member | What the platform reaches through it |
|---|---|---|---|
| 1 | `schema.py` | `schema` | The domain's record types and their version. |
| 2 | `ml_spec.py` | `ml_spec` | Features, target and the training frame for the ML spine. |
| 3 | `generator.py` | `generator` | Synthetic-world generation (demo + ML training data) **and** the client-facing demand series. |
| 4 | `tools.py` | `tools` | The action-tool registry, its risk tiers and the allowlist. |
| 5 | `personas.py` | `personas` | Who is asking, and the data each may see. |
| 6 | `prompts.py` | `prompts` | The persona system prompt, and the platform floor. |
| 7 | `memory_spec.py` | `memory_spec` | What counts as a durable fact — and `SKILLS_DIR`. |
| 8 | `roster.py` | `roster` | The specialists routed between, and the fan-out team. |
| 9 | `corpus/` | `corpus` | The seed corpus loader (a data directory + `__init__.py`). |
| 10 | `skills/` | *(none)* | The directory named by `memory_spec.SKILLS_DIR`. |

Plus the two identity members on the package itself:

```python
DOMAIN_ID: str            # stable machine id of the loaded domain
DOMAIN_DESCRIPTION: str   # one paragraph — AND the guardrails' allowed_topics
```

> `DOMAIN_DESCRIPTION` is **not metadata**. `backend/src/app/guardrails/__init__.py` imports it and wires it straight in as `allowed_topics`. It is a **control input**. A vague description is a loose rail; an absent one is no rail at all. Conformance check #2 asserts it is substantive.

---

## 3. Every sub-Protocol, in full

Signatures below are copied from `aegis/src/aegis/adapter.py`. `...` in a default position means the Protocol declares a default but does not fix its value.

### 3.1 Piece 1 — `SchemaModule`

```python
SCHEMA_VERSION: str
```

That is the entire required surface. The platform passes domain records around **opaquely** — it never introspects a field. `SCHEMA_VERSION` is written onto generated datasets so a corpus or a trained model can be told apart from one produced by a different shape of the same domain.

Reference implementation: `SCHEMA_VERSION = "1.0.0"`, seven `StrEnum`s, six pydantic `BaseModel`s, and a `SyntheticDataset` container.

### 3.2 Piece 2 — `MLSpecModule`

```python
FEATURES: list[Any]
FEATURE_NAMES: list[str]
TARGET: Any

def training_frame(self, *, num_records: int = ..., seed: int = ...) -> pd.DataFrame: ...
def describe_prediction(self, resp: Any, *, top_k: int = 3) -> str: ...
```

- `FEATURES` — ordered feature specs; each carries `.name` and `.dtype`. The categorical subset is derived from `.dtype == "categorical"` when `CATEGORICAL_FEATURES` is absent.
- `FEATURE_NAMES` — `[f.name for f in FEATURES]`.
- `TARGET` — `.name` is the predicted column, `.task` is `"regression"` | `"classification"`. `.unit` is read by `python -m app.ml` and by `describe_prediction`.
- `training_frame` — one column per `FEATURE_NAMES`, plus the `TARGET.name` column. **The keyword is `num_records`**, deliberately domain-neutral, because the core Protocol names it and a Protocol spelling it `num_requests` would force every future domain to call its rows "requests".
- `describe_prediction` — renders one prediction as the domain's own decision-support sentence.

**`CATEGORICAL_FEATURES` and `NUMERIC_FEATURES` are not Protocol members but you should declare them.** `resolve_spec` prefers an explicit `CATEGORICAL_FEATURES` and only derives it from `FEATURES[].dtype` when absent.

### 3.3 Piece 3 — `GeneratorModule`

```python
DOMAIN_SERIES_LABEL: str
DOMAIN_SERIES_UNIT: str

def domain_series_events(self, *, num_records: int = ..., seed: int = ...) -> Sequence[tuple[Any, float]]: ...
def generate_synthetic_sync(self, config: Any | None = None) -> Any: ...
async def generate_synthetic(self, config: Any | None = None) -> Any: ...
```

Both entry points must produce the **same structure and the same labels**. The sync one exists because it is called from inside a running event loop (where `asyncio.run` raises) *and* from offline training, and it must therefore need **no model call**.

`DOMAIN_SERIES_LABEL` is a **sentence a client reads** — it is the `/forecast` chart title. `domain_series_events` returns `(timestamp, value)` arrival events. Prefer arrivals over completions: arrivals are what a client plans capacity against, and the series is complete at the recent end.

Reference values: `DOMAIN_SERIES_LABEL = "Service requests opened per day"`, `DOMAIN_SERIES_UNIT = "requests"`, `domain_series_events(*, num_records: int = 1400, seed: int = 11)`.

### 3.4 Piece 4 — `ToolSpecLike` and `ToolsModule`

```python
class ToolSpecLike(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def risk(self) -> RiskLevel: ...
    def definition(self) -> dict[str, Any]: ...
```

> **`risk` is the load-bearing field.** It is the **only** signal deciding whether a proposed action routes to the human approval gate (compared against `AgentConfig.gate_min_risk`, platform default `HIGH`). A tool registered without one is not a mild annotation gap — **it is an ungated action.**

Two optional booleans are read when present and are deliberately *not* Protocol members, because a domain may legitimately declare neither: **`destructive`** (the call overwrites state a reader would miss) and **`idempotent`** (repeating the identical call converges). These are the MCP hints a client acts on. Risk does not imply idempotency: a note-append is LOW risk and *not* idempotent, while a gated status change is HIGH risk and *is*. Omit both and the conservative reading is published — safe, but it says less than you know.

```python
class ToolsModule(Protocol):
    @property
    def TOOL_REGISTRY(self) -> Mapping[str, ToolSpecLike]: ...
    @property
    def ALLOWLIST(self) -> Mapping[str, frozenset[str]]: ...

    def is_allowed(self, persona_id: str, tool_name: str) -> bool: ...
    def tools_for(self, persona_id: str) -> Sequence[ToolSpecLike]: ...
    def tool_definitions_for(self, persona_id: str) -> list[dict[str, Any]]: ...
    async def run_tool(self, persona_id: str, tool_name: str, args: dict[str, Any], ctx: Any) -> Any: ...
```

`TOOL_REGISTRY` is the whole vocabulary: **a name absent from it is treated as `HIGH` risk by the platform**, so a hallucinated tool can never slip under the gate. `ALLOWLIST` is checked *before* any side effect. `run_tool` returns an object with `ok` / `summary` (structural `ToolOutcome`).

### 3.5 Piece 5 — `PersonasModule`

```python
DEFAULT_PERSONA_ID: str

@property
def PERSONAS(self) -> Mapping[str, Any]: ...
@property
def PERSONA_BY_ROLE(self) -> Mapping[Any, str]: ...

def get_persona(self, persona_id: str | None) -> Any: ...
def persona_for_role(self, role: Any) -> str: ...
```

A persona is **not a UI label**: its `data_scope` becomes a retrieval filter and its entry in `ALLOWLIST` becomes the tool set. Each persona object carries `.id`, `.data_scope`, `.prompt_key`.

`PERSONA_BY_ROLE` maps coarse RBAC roles (`aegis.governance.types.Role`, or its string value) → persona id. **Every role must appear; every value must be a key of `PERSONAS`.** Both are checked by conformance.

> **This is the one that bites the moment a human signs in.** Every authenticated principal resolves through `persona_for_role(role)`. Re-voicing `PERSONAS` without re-pointing `PERSONA_BY_ROLE` makes **every login raise `KeyError`** while the adapter suite, the agent suite and ruff all stay green — none of them go through the login path. The host used to decide this itself with two persona ids hardcoded in an `if` in `app/api/routes.py`; that is exactly how the failure was found.

Roles to cover: `admin`, `ai_team`, `devops`, `client`.

### 3.6 Piece 6 — `PromptsModule`

```python
def render_system_prompt(self, persona: Any, *, extra_context: str | None = None) -> str: ...
def render_platform_floor(self, persona: Any | None) -> str: ...
```

`render_system_prompt` is the **task half**. `render_platform_floor` is the **boundary half** — the preamble plus the persona's *live* data scope and tool allowlist, derived from the enforcement tables rather than written by hand. An LLM-Ops prompt version replaces the first and is composed **over** the second, never instead of it.

Reference composition order: `base_task_prompt` → `render_platform_floor(persona)` → `extra_context`, joined with `"\n\n"`.

### 3.7 Piece 7 — `MemorySpecModule`

```python
FACT_TYPES: list[str]
PROFILE_FIELDS: list[str]
FACT_EXTRACTION_PROMPT: str
IMPORTANCE_HINTS: str
SKILLS_DIR: str
FactSchema: type[Any]
FactExtraction: type[Any]

def memory_subject_for(self, user_id: str | int | None, persona_id: str | None = None) -> str | None: ...
def render_profile(self, profile: dict[str, Any]) -> str: ...
def select_skills(self, query: str, persona_id: str | None, available: list[str]) -> list[str] | None: ...
```

Structurally this is `aegis.memory.spec.MemorySpec` — the module a host installs with `aegis.memory.set_default_spec(...)` — plus `memory_subject_for`, which decides **whose** memory a turn reads and writes. Getting that wrong is a **cross-subject data leak**, not a quality regression.

`PROFILE_ALIASES` (optional): predicate spellings your extractor emits mapped onto `PROFILE_FIELDS` entries. Absent means "no aliases", which is a legitimate statement. This table used to live in `aegis/memory/consolidate.py` naming the shipped domain's fields, where it quietly matched nothing after a retarget.

> **Trap.** `memory_spec` is deliberately **not** re-exported by name through `adapter/__init__.py`. Its consumer binds to the **module object**: `backend/src/app/memory/__init__.py` calls `set_default_spec(app.adapter.memory_spec)`. So the module must keep its **path and its symbol names**, not just its behaviour.

### 3.8 Piece 8 — `AgentRosterLike` and `RosterModule`

```python
class AgentRosterLike(Protocol):
    @property
    def specialists(self) -> Sequence[Any]: ...
    @property
    def default_role(self) -> str: ...
    def roles(self) -> list[str]: ...
    def named(self) -> list[Any]: ...

class RosterModule(Protocol):
    def agent_roster(self) -> AgentRosterLike: ...
    def sub_agent_roster(self) -> Sequence[Any]: ...
```

**Two rosters, two mechanisms, not interchangeable.**

- `agent_roster()` is the **supervisor's hand-off set**. Every role it names must have a handler node in the graph.
- `sub_agent_roster()` is the **fan-out team**. It is deliberately empty-able; an absent one means every turn runs single-lane rather than fanning out to agents the domain never declared.

`default_role` is load-bearing: it is where a turn goes when no specialist matched. `AgentRoster.default_role` returns the **first specialist in declaration order** when none is marked `is_default=True` — so forgetting the flag silently promotes whichever specialist happens to be written first.

Each `SubAgentSpec` carries a `tool_allowlist` of **literal tool names**. The allowlist is *intersected* with `TOOL_REGISTRY`, so a stale name is silently dropped and the sub-agent runs with fewer tools than you think — or none.

### 3.9 Piece 9 — `CorpusModule`

```python
def load_seed_corpus(self) -> list[Any]: ...
```

A domain with **no** corpus is legal and honest — the loader may return an empty list, and retrieval then returns no candidates instead of reaching for someone else's documents. What is not legal is *absent*: a missing loader is the difference between "this domain ships no seed knowledge" and "the seed knowledge silently failed to load".

Reference implementation reads `*.md` files from the package directory with `---`-delimited frontmatter keys `id`, `kind`, `title`, `category`, `tags` (`[a, b, c]` form), body after the closing `---`, and stamps `source="seed"`.

### 3.10 Piece 10 — `skills/`

No member. A directory of Markdown playbooks named by `memory_spec.SKILLS_DIR`, discovered at call time, chosen per query by `memory_spec.select_skills`.

---

## 4. The two proofs

Both are needed, and they are complementary.

### Proof 1 — presence and shape

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import app.adapter
from aegis.adapter import DomainAdapter, missing_members
print('missing:', missing_members(app.adapter))
print('satisfies:', isinstance(app.adapter, DomainAdapter))
")
```

PowerShell:

```powershell
Push-Location C:\aegis\backend
$env:PYTHONPATH = "src;..\aegis\src"
.\.venv\Scripts\python.exe -c "import app.adapter; from aegis.adapter import DomainAdapter, missing_members; print('missing:', missing_members(app.adapter)); print('satisfies:', isinstance(app.adapter, DomainAdapter))"
Pop-Location
```

`missing: []` means every member **exists**. `isinstance` is `runtime_checkable`, which verifies existence and **does not look at a single signature**. A type checker is what catches the member that exists with the wrong shape. An adapter that passes only this check is exactly the "working-looking but wrong" state the Protocol exists to remove.

### Proof 2 — the conformance suite

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)
```

Fourteen checks. No database, no Redis, no key, no model call, nothing async. Well under a second. **Run it after every step, not only at the end** — it is the fastest signal in the repo that the adapter is *wired* and not merely present.

`--aegis-adapter` is required; its absence stops with one usage error naming the flag, not with fourteen skips. The environment variable `AEGIS_ADAPTER` is the alternative, and the command line wins over it:

```bash
AEGIS_ADAPTER=app.adapter pytest --pyargs aegis.conformance
```

---

## 5. The fourteen checks, by name

Source: `aegis/src/aegis/conformance/test_conformance.py`. Listed in **test-function order**, which is the ordinal this documentation uses.

> **Numbering caveat.** The file groups the fourteen tests under **eleven** section headers, and `AGENTS.md` refers to the vocabulary check as "check 11" (its section number) while it is the **14th** test function. When you report a failure, quote the **test name**, never the number.

| # | Test name | Group | What it catches |
|---|---|---|---|
| 1 | `test_every_contract_member_is_present` | contract | A piece on disk that nothing imported, so it is not an attribute of the package. |
| 2 | `test_domain_identity_is_a_usable_topical_rail` | identity | `DOMAIN_ID` / `DOMAIN_DESCRIPTION` absent or too thin to be an `allowed_topics` rail. |
| 3 | `test_every_roster_role_has_a_handler_node` | roster | A specialist role that is not a key of `SPECIALIST_NODES` — it falls back to `qa` **with a log warning, not an exception**, and the `routing` stream event still names it. *The highest-value check in the suite.* |
| 4 | `test_the_roster_default_role_is_declared_and_routable` | roster | `default_role` is not one of the roster's own specialists, or has no graph node — so *every unmatched turn* takes the fallback path. |
| 5 | `test_every_tool_declares_a_risk_tier` | tools | A tool with no valid `RiskLevel` — an ungated action. |
| 6 | `test_allowlists_name_registered_tools_and_known_personas` | tools | A typo in `ALLOWLIST` or in a sub-agent `tool_allowlist`. Every typo fails **open into silence**: a misspelled persona key gives that persona no tools; a misspelled tool name never appears in the model's `tools=` payload. |
| 7 | `test_every_persona_the_adapter_declares_resolves` | personas | `DEFAULT_PERSONA_ID` not in `PERSONAS` (every anonymous request 500s), or `PERSONA_BY_ROLE` missing a role / naming a dead persona (every login raises `KeyError`). |
| 8 | `test_the_system_prompt_never_drops_the_platform_floor` | prompts | A rendered system prompt that does not contain its platform floor verbatim. |
| 9 | `test_memory_spec_satisfies_the_memory_contract` | memory_spec | A member `aegis.memory.spec.MemorySpec` requires that is missing — which otherwise fails only the first time a conversation is consolidated, i.e. never in a demo. |
| 10 | `test_skills_directory_holds_at_least_one_playbook` | skills | `SKILLS_DIR` pointing nowhere: zero playbooks discovered, nothing reported. |
| 11 | `test_every_playbook_is_reachable_from_select_skills` | skills | A playbook the selector can never name, or a selector naming a playbook that is gone. Reads the selector's compiled string constants **and** its module's top-level constants, so it does not matter whether the keyword table sits inside the function or beside it. Falls back to a behavioural probe. |
| 12 | `test_ml_spec_resolves_to_the_domain_not_the_fallback` | ml_spec | **`resolve_spec` returning `FALLBACK_SPEC`** — four columns named `feature_0`…`feature_3` predicting `target`. The backstop for §7 below. |
| 13 | `test_seed_corpus_records_carry_identity_and_chunk` | corpus | A corpus record with no stable id, or a body no chunker can split — text in the index nothing can be traced back to. Only constrains the records that exist; an empty corpus passes. |
| 14 | `test_no_shipped_domain_vocabulary_survives_outside_the_adapter` | **the core** | Any module outside the adapter still naming the shipped domain. **This is the one check that reads the core, not your adapter.** |

### 5.1 What check #14 actually does

`aegis/src/aegis/conformance/_vocabulary.py` holds a frozen tuple `SHIPPED_VOCABULARY` and a `SHIPPED_DOMAIN_ID`. `core_files()` walks the package roots minus the adapter directory and minus `__pycache__`, `tests`, `node_modules`, `.venv`; `scan_for_terms()` does a plain **case-sensitive substring scan** over every line — so it sees the word in a docstring, a comment, a dict key and a string literal alike.

Two halves:

1. **Unconditional**: no core module may contain any listed term.
2. **When the loaded adapter's `DOMAIN_ID` equals `SHIPPED_DOMAIN_ID`**, every listed term must still be found *inside* the adapter — a stale entry that no longer means anything is a failure, not decoration.

`MIN_CORE_FILES = 20` is the anti-vacuity floor: a check that silently finds nothing must fail, not pass.

Selection rule for the list: *"Each one is specific enough that an innocent occurrence is not plausible. A generic word ('customer', 'client', 'request') is deliberately absent: this check must never be the reason somebody deletes a true sentence from a core docstring, or its first false positive is the last time anybody believes it."*

**The scan is Python-only.** `web/` is not covered. See §9.

---

## 6. **What the fourteen checks do NOT cover**

Read this twice.

> **There is NO conformance check for generator ↔ latent coupling.** Zero references to the generator exist in `test_conformance.py`. A target that is **pure noise** — drawn independently of the features — passes **all fourteen checks**, the whole backend suite, the agent suite and ruff.
>
> The only native signal in the entire platform is **`distinct=False` on the last line of `python -m app.ml`**, which you will read minutes before the demo.
>
> `aegis_ml.data.latent.assert_learnable` is what actually catches it, and it fails in **seconds**. It fits a fast model on `training_frame(num_records=1200)` and asserts held-out R² (or accuracy) clears a floor, raising `LabelNotLearnableError` with the measured number.
>
> **This is the single most expensive trap in the whole exercise.** See `docs/04-synthetic-data.md`.

Other gaps the fourteen do not cover: target leakage, class imbalance, realistic noise levels, drift, and whether `describe_prediction` says anything true. `aegis_ml` covers all of those.

---

## 7. The `FALLBACK_SPEC` trap, verbatim

`aegis/src/aegis/ml/spec.py`:

```python
FALLBACK_SPEC = ResolvedSpec(
    features=["feature_0", "feature_1", "feature_2", "feature_3"],
    target="target",
    task="regression",
)

def resolve_spec(spec: MLSpec | None = None) -> ResolvedSpec:
    if isinstance(spec, ResolvedSpec):
        return spec
    if spec is None:
        return FALLBACK_SPEC

    candidate: object = spec
    features = getattr(candidate, "FEATURE_NAMES", None) or getattr(candidate, "features", None)
    target_obj = getattr(candidate, "TARGET", None)
    target = getattr(target_obj, "name", None) or getattr(candidate, "target", None)
    if not features or not target:
        return FALLBACK_SPEC          # ← nothing raises
    ...
```

Also read leniently: `task` from `TARGET.task` (or lowercase `task`), the categorical subset from `CATEGORICAL_FEATURES` (or derived from `FEATURES[].dtype == "categorical"`), and `frame_provider` from a callable `training_frame`.

And note `_coerce_task`: anything not in `{"classification", "classify", "clf", "categorical", "binary"}` becomes `"regression"`. **A typo in your task string silently trains a regressor on class labels.**

**Prevention, not detection:** declare the problem once as an `aegis_ml.contracts.spec.MLProblem` and let `aegis_ml` *generate* `ml_spec.py`. The five names then cannot be misspelled. Check #12 is the backstop, not the plan.

---

## 8. The `SPECIALIST_NODES` constraint

`aegis/src/aegis/agent/graph.py`:

```python
SPECIALIST_NODES: dict[str, str] = {
    "qa":     "recall_memory",   # the full retrieve -> plan -> gate -> act pipeline
    "memory": "answer_memory",   # answers from long-term memory, skipping RAG/tools
    "team":   "plan_team",       # router-written fan-out; NOT a roster role
}
```

A roster role that is not a key here **is not routable**. It falls back to the `qa` pipeline and logs a warning — it does not raise.

| What you want | What to do |
|---|---|
| Domain-specific specialists | **Re-voice `qa` and `memory`**: change `description` and `keywords` freely, keep the two role strings. No core edit. **Do this.** |
| A genuinely new third specialist | Requires a handler node in `graph.py` plus a `SPECIALIST_NODES` entry. That is a **sanctioned but reported** core edit. Avoid it under time pressure. |
| `team` | **Never declare it in a roster.** The router writes it when the depth classifier chooses fan-out. |

Exactly one specialist must be `is_default=True`.

---

## 9. Host-bound symbols beyond the Protocol

**Verified by grep against `/Users/yrevash/aegis/backend/src/` on the day this was written.** These names are bound by host modules *outside* the adapter. Satisfying the Protocol is not enough — drop one of these and the host raises `ImportError` at startup.

### From `app.adapter` (the package top level)

```
DEFAULT_PERSONA_ID   DOMAIN_DESCRIPTION   DOMAIN_SERIES_LABEL   DOMAIN_SERIES_UNIT
GeneratorConfig      InMemoryRecordStore  TARGET                TOOL_REGISTRY
ToolContext          agent_roster         domain_series_events  generate_synthetic_sync
get_persona          is_allowed           load_seed_corpus      memory_spec
ml_spec              persona_for_role     render_platform_floor render_system_prompt
run_tool             sub_agent_roster     tool_definitions_for  training_frame
```

### From `app.adapter.tools` (imported as a submodule)

```
ALLOWLIST   AuditFn   RecordStore   TOOL_REGISTRY   ToolActionResult
ToolContext   ToolNotAllowedError   UnknownToolError   is_allowed
```

### From `app.adapter.roster`

```
sub_agent_roster
```

### From `app.adapter.memory_spec`

```
FACT_TYPES   memory_subject_for
```
(plus `FACT_EXTRACTION_PROMPT`, bound by `backend/tests/`.)

### Also in `__all__`, and to keep

`SyntheticDataset` — `SKILL.md` names it specifically as *"the container the generator returns and the ML spine reads"*. Also `Persona`, `ScopeKind`, `AgentRoster`, `RosterSpecialist`, `PLATFORM_FLOOR`, `SYSTEM_PROMPTS`, `feature_matrix`, `features_for_request`, `generate_synthetic`, `tools_for`, `describe_prediction`.

> **Verified difference from the older internal note:** `Persona` and `ScopeKind` are **not** bound by any host module outside the adapter — `ScopeKind` is imported only by `adapter/prompts.py` (adapter-internal) and `Persona` only appears in host code as prose. Keep them anyway: `Persona` is in `__all__` and `backend/tests/` reads them. `ToolActionResult`, `ToolNotAllowedError` and `UnknownToolError` **are** bound (by `app/mcp/server.py` and `backend/tests/`), as the note said, and `AuditFn` and `RecordStore` are bound too and were missing from that note.

**Regenerate this list yourself before you rely on it:**

```bash
cd /Users/yrevash/aegis/backend/src && python3 - <<'PY'
import re, pathlib, collections
syms = collections.defaultdict(set)
for p in pathlib.Path('.').rglob('*.py'):
    if 'app/adapter/' in str(p):
        continue
    txt = p.read_text(encoding='utf-8', errors='ignore')
    for pat in (r'from\s+(app\.adapter[\w\.]*)\s+import\s+\(([^)]*)\)',
                r'from\s+(app\.adapter[\w\.]*)\s+import\s+([^\(\n]+)'):
        for m in re.finditer(pat, txt):
            for n in re.split(r'[,\n]', m.group(2)):
                n = n.split('#')[0].strip().rstrip(',')
                if n:
                    syms[m.group(1)].add(n.split(' as ')[0].strip())
for mod in sorted(syms):
    print(mod, '->', sorted(syms[mod]))
PY
```

---

## 10. Rebuilding `__all__` is expected

`SKILL.md`: *"You **will** edit `__init__.py`, and its `__all__`, and that is correct."* Step 1 replaces the entity models it re-exports and step 2 renames the latent function it re-exports, so leaving `__all__` untouched is impossible.

What must stay stable is not the list of names — it is **the contract**:

1. the nine module members reachable as attributes, plus `DOMAIN_ID` and `DOMAIN_DESCRIPTION`;
2. the member names *inside* each piece that the sub-Protocols name;
3. `SyntheticDataset` as the container the generator returns and the ML spine reads;
4. the host-bound symbols in §9.

---

## 11. Files in the Aegis checkout you must NOT touch

Everything outside `backend/src/app/adapter/`, concretely: `app.agent`, `app.core`, `app.retrieval`, `app.memory`, `app.ml`, `app.guardrails`, `app.ops`, `app.eval`, `app.observability`, `app.data`, `app.mcp`, `app.api`, `app.platform`, `app.forecast`, `app.seed` — and all of `aegis/`.

**Two sanctioned exceptions**, both of which must be reported:

1. `aegis/src/aegis/conformance/_vocabulary.py` — **required**, see `docs/07-integration-with-aegis.md` §6.
2. `aegis/src/aegis/agent/graph.py` `SPECIALIST_NODES` — only if the domain truly needs a third specialist path. Avoid.

**In `backend/tests/adapter/`, rewrite everything except** `test_piece_manifest.py`, `test_domain_adapter_protocol.py`, `test_conformance_suite.py`, and the `broken_adapter/` fixture directory. Those check *structure*, not domain, and `broken_adapter/` is deliberately self-contained and imports nothing of yours.

---

## 12. Next

`docs/03-authoring-a-domain.md` — how to fill the ten pieces from a problem statement.
