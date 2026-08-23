# Day-of checklist — filling these templates and landing them in Aegis

Running order, per-step verify, and the four things outside `templates/adapter/` that a
retarget still has to touch. The authoritative procedure is
[`SKILL.md`](../../../aegis/SKILL.md); this is that procedure with the `aegis_ml` steps
folded in. Where they disagree, `SKILL.md` is right.

Every command is written from a repository root, and each `cd` is wrapped in a subshell
so a run of them one after another in one terminal works.

---

## 0 · Before you write anything

```bash
# Aegis: install, because a fresh checkout has no .venv at all.
(cd /Users/yrevash/aegis && ./scripts/bootstrap.sh)          # macOS / Linux
# .\scripts\install-windows.ps1                              # Windows

# aegis_ml: resolve and check the ML side.
(cd /Users/yrevash/aegis_ml && uv sync --extra dev && uv run aegis-ml doctor)

# The GREEN BASELINE. Run it before you change anything.
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src \
    .venv/bin/python -m pytest tests/adapter tests/agent -q)
```

**Write down the number you actually get.** That is your regression baseline. If it is
not green now, you are about to attribute a pre-existing failure to your own edit and
lose an hour.

Then: problem statement → `prompts/00-intake.md` → **Domain Brief**. Do not start
piece 1 without one. The Brief must name: entities and their enums; the ML target and
its unit; the features and their dtypes; the **latent drivers** and their direction;
the personas and their data scopes; the tools and their risk tiers; the roster; the
series label and unit.

---

## Read this before piece 1 — the whole suite goes red, and stays red

`backend/tests/conftest.py` imports through `app.adapter`. The moment you replace the
record types in piece 1, **every test in the repository fails at import** — hundreds of
lines of `ImportError` that say nothing about whether your edit was right. It stays that
way until piece 8 lands and the registry's imports all resolve again.

That is expected. It is not a signal. Do not chase it, and do not "fix" it by loosening
a conftest. The per-piece verifies below are written against one file at a time for
exactly this reason — and **they only mean something once you have rewritten the tests
they run** (see step 12).

The one check that is green from your first edit to your last, and needs no
infrastructure at all, is the conformance suite. Lean on it:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)
```

Fourteen checks, no database, no key, under a second. Run it after every step.

---

## The ten pieces, in order

The order is not arbitrary — each piece consumes the vocabulary the previous one
defined. Fill them in `templates/adapter/`, then sync (step 11) and verify.

### 1 · `schema.py` — the vocabulary
Entities as pydantic v2 models, categoricals as `StrEnum`, `SCHEMA_VERSION` set,
`SyntheticDataset` kept **by name**. The target field is nullable and populated only on
finished records.

> **Verify:** `pytest tests/adapter/test_schema.py -q`

### 2 · `ml_spec.py` — the predictable signal
`FEATURES` (order is the contract), `FEATURE_NAMES`, `CATEGORICAL_FEATURES`,
`NUMERIC_FEATURES`, `TARGET`, the renamed `latent_*` function, `features_for_*`,
`feature_matrix`, `training_frame(*, num_records, seed)` — **the keyword is
`num_records`**, named by the core Protocol — and a re-voiced `describe_prediction`.
Keep the module pure-Python at import time; pandas is imported inside `training_frame`.

Set `TARGET_R2` in the **0.45–0.80** band and leave `noise_scale=None` so sigma is
derived, not guessed.

> **Verify:** `pytest tests/adapter/test_ml_spec.py -q`
> plus `pytest --pyargs aegis.conformance --aegis-adapter app.adapter -q -k ml_spec`

### 3 · `generator.py` — data before there is data
Seeded structure + LLM prose + templated fallback. **The label is drawn around piece 2's
latent function and nothing else.** `generate_synthetic_sync(config)` must return
schema-valid records with no LLM available at all. Re-voice `DOMAIN_SERIES_LABEL` (a
sentence a jury reads), `DOMAIN_SERIES_UNIT`, and point `domain_series_events` at your
own record collection — arrivals, not completions.

> **Verify:** `pytest tests/adapter/test_generator.py -q`
> then `aegis-ml contract` — the held-out-R² floor, in seconds, before anything expensive.

### 4 · `tools.py` — the real actions
One typed, audited, idempotent, reversible handler per action; each registered in
`TOOL_REGISTRY` with an honest `risk`, and allowlisted per persona in `ALLOWLIST`.
**Include a read-only lookup tool** or the planner can never legitimately obtain an id.
Assert `destructive` / `idempotent` per tool. Decide the ML SLOT now or note it for
step 15.

> **Verify:** `pytest tests/adapter/test_tools.py tests/adapter/test_allowlist.py -q`

### 5 · `personas.py` — who is served
`PERSONAS`, `DEFAULT_PERSONA_ID` (must be a key of `PERSONAS`), **`PERSONA_BY_ROLE`
covering every RBAC role**, `get_persona`, `persona_for_role`. Every persona id used as
an `ALLOWLIST` key must exist here.

### 6 · `prompts.py` — who the agent is
One base prompt per persona's `prompt_key`. Never hand-write a tool list — `_tools_clause`
derives it. **Leave `PLATFORM_FLOOR` alone**; check #8 asserts it survives composition.

> **Verify 5 and 6 together:** `pytest tests/adapter/test_registry.py -q`

### 7 · `memory_spec.py` — what is worth remembering
`FACT_TYPES`, `PROFILE_FIELDS`, `PROFILE_ALIASES`, `FACT_EXTRACTION_PROMPT`,
`IMPORTANCE_HINTS`, `memory_subject_for` (**decide the scope deliberately — the wrong
one is a cross-subject leak**), `render_profile`, `select_skills` + `SKILL_HINTS`,
`SKILLS_DIR`. Keep the module path and every symbol name: consumers bind to the module
object.

### 8 · `roster.py` — who the supervisor may route to
Re-voice `qa` and `memory`; **keep those two role strings**. Exactly one `is_default`.
Never declare `team`. Re-point every `sub_agent_roster()` `tool_allowlist` at names that
exist in your `TOOL_REGISTRY` — stale names are dropped in silence.

Adding a genuinely new specialist needs a core edit (`SPECIALIST_NODES` +
a handler node in `aegis/src/aegis/agent/graph.py`). That is sanctioned, and it **must
be reported**.

> **Verify:** `pytest --pyargs aegis.conformance --aegis-adapter app.adapter -q -k roster`
> then `pytest tests/agent/test_router.py -q`

### 9 · `corpus/` — the seed knowledge
Replace `EXAMPLE.md` with your own `*.md`, same frontmatter keys. Unique `id`, chunkable
`body`. **Delete `EXAMPLE.md`.**

### 10 · `skills/` — how to act
Replace `EXAMPLE.md` with real playbooks — and **edit `SKILL_HINTS` in the same commit**.
A playbook no literal can name is never selected and nothing warns. **Delete
`EXAMPLE.md` and its three hint rows.**

---

## 11 · Sync — `rsync -a --delete`, never `cp -r`

```bash
rsync -a --delete \
    /Users/yrevash/aegis_ml/templates/adapter/ \
    /Users/yrevash/aegis/backend/src/app/adapter/
```

A plain `cp -r` leaves the previous domain's `corpus/*.md` and `skills/*.md` behind.
`load_seed_corpus()` will ingest them without complaint and retrieval will serve the old
domain's policies under your domain's name, cited and confident.

Sanity-check what landed:

```bash
ls /Users/yrevash/aegis/backend/src/app/adapter/corpus/ \
   /Users/yrevash/aegis/backend/src/app/adapter/skills/
grep -rn "TODO(domain)" /Users/yrevash/aegis/backend/src/app/adapter/ | head
```

Any `TODO(domain)` still there is a blank you did not fill.

---

## 12 · Rewrite `backend/tests/adapter/*`

Not domain-neutral scaffolding: those files carry between 3 and 26 shipped-domain
literals each. Rewriting them is **part of each piece, not a follow-up** — a per-piece
verify against an un-rewritten test is meaningless.

**Leave these four alone.** They check structure, not domain:

- `test_piece_manifest.py`
- `test_domain_adapter_protocol.py`
- `test_conformance_suite.py`
- `broken_adapter/` (deliberately self-contained; imports nothing of yours)

---

## 13 · The one sanctioned core edit: `_vocabulary.py`

```
aegis/src/aegis/conformance/_vocabulary.py
```

Conformance check #14 scans every module **outside** the adapter for the shipped
domain's words and fails naming the file and the line. After a retarget, `SHIPPED_
VOCABULARY` must list *your* domain's distinctive terms and `SHIPPED_DOMAIN_ID` must be
your `DOMAIN_ID` — otherwise the check is quarantining a vocabulary nobody uses.

Selection rule: each term must be specific enough that an innocent occurrence is not
plausible. A generic word ("customer", "request", "item") is exactly what makes a check
nobody believes.

**This edit is sanctioned by `SKILL.md` and must be named in your report.** So must any
`SPECIALIST_NODES` edit from step 8.

---

## 14 · The console — four files the Python scan cannot see

`web/` is outside the adapter and outside the conformance check, which scans Python.
These four carry shipped-domain literals and **will show the old domain's words on
screen after an otherwise perfect retarget**:

| File | What it names |
|---|---|
| `web/src/config/personas.ts` | the persona ids, and the tool names in its prose |
| `web/src/components/ops/opsShared.ts` | `PROMPT_KEY`, and tool names in two prompt strings |
| `web/src/components/sim/SimulationView.tsx` | the persona id the scripted demo runs as |
| `web/src/components/ml/MLOpsView.tsx` | a literal ML feature row |

Also check `web/src/components/console/memorySubject.ts` if you changed
`memory_subject_for`'s scope — its own docstring says it must change when that does.

Re-voice them by hand and say in your report that you did.

---

## 15 · OpenAPI + TS client regeneration

Any change to a request/response shape the console reads (the persona list, the tool
catalogue, an ML route) needs the schema and the generated client regenerated, or the
console type-checks against yesterday's API:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src \
    .venv/bin/python -m app.api.export_openapi)     # emit the schema
(cd /Users/yrevash/aegis/web && npm run generate:client && npm run typecheck)
```

Check the repo's own script names before running these — if the commands differ, the
repo is right and this line is stale.

---

## 16 · ML: train, gate, promote

```bash
# The Aegis spine on your new spec. READ THE LAST LINE.
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src \
    .venv/bin/python -m app.ml | tail -1)
#   distinct=False  ⇒  the spine learned nothing. Go back to piece 3's trap,
#                      not to the prompt.

# The AutoML search, in the isolated trainer venv, returning a portable recipe.
(cd /Users/yrevash/aegis_ml && uv run aegis-ml train --tier all)
(cd /Users/yrevash/aegis_ml && uv run aegis-ml promote)   # the gate: refuses loudly
(cd /Users/yrevash/aegis_ml && uv run aegis-ml drift)
```

Then wire the ML tools: uncomment the SLOT in `tools.py`, grant the tool names in
`ALLOWLIST`, and grant them in the relevant `sub_agent_roster()` `tool_allowlist`.
Skipping any one of the three is silent.

---

## 17 · The final gate

```bash
# 0. Structural — seconds, and it catches a whole piece you forgot.
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import app.adapter
from aegis.adapter import DomainAdapter, missing_members
assert not missing_members(app.adapter), missing_members(app.adapter)
assert isinstance(app.adapter, DomainAdapter)
print('adapter contract: satisfied')
")

# 1. Conformance — fourteen checks, no infrastructure, under a second.
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)

# 2. Adapter + the whole agent graph, on fakes. The first command that can pass
#    again once piece 8 lands AND tests/adapter/* has been rewritten.
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    tests/adapter tests/agent -q)

# 3. The full backend suite.
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest -q)

# 4. The core package, untouched by your edits — as green as it was before you started.
(cd /Users/yrevash/aegis/aegis && PYTHONPATH=src ../backend/.venv/bin/python -m pytest -q)

# 5. Lint, both trees.
/Users/yrevash/aegis/backend/.venv/bin/python -m ruff check \
    /Users/yrevash/aegis/aegis /Users/yrevash/aegis/backend
(cd /Users/yrevash/aegis_ml && uv run ruff check src templates tests)

# 6. The label is learnable, measured rather than asserted.
(cd /Users/yrevash/aegis_ml && uv run pytest tests/test_label_is_learnable.py -q)
```

`tests/adapter/test_piece_manifest.py` is the tripwire for the structure itself: add a
ninth module and it fails until `SKILL.md`, `adapter/README.md` and every `piece N of M`
docstring are updated together. That is intentional — a piece missing from the checklist
is a piece nobody swaps.

---

## 18 · What to report

- New dependencies (**do not add them to `aegis/pyproject.toml` yourself** — name them).
- New environment variables.
- **Every core edit**, with the reason: the `_vocabulary.py` edit from step 13, and any
  `SPECIALIST_NODES` edit from step 8.
- That you re-voiced the four (or five) `web/` console files.
- The output of all six final-gate commands.
- The last line of `python -m app.ml` (`distinct=True`), and the promotion gate's
  decision with its numbers — measured coverage against requested, the primary metric
  against the champion's, and the worst slice.

---

## The traps, on one page

| Piece | Silent failure | What you would see instead |
|---|---|---|
| 1 | Target field always populated | A bigger training set that is partly leakage |
| 2 | `FEATURE_NAMES`/`TARGET.name` misspelled | `resolve_spec` returns `FALLBACK_SPEC`: four columns of noise. Only `distinct=False` shows it |
| 3 | Label not drawn around the latent function | R² ≈ 0, enormous conformal interval, SHAP with nothing to attribute. **All 14 checks pass** |
| 3 | Noise too small | R² ≈ 0.99. SHAP re-reads your coefficients back to you |
| 3 | `DOMAIN_SERIES_LABEL` left alone | The old domain's sentence on the client-facing chart, forever |
| 4 | Tool not registered | Treated as HIGH: an over-cautious gate that looks like a policy choice |
| 4 | `risk` too low | A consequential write runs with no human in the loop. There is no second signal |
| 5 | `PERSONA_BY_ROLE` not re-pointed | **Every login raises `KeyError`** while every suite stays green |
| 6 | Floor replaced instead of composed | Check #8 fails — and if it did not, a tenant prompt could remove the platform's rules |
| 7 | `memory_subject_for` scoped wrong | A cross-subject data leak that looks completely normal in every test |
| 7/10 | Playbook renamed, `SKILL_HINTS` not | Never selected. `select_skills` returns `None` and nothing warns |
| 8 | Roster role not in `SPECIALIST_NODES` | Falls back to `qa` with a warning — **does not raise** |
| 8 | Stale `tool_allowlist` name | Intersected away in silence; the lane runs with fewer tools than you think |
| 9 | `cp -r` instead of `rsync --delete` | The old domain's documents still ingested and cited |
| — | `web/` untouched | The old domain's words on screen after a perfect Python retarget |
