# PROMPT 14 · The final gate

**Never report success without pasting real output.**

> *"A test that is weakened, skipped or deleted to make a change pass is a regression, not a fix."* — `AGENTS.md`

---

## Role

You are proving the retarget is complete and correct, and writing the report.

---

## Run every command. In this order. Paste the output.

### 0 · The structural check — seconds, and it catches a whole piece you forgot

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import app.adapter
from aegis.adapter import DomainAdapter, missing_members
assert not missing_members(app.adapter), missing_members(app.adapter)
assert isinstance(app.adapter, DomainAdapter)
print('adapter contract: satisfied')
print('DOMAIN_ID:', app.adapter.DOMAIN_ID)
")
```

```powershell
Push-Location C:\aegis\backend; $env:PYTHONPATH = "src;..\aegis\src"
.\.venv\Scripts\python.exe -c "import app.adapter; from aegis.adapter import DomainAdapter, missing_members; assert not missing_members(app.adapter), missing_members(app.adapter); assert isinstance(app.adapter, DomainAdapter); print('adapter contract: satisfied'); print('DOMAIN_ID:', app.adapter.DOMAIN_ID)"
Pop-Location
```

**Expected:** `adapter contract: satisfied`, and your `DOMAIN_ID`.

### 1 · The host-bound symbols

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import importlib
required = {
 'app.adapter': ['DEFAULT_PERSONA_ID','DOMAIN_DESCRIPTION','DOMAIN_ID','DOMAIN_SERIES_LABEL',
   'DOMAIN_SERIES_UNIT','GeneratorConfig','InMemoryRecordStore','TARGET','TOOL_REGISTRY',
   'ToolContext','agent_roster','domain_series_events','generate_synthetic_sync','get_persona',
   'is_allowed','load_seed_corpus','memory_spec','ml_spec','persona_for_role',
   'render_platform_floor','render_system_prompt','run_tool','sub_agent_roster',
   'tool_definitions_for','tools_for','training_frame','SyntheticDataset'],
 'app.adapter.tools': ['ALLOWLIST','AuditFn','RecordStore','TOOL_REGISTRY','ToolActionResult',
   'ToolContext','ToolNotAllowedError','UnknownToolError','is_allowed'],
 'app.adapter.roster': ['sub_agent_roster'],
 'app.adapter.memory_spec': ['FACT_TYPES','FACT_EXTRACTION_PROMPT','memory_subject_for'],
}
bad = [f'{m}.{n}' for m, ns in required.items()
       for n in ns if not hasattr(importlib.import_module(m), n)]
assert not bad, 'missing host-bound symbols: ' + ', '.join(bad)
print('every host-bound symbol present')
")
```

### 2 · Conformance — fourteen checks, no infrastructure, under a second

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)
```

**Expected: 14 passed.** Not 13, not 14 with a skip. If check #14 fails naming a core file, see `docs/09-troubleshooting.md` §14 — it is either a real leak or a stale `_vocabulary.py`.

### 3 · The adapter and the whole agent graph, on fakes

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter tests/agent -q)
```

This is the first command that can pass again once piece 8 landed — **and only once `tests/adapter/*` is rewritten for your domain**, which was part of pieces 1–8, not a follow-up. Compare against the baseline you recorded in step 0.2 of `docs/07-integration-with-aegis.md`.

### 4 · The full backend suite

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest -q)
```

### 5 · The core package — untouched apart from `_vocabulary.py`, so as green as before

```bash
(cd /Users/yrevash/aegis && PYTHONPATH=aegis/src backend/.venv/bin/python -m pytest aegis -q)
```

### 6 · Lint — must be clean

```bash
/Users/yrevash/aegis/backend/.venv/bin/python -m ruff check /Users/yrevash/aegis/aegis /Users/yrevash/aegis/backend
```

### 7 · The console

```bash
(cd /Users/yrevash/aegis/web && npx tsc --noEmit && npm test && npx next build)
```

### 8 · The ML spine, on your spec

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml)
```

**Read the last line. `distinct=True` is the pass signal.** And check the first line names **your** target and task — `target='target'` with four `feature_N` columns means `resolve_spec` returned `FALLBACK_SPEC`.

### 9 · This package

```bash
cd /Users/yrevash/aegis_ml
uv run pytest tests -q
uv run ruff check src reference tests
uv run aegis-ml contract
uv run aegis-ml doctor
```

### 10 · The leak sweep — both languages

```bash
cd /Users/yrevash/aegis

# Python: what check #14 scans
grep -rn "operations_lead\|dataset\.requests\|Service requests opened per day\|num_requests\|queue_depth_at_open\|agent_tenure_months\|reopened_count\|customer_tier\|description_length\|resolution_hours\|ServiceRequest\|SupportAgent\|update_request_status\|assign_request\|add_case_note\|closing_requests\|de_escalation\|service_request_management" \
  aegis/src backend/src --include="*.py" | grep -v "aegis/src/aegis/conformance/"

# TypeScript: what nothing scans
grep -rn "operations_lead\|update_request_status\|assign_request\|add_case_note\|find_requests\|queue_depth_at_open\|agent_tenure_months\|reopened_count\|description_length\|customer_tier\|resolution_hours\|ServiceRequest\|SupportAgent" \
  web/src web/tests 2>/dev/null

# and the prose no list can enumerate
grep -rniE "service request|support agent|case note|help ?desk" web/src backend/src/app/adapter | grep -v node_modules
```

**All three must return nothing.**

### 11 · The stale data files

```bash
ls /Users/yrevash/aegis/backend/src/app/adapter/corpus/*.md
ls /Users/yrevash/aegis/backend/src/app/adapter/skills/*.md
```

Neither may contain `kb_request_closure.md`, `policy_escalation.md`, `runbook_login_failures.md`, `closing_requests.md` or `de_escalation.md`. If they do, you used `cp -r`. Redo with `rsync -a --delete`.

### 12 · The running system

```bash
(cd /Users/yrevash/aegis && ./scripts/dev-native.sh)                                          # or .\scripts\start.ps1 -Mode full
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.seed)
(cd /Users/yrevash/aegis/web && npm run dev)
```

Open **http://localhost:3000**. Sign in as `admin` / `demo` — **and as `client` / `demo`**, because that is the login path no test in the repository exercises and where a stale `PERSONA_BY_ROLE` raises `KeyError`.

Then, with your own eyes:

- [ ] **Both logins succeed.** (This is the `KeyError` check.)
- [ ] Sample query 0 gives a cited answer about your domain.
- [ ] Sample query 1 fans out; every lane label is yours.
- [ ] Sample query 2 routes to `memory`.
- [ ] A query that needs a prediction shows the interval **and** the drivers in the transcript.
- [ ] A HIGH-risk action **pauses at the human gate**, and the approvals inbox shows it.
- [ ] Approving it completes the run.
- [ ] The ML-Ops panel predicts in **your** unit, with an interval, and `unknown_features` is empty.
- [ ] The forecast chart title is **your** `DOMAIN_SERIES_LABEL`.
- [ ] The simulation's two lanes visibly differ in scope.
- [ ] No screen anywhere says "service request", "support agent" or "case note".

---

## **Do not quote test counts from any document**

`AGENTS.md` and `README.md` in the Aegis repo **disagree with each other** — 2247 vs 2268 core tests, 1121 vs 1174 backend tests — and `SKILL.md`'s "132 passed" for the fast subset has drifted too. Every one of those numbers is stale.

**Record the numbers you actually get.** Report them as *your* baselines, and say when they were measured. A number typed into a README is a number nobody re-derives.

---

## The report

```markdown
# Retarget report — <Domain Title>

## What was built
- `DOMAIN_ID`: `<domain_id>`
- Ten adapter pieces, at `backend/src/app/adapter/`
- ML target: `<target>` (`<task>`, unit `<unit>`)
- Personas: `<a>` (ALL) · `<b>` (OWN on `<field>`)
- Tools: <n> domain tools (<n> HIGH-risk) + 5 read-only ML tools

## Core edits made
1. **`aegis/src/aegis/conformance/_vocabulary.py`** — REQUIRED and sanctioned by
   `AGENTS.md` invariant 5 and `SKILL.md`. `SHIPPED_DOMAIN_ID` changed from
   `service_request_management` to `<domain_id>`; `SHIPPED_VOCABULARY` replaced with
   <n> terms from this domain. Conformance check #14 requires both halves.
   <list the terms>
2. <any other core edit, or "None.">

## Console files re-voiced (outside the Python-only conformance scan)
- `web/src/config/personas.ts` — persona ids, names, blurbs, sample queries
- `web/src/components/ops/opsShared.ts` — `PROMPT_KEY`, two prompt strings
- `web/src/components/sim/SimulationView.tsx` — `SIM_QUERY`, both persona ids, labels
- `web/src/components/ml/MLOpsView.tsx` — the literal `EXAMPLE.features` row
- <any fifth file found, or "No fifth file found.">

## New dependencies
`aegis-ml[serve]` installed into `backend/.venv` (pure-Python-or-already-present; the
caps hold: pandas <x>, numpy <x>, numba <x>). `aegis-ml[strong]` in the separate
`.venv-ml`. **Not added to `aegis/pyproject.toml`.**

## New environment variables
`AEGIS_ML_AEGIS_ROOT`, `AEGIS_ML_REGISTRY_DIR`, `AEGIS_ML_TRAINER_VENV`
<and any of the ENABLE_* switches you set>

## Test results — measured on <date>, not quoted from any document
| Command | Result |
|---|---|
| structural check | `adapter contract: satisfied` |
| conformance (14 checks) | <paste> |
| `tests/adapter tests/agent` | <paste> |
| full backend suite | <paste> |
| core package suite | <paste> |
| ruff | <paste> |
| console tsc / test / build | <paste> |
| `aegis_ml` tests + ruff | <paste> |
| leak sweep (py + ts + prose) | no matches |

## `python -m app.ml`
<paste all four lines, including `distinct=True`>

## ML results
| Metric | Value |
|---|---|
| Primary metric | `<name>` = <value> |
| Requested coverage | 0.90 |
| **Measured (empirical) coverage** | <value> |
| Worst slice | `<feature>=<level>` <value> (n=<rows>) |
| Leaderboard: winner | `<name>` (<tier>) <value>, portable=<bool> |
| Leaderboard: baseline | <value> — margin <delta> |
| Accuracy ceiling (non-portable) | `<name>` <value> — reported, not promoted |
| `data_source` | `provided` / `spec_provider` |
| `dataset_digest` | `sha256:<...>` |
| Gate decision | promoted / rejected, with every criterion's number |
| Drift verdict | `<pass\|warn\|block>`, drifted_share <value> |
| NannyML estimated `<metric>` | <value> (an ESTIMATE, not a measurement) |

## Synthetic-data realism
| Property | Value |
|---|---|
| `target_r2` designed for | <value> |
| Held-out score achieved | <value> |
| Noise model | Gaussian, heteroscedastic (σ scales with `<feature>`) |
| Unobserved confounder | `<name>`, σ=<value>, per `<group>` |
| MAR missingness | `<feature>` <n>% when `<other>` == `<value>` |
| Deliberately irrelevant features | `<a>`, `<b>` — flat in SHAP, as intended |
| Interaction term | `<a>` × `<b>` |
| Leakage scan | max single-feature score <value> < 0.98 |

## Manual verification
Logged in as `admin` **and** as `client` — both succeed (the `PERSONA_BY_ROLE`
`KeyError` path, which no test exercises). <then the §12 checklist, ticked>

## Known gaps
<anything you did not finish, plainly>
```

---

## Final checklist

- [ ] All 12 commands run, output pasted.
- [ ] Conformance: **14 passed**.
- [ ] `distinct=True`.
- [ ] Both suites at or above the baseline **you** recorded.
- [ ] ruff clean; console builds.
- [ ] All three leak sweeps return nothing.
- [ ] The reference domain's 3 documents and 2 playbooks are gone.
- [ ] **No core edit except `_vocabulary.py`** — and it is reported with its terms.
- [ ] The four console files are named in the report.
- [ ] No test count quoted from any document.
- [ ] Signed in as **both** `admin` and `client`.
- [ ] The human gate fired on a real HIGH-risk action and an approval completed the run.
- [ ] The ML evidence appeared in a transcript with its interval and drivers.
- [ ] No screen anywhere shows the reference domain's words.

---

`prompts/CHECKLIST.md` is the one-page version of all of this.
