# 07 · Integration with Aegis — the day-of procedure

Nine steps, end to end, with exact commands. Every path is absolute or explicitly repo-relative.

Repo roots referenced throughout:

```
ML_ROOT     = /Users/yrevash/aegis_ml
AEGIS_ROOT  = /Users/yrevash/aegis
ADAPTER_DIR = /Users/yrevash/aegis/backend/src/app/adapter
```

Windows equivalents assume `C:\aegis_ml` and `C:\aegis`; see `docs/08-windows.md`.

---

## Step 0 — `aegis-ml doctor`

```bash
cd /Users/yrevash/aegis_ml && uv sync --extra dev && uv run aegis-ml doctor
```

```powershell
Set-Location C:\aegis_ml; uv sync --extra dev; uv run aegis-ml doctor
```

Prints and checks:

| Line | What a failure means |
|---|---|
| Python, `aegis_ml` version | — |
| Resolved versions of pandas, numpy, sklearn, xgboost, mapie, shap, pandera | A cap violation. See `docs/09-troubleshooting.md`. |
| Available AutoML tiers, with a reason per unavailable one | Distinguishes *disabled by policy* from *not installed*. |
| Trainer venv path and whether its interpreter exists | `TrainerVenvMissingError` — create it, §1.1. |
| `settings.artifact_path` and whether its directory is writable | Promotion will fail at the last step otherwise. |
| `settings.aegis_root` and whether the adapter directory exists | Wrong checkout path. Set `AEGIS_ML_AEGIS_ROOT`. |
| Postgres reachability (only if `AEGIS_ML_POSTGRES_DSN` is set) | Optional; the filesystem registry is complete without it. |
| The TabPFN licence notice | Informational, always printed when the tier is enabled. |
| `assert_learnable` against the current adapter's `training_frame` | The big one. See `docs/04-synthetic-data.md`. |

Also, in the Aegis checkout:

```bash
cd /Users/yrevash/aegis && ./scripts/bootstrap.sh
```

Idempotent, no Docker, no GPU, no database. Installs `backend/.venv` with every extra plus the console's npm dependencies. **A fresh clone has no `backend/.venv` at all** — every command below runs `backend/.venv/bin/python`.

### 0.1 The trainer venv

```bash
cd /Users/yrevash/aegis_ml
uv venv .venv-ml --python 3.11
uv pip install --python .venv-ml -e '.[strong,serve]'
```

```powershell
Set-Location C:\aegis_ml
uv venv .venv-ml --python 3.11
uv pip install --python .venv-ml -e ".[strong,serve]"
```

If time is short, skip it. `baseline` and `flaml` run in the serving venv and produce a portable recipe; `autogluon` and `tabpfn` will be reported as unavailable **with a reason**, which is honest and costs you a leaderboard row, not the demo.

### 0.2 The green baseline

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter tests/agent -q)
```

**Write down whatever number you actually get.** Do not use a number from any document — `AGENTS.md` and `README.md` in the Aegis repo disagree with each other (2247 vs 2268 core; 1121 vs 1174 backend), and `SKILL.md`'s "132 passed" for this subset has drifted too. Your number is your regression baseline and it should only ever grow.

---

## Step 1 — Problem statement → Domain Brief

Open `prompts/00-intake.md`. Feed it the problem statement. Write the result to:

```
/Users/yrevash/aegis_ml/DOMAIN_BRIEF.md
```

**Do not skip this and do not write adapter code first.** Every prompt-pack from `01-schema.md` onward reads the Brief. See `docs/03-authoring-a-domain.md` §2.

---

## Step 2 — Fill the ten pieces

Author **inside `aegis_ml`**, not in the Aegis checkout:

```bash
cd /Users/yrevash/aegis_ml
uv run aegis-ml init --domain <domain_id> --out reference/adapter
```

Then fill the pieces in order — `schema` → `ml_spec` → `generator` → `tools` → `personas` → `prompts` → `memory_spec` → `roster` → `corpus` → `skills` — using `prompts/01-schema.md` … `prompts/10-skills.md`.

Working here rather than in the Aegis tree means the backend suite does not go red at import while you are mid-flight, and you can run the cheap gate against the adapter continuously.

Verify continuously (this works before the sync, because `reference/` is importable from `ML_ROOT`):

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
import reference.adapter as a
from aegis.adapter import DomainAdapter, missing_members
print('missing:', missing_members(a)); print('satisfies:', isinstance(a, DomainAdapter))
"
```

If `aegis` is not importable from the `aegis_ml` venv, add it to the path:

```bash
PYTHONPATH=/Users/yrevash/aegis/aegis/src uv run python -c "..."
```

---

## Step 3 — The cheap gate, before anything expensive

```bash
cd /Users/yrevash/aegis_ml && uv run aegis-ml contract
```

pandera + `assert_learnable` + leakage scan. **Seconds.** If `LabelNotLearnableError` fires, stop and fix the generator — nothing downstream is worth running. See `docs/04-synthetic-data.md` §6.

Also run the conformance suite against the authored adapter before you sync anything:

```bash
(cd /Users/yrevash/aegis/backend && \
 PYTHONPATH=src:../aegis/src:/Users/yrevash/aegis_ml \
 .venv/bin/python -m pytest --pyargs aegis.conformance --aegis-adapter reference.adapter -q)
```

Expect check #14 (vocabulary quarantine) to **fail at this point** — the core still names the shipped domain and your `DOMAIN_ID` no longer matches `SHIPPED_DOMAIN_ID`. Step 6 fixes it. The other thirteen should be green.

---

## Step 4 — **Sync, do not copy**

```bash
rsync -a --delete \
  /Users/yrevash/aegis_ml/reference/adapter/ \
  /Users/yrevash/aegis/backend/src/app/adapter/
```

PowerShell (no rsync; `robocopy /MIR` is the equivalent):

```powershell
robocopy C:\aegis_ml\reference\adapter C:\aegis\backend\src\app\adapter /MIR /NFL /NDL /NJH /NJS
```

> **Note the trailing slash on the rsync source.** `reference/adapter/` copies the *contents*; `reference/adapter` would create `adapter/adapter/`.
>
> **Why not `cp -r`.** A plain copy overwrites the Python modules but leaves everything you did not replace: the reference domain's **3 corpus documents** (`kb_request_closure.md`, `policy_escalation.md`, `runbook_login_failures.md`) and **2 skill playbooks** (`closing_requests.md`, `de_escalation.md`). Retrieval will ingest and serve those documents alongside yours, and `select_skills` may still name the stale playbooks. Nothing raises; the agent just cites the wrong domain's policy on stage.
>
> `robocopy /MIR` mirrors, i.e. it deletes destination files absent from the source — the same semantics as `--delete`.

Clean the stale bytecode, which `--delete` will not remove if it is in a `__pycache__` you did not mirror:

```bash
find /Users/yrevash/aegis/backend/src/app/adapter -name '__pycache__' -type d -exec rm -rf {} +
```

```powershell
Get-ChildItem C:\aegis\backend\src\app\adapter -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

Then confirm the shipped domain's files are gone:

```bash
ls /Users/yrevash/aegis/backend/src/app/adapter/corpus/*.md
ls /Users/yrevash/aegis/backend/src/app/adapter/skills/*.md
```

Neither listing may contain `kb_request_closure.md`, `policy_escalation.md`, `runbook_login_failures.md`, `closing_requests.md` or `de_escalation.md`.

**Now the contract check, against the real path:**

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import app.adapter
from aegis.adapter import DomainAdapter, missing_members
assert not missing_members(app.adapter), missing_members(app.adapter)
assert isinstance(app.adapter, DomainAdapter)
print('adapter contract: satisfied')
")
```

---

## Step 5 — Rewrite `backend/tests/adapter/*`

These files are **not** domain-neutral scaffolding. They carry between 3 and 26 shipped-domain literals each:

| File | Shipped-domain literals | Action |
|---|---|---|
| `test_tools.py` | 26 | **Rewrite** |
| `test_allowlist.py` | 19 | **Rewrite** |
| `test_ml_spec.py` | 13 | **Rewrite** |
| `test_schema.py` | 9 | **Rewrite** |
| `test_generator.py` | 7 | **Rewrite** |
| `test_registry.py` | 3 | **Rewrite** |
| `conftest.py` | fixtures over your records | **Rewrite** |
| `test_piece_manifest.py` | — | **LEAVE ALONE** — counts pieces on disk |
| `test_domain_adapter_protocol.py` | — | **LEAVE ALONE** — checks structure |
| `test_conformance_suite.py` | — | **LEAVE ALONE** |
| `broken_adapter/` | — | **LEAVE ALONE** — deliberately self-contained, imports nothing of yours |

While rewriting `test_ml_spec.py`, **add the learnability assertion** so it runs on every suite invocation:

```python
def test_the_generated_label_is_learnable() -> None:
    """The target must be a function of the features, not an independent draw.

    Nothing in the conformance suite checks this — a noise target passes all fourteen
    checks — and the only native symptom is `distinct=False` from `python -m app.ml`,
    read minutes before a demo. This fails in seconds instead.
    """
    from aegis_ml.data.latent import assert_learnable
    from app.adapter import ml_spec

    assert_learnable(
        ml_spec.training_frame(num_records=1200, seed=7),
        target=ml_spec.TARGET.name,
        task=ml_spec.TARGET.task,
        floor=0.15,
    )
```

(That requires `aegis-ml[serve]` to be installed into `backend/.venv`. It resolves cleanly under the caps — that is what the `[serve]` extra is for. `uv pip install --python /Users/yrevash/aegis/backend/.venv -e '/Users/yrevash/aegis_ml[serve]'`.)

Verify:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter -q)
```

---

## Step 6 — Edit `_vocabulary.py` — **the one required core edit**

```
/Users/yrevash/aegis/aegis/src/aegis/conformance/_vocabulary.py
```

**This edit is required and it is sanctioned.** `AGENTS.md` invariant 5: *"The quarantined word list is `aegis/src/aegis/conformance/_vocabulary.py`; if you change what the reference adapter calls things, update it in the same commit (the check fails when a listed word no longer appears in the adapter either)."*

Everything else under `/Users/yrevash/aegis/aegis/src/aegis/` stays untouched.

Replace `SHIPPED_DOMAIN_ID` and every entry of `SHIPPED_VOCABULARY` with **your** domain's words:

```python
SHIPPED_DOMAIN_ID = "<your_domain_id>"

SHIPPED_VOCABULARY: tuple[str, ...] = (
    # piece 5 — persona ids the login path might decide between
    "<your_staff_persona_id>",
    # piece 3 — the demand series' record collection and its client-facing title
    "dataset.<your_record_collection>",
    "<Your Series Label Sentence>",
    "num_<your_records>",
    # piece 2 — the ML feature and target names
    "<feature_1>", "<feature_2>", "<feature_3>", "<target_name>",
    # piece 1 — the record types
    "<YourRecordType>", "<YourOtherRecordType>",
    # piece 4 — the action tools
    "<your_high_risk_tool>", "<your_medium_risk_tool>", "<your_low_risk_tool>",
    # piece 10 — the playbooks, and the domain's own id
    "<your_playbook_a>", "<your_playbook_b>",
    SHIPPED_DOMAIN_ID,
)
```

**The selection rule, from the module's own docstring:** *"Each one is specific enough that an innocent occurrence is not plausible — that is the selection rule. A generic word ('customer', 'client', 'request') is deliberately absent: this check must never be the reason somebody deletes a true sentence from a core docstring, or its first false positive is the last time anybody believes it."*

So: **do not list generic nouns.** List the exact identifiers — persona ids, record class names, feature column names, tool names, playbook filenames without `.md`, the series label sentence verbatim.

**Both halves of check #14 must pass:**

1. No core module contains any listed term. (If one does, you have a real leak — find it and move it into the adapter.)
2. Because your adapter's `DOMAIN_ID` now equals `SHIPPED_DOMAIN_ID`, **every listed term must still be found inside your adapter.** A word you listed but never used is a failure, not decoration.

Do not change `MIN_CORE_FILES`, `_SKIP_DIRS`, `core_files()` or `scan_for_terms()`.

**Report this edit.** It is expected; it is still a core edit.

---

## Step 7 — Re-voice the four console files

`web/` is outside the adapter **and outside the conformance check, which scans Python only.** These four files carry shipped-domain literals and **will show the old domain's words on screen after an otherwise perfect retarget.** All four verified present.

| File (absolute) | What it names |
|---|---|
| `/Users/yrevash/aegis/web/src/config/personas.ts` | The persona ids, and the tool names in its prose. Its own docstring says *"Every `id` here must exist in the backend adapter's persona table"* and that *"`POST /query` … answers **400 Unknown persona**"* for an invented id. |
| `/Users/yrevash/aegis/web/src/components/ops/opsShared.ts` | `PROMPT_KEY`, and the tool names in two prompt strings |
| `/Users/yrevash/aegis/web/src/components/sim/SimulationView.tsx` | The persona id it drives the scripted demo with |
| `/Users/yrevash/aegis/web/src/components/ml/MLOpsView.tsx` | **A literal ML feature row** — the same defect as the trainer's old sanity probe |

`prompts/13-console.md` has the exact edits. Find every remaining occurrence:

```bash
cd /Users/yrevash/aegis/web
grep -rn "operations_lead\|update_request_status\|assign_request\|add_case_note\|find_requests\|queue_depth_at_open\|agent_tenure_months\|reopened_count\|description_length\|customer_tier\|resolution_hours\|service_request\|Service requests opened per day" src/ tests/ 2>/dev/null
```

```powershell
Set-Location C:\aegis\web
Select-String -Path src\*,tests\* -Recurse -Pattern "operations_lead|update_request_status|assign_request|add_case_note|find_requests|queue_depth_at_open|agent_tenure_months|reopened_count|description_length|customer_tier|resolution_hours|service_request|Service requests opened per day"
```

Then:

```bash
(cd /Users/yrevash/aegis/web && npx tsc --noEmit && npm test && npx next build)
```

> The real fix is for the API to serve the persona list and the feature spec so the console reads them like everything else. There is no such endpoint yet, and **inventing one mid-retarget is not the moment.** Re-voice by hand and say in your report that you did.

---

## Step 8 — OpenAPI and the TypeScript client

Only needed if you changed a route, a request model, a response model or a `StreamEvent` variant. A pure adapter retarget usually does not — but `backend/openapi.json` is **committed and snapshot-tested**, so run it and check whether it moved.

```bash
/Users/yrevash/aegis/backend/.venv/bin/python /Users/yrevash/aegis/scripts/build_openapi.py
(cd /Users/yrevash/aegis/web && npm run gen:api)
```

```powershell
C:\aegis\backend\.venv\Scripts\python.exe C:\aegis\scripts\build_openapi.py
Set-Location C:\aegis\web; npm run gen:api
```

Check-only variant (fails if the generated client is stale): `npm run gen:api:check`.

`backend/tests/api/test_openapi_snapshot.py` fails if the committed snapshot is out of date, and asserts `additionalProperties: false` for every request body — request models carry `extra="forbid"` because pydantic's default drops an unknown field in silence and answers 200, which has swallowed a request field four times in this project.

---

## Step 9 — Train, promote, monitor

In this order.

```bash
# 9.1 — the Aegis spine, on your spec. distinct=True is the pass signal.
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml)

# 9.2 — the AutoML search (uses .venv-ml for the strong tiers)
cd /Users/yrevash/aegis_ml && uv run aegis-ml train --tier all

# 9.3 — metrics, coverage, slices, SHAP, card
uv run aegis-ml eval

# 9.4 — the five-criterion gate; on pass, writes backend/.artifacts/ml_spine.joblib
uv run aegis-ml promote

# 9.5 — drift + label-free performance estimation
uv run aegis-ml drift
```

Read the last line of 9.1:

```
  sanity: lowest-labelled row=3.2 minutes  highest-labelled row=71.8 minutes  (distinct=True)
```

`distinct=False` means the spine learned nothing. Go to `docs/04-synthetic-data.md`, not to the prompts.

Read `card.json` per `docs/05-ml-pipelines.md` §7: metric in the target band, empirical coverage within tolerance of requested, no collapsed slice, `data_source` not `"synthetic"`, `dataset_digest` present.

---

## The final gate

Run all six, in order, and paste the real output into your report. Never claim success without it.

```bash
# 0 — the structural check
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import app.adapter
from aegis.adapter import DomainAdapter, missing_members
assert not missing_members(app.adapter), missing_members(app.adapter)
assert isinstance(app.adapter, DomainAdapter)
print('adapter contract: satisfied')
")

# 1 — conformance: fourteen checks, no infrastructure, under a second
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)

# 2 — the adapter and the whole agent graph, on fakes
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter tests/agent -q)

# 3 — the full backend suite
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest -q)

# 4 — the core package, untouched apart from _vocabulary.py, so as green as before
(cd /Users/yrevash/aegis && PYTHONPATH=aegis/src backend/.venv/bin/python -m pytest aegis -q)

# 5 — lint
/Users/yrevash/aegis/backend/.venv/bin/python -m ruff check /Users/yrevash/aegis/aegis /Users/yrevash/aegis/backend

# 6 — the console
(cd /Users/yrevash/aegis/web && npx tsc --noEmit && npm test && npx next build)

# 7 — this package
cd /Users/yrevash/aegis_ml && uv run pytest tests -q && uv run ruff check src reference tests
```

`tests/adapter/test_piece_manifest.py` is the tripwire for the structure itself: add a ninth module and it fails until `SKILL.md`, `adapter/README.md` and every `piece N of M` docstring are updated together. That is intentional.

---

## 7·bis — How `aegis_ml.serve.tools` puts ML into the agent loop

> **The problem.** The Aegis README's request path names an `ml_predict` node. **There is no such node.** `aegis/src/aegis/agent/graph.py` declares neither an `ml_predict` entry in `NODE_LABELS` nor a builder call for one, and `describe_prediction` — the adapter member that renders a prediction into the plan — has **zero consumers** across `backend/src/`, `aegis/src/` and `web/src/`. The prose describes an intention that was never wired.

**The answer: route ML through adapter *tools*, which are already wired, already gated, already audited, and already visible in the console.** No core edit.

`aegis_ml.serve.tools` ships ready-made `ToolSpec`s that drop into your `TOOL_REGISTRY`:

| Tool | Risk | read_only | idempotent | What it does |
|---|---|---|---|---|
| `predict_outcome` | LOW | ✔ | ✔ | Runs the promoted spine on a record and returns the prediction **with its conformal interval** |
| `explain_prediction` | LOW | ✔ | ✔ | Top-k signed SHAP drivers, rendered by `describe_prediction` |
| `whatif_scenario` | LOW | ✔ | ✔ | Re-predicts with one or more features overridden, and reports the delta |
| `forecast_series` | LOW | ✔ | ✔ | The domain demand series with conformal bands |
| `check_model_health` | LOW | ✔ | ✔ | Champion run id, metric, requested vs empirical coverage, last drift verdict |

Wire them in `tools.py`:

```python
from aegis_ml.serve.tools import ml_tool_specs

TOOL_REGISTRY: dict[str, ToolSpec] = {
    **{spec.name: spec for spec in ml_tool_specs(ToolSpec)},
    "find_procedures":  ToolSpec(...),
    "resequence_list":  ToolSpec(...),
}

ALLOWLIST: dict[str, frozenset[str]] = {
    "theatre_coordinator": frozenset(TOOL_REGISTRY),
    "surgeon": frozenset({"predict_outcome", "explain_prediction", "add_theatre_note"}),
}
```

`ml_tool_specs` is handed your own `ToolSpec` class so the tools are constructed in **your** domain's shape — the package does not import from `app.*`, honouring invariant 1.

This finally gives `describe_prediction` a consumer, and it respects the platform's stated rule: **ML informs, it never gates.** All five tools are LOW and read-only. The human gate still fires on *your* HIGH-risk write tools, never on model confidence.

Verify the round trip:

```bash
cd /Users/yrevash/aegis_ml && uv run pytest tests/test_ml_tools_roundtrip.py -q
```

---

## What to report when you are done

From `SKILL.md`, plus this package's additions:

1. New dependencies (**do not add them to `aegis/pyproject.toml` yourself — name them**).
2. New environment variables.
3. **The `_vocabulary.py` edit**, with the old and new `SHIPPED_DOMAIN_ID` and the term list.
4. Any *other* core edit and why (there should be none; a `SPECIALIST_NODES` addition is the only other sanctioned one).
5. **The four console files you re-voiced.**
6. The outputs of all the final-gate commands, pasted verbatim.
7. Your recorded test baselines — the numbers you actually got, not any number from a document.
8. The ML numbers: primary metric, requested vs empirical coverage, the worst slice, the leaderboard margin over baseline, and the gate decision.

---

## Next

`docs/08-windows.md` · `docs/09-troubleshooting.md` · `docs/10-architecture-decisions.md`
