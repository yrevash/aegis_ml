# PROMPT 12 · Integration — sync, tests, the vocabulary edit

---

## Role

You are taking a finished ten-piece adapter out of `aegis_ml` and into the Aegis checkout, correctly, in a way that leaves both suites green and the core still domain-ignorant.

**Prerequisite:** `missing_members(reference.adapter) == []`, `isinstance(reference.adapter, DomainAdapter)`, and `aegis-ml contract` passes.

---

## Paths

```
SRC = /Users/yrevash/aegis_ml/reference/adapter/
DST = /Users/yrevash/aegis/backend/src/app/adapter/
```

---

## Step 1 — **Sync, do not copy**

```bash
rsync -a --delete \
  /Users/yrevash/aegis_ml/reference/adapter/ \
  /Users/yrevash/aegis/backend/src/app/adapter/

find /Users/yrevash/aegis/backend/src/app/adapter -name '__pycache__' -type d -exec rm -rf {} +
```

```powershell
robocopy C:\aegis_ml\reference\adapter C:\aegis\backend\src\app\adapter /MIR /NFL /NDL /NJH /NJS
Get-ChildItem C:\aegis\backend\src\app\adapter -Recurse -Directory -Filter __pycache__ |
  Remove-Item -Recurse -Force
```

> **Note the trailing slash on the rsync source.** `reference/adapter/` copies the *contents*; without it you get `adapter/adapter/`.
>
> **`cp -r` is wrong.** It overwrites the Python modules and leaves everything you did not replace: the reference domain's **3 corpus documents** and **2 skill playbooks**. Retrieval will ingest and serve them, and `select_skills` may still name the stale playbooks. Nothing raises; the agent just cites the wrong domain's policy on stage.
>
> `robocopy /MIR` mirrors — it deletes destination files absent from the source, the same semantics as `--delete`. Note robocopy uses exit codes 0–7 for success.

**Verify the old data files are gone:**

```bash
ls /Users/yrevash/aegis/backend/src/app/adapter/corpus/*.md
ls /Users/yrevash/aegis/backend/src/app/adapter/skills/*.md
```

Neither may contain `kb_request_closure.md`, `policy_escalation.md`, `runbook_login_failures.md`, `closing_requests.md` or `de_escalation.md`.

**Fix the imports.** The synced modules import from `reference.adapter.*`; inside Aegis they must import from `app.adapter.*`:

```bash
cd /Users/yrevash/aegis/backend/src/app/adapter
grep -rn "reference\.adapter" .
sed -i '' 's/reference\.adapter/app.adapter/g' *.py corpus/*.py    # macOS
# sed -i    's/reference\.adapter/app.adapter/g' *.py corpus/*.py   # Linux
```

```powershell
Get-ChildItem C:\aegis\backend\src\app\adapter -Recurse -Filter *.py | ForEach-Object {
  (Get-Content $_.FullName) -replace 'reference\.adapter', 'app.adapter' | Set-Content $_.FullName
}
```

(Better: write the adapter with `from app.adapter.x import y` from the start and add `/Users/yrevash/aegis/backend/src` to `PYTHONPATH` while authoring. Then this step is a no-op.)

---

## Step 2 — The structural check

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import app.adapter
from aegis.adapter import DomainAdapter, missing_members
assert not missing_members(app.adapter), missing_members(app.adapter)
assert isinstance(app.adapter, DomainAdapter)
print('adapter contract: satisfied')
")
```

`missing_members` naming a piece almost always means `__init__.py` does not import that submodule — **a submodule is an attribute of its package only once something imports it.** This bit the reference adapter itself: `missing_members` returned `['memory_spec']` with the file on disk and named in the manifest.

Then verify the host-bound symbols survived:

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
bad = []
for mod, names in required.items():
    m = importlib.import_module(mod)
    bad += [f'{mod}.{n}' for n in names if not hasattr(m, n)]
assert not bad, 'host-bound symbols missing: ' + ', '.join(bad)
print('every host-bound symbol present')
")
```

---

## Step 3 — Rewrite `backend/tests/adapter/*`

| File | Shipped-domain literals | Action |
|---|---|---|
| `test_tools.py` | 26 | **Rewrite** |
| `test_allowlist.py` | 19 | **Rewrite** |
| `test_ml_spec.py` | 13 | **Rewrite** |
| `test_schema.py` | 9 | **Rewrite** |
| `test_generator.py` | 7 | **Rewrite** |
| `test_registry.py` | 3 | **Rewrite** |
| `conftest.py` | fixtures over your records | **Rewrite** |
| `test_piece_manifest.py` | — | **LEAVE ALONE** |
| `test_domain_adapter_protocol.py` | — | **LEAVE ALONE** |
| `test_conformance_suite.py` | — | **LEAVE ALONE** |
| `broken_adapter/` | — | **LEAVE ALONE** |

The three untouched files check *structure*, not domain. `broken_adapter/` is deliberately self-contained and imports nothing of yours — it is the negative fixture proving the conformance suite fails when it should.

**Add the learnability assertion while rewriting `test_ml_spec.py`:**

```python
def test_the_generated_label_is_learnable() -> None:
    """The target must be a function of the features, not an independent draw.

    Nothing in the conformance suite checks this — a pure-noise target passes all
    fourteen checks — and the only native symptom is `distinct=False` from
    `python -m app.ml`, read minutes before a demo. This fails in seconds instead.
    """
    from aegis_ml.data.latent import assert_learnable
    from app.adapter import ml_spec

    assert_learnable(
        ml_spec.training_frame(num_records=1200, seed=7),
        target=ml_spec.TARGET.name,
        task=ml_spec.TARGET.task,
        floor=0.15,
    )


def test_the_spec_resolves_to_this_domain_not_the_fallback() -> None:
    """`resolve_spec` reads leniently and returns FALLBACK_SPEC without raising."""
    from aegis.ml.spec import FALLBACK_SPEC, resolve_spec
    from app.adapter import ml_spec

    resolved = resolve_spec(ml_spec)
    assert resolved is not FALLBACK_SPEC
    assert resolved.target == ml_spec.TARGET.name
    assert resolved.features == ml_spec.FEATURE_NAMES
    assert resolved.frame_provider is not None
```

Install `aegis-ml[serve]` into the backend venv so those imports resolve:

```bash
uv pip install --python /Users/yrevash/aegis/backend/.venv -e '/Users/yrevash/aegis_ml[serve]'
/Users/yrevash/aegis/backend/.venv/bin/python -c "import pandas, numpy, numba; print(pandas.__version__, numpy.__version__, numba.__version__)"
```

The `[serve]` extra is pure-Python-or-already-present **by construction** and resolves under the caps. **Never install `[strong]` there.** If the version check moves, see `docs/09-troubleshooting.md` §17.

---

## Step 4 — **Edit `_vocabulary.py` — the one required core edit**

```
/Users/yrevash/aegis/aegis/src/aegis/conformance/_vocabulary.py
```

**Required and sanctioned.** `AGENTS.md` invariant 5: *"if you change what the reference adapter calls things, update it in the same commit (the check fails when a listed word no longer appears in the adapter either)."*

Replace `SHIPPED_DOMAIN_ID` and every entry of `SHIPPED_VOCABULARY` with your Brief's §14 list:

```python
SHIPPED_DOMAIN_ID = "surgical_scheduling"
"""``DOMAIN_ID`` of the adapter the vocabulary below belongs to."""

SHIPPED_VOCABULARY: tuple[str, ...] = (
    # piece 5 — the persona ids the login path used to decide between
    "theatre_coordinator",
    # piece 3 — the demand series' record collection and its client-facing title
    "dataset.procedures",
    "Procedures scheduled per day",
    "num_procedures",
    # piece 2 — the ML feature and target names
    "slot_position",
    "prior_overrun_mins",
    "equipment_swaps",
    "asa_grade",
    "surgeon_seniority",
    "slot_overrun_minutes",
    # piece 1 — the record types
    "TheatreDay",
    "Procedure",
    # piece 4 — the action tools
    "resequence_list",
    "reassign_theatre",
    "add_theatre_note",
    # piece 10 — the playbooks, and the domain's own id
    "containing_overrun",
    "cancelling_a_case",
    SHIPPED_DOMAIN_ID,
)
```

**The selection rule**, from the module's own docstring:

> *"Each one is specific enough that an innocent occurrence is not plausible — that is the selection rule. A generic word ('customer', 'client', 'request') is deliberately absent: this check must never be the reason somebody deletes a true sentence from a core docstring, or its first false positive is the last time anybody believes it."*

So list **exact identifiers only** — persona ids, record class names, feature columns, tool names, playbook filenames without `.md`, and the series label sentence verbatim. Never generic nouns.

**Both halves of check #14 must pass:**

1. No core module contains any listed term. If one does, you have a **real leak** — find it and move it into the adapter.
2. Because your `DOMAIN_ID` now equals `SHIPPED_DOMAIN_ID`, **every listed term must still be found inside your adapter.** A word you listed but never used is a failure, not decoration.

**Do not change** `MIN_CORE_FILES` (the anti-vacuity floor), `_SKIP_DIRS`, `core_files()` or `scan_for_terms()`.

**Report this edit.** It is expected; it is still a core edit.

Then run the check on its own:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q -k vocabulary -x)
```

---

## Step 5 — Sweep for real leaks

Before trusting the check, look yourself:

```bash
cd /Users/yrevash/aegis
grep -rn "operations_lead\|dataset\.requests\|Service requests opened per day\|num_requests\|queue_depth_at_open\|agent_tenure_months\|reopened_count\|customer_tier\|description_length\|resolution_hours\|ServiceRequest\|SupportAgent\|update_request_status\|assign_request\|add_case_note\|closing_requests\|de_escalation\|service_request_management" \
  aegis/src backend/src --include="*.py" | grep -v "aegis/src/aegis/conformance/"
```

That must return **nothing**. Any hit outside `conformance/` is a core module that still knows the old domain.

---

## Step 6 — The console

See `prompts/13-console.md`. Four files, all verified present, all outside the Python-only scan.

---

## Step 7 — OpenAPI and the TS client

Only needed if you changed a route, a request/response model or a `StreamEvent` variant. `backend/openapi.json` is **committed and snapshot-tested**, so run it and see whether it moved:

```bash
/Users/yrevash/aegis/backend/.venv/bin/python /Users/yrevash/aegis/scripts/build_openapi.py
(cd /Users/yrevash/aegis/web && npm run gen:api)
```

```powershell
C:\aegis\backend\.venv\Scripts\python.exe C:\aegis\scripts\build_openapi.py
Set-Location C:\aegis\web; npm run gen:api
```

`npm run gen:api:check` fails without writing, if you only want to know.

---

## Step 8 — Wire the ML tools into the agent loop

There is **no `ml_predict` node** in `graph.py`, and `describe_prediction` has **zero consumers**. Route ML through tools instead — no core edit:

```python
# in app/adapter/tools.py
from aegis_ml.serve.tools import ml_tool_specs

TOOL_REGISTRY.update({spec.name: spec for spec in ml_tool_specs(ToolSpec)})
```

All five are LOW and read-only, so **ML informs and never gates**; the human gate still fires on your HIGH-risk writes.

Add them to `ALLOWLIST` for the personas that should have them, and to at least one `SubAgentSpec.tool_allowlist`.

```bash
cd /Users/yrevash/aegis_ml && uv run pytest tests/test_ml_tools_roundtrip.py -q
```

---

## Step 9 — Train and promote

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml)
cd /Users/yrevash/aegis_ml && uv run aegis-ml train --tier all && uv run aegis-ml eval && uv run aegis-ml promote && uv run aegis-ml drift
```

Then restart the backend — `get_model()` caches the artifact and will not notice the swap.

See `prompts/11-ml-pipeline.md` for what to read in the output.

---

## The gate

Run every one. Paste the real output. See `prompts/14-final-gate.md`.

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q)
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter tests/agent -q)
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest -q)
(cd /Users/yrevash/aegis && PYTHONPATH=aegis/src backend/.venv/bin/python -m pytest aegis -q)
/Users/yrevash/aegis/backend/.venv/bin/python -m ruff check /Users/yrevash/aegis/aegis /Users/yrevash/aegis/backend
(cd /Users/yrevash/aegis/web && npx tsc --noEmit && npm test && npx next build)
```

---

## Checklist

- [ ] `rsync -a --delete` (or `robocopy /MIR`) used — **not** `cp -r`.
- [ ] The reference domain's 3 corpus documents and 2 playbooks are gone.
- [ ] `__pycache__` cleared under the adapter directory.
- [ ] Imports rewritten from `reference.adapter.*` to `app.adapter.*`.
- [ ] `missing_members(app.adapter) == []` and `isinstance(...) is True`.
- [ ] Every host-bound symbol resolves (the §2 script passes).
- [ ] `backend/tests/adapter/*` rewritten, with the four structural files untouched.
- [ ] `test_ml_spec.py` carries the learnability assertion.
- [ ] `aegis-ml[serve]` installed into `backend/.venv`, and pandas/numpy/numba unmoved.
- [ ] `_vocabulary.py` updated — **and reported**.
- [ ] The manual leak grep returns nothing outside `conformance/`.
- [ ] The four console files re-voiced (`prompts/13-console.md`).
- [ ] `openapi.json` regenerated if a route or model moved; TS client regenerated.
- [ ] The five ML tools registered, allowlisted and round-trip tested.
- [ ] **No core edit other than `_vocabulary.py`.** If there is one, it is a `SPECIALIST_NODES` addition and it is in the report.

---

## Next

`prompts/13-console.md`.
