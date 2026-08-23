# 00 · START HERE

**You are an agent. You have just been handed a problem statement and this repository. This file is a two-minute orientation. Read it completely before you do anything else.**

---

## 1. What is in front of you

There are **two checkouts** on this machine and you will work across both.

| Path | What it is | May you edit it? |
|---|---|---|
| `/Users/yrevash/aegis_ml/` | **This repo.** A pip-installable package (`aegis-ml`) holding the ML/MLOps machinery Aegis lacks, adapter templates, authoring prompt-packs, and a fully worked reference domain. | Yes — this is your workspace. |
| `/Users/yrevash/aegis/` | **The Aegis platform.** A domain-agnostic enterprise agentic-AI platform: an importable core (`aegis/src/aegis/`), a FastAPI host (`backend/src/app/`), a Next.js console (`web/`). | Only `backend/src/app/adapter/`, `backend/tests/adapter/`, four `web/` files, and **exactly one** core file. See §4. |

Aegis is retargeted to a new problem by writing **one thing**: a *domain adapter* satisfying the `aegis.adapter.DomainAdapter` Protocol. Eleven members across ten pieces. That is the whole job. The agent graph, human gate, memory, retrieval, RBAC, tracing, guardrails and console all keep working untouched.

`aegis_ml` exists so that you do not spend the morning re-deriving that contract and hand-rolling ML from scratch.

---

## 2. What you are about to do

Nine steps, in this order. Each has a prompt-pack in `prompts/` and a chapter in `docs/`.

| # | Step | Prompt-pack | Doc |
|---|---|---|---|
| 0 | Verify the environment | — | `docs/08-windows.md` (Windows) |
| 1 | Problem statement → **Domain Brief** | `prompts/00-intake.md` | `docs/03-authoring-a-domain.md` |
| 2 | Fill the ten adapter pieces from the Brief | `prompts/01-schema.md` … `prompts/10-skills.md` | `docs/02-domain-adapter-contract.md`, `docs/03-authoring-a-domain.md` |
| 3 | Prove the synthetic label is learnable | `prompts/04-generator.md` | `docs/04-synthetic-data.md` |
| 4 | Sync the adapter into Aegis | `prompts/12-integration.md` | `docs/07-integration-with-aegis.md` |
| 5 | Rewrite `backend/tests/adapter/*` | `prompts/12-integration.md` | `docs/07-integration-with-aegis.md` |
| 6 | Edit the quarantined vocabulary list | `prompts/12-integration.md` | `docs/07-integration-with-aegis.md` |
| 7 | Re-voice the four console files | `prompts/13-console.md` | `docs/07-integration-with-aegis.md` |
| 8 | Train, evaluate, promote, monitor | `prompts/11-ml-pipeline.md` | `docs/05-ml-pipelines.md`, `docs/06-mlops-registry-drift.md` |
| 9 | Final gate | `prompts/14-final-gate.md` | `prompts/CHECKLIST.md` |

---

## 3. The reading order

Read in this order. Do not skip 02 or 04.

1. **`docs/01-what-is-aegis.md`** — what the platform is, and **what it already gives you for free**. Read this so you do not rebuild conformal prediction, SHAP, forecasting, guardrails, RBAC or the human gate. They exist.
2. **`docs/02-domain-adapter-contract.md`** — the contract, in full. The reference you will return to constantly.
3. **`docs/03-authoring-a-domain.md`** — how to go from a problem statement to ten filled pieces.
4. **`docs/04-synthetic-data.md`** — **the most technically important document in this repo.** The single most expensive failure available to you is described here.
5. **`docs/05-ml-pipelines.md`** and **`docs/06-mlops-registry-drift.md`** — the ML half.
6. **`docs/07-integration-with-aegis.md`** — the day-of procedure, end to end, with exact commands.
7. `docs/08-windows.md` if you are on Windows. `docs/09-troubleshooting.md` when something breaks. `docs/10-architecture-decisions.md` when you need to justify a choice to a judge.

---

## 4. The four things that will cost you the demo

Memorise these now. Each is expanded later; none of them raises an exception.

### 4.1 A target that is noise passes every automated check Aegis has

There is **no conformance check** that the generated label is coupled to the features. A target drawn independently of the features passes **all fourteen** conformance checks and the entire backend suite. The only native symptom is `distinct=False` on the last line of `python -m app.ml` — read minutes before a demo.

`aegis_ml.data.latent.assert_learnable` catches it in seconds. Run it before anything expensive. See `docs/04-synthetic-data.md`.

### 4.2 `resolve_spec` silently returns four columns of noise

`aegis/src/aegis/ml/spec.py` reads your ML spec leniently:

```python
features = getattr(candidate, "FEATURE_NAMES", None) or getattr(candidate, "features", None)
target_obj = getattr(candidate, "TARGET", None)
target = getattr(target_obj, "name", None) or getattr(candidate, "target", None)
if not features or not target:
    return FALLBACK_SPEC          # features "feature_0".."feature_3", target "target"
```

Nothing raises. Misspell `FEATURE_NAMES` and the trustworthy spine trains happily on generated noise and serves the result as domain evidence. Conformance check #12 (`test_ml_spec_resolves_to_the_domain_not_the_fallback`) is the backstop; generating `ml_spec.py` from a single `MLProblem` spec (`aegis_ml.contracts.spec`) is the prevention.

### 4.3 Four console files show the old domain's words after a perfect Python retarget

The vocabulary-quarantine conformance check scans **Python only**. These four TypeScript files are outside it and carry shipped-domain literals:

```
/Users/yrevash/aegis/web/src/config/personas.ts
/Users/yrevash/aegis/web/src/components/ops/opsShared.ts
/Users/yrevash/aegis/web/src/components/sim/SimulationView.tsx
/Users/yrevash/aegis/web/src/components/ml/MLOpsView.tsx
```

All four verified present. Re-voice them by hand. See `prompts/13-console.md`.

### 4.4 `cp -r` leaves the old domain's documents behind

A plain copy into `backend/src/app/adapter/` leaves the reference domain's corpus documents and skill playbooks in place, and retrieval will serve them. Use:

```bash
rsync -a --delete /Users/yrevash/aegis_ml/reference/adapter/ /Users/yrevash/aegis/backend/src/app/adapter/
```

---

## 5. The one sanctioned core edit

Everything under `/Users/yrevash/aegis/aegis/src/aegis/` stays untouched **with exactly one exception**:

```
/Users/yrevash/aegis/aegis/src/aegis/conformance/_vocabulary.py
```

Editing it is **required**, not tolerated. Conformance check #14 (`test_no_shipped_domain_vocabulary_survives_outside_the_adapter`) asserts that every word in `SHIPPED_VOCABULARY` still appears inside the *currently loaded* adapter when that adapter's `DOMAIN_ID` matches `SHIPPED_DOMAIN_ID`. Retarget the adapter without updating this list and you have a stale quarantine list. `AGENTS.md` §5 says so explicitly: *"if you change what the reference adapter calls things, update it in the same commit."*

Report the edit. It is expected. Nothing else in `aegis/src/aegis/` may change.

(One *further* core edit is sanctioned but almost never needed: adding a third specialist requires a `SPECIALIST_NODES` entry in `aegis/src/aegis/agent/graph.py`. Avoid it — re-voice the existing `qa` and `memory` roles instead. See `docs/02-domain-adapter-contract.md` §8.)

---

## 6. Do not quote test counts

`AGENTS.md` and `README.md` in the Aegis repo **disagree** with each other (2247 vs 2268 core tests; 1121 vs 1174 backend tests). Both are drifted. **Record whatever number you actually get** when you run the suites, and use that as your regression baseline. Never quote a number from a document.

---

## 7. Your first three commands

Run these now, in order, from a shell.

```bash
# 1 — the ML package resolves and its tooling is present
cd /Users/yrevash/aegis_ml && uv sync --extra dev && uv run aegis-ml doctor

# 2 — the Aegis backend venv exists and the adapter contract is currently satisfied
cd /Users/yrevash/aegis && ./scripts/bootstrap.sh   # idempotent; skip if backend/.venv exists
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import app.adapter
from aegis.adapter import DomainAdapter, missing_members
print('missing:', missing_members(app.adapter))
print('satisfies:', isinstance(app.adapter, DomainAdapter))
")

# 3 — the green baseline you will regress against. WRITE THE NUMBER DOWN.
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter tests/agent -q)
```

Windows PowerShell equivalents are in `docs/08-windows.md` §3.

Expected from command 2: `missing: []` and `satisfies: True`. If not, the checkout is broken — fix that before writing a line of domain code.

---

## 8. Your next step

**Read `docs/01-what-is-aegis.md`, then `docs/02-domain-adapter-contract.md`. Then open `prompts/00-intake.md` and turn the problem statement into a Domain Brief.**

Do not write adapter code before the Brief exists. Every prompt-pack from `01-schema.md` onward reads the Brief as its input, and a piece written without it will disagree with the pieces written after it.
