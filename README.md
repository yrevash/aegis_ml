# aegis_ml

**A SOTA ML/MLOps adapter factory for the [Aegis](../aegis) agentic-AI platform.**

Aegis is retargeted to a new problem by writing exactly one thing: a *domain adapter* satisfying `aegis.adapter.DomainAdapter` — eleven members across ten pieces. `aegis_ml` is the base you start from: the ML/MLOps machinery Aegis lacks, templates and authoring prompt-packs for all ten pieces, and a fully worked reference domain.

> **Agent? Start at [`docs/00-START-HERE.md`](docs/00-START-HERE.md).**

---

## Why this exists

On hackathon day you hand a coding agent a problem statement and a repository. Without a base, it spends the morning re-deriving the adapter contract and hand-rolling ML from scratch — and it hits four traps that are all completely silent.

`aegis_ml` removes that morning. It carries:

- **the ML/MLOps machinery Aegis has no answer for** — AutoML search, HPO, data contracts, a model registry with a promotion gate, drift and label-free performance estimation, ONNX export, pipelines;
- **templates and prompt-packs** for all ten adapter pieces, each with its contract, its named trap and its verify command;
- **a fully worked reference domain**, green end to end before the day, so the agent pattern-matches working code rather than empty templates.

### Critical constraint: Aegis already has a serious ML spine

`aegis.ml` is XGBoost + HistGradientBoosting soft-voting, **MAPIE split-conformal** calibrated on a disjoint split, **SHAP TreeExplainer** averaged by member weight, a `ModelCard` separating *requested* from *empirical* coverage, SHA-256 dataset digests, and `MLModelUnavailableError` instead of a silent fallback. `aegis.forecast` is Nixtla StatsForecast with `ConformalIntervals` and rolling-origin backtests.

**`aegis_ml` extends that spine. It never replaces it.** The AutoML search finds a better configuration and hands it back as a portable JSON `Recipe`; the Aegis spine then fits it, keeping conformal calibration, SHAP, the card and the digest intact.

---

## The four traps this package exists to prevent

Each is silent. Each passes every automated check Aegis has.

| # | Trap | Guard |
|---|---|---|
| 1 | **A pure-noise target passes all 14 conformance checks.** There is no check for generator↔latent coupling. The only native signal is `distinct=False` on the last line of `python -m app.ml` — read minutes before the demo. | `aegis_ml.data.latent.assert_learnable`, in **seconds** |
| 2 | **`resolve_spec` silently returns `FALLBACK_SPEC`** — four columns of generated noise — when `FEATURE_NAMES` or `TARGET.name` is missing or misspelled. Nothing raises. | `aegis_ml.contracts.spec` **generates** `ml_spec.py`, so the five names cannot be typoed |
| 3 | **Four `web/` console files carry shipped-domain literals** and are outside the Python-only vocabulary scan. | `prompts/13-console.md` names all four |
| 4 | **`cp -r` leaves the old domain's 3 corpus documents and 2 playbooks behind**, and retrieval serves them. | `rsync -a --delete` / `robocopy /MIR` |

---

## Quickstart

### Both venvs

Two virtualenvs, deliberately. The Aegis backend carries hard caps — `pandas>=2.2,<2.4`, `numpy>=1.26,<2.5`, `numba==0.67.0` — that AutoGluon, TabPFN-2.5 and torch will not resolve under. Install everything; isolate the heavy half.

```bash
# 1 — this package, and the serving-safe tier
cd /Users/yrevash/aegis_ml
uv sync --extra dev
uv run aegis-ml doctor

# 2 — the isolated trainer venv (AutoGluon, TabPFN, torch, SDV)
uv venv .venv-ml --python 3.11
uv pip install --python .venv-ml -e '.[strong,serve]'

# 3 — the serving-safe extra into the Aegis backend venv
uv pip install --python /Users/yrevash/aegis/backend/.venv -e '/Users/yrevash/aegis_ml[serve]'
```

<details>
<summary>Windows PowerShell</summary>

```powershell
Set-Location C:\aegis_ml
uv sync --extra dev
uv run aegis-ml doctor

uv venv .venv-ml --python 3.11
uv pip install --python .venv-ml torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-ml -e ".[strong,serve]"

uv pip install --python C:\aegis\backend\.venv -e "C:\aegis_ml[serve]"
```

Install torch from the CPU index **first**, so AutoGluon resolves against it. `autogluon.tabular` and `autogluon.timeseries` install on Windows; `autogluon.multimodal` does not. Full detail in [`docs/08-windows.md`](docs/08-windows.md).
</details>

The trainer venv is **skippable** — `baseline` and `flaml` run in the serving venv and produce a portable recipe. The strong tiers are then reported as unavailable *with a reason*, which is honest and costs a leaderboard row, not the demo.

### The pipeline

```bash
cd /Users/yrevash/aegis_ml

uv run aegis-ml doctor       # environment, tiers, paths, learnability
uv run aegis-ml contract     # pandera + assert_learnable + leakage — seconds
uv run aegis-ml train --tier all
uv run aegis-ml eval
uv run aegis-ml promote      # the five-criterion gate
uv run aegis-ml drift        # Evidently + NannyML
```

Full CLI: `doctor | init | contract | synth | train | eval | promote | drift | forecast | card | export`, plus `registry list|show|champion|diff|rollback`.

---

## Layout

```
/Users/yrevash/aegis_ml/
├── README.md                     ← you are here
├── finalplan.md                  the plan, with the architecture decisions and research appendix
├── pyproject.toml                extras: [serve] [strong] [mlops] [dev]
│
├── docs/                         00-START-HERE · 01-what-is-aegis · 02-domain-adapter-contract
│                                 03-authoring-a-domain · 04-synthetic-data · 05-ml-pipelines
│                                 06-mlops-registry-drift · 07-integration-with-aegis
│                                 08-windows · 09-troubleshooting · 10-architecture-decisions
│
├── prompts/                      the authoring packs, read on the day
│   ├── 00-intake.md              problem statement → Domain Brief
│   ├── 01-schema.md … 10-skills.md    one per adapter piece
│   ├── 11-ml-pipeline.md  12-integration.md  13-console.md  14-final-gate.md
│   └── CHECKLIST.md              the whole day on one page
│
├── src/aegis_ml/
│   ├── settings.py               pydantic-settings; AEGIS_ML_* env
│   ├── cli.py                    typer
│   ├── contracts/                spec · frames · protocols · errors      (pydantic-only)
│   ├── data/                     synth (SDV) · latent · splits · profile · contract_check
│   ├── features/                 pipeline (skrub) · leakage
│   ├── automl/                   tiers · search · hpo (Optuna) · recipe · runner
│   ├── forecast/                 engine (wraps aegis.forecast) · ml_forecast · foundation · backtest
│   ├── evaluate/                 metrics · cv · calibration · slices · gate
│   ├── explain/                  shap_report · pdp · reason_codes · card
│   ├── registry/                 store · promote · mlflow_mirror · db
│   ├── monitor/                  log · drift (Evidently) · perf (NannyML) · alerts
│   ├── export/                   onnx
│   ├── serve/                    router (FastAPI) · tools (adapter ToolSpecs)
│   └── pipelines/                flows · prefect_shim · manifest
│
├── templates/adapter/            the 10 pieces as annotated skeletons
├── reference/adapter/            a FULL worked domain, green end to end
├── config/                       automl · forecast · monitoring · pipeline · contracts
├── registry_store/               the filesystem model registry
└── tests/
```

---

## Architecture, in one paragraph

`aegis_ml` runs the AutoML search in an **isolated trainer venv** (AutoGluon, TabPFN-2.5, torch, SDV) and returns the winning configuration to the serving venv as a **portable JSON `Recipe`** — an explicit allowlist of tree learners `shap.TreeExplainer` supports, so the **Aegis spine** re-fits it and keeps MAPIE conformal calibration, SHAP attribution, the model card and the dataset digest. That is the keystone: **full AutoML benefit with zero changes inside `aegis/`.** The registry is filesystem-first with an optional MLflow mirror, and promotion atomically replaces `backend/.artifacts/ml_spine.joblib` — the exact path `aegis.ml.get_model()` already loads. Pipelines are plain Python functions with Prefect applied as a decorator when it is importable, so a trained artefact never depends on a server being up. Monitoring pairs **Evidently** (drift, needs labels for performance) with **NannyML CBPE/DLE** (performance estimated *without* labels), and everything the latter produces is spelled `estimated_*` so it can never be read as a measurement. ML reaches the agent through adapter **tools** (`aegis_ml.serve.tools`) rather than a graph node, because the `ml_predict` node the Aegis README describes does not exist — and tools are already gated, audited and streamed to the console, so this needs no core edit.

Rationale, alternatives and sources: [`docs/10-architecture-decisions.md`](docs/10-architecture-decisions.md).

---

## Design discipline, inherited from Aegis

| Rule | Here |
|---|---|
| **Light types, heavy impl** | `aegis_ml.contracts` imports **pydantic and nothing else**, with a test asserting it in a subprocess |
| **Requested vs measured is a naming rule** | Never one field: `requested_coverage` / `empirical_coverage`, `conformal_coverage` / `conformal_coverage_empirical`. NannyML output is `estimated_*` throughout |
| **Refuse rather than degrade** | Nine typed errors, each carrying its remedy. `AutoMLTierUnavailableError` exists so "not installed" and "found nothing better" are never indistinguishable |
| **Optional deps name the exact command** | `aegis_ml._require.require(extra, module)`. Never `except ImportError: pass` |
| **Tooling** | Python 3.11, ruff `E,F,I,UP,B,SIM,ANN,D`, line-length 100, Google docstrings — mirroring `aegis/pyproject.toml` exactly, so one `ruff check` across both trees gives one verdict |

---

## Licence notice — TabPFN-2.5

**TabPFN-2.5 weights are distributed under the Prior Labs License: research and evaluation use are permitted, commercial and production use are NOT.**

The `tabpfn` tier is **on by default**, because at the 1k–10k-row scale this factory generates TabPFN-2.5 has a ~100% win rate against default XGBoost and a hackathon demo is evaluation use. Its score is reported as an **accuracy ceiling** and never promoted as the serving model — it is not portable, because the prediction *is* the pretrained transformer.

The notice is a module constant (`aegis_ml.automl.tiers.TABPFN_LICENSE_NOTICE`), not a docstring, precisely so it is **copied into data** — `Recipe.notes`, `Candidate.detail`, and every model card it touches.

Switch the tier off entirely:

```bash
export AEGIS_ML_ENABLE_TABPFN=0        # bash
$env:AEGIS_ML_ENABLE_TABPFN = "0"      # PowerShell
```

The AutoGluon tier matches it given budget, so nothing collapses if you do.

---

## Documentation

| | |
|---|---|
| [`docs/00-START-HERE.md`](docs/00-START-HERE.md) | **The entry point.** Two-minute orientation, reading order, first three commands |
| [`docs/01-what-is-aegis.md`](docs/01-what-is-aegis.md) | The platform — and **what it already gives you for free**, so you do not rebuild it |
| [`docs/02-domain-adapter-contract.md`](docs/02-domain-adapter-contract.md) | The Protocol in full, the 14 conformance checks by name, the host-bound symbols |
| [`docs/03-authoring-a-domain.md`](docs/03-authoring-a-domain.md) | Problem statement → ten filled pieces, in the right order |
| [`docs/04-synthetic-data.md`](docs/04-synthetic-data.md) | **The most technically important document here** |
| [`docs/05-ml-pipelines.md`](docs/05-ml-pipelines.md) | Flows, AutoML tiers, HPO, the portable recipe, reading the outputs |
| [`docs/06-mlops-registry-drift.md`](docs/06-mlops-registry-drift.md) | Registry, champion/challenger, the five-criterion gate, rollback, drift |
| [`docs/07-integration-with-aegis.md`](docs/07-integration-with-aegis.md) | The nine-step day-of procedure, with exact commands |
| [`docs/08-windows.md`](docs/08-windows.md) | Two venvs on Windows, PowerShell for every command, no Docker or WSL |
| [`docs/09-troubleshooting.md`](docs/09-troubleshooting.md) | Symptom → cause → fix. **If it did not raise, it is in here** |
| [`docs/10-architecture-decisions.md`](docs/10-architecture-decisions.md) | D1–D6 as ADRs, the SOTA tool choices, the risk register, the sources |

Authoring packs: [`prompts/`](prompts/) — one per adapter piece plus intake, ML, integration, console and the final gate. One-page version: [`prompts/CHECKLIST.md`](prompts/CHECKLIST.md).

---

## Verification

```bash
cd /Users/yrevash/aegis_ml && uv sync --extra dev && uv run aegis-ml doctor
uv run pytest tests -q
uv run ruff check src reference tests

# the reference domain satisfies the real contract
uv run python -c "import reference.adapter as a; from aegis.adapter import DomainAdapter, missing_members; \
print('missing:', missing_members(a)); print('satisfies:', isinstance(a, DomainAdapter))"

(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src:/Users/yrevash/aegis_ml \
  .venv/bin/python -m pytest --pyargs aegis.conformance --aegis-adapter reference.adapter -q)

# full pipeline end to end
cd /Users/yrevash/aegis_ml && make demo

# the spine still trains, and the label is learnable — distinct=True is the pass signal
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml | tail -1)
```

**Definition of done:** `make demo` produces a promoted artifact whose measured conformal coverage is inside tolerance, the reference adapter passes all fourteen conformance checks, `python -m app.ml` prints `distinct=True`, and both lockfiles are committed and reproduce on Windows.

> **Do not quote test counts.** `AGENTS.md` and `README.md` in the Aegis repo disagree with each other (2247 vs 2268 core; 1121 vs 1174 backend). Record whatever number you actually get, and use that as your baseline.
