# 08 · Windows

Everything in this repository runs natively on Windows. **No Docker, no WSL, no GPU.** That is a deliberate Aegis constraint, not a limitation to work around, and it is why Redis is Memurai and every store is a native local install.

Paths in this document assume `C:\aegis` and `C:\aegis_ml`. Adjust if your checkouts differ.

---

## 1. Prerequisites

| Tool | Get it |
|---|---|
| Python 3.11 | `winget install Python.Python.3.11` — or the python.org installer. **3.11 specifically**; Aegis targets it. |
| `uv` | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Node 20+ | `winget install OpenJS.NodeJS.LTS` |
| Git | `winget install Git.Git` |
| Build tools | Visual Studio Build Tools with the C++ workload. Some wheels still compile. |

Aegis's own installer handles the rest:

```powershell
Set-Location C:\aegis
.\scripts\install-windows.ps1
```

Elevated PowerShell. Installs the toolchain, the native stores, `backend\.venv` with every extra, and the console's npm dependencies. Idempotent — re-run it if it stops on a missing prerequisite. Full detail in `C:\aegis\INSTALL.md`.

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 2. The two-venv setup

### 2.1 Why the split exists

`C:\aegis\backend\pyproject.toml` carries hard caps that the whole backend depends on:

| Cap | Imposed by |
|---|---|
| `pandas>=2.2,<2.4` | nemoguardrails |
| `numpy>=1.26,<2.5` | presidio-analyzer 2.2.364 declares `numpy<2.5`; numba/llvmlite (a shap dependency) have no release supporting numpy 2.5 — without the cap the resolver drags numba back to an ancient release that cannot build against numpy 2.4 |
| `numba==0.67.0` | `[tool.uv] constraint-dependencies` |
| `litellm==1.96.0` | 1.96.2 regressed the gateway path |
| `presidio-analyzer==2.2.364` | a free resolve back-solves to the older 2.2.362 around pydantic/numpy |

AutoGluon 1.6 + TabPFN-2.5 + torch will not resolve cleanly inside that. **Installing them into `backend\.venv` is the single most likely way to lose the morning.** So: install everything, isolate the heavy half.

| Venv | Path | Contents |
|---|---|---|
| **serving** | `C:\aegis\backend\.venv` (already exists) | the backend, plus `aegis-ml[serve]` — pandas, numpy, sklearn, joblib, xgboost, mapie, shap, pandera, skrub, optuna, flaml, evidently, nannyml, pyarrow. All resolvable under the caps. |
| **trainer** | `C:\aegis_ml\.venv-ml` (you create it) | `aegis-ml[strong,serve]` — autogluon.tabular, autogluon.timeseries, tabpfn, tabpfn-extensions, torch (CPU), sdv, mlforecast, lightgbm, catboost. Unconstrained resolve. |

They never share an interpreter. The bridge between them is a **portable JSON `Recipe`** (`docs/05-ml-pipelines.md` §4).

### 2.2 Create the ML package venv

```powershell
Set-Location C:\aegis_ml
uv sync --extra dev
uv run aegis-ml doctor
```

### 2.3 Create the trainer venv

```powershell
Set-Location C:\aegis_ml
uv venv .venv-ml --python 3.11
uv pip install --python .venv-ml -e ".[strong,serve]"
```

**torch CPU wheels.** `uv` resolves the default PyPI torch build, which on Windows is CPU-only unless you asked for CUDA. If it tries to pull a large CUDA build, pin the CPU index explicitly:

```powershell
uv pip install --python .venv-ml torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-ml -e ".[strong,serve]"
```

Install torch **first**, then the rest, so AutoGluon resolves against the torch already present rather than dragging in another.

**AutoGluon on Windows.** `autogluon.tabular` and `autogluon.timeseries` install and run. **`autogluon.multimodal` does not** — do not install it, and do not add it to any extra. Python 3.10–3.13 is supported; use 3.11 to match everything else.

### 2.4 Install `aegis-ml[serve]` into the backend venv

Needed so the backend suite can run `assert_learnable` and so the ML adapter tools import:

```powershell
uv pip install --python C:\aegis\backend\.venv -e "C:\aegis_ml[serve]"
```

Then confirm nothing moved:

```powershell
C:\aegis\backend\.venv\Scripts\python.exe -c "import pandas, numpy, numba; print(pandas.__version__, numpy.__version__, numba.__version__)"
```

Expect pandas `2.2.x`–`2.3.x`, numpy `<2.5`, numba `0.67.0`. Anything outside that means the caps were violated — see `docs/09-troubleshooting.md`.

### 2.5 Where the interpreters are

`aegis_ml.settings.trainer_python` handles both layouts:

```python
win = self.trainer_venv / "Scripts" / "python.exe"
return win if win.exists() else self.trainer_venv / "bin" / "python"
```

So `C:\aegis_ml\.venv-ml\Scripts\python.exe` is found automatically. `AEGIS_ML_TRAINER_VENV` overrides the location.

---

## 3. PowerShell equivalents of every command

### 3.1 The `cd`-in-a-subshell idiom

Every command in `AGENTS.md` and `SKILL.md` is written from the Aegis repository root with each `cd` wrapped in a POSIX subshell `( ... )`, so a run of them one after another in one terminal works. PowerShell has no `( ... )` subshell for this; use `Push-Location` / `Pop-Location`.

```powershell
Push-Location C:\aegis\backend
$env:PYTHONPATH = "src;..\aegis\src"
# ... commands ...
Pop-Location
```

> **`PYTHONPATH` uses `;` on Windows, not `:`.** This is the single most common transcription error. `PYTHONPATH=src:../aegis/src` becomes `$env:PYTHONPATH = "src;..\aegis\src"`.

`$env:PYTHONPATH` persists for the session. Clear it when you are done: `Remove-Item Env:\PYTHONPATH`.

### 3.2 Command-by-command translation

| Purpose | bash | PowerShell |
|---|---|---|
| Bootstrap Aegis | `./scripts/bootstrap.sh` | `.\scripts\install-windows.ps1` |
| Start (full) | `./scripts/dev-native.sh` | `.\scripts\start.ps1 -Mode full` |
| Console dev server | `cd web && npm run dev` | `Set-Location C:\aegis\web; npm run dev` |
| Seed accounts | `(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.seed)` | see §3.3 |
| Adapter contract check | see `docs/02` §4 | see §3.3 |
| Conformance suite | see `docs/02` §4 | see §3.3 |
| Backend suite | `(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest -q)` | see §3.3 |
| Core suite | `(cd aegis && PYTHONPATH=src ../backend/.venv/bin/python -m pytest -q)` | see §3.3 |
| Lint | `backend/.venv/bin/python -m ruff check aegis backend` | `C:\aegis\backend\.venv\Scripts\python.exe -m ruff check C:\aegis\aegis C:\aegis\backend` |
| Train the spine | `(cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml)` | see §3.3 |
| OpenAPI snapshot | `backend/.venv/bin/python scripts/build_openapi.py` | `C:\aegis\backend\.venv\Scripts\python.exe C:\aegis\scripts\build_openapi.py` |
| TS client | `(cd web && npm run gen:api)` | `Set-Location C:\aegis\web; npm run gen:api` |
| Console build/test | `(cd web && npx tsc --noEmit && npm test && npx next build)` | `Set-Location C:\aegis\web; npx tsc --noEmit; npm test; npx next build` |
| Sync the adapter | `rsync -a --delete SRC/ DST/` | `robocopy SRC DST /MIR /NFL /NDL /NJH /NJS` |
| Clear `__pycache__` | `find DIR -name '__pycache__' -type d -exec rm -rf {} +` | `Get-ChildItem DIR -Recurse -Directory -Filter __pycache__ \| Remove-Item -Recurse -Force` |
| Grep the console | `grep -rn "pat" src/` | `Select-String -Path src\* -Recurse -Pattern "pat"` |
| Set an env var | `export AEGIS_ML_ENABLE_TABPFN=0` | `$env:AEGIS_ML_ENABLE_TABPFN = "0"` |
| Tail the last line | `... \| tail -1` | `... \| Select-Object -Last 1` |

### 3.3 The block you will paste most

```powershell
Push-Location C:\aegis\backend
$env:PYTHONPATH = "src;..\aegis\src"

# seed the demo accounts
.\.venv\Scripts\python.exe -m app.seed

# the adapter contract
.\.venv\Scripts\python.exe -c "import app.adapter; from aegis.adapter import DomainAdapter, missing_members; print('missing:', missing_members(app.adapter)); print('satisfies:', isinstance(app.adapter, DomainAdapter))"

# the fourteen conformance checks
.\.venv\Scripts\python.exe -m pytest --pyargs aegis.conformance --aegis-adapter app.adapter -q

# the fast loop
.\.venv\Scripts\python.exe -m pytest tests\adapter tests\agent -q

# the full backend suite
.\.venv\Scripts\python.exe -m pytest -q

# train the spine; read the last line for distinct=True
.\.venv\Scripts\python.exe -m app.ml

Pop-Location

# the core suite, run with the backend's interpreter
Push-Location C:\aegis\aegis
$env:PYTHONPATH = "src"
C:\aegis\backend\.venv\Scripts\python.exe -m pytest -q
Pop-Location

Remove-Item Env:\PYTHONPATH
```

### 3.4 The `aegis-ml` CLI

Identical on both platforms — `uv run` handles the interpreter.

```powershell
Set-Location C:\aegis_ml
uv run aegis-ml doctor
uv run aegis-ml contract
uv run aegis-ml train --tier all
uv run aegis-ml eval
uv run aegis-ml promote
uv run aegis-ml drift
```

---

## 4. Environment variables

```powershell
# point aegis_ml at the Aegis checkout (default: the sibling directory ../aegis)
$env:AEGIS_ML_AEGIS_ROOT = "C:\aegis"

# relocate the registry
$env:AEGIS_ML_REGISTRY_DIR = "C:\aegis_ml\registry_store"

# the trainer venv
$env:AEGIS_ML_TRAINER_VENV = "C:\aegis_ml\.venv-ml"

# switch off a tier by policy (distinct from "not installed")
$env:AEGIS_ML_ENABLE_TABPFN = "0"
$env:AEGIS_ML_ENABLE_AUTOGLUON = "0"

# optional mirrors
$env:AEGIS_ML_ENABLE_MLFLOW = "1"
$env:AEGIS_ML_POSTGRES_DSN = "postgresql+asyncpg://..."

# the conformance suite's adapter, as an alternative to --aegis-adapter
$env:AEGIS_ADAPTER = "app.adapter"
```

Prefix is `AEGIS_ML_` for everything in `aegis_ml.settings.Settings`, over the **field name**. So the field `enable_tabpfn` is `AEGIS_ML_ENABLE_TABPFN` — **not** `AEGIS_ML_TABPFN`.

To make one persist across sessions:

```powershell
[Environment]::SetEnvironmentVariable("AEGIS_ML_AEGIS_ROOT", "C:\aegis", "User")
```

---

## 5. Native stores — no Docker

| Store | Windows | Notes |
|---|---|---|
| Postgres | native installer | The `full` run mode wants it. `lite` uses SQLite for audit. |
| **Redis** | **Memurai** | Same wire protocol, same port, **no config change**. `winget install Memurai.MemuraiDeveloper` |
| Neo4j | Neo4j Desktop or the Windows service | |
| Qdrant | the Windows binary release | |
| Phoenix | `pip`-installed, runs in-process | |

`.\scripts\install-windows.ps1` sets these up. `.\scripts\preflight.ps1` checks them before a run.

**You do not need any of them to do the work in this repository.** The conformance suite, the adapter contract check, `tests/adapter`, `tests/agent`, `python -m app.ml` and every `aegis-ml` command run with no database, no Redis and no key. That is deliberate — it is what makes the loop fast enough to run after every step.

---

## 6. Windows-specific gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: app` | `PYTHONPATH` used `:` instead of `;` | `$env:PYTHONPATH = "src;..\aegis\src"` |
| `cd: no such file or directory` after chaining commands | The POSIX subshell idiom does not exist in PowerShell | Use `Push-Location` / `Pop-Location` |
| Path too long during `uv pip install` | AutoGluon's dependency tree is deep | Enable long paths: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`, then reboot |
| torch pulls a multi-GB CUDA build | Default index | `--index-url https://download.pytorch.org/whl/cpu`, installed first |
| `autogluon.multimodal` fails to build | Unsupported on Windows | Do not install it. `tabular` and `timeseries` only. |
| A wheel tries to compile and fails | No C++ toolchain | Install Visual Studio Build Tools with the C++ workload |
| `robocopy` exits with code 1 and the shell treats it as failure | robocopy uses 0–7 for success | Check `$LASTEXITCODE -lt 8`, or append `; if ($LASTEXITCODE -lt 8) { $global:LASTEXITCODE = 0 }` |
| Line endings churn the diff | `core.autocrlf` | `.gitattributes` is committed in the Aegis repo — leave it alone |
| `Select-String` finds nothing that `grep` found | Different regex dialect and default recursion | Use `-Path src\* -Recurse` and `-Pattern` with a .NET regex |
| The console shows stale types after a backend change | The TS client is generated, not live | `npm run gen:api` |

---

## 7. Pre-hackathon verification on the Windows box

Do this **before** the day. The two-venv resolve has never been done under these caps until you do it.

```powershell
# 1 — both venvs resolve, and both lockfiles are committed
Set-Location C:\aegis_ml
uv sync --extra dev
uv venv .venv-ml --python 3.11
uv pip install --python .venv-ml torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-ml -e ".[strong,serve]"
uv run aegis-ml doctor

# 2 — the caps in the backend venv survived installing aegis-ml[serve]
uv pip install --python C:\aegis\backend\.venv -e "C:\aegis_ml[serve]"
C:\aegis\backend\.venv\Scripts\python.exe -c "import pandas, numpy, numba, shap, mapie, xgboost; print(pandas.__version__, numpy.__version__, numba.__version__)"

# 3 — the reference domain is green end to end
Set-Location C:\aegis_ml
uv run pytest tests -q

# 4 — the Aegis suites are at their baseline. WRITE THE NUMBERS DOWN.
Push-Location C:\aegis\backend
$env:PYTHONPATH = "src;..\aegis\src"
.\.venv\Scripts\python.exe -m pytest -q
Pop-Location
```

If step 1 or 2 fails, that is the finding, and it is worth more than anything else you could do that day. **Commit both lockfiles once they resolve** — that is the difference between a five-minute setup and a lost morning.

---

## 8. Next

`docs/09-troubleshooting.md`.
