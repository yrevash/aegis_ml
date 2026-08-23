# Known issues — open, as of 2026-08-24

Everything here was found by **running the code**, not by reading it. Each entry says what
is wrong, how bad it is, how to reproduce it, and what the fix looks like. Nothing in this
file is fixed yet; it is the to-do list.

Severity: **P1** breaks something a user will hit · **P2** wrong or misleading but survivable
· **P3** cosmetic, stale, or a note for later.

---

## P1 — will be hit

### 1. `doctor` and the tier module disagree about TabPFN
`src/aegis_ml/cli.py:190` (`_tier_report`) reimplements tier availability instead of calling
`aegis_ml.automl.tiers.tier_status()`. It checks only importability, so it misses the
weights/token gate added to `tiers.unavailable_reason`.

Reproduce, in the trainer venv where `tabpfn` is installed but has no weights:
```
$ .venv-ml/bin/python -m aegis_ml.cli doctor | grep tabpfn
  RUNS     tabpfn       tabpfn 8.4.0                     # ← wrong

$ .venv-ml/bin/python -c "from aegis_ml.automl.tiers import unavailable_reason; print(unavailable_reason('tabpfn'))"
  importable, but no model weights are available and TABPFN_TOKEN is unset ...   # ← right
```
Two sources of truth for the same question, and the wrong one is the one a human reads
first on hackathon morning.

**Fix:** delete `_tier_report` and render from `tiers.tier_status()`. One function, one
answer. Keep the settings-flag distinction the CLI currently makes — `tier_status` already
reports it.

### 2. No test suite
`tests/` is empty. The agent writing it was stopped before it produced anything, so there
is no regression net at all. Individual modules were verified live by their authors, and
`scripts/run_demo.py` exercises the happy path, but nothing guards against a change
silently breaking a module tomorrow.

**Fix:** write it. Highest-value tests, in order: the dep-free guarantee for
`aegis_ml.contracts` (subprocess, assert no pandas/numpy/sklearn/torch in `sys.modules`);
the realism band and both learnability guards; recipe JSON round-trip and portability
refusal; metric direction correctness in `HIGHER_IS_BETTER`; the promotion gate's five
criteria; registry promote/rollback atomicity against `tmp_path`.

### 3. `reference/` is incomplete
Present: `problem.py`, `adapter/{__init__,schema,ml_spec,generator}.py`. **Missing:**
`tools.py`, `personas.py`, `prompts.py`, `memory_spec.py`, `roster.py`, `corpus/__init__.py`,
the 3 corpus documents, the 2 skill playbooks, and `reference/README.md`.

Until those exist, the reference domain cannot satisfy `aegis.adapter.DomainAdapter` and
`pytest --pyargs aegis.conformance --aegis-adapter reference.adapter` cannot pass — which
is the single claim the reference domain exists to make.

---

## P2 — wrong or misleading

### 4. `AEGIS_ML_TABPFN` is not the env var
The correct name is **`AEGIS_ML_ENABLE_TABPFN`** — `Settings` uses `env_prefix="AEGIS_ML_"`
over the field `enable_tabpfn`. `finalplan.md` (decision D6) and one line of `RESOLUTION.md`
name the wrong variable. `tiers.TABPFN_LICENSE_NOTICE` and the `docs/` tree already have it
right.

**Fix:** correct D6 in `finalplan.md` and the TabPFN section of `RESOLUTION.md`.

### 5. `docs/` was written against a mid-flight tree
The documentation agent read the repo while other agents were still writing it, and several
docs describe `cli.py`, `data/`, `features/`, `explain/`, `registry/`, `monitor/`, `export/`,
`serve/`, `pipelines/`, `forecast/` as *planned but not present*. They are all present and
full now. The agent documented them to the signatures in `finalplan.md`, which mostly match
what was built — but "mostly" is not "verified".

**Fix:** one pass over `docs/03`, `docs/05`, `docs/06`, `docs/07` checking every named
symbol against the real module. Anything that says "not yet present" is stale.

### 6. `data/synth.py` has never been executed
Written against SDV 1.37 APIs (`Metadata.detect_from_dataframe`, `update_column`,
`sdv.single_table.*Synthesizer`, `evaluate_quality`, `sdmetrics.single_table.NewRowSynthesis`)
while SDV was not installed. SDV **1.38.1** is now in `.venv-ml`, and 1.38 ≠ 1.37 — the
signatures are plausible but unproven.

**Fix:** run it against `.venv-ml` on a real frame. It is the one module in the package with
no execution evidence behind it.

### 7. ONNX export has two real limits
Both found by measurement, both now documented in the module docstring, neither fixed:
- **Learned NaN routing does not survive conversion.** The same RandomForest measured
  `2.4e-06` max abs difference on complete rows and **`17.7`** on rows containing NaNs.
  Since the synthetic data deliberately carries ~4% MAR missingness, this is not a corner case.
- **`HistGradientBoosting` will not convert** on skl2onnx 1.20 + current onnx — `TypeError`
  inside the TreeEnsemble attributes. `to_onnx` catches and re-raises with context.

**Fix (or accept):** ONNX is off by default in `config/pipeline.toml` (`export_onnx = false`)
and nothing in Aegis serves ONNX, so this is currently a documented limitation rather than a
blocker. If it is ever turned on, the NaN behaviour must be surfaced on the model card.

### 8. Three submodules are shadowed by function re-exports
`data.profile`, `automl.search` and `explain.reason_codes` are each a module whose package
`__init__` re-exports a function of the same name, so `aegis_ml.data.profile` resolves to
the function and `from aegis_ml.data.profile import ...` breaks. This caused two real
crashes during development. Call sites now import via the full module path with a comment,
but the trap is still armed for the next caller.

**Fix:** rename either the function or the module in each pair. `profile_frame`,
`run_search`, `build_reason_codes` would all read fine.

### 9. `registry/db.py` cannot be exercised in the serving venv
It needs `sqlalchemy[asyncio]` (which needs `greenlet`) and `aiosqlite` for tests; neither is
in `.venv`. It was verified live against a real async SQLite database by installing them
temporarily, then uninstalling to leave the venv as found.

**Fix:** add both to the `[dev]` extra so the module is testable by default.

### 10. `eval_flow` with no frame is in-sample
Re-scoring a registered run without supplying fresh data uses the run's whole reference
frame, training rows included. It is now labelled `IN-SAMPLE` in the output and explicitly
not presented as generalisation evidence — but the default is still the misleading one.

**Fix:** consider requiring an explicit `--allow-in-sample` flag rather than labelling
after the fact.

---

## P3 — notes, stale prose, upstream

### 11. TabPFN needs one-time browser setup
`pip install tabpfn` succeeds, `import tabpfn` succeeds, and `.fit()` raises
`TabPFNLicenseError`. Prior Labs gates weight download behind licence acceptance plus a
`TABPFN_TOKEN`. Fully documented in `RESOLUTION.md`; the tier now reports it instead of
crashing. **Needs a browser and a network — do it before hackathon day, not on it.**

### 12. The lightgbm ↔ nannyml ↔ scikit-learn triangle
`nannyml` pins `lightgbm<4.6`; lightgbm `<4.6`'s *sklearn wrapper* is broken against
sklearn ≥1.8 (`check_X_y(force_all_finite=)`, removed in 1.8). Measured: NannyML's DLE works,
FLAML with `lgbm` works (both use LightGBM's native API), only a direct
`LGBMRegressor(...).fit(...)` raises. Resolved by design — `is_portable_kind` drops that
member with its reason recorded. Full analysis in `RESOLUTION.md`.

### 13. Conformance-check numbering is ambiguous
Aegis has **14 test functions** grouped under **11 section headers**, and `AGENTS.md` refers
to the vocabulary check as "check 11" (its section) while it is test function #14. Quote the
**test name**, never the ordinal. Some of our docs still use ordinals.

### 14. Aegis's own console prose is stale
`web/src/config/personas.ts` argues at length that no sample query reaches the human gate and
that fixing it "needs a read-side tool in `backend/src/app/adapter/tools.py` (a LOW-risk
`find_requests`)". That tool **now exists** — the reference adapter has 4 tools, not 3.
`prompts/13-console.md` turns this into an opportunity rather than a correction.

### 15. Aegis's own reference generator is too easy
`backend/src/app/adapter/generator.py` uses a flat `noise_scale=4.0` against a latent signal
spreading ~30 hours, which lands around **R² 0.97–0.98**. That is the "label is a closed-form
function of the inputs" failure this package exists to prevent, present in the code we pattern-
matched from. Our templates and reference domain use calibrated sigma instead. Not our bug,
but worth knowing before copying anything from it.

### 16. Aegis's own docs disagree with themselves
`README.md` says the core is "30 packages", `AGENTS.md` says "~27 subpackages"; test counts
differ too (2247 vs 2268, 1121 vs 1174). Do not quote either — record what you actually get.

---

## What IS verified

For contrast, so this file is not read as a verdict on the whole package:

| Claim | Evidence |
|---|---|
| Both dependency tiers resolve | `requirements-serve.lock.txt` (153), `requirements-strong.lock.txt` (1174); `pandas`/`numpy`/`sklearn` identical across both |
| `ruff check src` | 0 errors across 59 modules |
| No mocks/stubs/empty bodies in `src/` | `scripts/audit_no_mocks.py` PASS, 14 reviewed opt-outs |
| Every module imports | 52/52 |
| `aegis-ml doctor` | exits 0 in both venvs |
| End-to-end `full_flow` | r² **0.6391** on 280 held-out rows; empirical conformal coverage **93.57%** vs 90% requested; gate 5/5; artifact promoted; drift on shifted frame → `block`; 13 artifacts + `RUN_SUMMARY.md` |
| Data realism | R² 0.656 / 0.588 / 0.6391, accuracy 0.743 — inside the 0.45–0.80 band, never above it |
| Learnability guards | noise target → `LabelNotLearnableError`; deterministic target → R² 0.994 flagged `suspiciously_easy` |
| AutoGluon | R² 0.5411, 10 models, 20 s |
| Evidently drift | stable → 0.000 share `pass`; shifted → 0.429 share `block`, correct columns named |
| NannyML label-free estimate | `estimated_rmse = 2.07 [1.71, 2.44]` |
| Optuna HPO | r² 0.6158 → 0.6556 in 12 trials; SQLite study resumes |
| Trainer-venv subprocess bridge | real child process, identical result in-process vs cross-process; crash path surfaces the traceback |
