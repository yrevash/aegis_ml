# Known issues — as of 2026-08-24

> Resolved entries are struck through and kept, not deleted: how a defect was found is
> usually more useful than the fact that it is gone.
>
> **Status: 12 of 21 fixed.** Five more issues surfaced while writing the beginner docs
> (#17–21); the two serious ones are fixed.
>
> Previously: 10 of 16 fixed. The 6 remaining are 1 accepted limitation (#7, ONNX, off by
> default), 1 action for the operator (#11, TabPFN token — needs a browser), and 4 defects in
> *Aegis's own* repo (#13–16) recorded so they are not inherited by accident.

Everything here was found by **running the code**, not by reading it. Each entry says what
is wrong, how bad it is, how to reproduce it, and what the fix looks like. Nothing in this
file is fixed yet; it is the to-do list.

Severity: **P1** breaks something a user will hit · **P2** wrong or misleading but survivable
· **P3** cosmetic, stale, or a note for later.

---

## P1 — will be hit

### 1. ~~`doctor` and the tier module disagree about TabPFN~~ — FIXED
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

**Fixed** in `cli.py`: `_tier_report` is now a thin projection of `tiers.tier_status()` and
holds no availability logic of its own. Verified in both venvs — the serving venv reports
both strong tiers as not importable, and the trainer venv correctly reports `skipped tabpfn`
with the weights/token remedy.

### 2. ~~No test suite~~ — FIXED
`tests/` is empty. The agent writing it was stopped before it produced anything, so there
is no regression net at all. Individual modules were verified live by their authors, and
`scripts/run_demo.py` exercises the happy path, but nothing guards against a change
silently breaking a module tomorrow.

**Fixed.** `tests/` now holds 14 files: **314 passed** in ~4.5 min with the Aegis checkout on
`PYTHONPATH` (306 passed + 3 skipped without it — the skips are the three `DomainAdapter`
Protocol checks). Test doubles live only in `tests/fixtures/`, and `tests/test_meta.py` runs
`scripts/audit_no_mocks.py` as a test so a mock reaching `src/` fails the suite.

It immediately earned its keep: a strict-xfail found that `store._validate_run_id` documented
itself as rejecting a run id that would "collide with a shell glob" while only checking for
path escape — `run[0-9]` and `wild*card` passed. Not an escape hole, but a run directory named
`wild*card` makes `rm -rf runs/<id>` mean something other than what it reads like. The charset
check now exists and the xfail is a live regression test.

### 3. ~~`reference/` is incomplete~~ — FIXED
Present: `problem.py`, `adapter/{__init__,schema,ml_spec,generator}.py`. **Missing:**
`tools.py`, `personas.py`, `prompts.py`, `memory_spec.py`, `roster.py`, `corpus/__init__.py`,
the 3 corpus documents, the 2 skill playbooks, and `reference/README.md`.

**Fixed.** All ten pieces are present. Verified: `missing_members()` returns `[]`,
`isinstance(reference.adapter, DomainAdapter)` is True, `resolve_spec` resolves to the domain
rather than `FALLBACK_SPEC`, learnability R² 0.6236 (in band, not suspiciously easy), and
**`pytest --pyargs aegis.conformance --aegis-adapter reference.adapter` → 14 passed**.

---

## P2 — wrong or misleading

### 4. ~~`AEGIS_ML_TABPFN` is not the env var~~ — FIXED
The correct name is **`AEGIS_ML_ENABLE_TABPFN`** — `Settings` uses `env_prefix="AEGIS_ML_"`
over the field `enable_tabpfn`. `finalplan.md` (decision D6) and one line of `RESOLUTION.md`
name the wrong variable. `tiers.TABPFN_LICENSE_NOTICE` and the `docs/` tree already have it
right.

**Fixed** in `finalplan.md` (D6 and the risk register). `RESOLUTION.md` and `docs/` were
already correct. Verified: `grep -rn 'AEGIS_ML_TABPFN[^_]' .` returns nothing.

### 5. ~~`docs/` was written against a mid-flight tree~~ — FIXED
The documentation agent read the repo while other agents were still writing it, and several
docs describe `cli.py`, `data/`, `features/`, `explain/`, `registry/`, `monitor/`, `export/`,
`serve/`, `pipelines/`, `forecast/` as *planned but not present*. They are all present and
full now. The agent documented them to the signatures in `finalplan.md`, which mostly match
what was built — but "mostly" is not "verified".

**Fix:** one pass over `docs/03`, `docs/05`, `docs/06`, `docs/07` checking every named
symbol against the real module. Anything that says "not yet present" is stale.

### 6. ~~`data/synth.py` has never been executed~~ — FIXED
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

### 8. ~~Three submodules are shadowed by function re-exports~~ — FIXED
`data.profile`, `automl.search` and `explain.reason_codes` are each a module whose package
`__init__` re-exports a function of the same name, so `aegis_ml.data.profile` resolves to
the function and `from aegis_ml.data.profile import ...` breaks. This caused two real
crashes during development. Call sites now import via the full module path with a comment,
but the trap is still armed for the next caller.

**Fix:** rename either the function or the module in each pair. `profile_frame`,
`run_search`, `build_reason_codes` would all read fine.

### 9. ~~`registry/db.py` cannot be exercised in the serving venv~~ — FIXED
It needs `sqlalchemy[asyncio]` (which needs `greenlet`) and `aiosqlite` for tests; neither is
in `.venv`. It was verified live against a real async SQLite database by installing them
temporarily, then uninstalling to leave the venv as found.

**Fix:** add both to the `[dev]` extra so the module is testable by default.

### 10. ~~`eval_flow` with no frame is in-sample~~ — FIXED
Re-scoring a registered run without supplying fresh data uses the run's whole reference
frame, training rows included. It is now labelled `IN-SAMPLE` in the output and explicitly
not presented as generalisation evidence — but the default is still the misleading one.

**Fixed.** `eval_flow` now takes `allow_in_sample: bool = False` and raises
`InSampleEvaluationError` otherwise; `aegis-ml eval` gained `--allow-in-sample`. Verified:
refusal exits 1, opt-in exits 0 and reports `r2=0.7587` **labelled IN-SAMPLE** against the
honest held-out `0.7224` — which is precisely the optimism the default now refuses to
produce silently.

---

## Found while writing the beginner docs — all fixed

### 17. ~~`config/*.toml` was read by nothing~~ — FIXED
All five files opened with "Read by `aegis_ml.<module>`" and **no code read them** — there
was no `tomllib` import anywhere in the package. Editing `automl.toml`'s `time_budget` on
hackathon morning would have changed nothing, silently. Exactly the class of no-op this
project exists to eliminate, sitting in the config directory the whole time.

**Fixed**: `aegis_ml/config.py` loads them into `Settings` via an explicit key→field table,
layered *beneath* the environment (`AEGIS_ML_*` > TOML > field default). 16 settings now
load from the files. The mapping is written out rather than derived, so a typo'd section
cannot be absorbed as "a key for a setting that doesn't exist yet" — and `unknown_keys()`
reports the 19 keys nothing consumes, which `aegis-ml doctor` prints under
**NOT CONSUMED … editing them does nothing**.

### 18. ~~Two realism bands disagreed~~ — FIXED
`config/contracts.toml` said classification accuracy `[0.65, 0.88]`; `flows.REALISM_ACCURACY_BAND`
— what `doctor`, `data_flow` and the charts actually used — said `(0.62, 0.92)`. Harmless
while the TOML was inert; a behaviour change the moment it was wired up.

**Fixed**: `realism_band_for` and `doctor` now read `settings.realism_r2_band` /
`realism_accuracy_band`; the module constants remain their defaults. The TOML was aligned to
the values that were genuinely running, so wiring it up changed nothing.

### 19. `aegis-ml` console script cannot import a cwd-relative adapter — OPEN
`aegis-ml contract --adapter reference.problem` fails in an `importlib` traceback because the
entry point does not put the working directory on `sys.path`. `python -m aegis_ml.cli` works,
because `-m` adds it. Documented in `docs/learn/08-your-first-run.md §0` and its troubleshooting
table. **Fix**: prepend `Path.cwd()` to `sys.path` in the CLI entry point.

### 20. `visuals/manifest.json` mislabels the leaderboard best — OPEN
The leaderboard slot labels `extra_trees` (the *lowest* score) as `"best"` and derives a
negative `ceiling_gap` from it. The rendered chart is correct; only that JSON field is wrong.
Cosmetic unless something downstream trusts the manifest.

### 21. `RUN_SUMMARY.md` labels the wrong pair "realism band" — OPEN
It prints `[0.15, 0.95]` — the *learnability guard* bounds (`learnable_r2_floor` and
`latent.R2_CEILING`) — under the heading "realism band", while the console output for the same
run prints `[0.45, 0.80]` from `realism_band_for`. Two different bands, one label.

---

## P3 — notes, stale prose, upstream

### 11. TabPFN needs one-time browser setup
`pip install tabpfn` succeeds, `import tabpfn` succeeds, and `.fit()` raises
`TabPFNLicenseError`. Prior Labs gates weight download behind licence acceptance plus a
`TABPFN_TOKEN`. Fully documented in `RESOLUTION.md`; the tier now reports it instead of
crashing. **Needs a browser and a network — do it before hackathon day, not on it.**

### 12. ~~The lightgbm ↔ nannyml ↔ scikit-learn triangle~~ — HANDLED
`nannyml` pins `lightgbm<4.6`; lightgbm `<4.6`'s *sklearn wrapper* is broken against
sklearn ≥1.8 (`check_X_y(force_all_finite=)`, removed in 1.8). Measured: NannyML's DLE works,
FLAML with `lgbm` works (both use LightGBM's native API), only a direct
`LGBMRegressor(...).fit(...)` raises. **Was still killing the FLAML tier**, because `is_portable_kind` checked only importability
and `_search_flaml` builds its `estimator_list` from that predicate — so FLAML accepted
`lgbm` and died partway through with a keyword-argument error. Observed in a demo run.

**Fixed**: `is_portable_kind` now also requires that the estimator can be *fitted* here, via
a cached two-row probe (`recipe._wrapper_usable`). Version arithmetic across three packages
would not have answered the actual question; a fit does. LightGBM is now correctly excluded,
FLAML runs clean, and in the latest demo the FLAML tier **wins** at r²=0.7328. Full analysis
in `RESOLUTION.md`.

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
| Test suite | **315 passed**, 0 failed, 0 xfail |
| Per-run visuals | 9 PNGs + `index.html` (0 external refs) + `interactive.html`, written automatically by the `visuals` stage in `train_flow` and `drift_flow` |
| Held-out split provenance | recovered and **verified**: re-scoring the persisted model on the recovered rows reproduces the registered r²=0.722406014223 exactly; a wrong seed gives 0.7805 and is rejected |
| Trainer-venv subprocess bridge | real child process, identical result in-process vs cross-process; crash path surfaces the traceback |
