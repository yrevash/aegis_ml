# Session log — building `aegis_ml`

Handoff context for the next agent. Written 2026-08-24 by the session that built this
package from an empty directory. Read this before changing anything.

---

## 1. What this repo is, and why it exists

The user is entering a hackathon. **The problem statement is not known until the day.** They
have an existing agentic-AI platform, **Aegis**, at `/Users/yrevash/aegis`, which is
retargeted to a new domain by writing exactly one thing: a *domain adapter* satisfying
`aegis.adapter.DomainAdapter` — 11 members across 10 pieces.

`aegis_ml` is the base they hand to a coding agent on the day: the ML/MLOps machinery Aegis
lacks, templates for all ten adapter pieces, teaching + procedural docs, and a **fully
worked reference domain** proving the whole chain green in advance.

The starting point was a plan written by another model (DeepSeek), preserved in
`.deepseek-archive/`. It was unusually accurate about Aegis — 26 of 26 line counts exact —
but had three substantive errors, corrected in `finalplan.md`. The most expensive:

> It claimed the Aegis conformance suite checks that the generator's labels come from the
> ml_spec's latent function. **It does not.** There are zero references to the generator in
> `test_conformance.py`. A target that is pure noise passes all 14 checks. The only native
> signal is `distinct=False` on the last line of `python -m app.ml`.

That gap is why `aegis_ml.data.latent.assert_learnable` exists.

---

## 2. Architecture — the decisions that constrain everything else

Full rationale in `finalplan.md` (D1–D6) and `docs/10-architecture-decisions.md`.

**D1 — two virtualenvs, one portable recipe.** The Aegis backend carries hard caps
(`pandas<2.4`, `numpy<2.5`, `numba==0.67.0`) that AutoGluon/TabPFN/torch cannot satisfy. So
the AutoML *search* runs in an isolated `.venv-ml` and its answer crosses back as a JSON
`Recipe`, which the serving venv re-fits — keeping Aegis's MAPIE conformal calibration, SHAP
and ModelCard. This is the keystone; do not casually collapse the two venvs.

Both tiers resolve to **identical** `pandas 2.3.3 / numpy 2.4.6 / scikit-learn 1.9.0`, which
is what makes recipe portability sound. Verified, not assumed — see `RESOLUTION.md`.

**D2 — ML reaches the agent through tools, not the graph.** `describe_prediction` had zero
consumers in Aegis and the `ml_predict` node its README describes does not exist in
`graph.py`. Rather than edit core, `aegis_ml.serve.tools` ships five LOW-risk read-only tool
specs that drop into an adapter's `TOOL_REGISTRY`. Aegis's rule holds: *ML informs, it never
gates; the human gate fires on a tool's risk tier.*

**D3 — filesystem registry is the source of truth.** Promotion atomically replaces
`backend/.artifacts/ml_spine.joblib`, the file `aegis.ml.get_model()` already loads. **Zero
Aegis core changes.** MLflow is an optional mirror only.

**D4 — pipelines are plain Python; Prefect is a decorator.** A trained artifact must never
depend on a server being up.

**D6 — TabPFN is enabled but licence-gated.** Prior Labs License: research/evaluation yes,
commercial/production no. See §6.

---

## 3. Repo map

```
src/aegis_ml/
  contracts/   pydantic-only, dep-free (enforced by a subprocess test)
  data/        latent function + realism, splits, profiling, SDV, contract check
  features/    ColumnTransformer (must mirror Aegis's), skrub, leakage detection
  automl/      4 tiers, Optuna HPO, the portable Recipe, trainer-venv subprocess bridge
  evaluate/    metrics, CV, calibration, slices, the 5-criterion promotion gate
  explain/     SHAP, PDP, reason codes, extended model card
  registry/    filesystem store, promote/rollback, optional MLflow + Postgres
  monitor/     prediction log, Evidently drift, NannyML label-free estimation, alerts
  report/      per-run visuals: 12 figure functions, bundle, self-contained index.html
  dashboard/   `aegis-ml dashboard` — hub page + MLflow UI + Optuna Dashboard, all local
  forecast/    thin wrapper over aegis.forecast + mlforecast candidates
  serve/       FastAPI router + the five ML adapter tools
  pipelines/   7 flows, stage graph with caching/resume, Prefect shim
  export/      ONNX (off by default — see §6)
templates/adapter/   the 10 pieces as annotated skeletons
reference/           worked cold-chain-logistics domain — passes 14/14 conformance
docs/                0*.md reference track, learn/ beginner track, images/ committed charts
tests/               314 tests; doubles live ONLY in tests/fixtures/
scripts/             audit_no_mocks.py, run_demo.py
config/              5 TOML files, every value commented with its rationale
```

Key documents: `finalplan.md` (architecture + SOTA research with sources), `RESOLUTION.md`
(dependency forensics), `ISSUES.md` (open/fixed ledger — **read this first**).

---

## 4. House rules — these are enforced, not aspirational

1. **No mocks, stubs, fakes or TODOs anywhere in `src/`.** `scripts/audit_no_mocks.py`
   fails the build on forbidden tokens, empty function bodies and swallowed `ImportError`s.
   `tests/test_meta.py` runs it as a test. The only escape is a reason-carrying
   `# audit-ok: <why>` on the preceding line — there are 15, all reviewed, all recording
   their failure rather than hiding it.
2. **No silent fallbacks.** A control that cannot run fails closed and says so. Optional
   deps go through `aegis_ml._require.require()`, which names the exact install command.
3. **Requested vs measured are always two fields.** `conformal_coverage` /
   `conformal_coverage_empirical`. Never one field that means whichever the reader assumes.
4. **Test doubles live only in `tests/fixtures/`.**
5. Python 3.11+, ruff `E,F,I,UP,B,SIM,ANN,D`, line length 100, Google docstrings that carry
   the *reasoning* — why the code is shaped this way and what breaks otherwise.

---

## 5. Verified state (measured, not claimed)

| check | result |
|---|---|
| `scripts/run_demo.py` end to end | **exit 0** |
| test suite | **314 passed**, 0 failed, 0 xfail (~4.5 min) |
| Aegis conformance vs `reference.adapter` | **14/14 passed** |
| `ruff check src reference scripts tests` | **0 errors** |
| no-mocks audit | **PASS**, 64 source files |
| `aegis-ml doctor` | exit 0 in both venvs |
| realism (data honesty) | held-out R² **0.6719** in band [0.45, 0.80]; oracle 0.7397; accuracy 0.8468 |
| model quality | test r² **0.7199–0.7224**; conformal coverage requested 90% → **measured 91.2–91.4%** |
| gate | 5/5 criteria, promoted |
| drift on shifted frame | **block**, 70% of features drifted, correct columns named |
| NannyML label-free | `estimated_rmse` ≈ 6.6 (named "estimated" throughout) |
| AutoGluon | R² 0.5411, 10 models, 20 s |
| per-run visuals | 9 PNGs + `index.html` (0 external refs) + `interactive.html` |
| dashboard | `aegis-ml dashboard` — hub 200, MLflow 200, Optuna 200; 0 orphaned processes on shutdown; 0 external network requests |

Reproduce: `.venv/bin/python scripts/run_demo.py`.

---

## 6. Things that will bite you

**TabPFN needs one-time browser setup.** `pip install` and `import` both succeed, then
`.fit()` raises `TabPFNLicenseError`. Prior Labs gates weights behind licence acceptance +
`TABPFN_TOKEN`. `automl/tiers.py` now probes for this so it reports instead of crashing
mid-search. **Operator action, needs a browser — not hackathon morning.** Steps in
`RESOLUTION.md`.

**lightgbm ↔ nannyml ↔ scikit-learn.** `nannyml` pins `lightgbm<4.6`; lightgbm `<4.6`'s
*sklearn wrapper* calls `check_X_y(force_all_finite=)`, removed in sklearn 1.8. NannyML and
FLAML both use LightGBM's native API and are fine; only a direct `LGBMRegressor().fit()`
raises. `is_portable_kind` now probes with a real two-row fit — **importable is not
fittable**, and conflating them killed the whole FLAML tier once. Do not "simplify" that
probe back into a version check.

**ONNX is off by default.** Two measured limits: sklearn's learned NaN routing does not
survive conversion (max abs diff **17.7** on NaN rows vs 2.4e-06 on clean), and
`HistGradientBoosting` will not convert on skl2onnx 1.20. Nothing in Aegis serves ONNX.

**The held-out split is not persisted by older runs.** `report/bundle.recover_split`
re-derives it and accepts a candidate *only if re-scoring the model reproduces the registered
metric exactly*. New runs write `split.json`. Never plot from an unverified split.

**Aegis's own repo has defects we deliberately did not inherit** (ISSUES #13–16): its
reference generator sits at R² 0.97–0.98 (flat `noise_scale=4.0`); four `web/` console files
carry shipped-domain literals and are outside the Python-only vocabulary scan; its docs
disagree with themselves on counts. Do not pattern-match its generator.

---

## 7. What is still open

`ISSUES.md` is authoritative. **10 of 16 fixed.** Remaining: #7 ONNX (accepted), #11 TabPFN
token (operator action), #13–16 (Aegis's own repo). Nothing is blocking.

Judgement calls worth revisiting if requirements change: `eval_flow` now *refuses*
in-sample re-scoring unless `--allow-in-sample` (in-sample reads 0.7587 vs honest 0.7224);
`card.json` is in `STANDARD_ARTIFACTS` but never written; `forecast_flow` needs
`aegis[forecast]`, absent here, so figure 10 is honestly omitted from every real bundle.

---

## 8. How this was built

Roughly a dozen subagents on disjoint module groups, each required to **execute** its code
against the real venv rather than only type-check it. That constraint is why the package
works, and nearly every serious bug below was found by running, never by reading:

- the confounder and the noise were **the same standard normal vector** — both rebuilt their
  generator from the same seed. Achieved R² sat at 0.31 against a declared 0.62, silently.
- `log_loss` silently mismatched probability columns, returning **3.03** for a model whose
  true log-loss is 0.52 — a UserWarning was the only symptom.
- the SHAP path did not work at all as first written: `shap.maskers.Independent` uses
  `numpy.isclose`, which raises on string columns.
- `is_portable_kind` meant "importable" and killed the FLAML tier.
- `_validate_run_id` promised to reject shell-glob metacharacters and only checked path
  escape, so `wild*card` became a legal run directory name.

**If you take one working practice from this session: make the agent run the thing.**

Git identity is `yrevash <yashtiwari9182@gmail.com>`; no co-author trailers. Remote is
`https://github.com/yrevash/aegis_ml` (public).
