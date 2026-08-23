# 09 · Troubleshooting

Symptom → cause → fix. Ordered roughly by how much time each one costs when you do not know it.

**Rule zero:** if something looks wrong and no exception was raised, it is in this document. Aegis's failure modes are silent by design of the defects they descend from — *"a wiring mistake that raises will be found in the first minute by anyone; a wiring mistake that logs a warning and answers as QA will not be found at all."*

---

## 1. The quick table

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `distinct=False` on the last line of `python -m app.ml` | The generated label is not a function of the features. | §2 |
| 2 | Held-out R² ≈ 0.99 (or accuracy ≈ 1.0) | Target leakage, or `noise_scale` ≈ 0. | §3 |
| 3 | Conformance check #12 fails: *resolves to the fallback* | A misspelled `FEATURE_NAMES` / `TARGET.name`, or `ml_spec` not importable. | §4 |
| 4 | `MLModelUnavailableError` / HTTP 503 from `/ml/*` | No artifact at `backend/.artifacts/ml_spine.joblib`, or it was written to the library path. | §5 |
| 5 | **Every login raises `KeyError`** after re-voicing `PERSONAS` | `PERSONA_BY_ROLE` still names dead persona ids. | §6 |
| 6 | A playbook is never selected; the agent acts without procedural guidance | The `hints` table in `select_skills` still names the old filenames. | §7 |
| 7 | The console shows the old domain's words | The four `web/` files are outside the Python-only vocabulary scan. | §8 |
| 8 | The corpus still returns the old domain's documents | `cp -r` instead of `rsync -a --delete`. | §9 |
| 9 | `ImportError` / `ModuleNotFoundError` around torch, autogluon, tabpfn | Wrong venv. The heavy tiers live in `.venv-ml`. | §10 |
| 10 | Drift fires but nothing actually changed | Small-sample noise on a per-feature statistical test. | §11 |
| 11 | Empirical coverage far *above* requested (e.g. 0.99 vs 0.90) | Under-confident model; usually too little calibration data or a noise target. | §12 |
| 12 | Hundreds of `ImportError`s across the whole backend suite | Expected mid-retarget. | §13 |
| 13 | Conformance check #14 fails naming a core file | A real domain leak, **or** a stale `_vocabulary.py`. | §14 |
| 14 | A new specialist answers as QA, and nothing warned | Its role is not in `SPECIALIST_NODES`. | §15 |
| 15 | A sub-agent runs with no tools | A stale name in `tool_allowlist`, silently intersected away. | §16 |
| 16 | `pandas`/`numpy`/`numba` version conflict in `backend/.venv` | Something heavy was installed into the serving venv. | §17 |
| 17 | `TrainerVenvMissingError` | `.venv-ml` absent. | §18 |
| 18 | `RecipeNotPortableError` | The AutoML winner cannot be re-fitted in the serving venv. | §19 |
| 19 | `data_source: "synthetic"` in the model card | The spine fell back to its own noise synthesiser. | §20 |
| 20 | An unseen categorical level silently ignored | `handle_unknown="ignore"` on the one-hot encoder. | §21 |

---

## 2. `distinct=False`

```
sanity: lowest-labelled row=24.1 hours  highest-labelled row=24.1 hours  (distinct=False)
```

**What it means.** `backend/src/app/ml/__main__.py` takes the two rows at the extremes of your own training frame's label and asks whether the fitted model separates them. It does not. The model predicts the same value for the best and worst rows in your data.

**Cause, in order of likelihood:**

1. **The label is not `latent_fn(features) + noise`.** The generator draws the target from its own distribution.
2. The latent function is called, but with a *different* feature dict than the training frame uses — a re-derivation instead of a call to `ml_spec.features_for_*`.
3. `noise_scale` is enormous relative to `Var(latent)`, so the signal is buried.
4. The latent function's drivers are all near-zero coefficients.

**Fix.** Go to `docs/04-synthetic-data.md` §4–§6. Do **not** touch the prompts, the model or the pipeline — the problem is in `generator.py`.

**Prevent it happening again:**

```bash
cd /Users/yrevash/aegis_ml && uv run aegis-ml contract
```

`assert_learnable` fails in seconds with the measured number instead of `distinct=False` minutes before a demo. Add it to your rewritten `backend/tests/adapter/test_ml_spec.py` (see `docs/07-integration-with-aegis.md` §5).

> **This is not covered by any of the fourteen conformance checks.** A pure-noise target passes all of them, plus the whole backend suite, plus ruff.

---

## 3. R² suspiciously near 1.0

**Cause A — target leakage.** A feature is a deterministic (or near-deterministic) function of the label. Classic examples: a `resolved_at` timestamp when the label is a duration from `created_at`; a `total` when the label is one of its components; an SLA-breach flag when the label is the duration the SLA compares against.

```bash
cd /Users/yrevash/aegis_ml && uv run aegis-ml contract    # runs the leakage scan
```

`TargetLeakageError` names the feature and its single-feature score. Drop it from `FEATURES`, or declare it intentional via config if it is genuinely available at prediction time.

**Cause B — `noise_scale` ≈ 0.** Derive it instead of typing it:

```python
from aegis_ml.data.latent import calibrate_noise
sigma = calibrate_noise(problem, latent_fn, target_r2=0.62, n=2000, seed=7)
```

**Cause C — no unobserved confounder.** If every driver of the label is a declared feature and the noise is small, a strong model *can* reach 0.99. Add a per-group offset drawn from the same seeded RNG that never appears in `FEATURES`. See `docs/04-synthetic-data.md` §6.

**Why this matters.** A judge who sees R² = 0.997 on synthetic data learns that you generated the answer and then predicted it. Target band: **R² 0.45–0.80**, **accuracy 0.65–0.88**.

---

## 4. Conformance check #12: *resolves to the fallback*

`test_ml_spec_resolves_to_the_domain_not_the_fallback` failing means `aegis.ml.spec.resolve_spec(your_ml_spec)` returned `FALLBACK_SPEC` — four columns called `feature_0`…`feature_3` predicting `target`.

**Diagnose in one command:**

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
from app.adapter import ml_spec
from aegis.ml.spec import resolve_spec
print('FEATURE_NAMES:', getattr(ml_spec, 'FEATURE_NAMES', '<<MISSING>>'))
print('TARGET       :', getattr(ml_spec, 'TARGET', '<<MISSING>>'))
print('TARGET.name  :', getattr(getattr(ml_spec, 'TARGET', None), 'name', '<<MISSING>>'))
print('resolved     :', resolve_spec(ml_spec))
")
```

**Causes:**

| What you see | Cause |
|---|---|
| `FEATURE_NAMES: <<MISSING>>` | Misspelled (`FEATURES_NAMES`, `FEATURE_NAME`), or defined but not at module scope. |
| `FEATURE_NAMES: []` | Empty list — `not features` is `True`, so the fallback fires. |
| `TARGET.name: <<MISSING>>` | `TARGET` is a plain string, or your spec class calls the field something else. It must carry `.name`. |
| Everything present but `task` is wrong | `_coerce_task` maps anything outside `{"classification","classify","clf","categorical","binary"}` to `"regression"`. A typo silently trains a regressor on class labels. |
| The right names but the wrong categoricals | `CATEGORICAL_FEATURES` absent **and** `FEATURES[].dtype` is not the literal string `"categorical"`. |

**The permanent fix** is to stop hand-writing the file: declare the problem once as `aegis_ml.contracts.spec.MLProblem` and generate `ml_spec.py` from it. `MLProblem` refuses a categorical with no declared `levels`, refuses a name that is not a valid Python identifier, refuses duplicate feature names, and refuses a target that is also a feature ("that is perfect leakage").

---

## 5. `MLModelUnavailableError` / HTTP 503

**Cause A — no artifact.** Train it:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml)
ls -l /Users/yrevash/aegis/backend/.artifacts/ml_spine.joblib
```

**Cause B — the artifact is in the wrong directory, which is the expensive one.**

There are two `DEFAULT_ARTIFACT_PATH` constants:

| Constant | Resolves to |
|---|---|
| `app.ml.DEFAULT_ARTIFACT_PATH` | `backend/.artifacts/ml_spine.joblib` ← **`app.ml.get_model()` loads this** |
| `aegis.ml.DEFAULT_ARTIFACT_PATH` (re-exported by `app.ml.model`) | inside the installed `aegis` package |

Training through the library constant writes where nothing loads from, so **training appears to succeed and the endpoints keep answering 503**, with the two paths differing by a directory nobody looks at. `backend/src/app/ml/__main__.py` carries a comment about exactly this.

Always import the host constant: `from app.ml import DEFAULT_ARTIFACT_PATH, train`.

`aegis_ml.settings.artifact_path` computes the host path from `settings.aegis_root`, so `aegis-ml promote` writes to the right place. Check it:

```bash
cd /Users/yrevash/aegis_ml && uv run python -c "from aegis_ml import settings; print(settings.artifact_path, settings.artifact_path.exists())"
```

**Cause C — the process cached the old model.** `get_model()` holds a process-wide singleton and does **not** notice a file swap. Restart the backend after a promotion or a rollback.

**Cause D — the adapter is not importable.** `app.ml.train()` refuses rather than falling back:

```
The domain adapter (app.adapter.ml_spec) is not importable, so there is no domain spec
to train on. Refusing to fall back to the built-in noise synthesiser and serve it as
domain evidence — pass an explicit spec to app.ml.train(), or repair the adapter.
```

That is invariant 4 working correctly. Fix the adapter.

---

## 6. Every login raises `KeyError`

**The single most common retarget failure, and no test in the repository catches it.**

**Cause.** You re-voiced `PERSONAS` (step 5 of the procedure explicitly instructs you to) and did not re-point `PERSONA_BY_ROLE`. Every authenticated principal resolves its persona through `persona_for_role(role)`, which reads that table. An entry naming a persona that no longer exists raises `KeyError` **at the login boundary**.

The adapter suite, the agent suite, ruff and thirteen of the fourteen conformance checks all stay green, because **none of them go through the login path.**

**Fix.** In `personas.py`:

```python
PERSONA_BY_ROLE: dict[Role, str] = {
    Role.ADMIN:   "<your_staff_persona_id>",
    Role.AI_TEAM: "<your_staff_persona_id>",
    Role.DEVOPS:  "<your_staff_persona_id>",
    Role.CLIENT:  "<your_end_user_persona_id>",
}
```

**Every** RBAC role must have an entry, and **every** value must be a key of `PERSONAS`.

**Also check `DEFAULT_PERSONA_ID`** — `PERSONAS[DEFAULT_PERSONA_ID]` is evaluated for every request that names no persona, so a stale value 500s every anonymous request.

Conformance check #7 (`test_every_persona_the_adapter_declares_resolves`) covers both. Run it.

**And check the console:** `web/src/config/personas.ts` sends persona ids to `POST /query`, which answers **400 Unknown persona** for an id the adapter does not declare. See §8.

---

## 7. A playbook is never selected

**Symptom.** The agent answers without following your procedure. No warning anywhere. You read it as a prompt problem and spend an hour in `prompts.py`.

**Cause.** Skills are selected **by filename**, through a literal keyword → filename `hints` dict inside `memory_spec.select_skills`. The reference ships:

```python
hints = {
    "close": "closing_requests", "resolve": "closing_requests", "duplicate": "closing_requests",
    "angry": "de_escalation", "frustrated": "de_escalation",
    "escalate": "de_escalation", "complaint": "de_escalation",
}
```

Rename `closing_requests.md` and the entry still reads `"closing_requests"`, which is then filtered out by `skill in available` — so the renamed playbook can never be selected again and the stale entry can never fire. Both halves are silent: `select_skills` returns its other matches, or `None`, and the turn proceeds.

**Fix.** Piece 10 is **two edits**: the `*.md` files *and* the `hints` table in `memory_spec.py`. Do both in the same commit.

Conformance check #11 (`test_every_playbook_is_reachable_from_select_skills`) covers it, reading the selector's compiled string constants **and** its module's top-level constants, so it works whether the table sits inside the function or beside it. It also probes behaviourally and asserts the selector never returns a name outside the `available` list it was handed.

Check #10 (`test_skills_directory_holds_at_least_one_playbook`) catches the other half: a `SKILLS_DIR` pointing nowhere discovers zero playbooks and reports nothing at all.

---

## 8. The console shows the old domain's words

**Cause.** The vocabulary-quarantine check scans **Python only**. `web/` is outside the adapter *and* outside the check.

Four files, all verified present:

```
/Users/yrevash/aegis/web/src/config/personas.ts
/Users/yrevash/aegis/web/src/components/ops/opsShared.ts
/Users/yrevash/aegis/web/src/components/sim/SimulationView.tsx
/Users/yrevash/aegis/web/src/components/ml/MLOpsView.tsx
```

**Fix.** Re-voice them by hand — `prompts/13-console.md` has the exact edits. Then sweep for anything you missed:

```bash
cd /Users/yrevash/aegis/web
grep -rn "operations_lead\|update_request_status\|assign_request\|add_case_note\|find_requests\|queue_depth_at_open\|agent_tenure_months\|reopened_count\|description_length\|customer_tier\|resolution_hours\|service_request\|Service requests opened per day" src/ tests/
```

**The worst of the four is `MLOpsView.tsx`**, which carries a literal ML feature row — the same defect as the trainer's old sanity probe. After a retarget the panel sends feature keys your model has never heard of, every one of them lands in `MLExplainResponse.unknown_features`, every declared feature lands in `imputed_features`, and the panel shows a prediction made entirely from training medians.

**And `personas.ts` breaks the console outright**, not just cosmetically: its ids are sent as `QueryRequest.persona`, and the backend answers `400 Unknown persona` for an id the adapter does not declare.

---

## 9. The corpus still serves the old domain

**Symptom.** Retrieval cites documents about the reference domain. Every citation resolves; the content is just from the wrong world.

**Cause.** `cp -r` overwrites the Python modules but leaves the data files it did not replace: 3 corpus documents (`kb_request_closure.md`, `policy_escalation.md`, `runbook_login_failures.md`) and 2 skill playbooks (`closing_requests.md`, `de_escalation.md`).

**Fix.**

```bash
rsync -a --delete /Users/yrevash/aegis_ml/reference/adapter/ /Users/yrevash/aegis/backend/src/app/adapter/
find /Users/yrevash/aegis/backend/src/app/adapter -name '__pycache__' -type d -exec rm -rf {} +
ls /Users/yrevash/aegis/backend/src/app/adapter/corpus/*.md
ls /Users/yrevash/aegis/backend/src/app/adapter/skills/*.md
```

Windows: `robocopy C:\aegis_ml\reference\adapter C:\aegis\backend\src\app\adapter /MIR`.

**If retrieval already ingested them**, deleting the files is not enough — reindex, or drop the collection and re-seed.

---

## 10. Import errors from the venv split

| Error | Which venv you are in | Fix |
|---|---|---|
| `ModuleNotFoundError: autogluon` / `tabpfn` / `torch` | serving venv | Correct. They live in `.venv-ml`. Use `aegis_ml.automl.runner`, which shells to `settings.trainer_python`. |
| `ModuleNotFoundError: app` | anywhere | `PYTHONPATH` not set, or `:` used instead of `;` on Windows. |
| `ModuleNotFoundError: aegis` | the `aegis_ml` venv | Add the core to the path: `PYTHONPATH=/Users/yrevash/aegis/aegis/src`. `aegis` is a *sibling checkout*, deliberately not a dependency of this package. |
| `ModuleNotFoundError: aegis_ml` in the backend | not installed there | `uv pip install --python /Users/yrevash/aegis/backend/.venv -e '/Users/yrevash/aegis_ml[serve]'` |
| `AutoMLTierUnavailableError` | serving venv, asked for a strong tier | Expected and correct. The message names the tier, the missing module and the install command. **It does not silently fall through to a weaker tier** — a caller who asked for AutoGluon and got plain XGBoost must be able to tell that apart. |

Never `pip install autogluon` into `backend/.venv`. See §17.

---

## 11. Drift firing on sampling noise

**Symptom.** `drifted_share` above the warn threshold on data you know is from the same distribution.

**Cause.** Per-feature drift is a statistical test (KS, chi-square, PSI, Wasserstein depending on dtype and Evidently's automatic choice). At small `n_current_rows`, a test on each of nine features will flag one or two by chance alone — that is what a p-value means.

**Diagnose:**

```bash
uv run aegis-ml drift --run <run_id> --current <live.parquet> --verbose
```

Read `n_current_rows` and `drifted_features` from the `DriftReport`.

| Signal | Reading |
|---|---|
| `n_current_rows < 500` | **Not enough data to conclude anything.** Wait. |
| A different feature set flags on each window | Noise. |
| The *same* features flag on consecutive windows | Real. |
| `drifted_share` high but `prediction_drift` ≈ 0 | Inputs moved in ways the model does not care about. Usually benign. |
| `prediction_drift` high **and** the NannyML `estimated_metric_value` is falling | **Real, and it matters.** This is the pair to trust. |

**Fixes:**

- Raise `AEGIS_ML_DRIFT_SHARE_WARN` / `AEGIS_ML_DRIFT_SHARE_BLOCK` (defaults 0.2 / 0.4) if you are running small windows.
- Increase the window size before evaluating.
- **Trust the NannyML estimate over the drift share.** Data drift is a proxy; estimated performance is the thing you actually care about.

**Remember what drift does not do:** it never withdraws the serving model. *"Aegis serves the model it has and flags it. This blocks PROMOTION of anything calibrated on the drifted reference."* Withdrawing a model on a drift signal turns a quality warning into an outage.

---

## 12. Empirical coverage far above requested

```
conformal_coverage:           0.90
conformal_coverage_empirical: 0.994
```

**This is a failure, not a success.** Requesting 90% and measuring 99% means the intervals are far wider than they need to be — the model is under-confident and the interval is nearly useless for a decision.

**Causes:**

1. **Too few calibration rows.** The conformal quantile is estimated from the calibration split; with too few rows it lands conservatively. `aegis/ml/model.py` has `_min_calibration_rows(confidence_level)` as a floor. Raise `num_records`.
2. **A noise target.** With no signal, the honest 90% interval spans nearly the whole target range and trivially covers 99% of a *finite* test split. Check §2 first.
3. **Heavy-tailed noise.** A few extreme residuals push the quantile out. Expected if you made the noise heteroscedastic (which you should have) — check whether the *width* varies with the feature you scaled it by. If it does, this is working as designed and you should say so.

Report both fields always. **Never one field that means whichever the reader assumes.**

---

## 13. Hundreds of `ImportError`s across the whole suite

**This is expected. Do not chase it.**

`backend/tests/conftest.py` imports through `app.adapter`. The moment you replace the entity models in piece 1, **every test in the repository fails at import** — a wall of `ImportError`, hundreds of lines, none of it about whether your edit was right. It stays that way until piece 8 lands and the registry's re-exports resolve again.

**Do not "fix" it by loosening a conftest.**

Two things are meaningful mid-flight:

1. **The conformance suite** — green from your first edit to your last, no infrastructure, under a second.
2. **The per-piece verify commands**, but only once you have rewritten the test file each one runs.

Authoring inside `aegis_ml/` and syncing at the end (`docs/07` §2) means the wall does not appear until you sync. That is the main reason to work that way.

---

## 14. Conformance check #14 fails naming a core file

`test_no_shipped_domain_vocabulary_survives_outside_the_adapter`. Two very different situations.

**Situation A — a real leak.** A term from `SHIPPED_VOCABULARY` appears in a module outside the adapter. Open the file and line it names. Something domain-specific has been written into the core or the host. **Move it into the adapter.** *"If you think you must edit a core file to finish, the check is the fastest way to find out whether the leak is real and where."*

**Situation B — a stale `_vocabulary.py`.** Your adapter's `DOMAIN_ID` now equals `SHIPPED_DOMAIN_ID`, so the check *also* requires every listed term to still appear **inside** your adapter. A word you left in the list that your domain never uses fails.

**Fix for B:** update `aegis/src/aegis/conformance/_vocabulary.py`. **This edit is required and sanctioned** — see `docs/07-integration-with-aegis.md` §6. It is the one core file you may change, and you must report it.

**Do not** delete the failing term from a core docstring just to make the check pass unless the term genuinely does not belong there. The list's own docstring: *"this check must never be the reason somebody deletes a true sentence from a core docstring."*

**Do not** change `MIN_CORE_FILES` (the anti-vacuity floor), `_SKIP_DIRS`, `core_files()` or `scan_for_terms()`.

---

## 15. A specialist answers as QA and nothing warned

**Cause.** `aegis/src/aegis/agent/graph.py`:

```python
SPECIALIST_NODES: dict[str, str] = {"qa": "recall_memory", "memory": "answer_memory", "team": "plan_team"}
```

A roster role outside that set falls back to the `qa` pipeline **with a log warning, not an exception** — and the `routing` stream event still names your specialist, so the console shows it being chosen. The build-time warning meant to catch this could not fire either: it iterated `roster.roles` (the bound method, not the list) and the surrounding `except Exception` swallowed the `TypeError`.

**Fix.** Re-voice `qa` and `memory` — change `description` and `keywords` freely, **keep the two role strings**. No core edit, and it is what you should do under time pressure.

If the domain genuinely needs a third path, adding a handler node plus a `SPECIALIST_NODES` entry is the **one other sanctioned core edit**, and it must be reported rather than done quietly.

**Never declare `team` in a roster.** The router writes it when the depth classifier chooses fan-out.

Conformance checks #3 and #4 cover this. Check #4 is the worse case: `AgentRoster.default_role` returns the **first specialist in declaration order** when none is marked `is_default=True`, so forgetting the flag silently promotes whichever specialist happens to be written first — and if that role has no node, *every unmatched turn* takes the fallback path.

---

## 16. A sub-agent runs with no tools

**Cause.** `SubAgentSpec.tool_allowlist` holds **literal tool names** and is *intersected* with `TOOL_REGISTRY`. A stale name is silently dropped. The shipped `data` lane allowlists `{"update_request_status", "add_case_note"}` — both of which your retarget deletes.

**Fix.** Re-point every `tool_allowlist` to names that exist in your registry, and re-voice each spec's `label` and `system_prompt` while you are there — they are read by the model and shown on screen.

Conformance check #6 covers it, along with the same failure in `ALLOWLIST`: a misspelled persona key gives that persona **no tools at all**, and a misspelled tool name simply never appears in the model's `tools=` payload. Neither raises, and the agent answers the question anyway.

---

## 17. `pandas` / `numpy` / `numba` conflict in the backend venv

**Cause.** Something heavy was installed into `backend/.venv`, and the resolver moved a capped package.

```bash
/Users/yrevash/aegis/backend/.venv/bin/python -c "import pandas, numpy, numba; print(pandas.__version__, numpy.__version__, numba.__version__)"
```

Required: pandas `>=2.2,<2.4`, numpy `>=1.26,<2.5`, numba `==0.67.0`. Also `litellm==1.96.0` and `presidio-analyzer==2.2.364`.

| Cap | Why |
|---|---|
| `pandas<2.4` | nemoguardrails |
| `numpy<2.5` | presidio-analyzer 2.2.364 declares it; numba/llvmlite (a shap dependency) have no numpy-2.5 release, and without the cap the resolver drags numba back to an ancient version that cannot build against numpy 2.4 |
| `numba==0.67.0` | pinned in `[tool.uv] constraint-dependencies` |
| `litellm==1.96.0` | 1.96.2 regressed the gateway path |
| `presidio-analyzer==2.2.364` | a free resolve back-solves to 2.2.362 around pydantic/numpy |

**Fix.** Reinstall the backend venv from its lockfile:

```bash
cd /Users/yrevash/aegis/backend && uv sync
```

Then reinstall only the `[serve]` extra of this package, which is pure-Python-or-already-present *by construction*:

```bash
uv pip install --python /Users/yrevash/aegis/backend/.venv -e '/Users/yrevash/aegis_ml[serve]'
```

**Never install `[strong]` into `backend/.venv`.** That is what `.venv-ml` exists for.

---

## 18. `TrainerVenvMissingError`

The message carries the fix:

```
Trainer venv not found at '.venv-ml'. Create it with:
  uv venv .venv-ml --python 3.11
  uv pip install --python .venv-ml -e '.[strong,serve]'
The heavy tiers live there on purpose: AutoGluon/TabPFN/torch will not resolve under
the backend's pandas<2.4 / numpy<2.5 / numba==0.67.0 caps.
```

On Windows install torch from the CPU index **first** — see `docs/08-windows.md` §2.3.

If you are short on time, run without it: `uv run aegis-ml train --tier baseline,flaml`. Both run in the serving venv and produce a portable recipe. The strong tiers will be reported as unavailable **with a reason**, which is honest.

---

## 19. `RecipeNotPortableError`

```
Recipe member 'TabPFNRegressor' is not portable into the serving venv: <reason>.
Report its leaderboard score as the accuracy ceiling and export it to ONNX for a
side-by-side predictor — do not promote it as the spine.
```

**Working as designed.** The two-venv split is only sound because the winning configuration crosses as JSON and is re-fitted by the Aegis spine. A recipe naming an estimator the serving venv cannot construct breaks that guarantee, so it is refused rather than half-applied.

`recipe.PORTABLE_KINDS` is an explicit allowlist of tree learners `shap.TreeExplainer` supports — XGBoost, HistGradientBoosting, RandomForest, ExtraTrees, LightGBM. Adding a non-tree member would produce a model that trains, scores, promotes, and then **raises inside `explain()` on the first request that asks why**.

**What to do:** promote the best *portable* recipe, quote the non-portable winner as the accuracy ceiling in the card, and export it to ONNX for a side-by-side (`aegis-ml export --onnx`). That is a better demo slide than pretending the ceiling does not exist.

If the member genuinely *is* constructible in the serving venv (e.g. LightGBM, which is in `[strong]` but pip-installable into the backend), install it there and re-run. `is_portable_kind` checks importability, not just the allowlist.

---

## 20. `data_source: "synthetic"` in the model card

**What it means.** The spine trained on **its own built-in noise synthesiser**, not on your data. The model carries **no domain signal**, and `MLExplainResponse.data_source` will say `"synthetic"` on every prediction.

The three values:

| `data_source` | Meaning |
|---|---|
| `"provided"` | An explicit frame was passed to `train()`. Good. |
| `"spec_provider"` | The frame came from your spec's `training_frame` callable. Good. |
| `"synthetic"` | **The spine generated noise.** Bad. |

**Cause.** `resolve_spec` found no callable `training_frame` on your `ml_spec` — usually because it is misspelled, is not at module scope, or is a non-callable attribute. Note `resolve_spec` sets `provider = None` when `training_frame` is not callable, and does so **silently**.

**Fix.** Ensure `ml_spec.training_frame` is a module-level function with the signature `(*, num_records: int = ..., seed: int = ...) -> pd.DataFrame`. Then re-check §4's diagnostic.

**The field is the honesty signal.** Downstream code and the UI must be able to discount the evidence on it alone. If you see it, do not ship the model.

---

## 21. An unseen categorical level is silently ignored

**Symptom.** No error. A slightly worse model, a wider conformal interval, and SHAP showing near-zero contribution for a feature you know matters — on *some* rows.

**Cause.** `aegis.ml.model.train` one-hot-encodes with `handle_unknown="ignore"`. An unseen level **does not raise**: it encodes to an all-zero block and the row is scored as if the feature were absent. A generator emitting `"REFRIGERATED "` (trailing space) for 3% of rows produces exactly this, with no error anywhere in the stack.

**Fix.** The pandera contract catches it at the boundary:

```bash
cd /Users/yrevash/aegis_ml && uv run aegis-ml contract
```

`aegis_ml.contracts.spec.FeatureSpec` **refuses a categorical that declares no `levels`** for this exact reason: *"the data contract cannot check an open set, and an unseen level encodes to all-zeros without raising."*

**Prevention.** Derive levels from your `StrEnum`s, never type them twice:

```python
FeatureSpec(name="priority", dtype="categorical", levels=[p.value for p in Priority], ...)
```

---

## 22. When you are genuinely stuck

In order:

1. **Run the conformance suite.** Fourteen checks, no infrastructure, under a second. Every failure prints what is wrong, the edit that fixes it, what happens if you leave it, and the defect it came from.
2. **Run the structural check.** `missing_members(app.adapter)` names a whole piece you forgot.
3. **Run `aegis-ml contract`.** pandera + learnability + leakage, in seconds.
4. **Read the last line of `python -m app.ml`.** `distinct=True` is the pass signal.
5. **Grep the console** for the shipped vocabulary (§8).
6. **Re-read `SKILL.md`** — it is the authoritative retargeting procedure and nothing else supersedes it.
7. **Ask whether the failure raised.** If it did not, it is in this document.

---

## 23. Next

`docs/10-architecture-decisions.md`.
