# PROMPT 11 · The ML pipeline — train, evaluate, promote, monitor

---

## Role

You are running the ML half: turning the adapter's `training_frame` into a promoted, explainable, monitored model, and producing the artefacts a demo is built from.

**Prerequisite: all ten pieces exist and `aegis-ml contract` passes.** If `assert_learnable` has not gone green, stop and go back to `prompts/03-generator.md`. Nothing here is worth running against a noise target.

---

## Read first

- `/Users/yrevash/aegis_ml/docs/05-ml-pipelines.md`
- `/Users/yrevash/aegis_ml/docs/06-mlops-registry-drift.md`

---

## Step 1 — Confirm the environment

```bash
cd /Users/yrevash/aegis_ml && uv run aegis-ml doctor
```

```powershell
Set-Location C:\aegis_ml; uv run aegis-ml doctor
```

Read four lines:

| Line | Must say |
|---|---|
| AutoML tiers | Which of `baseline` / `flaml` / `autogluon` / `tabpfn` will run, and a **reason** for each that will not |
| Trainer venv | An interpreter exists, or you are running `baseline,flaml` only |
| `artifact_path` | `/Users/yrevash/aegis/backend/.artifacts/ml_spine.joblib`, directory writable |
| Learnability | The current adapter's held-out score |

A disabled tier (`AEGIS_ML_ENABLE_TABPFN=0`) and an uninstalled tier produce **different** reason strings. That distinction is deliberate — never let "not installed" and "found nothing better" look the same on a leaderboard.

---

## Step 2 — The cheap gate

```bash
uv run aegis-ml contract
```

Three stages, seconds each:

1. **pandera** — dtypes, ranges, null policy and **the categorical level sets**. The level check matters because `aegis.ml.model.train` one-hot-encodes with `handle_unknown="ignore"`: an unseen level does not raise, it encodes to an all-zero block and the row is scored as if the feature were absent.
2. **`assert_learnable`** — the check nothing in Aegis performs. `LabelNotLearnableError` names the measured score and the floor.
3. **leakage** — every feature scored alone against the target; anything above `0.98` raises `TargetLeakageError`.

**If any of these fails, stop.** Everything downstream is expensive and meaningless.

---

## Step 3 — Train the Aegis spine first

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml)
```

Do this **before** the AutoML search. It is fast, it uses no trainer venv, and it produces the baseline you compare everything against. It also prints the sanity probe:

```
Training ML spine on domain spec: target='slot_overrun_minutes' task=regression
  categorical: ['procedure_type', 'asa_grade', 'surgeon_seniority', 'theatre_id']
  numeric    : ['slot_position', 'booked_minutes', 'prior_overrun_mins', 'patient_bmi', 'equipment_swaps']
Saved artifact → .../backend/.artifacts/ml_spine.joblib (24 encoded cols)
  sanity: lowest-labelled row=2.4 minutes  highest-labelled row=88.1 minutes  (distinct=True)
```

Check three things:

- **`target=` and `task=` are yours** — if they say `target='target'` with four `feature_N` columns, `resolve_spec` returned `FALLBACK_SPEC`. See `docs/09-troubleshooting.md` §4.
- **The categorical/numeric split is right.**
- **`distinct=True`.** `False` means the spine learned nothing. Go to `docs/04-synthetic-data.md`, not to the prompts.

---

## Step 4 — The AutoML search

```bash
cd /Users/yrevash/aegis_ml
uv run aegis-ml train --tier all
```

Variants:

```bash
uv run aegis-ml train --tier baseline,flaml        # no trainer venv needed
uv run aegis-ml train --tier all --no-hpo          # skip Optuna; the tiers give most of the gain
uv run aegis-ml train --tier all --budget 300      # seconds per tier
uv run aegis-ml train --reuse-recipe               # refit an existing recipe, fast
```

What happens:

1. `resolve_tiers` decides which tiers run and **writes down why each other one did not** — the same call produces both halves, so a tier cannot be dropped without a reason.
2. The portable tiers (`baseline`, `flaml`) run in the serving venv. The strong tiers shell out to `.venv-ml` via `aegis_ml.automl.runner`, parquet in, JSON + joblib out.
3. Optuna refines the winner (TPE + HyperbandPruner, SQLite-resumable — re-running resumes from trial *n*).
4. The winning configuration crosses back as a **`Recipe`**, and the **Aegis spine** fits it — keeping MAPIE conformal calibration, SHAP, the `ModelCard` and the `dataset_digest`.

**If the winner is not portable** (TabPFN, AutoGluon's stack), `RecipeNotPortableError` fires and the best portable runner-up is promoted instead, with the ceiling recorded in `Recipe.notes`. That is correct: `PORTABLE_KINDS` is an allowlist of tree learners `shap.TreeExplainer` supports, and a non-tree member produces a model that trains, scores, promotes — and then raises inside `explain()` on the first request that asks *why*.

---

## Step 5 — Evaluate

```bash
uv run aegis-ml eval
```

Produces `card.json`, `card.html`, `shap.html`, `slices.json`, `profile.html`, `drift_ref.parquet`.

**Read these five numbers, in this order:**

| # | Field | Must be |
|---|---|---|
| 1 | `metric_name` / `metric_value` | R² **0.45–0.80**, accuracy **0.65–0.88**. Above 0.90 is a bug report, not a good result — see `docs/09-troubleshooting.md` §3. |
| 2 | `conformal_coverage` vs `conformal_coverage_empirical` | Requested 0.90, measured within ±0.05. **Far above is also a failure** — the intervals are too wide and the model is under-confident. |
| 3 | worst row of `slices.json` | No slice collapsed. |
| 4 | `leaderboard.json` rank 1 vs `baseline` | The margin. A 0.01 margin means ship the baseline and say so. |
| 5 | `data_source` and `dataset_digest` | `"provided"` or `"spec_provider"` — **never `"synthetic"`**, which means the spine fell back to its own noise synthesiser. Digest present. |

Open `shap.html` and confirm two things with your eyes:

- The deliberately irrelevant features you declared are **flat**. That is the proof the explanation is real.
- The top drivers are the ones you wrote into the latent function, with the **signs you gave them**. If a driver you made positive shows up negative, something is wired backwards.

---

## Step 6 — Promote

```bash
uv run aegis-ml promote
```

Five criteria, all must hold, each reported with its number:

| # | Criterion | Threshold |
|---|---|---|
| 1 | beats the champion on the primary metric | `promote_min_gain` = 0.005 |
| 2 | `empirical_coverage ≥ requested_coverage − δ` | `coverage_tolerance` = 0.05 |
| 3 | every pandera contract passes | — |
| 4 | worst slice no worse than the champion's worst | same ε |
| 5 | no target leakage | `leakage_threshold` = 0.98 |

On pass: `registry_store/runs/<run_id>/model.joblib` is **atomically replaced** over `/Users/yrevash/aegis/backend/.artifacts/ml_spine.joblib`, with the previous artifact retained as `ml_spine.previous.joblib`.

On failure: `PromotionRejectedError` lists every failed criterion with its measured number, and **the champion is unchanged**.

> A rejected promotion is **a demo asset, not a problem**. "The gate refused this challenger because one slice collapsed" is the single most convincing thing you can show an enterprise judge. Keep one.

Restart the backend afterwards — `get_model()` caches the artifact in a process-wide singleton and will not notice a file swap.

---

## Step 7 — Monitor

```bash
uv run aegis-ml drift
uv run aegis-ml drift --current registry_store/reports/predictions/<date>.jsonl
```

Two tools, because neither alone is enough:

- **Evidently 0.7+** — data, target and prediction drift against the run's stored `drift_ref.parquet`. **Requires labels for performance metrics.**
- **NannyML CBPE/DLE** — estimates performance **before ground truth arrives**. Everything it produces is spelled `estimated_*` so it is never read as a measurement.

Verdicts: `pass` < 0.20, `warn` 0.20–0.40, `block` > 0.40 (`drifted_share`).

> **Drift never withdraws the serving model.** *"Aegis serves the model it has and flags it. This blocks PROMOTION of anything calibrated on the drifted reference."* Say that out loud — withdrawing on a drift signal turns a quality warning into an outage, and drift detectors fire on sampling noise.

If drift fires on data you know is fine, read `n_current_rows`. Below 500, a per-feature test on nine features will flag one or two by chance. **Trust the NannyML estimate over the drift share.**

---

## Step 8 — Forecast

```bash
uv run aegis-ml forecast
```

Runs `aegis.forecast` over your `domain_series_events()`, with `mlforecast` candidates added. Produces a horizon with conformal bands and a rolling-origin backtest, **reporting the losers as well as the winner** — which is what tells you whether AutoARIMA beat SeasonalNaive by a nose or a mile.

Confirm the chart title is **your** `DOMAIN_SERIES_LABEL`, in the client's language. That sentence is read by a judge.

---

## Step 9 — Export (optional)

```bash
uv run aegis-ml export --onnx
```

Round-trip-validated: the ONNX predictions must match the sklearn predictions to a stated tolerance or the export raises.

**Carry the caveat honestly: MAPIE intervals and SHAP attributions do not export.** The value is a portable point-predictor (~0.029 ms/request) and the validation, **not a new serving path**. It is also the escape hatch for a non-portable AutoML winner.

---

## The full flow, in one command

```bash
uv run aegis-ml train --tier all && uv run aegis-ml eval && uv run aegis-ml promote && uv run aegis-ml drift
```

Or:

```bash
uv run python -c "from aegis_ml.pipelines.flows import full_flow; print(full_flow().model_dump_json(indent=2))"
```

`full_flow` returns a `RunManifest` — every stage with its duration and `ok`/`error`, so a partial run says which stage stopped it.

---

## Wiring ML into the agent

The Aegis README's request path names an `ml_predict` node. **There is none** — `graph.py` declares no such node, and `describe_prediction` has zero consumers. The answer is adapter *tools*, which need no core edit:

```python
# in your tools.py
from aegis_ml.serve.tools import ml_tool_specs

TOOL_REGISTRY.update({spec.name: spec for spec in ml_tool_specs(ToolSpec)})
```

Five tools, all **LOW and read-only**, because ML informs and never gates:
`predict_outcome`, `explain_prediction`, `whatif_scenario`, `forecast_series`, `check_model_health`.

Verify the round trip:

```bash
uv run pytest tests/test_ml_tools_roundtrip.py -q
```

Then, end to end, ask the running agent something that needs a prediction and confirm the transcript carries the interval and the drivers — that is `describe_prediction` finally having a consumer.

---

## Checklist

- [ ] `aegis-ml doctor` is clean; every unavailable tier has a **reason**.
- [ ] `aegis-ml contract` passes: pandera, learnability, no leakage.
- [ ] `python -m app.ml` prints **your** target and task, and `distinct=True`.
- [ ] The leaderboard has more than one row, and the losers are on it.
- [ ] Metric in the target band — **not above 0.90**.
- [ ] `conformal_coverage_empirical` within ±0.05 of `conformal_coverage`, in **both** directions.
- [ ] No slice collapsed.
- [ ] `data_source` is not `"synthetic"`; `dataset_digest` is present.
- [ ] `shap.html`: the irrelevant features are flat, the real drivers have the right signs.
- [ ] The gate produced a `GateDecision` with numbers in `reasons`, pass or fail.
- [ ] `backend/.artifacts/ml_spine.joblib` was written and the backend was restarted.
- [ ] A drift report exists, with a NannyML `estimated_*` figure.
- [ ] The forecast chart title is your `DOMAIN_SERIES_LABEL`.
- [ ] The five ML tools are in `TOOL_REGISTRY`, all LOW and read-only.
- [ ] Any TabPFN-touched artefact carries the Prior Labs licence notice.

---

## Next

`prompts/12-integration.md`.
