# 08 · Your first run

[← 07](07-how-it-plugs-into-aegis.md) · [Index](00-index.md) · Next: [09 · Glossary](09-glossary.md)

Hands-on. Every command below was run against this repository while writing this page, and the
output excerpts are real.

Everything assumes you are in the repository root:

```bash
cd /Users/yrevash/aegis_ml
```

---

## 0. A note on how to invoke the CLI

Two forms work, and they are **not** interchangeable:

```bash
.venv/bin/python -m aegis_ml.cli <command>    # ← use this one
.venv/bin/aegis-ml <command>                  # console script
```

`python -m` puts the current directory on `sys.path`, so `--adapter reference.problem`
resolves. The console script does not, and you get an `ImportError` traceback. If you prefer
the short form, export `PYTHONPATH=.` first. This page uses `python -m` throughout.

---

## 1. Install

Two virtualenvs, for the reasons in [chapter 04 §6](04-the-pipeline.md#6-two-virtualenvs-one-portable-recipe-decision-d1).

```bash
# serving venv — everything that co-installs with Aegis
uv venv .venv --python 3.11
uv pip install --python .venv -e '.[dev]'

# trainer venv — the heavy half, isolated (optional; skip it and two tiers report as skipped)
uv venv .venv-ml --python 3.11
uv pip install --python .venv-ml -e '.[strong,serve]'
```

Or, equivalently, `make install` and `make install-strong`.

The serving install resolves 153 packages; the trainer install resolves 1,174. Exact pins are
frozen in `requirements-serve.lock.txt` and `requirements-strong.lock.txt`. Crucially,
`pandas` / `numpy` / `scikit-learn` resolve to **2.3.3 / 2.4.6 / 1.9.0** in *both*.

---

## 2. `doctor` — run this first, always

```bash
.venv/bin/python -m aegis_ml.cli doctor
```

Real output, abridged:

```
── environment ────────────────────────────────────────────────────────────────
  python           3.11.11  (macOS-26.5.1-arm64-arm-64bit)
  aegis_ml         0.1.0
  aegis            NOT importable — the host platform is not on this path
                   fix: install it, or run from the backend venv where it lives

── resolved versions ──────────────────────────────────────────────────────────
   pandas 2.3.3 · numpy 2.4.6 · sklearn 1.9.0 · xgboost 2.1.4 · shap 0.51.0
   mapie 1.5.0 · pandera 0.32.1 · skrub 0.10.0 · optuna 4.9.0 · flaml 2.6.0
   evidently 0.7.21 · nannyml 0.13.1
!! autogluon.tabular    not installed
!! tabpfn               not installed

── AutoML tiers ───────────────────────────────────────────────────────────────
  RUNS     baseline     sklearn + xgboost
  RUNS     flaml        flaml 2.6.0
  skipped  autogluon    not importable in this interpreter: autogluon.tabular. Install with
                        `uv pip install 'aegis-ml[strong]'`, or run the search through
                        aegis_ml.automl.runner, which executes it inside the trainer venv.
  skipped  tabpfn       not importable in this interpreter: tabpfn. …

  LICENCE  TabPFN-2.5 weights are distributed under the Prior Labs License: research and
           EVALUATION use are permitted; commercial and production use are NOT. …
           Set AEGIS_ML_ENABLE_TABPFN=0 to switch it off.

── paths ──────────────────────────────────────────────────────────────────────
  trainer python   /Users/yrevash/aegis_ml/.venv-ml/bin/python (exists)
  artifact_path    /Users/yrevash/aegis/backend/.artifacts/ml_spine.joblib
                   directory writable · artifact present
  registry_dir     /Users/yrevash/aegis_ml/registry_store (writable)

── data realism ───────────────────────────────────────────────────────────────
  bands            regression R² (0.45, 0.8), classification accuracy (0.62, 0.92)
  reference frame  run cold_chain_logistics-…-34e3f5 (cold_chain_logistics):
                   held-out score 0.702 vs band [0.45, 0.80] — INSIDE

── VERDICT: ready ─────────────────────────────────────────────────────────────
  ✓ nothing essential is broken
```

Read three things from it:

* **`aegis NOT importable` is fine here.** This venv is the standalone one. The Aegis platform
  lives in a different checkout and is only needed when you actually wire the adapter in.
* **A skipped tier tells you why and how to fix it.** It never silently vanishes.
* **The realism line is the honesty check** — it refits a probe on the stored reference frame
  and states whether the score is inside the band. `--strict` turns "outside the band" into a
  non-zero exit.

Exit code 0 means the morning can proceed.

---

## 3. `contract` — is the data fit to train on?

Seconds, and it runs before anything expensive.

```bash
.venv/bin/python -m aegis_ml.cli contract \
  --adapter reference.problem \
  --data registry_store/runs/cold_chain_logistics-20260824T030131425-34e3f5/reference.parquet
```

Real output:

```
contract      PASS  (2034 rows, 11 columns)
leakage       none
learnability  0.7018  band [0.45, 0.80] — inside
contract check passed
```

Three checks in one line each: the pandera data contract, the leakage audit, and
`assert_learnable`. Exit code is non-zero if any of them fails. This is the command that
catches [chapter 03](03-the-data-problem.md)'s two failure modes in seconds instead of at demo
time.

---

## 4. `run_demo.py` — the whole thing, end to end

```bash
.venv/bin/python scripts/run_demo.py
```

It takes a few minutes (the AutoML search alone spent **171.8 s** in the committed run) and
does seven things:

1. **Generate** a seeded cold-chain world and run the domain's own quality gate over it
   (referential integrity, class coverage, temporal consistency, PII).
2. **Realism first** — `assert_learnable` and `realism_report`, printed *before* anything
   expensive.
3. `data_flow` — contract, profile, learnability, realism, leakage, three-way split, frozen
   reference frame.
4. `train_flow` — AutoML search, Optuna HPO, fit, conformal calibration, slice sweep.
5. `promote_flow` — the five-criterion gate. **A refusal here is a successful demo**, and the
   script says so and keeps going.
6. `drift_flow` — against a frame deliberately shifted the way this domain degrades: a hot
   season on longer, cheaper lanes.
7. The **secondary classification target**, measured too.

Then it writes `registry_store/RUN_SUMMARY.md`. Here is the real one, abridged — the same
numbers used throughout this track:

```
## Data
- shipments generated: 2600, labelled (received and assayed): 2034
- calibrated for an oracle R² of 0.74; i.i.d. noise σ 5.8171 percentage points,
  unobserved-confounder σ 4.7496
- realised excursion share 0.2788, realised sensor_gap_minutes missingness 0.0423
- domain quality gate: PASS

## Realism — primary target spoilage_risk_pct (regression, %)
| held-out R² (measured)                  | 0.6719 |
| oracle R² (knows the generating function)| 0.7397 |
| headroom (achieved ÷ oracle)            | 90.8%  |
| analytic R² ceiling                     | 0.7400 |
| suspiciously easy?                      | False  |
| noise-to-signal                         | 0.593  |
| unobserved confounders   | unrecorded_tarmac_delay, undocumented_precool_quality |
| confounder share of variance            | 10.4%  |
| heteroscedastic on   | transit_hours (1.48× spread, top vs bottom quartile) |
| features with NO driver                 | origin_region, payload_kg |

## Realism — secondary target excursion_flag (classification)
- held-out accuracy: 0.8468 (band [0.62, 0.92])
- majority-class rate 0.7210, so the floor is 0.7410 — the classifier clears a constant
  predictor by +0.1257

## Train
- r2 on the held-out test split: 0.7199
- conformal coverage requested 90%, achieved 91.4%
- winning tier: flaml; rows train/calibration/test: [1301, 326, 407]
- worst slice: handoff_count=q2 (1.0, 2.0] → 0.4513
```

> `run_demo.py` **promotes**, which replaces `backend/.artifacts/ml_spine.joblib` in the Aegis
> checkout (archiving what it displaced). Know that before you run it on a machine where that
> matters.

### Where the artifacts land

```
registry_store/
├── RUN_SUMMARY.md
├── index.json
├── reports/                       drift HTML + JSON, the data profile
└── runs/<run_id>/
    ├── entry.json  problem.json  manifest.json  metrics.json  split.json
    ├── model.joblib  recipe.json  leaderboard.json  gate_inputs.json  drift.json
    ├── reference.parquet  current.parquet
    ├── card.md  card.html  shap.html  profile.html
    └── visuals/  9 PNGs + index.html + interactive.html + manifest.json
```

---

## 5. Open the visual bundle

```bash
open registry_store/runs/<run_id>/visuals/index.html
```

One page, in the order of [chapter 05](05-reading-the-charts.md):

```
cold_chain_logistics — spoilage_risk_pct
  Prediction vs measured, with the conformal band
  Residuals across the prediction range
  Conformal coverage — requested vs measured, overall and by segment
  Global SHAP attribution — every declared feature
  Performance by segment
  Leaderboard — every candidate the search scored
  Realism — is this data honestly hard?
  Feature distributions and missingness
  Drift — reference vs current distributions
  Forecast with conformal band and backtest origins      ← omitted, with its reason
```

It is **fully self-contained**: images are inlined, and a scan of the file finds **zero**
`http://` or `https://` references. It opens on a laptop with no network. `interactive.html`
is the same content with interactive plots.

---

## 6. `visuals` — rebuild the charts

```bash
.venv/bin/python -m aegis_ml.cli visuals --run-id cold_chain_logistics-20260824T030131425-34e3f5
```

Real output (this took 45 s, most of it recomputing SHAP over 300 rows):

```
  ✓ cold_chain_logistics-20260824T030131425-34e3f5: 9 figures rendered, 1 omitted
      - 10_forecast.png: no forecast payload for this run — forecast_flow writes one per
        series and this run registered a tabular model, not a series

open /Users/yrevash/aegis_ml/registry_store/runs/…/visuals/index.html
```

Useful flags: `--all` (rebuild every run), `--domain-id` (restrict `--all`),
`--shap-samples` (default 300), `--open`.

---

## 7. Look at what you have

```bash
.venv/bin/python -m aegis_ml.cli registry
```

```
run_id                                         stage       metric         value   req     emp  created
--------------------------------------------------------------------------------------------------
cold_chain_logistics-20260824T030131425-34e3f5 production  r2            0.7199   90%   91.4%  2026-08-24T03:04:55.766763+00:00

registry: /Users/yrevash/aegis_ml/registry_store
```

And the model card:

```bash
.venv/bin/python -m aegis_ml.cli card --run-id <run_id> --format md
```

```
# Model card — cold_chain_logistics / cold_chain_logistics-20260824T030131425-34e3f5

## Data
- Training rows: 1301
- Calibration rows (disjoint): 326
- Held-out test rows: 407
- Dataset digest: sha256:02755eb024b48227569db71809ea4f9191212d89b41b2598986e6c9ffd6d6367

## Model
- Recipe tier: flaml
- Ensemble: xgb_limitdepth (XGBRegressor, w=1.0)

### Leaderboard (losers included — the margin is the finding)
| ridge_reference       | baseline | 0.7460 | NO  |     |
| flaml_xgb_limitdepth  | flaml    | 0.7379 | yes | yes |
| flaml_xgboost         | flaml    | 0.7286 | yes |     |
…
```

`--format html` and `--format json` also work; `--out <path>` writes to a file.

---

## 8. Watch a refusal happen

This is worth doing once, because refusing well is most of what this package does.

```bash
.venv/bin/python -m aegis_ml.cli eval --run-id cold_chain_logistics-20260824T030131425-34e3f5
```

Real output:

```
InSampleEvaluationError: Re-scoring run '…-34e3f5' with no fresh frame would measure it on
its own frozen reference frame — the WHOLE dataset, training rows included. That number is
optimistic by construction and is not evidence about unseen data, so it is not the default.
Supply fresh labelled data (frame=<DataFrame>, source=<path>, or `--data`), or pass
allow_in_sample=True to ask for the artifact-loads-and-predicts integrity check on purpose.

stage     status  secs  rows  detail
--------  ------  ----  ----  ------------------------------------------------------------
load_run  ok      0.85  -
ingest    FAILED  0.00  -     InSampleEvaluationError: Re-scoring run 'cold_chain_logisti…

eval_flow run=…-34e3f5 total=0.85s  FAILED — InSampleEvaluationError: …
```

Three things in that output are the design working:

* The message names **what is wrong, why it matters, and every way to fix it.**
* The **stage table** shows exactly where it stopped — `load_run ok`, `ingest FAILED`.
* The manifest was still written, so the lineage record survives the failure.

To get the honest number, supply fresh data with `--data`. To ask for the in-sample
integrity check on purpose, add `--allow-in-sample` — the output is then labelled `IN-SAMPLE`.
Measured on this model: in-sample **0.7587** against held-out **0.7224**.

---

## 9. Promote and drift

```bash
.venv/bin/python -m aegis_ml.cli promote --run-id <run_id>
.venv/bin/python -m aegis_ml.cli drift   --run-id <run_id> --data <current.csv|parquet>
```

`promote` prints the `GateDecision` — five criteria, each with its number — and exits non-zero
on a refusal. `--force` overrides while recording the override.

`drift` compares your frame against the run's frozen `reference.parquet`, writes
`drift.json` plus a self-contained HTML report in `registry_store/reports/`, refreshes the
visual bundle, and prints the verdict. The committed run's:

```
n_reference_rows       2034
n_current_rows         934
drifted_share          0.7
drifted_features       ambient_temp_c, carrier_tier, handoff_count, product_class,
                       route_class, sensor_gap_minutes, transit_hours
estimated_metric_name  estimated_rmse
estimated_metric_value 6.6649
verdict                block
```

Undo the last promotion:

```bash
.venv/bin/python -m aegis_ml.cli rollback --domain-id cold_chain_logistics
```

---

## 10. Every command, one table

| Command | Required flags | Notes |
|---|---|---|
| `doctor` | — | `--problem`, `--domain-id`, `--strict` |
| `init` | — | `--out`, `--target`, `--task`, `--unit`, `--templates`, `--force` |
| `contract` | `--data` | `--problem` or `--adapter` |
| `synth` | `--data` | `--rows`, `--model gaussian_copula\|ctgan` |
| `train` | — | `--problem`/`--adapter`, `--data`, `--tier` (repeatable), `--time-budget`, `--seed`, `--trainer-venv`, `--hpo/--no-hpo`, `--force`, `--resume-from`, `--full` |
| `eval` | `--run-id` | `--data`, `--allow-in-sample` |
| `promote` | `--run-id` | `--force` |
| `rollback` | `--domain-id` | — |
| `drift` | `--run-id`, `--data` | — |
| `forecast` | `--data` | `--ts-column`, `--value-column`, `--horizon`, `--freq`, `--unit`, `--level`, `--ml-candidates`, `--out` |
| `card` | `--run-id` | `--format md\|html\|json`, `--out` |
| `export` | `--run-id` | `--out`, `--validate/--no-validate` |
| `visuals` | `--run-id` **or** `--all` | `--domain-id`, `--shap-samples`, `--open` |
| `registry` | — | `--domain-id`, `--limit`, `--json` |
| `serve` | — | `--host`, `--port 8099`, `--prefix /ml`, `--reload` |

`make` shortcuts: `install`, `install-strong`, `doctor`, `audit`, `lint`, `test`, `demo`,
`clean`.

---

## 11. If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError` on `--adapter reference.problem` | the console script does not add the cwd | use `python -m aegis_ml.cli`, or `PYTHONPATH=.` |
| A tier is `skipped` | not installed in this interpreter | the reason line quotes the exact install command |
| `TabPFNLicenseError` mid-search | weights are gated behind a licence + token | see [`RESOLUTION.md`](../../RESOLUTION.md); needs a browser, so do it **before** the day |
| `LabelNotLearnableError` | the target carries no signal | [chapter 03](03-the-data-problem.md) |
| `flagged suspiciously_easy` | the target is trivially recoverable | also [chapter 03](03-the-data-problem.md) |
| `InSampleEvaluationError` | `eval` with no fresh data | supply `--data`, or opt in explicitly |
| `FrameSourceMissingError` | no frame and no way to get one | pass `--data` |
| ONNX export mismatch on rows with nulls | learned NaN routing does not survive conversion | it is off by default; see issue #7 in [`ISSUES.md`](../../ISSUES.md) |

Deeper: [`docs/09-troubleshooting.md`](../09-troubleshooting.md) and
[`ISSUES.md`](../../ISSUES.md), which records what is fixed, what is open, and how each defect
was found.

Next: [09 · Glossary](09-glossary.md)
