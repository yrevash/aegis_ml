# 06 · MLOps — registry, promotion, drift

Module map: `aegis_ml.registry`, `aegis_ml.evaluate.gate`, `aegis_ml.monitor`, `aegis_ml.export`.

---

## 1. The registry is the filesystem

**Design rule: the filesystem registry is the source of truth. MLflow is an optional mirror. Nothing in the critical path depends on a server being up.**

```
/Users/yrevash/aegis_ml/registry_store/
├── index.json                     domain_id → {champion, challengers[], history[]}
├── runs/
│   ├── 20260824T0914-a3f1/        one immutable run
│   │   ├── manifest.json          RunManifest
│   │   ├── entry.json             RegistryEntry (run_id, domain_id, created_at, stage, result, gate, paths)
│   │   ├── model.joblib
│   │   ├── recipe.json
│   │   ├── leaderboard.json
│   │   ├── card.json  card.html
│   │   ├── shap.html  profile.html
│   │   ├── slices.json
│   │   ├── drift_ref.parquet
│   │   └── drift.html
│   └── 20260824T1102-b77c/
├── champion/
│   ├── <domain_id>.json           → points at a run_id
│   └── <domain_id>.previous.json  → the one before it, for rollback
├── hpo/<domain_id>.db             Optuna SQLite study
├── cache/frame-<digest>.parquet
└── reports/                       drift reports over time
```

`settings.registry_dir` (`AEGIS_ML_REGISTRY_DIR`) relocates the whole tree. `settings.reports_dir` defaults to `registry_store/reports`.

**A run directory is immutable once written.** Promotion never rewrites a run; it repoints `champion/<domain_id>.json`. That is what makes rollback a one-line operation.

`RegistryEntry.stage` is one of `"staging"` | `"production"` | `"archived"`.

---

## 2. Champion / challenger

| Role | Meaning |
|---|---|
| **champion** | The model currently serving. `champion/<domain_id>.json` names its run, and its `model.joblib` has been copied to `backend/.artifacts/ml_spine.joblib`. |
| **challenger** | Any newer run in `staging`. It has metrics and a card but is not serving. |
| **previous** | The champion before the current one. Kept for rollback, always. |

```bash
uv run aegis-ml registry list                 # every run, newest first, with stage and metric
uv run aegis-ml registry show <run_id>        # the full RegistryEntry
uv run aegis-ml registry champion             # which run is serving, and since when
uv run aegis-ml registry diff <a> <b>         # side-by-side metrics, slices, leaderboard
```

The first run for a domain has **no champion**. The gate treats that as a distinct case: criterion 1 is vacuously satisfied and `GateDecision.champion_run_id` is `None`. It is still subject to criteria 2–5 — a first model that fails its coverage check is not promoted just because it is first.

---

## 3. The promotion gate — five criteria

`aegis_ml/src/aegis_ml/evaluate/gate.py`. **A challenger is promoted only if all five hold, and each is reported with its number.**

| # | Criterion | Threshold | Why |
|---|---|---|---|
| 1 | **Beats the champion on the primary metric by ≥ ε** | `settings.promote_min_gain` = `0.005` | A model that is 0.0002 better is noise. Requiring a margin stops the registry churning. Direction comes from `metrics.higher_is_better`, which **raises** on an unknown metric rather than assuming. |
| 2 | **`empirical_coverage ≥ requested_coverage − δ`** on the held-out split | `settings.coverage_tolerance` = `0.05` | The conformal interval is the whole trust claim. A 90% interval that covers 78% of held-out truth is a lie with error bars on it. |
| 3 | **Every pandera contract passes** | — | The training frame matched its declared schema: dtypes, ranges, nulls and the categorical level sets. |
| 4 | **The worst slice is no worse than the champion's worst slice** | same ε | The aggregate metric cannot see a model that improves on average while collapsing on one region. |
| 5 | **No target leakage flagged** | `settings.leakage_threshold` = `0.98` | A feature scoring 0.99 alone against the target is leakage, not skill. |

Returns:

```python
class GateDecision(BaseModel):
    promoted: bool
    challenger_run_id: str
    champion_run_id: str | None = None
    reasons: list[str]              # populated on a PASS as well as a failure
    checks: dict[str, bool]
    metrics: dict[str, float]
```

> `reasons` is populated on a pass too. **"Promoted" with no figures is as opaque as "rejected" with no figures**, and the model card quotes both.

On failure, `promote_flow` raises `PromotionRejectedError` listing every failed criterion with its measured number, and **the champion is unchanged**. It refuses loudly; it never promotes silently.

Example output:

```
$ uv run aegis-ml promote --run 20260824T1102-b77c

Gate decision for 20260824T1102-b77c (champion 20260824T0914-a3f1)

  [PASS] metric          r2 0.6612 vs champion 0.6338   (+0.0274 >= 0.0050)
  [PASS] coverage        empirical 0.9083 vs requested 0.9000   (-0.0083 <= 0.0500)
  [PASS] contract        pandera: 9 columns, 0 violations
  [FAIL] worst_slice     region=latam  r2 0.3110 vs champion's worst 0.4802   (-0.1692)
  [PASS] leakage         max single-feature score 0.5412 < 0.9800

PromotionRejected: the champion is unchanged:
  - worst slice `region=latam` r2 0.3110 is 0.1692 below the champion's worst slice
```

---

## 4. Promotion writes exactly where Aegis loads from

```
registry_store/runs/<run_id>/model.joblib
        │  atomic replace (write to a temp file in the same directory, then os.replace)
        ▼
/Users/yrevash/aegis/backend/.artifacts/ml_spine.joblib
```

`settings.artifact_path` computes this from `settings.aegis_root`. That path is read off `backend/src/app/ml/__init__.py`'s own constant, which resolves to `backend/.artifacts/ml_spine.joblib`.

> **This is deliberately NOT the library path** `aegis/src/aegis/ml/artifacts/`. `app.ml.get_model()` loads from the host directory. Training through the library constant writes the artifact where nothing loads from, and the endpoints keep answering 503 — with the two paths differing by a directory nobody looks at. `backend/src/app/ml/__main__.py` carries a comment about exactly this.

The previous artifact is retained as `ml_spine.previous.joblib` before the replace.

**Result: this package needs no changes inside `aegis/` to serve a promoted model.** Aegis loads the file it always loaded.

---

## 5. Rollback

```bash
uv run aegis-ml registry rollback --domain <domain_id>
```

Repoints `champion/<domain_id>.json` at `champion/<domain_id>.previous.json`, restores `ml_spine.previous.joblib` over `ml_spine.joblib`, marks the demoted run `archived`, and prints both run ids and both metric values.

Two properties that matter:

1. **It never deletes a run directory.** Runs are immutable; rollback is a pointer move.
2. **It does not need the gate.** Rollback is an operator action, not a measurement.

If the backend is running, restart it or call `app.ml.load()` — `get_model()` caches the artifact in a process-wide singleton and will not notice a file swap on its own.

---

## 6. Drift — with labels and without

Two tools, because neither alone is enough.

### 6.1 Evidently — data, target and prediction drift

`aegis_ml/src/aegis_ml/monitor/drift.py`, Evidently **0.7+** using the modern `Report` / `Preset` API. (Prose targeting `evidently>=0.4` describes the removed 0.4.x surface — do not follow it.)

Compares live data against the run's stored `drift_ref.parquet`, not against a fresh generation. Produces:

```python
class DriftReport(BaseModel):
    run_id: str
    reference_digest: str | None
    n_reference_rows: int
    n_current_rows: int
    dataset_drift: bool
    drifted_share: float          # 0..1
    drifted_features: list[str]
    target_drift: float | None
    prediction_drift: float | None
    estimated_metric_name: str | None
    estimated_metric_value: float | None
    verdict: Literal["pass", "warn", "block"]
    html_report_path: str | None
```

Verdict thresholds: `settings.drift_share_warn = 0.2`, `settings.drift_share_block = 0.4`.

> **Evidently requires ground-truth labels for its performance metrics.** Its drift metrics do not, but "has the model got worse?" does. That is the gap NannyML fills.

### 6.2 NannyML — performance **without** ground truth

`aegis_ml/src/aegis_ml/monitor/perf.py`. **CBPE** (Confidence-Based Performance Estimation, classification) and **DLE** (Direct Loss Estimation, regression) estimate live performance *before ground truth arrives*.

This is the strongest single differentiator in the stack for an enterprise-trust demo, and it fits Aegis's thesis exactly: labels arrive days later, and "we cannot tell you how the model is doing until Friday" is not an answer an operations team accepts.

**Naming discipline.** Everything it produces is spelled `estimated_*` — `estimated_metric_name`, `estimated_metric_value` — so it can never be read as a measurement. Compare `aegis_ml.contracts.protocols.DriftReport`, where the field description says so explicitly.

When labels *do* arrive, `aegis-ml drift --with-labels <path>` computes the realised metric and prints it **next to** the earlier estimate. That side-by-side is the demo: "we said 0.61 without labels; the truth was 0.63."

`InsufficientLabelsError` is raised when a *measurement* is asked for with too few labelled rows, and its message points at the estimator instead: *"Use aegis_ml.monitor.perf (NannyML CBPE/DLE) to ESTIMATE performance without labels instead — it says it is an estimate, which an under-powered measurement does not."*

### 6.3 What drift does and does not do

```bash
uv run aegis-ml drift --run <run_id> --current <live.parquet>
```

| Verdict | `drifted_share` | Consequence |
|---|---|---|
| `pass` | < 0.20 | Nothing. |
| `warn` | 0.20 – 0.40 | Logged, shown on the card and in the console. Serving continues. |
| `block` | > 0.40 | `DriftThresholdExceededError`. **Blocks promotion** of anything calibrated on the drifted reference. |

> **Drift never withdraws the serving model.** From the error's own message: *"The serving model is NOT withdrawn — Aegis serves the model it has and flags it. This blocks PROMOTION of anything calibrated on the drifted reference."*
>
> That is the right behaviour and it is worth saying out loud in a demo. Withdrawing a model on a drift signal turns a quality warning into an outage, and drift detectors fire on sampling noise. See `docs/09-troubleshooting.md` for how to tell real drift from a small-sample artefact.

### 6.4 Alerts

`aegis_ml/src/aegis_ml/monitor/alerts.py` renders a verdict into a structured record and, when `settings.postgres_dsn` is set, writes it to `ml_drift_reports` (§8). There is no pager integration and there should not be one in a 24-hour build.

### 6.5 Prediction logging

`aegis_ml/src/aegis_ml/monitor/log.py` appends every served prediction — inputs, prediction, interval, top SHAP drivers, `run_id`, timestamp — to a JSONL file under `registry_store/reports/predictions/`. That file is the `--current` input to `drift_flow` and the input to NannyML. Without it, drift monitoring has nothing to monitor.

---

## 7. The optional MLflow mirror

```bash
export AEGIS_ML_ENABLE_MLFLOW=1
uv run aegis-ml train --tier all
mlflow ui --backend-store-uri sqlite:///registry_store/mlflow.db
```

`aegis_ml/src/aegis_ml/registry/mlflow_mirror.py` logs params, metrics, the card, the leaderboard and the artefacts to MLflow 3 **after** the filesystem registry has written them.

**It is a mirror, never the source of truth.** If MLflow is unreachable, the mirror logs a warning and the run continues, because nothing downstream reads from it. That is the one place in this package where a failure is tolerated rather than raised, and it is tolerated because the mirror is *by definition* redundant — the same information is already on disk.

Value: a good-looking UI for lineage and run comparison, one `pip install`, and it works against the Postgres you already have.

---

## 8. The optional Postgres tables

Nothing about ML is persisted relationally in Aegis today. `aegis_ml/src/aegis_ml/registry/db.py` adds three tables on `AegisBase`, so they inherit the platform's tenant scoping and RLS policy:

| Table | Columns (abbreviated) |
|---|---|
| `ml_runs` | `run_id`, `tenant_id`, `domain_id`, `created_at`, `stage`, `task`, `target`, `metric_name`, `metric_value`, `requested_coverage`, `empirical_coverage`, `dataset_digest`, `recipe_json`, `gate_json` |
| `ml_predictions` | `id`, `tenant_id`, `run_id`, `created_at`, `features_json`, `prediction`, `interval_low`, `interval_high`, `shap_json`, `trace_id` |
| `ml_drift_reports` | `id`, `tenant_id`, `run_id`, `created_at`, `drifted_share`, `drifted_features_json`, `estimated_metric_name`, `estimated_metric_value`, `verdict` |

Enabled by setting `AEGIS_ML_POSTGRES_DSN`. **Off by default** — the filesystem registry is complete without them.

> **Invariant 3 applies.** *"Every tenant-scoped table must be covered by `aegis.governance.rls`."* If you enable these tables, add them to the RLS plan in the same change. `audit_rls_enforcement` will report them as unprotected otherwise, and a table storing tenant data outside the plan is a cross-tenant leak.

---

## 9. ONNX export

```bash
uv run aegis-ml export --onnx --run <run_id>
```

`aegis_ml/src/aegis_ml/export/onnx.py`. Converts the fitted point-predictor via `skl2onnx` / `onnxmltools` and round-trip-validates it: the ONNX predictions must match the sklearn predictions to a stated tolerance, or the export raises rather than shipping a silently different model.

XGBoost→ONNX inference measures around **0.029 ms/request** on an Apple M1, and `onnxruntime` (CPU) is already present in the backend venv transitively via `fastembed`.

> **The caveat, carried honestly in the docstring and in the model card: MAPIE intervals and SHAP attributions do not export.** ONNX gives you the point prediction and nothing else. The value is a portable point-predictor and the round-trip validation, **not a new serving path** — swapping serving onto ONNX would silently drop the conformal interval and the explanation, which are the two things Aegis exists to provide.

This is also the escape hatch for a non-portable AutoML winner (`docs/05-ml-pipelines.md` §4): export it to ONNX, run it side by side, quote its number as the ceiling, and keep serving the portable ensemble that can explain itself.

---

## 10. The demo narrative this section supports

In order, each backed by a real artefact:

1. *"Here is the data contract — nine columns, declared ranges, declared categorical levels, enforced with pandera before anything trains."* → `card.json`, the contract stage.
2. *"Here is the search — four tiers, every candidate reported, winners and losers."* → `leaderboard.json`.
3. *"The strongest model was not portable, so we report it as the ceiling and promote the one that can explain itself."* → `Recipe.notes`, `Candidate.portable`.
4. *"The interval is calibrated, not asserted: we asked for 90% and measured 90.8% on held-out data."* → `conformal_coverage` vs `conformal_coverage_empirical`.
5. *"The promotion gate rejected the previous challenger because one slice collapsed."* → a real `GateDecision` with `promoted: false`.
6. *"We can estimate live performance before any labels arrive."* → NannyML CBPE/DLE.
7. *"And when the labels did arrive, the estimate was within 0.02."* → the side-by-side.
8. *"Rollback is one command and never deletes a run."* → `aegis-ml registry rollback`.

---

## 11. Next

`docs/07-integration-with-aegis.md`.
