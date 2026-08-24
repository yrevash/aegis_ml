# 06 · MLOps — registry, promotion, drift

Module map: `aegis_ml.registry`, `aegis_ml.evaluate.gate`, `aegis_ml.monitor`, `aegis_ml.export`.

---

## 1. The registry is the filesystem

**Design rule: the filesystem registry is the source of truth. MLflow is an optional mirror. Nothing in the critical path depends on a server being up.**

```
/Users/yrevash/aegis_ml/registry_store/
├── index.json                     list[RegistryEntry], newest first — DERIVED, rebuildable
├── runs/
│   ├── <domain_id>-<UTC stamp>-<rand>/     one run, e.g. cold_chain_logistics-20260823T213346076-e44917
│   │   ├── manifest.json          RunManifest (train_flow / full_flow)
│   │   ├── manifest_eval.json  manifest_promote.json  manifest_drift.json
│   │   ├── entry.json             RegistryEntry (run_id, domain_id, created_at, stage, result, gate, paths) — AUTHORITATIVE
│   │   ├── model.joblib
│   │   ├── recipe.json
│   │   ├── leaderboard.json
│   │   ├── metrics.json
│   │   ├── card.md  card.html
│   │   ├── shap.html  profile.html
│   │   ├── problem.json  gate_inputs.json
│   │   └── reference.parquet      the frozen drift reference
│   └── <another run>/
├── optuna/studies.db              one SQLite file, one Optuna study per domain
├── _cache/                        StageCache entries, keyed on frame digest + config
├── unregistered_artifacts/        a live ml_spine.joblib the registry had never seen
└── reports/                       <domain_id>_profile.html, <run_id>_drift.html/.json,
                                   forecasts/, predictions/<run_id>.jsonl
```

`settings.registry_dir` (`AEGIS_ML_REGISTRY_DIR`) relocates the whole tree. `settings.reports_dir` defaults to `registry_store/reports`.

**`index.json` is a cache and is marked as such.** `store.reindex()` rebuilds it by walking `runs/*/entry.json`. If the two ever disagree, **the run directory wins** — which is what makes a half-finished write survivable: a directory with an `entry.json` is a complete registry row even if the process died before the index was updated.

**A run directory is immutable once written**, apart from its own `stage` field. There is no `champion/` directory and no pointer file: the champion is simply the run whose `RegistryEntry.stage` is `"production"`, and `store.champion(domain_id)` is `list_runs(stage="production", limit=1)`. Promotion flips stages and replaces one file; rollback flips them back.

`RegistryEntry.stage` is one of `"staging"` | `"production"` | `"archived"`.

Every write goes through `store.atomic_write_bytes` or `store.atomic_copy`: content is written to a uniquely-named `.tmp` sibling, `fsync`-ed, then moved with `os.replace`. A reader sees either the whole previous version or the whole new one — never a truncated joblib, which loads as a corrupt model rather than as an error.

---

## 2. Champion / challenger

| Role | `RegistryEntry.stage` | Meaning |
|---|---|---|
| **champion** | `"production"` | The model currently serving. Its `model.joblib` has been copied to `backend/.artifacts/ml_spine.joblib`. `store.champion(domain_id)` returns it, or `None`. |
| **challenger** | `"staging"` | Any newer run. It has metrics and a card but is not serving. |
| **previous** | `"archived"` | Displaced champions, newest first. `promote.rollback` restores the most recent archived run that still has a stored `model.joblib`. |

```bash
uv run aegis-ml registry                        # every run, newest first, with both coverage numbers
uv run aegis-ml registry --domain-id <id>       # restrict to one domain
uv run aegis-ml registry --json --limit 50      # machine-readable
uv run aegis-ml card --run-id <run_id> --format json   # the full TrainResult for one run
```

`registry` is a single command with options, not a group with `list`/`show`/`champion`/`diff` subcommands. For "which model is actually live right now", `registry.current_artifact_info()` answers from the **bytes** — it digests `settings.artifact_path` and matches it against each run's stored `model.joblib` — so an artifact this registry never wrote reports `run_id=None, matched_by="unmatched"` instead of a guess.

The first run for a domain has **no champion**. The gate treats that as a distinct case: criterion 1 is vacuously satisfied and `GateDecision.champion_run_id` is `None`. It is still subject to criteria 2–5 — a first model that fails its coverage check is not promoted just because it is first.

---

## 3. The promotion gate — five criteria

`aegis_ml/src/aegis_ml/evaluate/gate.py`. **A challenger is promoted only if all five hold, and each is reported with its number.** The criterion keys are the module constant `CRITERIA` — quote these, not the ordinals:

```python
CRITERIA = ("beats_champion", "coverage_meets_request", "contracts_pass",
            "worst_slice_not_worse", "no_target_leakage")
```

| `checks` key | Criterion | Threshold (`GateConfig` field) | Why |
|---|---|---|---|
| `beats_champion` | **Beats the champion on the primary metric by ≥ ε** | `min_gain`, from `settings.promote_min_gain` = `0.005` | A model that is 0.0002 better is noise. Requiring a margin stops the registry churning. Direction comes from `metrics.higher_is_better`, which **raises** `UnknownMetricError` on an unknown metric rather than assuming. |
| `coverage_meets_request` | **`empirical_coverage ≥ requested_coverage − δ`** on the held-out split | `coverage_tolerance`, from `settings.coverage_tolerance` = `0.05` | The conformal interval is the whole trust claim. A 90% interval that covers 78% of held-out truth is a lie with error bars on it. |
| `contracts_pass` | **Every pandera contract passed** | `require_contracts` | Read from `gate_inputs.json`. When that file is absent the status is **UNPROVEN**, i.e. `contract_ok=False` — a gate input that was never recorded is not a passing one. |
| `worst_slice_not_worse` | **The worst slice is no worse than the champion's worst slice** | `slice_tolerance` | The aggregate metric cannot see a model that improves on average while collapsing on one region. |
| `no_target_leakage` | **No target leakage flagged** | `settings.leakage_threshold` = `0.98` | A feature scoring 0.99 alone against the target is leakage, not skill. |

`evaluate_gate(challenger, champion, *, contract_ok, leakage, config=None)` evaluates **every** criterion even after one has failed — short-circuiting would produce a record naming the first problem and hiding the second, and the second is the one that reappears after the first is fixed. A `champion` of `None` (the first model in a domain) passes `beats_champion` and `worst_slice_not_worse` trivially **and says so in `reasons`**; the other three still apply.

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

**Who raises what.** `promote_flow` does **not** raise on a refusal: it returns the `GateDecision` so the caller can render every number, and its `apply` stage skips with the reason recorded in the manifest. The `PromotionRejectedError` lives one layer down — `evaluate.gate.promote_or_raise` and `registry.promote.promote` raise it when handed a `promoted=False` decision without `force`. The CLI turns the refusal into **exit code 2**, so a Makefile or CI step fails on a rejected promotion. Either way **the champion is unchanged**: nothing promotes silently.

`--force` promotes despite a failed gate and does not fabricate the decision — `promoted` stays as the gate computed it, every failed check keeps its reason, and `"OVERRIDE: promoted with force=True despite the failures above."` is appended to `reasons`. An override a reader cannot see is indistinguishable from a pass.

Example output:

```
$ uv run aegis-ml promote --run-id cold_chain_logistics-20260824T1102376-b77c1a

promoted: False
champion: cold_chain_logistics-20260824T0914221-a3f107
  [x] beats_champion
  [x] coverage_meets_request
  [x] contracts_pass
  [ ] worst_slice_not_worse
  [x] no_target_leakage
      min_gain = 0.005
      challenger_r2 = 0.6612
      champion_r2 = 0.6338
      gain = 0.0274
      requested_coverage = 0.9
      coverage_tolerance = 0.05
      coverage_floor = 0.85
      empirical_coverage = 0.9083
      coverage_gap = 0.0083
      contract_ok = 1
      challenger_worst_slice = 0.311
      champion_worst_slice = 0.4802
      worst_slice_delta = -0.1692
      leakage_findings = 0
  - ...one sentence per criterion, on the pass as well as the failure...
  - REJECTED: 4/5 criteria passed. All five are required — they cover different
    failure modes and none substitutes for another.
$ echo $?
2
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

**Nothing is overwritten before it is preserved.** Before the replace, `_archive_live_artifact` copies the *live* bytes into the **outgoing champion's own run directory** as its `model.joblib`, so rollback has a byte-identical copy of what was actually serving. There is no `ml_spine.previous.joblib`. If the live artifact belongs to no registered run at all — the usual case on a fresh host, where `python -m app.ml` wrote it directly — it is preserved under `registry_store/unregistered_artifacts/<stamp>-<digest>.joblib` instead, because deleting it would be the one unrecoverable step in an otherwise reversible flow.

**Result: this package needs no changes inside `aegis/` to serve a promoted model.** Aegis loads the file it always loaded.

---

## 5. Rollback

```bash
uv run aegis-ml rollback --domain-id <domain_id>
```

`rollback` is a **top-level command**, not a `registry` subcommand. `registry.promote.rollback(domain_id)` walks the domain's `archived` runs newest-first, takes the first one that still has a stored `model.joblib` (archived staging runs that never served are skipped, not failed on), archives the current champion's live bytes into its own run directory, demotes it to `"archived"`, installs the restored model over `settings.artifact_path`, and marks the restored run `"production"`. A second call therefore rolls back one *more* step rather than ping-ponging between two models.

If no archived run for the domain has a stored model, it raises `FileNotFoundError` naming every archived run it examined and stating that the serving artifact is unchanged — a rollback that silently did nothing is the worst possible outcome of a rollback.

Two properties that matter:

1. **It never deletes a run directory.** Runs are immutable; rollback moves the `production` stage and replaces one file.
2. **It does not need the gate.** Rollback is an operator action, not a measurement.

If the backend is running, restart it or call `app.ml.load()` — `get_model()` caches the artifact in a process-wide singleton and will not notice a file swap on its own.

---

## 6. Drift — with labels and without

Two tools, because neither alone is enough.

### 6.1 Evidently — data, target and prediction drift

`aegis_ml/src/aegis_ml/monitor/drift.py`, Evidently **0.7+** using the modern `Report` / `Preset` API. (Prose targeting `evidently>=0.4` describes the removed 0.4.x surface — do not follow it.)

`drift_report(...)` compares live data against the run's stored `reference.parquet`, not against a fresh generation, and writes its HTML to `<reports_dir>/<run_id>_drift.html` with a JSON side-file. `NUMERIC_TEST` and `CATEGORICAL_TEST` are module constants naming the stat tests used, so the card can say which. Produces:

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

`estimate_performance(reference, current, problem, *, run_id)` returns a dict. When the **current frame happens to carry the target column**, the realised metric is computed too and returned alongside under `realised_metric_name` / `realised_metric_value`, with `labels_present_in_current` saying which case you are in. That side-by-side is the demo — *"we said 0.61 without labels; the truth was 0.63"* — and it needs no extra flag: there is no `--with-labels`, just a current frame that includes the label column when it becomes available.

`InsufficientLabelsError` is raised when a *measurement* is asked for with too few labelled rows, and its message points at the estimator instead: *"Use aegis_ml.monitor.perf (NannyML CBPE/DLE) to ESTIMATE performance without labels instead — it says it is an estimate, which an under-powered measurement does not."*

### 6.3 What drift does and does not do

```bash
uv run aegis-ml drift --run-id <run_id> --data <live.parquet>
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

`aegis_ml/src/aegis_ml/monitor/alerts.py` turns a `DriftReport` into a list of `Alert(level, ...)` via `evaluate_alerts`, and `raise_if_blocking` is what the **promotion** path calls to refuse. Levels map to actions, not to feelings: `info` is recorded and gates nothing (usually "not enough current rows to trust this"); `warn` is visible in the console and the model card while serving and promotion both continue; `block` refuses promotion and leaves serving untouched. There is no pager integration and there should not be one in a 24-hour build.

This module writes **no** database rows — `registry/db.py`'s only in-package caller is `monitor/log.py` (§6.5). Persisting a drift report relationally is `db.insert_drift_report`, called by the host if it wants it.

### 6.5 Prediction logging

`aegis_ml/src/aegis_ml/monitor/log.py` appends every served prediction to `registry_store/reports/predictions/<run_id>.jsonl` (`log_path(run_id)`). What is written by default is the **`feature_digest`**, not the raw feature vector — `feature_digest()` canonicalises and SHA-256s it — because a prediction log is a copy of production inputs and production inputs carry PII. Callers that need the values for a drift reference opt in with `store_features=True`. The file sink is written first and **always**; the `ml_predictions` mirror follows only when `settings.postgres_dsn` is configured, and a failure there *raises* rather than being dropped — a monitoring pipeline that silently loses half its rows reports a traffic drop that never happened. `log_prediction_async` is the same contract awaited on a FastAPI handler's loop, and `compact_to_parquet` / `read_log` turn the JSONL back into a frame. That frame is the `--data` input to `aegis-ml drift` and the input to NannyML. Without it, drift monitoring has nothing to monitor.

---

## 7. The optional MLflow mirror

```bash
export AEGIS_ML_ENABLE_MLFLOW=1
uv run aegis-ml train
mlflow ui --backend-store-uri sqlite:///registry_store/mlflow.db
```

`aegis_ml/src/aegis_ml/registry/mlflow_mirror.py` exposes `mirror_enabled()` and `mirror(entry)`, logging params, metrics, the card, the leaderboard and the artefacts to MLflow 3 **after** the filesystem registry has written them. `model.joblib` is deliberately **not** uploaded: it is large, it is already durable in the run directory, and uploading it invites the belief that MLflow is where the model lives. Promotion copies from `runs/<run_id>/model.joblib`.

**It is a mirror, never the source of truth.** When mirroring is off, `mirror()` logs one line saying so and returns `None` — a no-op with a receipt, so a user who expected a mirror finds out from the log rather than from an empty UI. When it is on and MLflow fails part-way through, it raises `MirrorFailedError`, whose message states plainly that the registry is unaffected; a pipeline that wants mirroring to be strictly best-effort catches *that type* explicitly. Nothing here catches it on the caller's behalf, because a silent fallback is the house rule this package exists to avoid.

Value: a good-looking UI for lineage and run comparison, one `pip install`, and it works against the Postgres you already have.

---

## 8. The optional Postgres tables

Nothing about ML is persisted relationally in Aegis today. `aegis_ml/src/aegis_ml/registry/db.py` adds three tables. **The base class is chosen at call time and reported, never guessed**: when `aegis.data` is importable the tables register on `AegisBase` (so the host's own `create_all` materialises them alongside every other Aegis table and they inherit its tenant scoping); standalone, a local `DeclarativeBase` named `aegis_ml.registry.db.MLBase` is used. Which one happened is on `MLTables.base_origin`, because *"which metadata did my tables land on"* is exactly the question a puzzling `create_all` raises.

| Table | Columns |
|---|---|
| `ml_runs` | `run_id` (**PK**, the same id as the run directory), `domain_id`, `tenant_id`, `created_at`, `stage`, `task`, `target`, `metric_name`, `metric_value`, `requested_coverage`, `empirical_coverage`, `dataset_digest`, `detail` |
| `ml_predictions` | `id`, `ts`, `tenant_id`, `run_id`, `feature_digest`, `prediction`, `prediction_label`, `interval_low`, `interval_high`, `detail` |
| `ml_drift_reports` | `id`, `ts`, `tenant_id`, `run_id`, `drifted_share`, `dataset_drift`, `verdict`, `detail` |

Four details that look like details and are not:

- **One `detail` JSON column per table, not a column per nested structure.** The recipe, leaderboard, gate, slices and per-feature drift scores all live there. It is `jsonb` on PostgreSQL and portable `JSON` everywhere else, which is what keeps `create_all` working on the SQLite database a test uses.
- **`requested_coverage` and `empirical_coverage` are two columns, never one.** A dashboard that `SELECT`s a single `coverage` cannot tell a reader whether the interval delivered what it promised.
- **`prediction` and `prediction_label` are separate**, because squeezing a class label into a float column via a class index is how a monitoring query silently starts averaging category codes.
- **`tenant_id` is a plain indexed column with no cross-package foreign key**, mirroring `aegis.ops.models.EvalResult`: isolation is enforced at the query/RLS layer, not by DDL that would couple two packages' migrations.

`insert_run` uses `session.merge`, not `add`, so mirroring the same run twice (once at training time, again after promotion flips its stage) converges instead of raising on the primary key. Timestamps default server-side, so a row's `ts` is the database's clock rather than one of N application clocks.

Enabled by setting `AEGIS_ML_POSTGRES_DSN`. **Off by default** — the filesystem registry is complete without them. `async_dsn()` rewrites a sync `postgresql://` DSN (which is what the backend's own config carries) to `postgresql+asyncpg://` and says so in the log, rather than letting `create_async_engine` fail several frames deep in a way that reads like a missing dependency.

SQLAlchemy is never imported at module scope — `aegis_ml.registry` is on the light CLI's import path, and listing runs must not require a database driver. `sqlalchemy[asyncio]` and `aiosqlite` are in the **`[dev]` extra** so the module is exercisable in a checkout; they are deliberately *not* in `[serve]`, because the relational mirror is for hosts that already run a database.

> **Invariant 3 applies.** *"Every tenant-scoped table must be covered by `aegis.governance.rls`."* If you enable these tables, add them to the RLS plan in the same change. `audit_rls_enforcement` will report them as unprotected otherwise, and a table storing tenant data outside the plan is a cross-tenant leak.

---

## 9. ONNX export

```bash
uv run aegis-ml export --run-id <run_id> [--out model.onnx] [--no-validate]
```

`aegis_ml/src/aegis_ml/export/onnx.py`. Three public names: `to_onnx`, `validate_roundtrip` and `register_converters` — the last is called explicitly rather than relied upon, because skl2onnx ships shape calculators for sklearn's own estimators but XGBoost/LightGBM members need `onnxmltools`' converters wired in before `convert_sklearn` will accept a pipeline containing them. `validate_roundtrip` measures the largest disagreement between the fitted sklearn model and the exported graph on real rows and refuses above `DEFAULT_TOLERANCE`. An export nobody round-tripped is a file whose predictions have never been compared to anything.

**Two limits found by running it, not by reading about it** (`ISSUES.md` §7):

- **Learned NaN routing does not survive conversion.** sklearn's tree learners route NaN down a direction they *learned* while fitting; the ONNX `TreeEnsemble` op does not reproduce that for every member type. The same RandomForest measured `2.4e-06` max abs difference on complete rows and **`17.7`** on rows containing NaNs — and the generated frames deliberately carry ~4% MAR missingness, so this is not a corner case. Impute before the model if the export has to be faithful.
- **`HistGradientBoosting` will not convert** on skl2onnx 1.20 + current onnx: a `TypeError` deep inside the TreeEnsemble attribute encoding, which `to_onnx` catches and re-raises with the member list and both likely causes attached.

ONNX is **off by default** in `config/pipeline.toml` (`export_onnx = false`) and nothing in Aegis serves ONNX, so both limits are documented rather than blocking. If it is ever turned on, the NaN behaviour must be surfaced on the model card.

> **MAPIE intervals and SHAP attributions do not export.** ONNX gives you the point prediction and nothing else. The value is a portable point-predictor and the round-trip validation, **not a new serving path** — swapping serving onto ONNX would silently drop the conformal interval and the explanation, which are the two things Aegis exists to provide. `onnxruntime` (CPU) is already present in the backend venv transitively via `fastembed`, and the spine's in-process predict is already sub-millisecond, so there is no speedup waiting to be claimed either.

This is also the escape hatch for a non-portable AutoML winner (`docs/05-ml-pipelines.md` §4): export it to ONNX, run it side by side, quote its number as the ceiling, and keep serving the portable ensemble that can explain itself.

---

## 10. The demo narrative this section supports

In order, each backed by a real artefact:

1. *"Here is the data contract — declared ranges, declared categorical levels, enforced with pandera before anything trains."* → `problem.json` + `gate_inputs.json`, and the `contract` stage in `manifest.json`.
2. *"Here is the search — four tiers, every candidate reported, winners and losers."* → `leaderboard.json`.
3. *"The strongest model was not portable, so we report it as the ceiling and promote the one that can explain itself."* → `Recipe.notes` (the `ACCURACY CEILING:` line), `Candidate.portable`.
4. *"The interval is calibrated, not asserted: we asked for 90% and measured 91.2% on held-out data."* → `metrics.json` → `requested_coverage` vs `empirical_coverage`, and `card.coverage` on the card.
5. *"The promotion gate rejected the previous challenger because one slice collapsed."* → a real `GateDecision` in `entry.json` with `promoted: false` and `worst_slice_not_worse: false`.
6. *"We can estimate live performance before any labels arrive."* → NannyML CBPE/DLE, `estimated_metric_*`.
7. *"And when the labels did arrive, the estimate was within 0.02."* → `realised_metric_*` beside it.
8. *"Rollback is one command and never deletes a run."* → `aegis-ml rollback --domain-id <id>`.

---

## 11. Next

`docs/07-integration-with-aegis.md`.
