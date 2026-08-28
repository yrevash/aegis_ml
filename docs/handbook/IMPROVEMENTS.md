# `aegis_ml` — prioritised improvements

*Audit date 2026-08-28. Every "what it does today" below was read in the source or reproduced
by running it; nothing here is inferred from documentation. Line numbers are as of commit
`a7a1bda`.*

`aegis_ml` is in unusually good shape for a package of its age. The dependency forensics are
real, the two-venv split is justified rather than assumed, `ruff` is clean across 73 source
files, the no-mocks audit passes with 17 reason-carrying opt-outs (I read all 17; 14 are sound, and the
three I would not sign off — `recipe.py:357`, `contract_check.py:220` and `:292` — appear as rows
15, 1 and 18 below), and 323 tests pass. Where the package is strong it is genuinely strong: `store.atomic_write_bytes`
is a correct temp+fsync+`os.replace` on the same directory with a post-install digest check;
`monitor/perf.py` never once drops the `estimated_` prefix; the RNG-stream collision that cost
the original session a day is pinned by a named regression test. The weakness is not in the
controls — it is in the seams *between* them. The package's thesis is "no silent fallbacks",
and the residual violations cluster in exactly one shape: a control that **did not run** is
handed to the next stage as an empty list, a `None`, or a default, and the next stage reads
that as **"ran, and was clean."** The promotion gate does this with the leakage audit; the
search-stage cache does it with a key that omits the split; `hpo.tune` does it with a resumed
Optuna study. The second weakness is coverage: the two largest modules in the package —
`pipelines/flows.py` (2,547 lines, all seven flows) and `cli.py` (1,543 lines, 17 commands) —
are the entire user-facing surface, and between them the test suite executes one function.
Neither weakness is architectural; both are a day's work to close.

---

> **Row 1 was fixed during this audit.** `_check_leakage` now takes
> `Sequence[object] | None`; `None` means the audit never ran and fails criterion 5 with
> *"UNPROVEN is not PASS"*. `promote_flow` no longer defaults a missing key to `[]`.
> Regression test: `tests/test_promotion_gate.py::test_leakage_audit_that_never_ran_is_not_a_pass`.
> 323 tests still pass. Everything else below is open.

## The table

| # | Area | What it does today | What I suggest | What that improves | Effort |
|---|---|---|---|---|---|
| 1 ✅ | Promotion gate — **FIXED 2026-08-28** | `evaluate/gate.py:441` sets `checks["no_target_leakage"] = not findings`, so an empty list means "audited, clean". Three producers hand it an empty list when the audit never ran: `data/contract_check.py:290-295` (records a *warning*, leaves `report.leakage = []`), `pipelines/flows.py:310-312` (`DataBundle.leakage` defaults to `[]` while `contract_ok` is required), and `flows.py:1775` `leakage=list(gate_inputs.get("leakage", []))` — directly under a note at `flows.py:1766` claiming "contract and leakage status are unknown, so **both** are treated as UNPROVEN". Only `contract_ok` is. | Change the gate parameter to `leakage: Sequence[object] \| None`, `None` ⇒ criterion 5 **not met**, mirroring how `empirical_coverage=None` is already handled at `gate.py:318`. Make `DataBundle.leakage` a required field. Have `contract_check._run_leakage` record the failure as an *issue*, not a warning — copying the `audit-ok` at `contract_check.py:235`, which already gets this right. | Removes the one path by which a model can be promoted 5/5 with the sentence "PASS no_target_leakage: the feature audit flagged nothing" printed about an audit that did not happen. This is the package's own `_validate_run_id` failure repeated inside the gate it exists to protect. | M |
| 2 | Pipeline caching | The AutoML search stage's cache key (`pipelines/flows.py:1033-1039`) is `[frame digest, tiers, budget, seed, use_trainer_venv]`. The stage consumes `bundle.train`, produced by `three_way_split(test_size=…, calibration_size=…)` at `flows.py:1016`. **Neither split fraction is in the key**, and nothing from `problem` beyond column names — so `primary_metric` and `requested_coverage` are invisible to it. | Add `test_size`, `calibration_size` and a digest of `problem.model_dump()` to the key parts. Separately, replace `default=repr` in `manifest.content_key` (`pipelines/manifest.py:97`) with a hard failure on a non-JSON part — `repr(DataFrame)` is a truncated preview, so two different frames hash identically. | Prevents `train_flow` re-run with a changed split or a changed `primary_metric` from adopting a recipe and leaderboard selected on different data under a different ranking, while the manifest honestly reports `status="cached"`. Today the registered `leaderboard.json` can quote scores never measured on the run's own frame. | M |
| 3 | Registry rollback | `registry/promote.py:304` picks the rollback target from `store.list_runs(stage="archived")`, ordered by **`created_at`** (`store.py:650`). `RegistryEntry` has no `archived_at` or `promoted_at`, so "the run I just displaced" is unrecoverable. Reproduced — promote C, then A, then B; rollback restores **C**, not A, and logs "restored run run-c" with no anomaly. `tests/test_registry.py:288` promotes in creation order, the one ordering where the bug cannot fire. | Stamp `promotion_seq` (or `archived_at`) onto `RegistryEntry` inside `promote.set_stage(previous, "archived")` and order rollback candidates by it. Cheaper: append to `registry_store/promotions.jsonl` and read the last displaced run from the ledger. | `rollback` is the one-command escape hatch on demo day. Today, any promotion sequence that is not also creation-ordered — which is what backfilling or re-promoting an older run produces — silently reinstates the wrong model into `ml_spine.joblib`. | M |
| 4 | HPO | `automl/hpo.py:95` keys the shared Optuna study on `domain::target::metric` only — nothing about the recipe or its members — and `load_if_exists=True` makes `study.best_trial` the best across *all previous invocations*. `_params_from_trial` recovers params by `f"{member.name}__"` prefix and on a miss silently returns `{**member.params}`. Reproduced: tuning an XGB recipe then a HistGB recipe on the same problem returns the **second recipe byte-identical**, annotated `"Tuned recipe scored r2=0.6563 … an improvement of 0.0654"` — a score belonging to the other model. That note reaches `Recipe.notes` and the model card. | Fold a fingerprint of member kinds and names into `study_name_for`. Make `_params_from_trial` raise when the prefix recovers nothing, rather than returning the untuned params. | Stops the tuner from certifying an untuned recipe as improved using another model's number. The trigger is on the hackathon path: `aegis-ml tune` run twice on one domain after the search picked a different winner — which it does, per ISSUES #22. | M |
| 5 | Test coverage | Function-level tracing of the full suite: `pipelines/flows.py` executes **1 of 57** functions (`realism_band_for`); `cli.py` **0 of 32** in-process — `tests/test_meta.py` shells out to `doctor` and `--help`, so 15 of 17 commands are never invoked. `tests/test_pipeline_end_to_end.py` does not import `flows` at all; it reassembles the pipeline by hand from components. `pipelines/manifest.py` (618 lines, the stage cache) has **no test file** — `grep -rl "StageCache\|content_key" tests/` is empty. `evaluate/cv.py` (617 lines, the nested-CV and temporal-shuffle refusals) has zero references. `features/pipeline.column_transformer`, whose docstring calls the Aegis mirror "load-bearing", has zero test references. Suite-wide: 204/717 functions executed (28.5%). | Add `tests/test_flows_smoke.py` marked `@pytest.mark.slow` that runs `full_flow` on a 300-row fixture and asserts the artifact set, the manifest stage statuses and the gate verdict; a `CliRunner`-based `tests/test_cli.py` covering every command's argument parsing and exit codes; `tests/test_manifest.py` for cache hit/miss, `content_key` stability, `SkipStage` and `optional→degraded`; a `cv.py` file asserting temporal-shuffle refusal and that nested CV scores below flat best-of-k on noise; and a characterisation test pinning `column_transformer`'s emitted column names and order. | Findings 1, 2, 4, 7, 10 and 21 in this table all live in code the suite never executes, and each is reproducible in under a second. 4,090 lines — the flows and the CLI — are the only things a user on hackathon morning actually touches, and they are the only things not regression-tested. | L |
| 6 | Serve tools | `serve/router.py:173` is `return _health_snapshot(None)`, and `serve/tools.py:584` gates both the champion and drift lookups behind `if domain_id:`. `GET /ml/health` therefore returns exactly `['artifact', 'fix_command', 'model_available', 'served_model']` — no drift, no champion. `router.py:191` documents it as returning the drift report and `router.py:333-340` writes three sentences on how a drifted model answers 200 "with its verdict attached". Separately, `tools.py:674` only mentions drift in the summary when `drift.get("verdict")` exists, so the "unavailable: no drift report recorded" string set at `tools.py:614` never reaches the sentence a planner reads. | Add a `domain_id` query parameter to the endpoint and thread it into `_health_snapshot`; default it to the single registered domain when there is exactly one. In `_compose_summary`, render the unavailable-string case explicitly rather than skipping it. | `check_model_health` is the tool whose own description (`tools.py:736-740`) tells a planner to call it *before citing any prediction as evidence*. Today it returns a healthy-looking 200 containing no health information, which is worse than returning nothing. | S |
| 7 | Promotion gate | `evaluate/gate.py:187-195`: when `champion is None`, criterion 1 returns PASS before any validity check on `challenger.metric_value`. Verified — a `NaN` metric with no champion is promoted 5/5. It then becomes the champion, and every later challenger computes `gain = x - nan → nan`, `nan >= min_gain → False`, so the domain becomes permanently un-promotable with no error anywhere. Related: `contracts/protocols.py:131-152` puts no bounds on `TrainResult` at all — `empirical_coverage=1.5, test_size=0` also promotes 5/5, while `CoverageReport` at `calibration.py:145` *does* carry `ge=0.0, le=1.0`. | Add `if not math.isfinite(challenger.metric_value): FAIL` at the top of `_check_metric`, before the champion branch. Mirror `CoverageReport`'s `Field` bounds onto `TrainResult.empirical_coverage`, `requested_coverage` and the three size fields. | Turns a first-run NaN — the normal outcome of an all-constant target or an empty test split — from a silent promotion that bricks the domain into a named refusal. The type that feeds the decision should be at least as constrained as the type that reports it. | S |
| 8 | Realism guarantee | `config/contracts.toml` exposes `suspiciously_easy_r2` / `suspiciously_easy_accuracy`, and `config.py:44-45` maps both into `Settings` — so `doctor` counts them as consumed and does **not** list them under NOT CONSUMED. But `data/latent._resolved_ceiling` (`latent.py:1646-1650`) returns the module constants and never reads `settings`. Verified: `AEGIS_ML_SUSPICIOUSLY_EASY_R2=0.30` leaves the ceiling at `0.95`. `_resolved_floor` twenty lines above *does* read settings. These two are the only dead pair among the 16 mapped settings. | Read `settings.suspiciously_easy_r2` / `_accuracy` in `_resolved_ceiling`. Then add a meta-test asserting that every field named in `config.TOML_TO_SETTING` has at least one `settings.<field>` reader in `src/` — `unknown_keys()` catches *unmapped* keys; nothing catches *mapped-but-unread* ones. | ISSUES #17 was "config files nothing reads". This is the same defect one layer down, in the half of the realism band that guards against a too-easy label — the failure this package was built to prevent — and in the exact form the #17 mitigation is blind to. | S |
| 9 | Strong tier | `automl/strong.py:451-452` does an unconditional `shutil.rmtree(destination)` on `runs/<id>/strong/`, and `run_search` calls `save_strong_model` once from `_search_autogluon` (`search.py:863`) and again from `_search_tabpfn` (`search.py:1006`). Reproduced: after the second save the first tier's artifact is gone, while `search.py:890` has already stamped `detail["persisted_to"]` onto the AutoGluon candidate, so the leaderboard names a file that no longer exists and `verify_strong` verifies the other tier's number. | Namespace the directory as `strong/<tier>/`, or refuse to overwrite a directory whose `manifest.json` names a different tier. | Does not fire today only because TabPFN is licence-gated — it fires the moment the operator completes the one-time setup ISSUES #11 tells them to do *before* the day. The "measured, not asserted" ceiling then silently belongs to a different model. | S |
| 10 | Conformal coverage | `pipelines/flows.py:783-802`: rows whose label is absent from `model.classes_` are dropped from the calibration score set (`if j is not None`) and from the empirical-coverage denominator (`if j is None: continue`), with no note. The note that *is* emitted reports `scores.size` — the post-filter count — so a reader cannot tell how many rows vanished. `measured = hits / total if total else 0.0` also fabricates a `0.0` where "could not measure" is the truth. | Count the dropped rows and put them in the coverage note and in `TrainResult.notes`; return `None` rather than `0.0` when `total == 0` (the gate already handles `None` correctly at `gate.py:318`). | A class the model has never seen is exactly the population whose prediction sets carry no guarantee, and it is currently the one population removed from the measurement. This is house rule 3 — requested vs measured — being satisfied on a denominator that quietly shrank. | S |
| 11 | Drift | `monitor/drift.py:491-492`: `share = len(drifted) / len(measured)`, where `measured` (line 481) holds only the columns Evidently returned a score for. `unmeasured` (line 482) reaches the JSON side-file at line 524 but **not** the `DriftReport` returned at lines 547-560 — the object the CLI, `alerts` and `check_model_health` consume. If 11 of 12 columns fail to score and the survivor is stable, `share = 0/1 = 0.0 → pass`. `tests/test_drift.py` has no case with a non-empty `features_not_measured`. | Add `n_measured` / `n_not_measured` to `DriftReport` and emit a distinct verdict (or refuse) when measured coverage falls below a configurable fraction of the declared columns. | The module's own docstring at `drift.py:96-97` says *"'no drift' and 'not enough data to tell' are the two answers a monitoring dashboard must never confuse."* `_MIN_ROWS` enforces that row-wise; nothing enforces it column-wise, and column-wise is where Evidently actually fails. | S |
| 12 | Settings | `settings.artifact_path` is a derived `@property` (`settings.py:102-110`), not a `BaseSettings` field, and `model_config` sets `extra="ignore"` (`settings.py:36`). So `AEGIS_ML_ARTIFACT_PATH` is accepted, discarded, and the promote writes to the real `~/aegis/backend/.artifacts/ml_spine.joblib`. **This happened during this audit**: a probe that believed it had redirected the path overwrote the live Aegis spine with 50 bytes. (`_archive_live_artifact` preserved the original and it was restored, digest-verified — that safety net worked exactly as designed.) | Switch `extra="ignore"` → `extra="forbid"`, or have `doctor` list every `AEGIS_ML_*` variable in the environment that matches no settings field. Consider promoting `artifact_path` to a real overridable field. | ISSUES #17 built `unknown_keys()` + `doctor` to end silent config no-ops — but only for the TOML layer. The environment layer, which sits *above* it in precedence, still absorbs typos and unsupported names silently, and the blast radius of this particular one is the file `aegis.ml.get_model()` loads. | S |
| 13 | CLI ergonomics | ISSUES #19 is real but its recorded fix would not work. `pyproject.toml:123` is `aegis-ml = "aegis_ml.cli:app"` — the Typer object — so `main()` at `cli.py:1537` is **dead code for the console script**, and "prepend `Path.cwd()` to `sys.path` in the CLI entry point" would change nothing. The failure is also unhandled: `aegis-ml contract --adapter reference.problem --data …` prints ~30 frames of `importlib` internals ending in `ModuleNotFoundError: No module named 'reference'`, with no remedy sentence, in a codebase whose house rule is that every error names its own fix. | Three parts: point `pyproject.toml:123` at `aegis_ml.cli:main`; insert `Path.cwd()` at `sys.path[0]` in `main()`; and wrap `importlib.import_module(adapter)` at `cli.py:160` in `except ModuleNotFoundError` that names the cwd, `sys.path[0]` and the `PYTHONPATH=.` workaround. | This is the first command an agent runs on hackathon morning and the first thing that fails. It currently produces a wall of stdlib traceback that says nothing about what to do — the highest-value single ergonomics fix in the package. | S |
| 14 | Slice evaluation | `evaluate/slices.py:14-18` promises "A skipped slice is recorded, never dropped", and `_segment_frames` at `slices.py:177-178` says unsliceable features are "not returned; the caller records them". `slice_report` never does. Reproduced on the reference problem: five numeric features vanish from the sweep with `skipped == []` and no note. Compounding it, `gate.py:378-391` collapses "no champion" and "champion's slices name a different metric" into the same PASS — a challenger with a collapsed 0.10 segment is promoted over a champion whose sweep used `rmse`. | Emit a `SkippedSlice` with `reason="feature could not be bucketed"` / `"declared feature absent from the evaluation frame"` at `slices.py:182` and `:188`. In `_check_worst_slice`, split the two cases: a genuine first model passes; an incomparable champion sweep FAILs with "re-evaluate the champion's slices on `{name}`" — the treatment `_check_metric:198` already gives a metric-name mismatch. | Criterion 4 is the gate's only defence against a model that is fine on average and broken for one population. Today it can pass on one measured segment out of forty, and it free-passes entirely whenever the champion's sweep used a different metric. | M |
| 15 | Recipe portability | `automl/recipe.py:461-463` documents: *"Dropped keys are returned, never discarded quietly — the caller writes them into `Recipe.notes`."* All five call sites discard the return (`recipe.py:532`, `search.py:408`, `search.py:496`, `hpo.py:178`, `hpo.py:386`). Measured on a real FLAML `rf` config: `max_leaves` is dropped, while the candidate's `detail["config"]` still shows it — so the leaderboard prints a configuration the promoted recipe was not fitted with. | Thread `dropped` into `Recipe.notes` at the three search/HPO sites — literally what the docstring already promises. | The two-venv design rests on the recipe being a faithful description of what was scored. A promoted model parameterised differently from the candidate whose number is on the card is the quiet version of the ISSUES #22 failure, and no reader can currently see it. | S |
| 16 | HPO | `automl/hpo.py:76-78` claims an untunable recipe is reported as such. `_TUNABLE_KINDS` is tree-only, so `RidgeCV` and `LogisticRegression` are absent, and the `untunable` note is appended **only on the improved branch** (`hpo.py:404-408`) — unreachable when the search space is empty. Reproduced on a `RidgeCV` recipe: all 8 trials score identically and the output reads *"Best trial … does NOT beat the untuned recipe"* with no mention that nothing was tunable. At the default `hpo_trials=60` that is 180 identical fits. | Move the `untunable` note out of the improved branch, and short-circuit with an explicit refusal when every member is untunable. | Since ISSUES #22 fixed the explainer dispatch, `ridge_reference` is what the search promotes — so this is now the *common* path, not a corner. An agent reads "tuning found nothing better" and concludes the model is at its ceiling, when in fact the tuner never moved a knob, and 600 s of the time budget was spent proving it. | S |
| 17 | Splitting | `data/splits.py`'s module docstring devotes a paragraph to *"A time series must never be shuffled … calibration rows drawn from after the rows the model trained on … void the coverage guarantee"* and ships `time_ordered_split` and `grouped_split`. Neither is called from any flow: `three_way_split` (random/stratified) is the only split reached from `flows.py:581` and `bundle.py:651`. `measure_learnability` calls `stratified_split` (`latent.py:1577`) too. Root cause: `MLProblem` (`contracts/spec.py:127-165`) has no `time_column` or `group_column`, so a domain has no way to *declare* that it is temporal. | Add optional `time_column` / `group_column` to `MLProblem`; have `flows._split` dispatch to `time_ordered_split` / `grouped_split` when either is set; refuse (rather than warn) when a feature carries a `datetime` dtype and neither is declared. `evaluate/cv.resolve_strategy` needs the same signal — `auto` currently never resolves to `time_series` and then defaults `shuffle=True`. | This is the largest structural gap and it lands squarely on hackathon-day risk: if the problem statement turns out to be temporal, every conformal interval the package produces is narrower than the truth, the realism band itself is measured under leakage, and nothing anywhere says so. Nothing in the code or `finalplan.md` declares temporal domains out of scope — the docstring reads as though the guard is live. | L |
| 18 | Leakage detection | `features/leakage.py:258` — `if feature.name in allowed or feature.name not in labelled.columns: continue` — silently narrows the scan when a declared feature is absent from the frame, while the docstring says "Score *every* feature on its own". `encode_frame` (`pipeline.py:152-158`) raises on the identical condition. Downstream, `contract_check.py:266-268` folds column-audit issues into `report.warnings`, so `report.ok` stays `True` and `flows.py:658` sets `contract_ok=True`. | Emit a "not scanned" record alongside the `LeakSignal` list, or raise as `encode_frame` does. Reclassify "declared in the spec but absent from the frame" from a warning to an issue in `_audit_columns`. | Combined with row 1, this means a spec/frame mismatch — the single most likely authoring error on the day — produces a clean contract report *and* a clean leakage report about columns nobody looked at. | S |
| 19 | Serve tools | `serve/tools.py:436-451` merges `{**features, **changes}` and diffs the predictions. `unknown_features` is on the response and `PredictOutcomeArgs`' description (`tools.py:108-111`) tells the caller to check it; `whatif_scenario` never reads it, and `_compose_summary` (`tools.py:242`) reads `imputed_features` but not `unknown_features`. A typo'd key in `changes` is dropped by the model, `delta` is `0.0`, and the summary asserts *"Changing carrier_ontime: 0.9 → 0.4 moves the prediction from 5.0 to 5.0 (+0.0)"*. | Intersect `after.unknown_features` with `parsed.changes`; return `ok=False` naming the unrecognised keys, or at minimum lead the summary with them. | A planner reads "+0.0" as a substantive finding — *this factor does not matter* — when the truth is *you named a field the model has never heard of*. The counterfactual tool is the one that most directly shapes what the agent tells a human. | S |
| 20 | Cross-venv verification | `verify_strong` reproduces the *non-promoted* ceiling exactly (recorded 0.6613144284511125, reproduced identically). Nothing does the equivalent for the recipe that actually ships: `Candidate.metric_value` is measured in `.venv-ml`, `flows.py:1161` computes an independent number in `.venv` on a different split, and the two are never reconciled. | Persist the trainer-venv holdout indices alongside the recipe, and add a `verify_recipe(run_id)` that re-fits the recipe in the serving venv on those exact rows and asserts the score matches within a stated tolerance — reporting the delta on the model card, the way version drift is already reported. | D1, the two-venv split, is the keystone of the whole design, and "the recipe re-fits to the same model on the other side" is the single claim it rests on. It is currently the one claim in the package that is asserted rather than measured — in a repo whose defining practice is the opposite. | M |
| 21 | Report labels | Two mislabels, both confirmed on disk. (a) `report/plots.py:786` sorts `reverse=not higher_is_better` (ascending for r², so `invert_yaxis` puts the winner on top), then line 844 takes `best = rows[0] if higher_is_better else rows[-1]` — the **worst** row. `visuals/manifest.json` for the shipped run reads `"selected": ridge 0.7460`, `"best": extra_trees 0.6531`, `"ceiling_gap": -0.0928`. (b) `scripts/run_demo.py:541` renders `achieved['floor']/['ceiling']` — the learnability guard, `[0.15, 0.95]` — under the label `realism band`, while line 296 of the same script prints the real band `[0.62, 0.92]` for the secondary target. Both appear in one document. | (a) `best = max(rows, key=…) if higher_is_better else min(rows, key=…)` — independent of the sort. (b) Relabel `run_demo.py:263` and `:541` as "learnability guard" and add a separate row from `realism_band_for(problem)`. The per-run `RUN_SUMMARY.md` written by `flows._render_run_summary:2306` is already correct. | ISSUES #20/#21, but with the root causes located: (a) is a one-line inversion that fires on **every** r²/accuracy run and exports a negative "headroom" as machine-readable JSON; (b) puts two different quantities under one word in the file a reviewer reads first. | S |
| 22 | SHAP | `explain/shap_report.py:264-265` draws `background = features.sample(n=50, random_state=seed)` and `sample = features.sample(n=300, random_state=seed)` from the same frame with the same seed. `pandas.sample` nests, so the background is exactly the first 50 rows of the explained sample — verified, 50/50 overlap, prefix. Separately, `shap_report.py:270` passes a wrapped `predict` *callable* to `build_explainer`, which routes to `PermutationExplainer` by construction (`explainers.py:292-299`), so the per-family `TreeExplainer`/`LinearExplainer` dispatch added by ISSUES #22 is never used by the report — ≈1.5M model row-evaluations for an attribution a tree could produce exactly. | Shuffle once and take the background from the head and the explained sample from the tail (disjoint), or simply derive the background seed as `seed + 1`. For the fast path: pass the unwrapped estimator plus the encode transform to `build_explainer` and fall back to the callable only when `model_family` returns `other`. | Every background row is currently explained against a background containing itself, which shrinks its attributions toward zero, and the background is not an independent draw from the reference population — which is exactly what `shap_report.py:235-237` says it must be. The fast path also removes the dominant cost of the report bundle. | S |
| 23 | Prediction log | `monitor/log.py:487-505` reads the JSONL, writes the parquet, then `os.truncate(jsonl, 0)`. Any row a concurrent worker appends between the read and the truncate is destroyed — in the normal path, not a crash path — and the `O_APPEND` design at `log.py:221-235` exists precisely so concurrent workers can write. `log.py:468` states "the parquet is written and `fsync`-ed before the JSONL is truncated"; `grep -rn fsync src/` hits only `store.py:249,303`. | Record the byte offset before reading and `os.truncate(jsonl, offset)`; add the `os.fsync` the docstring already promises. `store.py` has the correct pattern 200 lines away. | Compaction is the one operation in the package that deletes data. It currently loses whatever arrived while it ran, and the durability the docstring claims does not exist. | S |
| 24 | Test suite | `tests/test_reference_domain.py:43,52,61` guard with `pytest.importorskip("aegis", reason="needs PYTHONPATH=/Users/yrevash/aegis/aegis/src")`. `/Users/yrevash/aegis/aegis` is a directory with no `__init__.py`, so the natural near-miss `PYTHONPATH=/Users/yrevash/aegis` imports it as a **namespace package**: `importorskip` succeeds, then `from aegis.adapter import DomainAdapter` raises. Reproduced: `3 failed, 16 passed` instead of 3 skips. | `pytest.importorskip("aegis.adapter")` / `("aegis.ml.spec")` — skip on the module actually needed, not on a name a namespace package can satisfy. | The reason string names a path one segment deeper than `settings.aegis_root`, so a user who sets the obvious value gets three red tests that say nothing about their own work — on a morning when a red suite is the signal they are relying on. | S |
| 25 | Diagnostics | `_require.py:34-42` catches `ImportError`, which is also what a *present* package raises when its own import fails partway. Verified with a package whose `__init__` imports a missing module: the message is *"'brokenpkg' is required here and is not importable. Install it with: `uv pip install 'aegis-ml[strong]'`"* — the wrong remedy. Also `_require.py:14` imports only `importlib` while line 57 calls `importlib.util.find_spec`; under `python -S` that is an `AttributeError`, which is not in the caught tuple at line 59. | `import importlib.util` explicitly. In `require()`, check `find_spec(module.split(".")[0])` (or `exc.name`) first and, when the top-level package is present, raise "installed, but importing it failed" with the original message surfaced. | This is the function the entire no-silent-fallbacks doctrine routes through. Given the lightgbm ↔ nannyml ↔ sklearn triangle documented in SESSION.md §6, a partially-broken optional dependency is the *expected* failure mode here, and the message currently sends the reader to reinstall something already installed. | S |
| 26 | Docs vs code | Measured against the tree: `README.md:127` and `SESSION.md:116` say **314** tests (actual **323**); `README.md:132` and `ISSUES.md:279` say **14** reviewed opt-outs (actual **17**); the no-mocks audit reports **73** source files against README's 59–64. `card.json` is in `store.STANDARD_ARTIFACTS` (`store.py:80`) and mirrored at `mlflow_mirror.py:48`, and no writer for it exists anywhere. Every `aegis_ml.*` symbol quoted in the docs does resolve — the earlier drift rounds held. | Re-derive the three counts from `pytest --collect-only`, `scripts/audit_no_mocks.py` and the audit's own file count, in one pass over `README.md` / `SESSION.md` / `ISSUES.md`. Either write `card.json` from `explain/card.py` (it is the machine-readable form an agent would consume, and the only one missing) or drop it from `STANDARD_ARTIFACTS`. | ISSUES #16 criticises Aegis's own repo for exactly this — self-reported counts that disagree with each other — and instructs "record what you actually get". The counts are the cheapest thing in the repo to keep honest, and they are the first thing a reader checks the rest against. | S |

---

## P1 detail

### 1 — the gate cannot tell "clean" from "not checked"

Reproduce: `evaluate_gate(result, None, contract_ok=True, leakage=[])` prints
`PASS no_target_leakage: the feature audit flagged nothing.` Now delete `gate_inputs.json`
from a run directory and call `promote_flow` on it. `flows.py:1766` writes a note saying both
contract and leakage are treated as UNPROVEN; `flows.py:1775` then passes `[]`, and criterion 5
passes with that same sentence. The note documents a guarantee the line below it does not
implement. The fix is one signature change (`Sequence | None`) plus three call sites, and
`gate.py:318` is already the template — `require_contracts=False` fails closed there.
Today `train_flow` is safe only because `data_flow`'s leakage stage re-raises; the exposure is
`promote_flow` on any run whose gate inputs were not recorded, and any direct `check()` caller
who reads `report.ok` (which is unaffected by the leakage warning at `contract_check.py:290`).

### 2 — a stale AutoML result is adopted as this run's

`flows.py:1033-1039` hashes the frame but not the split. Change `test_size` from 0.2 to 0.3 and
re-run `train_flow`: the search stage hits cache, and the recipe plus `leaderboard.json` that
get registered were selected and scored on a different training set. The manifest says
`status="cached"`, which is true about reuse and silent about staleness. Same for editing
`primary_metric` in `ml_spec.py` — the ranking metric is not in the key, so a leaderboard ranked
by `r2` is reused for a run that now asks for `rmse`. Add `test_size`, `calibration_size` and
`sha256(problem.model_dump_json())` to the key parts. While there, `manifest.content_key`'s
`default=repr` (`manifest.py:97`) is a trap armed for the next `CacheSpec` author: `repr` of a
DataFrame is a truncated preview, so two frames differing in the middle hash identically.

### 3 — rollback restores the wrong run

```
promote run-c ; promote run-a ; promote run-b
champion            -> run-b
archived (by created_at) -> ['run-c', 'run-a']
rollback restored   -> run-c        # the displaced champion was run-a
```
`promote.py:304` walks `store.list_runs(stage="archived")`, and `store._sorted` (`store.py:650`)
orders by `created_at`. Nothing on `RegistryEntry` records *when* a run was archived, so the
information needed to reverse the last promotion is not stored anywhere. `tests/test_registry.py:288`
promotes `first` then `second` — creation-ordered, the one case where the bug is invisible.
Add `promotion_seq: int | None` to `RegistryEntry`, stamp it in `promote.set_stage(previous,
"archived")`, and sort rollback candidates by it descending.

### 4 — the tuner certifies an untuned recipe with another model's score

```
tune(xgb_recipe)   # writes trials into study "dom::target::r2"
tune(hist_recipe)  # same study name, load_if_exists=True
-> "Tuned recipe scored r2=0.6563 (trial #0), an improvement of 0.0654."
-> returned recipe is byte-identical to hist_recipe
```
`study_name_for` (`hpo.py:95`) contains nothing about the recipe, so `study.best_trial` is the
best across every prior invocation. `_params_from_trial` looks up `f"{member.name}__"` and, on a
miss, returns `{**member.params}` unchanged instead of refusing. Both halves need fixing: a
member fingerprint in the study name, and a raise in `_params_from_trial`.

### 5 — the surface a user touches is the surface with no tests

Function-level tracing of the whole suite (`sys.setprofile`, call events, 323 tests):
`flows.py` 1/57, `cli.py` 0/32 in-process, `manifest.py` 0/21, `cv.py` 0/9, `strong.py` 0/18,
`hpo.py` 0/9, `runner.py` 0/12; 204/717 overall. `tests/test_pipeline_end_to_end.py` imports
`run_search`, `three_way_split`, `evaluate_gate`, `promote` and `store` individually and wires
them together itself — a good component test, but it means the orchestration, the stage graph,
the caching, the manifest and every CLI argument are exercised only by `make demo`, which
nothing runs automatically. Six of the findings above (1, 2, 4, 7, 10, 21) sit in that
untested region, and each reproduces in well under a second.

### 6 — the health endpoint reports no health

```
$ curl localhost:8000/ml/health
{"artifact": …, "fix_command": …, "model_available": true, "served_model": …}
```
No `drift`, no `champion`. `router.py:173` calls `_health_snapshot(None)` and `tools.py:584`
gates both lookups on a truthy `domain_id`; there is no way to supply one, because the endpoint
takes no parameters. The docstring at `router.py:191` and the OpenAPI summary at `router.py:327`
both describe the drift verdict as part of the response. Add `domain_id: str | None = Query(None)`
and pass it through; when the registry holds exactly one domain, default to it.

### 7 — a NaN metric promotes, then locks the domain

```
TrainResult(metric_value=float("nan"), …), champion=None
-> promoted=True, 5/5 checks pass
```
`gate.py:187` short-circuits to PASS before `challenger.metric_value` is examined. The NaN run
becomes champion; thereafter every `gain = challenger - nan` is NaN, `nan >= min_gain` is False,
and criterion 1 can never pass again for that domain. Nothing reports why. One guard at the top
of `_check_metric` fixes it. The same session should bound `TrainResult`'s numeric fields —
`empirical_coverage=1.5` and `test_size=0` are currently accepted and promoted, while
`CoverageReport` in the same package already carries the right `Field` constraints.

### 8 — the too-easy ceiling is not wired to its own setting

```
$ AEGIS_ML_SUSPICIOUSLY_EASY_R2=0.30 python -c "…"
settings.suspiciously_easy_r2 = 0.3
ceiling actually used         = 0.95
```
`latent._resolved_ceiling` (`latent.py:1646-1650`) returns `R2_CEILING` / `ACCURACY_CEILING`
directly. Because `config.py:44-45` *maps* the keys, `doctor` does not list them as NOT CONSUMED
— so the mechanism built to catch inert config (ISSUES #17) reports them as working. The floor
path twenty lines up reads settings correctly, which is what makes this easy to miss. Fix the
one function, then add the meta-test: every field in `TOML_TO_SETTING` must have a
`settings.<field>` reader somewhere in `src/`.

---

## Deliberately not suggesting

**Collapsing `.venv` and `.venv-ml`.** The two-venv split is the one decision everything else
depends on, and `RESOLUTION.md` shows it was reached by resolving, not by guessing — both tiers
land on identical `pandas 2.3.3 / numpy 2.4.6 / scikit-learn 1.9.0`, which is what makes recipe
portability sound at all. The right response to the ~1.3 s bridge cost is row 20 (verify the
recipe crosses faithfully), not removing the boundary.

**Caching SHAP attributions across runs.** SHAP is the second-largest cost in the report bundle
and the temptation is obvious, but an attribution is a statement about a specific fitted model
against a specific background. A cache key that got it wrong would produce a plausible,
authoritative, wrong explanation — the exact failure class this package exists to eliminate,
in the artifact people trust most. Row 22's fast-path fix gets the same speedup with no cache.

**Making Prefect mandatory.** D4 is right: a trained artifact must never depend on a server
being up. `prefect_shim` costs almost nothing and the flows are plain Python — that is a feature
on a day when the network may not cooperate.

**A JS framework for the dashboard.** `dashboard/hub.py` renders server-side HTML with zero
external requests, which is why the dashboard works offline and why `index.html` is
self-contained. Adding React would trade that for nothing the hub actually needs.

**A hosted CI pipeline.** Tempting given row 5, but the suite's value depends on `.venv-ml`,
AutoGluon and ~1,174 resolved packages; a GitHub runner would either skip everything interesting
or take twenty minutes. The fix is to make `make test` cover the flows (row 5), not to move it
somewhere else.

**Splitting `flows.py` into seven modules.** 2,547 lines is large, but each flow reads top to
bottom in stage order, and `train_flow`'s own `noqa: PLR0915` comment gives the reason: *"one
linear pipeline; splitting it would hide the order."* That is correct. Test it (row 5) rather
than rearrange it.

**Fixing ONNX (ISSUES #7).** The two limits are measured, documented in the module docstring,
and the feature is off by default with nothing in Aegis consuming it. Turning it on would
require surfacing the NaN-routing divergence on the model card first — which is the real work,
and it is not worth doing for a path nobody uses.

**Replacing the filesystem registry with the Postgres mirror.** D3 is sound and the atomicity is
genuinely well built (same-directory temp, `fsync`, `os.replace`, post-install digest check).
A database would add an availability dependency to the one operation that must work when
everything else is on fire.
