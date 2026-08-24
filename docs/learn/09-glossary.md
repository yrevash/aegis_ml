# 09 · Glossary

[← 08](08-your-first-run.md) · [Index](00-index.md)

Alphabetical. Each entry links to the chapter that explains it properly.

**Adapter (domain adapter)** — the one directory that makes Aegis serve a specific business
problem: ten pieces, eleven members, satisfying the `DomainAdapter` Protocol structurally.
→ [07](07-how-it-plugs-into-aegis.md)

**AutoML** — automated model search: try many algorithms and settings, score them all on
held-out data, keep the best. Here it runs in four tiers — `baseline`, `flaml`, `autogluon`,
`tabpfn`. → [02 §7](02-ml-concepts-you-need.md#7-automl--searching-for-the-model-instead-of-picking-one)

**Calibration split** — a slice of data used *only* to measure how large the model's typical
errors are, so a conformal interval can be sized. Disjoint from both training and test.
→ [02 §3](02-ml-concepts-you-need.md#3-train-calibration-test--and-why-three)

**Champion / challenger** — the champion is the model currently serving (one per domain); a
challenger is a newly trained run that may replace it, but only by passing the gate.
→ [06 §2](06-mlops-registry-gate-drift.md#2-champion-and-challenger)

**Conformal interval** — a prediction range built from held-out residuals, carrying a coverage
guarantee that can be *measured* rather than assumed. Works with any model and assumes nothing
about the shape of the errors. → [02 §5](02-ml-concepts-you-need.md#5-conformal-prediction--the-crown-jewel)

**Confounder (unobserved)** — a driver that genuinely moves the target and is never emitted as
a column. It puts an honest ceiling on achievable accuracy that no model can cheat past.
→ [03 §4.1](03-the-data-problem.md#41-unobserved-confounders)

**Coverage** — the share of predictions whose interval actually contained the truth.
**Requested** coverage and **empirical** (measured) coverage are always two separate fields.
→ [02 §5](02-ml-concepts-you-need.md#requested-vs-measured--never-one-number)

**Drift** — live data no longer resembling the data the model was trained and calibrated on.
Measured against the frozen `reference.parquet`; verdicts `pass` / `warn` / `block`.
→ [06 §5](06-mlops-registry-gate-drift.md#5-drift)

**Estimated (vs measured)** — a NannyML performance figure computed *without* ground truth.
Every field carrying one is named `estimated_*` so it can never be misread as a measurement.
→ [06 §5.2](06-mlops-registry-gate-drift.md#52-nannyml--estimated-performance-before-the-labels-arrive)

**Feature** — an input column the model is allowed to look at. → [02 §1](02-ml-concepts-you-need.md#1-features-target-labels)

**Gate (promotion gate)** — the five criteria a challenger must all satisfy before it may
replace the champion: beats champion, coverage meets request, contracts pass, worst slice not
worse, no target leakage. → [06 §3](06-mlops-registry-gate-drift.md#3-the-five-promotion-criteria)

**Heteroscedastic** — the spread of the errors is not the same everywhere. Here the noise is
deliberately scaled with `transit_hours`, giving an adaptive interval something real to adapt
to. → [03 §4.2](03-the-data-problem.md#42-heteroscedastic-noise)

**Holdout / held-out split** — rows kept aside and never used for fitting or calibration. The
only source of an honest score. → [02 §3](02-ml-concepts-you-need.md#3-train-calibration-test--and-why-three)

**Hyperparameter** — a setting chosen *before* fitting (tree depth, learning rate, number of
trees), tuned here with an Optuna study. → [02 §8](02-ml-concepts-you-need.md#8-hyperparameter-optimisation-and-leakage)

**Label** — a known value of the target on a row you already have. Rows without labels can be
predicted for but not learned from or scored against. → [02 §1](02-ml-concepts-you-need.md#1-features-target-labels)

**Latent function** — the declared formula from features to target that a generator samples
labels *around*. Without one, the target is noise and there is nothing to learn.
→ [03 §2](03-the-data-problem.md#2-the-latent-function)

**Leakage (target leakage)** — a feature carrying information that would not exist at
prediction time. Produces the best held-out score in the run and the worst production
behaviour. → [02 §8](02-ml-concepts-you-need.md#8-hyperparameter-optimisation-and-leakage)

**MAR (missing at random)** — a value is absent, and whether it is absent depends on another
*observed* column, not on the missing value itself. Unlike MCAR, median imputation is then
systematically biased — which is what makes reporting imputed features worthwhile.
→ [03 §4.3](03-the-data-problem.md#43-mar-missingness)

**Marginal coverage** — coverage averaged over the whole population. It can look fine while a
subgroup's coverage does not, which is exactly what chart 03 shows.
→ [05 · chart 03](05-reading-the-charts.md#03--conformal-coverage--the-star)

**Portable recipe** — the AutoML search's answer expressed as JSON (`Recipe`): which
estimators, with which parameters. It crosses from the trainer venv to the serving venv, where
Aegis's own spine re-fits it. Portability is decided by an actual two-row fit, not by an
import. → [04 §6](04-the-pipeline.md#6-two-virtualenvs-one-portable-recipe-decision-d1)

**Promotion** — replacing the served artifact (`backend/.artifacts/ml_spine.joblib`)
atomically, archiving what it displaced. Only ever after a passing gate, or an explicitly
recorded `--force` override. → [06 §2](06-mlops-registry-gate-drift.md#2-champion-and-challenger)

**R²** — for regression: the share of the target's variance the model explains. 1.0 is
perfect, 0.0 is no better than predicting the average, negative is worse than that. On
synthetic data, **0.99 is a bug report**. → [02 §2](02-ml-concepts-you-need.md#2-regression-vs-classification)

**Realism band** — the held-out score a *realistic* frame should land in: R² `[0.45, 0.80]`,
accuracy `[0.62, 0.92]`. Below it the model looks broken; above it the data is a toy.
→ [03 §6](03-the-data-problem.md#6-the-learnability-guard-fires-in-both-directions)

**Registry** — the filesystem record of every run: what it scored, on what data, with what
settings, and which one is live. Rooted at `registry_store/`.
→ [06 §1](06-mlops-registry-gate-drift.md#1-what-a-model-registry-is)

**SHAP** — an attribution method answering, for one prediction, how much each feature pushed
the answer up or down, in the target's own units. Averaged absolute values give global
importance. → [02 §6](02-ml-concepts-you-need.md#6-shap--why-did-the-model-say-that)

**Slice** — the primary metric recomputed inside one segment of the data (a categorical level,
or a numeric quartile). The gate reads the **worst** slice, never the mean.
→ [05 · chart 05](05-reading-the-charts.md#05--slice-performance)

**Target** — the column you are predicting. `spoilage_risk_pct` in the reference domain.
→ [02 §1](02-ml-concepts-you-need.md#1-features-target-labels)

**Tier** — one rung of the AutoML search ladder: `baseline` → `flaml` → `autogluon` →
`tabpfn`. A tier that cannot run is recorded in `tiers_skipped` with its reason and the exact
install command — never silently absent. → [02 §7](02-ml-concepts-you-need.md#7-automl--searching-for-the-model-instead-of-picking-one)

---

### Also worth knowing

**Flow** — one of the seven pipeline functions in `pipelines/flows.py`; ordinary Python, with
Prefect applied as a decorator only if installed. → [04 §1](04-the-pipeline.md#1-the-seven-flows)

**Manifest** — the stage-by-stage lineage record every flow writes, including on failure.
→ [04 §1](04-the-pipeline.md#1-the-seven-flows)

**Dataset digest** — a SHA-256 over the feature and target columns plus their names.
Tamper-**evidence**: a mismatch proves the model was not fitted on the frame you believe.
→ [04 §3](04-the-pipeline.md#3-data_flow--nine-stages-all-cheap)

**Non-portable winner / accuracy ceiling** — a candidate that scored best but cannot be served
(a linear model under a tree-only SHAP explainer, an AutoGluon stack, a TabPFN model). Reported
as headroom, never as this model's performance.
→ [04 §7](04-the-pipeline.md#7-non-portable-winners-are-reported-not-hidden)

**Typed refusal** — a named exception raised instead of returning a plausible-looking number.
Nine of them live in `contracts/errors.py`, and each exists because the alternative was a
number a human would have believed. → [01 §6](01-what-problem-does-this-solve.md#6-what-good-looks-like-here)

[← Back to the index](00-index.md)
