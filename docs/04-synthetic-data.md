# 04 · Synthetic data

**This is the most technically important document in this repository.** The failure it describes is silent, passes every automated check Aegis has, and is discovered minutes before a demo.

---

## 1. The one rule

> **The label must be a function of the features, plus noise.**
>
> ```python
> y = latent_fn(features) + noise
> ```
>
> It must **not** be drawn independently.

If you draw the target from its own distribution and the features from theirs, you have produced a dataset in which no relationship exists. Then:

- Every model trains fine. No exception is raised anywhere.
- Held-out R² lands near 0 (or accuracy at the majority-class rate).
- MAPIE, doing its job **correctly**, produces a conformal interval that spans nearly the whole range of the target — because that is the honest 90% interval when you know nothing.
- SHAP produces attributions that are noise.
- The demo shows a prediction of "42 ± 380" and the story collapses.

`SKILL.md` names this trap: *"the generator must sample labels around your latent function. If it does not, the target is noise, the model finds nothing, and the conformal interval is honestly enormous."*

---

## 2. **Nothing in Aegis catches this**

Read this section before you decide to skip the verification step.

| Check | Catches a noise target? |
|---|---|
| All fourteen conformance checks | **No.** Zero references to the generator exist in `test_conformance.py`. |
| Conformance check #12 (`test_ml_spec_resolves_to_the_domain_not_the_fallback`) | **No.** It asserts the *spec* resolves to your columns, not that the *data* has signal. |
| `backend/tests/adapter/*` | No. |
| `backend/tests/agent/*` | No. |
| The full backend suite | No. |
| The core package suite | No. |
| `ruff` | No. |
| `python -m app.ml` last line | **Partially, and too late.** `distinct=False` means the model predicts the same value for the lowest- and highest-labelled rows of your own training frame. It is the only native signal, and you read it minutes before demo time. |
| **`aegis_ml.data.latent.assert_learnable`** | **Yes, in seconds.** |

The `python -m app.ml` probe is worth understanding, because it is your last line of defence. `backend/src/app/ml/__main__.py`:

```python
frame = training_frame(num_records=300)
ordered = frame.sort_values(target)
low  = ordered.iloc[0][features].to_dict()
high = ordered.iloc[-1][features].to_dict()
...
distinct = not np.isclose(low.prediction, high.prediction)
```

It takes the two rows at the extremes of your own label and asks whether the fitted model separates them. It is built from the adapter's spec, never from a literal row — it used to spell out the shipped domain's nine feature keys and therefore cried wolf on every *correct* retarget. That is fixed. `distinct=False` now means what it says.

---

## 3. **R² near 1.0 is a bug report, not a good result**

The opposite failure is just as bad, and easier to produce by accident.

| Held-out score | Verdict |
|---|---|
| R² < 0.15 (accuracy < 0.55) | **Broken.** `assert_learnable` raises. The label is noise. |
| R² 0.15 – 0.40 | Weak. Learnable but the demo will look unconvincing. Strengthen the drivers. |
| **R² 0.45 – 0.80** | **Target band.** This is what a real dataset looks like. |
| **Accuracy 0.65 – 0.88** | **Target band** for classification. |
| R² 0.90 – 0.98 | Suspicious. Your noise is too small or a driver is too dominant. |
| **R² > 0.99** | **A bug.** Either you have target leakage (a feature is a deterministic function of the label) or you set `noise_scale` to ~0. |

A judge who sees R² = 0.997 on synthetic data learns that you generated the answer and then predicted it. A judge who sees R² = 0.63 with a calibrated 90% interval whose *measured* coverage is 0.91 learns that you built something real.

`aegis_ml.features.leakage` scans for the first case: any single feature scoring above `settings.leakage_threshold` (default `0.98`) against the target alone raises `TargetLeakageError` naming the feature.

---

## 4. Designing a latent function

### 4.1 Structure

```python
def latent_<target>(features: dict) -> float:
    """Noise-free ground truth for one feature row.

    Monotone in every declared driver, which is what makes the target learnable by a
    tree model and gives the calibrated conformal interval meaning. Missing keys fall
    back to neutral values so a partial row still scores.
    """
    value = _INTERCEPT
    value += _LEVEL_TABLE_A.get(features.get("cat_feature_a", ""), _NEUTRAL_A)
    value -= _LEVEL_TABLE_B.get(features.get("cat_feature_b", ""), _NEUTRAL_B)
    value += 0.8 * float(features.get("num_feature_c", 0) or 0)
    value -= 0.5 * float(features.get("num_feature_d", 0) or 0)
    value += 0.02 * float(features.get("num_feature_c", 0) or 0) * float(features.get("num_feature_e", 0) or 0)
    return max(_FLOOR, round(value, 3))
```

Rules:

1. **Pure Python. No numpy, no pandas.** This module must import without the ML stack present.
2. **One dict lookup per categorical feature**, keyed on the enum `.value` string, with a neutral default.
3. **A signed coefficient per numeric feature.** Write the sign down in the Brief before you write the code.
4. **Monotone in every driver**, with one deliberate exception (§4.3). Monotone drivers are what a tree model learns cleanly and what makes a SHAP plot readable by a human.
5. **Floor or clamp the output** to whatever the target's physical range is (`max(0.0, ...)` for a duration, `min(100.0, max(0.0, ...))` for a percentage).
6. **Live in `ml_spec.py`, not in the generator.** Both the generator and the training frame call it, so it is the single source of truth. Change the drivers in one place and the data stays consistent.

### 4.2 Classification targets

Same shape, but the latent function returns a **score** and the label is drawn from it:

```python
def latent_<target>_score(features: dict) -> float:
    """Log-odds of the positive class."""
    ...

# in the generator:
import math
score = ml_spec.latent_excursion_score(features)
p = 1.0 / (1.0 + math.exp(-(score + rng.gauss(0.0, cfg.noise_scale))))
label = "excursion" if rng.random() < p else "nominal"
```

Drawing the class from a *noised logit* is what makes accuracy land in the 0.65–0.88 band. Thresholding the clean score deterministically (`label = score > 0`) gives you a perfectly separable problem and accuracy near 1.0.

### 4.3 The one interaction term

Include exactly one. Real data has interactions; a purely additive target is learned equally well by linear regression, which makes the tree ensemble look pointless and gives a boring SHAP story.

```python
value += 0.02 * float(features.get("slot_position", 0) or 0) * float(features.get("prior_overrun", 0) or 0)
```

Choose two features where the interaction is *physically sensible* and say so in the docstring. Then say it out loud in the demo: "the model found that slot position only matters when the day is already behind — that is the interaction term, and it is visible in the SHAP dependence plot."

---

## 5. Calibrating the noise to a target R²

Do not guess `noise_scale`. Derive it.

For a regression target where the signal is `s = latent_fn(features)` over your generated population and the noise is additive Gaussian with standard deviation σ:

```
R² ≈ Var(s) / (Var(s) + σ²)
```

Solving for σ:

```
σ = sqrt( Var(s) * (1 − target_r2) / target_r2 )
```

| `target_r2` | σ as a multiple of `sd(s)` |
|---|---|
| 0.90 | 0.33 |
| 0.80 | 0.50 |
| **0.70** | **0.65** |
| **0.60** | **0.82** |
| **0.50** | **1.00** |
| 0.40 | 1.22 |
| 0.30 | 1.53 |

Aim for `target_r2` between **0.50 and 0.70**. The measured held-out R² will land a little below it, because a finite model does not recover the latent function exactly.

`aegis_ml.data.latent` implements this:

```python
from aegis_ml.data.latent import sigma_for_r2, calibrate_noise

# Given the signal values your latent function produces over a sample population:
sigma = sigma_for_r2(signal_values, target_r2=0.62)

# Or let it draw the population itself from the problem spec:
sigma = calibrate_noise(problem, latent_fn, target_r2=0.62, n=2000, seed=7)
```

Then set `GeneratorConfig.noise_scale = sigma` — or better, compute it once at generator import time and use it as the field default, so the number in the code is derived rather than typed.

> **Why not just pick a number?** Because the scale of your latent function changes every time you adjust a coefficient. A `noise_scale` of `4.0` that gave R² = 0.65 yesterday gives R² = 0.93 after you double a driver's weight. Deriving it keeps the realism target stable while you iterate on the domain.

---

## 6. The realism requirements

A dataset that is *only* `latent_fn + Gaussian noise` is learnable but obviously synthetic. Six additions make it look like something that came off a real system, and each of them buys you a specific line in the demo.

| # | Requirement | How | What it buys |
|---|---|---|---|
| 1 | **Noise calibrated to a target R²** | §5 | Held-out R² in the 0.45–0.80 band instead of 0.99 |
| 2 | **Unobserved confounders** | Draw a per-group offset (per day, per site, per cohort) from the same seeded RNG, add it to the label, and **do not declare it in `FEATURES`** | An irreducible error floor. This is what stops any model reaching R² = 1.0, and it is the honest reason a conformal interval has width. |
| 3 | **Heteroscedastic noise** | Scale σ with a feature: `sigma = base * (0.5 + booked_minutes / 200.0)` | The conformal interval is *wider where the data is genuinely noisier* — which is exactly the behaviour adaptive conformal methods exist for, and it shows on a residual plot. |
| 4 | **MAR missingness** | Null out ~5–10% of one or two features, **conditional on another feature's value** (missing-at-random, not missing-completely-at-random) | Exercises the spine's imputation path and populates `MLExplainResponse.imputed_features`, which is a real honesty signal you can point at. |
| 5 | **Class imbalance** | For classification, aim for a 70/30 or 85/15 split, not 50/50 | Makes accuracy a bad metric and forces you to quote balanced accuracy / F1 / ROC-AUC — which is a point in your favour, not against. |
| 6 | **Genuinely irrelevant features** | Declare 1–2 features in `FEATURES` that the latent function **never reads** | SHAP correctly attributes ~0 to them. This is the single most convincing demonstration that the explanation is real rather than decorative. |
| 7 | **One interaction term** | §4.3 | Justifies the tree ensemble over a linear model. |

### 6.1 Worked: all seven at once

```python
# ml_spec.py — the latent function knows nothing about noise or missingness.
def latent_slot_overrun_minutes(features: dict) -> float: ...   # §4.1, incl. the interaction
                                                                # and NOT reading `theatre_id`

# generator.py
_SIGMA = calibrate_noise(PROBLEM, ml_spec.latent_slot_overrun_minutes, target_r2=0.62)

def _build_day(rng, cfg, day_index):
    # 2 — unobserved confounder: staffing pressure varies by day and is never a feature.
    staffing_pressure = rng.gauss(0.0, 6.0)
    ...

def _finalise(rng, cfg, proc, theatre, surgeon, staffing_pressure):
    features = ml_spec.features_for_procedure(proc, theatre=theatre, surgeon=surgeon)
    mean = ml_spec.latent_slot_overrun_minutes(features)

    # 3 — heteroscedastic: longer bookings are less predictable.
    sigma = _SIGMA * (0.5 + proc.booked_minutes / 200.0)
    overrun = max(0.0, mean + rng.gauss(0.0, sigma) + staffing_pressure)

    # 4 — MAR missingness: BMI is not recorded for ASA grade I patients ~35% of the time.
    bmi = proc.patient_bmi
    if proc.asa_grade == "I" and rng.random() < 0.35:
        bmi = None

    return proc.model_copy(update={"slot_overrun_minutes": round(overrun, 2),
                                   "patient_bmi": bmi})
```

`theatre_id` is declared in `FEATURES`, generated with a uniform draw, and never read by the latent function. That is requirement 6, and SHAP will show it flat.

### 6.2 Seeding

**One RNG instance, created once from `cfg.seed`, threaded through every builder.**

```python
rng = random.Random(cfg.seed)
```

Never `random.gauss(...)` (module-level global state), never a fresh `random.Random()` per record (defeats reproducibility), never `numpy.random` in a module that must import without numpy. A fixed `seed` must give a byte-identical dataset, because `dataset_digest` is a SHA-256 of the exact frame and provenance depends on it.

---

## 7. Verifying — the commands that actually catch it

### 7.1 `assert_learnable` — run this before anything expensive

```python
from aegis_ml.data.latent import assert_learnable
import app.adapter.ml_spec as ml_spec

assert_learnable(
    ml_spec.training_frame(num_records=1200, seed=7),
    target=ml_spec.TARGET.name,
    task=ml_spec.TARGET.task,
    floor=0.15,            # settings.learnable_r2_floor / learnable_accuracy_floor
)
```

It fits a fast model on a train split, scores on a held-out split, and raises `LabelNotLearnableError` with the measured number when the score is below the floor:

```
Label is not learnable: r2=0.0113 on a held-out split, below the floor of 0.1500.
The generator is drawing the target independently of the features. Fix the generator
so the label is `latent_fn(features) + noise` — see aegis_ml.data.latent. Training past
this point produces a model that has learned nothing and a conformal interval that is
honestly enormous.
```

Runs in seconds. **Wire it into your own `backend/tests/adapter/test_ml_spec.py` rewrite** so it runs on every suite invocation, and `aegis-ml doctor` runs it too.

### 7.2 `aegis-ml contract` — the full cheap gate

```bash
cd /Users/yrevash/aegis_ml && uv run aegis-ml contract
```

PowerShell:

```powershell
Set-Location C:\aegis_ml; uv run aegis-ml contract
```

Runs, in order:

1. **pandera validation** of `training_frame()` against the schema derived from your `MLProblem` — dtypes, ranges, null policy, and **the categorical level sets**.
2. **`assert_learnable`** — the check above.
3. **leakage scan** — every feature scored alone against the target; anything above `settings.leakage_threshold` (0.98) raises `TargetLeakageError`.

The pandera step matters more than it looks. `aegis.ml.model.train` one-hot-encodes with `handle_unknown="ignore"`, so an **unseen categorical level does not raise** — it encodes to an all-zero block and the row is scored as if the feature were absent. A generator that emits `"REFRIGERATED "` with a trailing space for 3% of rows produces a model that silently ignores that feature on those rows, a wider-than-necessary conformal interval, and no error anywhere in the stack. That is why `FeatureSpec` **refuses a categorical without declared `levels`**.

### 7.3 `realism_report` — the numbers you quote in the demo

```python
from aegis_ml.data.profile import realism_report

report = realism_report(frame, problem)
print(report.to_markdown())
```

Reports, per the requirements table in §6:

| Row | What it tells you |
|---|---|
| held-out R² / accuracy (and balanced accuracy) | Are you in the target band? |
| per-feature single-feature score | Leakage, and which drivers dominate |
| null rate per column, and its conditional structure | Is missingness MAR or MCAR? |
| class balance | For classification |
| numeric distributions (skew, kurtosis, range) | Anything degenerate |
| categorical level coverage vs the declared `levels` | Unseen or unused levels |
| correlation heat map | Unintended near-duplicates |
| features with ~zero SHAP | Requirement 6 confirmed |

`aegis_ml.data.profile` also exposes `skrub`'s `TableReport` for a free interactive HTML profile — `report.to_html(path)` — which drops straight into a demo tab.

### 7.4 The native backstop

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml | tail -1)
```

`distinct=True` is the pass signal. If you have run §7.1 you will never see `distinct=False`, which is the point.

---

## 8. The procedural + LLM hybrid

Aegis's own generator uses a three-layer pattern. **Copy it.**

```
1 · Procedural draws       seeded, deterministic, no network
      entity ids, categorical levels, numeric features, timestamps
      → and THE LABEL, from latent_fn(features) + noise
                    ↓
2 · LLM fabrication        optional, async, for the free-text fields only
      titles, descriptions, note bodies, document prose
                    ↓
3 · Templated fallback     always present
      f-string templates over the procedural draws
```

Two entry points:

```python
def generate_synthetic_sync(config=None) -> SyntheticDataset:
    """Fabricate a schema-valid synthetic dataset with no LLM and no `await`."""
    cfg = (config or GeneratorConfig()).model_copy(update={"use_llm": False})
    return _build(cfg)

async def generate_synthetic(config=None, *, complete=None) -> SyntheticDataset:
    """Same structure, same labels, optionally with LLM-written record text."""
```

### Why the sync path must need no LLM

Three independent reasons, all of them load-bearing:

1. **It is called from inside a running event loop.** `app/agent/deps.py` and `app/mcp/server.py` call `generate_synthetic_sync()` to build the shared record store. `asyncio.run` raises inside a running loop, so the sync path cannot await anything, and therefore cannot call a model.
2. **It is the training path.** `ml_spec.training_frame()` calls it. Training must be reproducible, offline, and free — an LLM call in the training path means a different dataset on every run and a `dataset_digest` that means nothing.
3. **It is what makes the system demonstrable while the model key is still being sorted out.** On the day, this is not hypothetical.

**The critical invariant: both entry points produce the same structure and the same labels.** The LLM touches only free-text fields. If the LLM path changed a categorical or a numeric, the demo data and the training data would disagree and you would never find out.

### The `complete` parameter

```python
CompleteFn = Callable[[str], Awaitable[str]]

async def generate_synthetic(config=None, *, complete: CompleteFn | None = None): ...
```

Injecting the completion function rather than importing a gateway keeps the generator testable and keeps the adapter from depending on `app.gateway`. When `complete is None` or the call fails, fall through to layer 3.

---

## 9. When to use SDV instead

The procedural + LLM hybrid is the **primary route** and the one to use by default. SDV covers a different case.

| Situation | Route |
|---|---|
| No real data. You are inventing the world from a problem statement. | **Procedural + latent function.** You control the ground truth, which means you can *state* what the model should learn and check that it did. |
| You have a real CSV with 200 rows and need 5,000. | **SDV.** `aegis_ml.data.synth`. |
| You have a real CSV and the relationships in it are the point. | **SDV.** |
| You have a real CSV but need a *known* latent function for the demo narrative. | Procedural, fitted to the CSV's marginals. Read the real distributions, keep your own latent function. |

```python
from aegis_ml.data.synth import fit_synthesizer, sample, quality_report

synth = fit_synthesizer(real_frame, problem, model="gaussian_copula")   # or "ctgan"
more  = sample(synth, n=5000, seed=7)
print(quality_report(real_frame, more).to_markdown())     # SDMetrics
```

Defaults: **GaussianCopula** (fast, no torch, good marginals and correlations), with **CTGAN** opt-in when the joint structure genuinely matters and you have the time and the trainer venv. SDV lives in the `[strong]` extra — it is in `.venv-ml`, not the backend venv.

> **The SDV caveat that matters here.** A synthesizer reproduces the relationships in the source data. If your source CSV has no signal, the synthetic copy has no signal either — and now you have no ground truth to compare against, so `assert_learnable` failing tells you about the *source*, not about a bug you can fix. **Run `assert_learnable` on the real CSV before you fit a synthesizer to it.**

---

## 10. Checklist

Before you sync the adapter into Aegis:

- [ ] The label is computed as `latent_fn(features) + noise` in the generator, calling `ml_spec`'s function — not a re-derivation.
- [ ] `features_for_*` is the same function the training frame uses.
- [ ] One seeded `random.Random(cfg.seed)` instance, threaded through every builder.
- [ ] `noise_scale` derived from a `target_r2` between 0.50 and 0.70, not typed by hand.
- [ ] At least one unobserved confounder that is **not** in `FEATURES`.
- [ ] Heteroscedastic noise scaled by at least one feature.
- [ ] MAR missingness on 1–2 features, conditional on another feature.
- [ ] 1–2 declared features the latent function never reads.
- [ ] Exactly one interaction term.
- [ ] Classification only: 70/30 or 85/15 imbalance, label drawn from a **noised** logit.
- [ ] `aegis-ml contract` passes: pandera + `assert_learnable` + no leakage.
- [ ] `realism_report` held-out score is in the target band — **not above 0.90**.
- [ ] `generate_synthetic_sync` returns a fully valid dataset with no LLM and no `await`.
- [ ] `generate_synthetic` produces the same structure and the same labels.
- [ ] `domain_series_events` returns arrival events over your own records, with a client-readable `DOMAIN_SERIES_LABEL`.
- [ ] A fixed seed gives a byte-identical frame.

---

## 11. Next

`docs/05-ml-pipelines.md`.
