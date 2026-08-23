# PROMPT 03 · Piece 3 — `generator.py`

**This is where the label is coupled to the latent function. Get this wrong and the demo dies, silently, with everything green.**

---

## Role

You are writing **piece 3 of 10**: a synthetic world that is simultaneously the demo's data and the ML spine's training set — plus the client-facing demand series `/forecast` charts.

---

## Inputs

- `DOMAIN_BRIEF.md` §2 (entities), §5 (latent drivers), §6 (realism targets), §7 (series).
- Piece 1's models and enums; piece 2's `latent_*` and `features_for_*`.
- Reference: `/Users/yrevash/aegis/backend/src/app/adapter/generator.py`.
- **`/Users/yrevash/aegis_ml/docs/04-synthetic-data.md` — read it fully first.**

## Output file

```
/Users/yrevash/aegis_ml/reference/adapter/generator.py
```

---

## The contract to satisfy

```python
@runtime_checkable
class GeneratorModule(Protocol):
    DOMAIN_SERIES_LABEL: str
    DOMAIN_SERIES_UNIT: str

    def domain_series_events(self, *, num_records: int = ..., seed: int = ...) -> Sequence[tuple[Any, float]]: ...
    def generate_synthetic_sync(self, config: Any | None = None) -> Any: ...
    async def generate_synthetic(self, config: Any | None = None) -> Any: ...
```

> *"Both entry points must produce the **same** structure and the same labels; the sync one exists because it is called from inside a running event loop (where `asyncio.run` raises) and from offline training, and it must therefore need **no model call**."*

---

## The trap — stated first, because it is the whole point

> **The generator must sample labels around `ml_spec`'s latent function.**
>
> If it does not: the target is noise, the model finds nothing, MAPIE honestly reports an enormous interval, SHAP shows garbage, and the demo collapses.
>
> **Nothing in Aegis catches this.** Not one of the fourteen conformance checks. Not the backend suite. Not ruff. The only native signal is `distinct=False` on the last line of `python -m app.ml`.
>
> `aegis_ml.data.latent.assert_learnable` catches it in seconds. Run it before anything else.

Three properties that must hold, and each has its own way of going wrong:

| Property | How it breaks |
|---|---|
| The features come from `ml_spec.features_for_*` | You build a feature dict inline "just for the label" and it drifts from the one the training frame uses. |
| The mean comes from `ml_spec.latent_*` | You re-derive the formula in the generator "to avoid a circular import". Now two formulas exist. |
| Noise comes from **one seeded RNG instance** | You use module-level `random.gauss`, or a fresh `random.Random()` per record. Reproducibility dies and `dataset_digest` means nothing. |

---

## What to write

### 1. `GeneratorConfig`

```python
class GeneratorConfig(BaseModel):
    """Knobs for one synthetic world."""

    num_theatres: int = Field(default=4, ge=1)
    num_surgeons: int = Field(default=9, ge=1)
    num_procedures: int = Field(default=400, ge=1)
    num_documents: int = Field(default=6, ge=0)
    completed_fraction: float = Field(default=0.75, ge=0.0, le=1.0)
    seed: int | None = None
    noise_scale: float = Field(default=_CALIBRATED_SIGMA, ge=0.0)
    use_llm: bool = True
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
```

> **Every count knob must be a positive integer field named `num_*`.** `app/demo_graph.py` scales them generically by reflecting over the model — a count spelled any other way is not scaled and the demo generates the default size regardless of what was asked for.

### 2. `_CALIBRATED_SIGMA` — derived, not typed

```python
from aegis_ml.data.latent import sigma_for_r2

_CALIBRATED_SIGMA: float = 11.4
"""Std-dev of the additive Gaussian noise on the target, in minutes.

Derived, not chosen: `sigma = sqrt(Var(signal) * (1 - target_r2) / target_r2)` for
target_r2 = 0.62 over a 2000-row draw. Recompute with
`aegis_ml.data.latent.calibrate_noise(PROBLEM, latent_slot_overrun_minutes,
target_r2=0.62)` whenever a latent coefficient changes — the scale of the latent
function moves with them, and a sigma that gave R2 0.62 yesterday gives 0.93 after
you double a driver's weight.
"""
```

Compute it once with `calibrate_noise` and paste the number with the formula in the docstring. Do not call the calibration at import time — it would make importing the adapter slow and non-deterministic.

### 3. The label computation — the ten lines that matter

```python
def _finalise_completed(
    rng: random.Random,
    cfg: GeneratorConfig,
    proc: Procedure,
    *,
    theatre: Theatre,
    surgeon: Surgeon,
    day_pressure: float,
) -> Procedure:
    """Compute the target and the finish timestamp for a completed procedure.

    The label is ``latent_fn(features) + noise`` by construction, using the SAME
    feature builder and the SAME latent function the training frame uses — so what the
    model sees is exactly what the label was computed from. Nothing in the platform
    enforces this coupling; ``aegis_ml.data.latent.assert_learnable`` is what catches
    it when it breaks.
    """
    features = ml_spec.features_for_procedure(proc, theatre=theatre, surgeon=surgeon)
    mean_minutes = ml_spec.latent_slot_overrun_minutes(features)

    # Heteroscedastic: a longer booking is intrinsically less predictable.
    sigma = cfg.noise_scale * (0.5 + proc.booked_minutes / 200.0)

    # `day_pressure` is the unobserved confounder: it moves the label, it is drawn
    # once per theatre day, and it is deliberately NOT a declared feature. It is the
    # irreducible error floor, and the honest reason the conformal interval has width.
    overrun = max(0.0, mean_minutes + rng.gauss(0.0, sigma) + day_pressure)

    finish = proc.scheduled_start + timedelta(minutes=proc.booked_minutes + overrun)
    return proc.model_copy(update={
        "slot_overrun_minutes": round(overrun, 2),
        "actual_finish": finish,
        "status": ProcedureStatus.FINISHED,
    })
```

For **classification**:

```python
score = ml_spec.latent_excursion_score(features)
prob = 1.0 / (1.0 + math.exp(-(score + rng.gauss(0.0, cfg.noise_scale))))
label = ExcursionFlag.EXCURSION if rng.random() < prob else ExcursionFlag.NOMINAL
```

Draw from a **noised logit**. Thresholding the clean score gives a separable problem and accuracy near 1.0.

### 4. MAR missingness

```python
# Missing-at-random, not missing-completely-at-random: BMI is not routinely recorded
# for ASA grade I patients. This exercises the spine's imputation path and populates
# MLExplainResponse.imputed_features, which is a real honesty signal.
bmi = proc.patient_bmi
if proc.asa_grade is AsaGrade.I and rng.random() < 0.35:
    bmi = None
```

### 5. The two entry points

```python
def generate_synthetic_sync(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Fabricate a schema-valid synthetic dataset with no LLM and no ``await``.

    Called from inside a running event loop (``app.agent.deps``, ``app.mcp.server``),
    where ``asyncio.run`` raises, and from offline training via
    ``ml_spec.training_frame``. It must therefore need no model call at all — which is
    also what makes the system demonstrable while a model key is still being sorted out.
    """
    cfg = (config or GeneratorConfig()).model_copy(update={"use_llm": False})
    return _build(cfg, complete=None)


async def generate_synthetic(
    config: GeneratorConfig | None = None,
    *,
    complete: CompleteFn | None = None,
) -> SyntheticDataset:
    """Fabricate a synthetic dataset, optionally with LLM-written record text.

    Same structure, same labels as the sync path. The LLM touches only free-text
    fields; if it changed a categorical or a numeric, the demo data and the training
    data would disagree and nothing would report it.
    """
```

`CompleteFn = Callable[[str], Awaitable[str]]`, **injected** rather than imported, so the adapter does not depend on `app.gateway` and the generator stays testable.

### 6. The templated fallback

Every LLM-written field needs an f-string template behind it. `generate_synthetic(config, complete=None)` must return a **fully schema-valid dataset**.

### 7. The demand series

```python
DOMAIN_SERIES_LABEL = "Procedures scheduled per day"
"""The /forecast chart title, in the client's language. This is a sentence a jury reads."""

DOMAIN_SERIES_UNIT = "procedures"
"""What the values are counted in — the chart's y-axis."""


def domain_series_events(*, num_records: int = 1400, seed: int = 11) -> list[tuple[datetime, float]]:
    """Return ``(timestamp, value)`` arrival events for the client-facing demand series.

    Arrivals, not completions: arrivals are what a client plans capacity against, and
    the series is complete at the recent end. A completions series is always missing
    the most recent, most interesting days.
    """
```

Give it weekday/weekend seasonality and a mild trend so `AutoARIMA`/`AutoETS` have something to beat `SeasonalNaive` with. `aegis.forecast` needs a minimum history — check `minimum_history()` and `season_length_for()` and make sure 1400 records span enough days.

### 8. Optional but cheap: `assess_quality`

```python
class DatasetQualityReport(BaseModel):
    referential_integrity: bool
    category_coverage: bool
    has_labels: bool
    temporal_consistency: bool
    pii_free: bool
    num_labelled: int
    level_counts: dict[str, int]

    @property
    def ok(self) -> bool: ...

def assess_quality(dataset: SyntheticDataset) -> DatasetQualityReport: ...
```

Demos well and costs twenty lines.

---

## Seeding

```python
def _build(cfg: GeneratorConfig, *, complete: CompleteFn | None) -> SyntheticDataset:
    rng = random.Random(cfg.seed)      # ONE instance, threaded through every builder
    theatres = _build_theatres(rng, cfg)
    surgeons = _build_surgeons(rng, cfg)
    days     = _build_days(rng, cfg)
    procs    = _build_procedures(rng, cfg, theatres, surgeons, days)
    ...
```

Never module-level `random`. Never `numpy.random` (this module must import without numpy). Never a fresh `Random()` per record. A fixed `seed` must give a byte-identical dataset, because `dataset_digest` is a SHA-256 of the exact frame.

---

## Verify — in this order

```bash
cd /Users/yrevash/aegis_ml

# 1 — determinism, and the sync path needs no LLM
uv run python -c "
from reference.adapter.generator import GeneratorConfig, generate_synthetic_sync
a = generate_synthetic_sync(GeneratorConfig(seed=7, num_procedures=100))
b = generate_synthetic_sync(GeneratorConfig(seed=7, num_procedures=100))
assert a.model_dump_json() == b.model_dump_json(), 'not deterministic'
print('deterministic; labelled:', len(a.labelled_procedures()), '/', len(a.procedures))
"

# 2 — THE CHECK. Seconds. Do this before anything expensive.
uv run python -c "
from aegis_ml.data.latent import assert_learnable
import reference.adapter.ml_spec as m
assert_learnable(m.training_frame(num_records=1200, seed=7),
                 target=m.TARGET.name, task=m.TARGET.task, floor=0.15)
print('label is learnable')
"

# 3 — the realism numbers you will quote
uv run python -c "
from aegis_ml.data.profile import realism_report
import reference.adapter.ml_spec as m
print(realism_report(m.training_frame(num_records=2000, seed=7), m.PROBLEM).to_markdown())
"

# 4 — the whole cheap gate
uv run aegis-ml contract

# 5 — the series
uv run python -c "
from reference.adapter.generator import DOMAIN_SERIES_LABEL, DOMAIN_SERIES_UNIT, domain_series_events
e = domain_series_events(num_records=1400, seed=11)
print(DOMAIN_SERIES_LABEL, '|', DOMAIN_SERIES_UNIT)
print('events:', len(e), 'first:', e[0], 'last:', e[-1])
"

# 6 — the async path, with and without a model
uv run python -c "
import asyncio
from reference.adapter.generator import GeneratorConfig, generate_synthetic
d = asyncio.run(generate_synthetic(GeneratorConfig(seed=7, num_procedures=50), complete=None))
print('async fallback ok; labelled:', len(d.labelled_procedures()))
"
```

After the sync:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter/test_generator.py -q)
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.ml | tail -1)   # distinct=True
```

### Reading step 3's output

| Held-out R² | Verdict |
|---|---|
| < 0.15 | **Broken.** Step 2 raises. The label is noise. |
| 0.15 – 0.40 | Weak. Strengthen the drivers or reduce sigma. |
| **0.45 – 0.80** | **Ship it.** |
| 0.90 – 0.98 | Suspicious. Too little noise, or one driver dominates. |
| > 0.99 | **A bug.** Leakage, or `noise_scale` ≈ 0. |

Accuracy target band: **0.65 – 0.88**.

---

## Checklist

- [ ] The label is `ml_spec.latent_*(ml_spec.features_for_*(...)) + noise` — both functions **called**, not re-derived.
- [ ] One `random.Random(cfg.seed)` instance, threaded through every builder.
- [ ] `noise_scale` derived from a `target_r2` in 0.50–0.70, with the formula in the docstring.
- [ ] Heteroscedastic sigma, scaled by a feature.
- [ ] An unobserved confounder, drawn per group, **not** in `FEATURES`.
- [ ] MAR missingness on 1–2 features, conditional on another.
- [ ] Classification only: label drawn from a **noised** logit, 70/30 or 85/15.
- [ ] Downstream timestamps derived from the label, so they stay consistent.
- [ ] `generate_synthetic_sync` does not `await`, does not import a gateway, forces `use_llm=False`.
- [ ] `generate_synthetic(cfg, complete=None)` returns a fully valid dataset.
- [ ] Both entry points produce the same structure and the same labels.
- [ ] Every count knob is `num_*: int` with `ge=1`.
- [ ] `DOMAIN_SERIES_LABEL` is a client-readable sentence; `DOMAIN_SERIES_UNIT` is a plural noun.
- [ ] `domain_series_events` returns **arrivals**, with seasonality, `(datetime, float)` tuples.
- [ ] A fixed seed gives a byte-identical dataset.
- [ ] `assert_learnable` passes and `realism_report` lands in the target band.

---

## Next

`prompts/04-tools.md`.
