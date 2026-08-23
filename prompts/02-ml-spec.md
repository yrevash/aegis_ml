# PROMPT 02 · Piece 2 — `ml_spec.py`

**This piece and piece 3 together decide whether the demo works. Read `docs/04-synthetic-data.md` before you write a line.**

---

## Role

You are writing **piece 2 of 10**: the single source of truth for what is predictable. It declares the feature contract, the target, **the latent ground-truth function**, and the training frame.

---

## Inputs

- `DOMAIN_BRIEF.md` §3 (target), §4 (features), §5 (latent drivers), §6 (realism).
- Piece 1's enums — derive every categorical level from them, never retype.
- Reference: `/Users/yrevash/aegis/backend/src/app/adapter/ml_spec.py`.

## Output file

```
/Users/yrevash/aegis_ml/reference/adapter/ml_spec.py
```

---

## The contract to satisfy

`aegis.adapter.MLSpecModule`:

```python
@runtime_checkable
class MLSpecModule(Protocol):
    FEATURES: list[Any]
    FEATURE_NAMES: list[str]
    TARGET: Any

    def training_frame(self, *, num_records: int = ..., seed: int = ...) -> pd.DataFrame: ...
    def describe_prediction(self, resp: Any, *, top_k: int = 3) -> str: ...
```

And, separately, `aegis.ml.spec.resolve_spec` reads — leniently, and **without raising** —
`FEATURE_NAMES`, `TARGET.name`, `TARGET.task`, `CATEGORICAL_FEATURES` (or `FEATURES[].dtype`), and a callable `training_frame`.

---

## The two traps, up front

### Trap 1 — `FALLBACK_SPEC`. This is the one that costs a whole session.

`aegis/src/aegis/ml/spec.py`:

```python
features = getattr(candidate, "FEATURE_NAMES", None) or getattr(candidate, "features", None)
target_obj = getattr(candidate, "TARGET", None)
target = getattr(target_obj, "name", None) or getattr(candidate, "target", None)
if not features or not target:
    return FALLBACK_SPEC       # features "feature_0".."feature_3", target "target"
```

**Nothing raises.** Misspell `FEATURE_NAMES`, define it inside a function, leave it empty, or give `TARGET` no `.name`, and the trustworthy spine trains happily on four columns of generated noise and serves the result as domain evidence.

And `_coerce_task` maps anything outside `{"classification","classify","clf","categorical","binary"}` to `"regression"` — so a typo in `task` silently trains a regressor on class labels.

Conformance check #12 (`test_ml_spec_resolves_to_the_domain_not_the_fallback`) is the backstop, not the plan. **Generate this file instead of typing it**, §"Generating" below.

### Trap 2 — the label must be `latent_fn(features) + noise`

The latent function you write here is the ground truth **piece 3 samples labels around**. If piece 3 draws the target independently, the target is noise, the model finds nothing, the conformal interval is honestly enormous — and **nothing in Aegis catches it**. Not one of the fourteen conformance checks. Only `distinct=False` from `python -m app.ml`, read minutes before the demo.

The coupling is *kept* in piece 3; the function it must call is defined *here*.

---

## Generating the file (preferred)

```python
# scripts/emit_spec.py, run once
from aegis_ml.contracts.spec import FeatureSpec, MLProblem, TargetSpec, emit_ml_spec_module
from reference.adapter.schema import AsaGrade, ProcedureType, SurgeonSeniority, TheatreId

PROBLEM = MLProblem(
    domain_id="surgical_scheduling",
    features=[
        FeatureSpec(name="procedure_type", dtype="categorical",
                    levels=[p.value for p in ProcedureType],
                    description="Kind of elective procedure; complexity drives theatre time."),
        FeatureSpec(name="asa_grade", dtype="categorical",
                    levels=[g.value for g in AsaGrade],
                    description="ASA physical status; sicker patients take longer."),
        FeatureSpec(name="slot_position", dtype="numeric", unit="position",
                    minimum=1, maximum=8,
                    description="Position in the day's list; delay accumulates downwards."),
        FeatureSpec(name="patient_bmi", dtype="numeric", unit="kg/m2",
                    minimum=16, maximum=55, nullable=True,
                    description="Body-mass index; affects access and positioning."),
        # ... 8-12 total
    ],
    target=TargetSpec(name="slot_overrun_minutes", task="regression", unit="minutes",
                      minimum=0, maximum=240,
                      description="Minutes past the booked finish this procedure runs."),
    primary_metric="r2",
    requested_coverage=0.9,
)

emit_ml_spec_module(PROBLEM, path="reference/adapter/ml_spec.py")
```

`MLProblem` **refuses**: a name that is not a valid Python identifier; a categorical with no `levels`; a categorical/`levels` mismatch; a classification target with fewer than two levels; a regression target that declares class levels; duplicate feature names; and a target that is also a feature (*"that is perfect leakage"*).

Then hand-write the two things the emitter cannot know: **the latent function** and **`describe_prediction`**.

---

## What the file must contain

```python
FeatureDType = Literal["categorical", "numeric", "boolean"]

class FeatureSpec(BaseModel):
    name: str
    dtype: FeatureDType
    description: str
    levels: list[str] | None = None

class TargetSpec(BaseModel):
    name: str
    task: Literal["regression", "classification"]
    unit: str | None = None
    description: str

FEATURES: list[FeatureSpec]           # ordered
FEATURE_NAMES: list[str]              # [f.name for f in FEATURES]
CATEGORICAL_FEATURES: list[str]       # not a Protocol member — declare it anyway
NUMERIC_FEATURES: list[str]
TARGET: TargetSpec

def latent_<target>(features: dict) -> float: ...
def features_for_<record>(record, *, <joins>) -> dict: ...
def feature_matrix(dataset) -> tuple[list[dict], list[float]]: ...
def training_frame(*, num_records: int = 1200, seed: int = 7) -> pd.DataFrame: ...
def describe_prediction(resp, *, top_k: int = 3) -> str: ...
```

### `latent_<target>` — the linchpin

```python
_INTERCEPT: float = 0.0
_PROCEDURE_BASE: dict[str, float] = {"hip_replacement": 40.0, "cataract": 5.0, ...}
_ASA_PENALTY: dict[str, float] = {"I": 0.0, "II": 6.0, "III": 15.0, "IV": 28.0}
_SENIORITY_GAIN: dict[str, float] = {"registrar": 0.0, "consultant": 8.0, ...}
_FLOOR: float = 0.0


def latent_slot_overrun_minutes(features: dict) -> float:
    """Compute the noise-free ground-truth overrun for a feature row.

    This is the deterministic latent function the generator samples around. It is
    monotone in every driver (later in the list is slower; a more senior surgeon is
    faster; a longer booking absorbs more), which is exactly what makes the target
    predictable for a tree model and gives calibrated conformal intervals meaning.

    Args:
        features: A feature dict as produced by :func:`features_for_procedure`.
            Missing keys fall back to neutral values so partial rows still score.

    Returns:
        Expected overrun in minutes, floored at zero.
    """
    minutes = _INTERCEPT
    minutes += _PROCEDURE_BASE.get(features.get("procedure_type", ""), 22.0)
    minutes += _ASA_PENALTY.get(features.get("asa_grade", ""), 6.0)
    minutes -= _SENIORITY_GAIN.get(features.get("surgeon_seniority", ""), 0.0)

    slot = float(features.get("slot_position", 1) or 1)
    prior = float(features.get("prior_overrun_mins", 0) or 0)
    minutes += 3.5 * slot
    minutes += 0.45 * prior
    minutes += 0.02 * slot * prior                       # the one interaction term
    minutes += 9.0 * float(features.get("equipment_swaps", 0) or 0)
    minutes -= 0.05 * float(features.get("booked_minutes", 0) or 0)
    minutes += 0.6 * max(0.0, float(features.get("patient_bmi", 25) or 25) - 25.0)
    # theatre_id and booking_channel are declared features and are deliberately
    # never read here: SHAP must show them flat.
    return max(_FLOOR, round(minutes, 3))
```

Rules: **pure Python, no numpy/pandas at module scope**; one dict lookup per categorical with a neutral default; a signed coefficient per numeric; monotone; floored; **and it lives here, not in the generator** — both the generator and the training frame call it, so it is the single source of truth.

For a **classification** target, return a *score* (log-odds) and let piece 3 draw the class from a noised logit. Thresholding the clean score deterministically gives a perfectly separable problem and accuracy near 1.0, which is a bug report.

### `features_for_<record>`

Returns a flat `{feature_name: value}` dict covering **every** entry in `FEATURES`, with categorical values as the enum `.value` strings. **Piece 3 must call this exact function** — a re-derivation is how the label stops matching what the model sees.

### `training_frame`

```python
def training_frame(*, num_records: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Build the ML spine's labelled training frame from a fresh synthetic world.

    Args:
        num_records: Records to synthesise. The keyword is deliberately domain-neutral:
            :class:`aegis.adapter.MLSpecModule` names it, and a core Protocol spelling it
            with this domain's noun would force every future domain to use that noun.
        seed: Seed for the synthetic world; a fixed seed gives an identical frame.
    """
    import pandas as pd                                        # inside the function
    from reference.adapter.generator import GeneratorConfig, generate_synthetic_sync

    dataset = generate_synthetic_sync(GeneratorConfig(seed=seed, num_procedures=num_records))
    rows, targets = feature_matrix(dataset)
    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    frame[TARGET.name] = targets
    return frame
```

- **The keyword is `num_records`.** Not `num_rows`, not your domain's noun.
- **Import pandas and the generator *inside* the function**, so this module loads and tests without the ML stack present and without a circular import.

### `describe_prediction`

Its output is **injected into the plan as evidence** and read aloud in a demo.

```python
def describe_prediction(resp: MLExplainResponse, *, top_k: int = 3) -> str:
    """Render an ML prediction as decision-support text for the agent's reasoning."""
    unit = f" {TARGET.unit}" if TARGET.unit else ""
    if isinstance(resp.prediction, (int, float)):
        head = f"Predicted {TARGET.name}: {float(resp.prediction):.1f}{unit}"
    else:
        head = f"Predicted {TARGET.name}: {resp.prediction}"

    lines = [f"ML decision-support ({TARGET.task}):", f"- {head}"]
    if resp.conformal_interval is not None and resp.conformal_confidence is not None:
        low, high = resp.conformal_interval
        lines.append(f"- {resp.conformal_confidence:.0%} confidence interval "
                     f"[{low:.1f}, {high:.1f}]{unit}")
    elif resp.prediction_set_size is not None and resp.conformal_confidence is not None:
        lines.append(f"- {resp.conformal_confidence:.0%} conformal set size "
                     f"{resp.prediction_set_size} (1 = confident)")
    drivers = [
        f"{f.feature} ({'+' if f.contribution >= 0 else '−'}{abs(f.contribution):.2f})"
        for f in resp.shap_attribution[:top_k]
    ]
    if drivers:
        lines.append("- Top drivers (SHAP): " + ", ".join(drivers))
    lines.append("Use this prediction to re-sequence the list before it slips.")
    return "\n".join(lines)
```

Re-voice the last line for your domain. `MLExplainResponse` fields: `prediction`, `conformal_interval`, `conformal_confidence`, `interval_width`, `prediction_set_size`, `shap_attribution` (list of `ShapFeature(feature, value, value_label, contribution)`), `data_source`, `imputed_features`, `unknown_features`. Import it under `TYPE_CHECKING` only.

---

## Verify

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
import reference.adapter.ml_spec as m
from aegis.ml.spec import FALLBACK_SPEC, resolve_spec
r = resolve_spec(m)
assert r is not FALLBACK_SPEC, 'FALLBACK_SPEC! FEATURE_NAMES or TARGET.name is wrong'
print('features   :', r.features)
print('target     :', r.target)
print('task       :', r.task)
print('categorical:', r.categorical_features)
print('provider   :', r.frame_provider is not None)
f = m.training_frame(num_records=200, seed=7)
print('frame      :', f.shape, list(f.columns))
assert list(f.columns) == m.FEATURE_NAMES + [m.TARGET.name]
assert f[m.TARGET.name].notna().all()
"
```

Then the check that matters:

```bash
uv run aegis-ml contract     # pandera + assert_learnable + leakage
```

After the sync:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter/test_ml_spec.py -q)
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q -k ml_spec)
```

### Checklist

- [ ] `FEATURE_NAMES`, `FEATURES`, `TARGET` are module-level and non-empty.
- [ ] `resolve_spec(ml_spec) is not FALLBACK_SPEC`.
- [ ] `TARGET.task` is exactly `"regression"` or `"classification"`.
- [ ] `TARGET.unit` is set for a regression target.
- [ ] `CATEGORICAL_FEATURES` is declared explicitly.
- [ ] Every categorical `FeatureSpec` declares `levels`, derived from a `StrEnum`.
- [ ] `latent_*` is pure Python: no numpy, no pandas, no schema-record argument.
- [ ] `latent_*` reads every feature **except** the deliberately irrelevant ones.
- [ ] Exactly one interaction term.
- [ ] `training_frame` takes `*, num_records, seed` and imports pandas inside the body.
- [ ] `training_frame()` returns `FEATURE_NAMES + [TARGET.name]`, with no null targets.
- [ ] `describe_prediction` names **your** target and unit and nothing from the old domain.
- [ ] `uv run aegis-ml contract` passes.

---

## Next

`prompts/03-generator.md` — where the coupling is actually kept.
