# `reference/` — the worked domain: pharmaceutical cold-chain logistics

This is the proof that `aegis_ml` and the Aegis adapter contract work end to end on a real
problem. It is not a fixture, not a sketch and not a set of stubs: it is a complete,
runnable domain that generates its own data, satisfies `aegis.adapter.DomainAdapter`
structurally, trains through the real pipelines, and is measured live by
[`scripts/run_demo.py`](../scripts/run_demo.py).

```
reference/
    __init__.py          package root; imports nothing heavy
    problem.py           the MLProblem + the LatentModel the pipelines consume
    adapter/             the ten pieces — an Aegis DomainAdapter
        schema.py        1 · entities and enums
        ml_spec.py       2 · features, targets, the declared causal story
        generator.py     3 · the seeded synthetic world
        tools.py         4 · the action tools + the five ML tools
        personas.py      5 · who is asking, and what they may see
        prompts.py       6 · the task prompt and the platform floor
        memory_spec.py   7 · durable facts, memory scoping, skill selection
        roster.py        8 · the supervisor's specialists and the fan-out team
        corpus/          9 · three seed documents (SOP, policy, runbook)
        skills/         10 · two procedural playbooks
```

## Why this domain

**Lexical disjointness.** Aegis ships a `service_request_management` reference domain, and
conformance check #14 quarantines that domain's vocabulary from every module outside the
adapter. A retarget to something adjacent — tickets, cases, work orders — satisfies that
check by accident. Cold-chain logistics shares essentially no nouns with it, so the
quarantine is genuinely exercised.

**Three ML shapes from one generator.** The domain naturally carries a regression target
(`spoilage_risk_pct`, `%`), a classification target (`excursion_flag`), and a time series
(`Shipments dispatched per day`) — from one schema, one world and one declared causal story.
Demonstrating all three without three unrelated fixtures is what makes the ML stack's
breadth visible.

**Real actions with real risk spread.** Looking a consignment up is free; annotating it is
cheap and reversible; rerouting it costs money and is visible to the customer; quarantining
it strands product a clinic is expecting. That is a genuine LOW/LOW/MEDIUM/HIGH ladder
rather than four tools tiered to make a table look full.

## The world

A pharmaceutical distributor moves temperature-controlled consignments — vaccines,
biologics, small-molecule product, diagnostic kits — from origin depots through transfer
hubs to clinics, cold stores and hospital pharmacies. Each consignment rides with a
`Carrier` under a declared `PackagingType`, changes custody some number of times, and is
instrumented with data loggers whose readings arrive as `SensorReading` records.

Entities: `Shipment` · `Carrier` · `Facility` · `SensorReading` · `Document`
(plus `ShipmentNote`, `DatasetMetadata`, `SyntheticDataset`).

## The supervised problem

| | |
|---|---|
| `DOMAIN_ID` | `cold_chain_logistics` |
| primary target | `spoilage_risk_pct` — regression, 0–100, unit `%` |
| secondary target | `excursion_flag` — classification, `no_excursion` / `excursion` |
| series | `Shipments dispatched per day`, unit `shipments` |
| features | 10 — five categorical, five numeric |
| requested coverage | 0.90 |

### The ten features

| # | feature | dtype | driver? | note |
|---|---|---|---|---|
| 1 | `carrier_tier` | categorical | yes | economy → validated; also drives the missingness |
| 2 | `route_class` | categorical | yes | direct → multi-leg; each transfer is exposure |
| 3 | `packaging_type` | categorical | yes | gel / PCM / active reefer / dry ice |
| 4 | `origin_region` | categorical | **no** | *deliberately irrelevant* |
| 5 | `product_class` | categorical | yes | thermal sensitivity of the payload |
| 6 | `transit_hours` | numeric | yes (`tanh`) | dominant continuous driver; saturating |
| 7 | `ambient_temp_c` | numeric | yes | linear |
| 8 | `handoff_count` | numeric | yes | linear |
| 9 | `payload_kg` | numeric | **no** | *deliberately irrelevant* |
| 10 | `sensor_gap_minutes` | numeric, **nullable** | yes (`log1p`) | MAR holes on economy lanes |

Every one of these is knowable at **booking time**. The logger readings that ultimately
*prove* an excursion are not features, because they do not exist when the question is asked
— that rule is what keeps this a forecasting problem rather than a leakage problem.

## The realism requirement — the point of the whole exercise

A latent function plus a whisper of Gaussian noise gives held-out R² ≈ 0.99. That number is
not a triumph, it is a tell: it says the label is a closed-form function of the inputs, SHAP
merely reads the coefficient table back to you, and the conformal interval collapses to a
hairline that impresses nobody and informs no decision.

Aegis's own reference generator gets this wrong — `noise_scale=4.0`, a flat constant against
a signal that spreads far wider, landing around R² 0.97. **This domain does not copy that.**
Seven devices, each a declared constant with a name, put the measured score in the middle of
the 0.45–0.80 band:

1. **Calibrated σ, not a guessed one.** `ml_spec.calibrated_noise_sigma` solves
   `σ = sqrt(var_signal · (1 − r²) / r²)` from the *measured* variance of the latent values
   the generator just computed. A hardcoded constant is correct exactly until the next
   coefficient edit, and then silently stops being.
2. **Two unobserved confounders** — `unrecorded_tarmac_delay` and
   `undocumented_precool_quality`. Both genuinely move the target; neither is ever emitted
   as a column. This is the honest ceiling no model can climb past, and
   `CONFOUNDER_SHARE = 0.4` puts two fifths of the irreducible error there rather than in
   featureless noise.
3. **Heteroscedastic noise** on `transit_hours`: residual spread runs from `σ/1.6` on the
   shortest lanes to `σ·1.6` on the longest, normalised to unit mean square so the total
   budget is *redistributed* rather than enlarged. An adaptive conformal interval now has a
   real reason to breathe.
4. **MAR missingness** (~4%) on `sensor_gap_minutes`, conditioned on `carrier_tier` — not
   MCAR. Under MCAR, median imputation is unbiased and demonstrating it proves nothing;
   under MAR the imputed rows are systematically riskier than the observed ones, which is
   when `MLExplainResponse.imputed_features` becomes information a reviewer can act on. The
   holes are punched **after** the label is computed, from the complete row: the interval
   existed and moved the outcome, it was simply never published.
5. **Two genuinely irrelevant features**, one categorical and one numeric. `origin_region`
   is drawn *independently* of `ambient_temp_c` precisely so it stays a clean negative
   control — real regions do correlate with real temperatures, and letting them correlate
   here would have leaked signal into the one column that is supposed to have none.
6. **One interaction plus two non-linear shapes.** Transit duration is gated on gel-pack
   packaging (a long lane costs far more under gel packs than under a powered reefer);
   `transit_hours` is `tanh`-shaped and `sensor_gap_minutes` is `log1p`-shaped. A purely
   additive latent function is exactly recoverable by ridge regression, which would make the
   boosted ensemble decorative and flatten the SHAP plot.
7. **Class imbalance and boundary label flips** on `excursion_flag`: 28/72, with 3% of
   labels corrupted among the rows *closest to the class boundary*. Uniform label noise is
   the wrong model — the shipment nobody could call either way is the one that gets
   mislabelled.

### Oracle R² is not achieved R², and the difference is measured

`TARGET_R2 = 0.74` is the **oracle** ceiling: what something that already knew the
generating function would score. A real model is always below it, because it has to
*estimate* that function from a finite sample. `realism_report`'s `headroom` field reports
achieved ÷ oracle, so the gap is a number rather than an excuse. Calibrating the oracle to
0.65 instead would put the achieved score near 0.48 — still inside the band, but close
enough to its floor that an ordinary seed-to-seed swing could drop the pipeline through it.

### Measured, on 2,000 generated shipments (≈1,550 labelled rows, seed 11)

```
rows               1548 labelled of 2000 generated
held-out R²        0.6236        band [0.45, 0.80]   ✓   suspiciously_easy: False
oracle R²          0.7403        (a model that KNEW the generating function)
headroom           84.2%         (achieved ÷ oracle)
analytic ceiling   0.7400        (= the declared TARGET_R2, recovered by measurement)
signal variance    156.55
noise σ            5.745 pts     i.i.d. measurement noise
confounder var     22.00         = 40.0% of the irreducible error, as declared
                                 = 10.4% of total target variance
noise-to-signal    0.5927
heteroscedasticity 1.47× residual spread, top vs bottom quartile of transit_hours
missingness        4.01% of sensor_gap_minutes (MAR on carrier_tier)
undriven features  origin_region, payload_kg
interactions       1             non-monotone drivers: none

held-out accuracy  0.8424        band [0.65, 0.88]   ✓   suspiciously_easy: False
majority class     0.7183        → floor 0.7383; clears a constant predictor by +0.124
class balance      no_excursion 0.719 / excursion 0.281
```

Re-derive these at any time:

```bash
.venv/bin/python -c "
from reference.problem import PROBLEM, LATENT, EXCURSION_PROBLEM
import reference.adapter as a
from aegis_ml.data.latent import assert_learnable, realism_report, measure_learnability
df = a.training_frame(num_records=2000)
print('learnable score:', assert_learnable(df, PROBLEM))
print(realism_report(df, PROBLEM, LATENT))
print(measure_learnability(a.excursion_frame(num_records=2000), EXCURSION_PROBLEM))
"
```

## One declared causal story, two evaluators

`ml_spec` holds the coefficient tables as **plain data** — `CATEGORICAL_EFFECTS`,
`NUMERIC_DRIVERS`, `INTERACTION`, `CONFOUNDERS`, `LATENT_INTERCEPT`.

* `ml_spec.latent_spoilage_risk` evaluates them **row by row, in pure Python**, because the
  generator must run with no numpy, no pandas and no scikit-learn: `ml_spec` is imported by
  spec resolution and by the conformance suite in environments where the ML extra is not
  installed at all.
* `problem.LATENT` re-expresses the **same tables** as an `aegis_ml.data.latent.LatentModel`,
  built by reading them rather than by re-typing them, so the vectorised pipeline code can
  calibrate σ over a frame, punch MAR holes and report an oracle R².

Two evaluators of one table is the honest resolution. Two *tables* would be exactly the trap
the generator's own docstring warns about: the formula typed twice, drifting silently the
first time someone edits a coefficient. And the proof they agree is not a comment — it is
`realism_report`'s `oracle_r2`, which scores `LATENT`'s signal against the labels the
*generator* wrote. If the two ever diverged, that number would collapse and the demo would
say so on its own front page.

## The agent surface

**Nine tools**, four the domain's own and five spliced in from `aegis_ml.serve.tools`:

| tool | risk | read-only | destructive | idempotent |
|---|---|---|---|---|
| `find_shipments` | LOW | ✓ | | ✓ |
| `add_shipment_note` | LOW | | | ✓ (content-hashed) |
| `reroute_shipment` | MEDIUM | | | ✓ |
| `quarantine_shipment` | **HIGH** | | ✓ | ✓ |
| `predict_outcome` | LOW | ✓ | | ✓ |
| `explain_prediction` | LOW | ✓ | | ✓ |
| `whatif_scenario` | LOW | ✓ | | ✓ |
| `forecast_series` | LOW | ✓ | | ✓ |
| `check_model_health` | LOW | ✓ | | ✓ |

`risk` is the **only** input to the human approval gate. `quarantine_shipment` is HIGH
because stranding product a clinic is expecting is the decision a human genuinely wants to
confirm; `reroute_shipment` is MEDIUM because the consignment keeps moving and the returned
inverse restores the previous lane exactly. Every ML tool is LOW and read-only, which
preserves the platform's rule: **ML informs; it never gates.**

The ML tools are built from *this adapter's own* `ToolSpec` class by
`ml_tool_specs(ToolSpec, problem=PROBLEM, result_cls=ToolActionResult)`, so the registry
stays homogeneous, the summaries carry the target's unit (`48.2 %`, not `48.2`), and
`aegis_ml` imports nothing from the adapter. Wiring them is three edits and all three are
done: registered in `TOOL_REGISTRY`, granted in `ALLOWLIST` for all three personas, and
granted in the `data` lane's `tool_allowlist` in `roster.py`.

**Three personas**, and the third earns its place:

| persona | role | scope | may reroute? | may quarantine? |
|---|---|---|---|---|
| `logistics_lead` | `admin` | ALL | ✓ | ✓ |
| `quality_auditor` | `ai_team` | ALL | ✗ | ✓ |
| `shipper_client` | `client` | OWN (`shipper_id`) | ✗ | ✗ |

`logistics_lead` and `quality_auditor` have *identical data scope and deliberately different
tool sets*, which is what makes the persona model legible: the boundary between them is
accountability, not secrecy. An auditor who could quietly move a consignment to a cheaper
lane would be auditing their own work.

`find_shipments` is **deliberately not** granted to `shipper_client`. That persona's scope is
OWN on `shipper_id` and the narrowing is applied by the data layer from the authenticated
subject — a value `ToolContext` does not carry — while the tool takes `shipper_id` as a
*filter*. Granting it would not be a roster line, it would be a scope change.

## Memory scoping — a decision, not a default

`memory_subject_for` returns `shipper:{id}` for the client persona and `user:{id}` for
operator-side personas. A durable cold-chain fact is almost always an *account* fact —
"our Lisbon site cannot take dry ice", "this product must never be frozen", "goods-in closes
at 15:00" are true no matter which coordinator is on the call, and re-learning them from each
new coordinator is exactly what long-term memory exists to prevent. Operator turns are
personal working context, and pooling three leads' preferences into one profile would make
all three incoherent. The differing **prefixes** guarantee the two namespaces cannot collide
even when the underlying ids do.

## Running it

```bash
make demo                       # the whole pipeline, end to end
.venv/bin/python scripts/run_demo.py

.venv/bin/python -m ruff check reference scripts

# the adapter contract, with the Aegis checkout on the path
PYTHONPATH=/Users/yrevash/aegis/aegis/src .venv/bin/python -c "
import reference.adapter as adapter
from aegis.adapter import DomainAdapter, missing_members
assert not missing_members(adapter), missing_members(adapter)
assert isinstance(adapter, DomainAdapter)
print('adapter contract: satisfied')
"
```

`scripts/run_demo.py` generates the world, prints the realism evidence **first**, then runs
`data_flow` → `train_flow` → `promote_flow` → `drift_flow` for real, and writes
`registry_store/RUN_SUMMARY.md`. It exits non-zero on any failure; a demo that prints a
traceback and exits 0 is a demo that will be believed when it should not be.

## Standing alone

Every `aegis` import in this package is resolved defensively — `RiskLevel` in `tools.py`,
`Role` in `personas.py`, `SubAgentSpec` in `roster.py`, `ModelRole` in `generator.py`. Each
falls back to a locally declared, **value-identical** `StrEnum` or dataclass, and a `StrEnum`
member hashes and compares as its string value, so a table keyed by a fallback member is
looked up correctly by the platform's own member and vice versa. That is what lets this
domain be run, tested and audited with no Aegis checkout on the path — while still satisfying
`aegis.adapter.DomainAdapter` exactly when the platform is there.
