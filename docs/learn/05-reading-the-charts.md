# 05 · Reading the charts

[← 04](04-the-pipeline.md) · [Index](00-index.md) · Next: [06 · MLOps](06-mlops-registry-gate-drift.md)

Every training run and every drift run writes a **visual bundle** to
`registry_store/runs/<run_id>/visuals/`: nine PNGs, an `index.html` that stitches them together
with no external references, and an `interactive.html`. Rebuild it any time with:

```bash
.venv/bin/aegis-ml visuals --run-id <run_id>
```

The nine images below are the real output of the committed run
`cold_chain_logistics-20260824T030131425-34e3f5` — r² 0.7199 over 407 held-out rows.

A chart that cannot be drawn is **omitted with a reason**, never faked. This run has no chart
10 (`10_forecast.png`), and its manifest says why: *"no forecast payload for this run —
`forecast_flow` writes one per series and this run registered a tabular model, not a series."*

---

## 01 · Prediction vs actual

![Predicted vs measured, with the 90% conformal band](../images/01_prediction_vs_actual.png)

**What it plots.** One dot per held-out row: the measured `spoilage_risk_pct` on the x axis,
the model's prediction on the y. The dashed line is `y = x` — perfect prediction. The shaded
ribbon is the 90 % conformal band, `±13.53` percentage points. Dots are coloured by whether
their interval contained the truth.

**Good looks like.** A cloud that hugs the diagonal along its whole length, with roughly the
requested share of dots inside the ribbon, and the misses scattered evenly.

**Bad looks like.** A cloud that bends away from the line at one end — the model is *biased*
in that range, and no interval width fixes bias. Or a flat horizontal cloud — the model is
predicting the mean and has learned nothing.

**From this run.** 372 inside, 35 outside — 91.40 % against a requested 90 %. But the orange
misses are **not** evenly scattered: almost all of them sit above measured 45 %, and the
right-hand tail past 60 % is nearly all orange. The constant-width band is too narrow up
there. The headline coverage number cannot see it. Chart 03 can.

---

## 02 · Residuals

![Residual against prediction, with rolling mean and ±1σ](../images/02_residuals.png)

**What it plots.** Residual (`measured − predicted`) on the y axis against the prediction on
the x. The dark line is a rolling mean over a 20-row window; the pale envelope is rolling ±1σ.

**Good looks like.** A band centred on zero whose *height* changes with the prediction, when —
and only when — the data is documented as heteroscedastic. This domain deliberately scales
noise with `transit_hours`, so the fan is the data behaving as declared.

**Bad looks like.** A rolling mean that drifts away from zero (bias, not noise). A visible
curve or step (structure the model did not capture). A perfectly uniform hairline band (see
[chapter 03](03-the-data-problem.md) — the data is a toy).

**From this run.** The title quantifies the fan: σ of the lowest decile **5.68** versus the
highest **10.57**, a ratio of 1.86. The rolling mean stays near zero across the range; overall
mean residual **+0.195** against σ **7.95** — no meaningful bias.

---

## 03 · Conformal coverage — the star

![Marginal coverage next to per-segment coverage](../images/03_conformal_coverage.png)

**What it plots.** Left: the requested level (90.00 %) beside the measured one (91.40 %).
Right: the same measurement repeated *within each segment* of the data — 20 segments, sorted
worst first. The dashed line is the requested 90 %; the dotted red line is the **85 % floor**
below which the promotion gate refuses.

**Good looks like.** Measured ≈ requested on the left, and the right-hand bars clustered
tightly around the dashed line.

**Bad looks like.** Measured far below requested (the interval is a fiction). Or — the
interesting case — measured *fine* on the left while one or more segments sit well below the
line on the right.

**From this run.** This is the finding of the whole bundle.

| | |
|---|---|
| Marginal (overall) coverage | **91.40 %** — comfortably above the requested 90 % |
| Segments below the requested level | **5 of 20** |
| Worst segment | **`route_class = multi_leg` at 82.9 %**, over 76 rows |

82.9 % is **below the 85 % gate floor**. The overall number is an average over these segments,
so it can stay inside tolerance while one of them does not. A shipper whose lanes are all
multi-leg is being handed an interval that is wrong about one shipment in six, while the model
card advertises one in ten.

This is what "marginal coverage" means as a technical term: the guarantee holds *on average
across the whole population*, not *for every subgroup*. Conditional coverage — the guarantee
holding within each segment — is strictly harder and is not what plain split conformal
provides. The right-hand panel exists so that the gap is visible rather than assumed away.

The chart even says so in its own annotation: *"The marginal figure above is an average over
these segments, so it can stay inside tolerance while one of them does not."*

**And note what this means for the gate.** Promotion criterion 2 reads the *marginal* number —
91.40 % against a floor of 85 % — so it passes. The dotted line on the right-hand panel is that
same floor drawn per segment, for the reader's benefit; the gate does not evaluate it. This
segment shortfall is a finding the chart surfaces and the gate does **not** catch. Criterion 4
protects segments in the *metric* (chart 05), not in coverage.

---

## 04 · Global SHAP

![Mean absolute SHAP for all ten declared features](../images/04_shap_global.png)

**What it plots.** For each feature, the average absolute SHAP value over 300 held-out rows —
how much that column moves a single prediction, in the target's own units (percentage points).
All ten declared features are shown, unfiltered.

**Good looks like.** The features your domain expert would name at the top; the features you
deliberately made irrelevant near zero; and no single bar dwarfing everything else.

**Bad looks like.** One feature carrying nearly all the attribution — usually **leakage**. Or
a feature that the domain says is decisive sitting at zero — the encoding is probably wrong.

**From this run.** `carrier_tier` leads at **3.945** points (17.3 %), then `route_class`
(3.372), `product_class` (3.346), `handoff_count` (3.185). Nothing dominates.

The two hatched grey bars are the point: `origin_region` (0.3824) and `payload_kg` (0.3525)
were **generated independently of the target**, and the chart labels them "declared not a
driver". Their combined share of attribution is **3.2 %**. Leaving an irrelevant feature in and
watching the model correctly ignore it is far better evidence than a chart where every column
happens to matter.

---

## 05 · Slice performance

![r2 by segment, worst highlighted](../images/05_slice_performance.png)

**What it plots.** The primary metric recomputed inside each segment — categorical levels
directly, numeric features cut into quartiles. 40 segments here. The dashed vertical line is
the whole-split score; the worst bar is highlighted.

**Good looks like.** A tight spread around the overall line.

**Bad looks like.** A long left tail. A model that improves on average while collapsing on one
segment is a regression *for everyone in that segment*, and the aggregate score is exactly the
instrument that cannot see it.

**From this run.** Overall r² 0.7199. Spread **0.305**, from `handoff_count = q2 (1.0, 2.0]`
at **0.4513** over 97 rows, up to `payload_kg = q1 (4.999, 19.02]` at **0.7566**.

Note which segments cluster at the bottom: `handoff_count = q2` (0.4513),
`route_class = last_mile_pool` (0.4569), `route_class = multi_leg` (0.4708),
`route_class = single_transfer` (0.4899). Journey *shape* is where this model is weakest —
which is a domain finding, not a bug: those are the lanes where the unobserved confounders
(tarmac delay, pre-cool quality) do the most work.

The 0.4513 worst slice is recorded on the gate decision, and becomes the floor every future
challenger must hold. See [chapter 06](06-mlops-registry-gate-drift.md).

---

## 06 · Leaderboard

![Every candidate the search scored, by tier](../images/06_leaderboard.png)

**What it plots.** Every candidate from every tier that ran, bar-coloured by tier, hatched if
not portable, with the promoted one outlined.

**Good looks like.** Several tiers represented, a visible margin between the winner and the
field, and any skipped tier accounted for in the run's `tiers_skipped`.

**Bad looks like.** One bar (nothing else was tried). All bars at the same value (see
[chapter 03](03-the-data-problem.md) — probably a target with no signal). A hatched bar
presented as the model's score.

**From this run.** 11 candidates across `baseline` and `flaml`.
`flaml_xgb_limitdepth` won at **0.7379**, fitted in 0.12 s, and was promoted. The plain
`xgboost` baseline scored 0.7111 and `aegis_spine` — the exact configuration Aegis would have
trained on its own — scored 0.7211, so the search bought about **+0.017 R²** over the status
quo. Small, real, and stated.

The hatched top bar `ridge_reference` at **0.746** is the highest score in the run and was not
promoted: it is a linear model and Aegis's SHAP explainer handles trees only. It is reported as
the accuracy ceiling. `autogluon` and `tabpfn` do not appear at all because the run used the
serving venv; both are listed in `tiers_skipped` with the exact install command.

---

## 07 · Realism

![Achieved vs band vs oracle vs ceiling; variance decomposition; missingness](../images/07_realism.png)

**What it plots.** Three panels answering "is this data honestly hard?" — covered in detail in
[chapter 03 §5](03-the-data-problem.md#5-the-realism-report).

**Good looks like.** The held-out bar inside the shaded band, below the oracle bar, with a
visible confounder + noise share in the middle panel.

**Bad looks like.** The held-out bar *above* the band. That is the R²-0.99 failure, and it is
the one nobody looks for.

**From this run.** probe 0.672 · held-out 0.720 · oracle 0.740 · analytic ceiling 0.740, all
inside `[0.45, 0.80]`. Variance: 74.0 % latent signal, 10.4 % unobserved confounders, 15.6 %
noise. One MAR rule realised at 4.23 %.

---

## 08 · Feature distributions

![Per-feature histograms over the 2034 reference rows](../images/08_feature_distributions.png)

**What it plots.** One panel per declared feature over the frozen reference frame — bar counts
for categoricals, histograms with a density curve for numerics — each annotated with its
missingness.

**Good looks like.** Shapes a domain expert recognises: payload heavily right-skewed, ambient
temperature roughly bell-shaped, handoff counts spiky at small integers. Categorical levels all
present with reasonable counts.

**Bad looks like.** A level with almost no rows (any per-segment metric on it is noise). A
numeric column that is secretly a category. A uniform distribution where the domain implies a
skew — a sign of a lazy generator.

**From this run.** 2,034 rows, ten features. Nine are marked `complete`;
`sensor_gap_minutes` carries the **4.23 % missing** annotation — the MAR rule from
[chapter 03 §4.3](03-the-data-problem.md#43-mar-missingness). `payload_kg` runs from 5 kg to
507 kg with a long right tail; `handoff_count` is discrete 0–7 and visibly spiky at 1 and 2 —
which is why the 40-segment slice sweep quartiles it into uneven groups (n = 176/97/43/91).

---

## 09 · Drift features

![Reference vs current distributions, strongest movement first](../images/09_drift_features.png)

**What it plots.** Reference (training-time) and current (live) distributions overlaid, one
panel per feature, sorted by how far each moved. Numeric features are compared with a
Kolmogorov–Smirnov statistic; categoricals with total variation distance. Panel titles say
whether the drift report **FLAGGED** the feature or judged it stable.

**Good looks like.** Overlapping distributions, small statistics, nothing flagged.

**Bad looks like.** Exactly what this chart shows.

**From this run.** 2,034 reference rows against 934 current rows, **7 of 10 features flagged**,
drifted share **0.70**, verdict **`block`**.

| Feature | Statistic | What moved |
|---|---|---|
| `route_class` | TV 0.544 | `multi_leg` went from ~18 % of shipments to ~72 % |
| `carrier_tier` | TV 0.468 | `economy` went from ~30 % to ~76 % |
| `ambient_temp_c` | KS 0.373 | the whole distribution shifted warmer, with a pile-up at the 40 °C cap |
| `handoff_count` | KS 0.315 | mass moved from 1 handoff to 2–3 |
| `sensor_gap_minutes` | KS 0.210 | longer telemetry gaps |
| `transit_hours` | KS 0.205 | longer lanes, with a pile-up at the cap |
| `product_class` | TV 0.044 | barely moved, but still flagged |

The three unflagged panels (`packaging_type` 0.040, `origin_region` 0.025, `payload_kg` 0.022)
are drawn in grey for contrast, which is what makes the flagged ones readable as *movement*
rather than as noise.

This current frame was **deliberately synthesised** by `full_flow` — a hot season on longer,
cheaper lanes, the way this domain actually degrades. The manifest labels it
`synthetic_stress_shift`, because a drift number computed against data we distorted ourselves
is a demonstration of the detector, not evidence about the world. Say that out loud when you
show it.

---

## The one-page version

| Chart | The single question it answers |
|---|---|
| 01 | Do the predictions track the truth, and does the band hold? |
| 02 | Is the error unbiased, and does its spread behave as documented? |
| 03 | Does the promised coverage hold **for every segment**, not just on average? |
| 04 | Which columns drive the answer — and do the irrelevant ones stay quiet? |
| 05 | Which segment is the model worst for? |
| 06 | What else was tried, and by how much did the winner win? |
| 07 | Is this data honestly hard, or a toy? |
| 08 | What does the raw data actually look like? |
| 09 | Has the world moved away from what we calibrated on? |

Next: [06 · MLOps: registry, gate, drift](06-mlops-registry-gate-drift.md)
