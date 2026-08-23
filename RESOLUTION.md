# Dependency resolution record — the `serve` tier

The single biggest risk in this project was never a modelling question. It was whether the
ML/MLOps stack could be installed **alongside** Aegis at all, given the hard caps the
backend carries (`backend/pyproject.toml`):

```
pandas>=2.2,<2.4      # nemoguardrails co-existence
numpy>=1.26,<2.5      # presidio-analyzer; numba/llvmlite via shap
[tool.uv] constraint-dependencies = ["numba==0.67.0", "litellm==1.96.0", "presidio-analyzer==2.2.364"]
```

A plan that adds nine or twelve dependencies without ever running the resolver is a plan
that discovers the conflict on the morning of the hackathon. So this was resolved first.

## Result — resolved and verified

| Package | Resolved | Cap it had to satisfy | Status |
|---|---|---|---|
| pandas | 2.3.3 | `>=2.2,<2.4` | inside |
| numpy | 2.4.6 | `>=1.26,<2.5` | inside |
| numba | 0.67.0 | `==0.67.0` (exact) | exact match |
| scikit-learn | 1.9.0 | `>=1.5` | inside |
| xgboost | 2.1.4 | `>=2.1` | inside |
| mapie | 1.5.0 | `>=1.4` | inside |
| shap | 0.51.0 | `>=0.46` | inside |
| pandera | 0.32.1 | `>=0.29` | inside |
| skrub | 0.10.0 | `>=0.5` | inside |
| optuna | 4.9.0 | `>=4.0` | inside |
| flaml | 2.6.0 | `>=2.3` | inside |
| evidently | 0.7.21 | `>=0.7` — modern `Report`/`presets` API | inside |
| nannyml | 0.13.1 | `>=0.13` | inside |
| pyarrow | 25.0.1 | `>=17` | inside |
| joblib | 1.5.3 | `>=1.4` | inside |
| typer | 0.27.1 | `>=0.15` | inside |

**Conclusion: the entire `serve` tier co-installs with Aegis with zero cap violations and
zero version overrides.** Nothing had to be loosened on either side.

Reproduce with:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv -e '.[dev]'
```

Exact pins are frozen in `requirements-serve.lock.txt`.

## The `strong` tier is deliberately NOT in this venv

AutoGluon, TabPFN-2.5 and torch bring their own pandas/numpy floors that would fight the
caps above. They live in a separate `.venv-ml`, and the AutoML search's answer crosses back
as a portable JSON `Recipe` — see decision **D1** in `finalplan.md`. That isolation is why
this table has no conflicts in it: the conflict was designed out rather than resolved.

```bash
uv venv .venv-ml --python 3.11
uv pip install --python .venv-ml -e '.[strong,serve]'
```

### The `strong` tier resolves too — verified, not assumed

| Package | Resolved |
|---|---|
| autogluon-tabular / -timeseries / -core / -common | 1.6.1 |
| tabpfn | 8.4.0 (the TabPFN-2.5 model; the package major jumped independently of the model version) |
| tabpfn-extensions | 0.6.0 |
| torch / torchvision / torchmetrics | 2.10.0 / 0.25.0 / 1.9.0 |
| sdv / ctgan / copulas | 1.38.1 / 0.12.1 / 0.14.1 |
| mlforecast | 0.14.0 |
| xgboost-cpu (AutoGluon's own) | 3.2.0 |

**The important line in that table is the one that is not there.** `pandas`, `numpy` and
`scikit-learn` resolve to **2.3.3 / 2.4.6 / 1.9.0** — byte-identical to the serving tier. So a
model fitted in the trainer venv and a model re-fitted from its recipe in the serving venv
see the same numerics, which is the assumption the portable-recipe design rests on. It was
worth checking rather than hoping.

Two pins had to be corrected to get here, both found by running the resolver:

- `mlforecast>=1.0` → `>=0.13,<0.15`. `autogluon.timeseries` pins `mlforecast<0.15`, and an
  unbounded floor made the entire `strong` extra unresolvable.
- `lightgbm>=4.6` → `>=4.0` (no ceiling). See the section below — `nannyml` needs `<4.6`,
  and the trainer venv installs `.[strong,serve]`, so both constraints apply at once.

Exact pins frozen in `requirements-strong.lock.txt` (1,174 packages).

---

## A three-way constraint worth knowing about: lightgbm ↔ nannyml ↔ scikit-learn

Found by executing the code, not by reading version metadata. Recording it because it is
non-obvious and it will otherwise be rediscovered on hackathon morning.

**The facts, each verified in this venv:**

1. `scikit-learn` removed the `force_all_finite=` keyword from `check_X_y` in 1.8. This
   project resolves to sklearn **1.9.0**.
2. `lightgbm` 4.5.0's *scikit-learn wrapper* (`LGBMRegressor` / `LGBMClassifier`) still
   passes that keyword, so every fit through the wrapper raises
   `TypeError: check_X_y() got an unexpected keyword argument 'force_all_finite'`.
   Fixed in lightgbm 4.6.
3. `nannyml>=0.13.0` requires `lightgbm>=3.3,<4.6` — so we **cannot** simply raise the floor
   without dropping NannyML, and NannyML is the one tool in the stack that estimates live
   performance *without ground-truth labels*.

**What is actually broken, and what is not.** The wrapper is the only casualty. Both
NannyML and FLAML drive LightGBM through its **native** `Dataset`/`train` API, which never
touches `check_X_y`:

| Path | Status | Evidence |
|---|---|---|
| `nannyml.DLE` fit + estimate | works | `estimated_rmse = 2.07 [1.71, 2.44]` on unlabelled current data |
| FLAML with `lgbm` in the estimator list | works | best estimator `lgbm`, r² 0.9310 |
| FLAML without `lgbm` | works | best estimator `xgboost`, r² 0.8711 |
| Direct `LGBMRegressor(...).fit(...)` | **raises** | `TypeError` as above |

**Resolution: change nothing, and let the recipe layer report it.** `lightgbm` stays where
NannyML pins it. FLAML keeps `lgbm` in its search list because FLAML's own path is fine. The
only affected code is our own direct-wrapper candidate, and `automl.recipe.is_portable_kind`
already handles exactly this case the right way — the member is dropped from the recipe with
its reason recorded in `Leaderboard.tiers_skipped`, rather than the tier silently vanishing.

That is the design working as intended: an unavailable estimator and an estimator that lost
on merit must never look the same on the leaderboard.

**If you want the wrapper back**, the trade is explicit: drop NannyML (losing label-free
performance estimation) and pin `lightgbm>=4.6`. Do not take that trade for a hackathon —
XGBoost and HistGradientBoosting cover the same ground, and label-free monitoring is the
more interesting thing to demo.

---

## TabPFN needs a one-time, browser-based setup — do it before the day

Found by running `.fit()`, not by installing. `pip install tabpfn` succeeds, the import
succeeds, and **then** `.fit()` raises:

```
tabpfn.errors.TabPFNLicenseError: TabPFN requires a one-time license acceptance to
download model weights for local inference, but no interactive terminal is available.
```

From TabPFN 8.x, Prior Labs gates the weight download behind a licence acceptance plus an
API token. The failure lands *inside* the search, after the earlier tiers have already spent
their budget — the worst possible moment.

`aegis_ml.automl.tiers` now probes for this at capability-report time, so it shows up in
`aegis-ml doctor` and in `Leaderboard.tiers_skipped` instead of as a traceback:

```
  baseline    available
  flaml       available
  autogluon   available
  tabpfn      importable, but no model weights are available and TABPFN_TOKEN is unset,
              so .fit() would raise TabPFNLicenseError mid-search. One-time setup: ...
```

**The setup, which needs a browser and a network — so not on hackathon morning:**

1. Register at <https://ux.priorlabs.ai>
2. Accept the licence on the **Licenses** tab
3. Copy the API key from the **Account** page
4. `export TABPFN_TOKEN="<key>"` (add it to your shell profile)
5. Run one fit to pull the weights into `~/Library/Caches/tabpfn` (`%LOCALAPPDATA%\tabpfn`
   on Windows). After that the tier works offline and the token is no longer needed.

**If you skip it, nothing breaks.** The tier reports itself unavailable with the reason,
and AutoGluon covers the same ground — measured here at R² 0.5411 across 10 models in 20 s
on data whose analytic ceiling is 0.65. Licence terms are unchanged: Prior Labs License,
research and evaluation permitted, commercial and production use not.
