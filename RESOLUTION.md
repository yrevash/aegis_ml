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
