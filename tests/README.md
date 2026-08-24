# Test suite

```bash
.venv/bin/python -m pytest tests -q                 # everything that needs no Aegis checkout
PYTHONPATH=/Users/yrevash/aegis/aegis/src \
  .venv/bin/python -m pytest tests -q               # + the DomainAdapter conformance checks
.venv/bin/python -m pytest tests -q -m "not slow"   # skip the few genuinely slow ones
```

## The one rule

**Test doubles live in `tests/fixtures/` and nowhere else.** Nothing under `src/` may contain
a mock, a fake, a stub, or an `if TESTING:` branch — that is what makes the shipped code
trustworthy, and `scripts/audit_no_mocks.py` enforces it mechanically (`tests/test_meta.py`
runs that audit as a test, so a violation fails the suite).

The corollary: **when a test fails, the code is wrong, not the test.** Several of these tests
exist because something in `src/` was wrong and nobody noticed. Do not edit `src/` to make a
test pass without understanding which of the two is actually mistaken.

## What each file covers

| File | Covers |
|---|---|
| `test_contracts_dep_free.py` | `aegis_ml.contracts` imports pandas/numpy/sklearn/torch/shap **not at all**, checked in a subprocess. Mirrors Aegis's own `test_types_is_dep_free.py`. The light API-schema layer depends on this. |
| `test_data_realism.py` | The point of the package: held-out score lands *inside* the band, a deterministic target trips the too-easy ceiling, a noise target raises `LabelNotLearnableError`, and the RNG streams are independent. |
| `test_data_contracts.py` | pandera accepts a valid frame and rejects unseen levels, out-of-range numerics, nulls in non-nullable columns, missing columns. |
| `test_splits_and_leakage.py` | Three-way split disjointness, zero group overlap, time-ordered refusing to shuffle, `min_calibration_rows` matching MAPIE's `ceil((n+1)*level) <= n`, and leakage detection. |
| `test_recipe_portability.py` | The two-venv keystone: JSON round-trip, `to_aegis_members`, `fit_recipe` really fitting, `RecipeNotPortableError`, and that `is_portable_kind` means **fittable**, not importable. |
| `test_metric_direction.py` | Every entry in `HIGHER_IS_BETTER`. A wrong direction silently promotes the worse model, so each is asserted individually. |
| `test_promotion_gate.py` | All five criteria, the no-champion first-model case, missing coverage failing, and `reasons` populated on PASS as well as FAIL. |
| `test_registry.py` | Save → reload → predict, atomic promote, rollback, `reindex()`, and run-id validation against path escape *and* shell-glob metacharacters. |
| `test_registry_db.py` | The optional Postgres/SQLAlchemy tables, against real async SQLite. Skipped without `sqlalchemy[asyncio]` + `aiosqlite`. |
| `test_drift.py` | Evidently: stable-vs-stable stays quiet; a deliberately shifted frame reports the shifted columns and blocks. |
| `test_determinism.py` | Same seed → identical frames, splits and predictions. This is what makes a demo repeatable. |
| `test_reference_domain.py` | The cold-chain domain satisfies `DomainAdapter`, `resolve_spec` does not fall back to noise, every skill is reachable, and every ML tool is LOW-risk read-only. |
| `test_pipeline_end_to_end.py` | A real `train_flow` → measure → gate → register cycle on a small frame. |
| `test_meta.py` | Every module imports, `aegis-ml doctor` exits 0, and the no-mocks audit passes. |

## Which tests need what

- **Nothing extra**: most of the suite.
- **`PYTHONPATH` to the Aegis checkout**: three tests in `test_reference_domain.py` that check the real `aegis.adapter.DomainAdapter` Protocol. They *skip* rather than fail without it, so a machine with no Aegis checkout still gets a green suite — but a skip is not a pass, so run them before believing the adapter conforms.
- **`sqlalchemy[asyncio]` + `aiosqlite`** (both in the `[dev]` extra): `test_registry_db.py`.
- **The trainer venv** (`.venv-ml`): nothing here. AutoGluon/TabPFN/SDV are exercised by `scripts/run_demo.py`, not by unit tests — they are too slow and too licence-gated to belong in a suite that must stay fast.

## Speed

The whole suite runs in roughly two minutes warm. Frames are 400–1500 rows and estimators
are deliberately tiny: these tests check *behaviour*, not accuracy. Anything genuinely slow
carries `@pytest.mark.slow`.
