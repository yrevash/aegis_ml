"""Shared fixtures, markers and the blast-radius guard for the whole suite.

Two things here are load-bearing rather than convenience.

**``_isolated_paths`` is autouse.** Every test runs with ``settings.aegis_root``,
``settings.registry_dir`` and ``settings.reports_dir`` repointed into its own ``tmp_path``.
``settings.artifact_path`` is derived from ``aegis_root``, so this makes it impossible for a
test to overwrite the real serving artifact at
``/Users/yrevash/aegis/backend/.artifacts/ml_spine.joblib`` even by accident — a promotion
test that forgot to isolate itself writes into a temp directory instead of over the model a
demo is about to load.

**The frame fixtures are session-scoped.** ``reference.adapter.ml_spec.training_frame`` runs
the real procedural generator; regenerating it per test would dominate the runtime. Tests
that mutate a frame take the ``frame`` fixture, which hands out a copy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# `pythonpath = ["src", "."]` in pyproject covers a normal `pytest` invocation; this makes
# the suite work when it is collected from another working directory too.
for _entry in (str(SRC_ROOT), str(REPO_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def pytest_configure(config: pytest.Config) -> None:
    """Register the suite's custom markers.

    Without this an unknown-mark warning appears for every marked test.
    """
    config.addinivalue_line(
        "markers",
        "slow: takes more than a couple of seconds (real AutoML search, full drift sweep).",
    )
    config.addinivalue_line(
        "markers",
        "aegis: needs the Aegis platform importable (PYTHONPATH=/Users/yrevash/aegis/aegis/src).",
    )


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repoint every settings path at ``tmp_path`` for the duration of one test.

    Autouse and unconditional. The registry, the reports directory and — through
    ``aegis_root`` — the serving artifact all move, so no test can reach the real ones.
    """
    from aegis_ml.settings import settings

    fake_aegis = tmp_path / "aegis_root"
    (fake_aegis / "backend" / ".artifacts").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "aegis_root", fake_aegis)
    monkeypatch.setattr(settings, "registry_dir", tmp_path / "registry")
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "registry" / "reports")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The checkout root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def problem() -> Any:
    """The reference domain's regression problem (``spoilage_risk_pct``)."""
    from reference.problem import PROBLEM

    return PROBLEM


@pytest.fixture(scope="session")
def excursion_problem() -> Any:
    """The reference domain's classification problem (``excursion_flag``)."""
    from reference.problem import EXCURSION_PROBLEM

    return EXCURSION_PROBLEM


@pytest.fixture(scope="session")
def latent() -> Any:
    """The reference domain's declared latent model."""
    from reference.problem import LATENT

    return LATENT


@pytest.fixture(scope="session")
def seed() -> int:
    """The reference domain's default seed."""
    from reference.problem import SEED

    return SEED


@pytest.fixture(scope="session")
def reference_frame() -> pd.DataFrame:
    """A real generated cold-chain frame — roughly 940 rows, built once for the session.

    Do not mutate it; take the ``frame`` fixture for that.
    """
    from reference.adapter import ml_spec
    from reference.problem import SEED

    return ml_spec.training_frame(num_records=1200, seed=SEED)


@pytest.fixture
def frame(reference_frame: pd.DataFrame) -> pd.DataFrame:
    """A private copy of :func:`reference_frame`, safe to mutate."""
    return reference_frame.copy()


@pytest.fixture(scope="session")
def excursion_frame() -> pd.DataFrame:
    """A real generated classification frame for ``excursion_flag``."""
    from reference.adapter import ml_spec
    from reference.problem import SEED

    return ml_spec.excursion_frame(num_records=1200, seed=SEED)
