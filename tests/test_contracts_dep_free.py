"""The dep-free guarantee: ``aegis_ml.contracts`` imports pydantic and nothing else.

Mirrors ``aegis/tests/ml/test_types_is_dep_free.py`` and exists for the same reason: the
backend's light API-schema layer names these shapes, and it must be able to do so without
pulling pandas, sklearn, torch or shap into a web process.

Every assertion runs in a **subprocess**. In-process it would be worthless — by the time
pytest has collected the rest of this suite, pandas and sklearn are already in
``sys.modules`` and a heavy import inside ``contracts`` would be invisible.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = str(_REPO / "src")

BANNED = (
    "pandas",
    "numpy",
    "sklearn",
    "torch",
    "shap",
    "mapie",
    "evidently",
    "nannyml",
    "xgboost",
    "joblib",
    "scipy",
    "matplotlib",
    "pandera",
)
"""Modules that must not appear in ``sys.modules`` after importing the light layer."""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a clean subprocess with ``src`` on the path."""
    env = {**os.environ, "PYTHONPATH": _SRC}
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
        cwd=str(_REPO),
    )


_GUARD = (
    "import sys\n"
    "{imports}\n"
    f"banned = {BANNED!r}\n"
    "hit = sorted(m for m in banned if m in sys.modules)\n"
    "assert not hit, hit\n"
    "print('clean')\n"
)


@pytest.mark.parametrize(
    "module",
    [
        "aegis_ml.contracts",
        "aegis_ml.contracts.spec",
        "aegis_ml.contracts.protocols",
        "aegis_ml.contracts.errors",
        "aegis_ml.settings",
    ],
)
def test_light_module_pulls_no_heavy_dependency(module: str) -> None:
    """Importing one light module alone must add none of the banned heavy deps."""
    proc = _run(_GUARD.format(imports=f"import {module}"))
    assert proc.returncode == 0, f"{module} pulled a heavy dep:\n{proc.stdout}\n{proc.stderr}"
    assert "clean" in proc.stdout


def test_all_light_modules_together_pull_no_heavy_dependency() -> None:
    """Importing the whole light surface at once must still be dep-free."""
    imports = "\n".join(
        f"import {m}"
        for m in (
            "aegis_ml.contracts",
            "aegis_ml.contracts.spec",
            "aegis_ml.contracts.protocols",
            "aegis_ml.contracts.errors",
            "aegis_ml.settings",
        )
    )
    proc = _run(_GUARD.format(imports=imports))
    assert proc.returncode == 0, f"Import guard failed:\n{proc.stdout}\n{proc.stderr}"


def test_contracts_frames_defers_pandera_to_call_time() -> None:
    """``aegis_ml.contracts.frames`` may be imported without pandera arriving with it.

    ``frames`` is the one module in the light package that touches an optional dependency;
    it reaches pandera inside its functions precisely so that importing the contracts
    package stays free. Importing it must therefore also stay free.
    """
    proc = _run(_GUARD.format(imports="import aegis_ml.contracts.frames"))
    assert proc.returncode == 0, f"frames pulled a heavy dep:\n{proc.stdout}\n{proc.stderr}"


def test_light_layer_still_exposes_every_public_name() -> None:
    """The dep-free layer is only useful if it actually re-exports the shapes.

    A module that imports nothing because it also *exports* nothing would pass the guard
    above and be worthless, so the guard is paired with a completeness check.
    """
    import aegis_ml.contracts as contracts

    for name in contracts.__all__:
        assert hasattr(contracts, name), f"{name} is in __all__ but not bound"
