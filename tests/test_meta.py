"""Meta checks: the repository's own guarantees about itself.

Three things, all of which are cheap and all of which have failed at least once in a
codebase of this shape:

* every module under ``src/aegis_ml`` imports cleanly in the serving venv;
* ``scripts/audit_no_mocks.py`` exits 0 — no placeholder, no empty body, no swallowed import;
* ``aegis-ml doctor`` exits 0 — the command a human runs first on demo morning.

The last two run through ``subprocess`` because their exit code is the contract.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _child_env(tmp_path: Path) -> dict[str, str]:
    """Environment for a subprocess: ``src`` on the path, registry pointed at a temp dir."""
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO)]),
        "AEGIS_ML_REGISTRY_DIR": str(tmp_path / "registry"),
        "AEGIS_ML_REPORTS_DIR": str(tmp_path / "registry" / "reports"),
    }


# ── every module imports ──────────────────────────────────────────────────────


def _module_names() -> list[str]:
    """Every importable module under ``aegis_ml``, including subpackages."""
    import aegis_ml

    return sorted(m.name for m in pkgutil.walk_packages(aegis_ml.__path__, "aegis_ml."))


def test_the_package_has_the_expected_module_count() -> None:
    """A sanity floor, so a collection bug cannot make the import sweep vacuous."""
    names = _module_names()
    assert len(names) >= 50, f"only {len(names)} modules found; the sweep below is not real"


@pytest.mark.parametrize("module", _module_names())
def test_every_module_imports_cleanly(module: str) -> None:
    """A module that only imports under one venv is a module nobody can review."""
    importlib.import_module(module)


def test_every_module_declares_all() -> None:
    """``__all__`` is the module's stated public surface; a missing one is an accident."""
    missing = []
    for name in _module_names():
        module = importlib.import_module(name)
        if not hasattr(module, "__all__"):
            missing.append(name)
    assert missing == [], f"modules with no __all__: {missing}"


def test_no_module_exports_a_name_it_does_not_bind() -> None:
    """An ``__all__`` entry with nothing behind it is an ImportError waiting for a consumer."""
    broken: list[str] = []
    for name in _module_names():
        module = importlib.import_module(name)
        for exported in getattr(module, "__all__", []):
            if not hasattr(module, exported):
                broken.append(f"{name}.{exported}")
    assert broken == [], f"names in __all__ that are not bound: {broken}"


# ── the no-mocks audit ────────────────────────────────────────────────────────


def test_audit_no_mocks_exits_zero(tmp_path: Path) -> None:
    """``src/`` carries no placeholder, no empty body and no swallowed import.

    This is the mechanical half of the user's requirement that test doubles stay out of
    shipped code. The other half is that this suite keeps every double in
    ``tests/fixtures/``.
    """
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(REPO / "scripts" / "audit_no_mocks.py")],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO),
        env=_child_env(tmp_path),
    )
    assert proc.returncode == 0, f"audit_no_mocks found placeholders:\n{proc.stdout}\n{proc.stderr}"
    assert "PASS" in proc.stdout


def test_no_test_double_lives_outside_tests_fixtures() -> None:
    """The suite's own half of the rule, enforced the same blunt way the audit enforces src/."""
    from tests.fixtures.doubles import BANNED_DOUBLE_TOKENS

    offenders: list[str] = []
    for path in sorted((REPO / "tests").rglob("*.py")):
        if path.parent.name == "fixtures":
            continue
        text = path.read_text(encoding="utf-8")
        for token in BANNED_DOUBLE_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(REPO)}: {token}")
    assert offenders == [], f"test doubles outside tests/fixtures/: {offenders}"


# ── the CLI ───────────────────────────────────────────────────────────────────


def test_doctor_exits_zero(tmp_path: Path) -> None:
    """``aegis-ml doctor`` is what a human runs first; exit 0 means the morning can proceed."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "aegis_ml.cli", "doctor"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
        env=_child_env(tmp_path),
    )
    assert proc.returncode == 0, f"doctor failed:\n{proc.stdout}\n{proc.stderr}"
    assert "VERDICT: ready" in proc.stdout


def test_doctor_reports_the_tiers_it_can_and_cannot_run(tmp_path: Path) -> None:
    """A skipped tier is never silent: an empty slot and a missing dep look identical otherwise."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "aegis_ml.cli", "doctor"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
        env=_child_env(tmp_path),
    )
    assert "baseline" in proc.stdout
    assert "skipped" in proc.stdout, "no tier is skipped in the serving venv — is doctor lying?"
    assert "autogluon" in proc.stdout


def test_doctor_and_the_tier_module_agree(tmp_path: Path) -> None:
    """Two sources of truth for tier availability is exactly the bug ISSUES.md #1 recorded.

    ``_tier_report`` must be a projection of ``tiers.tier_status()``, not a reimplementation
    that checks importability only.
    """
    from aegis_ml.automl.tiers import tier_status

    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "aegis_ml.cli", "doctor"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
        env=_child_env(tmp_path),
    )
    tier_lines = {
        parts[1]: parts[0]
        for parts in (ln.split() for ln in proc.stdout.splitlines())
        if len(parts) >= 2 and parts[0] in ("RUNS", "skipped")
    }
    assert set(tier_lines) == set(tier_status()), (
        f"doctor's tier table lists {sorted(tier_lines)} but the module knows "
        f"{sorted(tier_status())}"
    )
    for tier, status in tier_status().items():
        runs_in_cli = tier_lines[tier] == "RUNS"
        runs_in_module = status == "available"
        assert runs_in_cli == runs_in_module, (
            f"doctor says {'RUNS' if runs_in_cli else 'skipped'} for {tier!r} but "
            f"tiers.tier_status() says {status!r}"
        )


def test_cli_help_exits_zero(tmp_path: Path) -> None:
    """The entry point resolves and typer builds the command tree."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "aegis_ml.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO),
        env=_child_env(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert "doctor" in proc.stdout


# ── optional dependencies fail closed ─────────────────────────────────────────


def test_require_names_the_exact_install_when_a_module_is_absent() -> None:
    """A missing optional dep must raise naming the fix, never fall through to a weaker path.

    Uses ``tests.fixtures.doubles.hidden_module`` — every optional dep is installed in this
    venv, so the absent-dependency branch cannot otherwise be reached.
    """
    from aegis_ml._require import require
    from tests.fixtures.doubles import hidden_module

    with hidden_module("autogluon"), pytest.raises(ImportError) as excinfo:
        require("aegis-ml[strong]", "autogluon.tabular")

    message = str(excinfo.value)
    assert "uv pip install 'aegis-ml[strong]'" in message
    assert "falls back" in message


def test_is_available_reports_rather_than_degrades() -> None:
    """``is_available`` is a capability probe; ``False`` IS the answer, not a downgrade."""
    from aegis_ml._require import is_available

    assert is_available("pandas") is True
    assert is_available("a_module_that_does_not_exist_anywhere") is False


def test_unavailable_tier_is_refused_with_a_reason() -> None:
    """Asking for a tier that cannot run raises rather than silently using a weaker one."""
    from aegis_ml.automl.tiers import require_tier, unavailable_reason
    from aegis_ml.contracts.errors import AutoMLTierUnavailableError

    reason = unavailable_reason("autogluon")
    assert reason, "autogluon is not installed in the serving venv; doctor says so"
    with pytest.raises(AutoMLTierUnavailableError):
        require_tier("autogluon")


def test_baseline_tier_is_always_available() -> None:
    """The floor every other tier must beat has to exist wherever the package is installed."""
    from aegis_ml.automl.tiers import unavailable_reason

    assert unavailable_reason("baseline") is None
