"""Load ``config/*.toml`` into :class:`~aegis_ml.settings.Settings`.

This module exists because of a defect a documentation pass found: the five files under
``config/`` each opened with "Read by ``aegis_ml.<module>``", and **nothing read them**.
There was no ``tomllib`` import anywhere in the package. Someone tuning
``automl.toml``'s ``time_budget`` on hackathon morning would have changed nothing, seen the
old behaviour, and gone looking in the wrong place — which is precisely the class of silent
no-op this project exists to eliminate.

The TOML files are *sectioned by topic* (``[search]``, ``[gate]``, ``[drift]``) while
``Settings`` is deliberately flat, so the two are bridged by an explicit table rather than
by clever auto-nesting. The table is the contract: a key not in it is not a setting, and
:func:`unknown_keys` reports any it finds instead of ignoring them — an ignored key in a
config file is the same silent no-op in miniature.

Precedence, highest first: environment variables (``AEGIS_ML_*``) → ``config/*.toml`` →
the field defaults in ``Settings``. Environment wins so a one-off override never requires
editing a file that is under version control.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

__all__ = ["CONFIG_DIR", "TOML_TO_SETTING", "load_config_overrides", "unknown_keys"]

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
"""Where the TOML files live, relative to the installed package."""

#: ``(file stem, section, key)`` → ``Settings`` field name.
#:
#: Written out rather than derived. A derived mapping would silently absorb a typo in a
#: section name as "a key for a setting that does not exist yet"; this one cannot.
TOML_TO_SETTING: dict[tuple[str, str, str], str] = {
    ("automl", "search", "time_budget"): "automl_time_budget",
    ("automl", "search", "seed"): "random_seed",
    ("automl", "hpo", "n_trials"): "hpo_trials",
    ("automl", "hpo", "timeout"): "hpo_timeout",
    ("contracts", "leakage", "threshold"): "leakage_threshold",
    ("contracts", "realism", "r2_band"): "realism_r2_band",
    ("contracts", "realism", "accuracy_band"): "realism_accuracy_band",
    ("contracts", "realism", "suspiciously_easy_r2"): "suspiciously_easy_r2",
    ("contracts", "realism", "suspiciously_easy_accuracy"): "suspiciously_easy_accuracy",
    ("monitoring", "drift", "warn_share"): "drift_share_warn",
    ("monitoring", "drift", "block_share"): "drift_share_block",
    ("monitoring", "gate", "min_gain"): "promote_min_gain",
    ("monitoring", "gate", "coverage_tolerance"): "coverage_tolerance",
    ("forecast", "series", "level"): "requested_coverage",
    ("pipeline", "prefect", "enabled"): "enable_prefect",
    ("pipeline", "mlflow", "enabled"): "enable_mlflow",
}

#: Sections read for their own consumers rather than for ``Settings``. Listed so
#: :func:`unknown_keys` does not report them as mistakes.
_ADVISORY_SECTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("automl", "baseline"), ("automl", "flaml"), ("automl", "autogluon"),
        ("automl", "tabpfn"), ("contracts", "realism.structure"), ("contracts", "validation"),
        ("monitoring", "performance"), ("forecast", "models"), ("forecast", "intervals"),
        ("forecast", "backtest"), ("pipeline", "run"), ("pipeline", "retry"),
        ("pipeline", "output"),
    }
)


def _read(stem: str) -> dict[str, Any]:
    """Parse one ``config/<stem>.toml``, or return ``{}`` when it is absent.

    An absent file is legitimate — the package must run from a wheel with no ``config/``
    directory beside it — so this is the one place a missing input is not an error. A
    *malformed* file is still an error: it means someone edited it and got it wrong, and
    silently falling back to defaults there would hide the edit.
    """
    path = CONFIG_DIR / f"{stem}.toml"
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config_overrides() -> dict[str, Any]:
    """Return ``Settings`` field overrides drawn from ``config/*.toml``.

    Returns:
        Field name → value, for every mapped key present in the files. Values are passed
        through untouched; pydantic validates and coerces them, so a bad type fails loudly
        at construction rather than deep inside a flow.
    """
    overrides: dict[str, Any] = {}
    for (stem, section, key), field in TOML_TO_SETTING.items():
        data = _read(stem)
        block = data.get(section)
        if isinstance(block, dict) and key in block:
            overrides[field] = block[key]
    return overrides


def unknown_keys() -> list[str]:
    """Return ``file:section.key`` for every scalar key no setting consumes.

    Reported rather than ignored. A key sitting in a config file that nothing reads looks
    exactly like a key that works, and that misreading is what this module was written to
    end. ``aegis-ml doctor`` prints whatever this returns.
    """
    stray: list[str] = []
    for stem in ("automl", "contracts", "monitoring", "forecast", "pipeline"):
        for section, block in _read(stem).items():
            if not isinstance(block, dict) or (stem, section) in _ADVISORY_SECTIONS:
                continue
            for key in block:
                if isinstance(block[key], dict):
                    continue
                if (stem, section, key) not in TOML_TO_SETTING:
                    stray.append(f"{stem}.toml:{section}.{key}")
    return stray
