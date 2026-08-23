"""Runtime configuration, read from ``AEGIS_ML_*`` environment variables.

Paths default to positions relative to this checkout so a fresh clone runs with no
configuration at all, and every path a pipeline writes to is resolvable before any
expensive stage starts — ``aegis-ml doctor`` prints them.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "settings"]

_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Everything the pipelines need to locate, tune, or switch off.

    Attributes:
        aegis_root: The Aegis checkout. ``artifact_path`` is derived from it, and that is
            the file ``aegis.ml.get_model()`` loads — promotion writes exactly there, which
            is why this package needs no changes inside ``aegis/``.
        trainer_venv: The isolated venv holding AutoGluon/TabPFN/torch. Kept separate
            because they will not resolve under the backend's pandas<2.4 / numpy<2.5 /
            numba==0.67.0 caps.
        enable_tabpfn: TabPFN-2.5's weights are Prior Labs License — research and
            evaluation are permitted, commercial and production use are not. On by default
            because a hackathon demo is evaluation use; every card it touches prints the
            notice. Set ``AEGIS_ML_ENABLE_TABPFN=0`` to switch it off.
    """

    model_config = SettingsConfigDict(env_prefix="AEGIS_ML_", extra="ignore")

    aegis_root: Path = _ROOT.parent / "aegis"
    registry_dir: Path = _ROOT / "registry_store"
    reports_dir: Path = _ROOT / "registry_store" / "reports"
    trainer_venv: Path = _ROOT / ".venv-ml"

    enable_tabpfn: bool = True
    enable_autogluon: bool = True
    enable_flaml: bool = True
    enable_mlflow: bool = False
    enable_prefect: bool = False

    automl_time_budget: int = 300
    hpo_trials: int = 60
    hpo_timeout: int = 600
    random_seed: int = 7

    requested_coverage: float = 0.9
    coverage_tolerance: float = 0.05
    promote_min_gain: float = 0.005
    learnable_r2_floor: float = 0.15
    learnable_accuracy_floor: float = 0.55
    leakage_threshold: float = 0.98

    drift_share_warn: float = 0.2
    drift_share_block: float = 0.4

    postgres_dsn: str | None = None

    @property
    def artifact_path(self) -> Path:
        """The joblib file ``aegis.ml.get_model()`` loads in the backend host.

        Read off ``backend/src/app/ml/__init__.py``'s own constant, which resolves to
        ``backend/.artifacts/ml_spine.joblib`` — deliberately NOT the library's
        ``aegis/src/aegis/ml/artifacts/`` path. Training through the library constant
        writes where nothing loads from, and the endpoints keep answering 503.
        """
        return self.aegis_root / "backend" / ".artifacts" / "ml_spine.joblib"

    @property
    def adapter_dir(self) -> Path:
        """The one directory a retarget changes."""
        return self.aegis_root / "backend" / "src" / "app" / "adapter"

    @property
    def trainer_python(self) -> Path:
        """Interpreter inside the isolated trainer venv (POSIX and Windows layouts)."""
        win = self.trainer_venv / "Scripts" / "python.exe"
        return win if win.exists() else self.trainer_venv / "bin" / "python"


settings = Settings()
"""Process-wide settings singleton."""
