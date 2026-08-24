"""Deliberately-degenerate frames, derived from the real generated one.

Each of these is a failure a real generator produces, reconstructed on purpose:

* ``noise_free_target`` — the latent function evaluated with the noise term switched off,
  which is what a generator that forgot to sample *around* the latent function emits. The
  learnability probe must call it ``suspiciously_easy``.
* ``pure_noise_target`` — a label drawn independently of every feature, which is the trap
  ``SKILL.md`` names. ``assert_learnable`` must refuse it.
* ``leaky_frame`` — an affine restatement of the target added as a feature column, the
  canonical leak.
* ``shifted_frame`` — a copy whose named columns have been moved, for drift.

Nothing here patches ``src/``: every frame is ordinary pandas built from the real one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from aegis_ml.contracts.spec import MLProblem
    from aegis_ml.data.latent import LatentModel

__all__ = [
    "LEAK_COLUMN",
    "SHIFTED_COLUMNS",
    "leaky_frame",
    "noise_free_target",
    "pure_noise_target",
    "shifted_frame",
    "with_leak_declared",
]

LEAK_COLUMN = "settled_risk_pct"
"""Name of the injected leaking column — a plausible-sounding after-the-fact measurement."""

SHIFTED_COLUMNS: tuple[str, ...] = (
    "transit_hours",
    "ambient_temp_c",
    "payload_kg",
    "handoff_count",
    "carrier_tier",
)
"""The columns :func:`shifted_frame` moves. Drift must name every one of them."""


def noise_free_target(
    frame: pd.DataFrame, problem: MLProblem, latent: LatentModel
) -> pd.DataFrame:
    """Replace the target with the latent signal itself — no noise, no confounder.

    Args:
        frame: A real generated frame.
        problem: Its problem, read for the target name.
        latent: The declared latent model.

    Returns:
        A copy whose target is a deterministic function of the features.
    """
    out = frame.copy()
    signal = latent.signal_frame(out)
    out[problem.target.name] = signal.clip(lower=latent.floor, upper=latent.ceiling)
    return out


def pure_noise_target(frame: pd.DataFrame, problem: MLProblem, *, seed: int = 0) -> pd.DataFrame:
    """Replace the target with a uniform draw independent of every feature.

    Args:
        frame: A real generated frame.
        problem: Its problem, read for the target name and bounds.
        seed: Draw seed, so the refusal is reproducible.

    Returns:
        A copy whose target carries no recoverable signal.
    """
    import numpy as np

    out = frame.copy()
    target = problem.target
    low = 0.0 if target.minimum is None else float(target.minimum)
    high = 100.0 if target.maximum is None else float(target.maximum)
    out[target.name] = np.random.default_rng(seed).uniform(low, high, size=len(out))
    return out


def leaky_frame(
    frame: pd.DataFrame, problem: MLProblem, *, factor: float = 1.0001
) -> pd.DataFrame:
    """Add an affine restatement of the target as a feature column.

    Args:
        frame: A real generated frame.
        problem: Its problem, read for the target name.
        factor: Multiplier applied to the target. The default is deliberately not ``1.0``
            so the column is not a literal duplicate — a detector that only catches exact
            copies would pass this and still miss every real leak.

    Returns:
        A copy carrying :data:`LEAK_COLUMN`.
    """
    out = frame.copy()
    out[LEAK_COLUMN] = out[problem.target.name] * factor
    return out


def with_leak_declared(problem: MLProblem) -> MLProblem:
    """Return ``problem`` with :data:`LEAK_COLUMN` declared as a numeric feature.

    The detector iterates declared features, so the leaking column has to be on the spec
    for the frame to represent an adapter that shipped it by mistake.
    """
    from aegis_ml.contracts.spec import FeatureSpec

    leak = FeatureSpec(
        name=LEAK_COLUMN,
        dtype="numeric",
        description="Risk as settled after the fact — not available at prediction time.",
    )
    return problem.model_copy(update={"features": [*problem.features, leak]})


def shifted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Move every column in :data:`SHIFTED_COLUMNS` hard enough that drift must see it.

    Args:
        frame: A real generated frame.

    Returns:
        A copy with a scaled/offset numeric block and a collapsed categorical.
    """
    out = frame.copy()
    out["transit_hours"] = out["transit_hours"] * 3.0 + 50.0
    out["ambient_temp_c"] = out["ambient_temp_c"] + 25.0
    out["payload_kg"] = out["payload_kg"] * 4.0
    out["handoff_count"] = out["handoff_count"] + 7
    out["carrier_tier"] = "economy"
    return out


def as_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Render a frame as plain dicts, for tests that compare content rather than dtypes."""
    return frame.to_dict(orient="records")
