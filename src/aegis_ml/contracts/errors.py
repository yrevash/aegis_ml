"""Typed refusals for the ML adapter factory — never a plausible-looking number.

Aegis's single most important rule, quoted from ``AGENTS.md``:

    "No silent fallbacks. A control that cannot run fails closed and says so — it never
    degrades quietly into something that looks like it worked. … a silent downgrade is
    worse than an outage because nobody goes looking."

Every error here exists because the alternative was a number a human would have believed.
Each carries the remedy in its message, following ``aegis.core.require``'s convention of
naming the exact command that fixes the problem.
"""

from __future__ import annotations

__all__ = [
    "AegisMLError",
    "AutoMLTierUnavailableError",
    "DriftThresholdExceededError",
    "InsufficientLabelsError",
    "LabelNotLearnableError",
    "PromotionRejectedError",
    "RecipeNotPortableError",
    "TargetLeakageError",
    "TrainerVenvMissingError",
]


class AegisMLError(RuntimeError):
    """Base for every refusal this package raises."""


class AutoMLTierUnavailableError(AegisMLError):
    """A requested AutoML tier is not installed in the active interpreter.

    Raised instead of silently falling through to a weaker tier: a caller who asked for
    AutoGluon and got a plain XGBoost baseline must be able to tell that apart from a
    caller who got what they asked for, because the leaderboard they publish says which
    one ran.
    """

    def __init__(self, tier: str, module: str, extra: str = "aegis-ml[strong]") -> None:
        """Name the tier, the import that failed, and the install that fixes it."""
        super().__init__(
            f"AutoML tier {tier!r} needs {module!r}, which is not importable here. "
            f"Install it into the trainer venv: `uv pip install '{extra}'` — or run the "
            f"search through the subprocess bridge (`aegis_ml.automl.runner`), which is "
            f"what the two-venv split exists for."
        )
        self.tier = tier
        self.module = module


class TrainerVenvMissingError(AegisMLError):
    """The isolated trainer virtualenv does not exist or has no interpreter."""

    def __init__(self, path: str) -> None:
        """Name the missing venv and the command that creates it."""
        super().__init__(
            f"Trainer venv not found at {path!r}. Create it with:\n"
            f"  uv venv {path} --python 3.11\n"
            f"  uv pip install --python {path} -e '.[strong,serve]'\n"
            f"The heavy tiers live there on purpose: AutoGluon/TabPFN/torch will not "
            f"resolve under the backend's pandas<2.4 / numpy<2.5 / numba==0.67.0 caps."
        )
        self.path = path


class RecipeNotPortableError(AegisMLError):
    """A search result cannot be re-fitted in the serving venv.

    The two-venv split is only sound because the winning configuration crosses as JSON
    and is re-fitted by the Aegis spine. A recipe naming an estimator the serving venv
    cannot construct breaks that guarantee, so it is refused rather than half-applied.
    """

    def __init__(self, kind: str, reason: str) -> None:
        """Name the estimator and why it cannot cross the venv boundary."""
        super().__init__(
            f"Recipe member {kind!r} is not portable into the serving venv: {reason}. "
            f"Report its leaderboard score as the accuracy ceiling and export it to ONNX "
            f"for a side-by-side predictor — do not promote it as the spine."
        )
        self.kind = kind


class LabelNotLearnableError(AegisMLError):
    """The generated target carries no recoverable signal from the features.

    This is the trap ``SKILL.md`` names and nothing in the platform enforces: "the
    generator must sample labels *around your latent function*. If it does not, the target
    is noise, the model finds nothing, and the conformal interval is honestly enormous."

    Aegis's 14 conformance checks do not cover it — the only native signal is
    ``distinct=False`` on the last line of ``python -m app.ml``, which is read minutes
    before a demo. This error fires in seconds instead.
    """

    def __init__(self, metric: str, value: float, floor: float) -> None:
        """Report the measured score against the floor it had to clear."""
        super().__init__(
            f"Label is not learnable: {metric}={value:.4f} on a held-out split, below the "
            f"floor of {floor:.4f}. The generator is drawing the target independently of "
            f"the features. Fix the generator so the label is `latent_fn(features) + noise` "
            f"— see aegis_ml.data.latent. Training past this point produces a model that "
            f"has learned nothing and a conformal interval that is honestly enormous."
        )
        self.metric = metric
        self.value = value
        self.floor = floor


class TargetLeakageError(AegisMLError):
    """A feature predicts the target too well to be legitimate."""

    def __init__(self, feature: str, score: float, threshold: float) -> None:
        """Name the leaking feature and its single-feature score."""
        super().__init__(
            f"Feature {feature!r} alone scores {score:.4f} against the target (threshold "
            f"{threshold:.4f}) — that is leakage, not signal. Drop it from FEATURES, or "
            f"declare it intentional via the `allow_leakage` config if it is genuinely "
            f"available at prediction time."
        )
        self.feature = feature


class InsufficientLabelsError(AegisMLError):
    """Not enough ground truth to compute the requested measurement."""

    def __init__(self, have: int, need: int, what: str) -> None:
        """Report what was available against what the measurement required."""
        super().__init__(
            f"{what} needs at least {need} labelled rows; {have} are available. Use "
            f"aegis_ml.monitor.perf (NannyML CBPE/DLE) to ESTIMATE performance without "
            f"labels instead — it says it is an estimate, which an under-powered "
            f"measurement does not."
        )
        self.have = have
        self.need = need


class PromotionRejectedError(AegisMLError):
    """The challenger did not clear the promotion gate.

    Carries every failed criterion with its measured number, because "rejected" without
    the figure is indistinguishable from a bug in the gate.
    """

    def __init__(self, reasons: list[str]) -> None:
        """List every criterion the challenger failed, each with its number."""
        joined = "\n  - ".join(reasons)
        super().__init__(f"Promotion rejected; the champion is unchanged:\n  - {joined}")
        self.reasons = reasons


class DriftThresholdExceededError(AegisMLError):
    """Measured drift exceeded the configured block threshold."""

    def __init__(self, metric: str, value: float, threshold: float) -> None:
        """Report the drift metric against its threshold."""
        super().__init__(
            f"Drift gate: {metric}={value:.4f} exceeds the block threshold {threshold:.4f}. "
            f"The serving model is NOT withdrawn — Aegis serves the model it has and flags "
            f"it. This blocks PROMOTION of anything calibrated on the drifted reference."
        )
        self.metric = metric
        self.value = value
        self.threshold = threshold
