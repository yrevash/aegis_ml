"""The four AutoML search tiers, and an availability probe that never imports them.

Why a probe at all. The tiers do not live in one interpreter. ``baseline`` and ``flaml``
are pure-Python and resolve inside the backend's ``pandas<2.4`` / ``numpy<2.5`` /
``numba==0.67.0`` caps; ``autogluon`` and ``tabpfn`` pull torch and only resolve in the
isolated trainer venv (decision D1). So *the same call* to :func:`available_tiers` gives a
different answer in the two processes, and the answer must be reported rather than acted
on quietly — hence :func:`tier_status`, whose strings land verbatim in
``Leaderboard.tiers_skipped``.

Why the probe uses ``importlib.util.find_spec`` and not ``import``. Importing
``autogluon.tabular`` costs seconds and drags torch into the address space; importing
``tabpfn`` may reach for model weights. A capability report must be cheap enough that
``aegis-ml doctor`` can run it before anything expensive starts, so this module imports
nothing heavier than :mod:`importlib` at module scope.

What breaks otherwise. The failure mode this module exists to prevent is the one named in
``aegis_ml._require``: a bare ``try/except ImportError`` around a tier turns "AutoGluon is
not installed" into "AutoGluon found nothing better than the baseline", and the leaderboard
shown in the demo cannot tell those apart. Every tier that does not run leaves a reason
string behind, always.
"""

from __future__ import annotations

from aegis_ml._require import is_available
from aegis_ml.contracts.errors import AutoMLTierUnavailableError
from aegis_ml.contracts.protocols import TierName
from aegis_ml.settings import settings

__all__ = [
    "TABPFN_LICENSE_NOTICE",
    "TIER_DESCRIPTIONS",
    "TIER_EXTRAS",
    "TIER_ORDER",
    "TIER_REQUIREMENTS",
    "available_tiers",
    "has_autotabpfn",
    "require_tier",
    "resolve_tiers",
    "tier_enabled",
    "tier_notes",
    "tier_status",
    "unavailable_reason",
]

TIER_ORDER: tuple[TierName, ...] = ("baseline", "flaml", "autogluon", "tabpfn")
"""Weakest-but-always-present to strongest.

Order is load-bearing twice: :func:`resolve_tiers` runs tiers in it so a time-budgeted
search spends its first seconds on the tier that is guaranteed to produce *something*
portable, and ties on the leaderboard are broken towards the earlier (cheaper) tier.
"""

TIER_REQUIREMENTS: dict[TierName, tuple[str, ...]] = {
    "baseline": ("sklearn",),
    "flaml": ("flaml",),
    "autogluon": ("autogluon.tabular",),
    "tabpfn": ("tabpfn",),
}
"""Tier → the modules that must be importable for it to run.

``baseline`` names ``sklearn`` and *not* ``xgboost``, which is deliberate. XGBoost ships in
the ``[serve]`` extra and in the Aegis backend venv, so it is normally there — but this
tier is the *only* one guaranteed to yield a portable recipe, and making that guarantee
depend on a second wheel is how a search ends up with nothing to return. Instead XGBoost is
checked per-member by :func:`aegis_ml.automl.recipe.is_portable_kind`: its absence removes
candidates from the tier, each with a recorded reason, rather than removing the tier.
"""

TIER_EXTRAS: dict[TierName, str] = {
    "baseline": "aegis-ml[serve]",
    "flaml": "aegis-ml[serve]",
    "autogluon": "aegis-ml[strong]",
    "tabpfn": "aegis-ml[strong]",
}
"""Tier → the install target that provides it, quoted verbatim in every error message."""

TIER_DESCRIPTIONS: dict[TierName, str] = {
    "baseline": (
        "sklearn + xgboost soft-voting, the same members aegis.ml.model builds, plus a "
        "linear reference floor. Always portable, and the explicit floor the other tiers "
        "must beat to justify themselves."
    ),
    "flaml": (
        "FLAML cost-frugal search under a wall-clock budget. Pure Python, so it runs in "
        "the serving venv and its winner is re-fittable there without a subprocess."
    ),
    "autogluon": (
        "AutoGluon TabularPredictor at preset 'best_quality' — multi-layer stacked "
        "ensembles. Its stack cannot be re-fitted in the serving venv, so it is reported "
        "as an accuracy ceiling rather than promoted as the spine."
    ),
    "tabpfn": (
        "TabPFN-2.5 tabular foundation model (plus AutoTabPFN when tabpfn_extensions is "
        "installed). Strongest at the 1k-10k row scale this factory generates, and not "
        "portable: the prediction IS the pretrained transformer."
    ),
}
"""Human-readable one-liners; ``aegis-ml doctor`` and the model card print these."""

TABPFN_LICENSE_NOTICE: str = (
    "TabPFN-2.5 weights are distributed under the Prior Labs License: research and "
    "evaluation use are permitted, commercial and production use are NOT. This tier's "
    "score is reported as an accuracy ceiling for evaluation purposes only. Set "
    "AEGIS_ML_ENABLE_TABPFN=0 to switch the tier off entirely."
)
"""The licence notice every artefact TabPFN touches must carry.

It is a module constant rather than a docstring because it has to be *copied into data* —
``Recipe.notes``, ``Candidate.detail``, the model card — not just read by a developer. A
licence condition that only exists in a docstring travels nowhere.
"""

_TIER_SWITCHES: dict[TierName, str] = {
    "flaml": "enable_flaml",
    "autogluon": "enable_autogluon",
    "tabpfn": "enable_tabpfn",
}
"""Tier → the ``settings`` flag that disables it. ``baseline`` has no switch on purpose:
something portable must always be able to run, or there is no recipe to hand back."""


def tier_enabled(tier: TierName) -> bool:
    """Return whether ``tier`` is switched on in settings.

    A disabled tier is a *policy* decision (``AEGIS_ML_ENABLE_TABPFN=0``), distinct from an
    uninstalled one, and the two produce different reason strings so the reader of a
    leaderboard can tell "we chose not to" from "we could not".

    Args:
        tier: The tier to check.

    Returns:
        ``True`` unless an ``AEGIS_ML_ENABLE_*`` flag turns this tier off.
    """
    switch = _TIER_SWITCHES.get(tier)
    if switch is None:
        return True
    return bool(getattr(settings, switch))


def _missing_modules(tier: TierName) -> list[str]:
    """Return the tier's required modules that are not importable in this interpreter."""
    return [m for m in TIER_REQUIREMENTS[tier] if not is_available(m)]


def unavailable_reason(tier: TierName) -> str | None:
    """Return why ``tier`` cannot run here, or ``None`` when it can.

    The string is written for a human reading ``Leaderboard.tiers_skipped`` in a model
    card, so it names the missing import *and* the command that supplies it. "Tier
    unavailable" with no remedy sends the reader to the source; this does not.

    Args:
        tier: The tier to check.

    Returns:
        A reason string, or ``None`` if the tier is enabled and importable.
    """
    if tier not in TIER_REQUIREMENTS:
        return f"unknown tier {tier!r}; known tiers are {list(TIER_ORDER)}"
    if not tier_enabled(tier):
        switch = _TIER_SWITCHES[tier]
        return (
            f"disabled by settings.{switch} (AEGIS_ML_{switch.upper()}=0) — this is a "
            f"policy choice, not a missing dependency"
        )
    missing = _missing_modules(tier)
    if missing:
        return (
            f"not importable in this interpreter: {', '.join(missing)}. Install with "
            f"`uv pip install '{TIER_EXTRAS[tier]}'`, or run the search through "
            f"aegis_ml.automl.runner, which executes it inside the trainer venv."
        )
    if tier == "tabpfn":
        return _tabpfn_weights_reason()
    return None


def _tabpfn_weights_reason() -> str | None:
    """Return why TabPFN cannot fit here despite importing cleanly, or ``None``.

    Importability is not availability for this tier, and the gap is not academic. From
    TabPFN 8.x the package imports fine with no weights on disk, then raises
    ``TabPFNLicenseError`` **inside** ``.fit()`` — after the search has already spent its
    budget on the earlier tiers and a user is watching a progress line. Prior Labs gates
    the weight download behind a one-time licence acceptance plus an API token.

    Probing at capability-report time turns that into a line in ``tiers_skipped`` and one
    in ``aegis-ml doctor``, which is the difference between "TabPFN is not set up on this
    machine" and a traceback mid-demo.

    Two things count as ready, and either is enough:

    * ``TABPFN_TOKEN`` is set, so the weights can be fetched on first use; or
    * a checkpoint is already in the local cache, so no network is needed at all — which
      is the state a machine is in after one successful run, and the state a hackathon
      laptop should be put in deliberately, in advance.

    Returns:
        A reason string naming the exact remedy, or ``None`` when the tier can fit.
    """
    import os

    if os.environ.get("TABPFN_TOKEN"):
        return None
    try:
        from tabpfn.model_loading import get_cache_dir

        cache = get_cache_dir()
        if cache.exists() and any(cache.rglob("*.ckpt")):
            return None
    # audit-ok: the probe's own failure IS a reason string below, never a silent pass.
    except Exception:  # noqa: BLE001 - any failure here means "cannot confirm weights"
        pass
    return (
        "importable, but no model weights are available and TABPFN_TOKEN is unset, so "
        ".fit() would raise TabPFNLicenseError mid-search. One-time setup: register at "
        "https://ux.priorlabs.ai, accept the licence on the Licenses tab, copy the API key "
        "from the Account page, then `export TABPFN_TOKEN=...` and run once to cache the "
        "weights locally. Do this BEFORE the day — it needs a browser and a network. "
        + TABPFN_LICENSE_NOTICE
    )


def available_tiers() -> dict[TierName, bool]:
    """Return, for every tier, whether it can run in this interpreter right now.

    Returns:
        Tier → availability, in :data:`TIER_ORDER`. Every tier appears, including the
        unavailable ones: a caller iterating this map cannot accidentally omit a tier from
        a report by iterating only what was present.
    """
    return {tier: unavailable_reason(tier) is None for tier in TIER_ORDER}


def tier_status() -> dict[TierName, str]:
    """Return a one-line status for every tier — ``"available"`` or the reason it is not.

    This is the function ``aegis-ml doctor`` prints and the one that populates
    ``Leaderboard.tiers_skipped`` (via :func:`resolve_tiers`).

    Returns:
        Tier → ``"available"`` or a remedy-carrying reason string.
    """
    return {tier: (unavailable_reason(tier) or "available") for tier in TIER_ORDER}


def has_autotabpfn() -> bool:
    """Return whether ``tabpfn_extensions``' post-hoc ensembling is importable.

    AutoTabPFN (the post-hoc ensemble over TabPFN base models) is a strict improvement on
    a bare TabPFN fit but lives in a *separate* distribution. Its absence downgrades the
    tabpfn tier rather than disabling it, and the downgrade is recorded on the candidate
    so nobody reads a plain-TabPFN score as an AutoTabPFN one.

    Returns:
        ``True`` if ``tabpfn_extensions`` can be imported.
    """
    return is_available("tabpfn_extensions")


def require_tier(tier: TierName) -> None:
    """Raise unless ``tier`` can run in this interpreter.

    Used where a caller has *explicitly asked* for one tier (``aegis-ml train --tier
    tabpfn``). Asking for AutoGluon and silently receiving an XGBoost baseline is exactly
    the confusion :class:`~aegis_ml.contracts.errors.AutoMLTierUnavailableError` exists to
    prevent — the leaderboard that gets published says which tier ran.

    Args:
        tier: The tier the caller demanded.

    Raises:
        AutoMLTierUnavailableError: If the tier is disabled or not importable.
    """
    reason = unavailable_reason(tier)
    if reason is None:
        return
    missing = _missing_modules(tier) if tier in TIER_REQUIREMENTS else []
    module = missing[0] if missing else TIER_REQUIREMENTS.get(tier, (tier,))[0]
    raise AutoMLTierUnavailableError(tier, module, TIER_EXTRAS.get(tier, "aegis-ml[strong]"))


def tier_notes(tier: TierName) -> list[str]:
    """Return notes that must travel with any result this tier produced.

    Currently only TabPFN carries one, and it is a licence condition rather than a
    footnote: :data:`TABPFN_LICENSE_NOTICE` must reach ``Recipe.notes`` and the model card
    of every run the tier touched, because the permission it grants is conditional.

    Args:
        tier: The tier that produced a result.

    Returns:
        Zero or more note strings to copy into the result.
    """
    return [TABPFN_LICENSE_NOTICE] if tier == "tabpfn" else []


def resolve_tiers(
    requested: list[TierName] | tuple[TierName, ...] | None = None,
) -> tuple[list[TierName], dict[str, str]]:
    """Split the requested tiers into the ones that will run and the ones that will not.

    This is the single place where "which tiers ran" is decided, so it is also the single
    place that produces the ``tiers_skipped`` map. Keeping the two halves of that decision
    in one return value makes it impossible to drop a tier from the run without also
    writing down why.

    Args:
        requested: Tiers the caller asked for, or ``None`` for all of
            :data:`TIER_ORDER`. Unknown names are not dropped — they are skipped *with a
            reason*, which is how a typo in ``--tier`` becomes visible.

    Returns:
        ``(to_run, skipped)`` where ``to_run`` is in :data:`TIER_ORDER` and ``skipped``
        maps tier name → reason, ready for ``Leaderboard.tiers_skipped``.
    """
    wanted: list[str] = list(requested) if requested is not None else list(TIER_ORDER)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in wanted:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    to_run: list[TierName] = []
    skipped: dict[str, str] = {}
    for name in ordered:
        if name not in TIER_REQUIREMENTS:
            skipped[name] = f"unknown tier {name!r}; known tiers are {list(TIER_ORDER)}"
            continue
        tier: TierName = name  # type: ignore[assignment]
        reason = unavailable_reason(tier)
        if reason is None:
            to_run.append(tier)
        else:
            skipped[tier] = reason

    for tier in TIER_ORDER:
        if tier not in seen:
            skipped[tier] = "not requested by the caller"

    to_run.sort(key=TIER_ORDER.index)
    return to_run, skipped
