"""Import an optional dependency, or raise naming the exact command that installs it.

Mirrors ``aegis.core.require``. Aegis's rule, quoted from ``AGENTS.md``: *"Optional deps go
through ``aegis.core.require(extra, module)``, which raises naming the exact ``pip install``.
Never ``except ImportError: pass``."*

The failure mode this prevents is specific and expensive: a bare ``try/except ImportError``
around an AutoML tier turns "AutoGluon is not installed" into "AutoGluon found nothing
better than the baseline", and the leaderboard the demo shows cannot tell them apart.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["is_available", "require"]


def require(extra: str, module: str) -> ModuleType:
    """Import ``module``, or raise :class:`ImportError` naming the install that fixes it.

    Args:
        extra: The install target, e.g. ``"aegis-ml[strong]"``.
        module: Dotted module path to import, e.g. ``"autogluon.tabular"``.

    Returns:
        The imported module.

    Raises:
        ImportError: When the module is absent, carrying the exact install command.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # noqa: TRY003 - the message IS the remedy
        raise ImportError(
            f"{module!r} is required here and is not importable. Install it with:\n"
            f"    uv pip install '{extra}'\n"
            f"Nothing here falls back to a weaker path on your behalf: a result produced "
            f"without {module!r} and a result produced with it must never be "
            f"indistinguishable."
        ) from exc


def is_available(module: str) -> bool:
    """Return whether ``module`` can be imported, without importing it eagerly.

    Used only for *capability reporting* — ``aegis-ml doctor`` and the leaderboard's
    ``tiers_skipped`` map. Never to silently choose a different code path: that decision
    is always recorded in the output the user reads.

    This is the one place in ``src/`` that catches an import failure without re-raising,
    and it is exempted explicitly rather than quietly: the caller's contract is "tell me
    whether this is here", so returning ``False`` IS the answer, not a degradation.
    """
    try:
        return importlib.util.find_spec(module) is not None
    # audit-ok: capability probe — False IS the answer here, not a silent downgrade.
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
