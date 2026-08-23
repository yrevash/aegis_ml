"""``@flow`` / ``@task`` that become Prefect when Prefect is there, and nothing when it is not.

**A trained artifact must never depend on a server being up.**

That is the whole design. An orchestrator earns its place by giving you retries, scheduling
and a UI over work that already runs; it does not earn the right to stand between a
training run and a model file. Every pipeline in :mod:`aegis_ml.pipelines.flows` is an
ordinary Python function that returns a typed result and writes its own manifest — the
decorators here are additive. Delete Prefect from the environment, or take the server down
mid-demo, and the flows keep producing artifacts identical to the ones the orchestrated
path produces.

The failure mode this prevents is specific: an orchestration framework that owns the entry
point turns "the scheduler is unreachable" into "we cannot train", and turns a local
reproduction of a production run into an infrastructure exercise. Aegis's own rule — a
control that cannot run fails closed and *says so* — cuts the other way here, because
Prefect is not a control. It is a convenience, so its absence is allowed, and the
convenience is simply not applied.

Activation requires **both** ``settings.enable_prefect`` and an importable ``prefect``.
The flag alone is not enough (the package may be missing) and the import alone is not
enough (a developer with Prefect installed for another project must not silently start
registering flow runs). When the flag is on and the import fails, that is a configuration
error the operator asked for, so it raises through :func:`aegis_ml._require.require` naming
the install — the one case here that is not allowed to pass quietly.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

from aegis_ml._require import is_available
from aegis_ml.settings import settings

__all__ = ["flow", "prefect_active", "task"]

F = TypeVar("F", bound=Callable[..., Any])


def prefect_active() -> bool:
    """Return whether the decorators will delegate to Prefect in this process.

    Returns:
        ``True`` only when ``AEGIS_ML_ENABLE_PREFECT`` is set **and** ``prefect`` imports.

    Read by ``aegis-ml doctor`` so the answer to "are my flows orchestrated right now?" is a
    printed fact rather than an inference from two separate settings.
    """
    return bool(settings.enable_prefect) and is_available("prefect")


def _prefect_attr(name: str) -> Any:  # noqa: ANN401 - returns a Prefect decorator factory
    """Return ``prefect.flow`` or ``prefect.task``, raising with the install command.

    Args:
        name: ``"flow"`` or ``"task"``.

    Returns:
        The Prefect decorator factory.

    Raises:
        ImportError: When ``settings.enable_prefect`` is on but Prefect is not installed —
            the operator asked for orchestration and did not get it, which is exactly the
            kind of silent downgrade this package refuses.
    """
    from aegis_ml._require import require

    module = require("aegis-ml[mlops]", "prefect")
    return getattr(module, name)


def _decorator(kind: str, *d_args: Any, **d_kwargs: Any) -> Any:  # noqa: ANN401
    """Build the ``flow``/``task`` decorator for one call site.

    Handles both spellings — ``@flow`` (bare) and ``@flow(name="x")`` (called) — by
    detecting the bare form as "exactly one positional argument, and it is callable".
    Supporting only one spelling would make the shim's usage differ from Prefect's, and a
    decorator you have to write differently depending on whether the orchestrator is
    present defeats the point of the shim.

    Args:
        kind: ``"flow"`` or ``"task"``.
        *d_args: Positional decorator arguments (at most the bare function).
        **d_kwargs: Keyword decorator arguments, forwarded to Prefect when active and
            ignored when not.

    Returns:
        Either the decorated function (bare form) or a decorator (called form).
    """
    bare = len(d_args) == 1 and callable(d_args[0]) and not d_kwargs

    def wrap(fn: F) -> F:
        if not prefect_active():
            # The identity path. functools.wraps is applied even though the function is
            # returned unchanged, so that the *decorated* and *undecorated* objects are
            # indistinguishable to introspection — the CLI reads __doc__ off these.
            @functools.wraps(fn)
            def passthrough(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                return fn(*args, **kwargs)

            passthrough.__aegis_ml_orchestrated__ = False  # type: ignore[attr-defined]
            return cast(F, passthrough)

        factory = _prefect_attr(kind)
        kwargs = dict(d_kwargs)
        kwargs.setdefault("name", fn.__name__)
        decorated = factory(**kwargs)(fn)
        functools.update_wrapper(decorated, fn, updated=())
        decorated.__aegis_ml_orchestrated__ = True  # type: ignore[attr-defined]
        return cast(F, decorated)

    if bare:
        return wrap(cast(F, d_args[0]))
    return wrap


def flow(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - a decorator with two spellings
    """Mark a function as a pipeline entry point.

    Works as ``@flow`` and as ``@flow(name="train", retries=1)``. Keyword arguments are
    forwarded to ``prefect.flow`` when orchestration is active and ignored otherwise —
    ignoring them is correct because they configure the orchestrator, and with no
    orchestrator there is nothing to configure. The function's behaviour, return value and
    written artifacts are identical either way.

    Args:
        *args: The bare function, when used as ``@flow``.
        **kwargs: Prefect flow options, when used as ``@flow(...)``.

    Returns:
        The decorated function, or a decorator.
    """
    return _decorator("flow", *args, **kwargs)


def task(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - a decorator with two spellings
    """Mark a function as one step of a pipeline.

    Same contract as :func:`flow`: identical results with and without Prefect. Retries
    configured here apply only under orchestration; the retries that matter to correctness
    are declared on :class:`~aegis_ml.pipelines.manifest.StageSpec`, which works in both
    modes and records every attempt into the manifest.

    Args:
        *args: The bare function, when used as ``@task``.
        **kwargs: Prefect task options, when used as ``@task(...)``.

    Returns:
        The decorated function, or a decorator.
    """
    return _decorator("task", *args, **kwargs)
