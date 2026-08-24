"""The doubles. One file, so they can be counted.

There is exactly one thing this suite cannot obtain from the real environment: the state of
a venv where an optional dependency is *absent*. Every dependency the package treats as
optional is installed in ``.venv``, so the "missing dependency" branch — which is the one
that must fail closed rather than degrade quietly — has no way to be reached without
arranging for an import to fail.

:func:`hidden_module` arranges exactly that, by inserting ``None`` into ``sys.modules`` for
the duration of a ``with`` block. Python's import machinery treats a ``None`` entry as a
definitive "not importable" and raises ``ImportError``, which is precisely the condition on
a machine that never installed the extra.

Nothing else in this suite is a double. Estimators are fitted, frames are generated,
registries are written to disk.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator

__all__ = ["hidden_module"]


@contextlib.contextmanager
def hidden_module(*names: str) -> Iterator[None]:
    """Make ``names`` (and their submodules) unimportable inside the block.

    Args:
        *names: Top-level module names, e.g. ``"lightgbm"``, ``"autogluon"``.

    Yields:
        Nothing; the effect is on ``sys.modules`` for the duration.
    """
    saved: dict[str, object] = {}
    prefixes = tuple(names)
    for key in list(sys.modules):
        if key in prefixes or key.startswith(tuple(f"{n}." for n in prefixes)):
            saved[key] = sys.modules.pop(key)
    for name in prefixes:
        sys.modules[name] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        for name in prefixes:
            sys.modules.pop(name, None)
        sys.modules.update(saved)  # type: ignore[arg-type]


BANNED_DOUBLE_TOKENS: tuple[str, ...] = (
    "unittest." + "mock",
    "Magic" + "Mock",
    "monkeypatch.setattr(aegis_ml",
)
"""Tokens that must not appear in a test module outside this package.

Assembled from fragments so that ``tests/test_meta.py`` — which greps every test file for
them — does not trip over its own list. This file is exempt from that grep by living in
``tests/fixtures/``, which is the entire point of the rule.

The third entry bans patching ``aegis_ml`` internals: a test that reaches into ``src/`` to
make itself pass is a test that no longer measures the shipped code. Patching *settings*
(``monkeypatch.setattr(settings, ...)``) is fine and is what ``conftest`` does — that is
configuration, not a double.
"""
