"""The only place in this repository where a test double may live.

The user's requirement, quoted from ``scripts/audit_no_mocks.py``: *"for mock keep things
separate not in code so it can be really trusted."* ``scripts/audit_no_mocks.py`` enforces
the ``src/`` half of that; this package is the other half — every stand-in, every hidden
module, every hand-built result object the tests need is constructed here and imported by
name, so a reader can enumerate the doubles in one directory listing.

Nothing here is imported by ``src/aegis_ml``. Ever.

Modules:
    builders: Real pydantic result objects (``TrainResult``, ``GateDecision``,
        ``SliceMetric``, ``RegistryEntry``) assembled from literal numbers. Data, not
        doubles — a gate test needs a challenger and a champion, and fitting two models to
        produce two floats would test the estimator rather than the gate.
    frames: Deliberately-degenerate frames — a noise-free target, a pure-noise target, an
        injected leaking column, a distribution-shifted copy.
    doubles: The actual doubles. Currently one: a context manager that makes a named module
        unimportable, so the "optional dependency is missing" path can be exercised in a
        venv where the dependency is in fact installed.
"""

from __future__ import annotations

__all__ = ["builders", "doubles", "frames"]

from tests.fixtures import builders, doubles, frames
