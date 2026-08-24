"""The ``aegis-ml dashboard`` surface: one hub page in front of two premade UIs.

Three ideas, in the order they matter:

1. **Do not rebuild what already exists.** MLflow's run comparison and Optuna Dashboard's
   parallel-coordinate plot are best-in-class and already installed. This package writes
   both of their stores as a side effect of doing its actual job — the Optuna study
   database exists so a search is resumable, not so a dashboard can read it — so pointing
   them at that data costs a subprocess each, not a frontend.
2. **Own the thing nobody else can show.** No off-the-shelf UI knows what a *promotion
   gate* is, or that a conformal interval has a requested coverage and a measured one that
   must be read next to each other. That verdict is the hub's job, and it is the only page
   here that is hand-built.
3. **Never show a number that was not measured.** Every figure the hub renders is read from
   an artifact in ``registry_store/``. Where the artifact is missing the page says which
   one and which command writes it. There is no default, no rounded-up estimate and no
   sample row anywhere in this package.

Submodules:

* :mod:`~aegis_ml.dashboard.theme` — the design tokens and the stylesheet.
* :mod:`~aegis_ml.dashboard.hub` — reads the registry and renders the page.
* :mod:`~aegis_ml.dashboard.services` — starts, probes and stops MLflow and Optuna Dashboard.
* :mod:`~aegis_ml.dashboard.server` — the loopback HTTP server for the page and the
  run directory's static artifacts.

Nothing here is imported by the training, gating, promotion or serving path. The dashboard
is a viewer; deleting this package changes nothing about what the pipelines produce.
"""

from __future__ import annotations

from aegis_ml.dashboard import hub, server, services, theme

__all__ = ["hub", "server", "services", "theme"]
