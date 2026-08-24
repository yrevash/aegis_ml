"""Per-run visual reports: one directory of figures per run, built from that run's artifacts.

Every run in ``registry_store/runs/`` gets ``visuals/``, and ``visuals/index.html`` is the
single file a human opens to decide whether to trust the model. The rest of the package is
what makes that page defensible:

* :mod:`aegis_ml.report.theme` — one palette, one set of rcParams, one stylesheet, so ten
  figures read as one report instead of ten.
* :mod:`aegis_ml.report.plots` — the figures. Each takes measured data and returns the
  numbers it drew; none of them reads a file or invents a series.
* :mod:`aegis_ml.report.bundle` — loads the run's artifacts, recovers and *verifies* the
  held-out split, renders what the artifacts support, and records a reason for everything
  they do not.
* :mod:`aegis_ml.report.index` — inlines it all into one self-contained page.

The rule the whole package is built around: **an input that is missing produces a recorded
omission, never a drawn zero.** A run whose SHAP stage failed and a run whose features
genuinely carry no attribution must not produce the same picture.

Entry points::

    from aegis_ml.report import build_bundle
    build_bundle("cold_chain_logistics-20260823T213346076-e44917")

or, for a run that is already registered, ``aegis-ml visuals <run_id>``.
"""

from __future__ import annotations

from aegis_ml.report.bundle import (
    CAPTIONS,
    PLOT_FILES,
    VISUALS_DIRNAME,
    MissingInput,
    build_bundle,
    bundle_dir,
    load_assets,
    recover_split,
)
from aegis_ml.report.index import render_index, write_index
from aegis_ml.report.theme import PALETTE, TIER_COLOURS

__all__ = [
    "CAPTIONS",
    "PALETTE",
    "PLOT_FILES",
    "TIER_COLOURS",
    "VISUALS_DIRNAME",
    "MissingInput",
    "build_bundle",
    "bundle_dir",
    "load_assets",
    "recover_split",
    "render_index",
    "write_index",
]
