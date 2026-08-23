"""The model registry: a directory tree, a promotion gate, and two optional mirrors.

Decision D3, in one sentence: **the filesystem is the source of truth, and promotion is
replacing the one joblib file ``aegis.ml.get_model()`` already loads.** Everything in this
subpackage follows from that.

* :mod:`~aegis_ml.registry.store` — run directories, atomic writes, a rebuildable index.
* :mod:`~aegis_ml.registry.promote` — champion/challenger swap, rollback, and an honest
  answer to "which model is actually serving right now?".
* :mod:`~aegis_ml.registry.mlflow_mirror` — optional MLflow copy for the UI and lineage.
* :mod:`~aegis_ml.registry.db` — optional SQLAlchemy tables filling the relational gap
  Aegis has for ML, following the ``eval_results`` precedent exactly.

Neither mirror is on the critical path. Both are switched off by default, and with both
off the registry still trains, gates, promotes, rolls back and serves — which is what
makes a demo survivable when a server is not.

Importing this package costs pydantic and the standard library. joblib, SQLAlchemy and
mlflow are all imported inside the functions that need them.

**One naming rule, applied deliberately.** ``promote`` is a *module* here, not a
re-exported function, because the two cannot both own the name: re-exporting the function
would shadow ``aegis_ml.registry.promote`` and break ``registry.promote.rollback(...)``
with an ``AttributeError`` that reads like a missing feature. So:

* ``from aegis_ml.registry.promote import promote`` — the function, by its full path.
* ``aegis_ml.promote`` — the same function, lazily re-exported at the package root, which
  is where the short spelling belongs.

``rollback``, ``current_artifact_info`` and the store helpers have no such clash and are
re-exported here for convenience.
"""

from __future__ import annotations

from aegis_ml.registry.promote import current_artifact_info, rollback, sha256_file
from aegis_ml.registry.store import (
    STANDARD_ARTIFACTS,
    artifact,
    champion,
    index_path,
    list_runs,
    load_entry,
    new_run_id,
    registry_root,
    reindex,
    run_dir,
    save_run,
    set_stage,
)

__all__ = [
    "STANDARD_ARTIFACTS",
    "artifact",
    "champion",
    "current_artifact_info",
    "index_path",
    "list_runs",
    "load_entry",
    "new_run_id",
    "registry_root",
    "reindex",
    "rollback",
    "run_dir",
    "save_run",
    "set_stage",
    "sha256_file",
]
