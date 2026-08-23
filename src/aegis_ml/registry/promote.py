"""Champion/challenger promotion — one atomic file replacement, and a way back.

The single fact that makes this package need **no changes inside** ``aegis/``:
``aegis.ml.get_model()`` loads one joblib file, and in the backend host that file is
``backend/.artifacts/ml_spine.joblib`` — exposed here as
:attr:`aegis_ml.settings.Settings.artifact_path`. So promotion is not an API call, a
registry transaction or a server-side flip. **Promotion is replacing that file**, and
demotion is putting the old one back.

Two disciplines make that safe rather than reckless:

1. **Nothing is overwritten before it is preserved.** The live artifact is copied into the
   outgoing champion's run directory first. If the incoming model turns out to be wrong at
   demo time, :func:`rollback` has a byte-identical copy of what was serving.
2. **The swap is atomic.** The challenger is copied to a temp file on the *same*
   filesystem and moved into place with :func:`os.replace`. A backend process calling
   ``get_model()`` mid-promotion opens either the whole old model or the whole new one —
   never a truncated joblib, which raises deep inside a pickle loader and looks like a
   corrupted install rather than a bad deploy.

The gate decides; this module only executes. A :class:`~aegis_ml.contracts.protocols.GateDecision`
with ``promoted=False`` raises :class:`~aegis_ml.contracts.errors.PromotionRejectedError`
carrying every reason, because "rejected" without the figures is indistinguishable from a
bug in the gate.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aegis_ml.contracts.errors import PromotionRejectedError
from aegis_ml.contracts.protocols import GateDecision, RegistryEntry
from aegis_ml.registry import store
from aegis_ml.settings import settings

__all__ = [
    "current_artifact_info",
    "promote",
    "rollback",
    "sha256_file",
]

_LOG = logging.getLogger(__name__)

_UNREGISTERED_BACKUP_DIR = "unregistered_artifacts"
"""Where a live artifact with no registry row behind it is preserved.

``python -m app.ml`` writes the serving artifact directly, so on a fresh host the very
first promotion overwrites a model this registry has never seen. Deleting it would be the
one unrecoverable step in an otherwise reversible flow.
"""


def sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a file, read in chunks.

    Streamed rather than slurped because a joblib spine with an ensemble inside it is
    tens of megabytes, and this runs on the request path of ``aegis-ml doctor``.

    Args:
        path: File to digest.
        chunk: Read size in bytes.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _iso(ts: float) -> str:
    """Render a POSIX timestamp as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _archive_live_artifact(previous: RegistryEntry | None) -> dict[str, Any]:
    """Preserve whatever is currently serving, before anything overwrites it.

    Three cases, all of which happen in practice:

    * **No live artifact.** First ever promotion on this host — nothing to preserve.
    * **A live artifact and a known champion.** Copied to ``runs/<champion>/model.joblib``
      unless that file already holds the same bytes. The digest comparison is what stops a
      re-promotion from clobbering a good archived copy with an identical one (harmless)
      or, worse, from skipping the archive because a file merely *exists* there (not
      harmless — it may be a different build).
    * **A live artifact and no champion row.** Preserved under
      ``registry_store/unregistered_artifacts/`` with a UTC stamp. This is the ``python -m
      app.ml`` case and it is the one an unconsidered implementation destroys.

    Args:
        previous: The outgoing champion entry, if the registry knows one.

    Returns:
        A record of what was preserved and where, for the promotion audit trail.
    """
    live = Path(settings.artifact_path)
    if not live.is_file():
        return {"archived": False, "reason": "no live artifact to preserve"}

    live_digest = sha256_file(live)
    if previous is not None:
        target = store.artifact(previous.run_id, "model.joblib")
        if target.is_file() and sha256_file(target) == live_digest:
            return {
                "archived": False,
                "reason": "outgoing champion already archived byte-identically",
                "path": str(target),
                "sha256": live_digest,
            }
        store.atomic_copy(live, target)
        _LOG.info("promote: archived live artifact into %s", target)
        return {"archived": True, "path": str(target), "sha256": live_digest}

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    target = (
        Path(store.registry_root())
        / _UNREGISTERED_BACKUP_DIR
        / f"{stamp}-{live_digest[:12]}.joblib"
    )
    if not target.is_file():
        store.atomic_copy(live, target)
    _LOG.warning(
        "promote: the live artifact %s has no registry entry behind it (it was most "
        "likely written by `python -m app.ml`). Preserved at %s before replacement.",
        live,
        target,
    )
    return {
        "archived": True,
        "path": str(target),
        "sha256": live_digest,
        "reason": "live artifact had no registry row; preserved outside runs/",
    }


def _install(source: Path) -> Path:
    """Copy ``source`` over the serving artifact atomically and verify the result.

    The post-copy digest check is not paranoia: a full disk truncates the temp file, and
    :func:`os.replace` will happily publish a truncated file. Verifying afterwards turns
    that into a loud failure at promotion time instead of a pickle error the next time the
    backend restarts — hours later, with nothing pointing at the deploy.

    Args:
        source: The challenger's ``model.joblib``.

    Returns:
        The serving artifact path.

    Raises:
        OSError: If the installed file does not match the source digest.
    """
    destination = Path(settings.artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_file(source)
    store.atomic_copy(source, destination)
    landed = sha256_file(destination)
    if landed != expected:
        raise OSError(
            f"promotion wrote {str(destination)!r} but its digest {landed} does not match "
            f"the source {expected} — the copy did not survive (a full disk is the usual "
            f"cause). The serving artifact is now in an unverified state: restore it with "
            f"aegis_ml.registry.promote.rollback() before serving traffic."
        )
    return destination


def _record_decision(
    entry: RegistryEntry, decision: GateDecision, *, forced: bool
) -> RegistryEntry:
    """Attach the decision that authorised this promotion to the entry, forcing included.

    A forced promotion is a promotion that bypassed the gate, and six weeks later nobody
    remembers which ones those were. It is written into three places that all travel with
    the run: ``gate.checks["forced"]``, a ``reasons`` line, and ``result.notes``. Whichever
    of the three a reader happens to open, the override is visible.
    """
    reasons = list(decision.reasons)
    checks = dict(decision.checks)
    notes = list(entry.result.notes)
    if forced:
        stamp = datetime.now(UTC).isoformat()
        line = (
            f"FORCED PROMOTION at {stamp}: the gate returned promoted="
            f"{decision.promoted} and was overridden explicitly (force=True)."
        )
        reasons.append(line)
        checks["forced"] = True
        notes.append(line)
    recorded = decision.model_copy(update={"reasons": reasons, "checks": checks})
    result = entry.result.model_copy(
        update={"notes": notes, "artifact_path": str(settings.artifact_path)}
    )
    return entry.model_copy(update={"gate": recorded, "result": result})


def promote(run_id: str, *, decision: GateDecision, force: bool = False) -> Path:
    """Install a challenger as the serving model, reversibly.

    Sequence, in the order that keeps every intermediate state recoverable:

    1. Refuse unless ``decision.promoted`` — or ``force=True``, which is recorded on the
       entry rather than merely allowed.
    2. Load the challenger entry and confirm its ``model.joblib`` exists.
    3. Preserve whatever is currently serving (see :func:`_archive_live_artifact`).
    4. Copy the challenger over :attr:`~aegis_ml.settings.Settings.artifact_path` via a
       temp file + :func:`os.replace`, then verify the digest.
    5. Mark the challenger ``production`` and the outgoing champion ``archived``.

    The stage flips come **last**. If step 4 fails, the registry still says the old model
    is in production — which is true, because it is.

    Args:
        run_id: The challenger run.
        decision: The gate's verdict, with the numbers behind it. Stored on the entry, so
            the model card and the registry quote the same figures.
        force: Promote despite a rejecting decision. Recorded permanently on the entry.

    Returns:
        The path now serving — :attr:`~aegis_ml.settings.Settings.artifact_path`.

    Raises:
        PromotionRejectedError: When the gate said no and ``force`` is False.
        FileNotFoundError: When the challenger run has no ``model.joblib`` to install.
    """
    if not decision.promoted and not force:
        reasons = list(decision.reasons) or [
            f"the gate returned promoted=False for run {run_id!r} and recorded no reason "
            f"— treat that as a gate bug, not as a passing model."
        ]
        raise PromotionRejectedError(reasons)

    entry = store.load_entry(run_id)
    source = store.artifact(run_id, "model.joblib")
    if not source.is_file():
        raise FileNotFoundError(
            f"run {run_id!r} has no model.joblib at {str(source)!r}, so there is nothing "
            f"to promote. Save the fitted spine with "
            f"aegis_ml.registry.store.save_run(entry, model=...) before promoting: the "
            f"registry promotes files it holds, never a model that only exists in memory."
        )

    previous = store.champion(entry.domain_id)
    if previous is not None and previous.run_id == run_id:
        _LOG.info(
            "promote: run %s is already the champion for %s; re-installing its artifact "
            "so the serving file and the registry cannot disagree",
            run_id,
            entry.domain_id,
        )
        previous = None

    archive = _archive_live_artifact(previous)
    installed = _install(source)

    promoted_entry = _record_decision(entry, decision, forced=force and not decision.promoted)
    promoted_entry = promoted_entry.model_copy(update={"stage": "production"})
    store.save_run(promoted_entry)

    if previous is not None:
        store.set_stage(previous.run_id, "archived")

    _LOG.info(
        "promote: %s is now serving at %s (previous champion: %s, archive: %s)",
        run_id,
        installed,
        previous.run_id if previous else "none",
        archive.get("path", "n/a"),
    )
    return installed


def rollback(domain_id: str) -> Path:
    """Restore the most recent archived production model for ``domain_id``.

    "Most recent archived" is the previous champion: :func:`promote` archives exactly the
    run it displaced, so walking archived entries newest-first and taking the first one
    with a stored ``model.joblib`` reverses the last promotion. Runs archived without a
    model file are skipped rather than failed on — an archived staging run that never
    served is not a rollback target.

    The current champion is demoted to ``archived`` and the restored run becomes
    ``production``, so a second call rolls back one more step rather than ping-ponging
    between two models.

    Args:
        domain_id: The adapter domain to roll back.

    Returns:
        The serving artifact path, now holding the restored model.

    Raises:
        FileNotFoundError: When no archived run for this domain has a stored model —
            named explicitly, because "rollback did nothing" must never be silent.
    """
    archived = store.list_runs(domain_id=domain_id, stage="archived")
    for candidate in archived:
        source = store.artifact(candidate.run_id, "model.joblib")
        if not source.is_file():
            continue

        current = store.champion(domain_id)
        if current is not None:
            _archive_live_artifact(current)
            store.set_stage(current.run_id, "archived")

        installed = _install(source)
        store.set_stage(candidate.run_id, "production")
        _LOG.warning(
            "rollback: restored run %s for domain %s over %s (displaced champion: %s)",
            candidate.run_id,
            domain_id,
            installed,
            current.run_id if current else "none",
        )
        return installed

    raise FileNotFoundError(
        f"no archived run with a stored model.joblib exists for domain {domain_id!r}, so "
        f"there is nothing to roll back to. Archived runs found: "
        f"{[e.run_id for e in archived] or 'none'}. The serving artifact is unchanged."
    )


def current_artifact_info() -> dict[str, Any]:
    """Describe the file that is actually serving right now.

    This answers the question every ML deploy eventually gets asked — *which model is
    live?* — from the file itself rather than from the registry's opinion of it. The
    registry can be wrong (someone re-ran ``python -m app.ml`` by hand); the bytes cannot.

    The run id is resolved by digest: the live file's SHA-256 is compared against each
    run's stored ``model.joblib``, champion first. A ``run_id`` of ``None`` with
    ``exists=True`` is the honest answer for an artifact this registry never wrote, and it
    is reported as ``matched_by="unmatched"`` rather than guessed from mtimes.

    Returns:
        A dict with ``path``, ``exists``, ``size_bytes``, ``modified_at`` (ISO-8601 UTC),
        ``sha256``, ``run_id``, ``domain_id``, ``stage`` and ``matched_by``.
    """
    path = Path(settings.artifact_path)
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": None,
        "modified_at": None,
        "sha256": None,
        "run_id": None,
        "domain_id": None,
        "stage": None,
        "matched_by": "missing",
    }
    if not path.is_file():
        info["hint"] = (
            "Nothing is serving: aegis.ml.get_model() will report the model as "
            "unavailable and the ML endpoints will answer 503. Train and promote, or run "
            "`python -m app.ml` in the backend to write a spine."
        )
        return info

    stat = path.stat()
    digest = sha256_file(path)
    info.update(
        {
            "size_bytes": stat.st_size,
            "modified_at": _iso(stat.st_mtime),
            "sha256": digest,
            "matched_by": "unmatched",
        }
    )

    entries = store.list_runs()
    ordered = [e for e in entries if e.stage == "production"] + [
        e for e in entries if e.stage != "production"
    ]
    for entry in ordered:
        candidate = store.artifact(entry.run_id, "model.joblib")
        if candidate.is_file() and sha256_file(candidate) == digest:
            info.update(
                {
                    "run_id": entry.run_id,
                    "domain_id": entry.domain_id,
                    "stage": entry.stage,
                    "matched_by": "sha256",
                }
            )
            return info

    info["hint"] = (
        "The live artifact matches no run in the registry. It was written outside this "
        "package (`python -m app.ml` does exactly that). Promoting will preserve it under "
        f"registry_store/{_UNREGISTERED_BACKUP_DIR}/ first."
    )
    return info
