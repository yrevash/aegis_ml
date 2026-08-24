"""The filesystem model registry — the source of truth, and deliberately boring.

Decision D3 of ``finalplan.md``: *the registry is filesystem-first*. Every fact about a
run lives in a directory a human can ``ls``, and every derived index can be thrown away
and rebuilt from those directories. Nothing here needs a server, a daemon, a migration or
a network round-trip, which is the whole reason a demo cannot lose its models when MLflow
or Postgres is down.

Layout under :attr:`aegis_ml.settings.Settings.registry_dir`::

    registry_store/
      index.json                     # list[RegistryEntry], newest first — DERIVED
      runs/<run_id>/
        model.joblib                 # the fitted spine; promotion copies THIS file
        entry.json                   # the full RegistryEntry — AUTHORITATIVE
        recipe.json                  # the portable AutoML recipe
        leaderboard.json             # every candidate, winners and losers
        card.json  card.md  card.html
        shap.html
        reference.parquet            # the drift reference frame
        metrics.json

``index.json`` is a cache and is marked as such: :func:`reindex` rebuilds it by walking
``runs/*/entry.json``. If the two ever disagree, the run directory wins. That asymmetry is
what makes a half-finished write survivable — a run directory with an ``entry.json`` is a
complete registry row even if the process died before the index was updated.

Every write goes through :func:`atomic_write_bytes` or :func:`atomic_copy`: content is
written to a uniquely-named ``.tmp`` sibling, ``fsync``-ed, then moved into place with
:func:`os.replace`. A reader therefore sees either the whole previous version or the whole
new one — never a truncated joblib, which loads as a corrupt model rather than an error.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from aegis_ml.contracts.protocols import RegistryEntry
from aegis_ml.settings import settings

__all__ = [
    "STANDARD_ARTIFACTS",
    "Stage",
    "artifact",
    "atomic_copy",
    "atomic_write_bytes",
    "champion",
    "index_path",
    "list_runs",
    "load_entry",
    "new_run_id",
    "registry_root",
    "reindex",
    "run_dir",
    "runs_root",
    "save_run",
    "set_stage",
]

_LOG = logging.getLogger(__name__)

Stage = Literal["staging", "production", "archived"]
"""The three lifecycle states, matching :class:`RegistryEntry.stage` exactly."""

STANDARD_ARTIFACTS: tuple[str, ...] = (
    "model.joblib",
    "entry.json",
    "recipe.json",
    "leaderboard.json",
    "card.json",
    "card.md",
    "card.html",
    "shap.html",
    "reference.parquet",
    "metrics.json",
)
"""The documented file names inside a run directory.

Not enforced — :func:`save_run` will write any name asked of it — but every tool in this
package reads these, so a run that carries them is a run every other stage understands.
"""

_LOCK_TIMEOUT_S = 15.0
"""How long :func:`_index_lock` waits before refusing. It refuses rather than proceeding
unlocked: two concurrent index writers produce an index missing one of the runs, and the
run directories would then be the only record — recoverable, but silently wrong until
someone runs :func:`reindex`."""

_LOCK_STALE_S = 120.0
"""A lock file older than this is assumed to belong to a process that died mid-write.
Without stale-breaking, one crashed trainer would wedge the registry until a human
deleted a dotfile they have never heard of."""


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
def registry_root() -> Path:
    """Return the registry root, creating it if it does not exist.

    Read from :data:`aegis_ml.settings.settings` on every call rather than captured at
    import time, so a test that repoints ``AEGIS_ML_REGISTRY_DIR`` (or mutates the
    settings singleton) is actually honoured instead of writing into the developer's
    real registry.

    Returns:
        The directory holding ``index.json`` and ``runs/``.
    """
    root = Path(settings.registry_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_root() -> Path:
    """Return ``<registry_dir>/runs``, creating it if needed."""
    root = registry_root() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def index_path() -> Path:
    """Return the path of the derived ``index.json`` cache."""
    return registry_root() / "index.json"


def run_dir(run_id: str) -> Path:
    """Return the directory for ``run_id``, creating it if needed.

    Args:
        run_id: The run identifier produced by :func:`new_run_id`.

    Returns:
        ``<registry_dir>/runs/<run_id>``.

    Raises:
        ValueError: If ``run_id`` contains a path separator or ``..``. A run id reaches
            this function from JSON that crossed a venv boundary, so it is treated as
            untrusted input: ``../../etc`` must not become a write target.
    """
    _validate_run_id(run_id)
    directory = runs_root() / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def artifact(run_id: str, name: str) -> Path:
    """Return the path of one artifact inside a run directory.

    The file is not required to exist — callers ask for the path both to read it and to
    decide whether to write it.

    Args:
        run_id: The run identifier.
        name: File name, e.g. ``"model.joblib"``. See :data:`STANDARD_ARTIFACTS`.

    Returns:
        ``<registry_dir>/runs/<run_id>/<name>``.

    Raises:
        ValueError: If ``name`` is absolute or tries to escape the run directory.
    """
    if not name or name != Path(name).name:
        raise ValueError(
            f"artifact name {name!r} must be a bare file name — a run directory is a flat "
            f"namespace so that a run can be copied, tarred or rsynced as one unit."
        )
    return run_dir(run_id) / name


#: Characters a run id may contain. Deliberately narrower than "a legal file name":
#: :func:`new_run_id` only ever emits ``<domain>-<timestamp>-<hex>``, so anything outside
#: this set is a caller inventing an id, and the cost of allowing it is paid later by
#: whoever types ``rm -rf runs/<id>``.
_RUN_ID_ALLOWED = re.compile(r"\A[A-Za-z0-9._-]+\Z")


def _validate_run_id(run_id: str) -> None:
    """Reject a run id that could escape ``runs/`` or collide with a shell glob.

    Two separate checks, because they fail differently.

    The **path** check (``run_id != Path(run_id).name``) is the security-relevant one: it
    stops ``../`` and absolute paths from escaping ``runs/``.

    The **charset** check is the one this docstring used to promise and not deliver. A glob
    metacharacter is a perfectly legal single path segment, so ``run[0-9]``, ``wild*card``
    and ``run?id`` all passed the path check — and then became directory names that every
    shell loop over ``runs/`` silently mishandles. Not an escape hole, but the kind of
    defect that surfaces as "the cleanup script deleted the wrong run", which is worse than
    an error at creation time. Found by ``tests/test_registry.py``.

    Raises:
        ValueError: If ``run_id`` is empty, escapes its directory, or contains anything
            outside :data:`_RUN_ID_ALLOWED`.
    """
    if not run_id or run_id != Path(run_id).name or run_id in {".", ".."}:
        raise ValueError(
            f"run_id {run_id!r} is not a single safe path segment. Run ids are generated "
            f"by aegis_ml.registry.store.new_run_id() and must stay filesystem-safe: they "
            f"become directory names, and they cross process boundaries as JSON."
        )
    if not _RUN_ID_ALLOWED.match(run_id):
        raise ValueError(
            f"run_id {run_id!r} contains characters outside [A-Za-z0-9._-]. A run id "
            f"becomes a directory name, so a glob metacharacter in it makes every shell "
            f"loop over runs/ ambiguous — `rm -rf runs/{run_id}` would not mean what it "
            f"reads like. Ids from new_run_id() are always within this set."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Atomic write primitives
# ──────────────────────────────────────────────────────────────────────────────
def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically.

    Content goes to a uniquely-named ``.tmp`` sibling (same directory, therefore the same
    filesystem, therefore :func:`os.replace` is a rename and not a copy), is flushed and
    ``fsync``-ed, and only then replaces the target. A concurrent reader sees either the
    complete old file or the complete new one.

    The temp name carries the pid and a random token because the naive ``path.tmp``
    collides when two runs write the same index concurrently — and a collision on a temp
    file is a half-written final file.

    Args:
        path: Destination.
        data: Bytes to write.

    Returns:
        ``path``, for chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    """Write UTF-8 ``text`` atomically. See :func:`atomic_write_bytes`."""
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> Path:  # noqa: ANN401 - any JSON value
    """Serialise ``payload`` as indented UTF-8 JSON and write it atomically.

    ``default=str`` is set deliberately: a ``Path`` or ``datetime`` that leaks into a
    detail dict must not turn a successful training run into a ``TypeError`` at the very
    last step, after the expensive part already succeeded.
    """
    body = json.dumps(payload, indent=2, sort_keys=False, default=str, ensure_ascii=False)
    return atomic_write_text(path, body + "\n")


def atomic_copy(src: Path, dest: Path) -> Path:
    """Copy ``src`` onto ``dest`` atomically, preserving nothing but the bytes.

    Used for the two copies that must never be observed half-done: archiving the live
    serving artifact into its run directory, and installing a challenger over
    :attr:`aegis_ml.settings.Settings.artifact_path`. ``aegis.ml.get_model()`` may be
    loading that file at any instant.

    Args:
        src: Existing file.
        dest: Destination path; its parent is created if missing.

    Returns:
        ``dest``.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
    """
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(
            f"cannot copy {str(src)!r}: it does not exist or is not a regular file"
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with open(src, "rb") as reader, open(tmp, "wb") as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


@contextmanager
def _index_lock() -> Iterator[None]:
    """Hold an exclusive-create lock file while mutating ``index.json``.

    ``O_CREAT | O_EXCL`` is the one cross-platform atomic "claim this name" primitive that
    works on Windows too, which matters because the runbook ships PowerShell commands
    alongside bash. A lock older than :data:`_LOCK_STALE_S` is broken on the assumption
    its owner died; waiting longer than :data:`_LOCK_TIMEOUT_S` raises rather than
    proceeding unlocked.

    Yields:
        None, with the lock held.

    Raises:
        TimeoutError: If the lock cannot be acquired in time.
    """
    lock = registry_root() / ".index.lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT_S
    handle: int | None = None
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > _LOCK_STALE_S:
                _LOG.warning(
                    "breaking stale registry index lock %s (age %.0fs) — its owner "
                    "almost certainly died mid-write; run dirs remain authoritative",
                    lock,
                    age,
                )
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"could not acquire the registry index lock {str(lock)!r} within "
                    f"{_LOCK_TIMEOUT_S:.0f}s. Another aegis-ml process is writing. Nothing "
                    f"was written here: proceeding without the lock would produce an index "
                    f"missing one of the runs."
                ) from None
            time.sleep(0.05)
    try:
        os.write(handle, f"{os.getpid()}\n".encode())
        yield
    finally:
        if handle is not None:
            os.close(handle)
        lock.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Run identity
# ──────────────────────────────────────────────────────────────────────────────
def _slug(text: str) -> str:
    """Reduce ``text`` to a filesystem- and URL-safe token."""
    kept = [c if (c.isalnum() or c in "-_") else "-" for c in text.strip().lower()]
    slug = "".join(kept).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug or "domain")[:48]


def new_run_id(domain_id: str) -> str:
    """Mint a sortable, filesystem-safe run id for ``domain_id``.

    Shape: ``<domain-slug>-<YYYYmmddTHHMMSSmmm>-<6 hex>``. The three properties that
    matter, in order:

    1. **Sortable.** The timestamp is fixed-width UTC, so a lexicographic sort of
       ``runs/`` is a chronological sort. ``list_runs`` and every ``ls`` agree.
    2. **Collision-proof.** Millisecond resolution is not enough when a sweep launches
       several runs in one loop iteration, so a random suffix is appended. Two runs
       sharing a directory would silently overwrite each other's model.
    3. **Stamped here, not by the caller.** The timestamp is taken from
       ``datetime.now(UTC)`` *inside* this function. A caller-supplied "now" is how a
       replayed pipeline ends up writing yesterday's id today.

    Args:
        domain_id: The adapter's ``DOMAIN_ID``; slugified into the prefix so a human
            scanning ``runs/`` can tell the domains apart without opening anything.

    Returns:
        The new run id. Nothing is created on disk — :func:`save_run` does that.
    """
    now = datetime.now(UTC)
    stamp = f"{now.strftime('%Y%m%dT%H%M%S')}{now.microsecond // 1000:03d}"
    return f"{_slug(domain_id)}-{stamp}-{secrets.token_hex(3)}"


def _utc_now_iso() -> str:
    """Return the current instant as an ISO-8601 UTC string."""
    return datetime.now(UTC).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Writing runs
# ──────────────────────────────────────────────────────────────────────────────
def _write_artifact(dest: Path, value: Any) -> None:  # noqa: ANN401 - deliberately open
    """Write one artifact, dispatching on the *type* of ``value``.

    The dispatch rule is explicit rather than clever, because the one ambiguity that
    matters is ``str``:

    * :class:`pathlib.Path` → an existing file to copy in.
    * ``bytes`` → written verbatim.
    * ``str`` → written as UTF-8 **text content**, never interpreted as a path. A model
      card that happens to be one line long must not be mistaken for a filename.
    * pydantic model / ``dict`` / ``list`` → JSON.
    * anything exposing ``to_parquet`` (a DataFrame) with a ``.parquet`` destination →
      written through pandas, so the drift reference frame round-trips with its dtypes.

    Args:
        dest: Destination path inside the run directory.
        value: The artifact, in one of the forms above.

    Raises:
        TypeError: For anything else — naming the type, because a silently skipped
            artifact is a run that looks complete and is not.
    """
    if isinstance(value, Path):
        atomic_copy(value, dest)
        return
    if isinstance(value, bytes | bytearray):
        atomic_write_bytes(dest, bytes(value))
        return
    if isinstance(value, str):
        atomic_write_text(dest, value)
        return
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        atomic_write_json(dest, dump(mode="json"))
        return
    if isinstance(value, dict | list):
        atomic_write_json(dest, value)
        return
    to_parquet = getattr(value, "to_parquet", None)
    if callable(to_parquet) and dest.suffix == ".parquet":
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        try:
            to_parquet(tmp, index=False)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return
    raise TypeError(
        f"cannot store artifact {dest.name!r}: unsupported type {type(value).__name__}. "
        f"Pass a Path to copy, bytes, a str of text content, a pydantic model, a "
        f"dict/list for JSON, or a DataFrame for a .parquet destination."
    )


def _dump_model(model: Any, dest: Path) -> None:  # noqa: ANN401 - any fitted estimator
    """Persist a fitted estimator to ``dest`` with joblib, atomically.

    joblib is imported here rather than at module scope because this module is on the
    import path of the dep-free CLI: listing runs must not require sklearn's transitive
    stack to be installed.
    """
    if isinstance(model, Path | str):
        source = Path(model)
        if not source.is_file():
            raise FileNotFoundError(
                f"model source {str(source)!r} does not exist. Pass an existing .joblib "
                f"path, or the fitted estimator object itself."
            )
        atomic_copy(source, dest)
        return
    from aegis_ml._require import require

    joblib = require("aegis-ml[serve]", "joblib")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        joblib.dump(model, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def save_run(
    entry: RegistryEntry,
    *,
    model: Any = None,  # noqa: ANN401 - a fitted estimator or a path to one
    artifacts: Mapping[str, Any] | None = None,
) -> Path:
    """Write a complete run directory and register it in the index.

    The order is load-bearing: every artifact lands first, then ``entry.json``, then the
    index. ``entry.json`` is what makes a directory a registry row, so a crash before it
    leaves an incomplete directory that :func:`reindex` correctly ignores — rather than an
    index entry pointing at a run with no model.

    Three artifacts are derived automatically when the caller does not supply them, so the
    documented layout is real rather than aspirational:

    * ``recipe.json`` from ``entry.result.recipe``
    * ``leaderboard.json`` from ``entry.result.leaderboard``
    * ``metrics.json`` — the flat scalar summary, carrying the *requested* and the
      *measured* coverage as two separate keys, per the house naming rule.

    Args:
        entry: The registry row. ``entry.paths`` is updated with everything written and
            the updated copy is what lands in ``entry.json``.
        model: Optional fitted estimator, or a path to an existing ``.joblib``. Stored as
            ``model.joblib`` — the exact file :func:`aegis_ml.registry.promote.promote`
            later copies over the serving artifact.
        artifacts: Optional ``name -> value`` map; see :func:`_write_artifact` for the
            accepted value types.

    Returns:
        The run directory.
    """
    directory = run_dir(entry.run_id)
    paths: dict[str, str] = dict(entry.paths)

    if model is not None:
        target = directory / "model.joblib"
        _dump_model(model, target)
        paths["model"] = str(target)

    for name, value in (artifacts or {}).items():
        dest = artifact(entry.run_id, name)
        _write_artifact(dest, value)
        paths[Path(name).stem] = str(dest)

    if entry.result.recipe is not None and "recipe" not in paths:
        dest = directory / "recipe.json"
        atomic_write_json(dest, entry.result.recipe.model_dump(mode="json"))
        paths["recipe"] = str(dest)
    if entry.result.leaderboard is not None and "leaderboard" not in paths:
        dest = directory / "leaderboard.json"
        atomic_write_json(dest, entry.result.leaderboard.model_dump(mode="json"))
        paths["leaderboard"] = str(dest)
    if "metrics" not in paths:
        dest = directory / "metrics.json"
        atomic_write_json(dest, _metrics_payload(entry))
        paths["metrics"] = str(dest)

    stored = entry.model_copy(update={"paths": paths})
    entry_file = directory / "entry.json"
    atomic_write_json(entry_file, stored.model_dump(mode="json"))
    _upsert_index(stored)
    _LOG.info("registry: saved run %s (%s) → %s", stored.run_id, stored.stage, directory)
    return directory


def _metrics_payload(entry: RegistryEntry) -> dict[str, Any]:
    """Flatten a run's scalars for ``metrics.json`` and the optional mirrors.

    Requested and measured coverage stay two keys with two names. Collapsing them into one
    ``coverage`` field is precisely the mistake ``contracts/protocols.py`` exists to
    prevent, and this file is the one a dashboard is most likely to read.
    """
    result = entry.result
    return {
        "run_id": entry.run_id,
        "domain_id": entry.domain_id,
        "created_at": entry.created_at,
        "stage": entry.stage,
        "task": result.task,
        "target": result.target,
        "metric_name": result.metric_name,
        "metric_value": result.metric_value,
        "requested_coverage": result.requested_coverage,
        "empirical_coverage": result.empirical_coverage,
        "training_size": result.training_size,
        "calibration_size": result.calibration_size,
        "test_size": result.test_size,
        "dataset_digest": result.dataset_digest,
        "tier": result.recipe.tier if result.recipe else None,
        "gate_promoted": entry.gate.promoted if entry.gate else None,
    }


def set_stage(run_id: str, stage: Stage) -> RegistryEntry:
    """Move a run to a new lifecycle stage, in the run directory and in the index.

    ``entry.json`` is rewritten first for the same reason :func:`save_run` orders its
    writes that way: the run directory is the truth and the index is a cache of it.

    Args:
        run_id: The run to move.
        stage: ``"staging"``, ``"production"`` or ``"archived"``.

    Returns:
        The updated entry.

    Raises:
        ValueError: On an unknown stage — an unrecognised stage string would otherwise sit
            in the index making the run invisible to both ``champion()`` and ``rollback()``.
    """
    if stage not in ("staging", "production", "archived"):
        raise ValueError(
            f"unknown stage {stage!r}; expected 'staging', 'production' or 'archived'"
        )
    entry = load_entry(run_id)
    updated = entry.model_copy(update={"stage": stage})
    atomic_write_json(run_dir(run_id) / "entry.json", updated.model_dump(mode="json"))
    _upsert_index(updated)
    _LOG.info("registry: run %s → stage %s", run_id, stage)
    return updated


# ──────────────────────────────────────────────────────────────────────────────
# Reading runs
# ──────────────────────────────────────────────────────────────────────────────
def load_entry(run_id: str) -> RegistryEntry:
    """Load one run's authoritative ``entry.json``.

    Never reads the index: the index is derived, and a stale index that disagrees with a
    run directory must not be able to hand back a wrong stage or a wrong metric.

    Args:
        run_id: The run to load.

    Returns:
        The entry.

    Raises:
        FileNotFoundError: If the run directory has no ``entry.json``, naming what a
            complete run looks like.
    """
    _validate_run_id(run_id)
    path = runs_root() / run_id / "entry.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no registry entry for run {run_id!r} (expected {str(path)!r}). A run "
            f"directory only counts as a registry row once entry.json is written; a "
            f"directory without one is an interrupted save, not a run."
        )
    return RegistryEntry.model_validate_json(path.read_text(encoding="utf-8"))


def _sorted(entries: list[RegistryEntry]) -> list[RegistryEntry]:
    """Sort newest first, breaking ties on run id so the order is total and stable."""
    return sorted(entries, key=lambda e: (e.created_at, e.run_id), reverse=True)


def reindex() -> list[RegistryEntry]:
    """Rebuild ``index.json`` by walking every ``runs/*/entry.json``.

    The index exists only so that ``list_runs`` does not open hundreds of small files. It
    is disposable by construction, and this function is the proof: delete it, run this,
    get it back. That is why a corrupt or missing index is a warning here and not an
    outage.

    Returns:
        Every entry found, newest first.
    """
    entries: list[RegistryEntry] = []
    for directory in sorted(runs_root().iterdir()):
        if not directory.is_dir():
            continue
        candidate = directory / "entry.json"
        if not candidate.is_file():
            continue
        try:
            entries.append(
                RegistryEntry.model_validate_json(candidate.read_text(encoding="utf-8"))
            )
        except (ValueError, OSError) as exc:
            _LOG.warning(
                "registry: skipping unreadable entry %s during reindex: %s", candidate, exc
            )
    entries = _sorted(entries)
    with _index_lock():
        atomic_write_json(index_path(), [e.model_dump(mode="json") for e in entries])
    return entries


def _read_index() -> list[RegistryEntry]:
    """Return the cached index, rebuilding it from the run directories if unusable."""
    path = index_path()
    if not path.is_file():
        return reindex()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [RegistryEntry.model_validate(row) for row in raw]
    except (ValueError, OSError) as exc:
        _LOG.warning(
            "registry: index.json is unreadable (%s); rebuilding it from run "
            "directories, which are the source of truth",
            exc,
        )
        return reindex()


def _upsert_index(entry: RegistryEntry) -> None:
    """Insert or replace one row in the index, under the lock, newest first."""
    with _index_lock():
        path = index_path()
        rows: list[RegistryEntry] = []
        if path.is_file():
            try:
                rows = [
                    RegistryEntry.model_validate(row)
                    for row in json.loads(path.read_text(encoding="utf-8"))
                ]
            except (ValueError, OSError) as exc:
                _LOG.warning("registry: discarding unreadable index (%s)", exc)
                rows = []
        rows = [row for row in rows if row.run_id != entry.run_id]
        rows.append(entry)
        atomic_write_json(path, [row.model_dump(mode="json") for row in _sorted(rows)])


def list_runs(
    *,
    domain_id: str | None = None,
    stage: Stage | None = None,
    limit: int | None = None,
) -> list[RegistryEntry]:
    """List registered runs, newest first.

    Args:
        domain_id: Restrict to one adapter domain.
        stage: Restrict to one lifecycle stage.
        limit: Return at most this many rows, applied *after* filtering — so
            ``list_runs(stage="production", limit=1)`` is the champion and not "the newest
            run, if it happens to be in production".

    Returns:
        Matching entries, newest first.
    """
    entries = _read_index()
    if domain_id is not None:
        entries = [e for e in entries if e.domain_id == domain_id]
    if stage is not None:
        entries = [e for e in entries if e.stage == stage]
    entries = _sorted(entries)
    if limit is not None:
        entries = entries[: max(0, limit)]
    return entries


def champion(domain_id: str) -> RegistryEntry | None:
    """Return the production entry for ``domain_id``, or ``None`` if there is none.

    "None" is a real and common answer — before the first promotion, the serving artifact
    may exist (written by ``python -m app.ml``) with no registry row behind it. Callers
    must handle that rather than assume a champion; :func:`aegis_ml.registry.promote.promote`
    backs such an unregistered artifact up before overwriting it.

    Args:
        domain_id: The adapter domain.

    Returns:
        The newest entry with ``stage == "production"``, or ``None``.
    """
    found = list_runs(domain_id=domain_id, stage="production", limit=1)
    return found[0] if found else None


def touch_created_at(entry: RegistryEntry) -> RegistryEntry:
    """Return ``entry`` with ``created_at`` stamped now, if it is blank.

    :class:`RegistryEntry` documents that the caller stamps ``created_at``; this is the
    one convenience that fills it in, because an entry with an empty timestamp sorts to
    the bottom of every listing and silently loses its place in history.
    """
    if entry.created_at:
        return entry
    return entry.model_copy(update={"created_at": _utc_now_iso()})
