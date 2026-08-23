"""Stage-level lineage for one pipeline execution — the record, not the log.

A log tells you what happened in the order it happened. A **manifest** tells you what each
stage consumed, what it produced, how long it took, whether it was skipped or served from
cache, and — when it failed — exactly where. That difference is what makes a re-run
auditable and a crash resumable, and it is why this module exists rather than a
``logging.info`` per step.

Three properties are load-bearing:

**Every stage is recorded, including the ones that did not run.** A skipped stage and a
stage that ran and produced nothing look identical in a log and are recorded distinctly
here (``status="skipped"`` carries ``skip_reason``). The same rule
:class:`~aegis_ml.contracts.protocols.Leaderboard` applies to tiers: an empty slot and an
unavailable dependency must never be indistinguishable.

**Cache hits are visible.** :class:`StageCache` makes a five-minute AutoML search skippable
on a re-run, which is only safe because the manifest says ``status="cached"`` and prints the
content key that was matched. A silently reused result is a result nobody can date.

**Failure is attributed to a stage, not to the flow.** ``manifest.error`` carries
``"<stage>: <ExceptionType>: <message>"``, so the first line of a failed run names the unit
to re-run rather than the flow that contained it.

The content-addressing rule: a stage's cache key is derived from the *inputs it actually
reads* (a frame digest, a config value, a seed), never from wall-clock time or a run id.
Two runs over the same bytes with the same configuration must collide; anything else makes
the cache a coin toss.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aegis_ml.contracts.protocols import RunManifest

__all__ = [
    "CacheSpec",
    "SkipStage",
    "StageCache",
    "StageGraph",
    "StageRecord",
    "StageSpec",
    "content_key",
    "finish_manifest",
    "new_manifest",
    "render_summary",
    "stage",
    "write_manifest",
]


def _now() -> str:
    """Return an ISO-8601 UTC timestamp — the one time format this package writes."""
    return datetime.now(UTC).isoformat()


class SkipStage(Exception):  # noqa: N818 - control flow, not an error condition
    """Raised inside a stage body to record it as *skipped* rather than failed.

    A stage that decides mid-body that it has nothing to do (no labels yet, an optional
    report module absent) must not be recorded as an error — a red manifest that is
    actually fine trains readers to ignore red manifests. It carries the reason, which is
    printed in the summary table.
    """

    def __init__(self, reason: str) -> None:
        """Record why the stage declined to run."""
        super().__init__(reason)
        self.reason = reason


def content_key(*parts: Any) -> str:
    """Return a stable ``sha256:`` digest over JSON-canonicalised ``parts``.

    Args:
        *parts: Anything JSON-serialisable that the stage's result depends on — a frame
            digest, a sorted tier list, a time budget, a seed. Non-serialisable values are
            rendered with ``repr``, which is stable for the scalars and small containers
            these keys are built from.

    Returns:
        ``"sha256:<hex>"``.

    Why a digest of the *inputs* and never of the output: a cache keyed on what a stage
    produced can only be validated by producing it again, which is the cost the cache
    exists to avoid.
    """
    payload = json.dumps(parts, sort_keys=True, default=repr, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class StageRecord:
    """The mutable row one stage writes into the manifest.

    The body of a stage is handed this object and annotates it as it goes —
    ``record.rows_in = len(frame)``, ``record.metric("r2", 0.61)``, ``record.note(...)``.
    Everything a reader needs to judge the stage lands here rather than in stdout, because
    stdout is not part of the artifact and this is.
    """

    name: str
    description: str = ""
    status: str = "running"
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    duration_seconds: float = 0.0
    attempts: int = 1
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    rows_in: int | None = None
    rows_out: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cache_key: str | None = None
    skip_reason: str | None = None
    error: str | None = None

    def metric(self, name: str, value: float) -> None:
        """Record one measured number for this stage.

        Args:
            name: Metric key, e.g. ``"r2"`` or ``"empirical_coverage"``.
            value: The measured value. Coerced to ``float`` so the manifest stays JSON.
        """
        self.metrics[name] = float(value)

    def note(self, text: str) -> None:
        """Append a human-readable finding a reader must not miss."""
        self.notes.append(text)

    def artifact(self, name: str, path: str | Path) -> None:
        """Record a file this stage wrote, by role name."""
        self.artifacts[name] = str(path)

    def to_dict(self) -> dict[str, Any]:
        """Render the record as the plain dict :class:`RunManifest` stores.

        Returns:
            A JSON-safe dict with empty optional fields dropped, so a manifest of ten
            clean stages stays readable instead of being 60% ``null``.
        """
        out: dict[str, Any] = {
            "stage": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 4),
        }
        if self.description:
            out["description"] = self.description
        if self.attempts != 1:
            out["attempts"] = self.attempts
        for key, value in (
            ("inputs", self.inputs),
            ("outputs", self.outputs),
            ("metrics", self.metrics),
            ("artifacts", self.artifacts),
            ("notes", self.notes),
        ):
            if value:
                out[key] = value
        for key, scalar in (
            ("rows_in", self.rows_in),
            ("rows_out", self.rows_out),
            ("cache_key", self.cache_key),
            ("skip_reason", self.skip_reason),
            ("error", self.error),
        ):
            if scalar is not None:
                out[key] = scalar
        return out


def new_manifest(run_id: str, flow: str) -> RunManifest:
    """Open a manifest for one flow execution.

    Args:
        run_id: The registry run id this execution belongs to.
        flow: Flow name, e.g. ``"train_flow"``.

    Returns:
        An open :class:`~aegis_ml.contracts.protocols.RunManifest` with no stages yet.
    """
    return RunManifest(run_id=run_id, flow=flow, started_at=_now())


def finish_manifest(manifest: RunManifest, *, error: BaseException | None = None) -> RunManifest:
    """Close a manifest, stamping the end time and the terminal verdict.

    Args:
        manifest: The manifest to close.
        error: The exception that ended the flow, if any. Recorded verbatim with its type,
            because "failed" without the exception text is a bug report nobody can action.

    Returns:
        The same manifest, mutated in place and returned for chaining.
    """
    manifest.finished_at = _now()
    if error is not None:
        manifest.ok = False
        manifest.error = f"{type(error).__name__}: {error}"
    return manifest


@contextmanager
def stage(manifest: RunManifest, name: str, description: str = "") -> Iterator[StageRecord]:
    """Record one stage's start, end, duration and outcome into ``manifest``.

    This is the primitive every flow ultimately writes through. :class:`StageGraph` adds
    declarative inputs/outputs, skip predicates, retries and caching on top of it, but a
    one-off stage in an ad-hoc script needs nothing more than this.

    Args:
        manifest: The open manifest to append to.
        name: Stage name, unique within the flow.
        description: One line saying what the stage is for; printed in the summary table.

    Yields:
        The :class:`StageRecord` the body annotates.

    Raises:
        Exception: Anything the body raises is recorded against this stage — with the
            stage name prefixed onto ``manifest.error`` so the failure names the unit to
            re-run — and then re-raised. Swallowing it here would produce a green manifest
            for a flow that did not finish, which is the exact failure this package refuses
            to ship.
    """
    record = StageRecord(name=name, description=description)
    started = time.perf_counter()
    try:
        yield record
    except SkipStage as skip:
        record.status = "skipped"
        record.skip_reason = skip.reason
    except BaseException as exc:
        record.status = "error"
        record.error = f"{type(exc).__name__}: {exc}"
        manifest.ok = False
        manifest.error = f"{name}: {record.error}"
        raise
    else:
        if record.status == "running":
            record.status = "ok"
    finally:
        record.duration_seconds = time.perf_counter() - started
        record.finished_at = _now()
        manifest.stages.append(record.to_dict())


@dataclass(frozen=True)
class CacheSpec:
    """How one stage's result is content-addressed and round-tripped through JSON.

    Only JSON-shaped results are cacheable on purpose. A cached fitted model would need a
    pickle whose validity depends on the exact library versions present at load time, and a
    stale-but-loadable estimator is precisely the silent wrong answer this package refuses.
    The expensive stage worth caching — the AutoML search — returns a
    :class:`~aegis_ml.contracts.protocols.Recipe`, which is JSON by construction.

    Attributes:
        key: ``context -> list of hashable parts``. Returning ``None`` disables caching for
            this run (e.g. the frame digest could not be computed), which is recorded.
        dumps: Result → JSON-safe dict.
        loads: JSON-safe dict → result.
    """

    key: Callable[[Mapping[str, Any]], Sequence[Any] | None]
    dumps: Callable[[Any], dict[str, Any]]
    loads: Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class StageSpec:
    """One declaratively described unit of a flow.

    Declaring ``inputs``/``outputs`` by context key is what turns the manifest into a
    lineage record: a reader can see that ``fit`` consumed ``train_frame`` and ``recipe``
    and produced ``model``, without reading the flow's source.

    Attributes:
        name: Unique stage name within the flow.
        description: One line for the summary table.
        inputs: Context keys this stage reads.
        outputs: Context keys this stage writes (the return value is bound to the first).
        skip_if: ``context -> str | None``. A returned string is the recorded skip reason;
            ``None`` means run. Used for genuinely conditional work, never to hide failure.
        retries: Extra attempts on failure. **Only ever non-zero for a stage whose failure
            can be transient** — the AutoML subprocess bridge. Retrying a deterministic
            stage burns the budget re-deriving the same exception.
        backoff_seconds: Base delay; attempt *n* waits ``backoff_seconds * 2 ** (n - 1)``.
        cache: Content-addressed cache policy, or ``None`` for an uncached stage.
        optional: When true a failure is recorded as ``status="degraded"`` and the flow
            continues. Reserved for *reporting* stages (SHAP HTML, data profile) whose
            absence costs a reader a picture, never a stage whose output the model or the
            gate depends on. The failure is written into the manifest and into the run's
            notes — degraded is loud, it is simply not fatal.
    """

    name: str
    description: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    skip_if: Callable[[Mapping[str, Any]], str | None] | None = None
    retries: int = 0
    backoff_seconds: float = 2.0
    cache: CacheSpec | None = None
    optional: bool = False


class StageCache:
    """A filesystem, content-addressed cache of JSON stage results.

    Keyed by ``<root>/<flow>/<stage>/<sha256>.json``. The key is a digest of the stage's
    declared inputs, so a re-run over unchanged data and configuration hits, and a re-run
    over a *changed* frame misses — which is the only behaviour that makes skipping a
    five-minute search safe to do by default.
    """

    def __init__(self, root: Path, *, flow: str, enabled: bool = True) -> None:
        """Create a cache rooted at ``root`` for one flow.

        Args:
            root: Directory to hold cache entries; created on first write.
            flow: Flow name, used as the first path segment so two flows never collide.
            enabled: ``False`` disables reads *and* writes — what ``--force`` sets.
        """
        self.root = Path(root) / flow
        self.enabled = enabled

    def path_for(self, stage_name: str, key: str) -> Path:
        """Return the file a ``(stage, key)`` pair maps to."""
        return self.root / stage_name / f"{key.replace('sha256:', '')}.json"

    def load(self, stage_name: str, key: str) -> dict[str, Any] | None:
        """Return the cached payload for ``key``, or ``None`` on a miss.

        A corrupt or unreadable entry is treated as a miss and the file is removed: a cache
        is an optimisation, and a half-written JSON file from a killed process must not be
        able to fail a run.
        """
        if not self.enabled:
            return None
        path = self.path_for(stage_name, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            return None

    def store(self, stage_name: str, key: str, payload: dict[str, Any]) -> Path | None:
        """Write ``payload`` under ``key``, returning the path (or ``None`` if disabled)."""
        if not self.enabled:
            return None
        path = self.path_for(stage_name, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path


class StageGraph:
    """Runs a flow's stages against a shared context, recording each into the manifest.

    The context is a plain dict keyed by the names stages declare in ``inputs``/``outputs``.
    It is deliberately not a typed object: stages produce dataframes, fitted estimators,
    pydantic models and paths, and forcing those through one schema would buy nothing that
    the declared input/output names do not already give a reader.

    Usage::

        graph = StageGraph(manifest, cache=StageCache(root, flow="train_flow"))
        graph.run(StageSpec(name="ingest", outputs=("frame",)), lambda rec: load_frame())
        frame = graph.context["frame"]
    """

    def __init__(
        self,
        manifest: RunManifest,
        *,
        cache: StageCache | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Bind a graph to one manifest.

        Args:
            manifest: The open manifest every stage records into.
            cache: Optional content-addressed cache; ``None`` disables caching entirely.
            context: Seed values for the shared context (e.g. ``{"problem": problem}``).
        """
        self.manifest = manifest
        self.cache = cache
        self.context: dict[str, Any] = dict(context or {})
        self.degraded: list[str] = []

    def run(self, spec: StageSpec, fn: Callable[[StageRecord], Any]) -> Any:  # noqa: ANN401
        """Execute one stage and bind its result into the context.

        The order of operations is deliberate: skip predicate → cache lookup → attempt loop.
        A skipped stage never consults the cache (there is nothing to serve), and a cache
        hit never retries (there is nothing to retry).

        Args:
            spec: The declarative description of the stage.
            fn: The body, taking the :class:`StageRecord` to annotate and returning the
                stage's result.

        Returns:
            The stage result — from cache, from the body, or ``None`` when skipped.

        Raises:
            Exception: Whatever the body raised, once ``spec.retries`` attempts are spent,
                unless ``spec.optional`` is set (in which case the failure is recorded as
                ``degraded`` and ``None`` is returned).
        """
        with stage(self.manifest, spec.name, spec.description) as record:
            record.inputs = list(spec.inputs)
            record.outputs = list(spec.outputs)

            if spec.skip_if is not None:
                reason = spec.skip_if(self.context)
                if reason:
                    raise SkipStage(reason)

            cache_key: str | None = None
            if spec.cache is not None and self.cache is not None:
                parts = spec.cache.key(self.context)
                if parts is None:
                    record.note("not cacheable this run: cache key inputs unavailable")
                else:
                    cache_key = content_key(spec.name, *parts)
                    record.cache_key = cache_key
                    cached = self.cache.load(spec.name, cache_key)
                    if cached is not None:
                        result = spec.cache.loads(cached)
                        record.status = "cached"
                        record.note("served from the content-addressed stage cache")
                        self._bind(spec, result)
                        return result

            result = self._attempt(spec, record, fn)
            if record.status == "degraded":
                return None

            if spec.cache is not None and self.cache is not None and cache_key is not None:
                path = self.cache.store(spec.name, cache_key, spec.cache.dumps(result))
                if path is not None:
                    record.artifact("cache_entry", path)

            self._bind(spec, result)
            return result

    def _attempt(
        self,
        spec: StageSpec,
        record: StageRecord,
        fn: Callable[[StageRecord], Any],
    ) -> Any:  # noqa: ANN401 - stage results are heterogeneous by design
        """Run the body, retrying only as ``spec.retries`` permits.

        Args:
            spec: The stage description (supplies the retry budget and optionality).
            record: The record to annotate with attempts and any degraded note.
            fn: The stage body.

        Returns:
            The body's result, or ``None`` when an optional stage failed.

        Raises:
            Exception: The last failure, for a non-optional stage with no attempts left.
        """
        last: BaseException | None = None
        for attempt in range(1, spec.retries + 2):
            record.attempts = attempt
            try:
                return fn(record)
            except SkipStage:
                raise
            except Exception as exc:  # noqa: BLE001 - re-raised below; the reason is recorded
                last = exc
                record.note(f"attempt {attempt} failed: {type(exc).__name__}: {exc}")
                if attempt <= spec.retries:
                    time.sleep(spec.backoff_seconds * (2 ** (attempt - 1)))
        assert last is not None  # noqa: S101 - the loop cannot exit without setting it
        if spec.optional:
            record.status = "degraded"
            record.error = f"{type(last).__name__}: {last}"
            self.degraded.append(f"{spec.name}: {record.error}")
            return None
        raise last

    def _bind(self, spec: StageSpec, result: Any) -> None:  # noqa: ANN401
        """Bind a stage result into the context under its declared output names.

        One declared output binds the result directly; several unpack a tuple positionally,
        which is what ``three_way_split`` returning ``(train, calibration, test)`` needs.
        """
        if not spec.outputs or result is None:
            return
        if len(spec.outputs) == 1:
            self.context[spec.outputs[0]] = result
            return
        values = tuple(result)
        if len(values) != len(spec.outputs):
            raise ValueError(
                f"stage {spec.name!r} declares {len(spec.outputs)} outputs "
                f"{spec.outputs} but returned {len(values)} values"
            )
        for name, value in zip(spec.outputs, values, strict=True):
            self.context[name] = value


def write_manifest(path: str | Path, manifest: RunManifest) -> Path:
    """Write ``manifest`` to ``path`` as indented JSON, creating parent directories.

    Args:
        path: Destination file, conventionally ``<run_dir>/manifest.json``.
        manifest: The manifest to serialise.

    Returns:
        The path written.

    The write is atomic (temp file then ``replace``) because a manifest is read by the
    promotion gate and the console: a half-written one would be a parse error at exactly
    the moment someone is trying to find out what went wrong.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


_STATUS_MARK = {
    "ok": "ok",
    "cached": "cached",
    "skipped": "skip",
    "degraded": "degraded",
    "error": "FAILED",
    "running": "running",
}


def render_summary(manifest: RunManifest) -> str:
    """Render the manifest as an aligned console table.

    Args:
        manifest: A closed (or open) manifest.

    Returns:
        A multi-line string: one row per stage with status, duration, row counts and the
        stage's headline metrics, then a footer with the verdict.

    Printed at the end of every flow because the manifest file answers "what happened" only
    for someone who knows to go and open it, and the person watching a training run does
    not yet know whether there is anything to look for.
    """
    header = ("stage", "status", "secs", "rows", "detail")
    rows: list[tuple[str, str, str, str, str]] = []
    for entry in manifest.stages:
        rows_in = entry.get("rows_in")
        rows_out = entry.get("rows_out")
        if rows_in is None and rows_out is None:
            rows_txt = "-"
        elif rows_out is None or rows_in == rows_out:
            rows_txt = str(rows_in if rows_in is not None else rows_out)
        else:
            rows_txt = f"{rows_in}→{rows_out}"
        detail_bits: list[str] = []
        metrics = entry.get("metrics") or {}
        detail_bits += [f"{k}={v:.4g}" for k, v in list(metrics.items())[:3]]
        if entry.get("skip_reason"):
            detail_bits.append(str(entry["skip_reason"]))
        if entry.get("error"):
            detail_bits.append(str(entry["error"]))
        rows.append(
            (
                str(entry.get("stage", "?")),
                _STATUS_MARK.get(str(entry.get("status")), str(entry.get("status"))),
                f"{float(entry.get('duration_seconds', 0.0)):.2f}",
                rows_txt,
                "; ".join(detail_bits),
            )
        )

    widths = [len(h) for h in header]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]
    widths[-1] = min(widths[-1], 60)

    def line(cells: Sequence[str]) -> str:
        parts = []
        for cell, width in zip(cells, widths, strict=True):
            text = cell if len(cell) <= width else cell[: width - 1] + "…"
            parts.append(text.ljust(width))
        return "  ".join(parts).rstrip()

    out = [line(header), line(["-" * w for w in widths])]
    out += [line(row) for row in rows]
    total = sum(float(e.get("duration_seconds", 0.0)) for e in manifest.stages)
    verdict = "OK" if manifest.ok else f"FAILED — {manifest.error}"
    out.append("")
    out.append(f"{manifest.flow} run={manifest.run_id} total={total:.2f}s  {verdict}")
    return "\n".join(out)
