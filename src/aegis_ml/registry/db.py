"""Optional relational persistence for ML — the gap Aegis genuinely has.

The exploration behind ``finalplan.md`` confirmed that Aegis persists **nothing** about ML
relationally: no predictions table, no model registry table, no drift table. The only
precedent for a metrics table anywhere in the platform is
``aegis/src/aegis/ops/models.py::EvalResult`` — table ``eval_results``, columns
``ts, run_id, prompt_key, tenant_id, metric, score, passed, detail`` — and this module
follows it deliberately, down to the details that look like details and are not:

* **``tenant_id`` is a plain indexed column, with no cross-package foreign key.** Aegis
  isolates tenants at the query/RLS layer (``aegis.memory`` does the same), and a DDL
  foreign key from an ML table into ``aegis.governance``'s ``tenants`` would couple two
  packages whose migrations ship separately.
* **JSON columns are ``jsonb`` on PostgreSQL and portable ``JSON`` everywhere else**, so
  ``create_all`` materialises on the SQLite database the unit tests use.
* **Timestamps default server-side**, so a row's ``ts`` is the database's clock, not one
  of N application clocks.

These tables are a **mirror, exactly like MLflow**. The filesystem registry
(:mod:`aegis_ml.registry.store`) remains the source of truth; nothing in train → gate →
promote → serve reads a row back from here. What they add is the thing a filesystem cannot
do well: *querying across runs and across time* — "every prediction this model made last
Tuesday", "drifted share by week", "which runs ever reached production".

Everything is constructed lazily by :func:`tables`. SQLAlchemy is not imported at module
scope, because ``aegis_ml.registry`` is on the import path of the light CLI and listing
runs must not require a database driver.

Base class selection is explicit and reported, never guessed:

* Running **inside the Aegis host**, ``aegis.data.AegisBase`` is importable, and these
  tables register on its metadata — so the host's own ``AegisBase.metadata.create_all``
  materialises them alongside every other Aegis table.
* Running **standalone** (this package's own tests, a trainer venv), a local
  :class:`~sqlalchemy.orm.DeclarativeBase` is used instead.

The choice is recorded in :attr:`MLTables.base_origin` and logged, because "which
metadata did my tables land on" is exactly the question a puzzling ``create_all`` raises.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aegis_ml._require import is_available, require
from aegis_ml.contracts.protocols import DriftReport, RegistryEntry
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

__all__ = [
    "MLTables",
    "async_dsn",
    "create_all",
    "engine_from_dsn",
    "insert_drift_report",
    "insert_prediction",
    "insert_run",
    "latest_drift",
    "recent_predictions",
    "recent_runs",
    "session_scope",
    "tables",
]

_LOG = logging.getLogger(__name__)

_SQLALCHEMY_INSTALL = "sqlalchemy[asyncio]>=2.0"
"""Install target quoted back to the user when SQLAlchemy is missing.

Not one of this package's own extras: the relational mirror is for hosts that already run
a database (the Aegis backend ships ``sqlalchemy[asyncio]``), so the honest instruction is
the package name rather than an ``aegis-ml[...]`` extra that would drag a driver into the
trainer venv for no reason.
"""

_ASYNCPG_INSTALL = "asyncpg>=0.30"
"""Driver quoted when a PostgreSQL DSN is used without an async driver installed."""


@dataclass(frozen=True)
class MLTables:
    """The three ORM classes, their base, and where that base came from.

    Attributes:
        base: The declarative base the tables registered on.
        run: ``ml_runs`` — one row per training run, the relational echo of a
            :class:`~aegis_ml.contracts.protocols.RegistryEntry`.
        prediction: ``ml_predictions`` — one row per served prediction.
        drift: ``ml_drift_reports`` — one row per drift evaluation.
        base_origin: ``"aegis.data.AegisBase"`` when running inside the host, else
            ``"aegis_ml.registry.db.MLBase"``. Reported so a surprising ``create_all``
            can be explained instead of investigated.
    """

    base: Any
    run: type[Any]
    prediction: type[Any]
    drift: type[Any]
    base_origin: str

    @property
    def all_tables(self) -> list[Any]:
        """The three ``Table`` objects, in dependency-free creation order."""
        return [self.run.__table__, self.prediction.__table__, self.drift.__table__]


_TABLES: MLTables | None = None


def tables() -> MLTables:
    """Build (once) and return the ORM classes.

    Deferred rather than declared at module scope for two reasons that both bite in
    practice: importing SQLAlchemy costs ~200 ms that the CLI's ``list`` path should not
    pay, and declaring classes at import time would bind them to whichever base was
    importable *then* — before a host has finished setting up ``aegis.data``.

    Returns:
        The cached :class:`MLTables`.

    Raises:
        ImportError: If SQLAlchemy is not installed, naming the install command.
    """
    global _TABLES  # noqa: PLW0603 - a deliberate one-shot module cache
    if _TABLES is not None:
        return _TABLES

    require(_SQLALCHEMY_INSTALL, "sqlalchemy")
    from sqlalchemy import (
        JSON,
        Boolean,
        DateTime,
        Float,
        Index,
        Integer,
        String,
        func,
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.orm import DeclarativeBase, mapped_column

    # ``JsonB`` in aegis.data, reproduced rather than imported so the tables also
    # materialise standalone: native jsonb on PostgreSQL, portable JSON everywhere else
    # (which is what keeps create_all working on the SQLite test database).
    json_b = JSON().with_variant(JSONB, "postgresql")
    timestamp = DateTime(timezone=True)

    base: Any
    origin: str
    if is_available("aegis.data"):
        from aegis.data import AegisBase  # type: ignore[import-not-found]

        base = AegisBase
        origin = "aegis.data.AegisBase"
    else:

        class MLBase(DeclarativeBase):
            """Standalone metadata for hosts without ``aegis.data`` on the path."""

        base = MLBase
        origin = "aegis_ml.registry.db.MLBase"

    _LOG.debug("aegis_ml relational mirror: tables registering on %s", origin)

    # Columns are declared with explicit types and no ``Mapped[...]`` annotations, unlike
    # ``aegis.ops.models``. That is forced by two constraints this module holds at once:
    # this file uses ``from __future__ import annotations`` (house rule), so annotations
    # are strings; and SQLAlchemy resolves those strings against the defining module's
    # globals, where ``Mapped`` deliberately does not exist because SQLAlchemy is imported
    # inside this function. The annotated form raises MappedAnnotationError. The schema
    # produced is identical — same tables, same types, same indexes — so the precedent is
    # followed where it counts: on disk.

    class MLRun(base):  # type: ignore[misc, valid-type]
        """One training run — the queryable echo of a ``RegistryEntry``.

        The primary key is the ``run_id`` string rather than a surrogate integer, because
        that id is minted by :func:`aegis_ml.registry.store.new_run_id` and is already the
        identifier every other layer speaks: the run directory name, the model card, the
        MLflow run name, and the ``run_id`` on every prediction row. A surrogate key would
        add a second identity for the same thing.

        ``requested_coverage`` and ``empirical_coverage`` are two columns, never one. The
        house rule from ``contracts/protocols.py`` applies to the schema as much as to the
        pydantic models: a dashboard that ``SELECT``s one ``coverage`` column cannot tell a
        reader whether the conformal interval delivered what it promised.
        """

        __tablename__ = "ml_runs"

        run_id = mapped_column(String(128), primary_key=True)
        domain_id = mapped_column(String(128), nullable=False, index=True)
        # Plain indexed column (no cross-package FK to aegis.governance ``tenants``),
        # mirroring aegis.ops.models.EvalResult exactly: isolation is enforced at the
        # query/RLS layer, not by DDL that would couple two packages' migrations.
        tenant_id = mapped_column(Integer(), nullable=True, default=None, index=True)
        created_at = mapped_column(timestamp, server_default=func.now(), index=True)
        stage = mapped_column(String(32), nullable=False, default="staging", index=True)
        task = mapped_column(String(32), nullable=False)
        target = mapped_column(String(128), nullable=False)
        metric_name = mapped_column(String(64), nullable=False, index=True)
        metric_value = mapped_column(Float(), nullable=False)
        requested_coverage = mapped_column(Float(), nullable=False)
        empirical_coverage = mapped_column(Float(), nullable=True, default=None)
        dataset_digest = mapped_column(String(64), nullable=True, default=None)
        detail = mapped_column(json_b, nullable=False, default=dict)

        __table_args__ = (Index("ix_ml_runs_domain_stage", "domain_id", "stage"),)

    class MLPrediction(base):  # type: ignore[misc, valid-type]
        """One served prediction — what makes drift monitoring real rather than a toy.

        ``feature_digest`` rather than the raw feature vector is the default written by
        :func:`aegis_ml.monitor.log.log_prediction`: a prediction log is a copy of
        production inputs, and production inputs carry PII. The digest still supports the
        two things the log is for — counting traffic and detecting duplicate/replayed
        inputs — without becoming a shadow dataset nobody governs. Callers that need the
        values for a drift reference opt in explicitly, and the values land in ``detail``.

        ``prediction`` is nullable and paired with ``prediction_label`` because both task
        types write here: a regression writes a float, a classification writes a class
        name (and, usually, its probability into ``detail``). Squeezing a class label into
        a float column via a class index is how a monitoring query silently starts
        averaging category codes.
        """

        __tablename__ = "ml_predictions"

        id = mapped_column(Integer(), primary_key=True, autoincrement=True)
        ts = mapped_column(timestamp, server_default=func.now(), index=True)
        tenant_id = mapped_column(Integer(), nullable=True, default=None, index=True)
        run_id = mapped_column(String(128), nullable=True, default=None, index=True)
        feature_digest = mapped_column(String(64), nullable=True, default=None, index=True)
        prediction = mapped_column(Float(), nullable=True, default=None)
        prediction_label = mapped_column(String(128), nullable=True, default=None)
        interval_low = mapped_column(Float(), nullable=True, default=None)
        interval_high = mapped_column(Float(), nullable=True, default=None)
        detail = mapped_column(json_b, nullable=False, default=dict)

        __table_args__ = (Index("ix_ml_predictions_run_ts", "run_id", "ts"),)

    class MLDriftReport(base):  # type: ignore[misc, valid-type]
        """One drift evaluation of live data against a run's stored reference frame.

        ``drifted_share`` is stored as the headline number rather than a p-value, and that
        is a measurement decision, not a formatting one: with twelve features, one feature
        crossing p<0.05 is the *expected* outcome under no drift at all. The share of
        drifted features against a threshold is the statistic that survives multiple
        comparisons; the per-feature scores live in ``detail`` for the human who wants them.

        A ``verdict`` of ``"block"`` never withdraws a serving model — see
        :mod:`aegis_ml.monitor.alerts`. It blocks *promotion* of anything calibrated on
        the drifted reference.
        """

        __tablename__ = "ml_drift_reports"

        id = mapped_column(Integer(), primary_key=True, autoincrement=True)
        ts = mapped_column(timestamp, server_default=func.now(), index=True)
        tenant_id = mapped_column(Integer(), nullable=True, default=None, index=True)
        run_id = mapped_column(String(128), nullable=True, default=None, index=True)
        drifted_share = mapped_column(Float(), nullable=False, default=0.0)
        dataset_drift = mapped_column(Boolean(), nullable=False, default=False)
        verdict = mapped_column(String(16), nullable=False, default="pass", index=True)
        detail = mapped_column(json_b, nullable=False, default=dict)

        __table_args__ = (Index("ix_ml_drift_run_ts", "run_id", "ts"),)

    _TABLES = MLTables(
        base=base,
        run=MLRun,
        prediction=MLPrediction,
        drift=MLDriftReport,
        base_origin=origin,
    )
    return _TABLES


# ──────────────────────────────────────────────────────────────────────────────
# Engine / session plumbing
# ──────────────────────────────────────────────────────────────────────────────
def async_dsn(dsn: str | None = None) -> str:
    """Normalise a DSN to an async driver URL.

    ``AEGIS_ML_POSTGRES_DSN`` is usually copied from the backend's own configuration,
    where it is a **sync** ``postgresql://`` URL. Handing that to
    ``create_async_engine`` raises a driver error several frames deep in SQLAlchemy that
    reads like a missing dependency. Rewriting the scheme here — and saying so in the log
    — is a translation, not a fallback: the target database is identical.

    Args:
        dsn: An explicit DSN, or ``None`` to read ``settings.postgres_dsn``.

    Returns:
        A DSN whose driver is async-capable.

    Raises:
        ValueError: When no DSN is configured at all.
    """
    raw = dsn or settings.postgres_dsn
    if not raw:
        raise ValueError(
            "no PostgreSQL DSN configured. Set AEGIS_ML_POSTGRES_DSN (or pass dsn=...) — "
            "the relational mirror is optional, so nothing else here needs it, but this "
            "call specifically asked for the database."
        )
    for prefix, replacement in (
        ("postgresql+asyncpg://", None),
        ("postgresql+psycopg://", None),
        ("postgresql://", "postgresql+asyncpg://"),
        ("postgres://", "postgresql+asyncpg://"),
    ):
        if raw.startswith(prefix):
            if replacement is None:
                return raw
            rewritten = replacement + raw[len(prefix) :]
            _LOG.debug("rewrote sync DSN scheme %r → asyncpg for the async engine", prefix)
            return rewritten
    return raw


def engine_from_dsn(dsn: str | None = None, **kwargs: Any) -> AsyncEngine:  # noqa: ANN401
    """Create an :class:`~sqlalchemy.ext.asyncio.AsyncEngine` for the mirror.

    Args:
        dsn: DSN, or ``None`` to use ``settings.postgres_dsn``.
        **kwargs: Passed through to ``create_async_engine`` (``echo``, ``pool_size``, …).

    Returns:
        A new engine. The caller owns it and must ``await engine.dispose()``.

    Raises:
        ImportError: If SQLAlchemy — or, for a PostgreSQL DSN, ``asyncpg`` — is missing.
    """
    require(_SQLALCHEMY_INSTALL, "sqlalchemy")
    url = async_dsn(dsn)
    if url.startswith("postgresql+asyncpg://"):
        require(_ASYNCPG_INSTALL, "asyncpg")
    from sqlalchemy.ext.asyncio import create_async_engine

    return create_async_engine(url, **kwargs)


@asynccontextmanager
async def session_scope(
    target: AsyncEngine | str | None = None,
) -> AsyncIterator[AsyncSession]:
    """Yield a committed-on-success session, disposing an engine it created itself.

    Args:
        target: An existing engine (not disposed — the caller owns it), a DSN string, or
            ``None`` to build one from ``settings.postgres_dsn``.

    Yields:
        An :class:`~sqlalchemy.ext.asyncio.AsyncSession`. Committed on clean exit, rolled
        back on exception — a half-written drift report is worse than none, because the
        next query averages it in.
    """
    require(_SQLALCHEMY_INSTALL, "sqlalchemy")
    from sqlalchemy.ext.asyncio import AsyncEngine as _AsyncEngine
    from sqlalchemy.ext.asyncio import async_sessionmaker

    owned = not isinstance(target, _AsyncEngine)
    engine = target if isinstance(target, _AsyncEngine) else engine_from_dsn(target)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise
    finally:
        if owned:
            await engine.dispose()


async def create_all(engine: AsyncEngine) -> None:
    """Create the three ML tables, and only those three.

    ``tables=`` is passed explicitly. When the base is ``aegis.data.AegisBase``, its
    metadata carries every Aegis table in the process; calling a bare ``create_all`` would
    quietly materialise the platform's whole schema from an ML pipeline — which is the
    host's job, done in the host's migration order.

    Args:
        engine: An async engine pointed at the target database.
    """
    schema = tables()
    async with engine.begin() as connection:
        await connection.run_sync(schema.base.metadata.create_all, tables=schema.all_tables)
    _LOG.info(
        "created ml_runs / ml_predictions / ml_drift_reports (metadata: %s)",
        schema.base_origin,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Writes
# ──────────────────────────────────────────────────────────────────────────────
async def insert_run(
    session: AsyncSession, entry: RegistryEntry, *, tenant_id: int | None = None
) -> str:
    """Mirror one registry entry into ``ml_runs``.

    Uses ``session.merge`` rather than ``add``: the same run is written once at training
    time and again after promotion flips its stage, and the run id is the primary key, so
    an insert would raise on the second call. Merge makes the mirror idempotent, which is
    what a mirror must be — re-running it must converge on the filesystem's truth rather
    than error.

    Args:
        session: An open async session.
        entry: The registry row to mirror.
        tenant_id: Optional tenant scope, stored as a plain indexed column.

    Returns:
        The ``run_id`` written.
    """
    schema = tables()
    result = entry.result
    detail: dict[str, Any] = {
        "training_size": result.training_size,
        "calibration_size": result.calibration_size,
        "test_size": result.test_size,
        "notes": result.notes,
        "paths": entry.paths,
        "slices": [s.model_dump(mode="json") for s in result.slices],
    }
    if result.recipe is not None:
        detail["recipe"] = result.recipe.model_dump(mode="json")
    if result.leaderboard is not None:
        detail["leaderboard"] = result.leaderboard.model_dump(mode="json")
    if entry.gate is not None:
        detail["gate"] = entry.gate.model_dump(mode="json")

    row = schema.run(
        run_id=entry.run_id,
        domain_id=entry.domain_id,
        tenant_id=tenant_id,
        created_at=_parse_iso(entry.created_at),
        stage=entry.stage,
        task=result.task,
        target=result.target,
        metric_name=result.metric_name,
        metric_value=float(result.metric_value),
        requested_coverage=float(result.requested_coverage),
        empirical_coverage=(
            None if result.empirical_coverage is None else float(result.empirical_coverage)
        ),
        dataset_digest=result.dataset_digest,
        detail=detail,
    )
    await session.merge(row)
    return entry.run_id


async def insert_prediction(
    session: AsyncSession,
    *,
    run_id: str,
    feature_digest: str,
    prediction: float | None = None,
    prediction_label: str | None = None,
    interval: tuple[float, float] | None = None,
    tenant_id: int | None = None,
    ts: datetime | None = None,
    detail: dict[str, Any] | None = None,
) -> int:
    """Append one prediction row.

    Args:
        session: An open async session.
        run_id: The model that produced the prediction.
        feature_digest: SHA-256 of the feature vector — see
            :func:`aegis_ml.monitor.log.feature_digest` for the canonicalisation.
        prediction: Numeric prediction, for regression.
        prediction_label: Class label, for classification.
        interval: ``(low, high)`` conformal bounds, when the spine produced them.
        tenant_id: Optional tenant scope.
        ts: Explicit timestamp; ``None`` lets the database stamp it, which is what keeps
            ordering coherent when several workers log concurrently.
        detail: Anything else worth keeping — raw features when the caller opted in,
            class probabilities, latency.

    Returns:
        The generated row id.
    """
    schema = tables()
    row = schema.prediction(
        tenant_id=tenant_id,
        run_id=run_id,
        feature_digest=feature_digest,
        prediction=None if prediction is None else float(prediction),
        prediction_label=prediction_label,
        interval_low=None if interval is None else float(interval[0]),
        interval_high=None if interval is None else float(interval[1]),
        detail=detail or {},
    )
    if ts is not None:
        row.ts = ts
    session.add(row)
    await session.flush()
    return int(row.id)


async def insert_drift_report(
    session: AsyncSession,
    report: DriftReport,
    *,
    tenant_id: int | None = None,
    detail: dict[str, Any] | None = None,
) -> int:
    """Append one drift evaluation.

    Args:
        session: An open async session.
        report: The measured report.
        tenant_id: Optional tenant scope.
        detail: Extra payload merged over the report's own fields — per-feature scores,
            the stat test used, the HTML path.

    Returns:
        The generated row id.
    """
    schema = tables()
    payload: dict[str, Any] = report.model_dump(mode="json")
    payload.update(detail or {})
    row = schema.drift(
        tenant_id=tenant_id,
        run_id=report.run_id,
        drifted_share=float(report.drifted_share),
        dataset_drift=bool(report.dataset_drift),
        verdict=report.verdict,
        detail=payload,
    )
    session.add(row)
    await session.flush()
    return int(row.id)


# ──────────────────────────────────────────────────────────────────────────────
# Reads
# ──────────────────────────────────────────────────────────────────────────────
async def recent_runs(
    session: AsyncSession,
    *,
    domain_id: str | None = None,
    stage: str | None = None,
    tenant_id: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return recent ``ml_runs`` rows as plain dicts, newest first.

    Dicts rather than ORM instances: these cross into JSON responses and model cards, and
    a detached ORM instance that lazy-loads after its session closed is a class of bug
    this mirror has no reason to invite.
    """
    schema = tables()
    from sqlalchemy import select

    statement = select(schema.run).order_by(schema.run.created_at.desc()).limit(limit)
    if domain_id is not None:
        statement = statement.where(schema.run.domain_id == domain_id)
    if stage is not None:
        statement = statement.where(schema.run.stage == stage)
    if tenant_id is not None:
        statement = statement.where(schema.run.tenant_id == tenant_id)
    rows = (await session.execute(statement)).scalars().all()
    return [_as_dict(row) for row in rows]


async def recent_predictions(
    session: AsyncSession,
    *,
    run_id: str | None = None,
    since: datetime | None = None,
    tenant_id: int | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Return recent ``ml_predictions`` rows as plain dicts, newest first."""
    schema = tables()
    from sqlalchemy import select

    statement = select(schema.prediction).order_by(schema.prediction.ts.desc()).limit(limit)
    if run_id is not None:
        statement = statement.where(schema.prediction.run_id == run_id)
    if since is not None:
        statement = statement.where(schema.prediction.ts >= since)
    if tenant_id is not None:
        statement = statement.where(schema.prediction.tenant_id == tenant_id)
    rows = (await session.execute(statement)).scalars().all()
    return [_as_dict(row) for row in rows]


async def latest_drift(
    session: AsyncSession, run_id: str, *, tenant_id: int | None = None
) -> dict[str, Any] | None:
    """Return the newest drift report for one run, or ``None`` if there is none."""
    schema = tables()
    from sqlalchemy import select

    statement = (
        select(schema.drift)
        .where(schema.drift.run_id == run_id)
        .order_by(schema.drift.ts.desc())
        .limit(1)
    )
    if tenant_id is not None:
        statement = statement.where(schema.drift.tenant_id == tenant_id)
    row = (await session.execute(statement)).scalars().first()
    return None if row is None else _as_dict(row)


def _as_dict(row: Any) -> dict[str, Any]:  # noqa: ANN401 - any mapped ORM instance
    """Flatten one ORM row into JSON-friendly primitives."""
    columns: Sequence[Any] = row.__table__.columns
    out: dict[str, Any] = {}
    for column in columns:
        value = getattr(row, column.name)
        out[column.name] = value.isoformat() if isinstance(value, datetime) else value
    return out


def _parse_iso(text: str) -> datetime:
    """Parse an ISO-8601 timestamp, forcing UTC when the string carries no offset.

    ``RegistryEntry.created_at`` is documented as ISO-8601 UTC, but a naive string reaches
    a ``TIMESTAMP WITH TIME ZONE`` column as local time, which shifts a run's history by
    whatever the host's offset happens to be. Assuming UTC here matches what every writer
    in this package actually stamps (``datetime.now(UTC)``).
    """
    if not text:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
