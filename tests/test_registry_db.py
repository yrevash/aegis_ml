"""The optional SQL mirror of the registry, exercised against a real aiosqlite database.

``registry/db.py`` needs ``sqlalchemy[asyncio]`` and a driver. Both are guarded with
``importorskip``: the filesystem registry is the source of truth and this mirror is
optional, so a venv without them must skip rather than fail.

``asyncio_mode = "auto"`` is configured, so the ``async def`` tests need no decorator.
"""

from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy", reason="registry.db needs sqlalchemy[asyncio]")
pytest.importorskip("aiosqlite", reason="registry.db needs an async driver")

from aegis_ml.contracts.protocols import DriftReport  # noqa: E402
from aegis_ml.registry import db  # noqa: E402
from tests.fixtures.builders import registry_entry, train_result  # noqa: E402


@pytest.fixture
async def engine(tmp_path):
    """A real, disposable aiosqlite engine with the ML tables created."""
    created = db.engine_from_dsn(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    await db.create_all(created)
    try:
        yield created
    finally:
        await created.dispose()


def test_async_dsn_rejects_a_sync_driver_or_upgrades_it() -> None:
    """A sync DSN handed to an async engine deadlocks; it must be converted or refused."""
    converted = db.async_dsn("postgresql://user:pw@localhost/db")
    assert "+asyncpg" in converted or "+psycopg" in converted, converted


def test_tables_are_declared_once_and_reused() -> None:
    """A second call must not redefine the metadata — SQLAlchemy raises on a duplicate table."""
    assert db.tables() is db.tables()


async def test_run_round_trips_through_the_database(engine) -> None:
    """Insert a registry row, read it back through ``recent_runs``."""
    result = train_result("run-db-1", 0.66)
    entry = registry_entry(result, stage="production")

    async with db.session_scope(engine) as session:
        run_id = await db.insert_run(session, entry)

    assert run_id == "run-db-1"

    async with db.session_scope(engine) as session:
        rows = await db.recent_runs(session, domain_id=entry.domain_id)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-db-1"
    assert rows[0]["stage"] == "production"
    assert rows[0]["metric_value"] == pytest.approx(0.66)


async def test_recent_runs_filters_by_stage(engine) -> None:
    """The filter the champion lookup depends on."""
    async with db.session_scope(engine) as session:
        await db.insert_run(session, registry_entry(train_result("staged", 0.5), stage="staging"))
        await db.insert_run(
            session, registry_entry(train_result("live", 0.7), stage="production")
        )

    async with db.session_scope(engine) as session:
        production = await db.recent_runs(session, stage="production")

    assert [row["run_id"] for row in production] == ["live"]


async def test_predictions_are_logged_with_their_interval(engine) -> None:
    """A prediction row without its interval is a number with no confidence attached."""
    async with db.session_scope(engine) as session:
        await db.insert_run(session, registry_entry(train_result("run-db-2", 0.6)))
        await db.insert_prediction(
            session,
            run_id="run-db-2",
            feature_digest="deadbeef",
            prediction=42.5,
            interval=(30.0, 55.0),
        )

    async with db.session_scope(engine) as session:
        rows = await db.recent_predictions(session, run_id="run-db-2")

    assert len(rows) == 1
    assert rows[0]["prediction"] == pytest.approx(42.5)
    assert rows[0]["interval_low"] == pytest.approx(30.0)
    assert rows[0]["interval_high"] == pytest.approx(55.0)


async def test_drift_report_round_trips_and_latest_wins(engine) -> None:
    """``latest_drift`` must return the newest report, not an arbitrary one."""
    async with db.session_scope(engine) as session:
        await db.insert_run(session, registry_entry(train_result("run-db-3", 0.6)))
        await db.insert_drift_report(
            session,
            DriftReport(run_id="run-db-3", drifted_share=0.1, verdict="pass", n_current_rows=100),
        )
        await db.insert_drift_report(
            session,
            DriftReport(run_id="run-db-3", drifted_share=0.5, verdict="block", n_current_rows=100),
        )

    async with db.session_scope(engine) as session:
        latest = await db.latest_drift(session, "run-db-3")

    assert latest is not None
    assert latest["verdict"] == "block"
    assert latest["drifted_share"] == pytest.approx(0.5)


async def test_latest_drift_is_none_when_nothing_was_measured(engine) -> None:
    """Absence is reported as absence, never as a zero that reads like "no drift"."""
    async with db.session_scope(engine) as session:
        assert await db.latest_drift(session, "never-measured") is None
