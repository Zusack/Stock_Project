"""Tests for DB performance helpers and registry batch operations."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.analysis import ticker_registry as registry
from src.analysis.db_perf import check_sla, is_db_locked_error
from src.analysis.ingest import init_db


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    registry.ensure_registry(db)
    return db


def test_is_db_locked_error():
    assert is_db_locked_error(sqlite3.OperationalError("database is locked"))
    assert not is_db_locked_error(sqlite3.OperationalError("no such table"))


def test_check_sla():
    assert check_sla("count_summary", 0.5)
    assert not check_sla("count_summary", 999.0)


def test_count_summary_single_query(temp_db: Path):
    registry.add_symbol(temp_db, "AAA")
    registry.add_symbol(temp_db, "BBB")
    summary = registry.count_summary(temp_db)
    assert summary["total"] >= 2
    assert "focus_cap" in summary


def test_bulk_remove_dead_no_history_batch(temp_db: Path):
    now = registry._utc_now()
    from src.analysis.db import db_connection

    with db_connection(temp_db, readonly=False) as conn:
        conn.execute(
            """
            INSERT INTO watchlist_tickers
            (symbol, added_at, skip_ingest, has_history, validation_status, last_ingest_status, pool)
            VALUES ('DEAD1', ?, 0, 0, 'delisted', 'No Price Data', 'universe')
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO watchlist_tickers
            (symbol, added_at, skip_ingest, has_history, validation_status, last_ingest_status, pool)
            VALUES ('DEAD2', ?, 0, 0, 'invalid', 'No Price Data', 'universe')
            """,
            (now,),
        )
        conn.commit()

    result = registry.bulk_remove_dead_no_history(temp_db, remove_from_csv=False)
    assert result["removed_db"] == 2
    assert registry.get_symbol(temp_db, "DEAD1") is None
    assert registry.get_symbol(temp_db, "DEAD2") is None
