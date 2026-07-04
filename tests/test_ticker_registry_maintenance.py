"""Watchlist maintenance helpers and coverage label formatting."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.analysis import ticker_registry as registry
from src.analysis.history_coverage import init_coverage_schema
from src.analysis.ingest import init_db
from src.views.data_ingest_view import format_coverage_label, format_coverage_tooltip


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    init_coverage_schema(path)
    return path


def _insert_row(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    has_history: int = 0,
    validation_status: str = "ok",
    last_ingest_status: str | None = None,
    skip_ingest: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO watchlist_tickers
        (symbol, added_at, skip_ingest, has_history, validation_status, last_ingest_status)
        VALUES (?, '2026-01-01', ?, ?, ?, ?)
    """,
        (symbol, skip_ingest, has_history, validation_status, last_ingest_status),
    )


def test_is_dead_symbol_predicate():
    dead = registry.WatchlistRow(
        symbol="DEAD",
        added_at=None,
        skip_ingest=False,
        last_ingest_at=None,
        last_ingest_status="No Price Data",
        has_history=False,
        validation_status="ok",
        validation_reason=None,
        last_validated_at=None,
    )
    assert registry.is_dead_symbol(dead)
    ok = registry.WatchlistRow(
        symbol="OK",
        added_at=None,
        skip_ingest=False,
        last_ingest_at=None,
        last_ingest_status="Success",
        has_history=True,
        validation_status="ok",
        validation_reason=None,
        last_validated_at=None,
    )
    assert not registry.is_dead_symbol(ok)


def test_list_dead_symbols_split_by_history(db_path: str):
    with sqlite3.connect(db_path) as conn:
        _insert_row(conn, "REM", has_history=0, last_ingest_status="No Price Data")
        _insert_row(conn, "HIST", has_history=1, validation_status="delisted")
        _insert_row(conn, "OK", has_history=1, last_ingest_status="Success")
        conn.commit()

    removable = registry.list_dead_symbols(db_path, has_history=False, limit=None)
    with_hist = registry.list_dead_symbols(db_path, has_history=True, limit=None)
    assert [r.symbol for r in removable] == ["REM"]
    assert [r.symbol for r in with_hist] == ["HIST"]


def test_count_summary_includes_removable(db_path: str):
    with sqlite3.connect(db_path) as conn:
        _insert_row(conn, "REM", has_history=0, validation_status="invalid")
        conn.commit()
    summary = registry.count_summary(db_path)
    assert summary["removable"] >= 1


def test_set_skip_ingest_preserves_validation_status(db_path: str):
    with sqlite3.connect(db_path) as conn:
        _insert_row(conn, "X", validation_status="delisted")
        conn.commit()
    registry.set_skip_ingest(db_path, "X", True, archive_reason="Paused by user")
    row = registry.get_symbol(db_path, "X")
    assert row is not None
    assert row.skip_ingest
    assert row.validation_status == "delisted"
    assert row.archive_reason == "Paused by user"


def test_format_coverage_gaps_days_label():
    row = registry.WatchlistRow(
        symbol="G",
        added_at=None,
        skip_ingest=False,
        last_ingest_at=None,
        last_ingest_status=None,
        has_history=True,
        validation_status="ok",
        validation_reason=None,
        last_validated_at=None,
        coverage_status="gaps",
        gap_count=3,
        gap_summary="2026-05-12..2026-05-14",
    )
    assert format_coverage_label(row) == "gaps (3 days)"
    tip = format_coverage_tooltip(row)
    assert "90-day" in tip
    assert "2026-05-12" in tip


def test_investigation_classifies_transient_error(monkeypatch):
    from src.analysis import ticker_investigation as inv_mod

    def fake_yahoo(_ticker: str):
        return {
            "history_rows": 0,
            "quote_type": None,
            "error": "429 Too Many Requests",
        }

    monkeypatch.setattr(inv_mod, "_fetch_yahoo_signals", fake_yahoo)
    monkeypatch.setattr(inv_mod, "_scan_yahoo_news", lambda _t: (None, []))
    result = inv_mod.investigate_symbol("FOO", None, use_sec=False)
    assert result.outcome == "transient_error"
