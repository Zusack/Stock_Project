"""Tests for smart bulk-ingest preflight (skip when no new market data)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.analysis import ticker_registry as registry
from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.history_coverage import assess_ingest_preflight, init_coverage_schema
from src.analysis.ingest import init_db
from src.analysis.market_calendar import last_completed_trading_day


def _insert_symbol(conn, symbol: str) -> None:
    conn.execute(
        """
        INSERT INTO watchlist_tickers
        (symbol, added_at, skip_ingest, has_history, validation_status, pool)
        VALUES (?, datetime('now'), 0, 1, 'ok', 'universe')
        """,
        (symbol,),
    )


def _insert_bar(conn, ticker: str, day: date) -> None:
    conn.execute(
        """
        INSERT INTO stock_history (Ticker, Date, Open, High, Low, Close, Volume)
        VALUES (?, ?, 1, 1, 1, 1, 1)
        """,
        (ticker, day.isoformat()),
    )


@pytest.fixture
def preflight_db(tmp_path):
    db_path = str(tmp_path / "preflight.db")
    init_db(db_path)
    registry.ensure_registry(db_path)
    init_coverage_schema(db_path)
    return db_path


def test_preflight_skips_when_benchmarks_current_after_holiday(preflight_db, monkeypatch):
    """Calendar may expect Friday, but benchmarks stop at Thursday (holiday)."""
    as_of = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)  # Saturday after Friday holiday
    required = last_completed_trading_day(as_of)
    assert required == date(2026, 7, 2)  # Thursday

    thursday = date(2026, 7, 2)
    with ingest_write_lock():
        with db_connection(preflight_db, readonly=False) as conn:
            for sym in ("SPY", "^GSPC", "^DJI", "AAPL", "MSFT"):
                _insert_symbol(conn, sym)
            for sym in ("SPY", "^GSPC", "^DJI", "AAPL", "MSFT"):
                _insert_bar(conn, sym, thursday)
            conn.commit()

    monkeypatch.setattr(
        "src.analysis.history_coverage.last_completed_trading_day",
        lambda _as_of=None: required,
    )

    result = assess_ingest_preflight(preflight_db, scope="universe", mode="smart")
    assert result.should_run is False
    assert "No new data available since the last ingest" in result.message
    assert result.effective_required_day == thursday


def test_preflight_runs_when_symbols_behind_market(preflight_db, monkeypatch):
    as_of = datetime(2026, 5, 20, 22, 0, tzinfo=timezone.utc)
    required = last_completed_trading_day(as_of)
    assert required == date(2026, 5, 20)

    with ingest_write_lock():
        with db_connection(preflight_db, readonly=False) as conn:
            for sym in ("SPY", "^GSPC", "LAG"):
                _insert_symbol(conn, sym)
            _insert_bar(conn, "SPY", required)
            _insert_bar(conn, "^GSPC", required)
            _insert_bar(conn, "LAG", date(2026, 5, 15))
            conn.commit()

    monkeypatch.setattr(
        "src.analysis.history_coverage.last_completed_trading_day",
        lambda _as_of=None: required,
    )

    result = assess_ingest_preflight(preflight_db, scope="universe", mode="smart")
    assert result.should_run is True
    assert result.stale_price_symbols == 1


def test_preflight_runs_for_full_mode(preflight_db, monkeypatch):
    thursday = date(2026, 7, 2)
    with ingest_write_lock():
        with db_connection(preflight_db, readonly=False) as conn:
            _insert_symbol(conn, "SPY")
            _insert_bar(conn, "SPY", thursday)
            conn.commit()

    monkeypatch.setattr(
        "src.analysis.history_coverage.last_completed_trading_day",
        lambda _as_of=None: thursday,
    )

    result = assess_ingest_preflight(preflight_db, scope="universe", mode="full")
    assert result.should_run is True
