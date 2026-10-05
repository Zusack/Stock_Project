"""Intraday bars schema, row persistence, and stream aggregation helpers."""

from __future__ import annotations

import pytest

from src.analysis.db import db_connection, load_intraday_bars
from src.analysis.ingest import init_db
from src.analysis.intraday_common import (
    INTRADAY_COLUMNS,
    interval_to_seconds,
    interval_to_yfinance,
)
from src.analysis.intraday_ingest import upsert_intraday_rows
from src.services.live_stream_service import LiveBarSnapshot, live_stream_service


def _sample_row(ts: str = "2026-06-10T14:31:00Z") -> tuple:
    snap = LiveBarSnapshot(
        ticker="AAPL",
        timestamp=ts,
        interval="1Min",
        open=100.0,
        high=101.0,
        low=99.5,
        close=100.5,
        volume=1200,
        vwap=None,
        trade_count=42,
    )
    return snap.to_row()


def test_interval_helpers():
    assert interval_to_seconds("1Min") == 60
    assert interval_to_seconds("5Min") == 300
    assert interval_to_seconds("1Hour") == 3600
    assert interval_to_yfinance("1Min") == "1m"
    assert interval_to_yfinance("1Hour") == "1h"


def test_snapshot_row_matches_columns():
    row = _sample_row()
    assert len(row) == len(INTRADAY_COLUMNS)
    assert row[0] == "AAPL"
    assert row[1] == "2026-06-10T14:31:00Z"
    assert row[2] == "1Min"
    assert row[6] == 100.5


def test_intraday_bars_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    assert upsert_intraday_rows(db, [_sample_row()]) == 1
    df = load_intraday_bars("AAPL", db, interval="1Min", max_points=10)
    assert df is not None
    assert len(df) == 1
    assert float(df.iloc[-1]["Close"]) == pytest.approx(100.5)

    with db_connection(db, readonly=True) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "intraday_bars" in tables


def test_resolve_stream_symbols_chart_only(monkeypatch):
    class _Cfg:
        db_path = "/unused/test.db"
        live_stream_symbols_source = "custom"

    monkeypatch.setattr(
        "src.services.live_stream_service.stock_config",
        lambda: _Cfg(),
    )
    monkeypatch.setattr(
        "src.services.live_stream_service.resolve_watchlist_symbols",
        lambda _db: [],
    )

    svc = live_stream_service()
    symbols = svc.resolve_stream_symbols(chart_symbol="aapl", extra_symbols=["msft", "AAPL"])
    assert symbols == ["AAPL", "MSFT"]


def test_resolve_stream_symbols_reserves_chart_and_extras(monkeypatch):
    """Focus list is large; chart + Also stream must still make the 30-cap."""

    class _Cfg:
        db_path = "/unused/test.db"
        live_stream_symbols_source = "focus"

    focus = [f"F{i:02d}" for i in range(40)]
    monkeypatch.setattr(
        "src.services.live_stream_service.stock_config",
        lambda: _Cfg(),
    )
    monkeypatch.setattr(
        "src.services.live_stream_service.resolve_watchlist_symbols",
        lambda _db: focus,
    )

    svc = live_stream_service()
    symbols = svc.resolve_stream_symbols(
        chart_symbol="AAPL",
        extra_symbols=["MSFT", "NVDA"],
    )
    assert symbols[:3] == ["AAPL", "MSFT", "NVDA"]
    assert len(symbols) == 30
    assert "AAPL" in symbols and "MSFT" in symbols and "NVDA" in symbols
