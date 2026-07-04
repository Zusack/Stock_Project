"""Leaderboard DB snapshot persistence and freshness."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.analysis.market_calendar import last_completed_trading_day
from src.analysis.leaderboard_cache import (
    FRESH,
    INGEST_NEEDED,
    STALE_DATA,
    LeaderboardRunMeta,
    assess_leaderboard_cache_freshness,
    compute_leaderboard_data_fingerprint,
    load_latest_leaderboard,
    save_leaderboard_snapshot,
)


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE watchlist_tickers (
            symbol TEXT PRIMARY KEY,
            price_max_date TEXT,
            coverage_status TEXT
        )
        """
    )
    as_of = last_completed_trading_day().isoformat()
    conn.execute(
        "INSERT INTO watchlist_tickers VALUES (?, ?, 'current')",
        ("AAPL", as_of),
    )
    conn.execute(
        """
        CREATE TABLE stock_history (
            Ticker TEXT, Date TEXT, "Adj Close" REAL, Volume REAL
        )
        """
    )
    conn.execute(
        "INSERT INTO stock_history VALUES ('AAPL', ?, 100.0, 1e6)",
        (as_of,),
    )
    conn.commit()
    conn.close()
    ensure_intelligence_schema(path)
    return path


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "composite_score": 80.0,
                "canslim_score": 5,
                "pattern_quality": 0.5,
                "rs_pct": 10.0,
                "volume_ratio": 1.2,
                "near_high_pct": 90.0,
                "pass_setup": True,
                "pass_pattern": False,
                "risk_flag": "",
                "latest_price": 150.0,
                "sector": "Tech",
                "industry": "Hardware",
                "notes": "",
                "news_sentiment": 0.5,
                "news_tags": None,
            }
        ]
    )


def test_save_and_load_roundtrip(db_path, monkeypatch):
    monkeypatch.setattr(
        "src.analysis.leaderboard_cache.stock_config",
        lambda: type(
            "Cfg",
            (),
            {
                "last_ingest_at": "2026-05-26 12:00 UTC",
                "last_universe_ingest_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "universe_ingest_interval_days": 7,
                "market_ticker": "^GSPC",
            },
        )(),
    )
    fp, as_of = compute_leaderboard_data_fingerprint(
        db_path, universe_key="focus", tickers=["AAPL"], market_ticker="^GSPC"
    )
    meta = LeaderboardRunMeta(
        scored_at=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
        universe_key="focus",
        universe_label="1 focus symbol(s)",
        market_ticker="^GSPC",
        full_universe=False,
        symbol_count=1,
        elapsed_sec=1.5,
        data_fingerprint=fp,
        data_as_of=as_of,
        tickers_json='["AAPL"]',
    )
    save_leaderboard_snapshot(db_path, _sample_df(), meta=meta)
    loaded = load_latest_leaderboard(db_path)
    assert loaded is not None
    assert len(loaded.df) == 1
    assert loaded.df.iloc[0]["ticker"] == "AAPL"
    assert loaded.meta.universe_key == "focus"
    assert loaded.meta.data_fingerprint == fp


def test_max_price_date_uses_stock_history_not_stale_metadata(db_path, monkeypatch):
    """price_max_date on watchlist_tickers can lag; fingerprint must follow stock_history."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE watchlist_tickers SET price_max_date = '2026-05-10' WHERE symbol = 'AAPL'"
    )
    conn.execute(
        "UPDATE stock_history SET Date = '2026-05-20' WHERE Ticker = 'AAPL'"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "src.analysis.leaderboard_cache.stock_config",
        lambda: type(
            "Cfg",
            (),
            {
                "last_ingest_at": "2026-05-26 12:00 UTC",
                "last_universe_ingest_at": "2026-05-26",
                "universe_ingest_interval_days": 7,
                "market_ticker": "^GSPC",
            },
        )(),
    )
    _fp, as_of = compute_leaderboard_data_fingerprint(
        db_path, universe_key="focus", tickers=["AAPL"], market_ticker="^GSPC"
    )
    assert as_of == "2026-05-20"


def test_freshness_stale_after_fingerprint_change(db_path, monkeypatch):
    cfg = type(
        "Cfg",
        (),
        {
            "last_ingest_at": "2026-05-26 12:00 UTC",
            "last_universe_ingest_at": "2026-05-20",
            "universe_ingest_interval_days": 7,
            "market_ticker": "^GSPC",
        },
    )()
    monkeypatch.setattr("src.analysis.leaderboard_cache.stock_config", lambda: cfg)

    fp, as_of = compute_leaderboard_data_fingerprint(
        db_path, universe_key="focus", tickers=["AAPL"], market_ticker="^GSPC"
    )
    meta = LeaderboardRunMeta(
        scored_at=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
        universe_key="focus",
        universe_label="test",
        market_ticker="^GSPC",
        full_universe=False,
        symbol_count=1,
        elapsed_sec=1.0,
        data_fingerprint=fp,
        data_as_of=as_of,
        tickers_json='["AAPL"]',
    )
    fresh = assess_leaderboard_cache_freshness(
        db_path, meta, current_universe_key="focus", tickers=["AAPL"]
    )
    assert fresh.state == FRESH

    cfg.last_ingest_at = "2026-05-27 08:00 UTC"
    stale = assess_leaderboard_cache_freshness(
        db_path, meta, current_universe_key="focus", tickers=["AAPL"]
    )
    assert stale.state == STALE_DATA
