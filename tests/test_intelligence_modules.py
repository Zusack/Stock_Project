"""Tests for market intelligence modules."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from src.analysis.data_quality import classify_ingest_error, record_ingest_event
from src.analysis.guidance import RecommendationBand, _assign_band, _build_confidence
from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.analysis.leaderboard import LeaderboardRow
from src.analysis.market_context import regime_alignment_score
from src.analysis.signal_quality import (
    BacktestCostModel,
    adjust_trade_return,
    summarize_trades,
)


def _temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    Path(path).unlink(missing_ok=True)
    from src.analysis.ingest import init_db

    init_db(path)
    return path


def test_classify_ingest_error():
    assert classify_ingest_error("Success") == "ok"
    assert classify_ingest_error("Price Error: Invalid comparison") == "datetime_compare"
    assert classify_ingest_error("Rate Limit (Skipped)") == "rate_limit"


def test_intelligence_schema_creates_tables():
    db = _temp_db()
    ensure_intelligence_schema(db)
    with sqlite3.connect(db) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "guidance_snapshots" in tables
    assert "alert_events" in tables


def test_record_ingest_event():
    db = _temp_db()
    record_ingest_event(db, "AAPL", phase="test", status="Success", price_rows=10)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT error_category, price_rows FROM data_quality_events WHERE symbol='AAPL'"
        ).fetchone()
    assert row[0] == "ok"
    assert row[1] == 10


def test_assign_band_high_priority():
    lb = LeaderboardRow(
        ticker="AAPL",
        composite_score=80,
        canslim_score=6,
        pattern_quality=0.8,
        rs_pct=12,
        volume_ratio=1.5,
        near_high_pct=92,
        pass_setup=True,
        pass_pattern=True,
        risk_flag="",
        latest_price=150,
        sector="Technology",
        notes="",
    )
    band = _assign_band(lb, news_sent=0.6, regime_align=0.7, market_regime="risk_on")
    assert band == RecommendationBand.HIGH_PRIORITY


def test_assign_band_exit_watch():
    lb = LeaderboardRow(
        ticker="XYZ",
        composite_score=40,
        canslim_score=2,
        pattern_quality=0.1,
        rs_pct=-5,
        volume_ratio=0.8,
        near_high_pct=50,
        pass_setup=False,
        pass_pattern=False,
        risk_flag="Below 50-day simple moving average",
        latest_price=10,
        sector="",
        notes="",
    )
    band = _assign_band(lb, news_sent=0.5, regime_align=0.3, market_regime="neutral")
    assert band == RecommendationBand.EXIT_WATCH


def test_regime_alignment():
    assert regime_alignment_score("risk_on", pass_setup=True, risk_flag="") > 0.7
    assert regime_alignment_score(
        "risk_off", pass_setup=True, risk_flag="Below 50-day simple moving average"
    ) < 0.5


def test_backtest_cost_model():
    cost = BacktestCostModel(slippage_bps=10, spread_bps=5, fee_per_trade=1.0)
    ret = adjust_trade_return(100.0, 110.0, cost=cost)
    assert ret < 0.10


def test_summarize_trades_empty():
    m = summarize_trades([])
    assert m["sample_count"] == 0


def test_summarize_trades_basic():
    trades = [
        {"Entry_Price": 100, "Exit_Price": 110},
        {"Entry_Price": 50, "Exit_Price": 45},
    ]
    m = summarize_trades(trades, cost=BacktestCostModel(slippage_bps=0, spread_bps=0))
    assert m["sample_count"] == 2
    assert 0 <= m["win_rate"] <= 1
