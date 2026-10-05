"""Tests for unified backtest engine on synthetic data."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.backtest_engine import (
    _index_loc_to_int,
    evaluate_comparator,
    evaluate_rule_group,
    run_backtest,
)
from src.analysis.backtest_common import period_bounds
from src.analysis.strategy_presets import get_preset
from src.analysis.strategy_spec import Rule, RuleGroup, RuleTarget


def test_index_loc_to_int_handles_slice_and_array():
    assert _index_loc_to_int(7) == 7
    assert _index_loc_to_int(slice(3, 8)) == 3
    assert _index_loc_to_int(np.array([4, 5, 6])) == 4
    assert _index_loc_to_int([9, 10]) == 9


def test_index_loc_to_int_from_duplicate_index():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03"])
    loc = idx.get_loc(pd.Timestamp("2024-01-02"))
    assert isinstance(loc, slice)
    assert _index_loc_to_int(loc) == 0


def test_run_backtest_raises_when_cancelled_before_start(tmp_path):
    from src.analysis.backtest_engine import BacktestCancelled, run_backtest
    import threading

    db = tmp_path / "t.db"
    _seed_db(str(db))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(BacktestCancelled):
        run_backtest(
            get_preset("golden_cross"),
            db_path=str(db),
            tickers=["TEST"],
            lookback_days=300,
            cancel_event=cancel,
        )


def test_run_backtest_cancel_during_progress(tmp_path):
    from src.analysis.backtest_engine import BacktestCancelled, run_backtest
    import threading

    db = tmp_path / "t.db"
    _seed_db(str(db), days=400)
    # Seed a second ticker so the loop has multiple iterations.
    _seed_db(str(db), ticker="TEST2", days=400)
    cancel = threading.Event()

    def _prog(p: float) -> None:
        if p >= 0.3:
            cancel.set()

    with pytest.raises(BacktestCancelled):
        run_backtest(
            get_preset("golden_cross"),
            db_path=str(db),
            tickers=["TEST", "TEST2"],
            lookback_days=300,
            progress_callback=_prog,
            cancel_event=cancel,
        )


def _seed_db(path: str, ticker: str = "TEST", days: int = 400) -> None:
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    trend = np.linspace(100, 150, len(dates)) + np.random.default_rng(42).normal(0, 1, len(dates))
    df = pd.DataFrame(
        {
            "Ticker": ticker,
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": trend,
            "High": trend + 1,
            "Low": trend - 1,
            "Close": trend,
            "Adj Close": trend,
            "Volume": 1_000_000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "Capital Gains": 0.0,
        }
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_history (
                Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL,
                Close REAL, "Adj Close" REAL, Volume REAL,
                Dividends REAL, "Stock Splits" REAL, "Capital Gains" REAL
            )
            """
        )
        df.to_sql("stock_history", conn, if_exists="append", index=False)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.db")
        _seed_db(path)
        yield path


def test_golden_cross_preset_uses_level_not_edge():
    spec = get_preset("golden_cross")
    assert spec.entry.rules[0].comparator == "above"
    assert spec.exit_rules.rules[0].comparator == "below"


def test_rsi_dip_preset_uses_level_not_edge():
    spec = get_preset("rsi_dip")
    assert spec.entry.rules[0].comparator == "below"
    assert spec.exit_rules.rules[0].comparator == "above"
    assert spec.entry.rules[0].target.value == 30.0
    assert spec.exit_rules.rules[0].target.value == 50.0


def test_trend_200_preset_uses_level_not_edge():
    spec = get_preset("trend_200")
    assert spec.entry.rules[0].comparator == "above"
    assert spec.exit_rules.rules[0].comparator == "below"


def test_golden_cross_already_in_regime_enters(temp_db):
    """When SMA50 > SMA200 for the whole window, strategy should not stay flat."""
    spec = get_preset("golden_cross", fast=5, slow=20)
    result = run_backtest(
        spec,
        db_path=temp_db,
        tickers=["TEST"],
        lookback_days=300,
        initial_capital=10000.0,
    )
    assert result is not None
    assert result.metrics.total_return_pct != 0.0
    assert len(result.trades) >= 1


def test_warmup_keeps_slow_sma_valid_early_in_window(temp_db):
    from src.analysis.backtest_engine import _slice_with_warmup, _spec_warmup_bars
    from src.analysis.db import load_entire_database

    raw = load_entire_database(temp_db, full_ohlcv=True, tickers=["TEST"])
    assert raw is not None and not raw.empty
    raw = raw[raw["Ticker"] == "TEST"].sort_index()
    p_start, p_end = period_bounds(raw, 120)
    warm = _slice_with_warmup(raw, p_start, p_end, _spec_warmup_bars(get_preset("golden_cross", fast=5, slow=20)))
    assert len(warm) > len(raw.loc[p_start:p_end])
    sma = warm["Adj Close"].rolling(20).mean()
    assert sma.loc[p_start:].notna().all()


def test_rsi_dip_enters_when_window_starts_oversold(tmp_path):
    """Level entry must catch an already-oversold start (crosses would miss)."""
    import sqlite3

    path = str(tmp_path / "rsi.db")
    dates = pd.date_range("2023-01-01", periods=400, freq="B")
    prices = np.ones(400, dtype=float) * 100.0
    prices[200:230] = np.linspace(100, 40, 30)
    prices[230:320] = np.linspace(40, 38, 90)
    prices[320:] = np.linspace(38, 80, 80)
    df = pd.DataFrame(
        {
            "Ticker": "TEST",
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": prices,
            "High": prices + 1,
            "Low": prices - 1,
            "Close": prices,
            "Adj Close": prices,
            "Volume": 1_000_000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "Capital Gains": 0.0,
        }
    )
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE stock_history (
                Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL,
                Close REAL, "Adj Close" REAL, Volume REAL,
                Dividends REAL, "Stock Splits" REAL, "Capital Gains" REAL
            )
            """
        )
        df.to_sql("stock_history", conn, if_exists="append", index=False)

    result = run_backtest(
        get_preset("rsi_dip"),
        db_path=path,
        tickers=["TEST"],
        lookback_days=150,
        initial_capital=10000.0,
    )
    assert result is not None
    assert len(result.trades) >= 1


def test_crosses_above_semantics():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    left = pd.Series([1, 2, 3, 2, 1], index=idx, dtype=float)
    right = pd.Series([2, 2, 2, 2, 2], index=idx, dtype=float)
    crosses = evaluate_comparator(left, right, "crosses_above")
    assert bool(crosses.iloc[2]) is True
    assert bool(crosses.iloc[1]) is False


def test_run_trend_backtest(temp_db):
    spec = get_preset("trend_200", period=50)
    result = run_backtest(
        spec,
        db_path=temp_db,
        tickers=["TEST"],
        lookback_days=300,
        initial_capital=10000.0,
    )
    assert result is not None
    assert len(result.equity_dates) > 0
    assert result.metrics.period_days > 0


def test_buy_hold_engine(temp_db):
    spec = get_preset("buy_hold")
    result = run_backtest(
        spec,
        db_path=temp_db,
        tickers=["TEST"],
        lookback_days=200,
        initial_capital=5000.0,
    )
    assert result is not None
    assert result.metrics.total_return_pct != 0 or len(result.equity_values) > 1


def test_rule_group_and_logic():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = np.linspace(100, 120, len(idx))
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1,
            "Low": prices - 1,
            "Adj Close": prices,
            "Volume": 1e6,
        },
        index=idx,
    )
    group = RuleGroup(
        logic="and",
        rules=[
            Rule(
                indicator="PRICE",
                comparator="above",
                target=RuleTarget(kind="value", value=50.0),
            )
        ],
    )
    mask = evaluate_rule_group(group, df)
    assert mask.all()
