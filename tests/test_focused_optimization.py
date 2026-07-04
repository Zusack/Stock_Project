"""Tests for focused momentum optimization."""

from __future__ import annotations

import sqlite3
import threading
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis import focused as focused_mod
from src.analysis.focused import (
    backtest_momentum_strategy,
    optimize_single_ticker,
    run_focused_optimization,
)


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "opt.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE stock_history (
            Ticker TEXT, Date TEXT, "Adj Close" REAL, Volume REAL
        )
        """
    )
    start = date(2020, 1, 1)
    tickers = ["AAA", "BBB", "CCC"]
    for sym in tickers:
        for i in range(120):
            d = (start + timedelta(days=i)).isoformat()
            price = 100.0 + i * 0.5 + (hash(sym) % 7)
            conn.execute(
                'INSERT INTO stock_history VALUES (?, ?, ?, ?)',
                (sym, d, price, 1_000_000),
            )
    conn.commit()
    conn.close()
    return path


def _make_momentum_df(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"Adj Close": prices}, index=dates)


def test_backtest_momentum_strategy_known_series():
    prices = [100.0] * 15 + [110.0] * 15 + [105.0] * 10
    df = _make_momentum_df(prices)
    ret = backtest_momentum_strategy(df, window_days=10, buy_threshold=0.02, sell_threshold=0.0)
    assert isinstance(ret, float)
    assert ret > 0.0


def test_optimize_single_ticker_returns_params():
    prices = np.linspace(100.0, 150.0, 80)
    task = ("TST", prices, [10, 20], [0.05], [0.0])
    result = optimize_single_ticker(task)
    assert result["Ticker"] == "TST"
    assert result["Baseline"] > 1.0
    assert result["Params"] is not None


def test_run_focused_progress_callback_phases(db_path):
    events: list[tuple[str, float, int, int]] = []

    def on_progress(phase, pct, done, total, detail):
        events.append((phase, pct, done, total))

    res = run_focused_optimization(
        db_path,
        target_tickers=["AAA"],
        use_parallel=False,
        windows=[10],
        buy_thresholds=[0.05],
        sell_thresholds=[0.0],
        progress_callback=on_progress,
    )
    assert res is not None
    assert res.stock_count == 1
    phases = [e[0] for e in events]
    assert "loading" in phases
    assert "optimizing" in phases
    assert "finalizing" in phases
    assert events[-1][1] == 1.0


def test_run_focused_legacy_float_progress_callback(db_path):
    seen: list[float] = []

    def on_progress(pct: float) -> None:
        seen.append(pct)

    run_focused_optimization(
        db_path,
        target_tickers=["AAA"],
        use_parallel=False,
        windows=[10],
        buy_thresholds=[0.05],
        sell_thresholds=[0.0],
        progress_callback=on_progress,
    )
    assert seen
    assert seen[-1] == 1.0


def test_run_focused_result_callback_streams_rows(db_path):
    rows: list[dict] = []

    def on_result(row: dict) -> None:
        rows.append(row)

    run_focused_optimization(
        db_path,
        target_tickers=["AAA", "BBB"],
        use_parallel=False,
        windows=[10],
        buy_thresholds=[0.05],
        sell_thresholds=[0.0],
        result_callback=on_result,
    )
    assert len(rows) == 2
    assert all("Alpha" in r for r in rows)


def test_run_focused_cancel_returns_partial(db_path):
    cancel = threading.Event()
    seen = 0

    def on_result(_row: dict) -> None:
        nonlocal seen
        seen += 1
        if seen >= 1:
            cancel.set()

    res = run_focused_optimization(
        db_path,
        use_parallel=False,
        windows=[10],
        buy_thresholds=[0.05],
        sell_thresholds=[0.0],
        result_callback=on_result,
        cancel_event=cancel,
    )
    assert res is not None
    assert res.cancelled
    assert 1 <= res.stock_count < 3


def test_filtered_load_passes_tickers_to_db(monkeypatch, db_path):
    from src.analysis import db as db_mod

    captured: dict = {}

    def fake_load(db_path_arg, tickers=None, **kwargs):
        captured["tickers"] = tickers
        return db_mod.load_entire_database(db_path_arg, tickers=tickers)

    monkeypatch.setattr(focused_mod, "load_entire_database", fake_load)
    run_focused_optimization(
        db_path,
        target_tickers=["AAA", "BBB"],
        use_parallel=False,
        windows=[10],
        buy_thresholds=[0.05],
        sell_thresholds=[0.0],
    )
    assert captured.get("tickers") == ["AAA", "BBB"]
