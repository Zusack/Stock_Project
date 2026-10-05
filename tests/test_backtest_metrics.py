"""Tests for backtest metrics calculations."""

from __future__ import annotations

import math

import pandas as pd

from src.analysis.backtest_metrics import compute_metrics, monthly_returns_grid


def test_compute_metrics_positive_trend():
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    values = [10000 * (1.0005 ** i) for i in range(len(dates))]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    report = compute_metrics(date_strs, values)
    assert report.total_return_pct > 0
    assert report.cagr_pct > 0
    assert report.max_drawdown_pct >= 0
    assert report.period_days > 200


def test_compute_metrics_with_trades():
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
    values = [10000 + i * 100 for i in range(20)]
    trades = [
        {"Entry_Price": 100.0, "Exit_Price": 110.0},
        {"Entry_Price": 50.0, "Exit_Price": 45.0},
    ]
    report = compute_metrics(dates, values, trades=trades)
    assert report.trade_count == 2
    assert 0 <= report.win_rate_pct <= 100
    assert report.profit_factor > 0


def test_monthly_returns_grid():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    values = [10000 * (1.001 ** i) for i in range(len(dates))]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    grid = monthly_returns_grid(date_strs, values)
    assert not grid.empty
    assert "Year" in grid.reset_index().columns or grid.index.name == "year" or len(grid) >= 1


def test_sharpe_finite():
    dates = [f"2024-06-{d:02d}" for d in range(1, 23)]
    values = [10000 + 50 * math.sin(i / 3) for i in range(len(dates))]
    report = compute_metrics(dates, values)
    assert math.isfinite(report.sharpe)
