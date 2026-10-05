"""Tests for CANSLIM market-direction SMA logic."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.analysis.canslim_core import (
    _format_canslim_lines,
    evaluate_risk_flags,
    market_direction_line,
    price_vs_sma50,
)


def _sample_index_df(*, with_gap_nan: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    close = pd.Series(range(100, 180), index=dates, dtype=float)
    if with_gap_nan:
        close.iloc[60] = float("nan")
        close.iloc[70] = float("nan")
    return pd.DataFrame({"Adj Close": close})


def test_price_vs_sma50_uses_ffill_like_dashboard():
    values = price_vs_sma50(_sample_index_df(with_gap_nan=True))
    assert values is not None
    price, sma = values
    assert price > sma


def test_price_vs_sma50_without_ffill_would_fail_on_gap_nan():
    df = _sample_index_df(with_gap_nan=True)
    raw = df["Adj Close"].rolling(50).mean().iloc[-1]
    assert pd.isna(raw)


def test_market_direction_line_names_index_and_passes():
    assert "PASS" in market_direction_line("^IXIC", True)
    assert "Nasdaq" in market_direction_line("^IXIC", True)
    assert "FAIL" in market_direction_line("^IXIC", False)


def test_evaluate_risk_flags_market_weak_only_when_pass_m_false():
    assert evaluate_risk_flags(price=100, stock_sma50=90, pass_m=False, near_high=0.9) == "Market weak"
    assert evaluate_risk_flags(price=100, stock_sma50=90, pass_m=True, near_high=0.9) == ""


def test_format_canslim_lines_includes_market_direction():
    last = pd.Series(
        {
            "Pass_C": True,
            "Pass_A": False,
            "Pass_N": True,
            "Pass_S": False,
            "Pass_L": True,
            "Pass_M": True,
            "High_52": 110.0,
        }
    )
    lines = _format_canslim_lines(
        last,
        market_ticker="^IXIC",
        meta={"C": "quarterly_eps", "A": "proxy_1y_return"},
        price=100.0,
        near_high=0.91,
        stock_ret=0.12,
        mkt_ret=0.05,
        vol_ratio=1.2,
    )
    assert any(line.startswith("[M]") for line in lines)
    assert any(line.startswith("[C]") and "PASS" in line for line in lines)
