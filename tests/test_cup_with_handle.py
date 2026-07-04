"""Tests for cup-with-handle pattern detection."""

import pandas as pd

from src.analysis.patterns.cup_with_handle import detect_cup_with_handle_series, scan_cup_with_handle


def test_scan_returns_none_on_short_data():
    df = pd.DataFrame(
        {"High": [1.0], "Low": [0.9], "Close": [1.0], "Volume": [100]},
        index=pd.date_range("2024-01-01", periods=1, freq="B"),
    )
    assert scan_cup_with_handle(df, 0) is None


def test_detect_series_columns(sample_ohlcv):
    out = detect_cup_with_handle_series(sample_ohlcv, min_warmup=60)
    assert "Pass_Pattern" in out.columns
    assert "Pivot" in out.columns
    assert "Pattern_Quality" in out.columns
    assert len(out) == len(sample_ohlcv)
