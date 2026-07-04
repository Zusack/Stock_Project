"""Cup-with-handle array path matches DataFrame API."""

import numpy as np
import pandas as pd

from src.analysis.patterns.cup_with_handle import (
    detect_cup_with_handle_series,
    scan_cup_with_handle,
    scan_cup_with_handle_arrays,
)


def test_scan_arrays_matches_dataframe_api(sample_ohlcv):
    close_col = "Close" if "Close" in sample_ohlcv.columns else "Adj Close"
    high = sample_ohlcv["High"].values
    low = sample_ohlcv["Low"].values
    close = sample_ohlcv[close_col].values
    volume = sample_ohlcv["Volume"].values
    end = len(sample_ohlcv) - 1
    m1 = scan_cup_with_handle(sample_ohlcv, end)
    m2 = scan_cup_with_handle_arrays(high, low, close, volume, end)
    if m1 is None:
        assert m2 is None
    else:
        assert m2 is not None
        assert abs(m1.pivot - m2.pivot) < 1e-6


def test_detect_series_still_aligned(sample_ohlcv):
    out = detect_cup_with_handle_series(sample_ohlcv, min_warmup=60)
    assert len(out) == len(sample_ohlcv)
    assert out["Pass_Pattern"].dtype == bool
