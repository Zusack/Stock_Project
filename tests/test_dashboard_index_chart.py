"""Dashboard index chart data prep — skip NaN closes before % normalization."""

from __future__ import annotations

import math

import pandas as pd

from src.analysis.db import load_prices_bulk
from src.services.stock_config import stock_config
from src.utils.format_utils import slice_price_df
from src.views.dashboard_view import _MARKET_INDICES


def test_index_pct_series_skip_nan_adj_close():
    cfg = stock_config()
    tickers = [t for t, _ in _MARKET_INDICES]
    data = load_prices_bulk(tickers, cfg.db_path)
    if not any(data.get(t) is not None and not data[t].empty for t in tickers):
        return

    for ticker, _label in _MARKET_INDICES:
        df = slice_price_df(data.get(ticker), "week")
        if df is None or df.empty:
            continue
        close = df["Adj Close"].astype(float).dropna()
        if len(close) < 2:
            continue
        base = float(close.iloc[0])
        pct = [((float(p) / base) - 1.0) * 100.0 for p in close.tolist()]
        assert all(math.isfinite(v) for v in pct), f"{ticker} produced non-finite % values"
