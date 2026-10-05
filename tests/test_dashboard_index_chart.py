"""Dashboard index chart data prep — skip NaN closes before % normalization."""

from __future__ import annotations

import math

import pandas as pd

from src.analysis.db import load_prices_bulk
from src.services.stock_config import stock_config
from src.utils.format_utils import slice_price_df
from src.views.components.price_charts import prepare_market_index_series
from src.views.dashboard_view import _MARKET_INDICES


def test_index_pct_series_skip_nan_adj_close():
    cfg = stock_config()
    tickers = [t for t, _ in _MARKET_INDICES]
    data = load_prices_bulk(tickers, cfg.db_path)
    if not any(data.get(t) is not None and not data[t].empty for t in tickers):
        return

    for ticker, label in _MARKET_INDICES:
        series = prepare_market_index_series(
            data.get(ticker),
            label=label,
            color="#3366cc",
            interval_key="week",
        )
        assert series, f"{ticker} produced no index series"
        pct = series[0].values
        assert all(math.isfinite(v) for v in pct), f"{ticker} produced non-finite % values"


def test_index_sma_series_only_when_requested():
    cfg = stock_config()
    tickers = [t for t, _ in _MARKET_INDICES]
    data = load_prices_bulk(tickers, cfg.db_path, max_days=400)
    ticker = next((t for t in tickers if data.get(t) is not None and len(data[t]) >= 55), None)
    if ticker is None:
        return
    label = dict(_MARKET_INDICES)[ticker]
    without = prepare_market_index_series(
        data[ticker], label=label, color="#3366cc", interval_key="month", include_sma50=False
    )
    with_sma = prepare_market_index_series(
        data[ticker], label=label, color="#3366cc", interval_key="month", include_sma50=True
    )
    assert len(without) == 1
    assert len(with_sma) == 2
    assert with_sma[1].dash_pattern is not None
    assert all(math.isfinite(v) for v in with_sma[1].values)

