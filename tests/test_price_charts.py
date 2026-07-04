"""Tests for native price chart builders (real Y values, bounded height)."""

from __future__ import annotations

from src.flet_v1_compat import apply_flet_v1_compat  # noqa: F401
from src.services.stock_config import stock_config
from src.views.components.price_charts import (
    IndexSeries,
    build_candlestick_chart,
    build_index_comparison_chart,
)
from src.views.ui_helpers import build_ticker_candlestick_chart


def _find_charts(ctrl, chart_type: str) -> list:
    found: list = []
    if type(ctrl).__name__ == chart_type:
        found.append(ctrl)
    for attr in ("content", "controls"):
        child = getattr(ctrl, attr, None)
        if child is None:
            continue
        items = child if isinstance(child, list) else [child]
        for item in items:
            found.extend(_find_charts(item, chart_type))
    return found


def _find_chart(ctrl, chart_type: str):
    charts = _find_charts(ctrl, chart_type)
    return charts[0] if charts else None


def test_candlestick_uses_real_dollar_y_values():
    dates = [f"2026-01-{i:02d}" for i in range(1, 11)]
    opens = [100.0 + i for i in range(10)]
    highs = [o + 2 for o in opens]
    lows = [o - 2 for o in opens]
    closes = [o + 1 for o in opens]

    col = build_candlestick_chart(
        None, "TEST", dates, opens, highs, lows, closes, height=360
    )
    chart = _find_chart(col, "CandlestickChart")
    assert chart is not None
    assert chart.height == 360
    assert chart.expand is False
    assert chart.min_y < chart.max_y
    assert len(chart.spots) == 10
    spot = chart.spots[0]
    assert spot.open == 100.0
    assert spot.high == 102.0
    assert spot.low == 98.0
    assert spot.close == 101.0
    ys = [spot.open, spot.high, spot.low, spot.close]
    assert all(chart.min_y <= y <= chart.max_y for y in ys)


def test_index_comparison_uses_real_percent_y_values():
    pct_a = [float(i) * 0.5 for i in range(20)]
    pct_b = [float(i) * 0.3 for i in range(20)]
    dates = [f"2026-01-{1 + i % 28:02d}" for i in range(20)]
    col = build_index_comparison_chart(
        None,
        [
            IndexSeries("S&P 500", pct_a, "#3730a3"),
            IndexSeries("Dow", pct_b, "#92400e"),
        ],
        dates,
        interval_key="month",
        height=320,
    )
    chart = _find_chart(col, "LineChart")
    assert chart is not None
    assert chart.height == 320
    assert chart.expand is False
    assert len(chart.data_series) >= 2
    ys = [p.y for p in chart.data_series[0].points]
    assert max(ys) > 0.0
    assert chart.min_y == 0.0
    assert chart.max_y > 0.0
    label_texts = [lab.label.value for lab in chart.left_axis.labels]
    assert all("nan" not in str(t).lower() for t in label_texts)


def test_ticker_candlestick_from_database():
    cfg = stock_config()
    col = build_ticker_candlestick_chart(None, "AAPL", cfg.db_path, height=360)
    chart = _find_chart(col, "CandlestickChart")
    if chart is None:
        assert type(col).__name__ in ("Container", "Column")
        return
    assert len(chart.spots) >= 2
    assert chart.height == 360
