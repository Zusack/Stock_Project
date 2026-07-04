"""Chart axis helpers for backtest views."""

from src.views.components.chart_factory import nice_y_axis_bounds, y_axis_decimals


def test_nice_y_axis_bounds_seven_ticks():
    min_y, max_y, ticks, interval = nice_y_axis_bounds([10.0, 20.0, 15.0], tick_count=7)
    assert len(ticks) == 7
    assert ticks[0] == min_y
    assert ticks[-1] == max_y
    assert interval > 0
    assert min_y < 10.0
    assert max_y > 20.0


def test_y_axis_decimals_for_small_span():
    _, _, ticks, _ = nice_y_axis_bounds([1.01, 1.05, 1.03], tick_count=5)
    assert y_axis_decimals(ticks) >= 2
