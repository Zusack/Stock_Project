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


def test_build_compare_equity_chart_passes_bottom_axis(monkeypatch):
    """Regression: compare overlay must supply bottom_axis to build_line_chart."""
    import src.flet_v1_compat  # noqa: F401
    import flet as ft
    from src.views import backtest_charts as bc

    captured = {}

    def _fake_build_line_chart(page, series, **kwargs):
        captured.update(kwargs)
        return ft.Container()

    monkeypatch.setattr(bc, "build_line_chart", _fake_build_line_chart)
    monkeypatch.setattr(
        bc,
        "normalized_price_axis",
        lambda *a, **k: (
            ft.ChartAxis(),
            type("S", (), {"norm_max": 10.0, "to_chart_y": staticmethod(lambda v: v)})(),
        ),
    )
    monkeypatch.setattr(
        bc,
        "normalized_time_axis",
        lambda *a, **k: (ft.ChartAxis(), 1.0),
    )
    monkeypatch.setattr(bc, "legend_chip", lambda *a, **k: ft.Text("x"))
    monkeypatch.setattr(bc.ThemeHelper, "chart_named", staticmethod(lambda *a, **k: "#0f0"))

    ctrl = bc.build_compare_equity_chart(
        None,
        [
            ("A", ["2020-01-01", "2020-02-01"], [100.0, 110.0]),
            ("B", ["2020-01-01", "2020-02-01"], [100.0, 105.0]),
        ],
    )
    assert ctrl is not None
    assert "bottom_axis" in captured
    assert captured["bottom_axis"] is not None
