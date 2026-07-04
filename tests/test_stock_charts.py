"""Tests for integer-normalized stock chart axis helpers."""

from __future__ import annotations

from src.flet_v1_compat import apply_flet_v1_compat  # noqa: F401
from src.views.components.stock_charts import (
    NormalizedYScale,
    format_time_label,
    normalize_y,
    normalized_price_axis,
    normalized_time_axis,
)


def test_normalize_y_maps_to_integer_ticks():
    scale = NormalizedYScale.from_values([100.0, 110.0], tick_count=5)
    assert scale.to_chart_y(scale.y_min) == 0.0
    assert abs(scale.to_chart_y(scale.y_max) - scale.norm_max) < 1e-6


def test_price_axis_uses_label_spacing_one():
    axis, scale = normalized_price_axis(None, [189.1, 190.5, 191.0])
    assert axis.label_spacing == 1.0
    assert len(axis.labels) == len(scale.tick_values)
    for i, lbl in enumerate(axis.labels):
        assert lbl.value == float(i)


def test_time_axis_integer_indices():
    labels = [f"2026-06-10T{h:02d}:00:00Z" for h in range(10)]
    axis, x_interval = normalized_time_axis(None, labels, fmt="intraday")
    assert axis.label_spacing >= 1.0
    assert all(lbl.value == float(int(lbl.value)) for lbl in axis.labels)


def test_format_time_label_intraday():
    assert format_time_label("2026-06-10T14:31:00Z", "intraday") == "14:31 UTC"


def _find_line_chart(control):
    if control.__class__.__name__ == "LineChart":
        return control
    for child in getattr(control, "controls", None) or []:
        found = _find_line_chart(child)
        if found is not None:
            return found
    content = getattr(control, "content", None)
    if content is not None:
        return _find_line_chart(content)
    return None


def test_build_multi_series_chart_defaults_to_non_expanding():
    from src.views.components.stock_charts import SeriesSpec, build_multi_series_chart

    chart = build_multi_series_chart(
        None,
        [SeriesSpec(label="A", values=[0.0, 1.0, 2.0], color="#3366cc")],
        ["2026-01-01", "2026-01-02", "2026-01-03"],
        signed_y=True,
        height=200,
    )
    line = _find_line_chart(chart)
    assert line is not None
    assert line.expand is False
    assert chart.expand is False


def test_multi_series_aligns_short_history_to_real_dates():
    """IPO-length series must plot on its calendar dates, not at axis start."""
    from src.views.components.stock_charts import SeriesSpec, build_multi_series_chart

    long_dates = [f"2026-01-{d:02d}" for d in range(1, 11)]
    ipo_dates = ["2026-01-08", "2026-01-09", "2026-01-10"]
    chart = build_multi_series_chart(
        None,
        [
            SeriesSpec(
                label="LONG",
                values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                color="#3366cc",
                timestamps=long_dates,
            ),
            SeriesSpec(
                label="IPO",
                values=[0.0, 1.0, 2.0],
                color="#cc6633",
                timestamps=ipo_dates,
            ),
        ],
        [],
        signed_y=True,
        height=200,
    )
    line = _find_line_chart(chart)
    assert line is not None
    assert len(line.data_series) == 2
    long_xs = [p.x for p in line.data_series[0].points]
    ipo_xs = [p.x for p in line.data_series[1].points]
    assert long_xs == [float(i) for i in range(10)]
    # IPO occupies the last three slots of the shared calendar axis.
    assert ipo_xs == [7.0, 8.0, 9.0]
    assert line.min_x == 0.0
    assert line.max_x == 9.0


def test_modular_arithmetic_fix():
    """Prices on fractional spacing fail fl_chart label test; integers pass."""
    y_min, y_step = 189.102, 0.796
    for i in range(7):
        price = y_min + y_step * i
        assert price % y_step != 0 or i == 0
        norm = normalize_y(price, y_min, y_step)
        assert abs(norm - float(i)) < 1e-6
        assert float(i) % 1.0 == 0.0
