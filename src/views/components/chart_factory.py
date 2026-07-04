"""
Chart factories (line / bar / scatter / histogram) and small chart utilities.

These wrap flet-charts so view code doesn't have to redo border, grid, axis
label, and tooltip styling on every chart. Color choices flow through
ThemeHelper so light and dark modes look right.
"""
from __future__ import annotations

import math

import flet as ft
import flet_charts as fchart

from src.utils.format_utils import shorten
from src.views.theme import Palette, ThemeHelper

CHART_AXIS_LABEL_SIZE = 11
CHART_AXIS_TITLE_SIZE = 12
CHART_MIN_PLOT_HEIGHT = 280


def format_signed_value(value: float, *, decimals: int = 2, suffix: str = "") -> str:
    """Format a numeric value with explicit + / − sign."""
    if value > 0:
        sign = "+"
    elif value < 0:
        sign = "−"
    else:
        sign = ""
    if decimals <= 0 and abs(value - round(value)) < 1e-9:
        body = str(int(round(value)))
    else:
        body = f"{abs(value):.{decimals}f}".rstrip("0").rstrip(".")
    return f"{sign}{body}{suffix}"


def compute_trend_insights(values: list[float], *, label: str = "Series") -> list[str]:
    """Short textual trend callouts from a numeric series."""
    if not values or len(values) < 2:
        return [f"{label}: not enough data for trend summary."]
    first = float(values[0])
    last = float(values[-1])
    change = last - first
    pct = (change / abs(first) * 100.0) if abs(first) > 1e-9 else 0.0
    window = values[-min(20, len(values)) :]
    mid = len(window) // 2
    early = sum(window[:mid]) / max(1, mid)
    late = sum(window[mid:]) / max(1, len(window) - mid)
    momentum = "rising" if late > early * 1.002 else "falling" if late < early * 0.998 else "flat"
    insights = [
        f"{label}: {format_signed_value(change, decimals=2)} "
        f"({format_signed_value(pct, decimals=1, suffix='%')}) over period.",
        f"Recent momentum: {momentum} (last {len(window)} points).",
    ]
    if any(v < 0 for v in values) and any(v > 0 for v in values):
        insights.append("Series crosses zero — compare vs baseline carefully.")
    return insights


def zero_baseline_series(
    page,
    *,
    min_x: float,
    max_x: float,
    y_value: float = 0.0,
) -> ft.LineChartData:
    """Horizontal zero-reference line when series span positive and negative."""
    return ft.LineChartData(
        data_points=[
            ft.LineChartDataPoint(min_x, y_value, show_tooltip=False),
            ft.LineChartDataPoint(max_x, y_value, show_tooltip=False),
        ],
        color=ThemeHelper.text_muted(page),
        stroke_width=1,
        curved=False,
    )


def nice_y_axis_bounds(
    values: list[float],
    *,
    tick_count: int = 7,
    padding_frac: float = 0.1,
    min_pad: float = 0.25,
) -> tuple[float, float, list[float], float]:
    """
    Compute padded y-axis min/max, tick values, and grid step for line charts.

    Returns (min_y, max_y, y_ticks, y_interval).
    """
    if tick_count < 2:
        tick_count = 2
    if not values:
        y_min, y_max = -1.0, 1.0
    else:
        y_min = float(min(values))
        y_max = float(max(values))
        span = y_max - y_min
        if span <= 1e-9:
            center = y_min
            half = max(abs(center) * 0.1, min_pad)
            y_min = center - half
            y_max = center + half
            span = y_max - y_min
        pad = max(span * padding_frac, min_pad)
        y_min -= pad
        y_max += pad
    y_interval = (y_max - y_min) / (tick_count - 1) if tick_count > 1 else 1.0
    y_ticks = [y_min + y_interval * i for i in range(tick_count)]
    return y_min, y_max, y_ticks, y_interval


def y_axis_decimals(y_ticks: list[float]) -> int:
    """Pick decimal places for y labels from tick span."""
    if not y_ticks:
        return 1
    span = abs(y_ticks[-1] - y_ticks[0])
    if span >= 100:
        return 0
    if span >= 10:
        return 1
    if span >= 1:
        return 2
    return 3


def nice_integer_y_ticks(max_bar_value: float) -> tuple[float, float | None, list[float]]:
    """
    Return (max_y, grid_interval, ticks) for bar charts with non-negative
    integer counts. Picks a "nice" step (1, 2, 5, 10, ...) so axis labels are
    legible.
    """
    if max_bar_value <= 0:
        return 1.0, 1.0, [0.0, 1.0]
    cap = max(1.0, math.ceil(float(max_bar_value) * 1.1))
    rough_step = max(1, int(cap // 5))
    step_f = float(rough_step)
    for cand in (1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250, 500, 1000):
        if float(cand) >= rough_step:
            step_f = float(cand)
            break
    ticks: list[float] = []
    t = 0.0
    while t <= cap + step_f * 0.001:
        ticks.append(round(t, 6))
        t += step_f
    max_y = ticks[-1] if ticks else cap
    if max_y < cap:
        while max_y < cap:
            max_y = round(max_y + step_f, 6)
            ticks.append(max_y)
    return float(max_y), step_f, ticks


def shorten_label(value: str, max_len: int = 12) -> str:
    return shorten(value, max_len)


def axis_labels(
    values: list[str], size: int = CHART_AXIS_LABEL_SIZE, max_len: int = 12
) -> list[ft.ChartAxisLabel]:
    """Build categorical labels for the bottom axis (indexed by position)."""
    labels = []
    for idx, value in enumerate(values):
        labels.append(
            ft.ChartAxisLabel(
                value=idx,
                label=ft.Container(
                    ft.Text(shorten_label(value, max_len=max_len), size=size),
                    padding=4,
                    tooltip=value,
                ),
            )
        )
    return labels


def numeric_axis_labels(
    values: list[float],
    page: ft.Page | None,
    size: int = CHART_AXIS_LABEL_SIZE,
    decimals: int = 0,
    *,
    signed: bool = False,
) -> list[ft.ChartAxisLabel]:
    """Build numeric labels for a value-based axis (e.g. y-ticks)."""
    color = ThemeHelper.chart_axis_label(page)
    labels: list[ft.ChartAxisLabel] = []
    for value in values:
        if signed:
            text = format_signed_value(float(value), decimals=decimals)
        elif decimals <= 0 and abs(value - round(value)) < 1e-9:
            text = str(int(round(value)))
        else:
            text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
        labels.append(ft.ChartAxisLabel(value=value, label=ft.Text(text, size=size, color=color)))
    return labels


def chart_axis_title(text: str, *, size: int = CHART_AXIS_TITLE_SIZE) -> ft.Text:
    return ft.Text(text, size=size, weight=ft.FontWeight.W_500)


def chart_empty_state(page: ft.Page | None, message: str) -> ft.Container:
    """Centered text placeholder for charts with no data."""
    return ft.Container(
        content=ft.Text(message, color=ThemeHelper.chart_empty_text(page), size=13),
        alignment=ft.alignment.center,
        padding=10,
    )


def chart_loading_state(
    page: ft.Page | None,
    message: str,
    *,
    submessage: str = "",
    height: int = CHART_MIN_PLOT_HEIGHT,
) -> ft.Container:
    """Animated spinner while chart data is loading."""
    body: list[ft.Control] = [
        ft.ProgressRing(width=36, height=36, stroke_width=3),
        ft.Text(message, size=13, color=ThemeHelper.text_primary(page), text_align=ft.TextAlign.CENTER),
    ]
    if submessage:
        body.append(
            ft.Text(
                submessage,
                size=11,
                color=ThemeHelper.text_muted(page),
                text_align=ft.TextAlign.CENTER,
            )
        )
    return ft.Container(
        content=ft.Column(
            body,
            spacing=12,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        alignment=ft.alignment.center,
        height=height,
        padding=16,
    )


def spaced_index_ticks(count: int, *, max_ticks: int = 6) -> list[int]:
    """Evenly spaced integer indices for chart axis labels (always includes first and last)."""
    if count <= 0:
        return []
    if count == 1:
        return [0]
    tick_count = min(max_ticks, count)
    if tick_count <= 2:
        return [0, count - 1]
    step = (count - 1) / (tick_count - 1)
    indices: list[int] = []
    seen: set[int] = set()
    for i in range(tick_count):
        idx = int(round(i * step))
        idx = max(0, min(idx, count - 1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return indices


def dynamic_content_slot(
    page: ft.Page | None,
    placeholder: str,
    *,
    size: int = 12,
) -> ft.Container:
    """Grows with child content — no fixed height (safe inside scrollable views)."""
    return ft.Container(
        content=ft.Text(placeholder, color=ThemeHelper.text_muted(page), size=size),
    )


def chart_panel_container(
    page: ft.Page | None,
    content: ft.Control,
    *,
    padding: int = 12,
    border_radius: int = 6,
    panel_height: int | None = None,
    min_height: int | None = None,
    expand: bool = False,
) -> ft.Container:
    """Bordered inset panel matching other chart surfaces."""
    kwargs: dict = {
        "content": content,
        "padding": padding,
        "border": ft.border.all(1, ThemeHelper.border_default(page)),
        "border_radius": border_radius,
        "expand": expand,
    }
    if panel_height is not None:
        kwargs["height"] = panel_height
    elif min_height is not None:
        # Flet 0.85 Container has no min_height; use height as a floor for chart panels.
        kwargs["height"] = min_height
    return ft.Container(**kwargs)


def build_bar_chart(
    page: ft.Page | None,
    groups: list[ft.BarChartGroup],
    bottom_labels: list[ft.ChartAxisLabel],
    left_title: str,
    left_labels: list[ft.ChartAxisLabel],
    *,
    height: int = 220,
    min_y: float = 0.0,
    max_y: float = 100.0,
    y_interval: float | None = None,
    bottom_labels_size: int = 60,
    left_labels_size: int = 44,
) -> ft.BarChart:
    grid_interval = y_interval if y_interval and y_interval > 0 else None
    return ft.BarChart(
        bar_groups=groups,
        bottom_axis=ft.ChartAxis(
            labels=bottom_labels,
            labels_size=bottom_labels_size,
            label_spacing=1.0,
            show_labels=True,
        ),
        left_axis=ft.ChartAxis(
            title=chart_axis_title(left_title),
            labels=left_labels,
            labels_size=left_labels_size,
            label_spacing=1.0,
            show_labels=True,
        ),
        border=ft.border.all(1, ThemeHelper.border_default(page)),
        height=height,
        min_y=min_y,
        max_y=max_y,
        tooltip_bgcolor=ThemeHelper.chart_tooltip_bg(page),
        horizontal_grid_lines=ft.ChartGridLines(
            interval=grid_interval, color=ThemeHelper.chart_grid_color(page)
        ),
    )


def build_histogram_chart(
    page: ft.Page | None,
    counts: list[int],
    *,
    bin_min: int = 0,
    bin_max: int = 10,
    x_title: str = "Bin",
    y_title: str = "Count",
    height: int = 220,
    width: float | None = None,
    expand: bool = True,
) -> ft.BarChart:
    """Vertical histogram: one bar per integer bin."""
    n_bins = bin_max - bin_min + 1
    if len(counts) != n_bins:
        raise ValueError(f"counts length {len(counts)} != {n_bins} bins for range [{bin_min}, {bin_max}]")
    max_count = max(counts) if counts else 0
    max_y, y_step, y_ticks = nice_integer_y_ticks(float(max_count))
    y_interval = 1.0
    norm_max = float(max(len(y_ticks) - 1, 1))
    color = ThemeHelper.chart_named(page, "price") if page else Palette.cyan.s400
    axis_color = ThemeHelper.chart_axis_label(page)
    bottom_labels = [
        ft.ChartAxisLabel(
            value=float(i),
            label=ft.Text(str(bin_min + i), size=9, color=axis_color),
        )
        for i in range(n_bins)
    ]
    left_labels = [
        ft.ChartAxisLabel(
            value=float(i),
            label=ft.Text(str(int(round(tick))), size=10, color=axis_color),
        )
        for i, tick in enumerate(y_ticks)
    ]
    groups: list[ft.BarChartGroup] = []
    for i, c in enumerate(counts):
        rating = bin_min + i
        norm_h = (float(c) / y_step) if y_step > 0 else 0.0
        groups.append(
            ft.BarChartGroup(
                x=int(i),
                bar_rods=[
                    ft.BarChartRod(
                        to_y=norm_h,
                        color=color,
                        width=12,
                        border_radius=3,
                        tooltip=f"{rating}: {c}",
                    )
                ],
            )
        )
    chart = build_bar_chart(
        page,
        groups,
        bottom_labels,
        y_title,
        left_labels,
        height=height,
        min_y=0.0,
        max_y=norm_max,
        y_interval=y_interval,
        bottom_labels_size=52,
        left_labels_size=44,
    )
    chart.bottom_axis.title = chart_axis_title(x_title)
    if width is not None:
        chart.width = width
    if expand is not None:
        chart.expand = expand
    return chart


def build_line_chart(
    page: ft.Page | None,
    series: list[ft.LineChartData],
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    left_axis: ft.ChartAxis,
    bottom_axis: ft.ChartAxis,
    x_interval: float | None = None,
    y_interval: float | None = None,
    height: int = 320,
    width: float | None = None,
    expand: bool = True,
    interactive: bool = True,
    on_event=None,
    use_tooltip: bool = True,
    baseline_y: float | None = None,
) -> ft.LineChart:
    kwargs: dict = dict(
        data_series=series,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        left_axis=left_axis,
        bottom_axis=bottom_axis,
        tooltip_bgcolor=ThemeHelper.chart_tooltip_bg(page),
        horizontal_grid_lines=ft.ChartGridLines(
            interval=y_interval, color=ThemeHelper.chart_grid_color(page)
        ),
        vertical_grid_lines=ft.ChartGridLines(
            interval=x_interval, color=ThemeHelper.chart_grid_color(page)
        ),
        height=height,
        width=width,
        expand=expand,
        interactive=interactive,
    )
    if baseline_y is not None:
        kwargs["baseline_y"] = baseline_y
    if on_event is not None:
        kwargs["on_event"] = on_event
    try:
        chart = ft.LineChart(**kwargs)
    except TypeError:
        kwargs.pop("tooltip_bgcolor", None)
        kwargs.pop("interactive", None)
        chart = ft.LineChart(**kwargs)
    if not use_tooltip:
        try:
            chart.tooltip = None
        except (AttributeError, TypeError):
            pass
    return chart


def build_scatter_chart(
    page: ft.Page | None,
    spots: list[fchart.ScatterChartSpot],
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    left_axis: ft.ChartAxis,
    bottom_axis: ft.ChartAxis,
    height: int = 320,
    width: float | None = None,
    expand: bool = False,
) -> fchart.ScatterChart:
    """Themed ScatterChart. Use with numeric X and Y values."""
    grid = ThemeHelper.chart_grid_color(page)
    border = ThemeHelper.border_default(page)
    tt_bg = ThemeHelper.chart_tooltip_bg(page)
    kwargs: dict = dict(
        spots=spots,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        left_axis=left_axis,
        bottom_axis=bottom_axis,
        horizontal_grid_lines=ft.ChartGridLines(color=grid),
        vertical_grid_lines=ft.ChartGridLines(color=grid),
        border=ft.border.all(1, border),
        interactive=True,
        show_tooltips_for_selected_spots_only=False,
        height=height,
        width=width,
        expand=expand,
        tooltip=fchart.ScatterChartTooltip(bgcolor=tt_bg),
    )
    try:
        return fchart.ScatterChart(**kwargs)
    except TypeError:
        kwargs.pop("tooltip", None)
        kwargs.pop("height", None)
        kwargs.pop("width", None)
        kwargs.pop("expand", None)
        return fchart.ScatterChart(**kwargs)


def legend_chip(label: str, color: str, size: int = 10) -> ft.Row:
    """Small colored square + label, suitable for chart legends.

    ``tight=True`` is required so chips keep intrinsic width. Without it, each
    chip Row expands to full parent width and a wrapping legend stacks as a
    single column (pushing the plot out of a fixed-height panel).
    """
    return ft.Row(
        [
            ft.Container(width=size, height=size, bgcolor=color, border_radius=2),
            ft.Text(label, size=10, no_wrap=True),
        ],
        spacing=5,
        tight=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def trend_insight_panel(page, insights: list[str]) -> ft.Container:
    """Compact callout panel for chart trend statistics."""
    if not insights:
        return ft.Container()
    return ft.Container(
        content=ft.Column(
            [ft.Text(line, size=11, color=ThemeHelper.text_muted(page)) for line in insights],
            spacing=2,
            tight=True,
        ),
        padding=ft.padding.only(top=6, bottom=4),
        border=ft.border.only(top=ft.BorderSide(1, ThemeHelper.border_subtle(page))),
    )


def chart_plot_column(
    page,
    chart: ft.Control,
    *,
    legend: list[ft.Control] | None = None,
    insights: list[str] | None = None,
    subtitle: str | None = None,
    expand: bool = True,
    plot_height: int | None = None,
) -> ft.Column:
    """Stack horizontal legend, fixed-height chart, and insights.

    Legend chips flow left-to-right (wrapping only when needed). The plot keeps
    an explicit height so it stays fully visible regardless of series count;
    the column grows with the legend instead of clipping the chart.
    """
    controls: list[ft.Control] = []
    if legend:
        controls.append(
            ft.Row(
                legend,
                spacing=12,
                run_spacing=6,
                wrap=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
    if subtitle:
        controls.append(
            ft.Text(subtitle, size=11, color=ThemeHelper.text_muted(page))
        )
    # Reserve a stable plot area so legend growth never shrinks the chart.
    chart_height = plot_height
    if chart_height is None:
        chart_height = getattr(chart, "height", None)
    if chart_height:
        try:
            chart.height = chart_height
        except (AttributeError, TypeError):
            pass
        controls.append(
            ft.Container(
                content=chart,
                height=float(chart_height),
                clip_behavior=ft.ClipBehavior.NONE,
            )
        )
    else:
        controls.append(chart)
    if insights:
        controls.append(trend_insight_panel(page, insights))
    return ft.Column(controls, spacing=8, tight=True, expand=expand)
