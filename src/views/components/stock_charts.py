"""
Professional stock chart builders with integer-normalized axes.

flet-charts (fl_chart) only renders axis labels when ``value % label_spacing ~= 0``.
Fractional price ticks (e.g. $189.10 with spacing 0.796) never pass that test.
We map data to integer Y indices (0, 1, 2, …) with ``label_spacing=1`` and show
formatted price/time text on each tick — the pattern used in official Flet examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import flet as ft
import flet_charts as fchart

from src.views.components.chart_factory import (
    CHART_AXIS_LABEL_SIZE,
    CHART_MIN_PLOT_HEIGHT,
    build_line_chart,
    chart_axis_title,
    chart_empty_state,
    chart_plot_column,
    compute_trend_insights,
    format_signed_value,
    legend_chip,
    nice_y_axis_bounds,
    trend_insight_panel,
    y_axis_decimals,
)
from src.views.theme import ThemeHelper

TimeFormat = Literal["intraday", "daily", "monthly", "auto"]
_LEFT_AXIS_LABEL_SIZE = 56
_BOTTOM_AXIS_LABEL_SIZE = 48
_DEFAULT_TICK_COUNT = 7


def normalize_y(value: float, y_min: float, y_step: float) -> float:
    """Map a data value into integer tick space for the chart Y axis."""
    if abs(y_step) < 1e-12:
        return 0.0
    return (float(value) - y_min) / y_step


def format_time_label(text: str, fmt: TimeFormat = "auto") -> str:
    """Format a timestamp string for axis/tooltip display."""
    if not text:
        return ""
    if fmt == "intraday" or (fmt == "auto" and "T" in text and len(text) >= 16):
        return f"{text[11:16]} UTC" if len(text) >= 16 else text
    if fmt == "monthly" or (fmt == "auto" and len(text) >= 7 and text[4] == "-"):
        return text[:7] if len(text) >= 7 else text
    if len(text) >= 10:
        try:
            month = int(text[5:7])
            day = int(text[8:10])
            months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
            if 1 <= month <= 12:
                return f"{months[month - 1]} {day}"
        except ValueError:
            pass
        return text[:10]
    return text


def _spaced_indices(count: int, *, tick_count: int = _DEFAULT_TICK_COUNT) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [0]
    ticks = min(max(2, tick_count), count)
    step = max(1, round((count - 1) / max(ticks - 1, 1)))
    indices: list[int] = []
    idx = 0
    while idx < count:
        indices.append(idx)
        idx += step
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return indices


@dataclass(frozen=True)
class NormalizedYScale:
    """Real-value bounds and tick step used to map prices onto integer Y indices."""

    y_min: float
    y_max: float
    y_step: float
    tick_values: list[float]
    norm_max: float

    @classmethod
    def from_values(
        cls,
        values: list[float],
        *,
        tick_count: int = _DEFAULT_TICK_COUNT,
        padding_frac: float = 0.1,
        min_pad: float = 0.25,
    ) -> NormalizedYScale:
        y_min, y_max, ticks, y_step = nice_y_axis_bounds(
            values,
            tick_count=tick_count,
            padding_frac=padding_frac,
            min_pad=min_pad,
        )
        norm_max = float(max(len(ticks) - 1, 1))
        return cls(y_min=y_min, y_max=y_max, y_step=y_step, tick_values=ticks, norm_max=norm_max)

    def to_chart_y(self, value: float) -> float:
        return normalize_y(value, self.y_min, self.y_step)


def normalized_value_axis(
    page: ft.Page | None,
    scale: NormalizedYScale,
    *,
    title: str,
    label_formatter: Callable[[float], str],
    label_size: int = _LEFT_AXIS_LABEL_SIZE,
) -> ft.ChartAxis:
    """Left axis with integer tick indices and formatted value labels."""
    color = ThemeHelper.chart_axis_label(page)
    labels = [
        ft.ChartAxisLabel(
            value=float(i),
            label=ft.Text(label_formatter(tick), size=CHART_AXIS_LABEL_SIZE, color=color),
        )
        for i, tick in enumerate(scale.tick_values)
    ]
    return ft.ChartAxis(
        title=chart_axis_title(title),
        labels=labels,
        label_spacing=1.0,
        labels_size=label_size,
        show_labels=True,
        show_min=True,
        show_max=True,
    )


def normalized_price_axis(
    page: ft.Page | None,
    values: list[float],
    *,
    title: str = "Price ($)",
    tick_count: int = _DEFAULT_TICK_COUNT,
    label_size: int = _LEFT_AXIS_LABEL_SIZE,
) -> tuple[ft.ChartAxis, NormalizedYScale]:
    scale = NormalizedYScale.from_values(values, tick_count=tick_count)
    decimals = y_axis_decimals(scale.tick_values)

    def _fmt(v: float) -> str:
        if decimals <= 0 and abs(v - round(v)) < 1e-9:
            body = str(int(round(v)))
        else:
            body = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
        return f"${body}"

    axis = normalized_value_axis(
        page, scale, title=title, label_formatter=_fmt, label_size=label_size
    )
    return axis, scale


def normalized_signed_axis(
    page: ft.Page | None,
    values: list[float],
    *,
    title: str,
    tick_count: int = _DEFAULT_TICK_COUNT,
    decimals: int | None = None,
    suffix: str = "%",
    label_size: int = _LEFT_AXIS_LABEL_SIZE,
    padding_frac: float = 0.1,
    min_pad: float = 0.25,
) -> tuple[ft.ChartAxis, NormalizedYScale]:
    scale = NormalizedYScale.from_values(
        values, tick_count=tick_count, padding_frac=padding_frac, min_pad=min_pad
    )
    dec = decimals if decimals is not None else y_axis_decimals(scale.tick_values)

    def _fmt(v: float) -> str:
        return format_signed_value(v, decimals=dec, suffix=suffix)

    axis = normalized_value_axis(
        page, scale, title=title, label_formatter=_fmt, label_size=label_size
    )
    return axis, scale


def normalized_time_axis(
    page: ft.Page | None,
    labels: list[str],
    *,
    title: str = "Time",
    tick_count: int = _DEFAULT_TICK_COUNT,
    fmt: TimeFormat = "auto",
    label_size: int = _BOTTOM_AXIS_LABEL_SIZE,
) -> tuple[ft.ChartAxis, float]:
    """Bottom axis with integer X indices and formatted time/category labels."""
    n = len(labels)
    if n <= 0:
        return ft.ChartAxis(title=chart_axis_title(title), label_spacing=1.0), 1.0

    indices = _spaced_indices(n, tick_count=tick_count)
    x_interval = float(max(1, round((n - 1) / max(len(indices) - 1, 1)))) if n > 1 else 1.0
    color = ThemeHelper.chart_axis_label(page)
    axis_labels = [
        ft.ChartAxisLabel(
            value=float(i),
            label=ft.Text(format_time_label(labels[i], fmt), size=CHART_AXIS_LABEL_SIZE, color=color),
        )
        for i in indices
    ]
    axis = ft.ChartAxis(
        title=chart_axis_title(title),
        labels=axis_labels,
        label_spacing=x_interval,
        labels_size=label_size,
        show_labels=True,
        show_min=True,
        show_max=True,
    )
    return axis, x_interval


def _zero_baseline_norm(
    page: ft.Page | None,
    *,
    min_x: float,
    max_x: float,
    scale: NormalizedYScale,
) -> ft.LineChartData | None:
    zero = 0.0
    if scale.y_min > zero or scale.y_max < zero:
        return None
    y_norm = scale.to_chart_y(zero)
    return ft.LineChartData(
        data_points=[
            ft.LineChartDataPoint(min_x, y_norm, show_tooltip=False),
            ft.LineChartDataPoint(max_x, y_norm, show_tooltip=False),
        ],
        color=ThemeHelper.text_muted(page),
        stroke_width=1,
        curved=False,
    )


@dataclass
class SeriesSpec:
    label: str
    values: list[float]
    color: str
    tooltips: list[str] | None = None
    # Per-point dates. When set on any series, all series are aligned to a shared
    # calendar axis so short histories (e.g. IPOs) plot on their real dates.
    timestamps: list[str] | None = None
    dash_pattern: list[int] | None = None
    stroke_width: float | None = None
    show_points: bool | None = None


def _master_timeline(
    series_list: list[SeriesSpec],
    timestamps: list[str],
) -> tuple[list[str], bool]:
    """Build the X-axis timeline.

    Returns ``(master_timestamps, date_aligned)``. When any series carries its
    own timestamps, the master axis is the sorted union of those dates so each
    point maps to its real calendar position rather than positional index 0..n.
    """
    per_series = [list(s.timestamps) for s in series_list if s.timestamps]
    if not per_series:
        n = max((len(s.values) for s in series_list), default=0)
        if len(timestamps) < n:
            timestamps = [*timestamps, *([""] * (n - len(timestamps)))]
        else:
            timestamps = list(timestamps[:n])
        return timestamps, False

    seen: set[str] = set()
    master: list[str] = []
    for ts_list in per_series:
        for ts in ts_list:
            if ts and ts not in seen:
                seen.add(ts)
                master.append(ts)
    for ts in timestamps:
        if ts and ts not in seen:
            seen.add(ts)
            master.append(ts)
    master.sort()
    return master, True


def build_multi_series_chart(
    page: ft.Page | None,
    series_list: list[SeriesSpec],
    timestamps: list[str],
    *,
    y_title: str = "Value",
    x_title: str = "Time",
    time_fmt: TimeFormat = "auto",
    signed_y: bool = False,
    y_suffix: str = "%",
    y_decimals: int | None = None,
    show_zero_baseline: bool = False,
    height: int = 320,
    expand: bool = False,
    interactive: bool = True,
    use_tooltip: bool = True,
    on_event=None,
    subtitle: str | None = None,
    insights_label: str = "Series",
    extra_insights: list[str] | None = None,
) -> ft.Control:
    """Multi-line chart with normalized axes, legend, and optional trend insights.

    Pass ``SeriesSpec.timestamps`` for each series when histories may start on
    different dates (Compare tab). Points are placed on a shared calendar axis.
    """
    if not series_list or not any(s.values for s in series_list):
        return chart_empty_state(page, "No data to chart.")

    timestamps, date_aligned = _master_timeline(series_list, timestamps)
    n = len(timestamps)
    if n < 1:
        return chart_empty_state(page, "No data to chart.")
    ts_to_x = {ts: i for i, ts in enumerate(timestamps)}

    all_y: list[float] = []
    for spec in series_list:
        all_y.extend(spec.values)

    if signed_y:
        left_axis, scale = normalized_signed_axis(
            page,
            all_y,
            title=y_title,
            decimals=y_decimals,
            suffix=y_suffix,
        )
    else:
        left_axis, scale = normalized_price_axis(page, all_y, title=y_title)

    bottom_axis, x_interval = normalized_time_axis(
        page, timestamps, title=x_title, fmt=time_fmt
    )

    line_series: list[ft.LineChartData] = []
    legend: list[ft.Control] = []
    for spec in series_list:
        # Integer-normalized axes span many X indices but only a few Y tick steps.
        # Curved splines overshoot that narrow band and fl_chart clips the stroke
        # entirely (hover/tooltips still hit the underlying spots).
        points: list[ft.LineChartDataPoint] = []
        if date_aligned:
            if not spec.timestamps:
                continue
            point_iter = list(zip(spec.timestamps, spec.values))
        else:
            point_iter = [
                (timestamps[i] if i < len(timestamps) else "", v)
                for i, v in enumerate(spec.values)
            ]

        for i, (ts, v) in enumerate(point_iter):
            if date_aligned:
                if not ts or ts not in ts_to_x:
                    continue
                x = float(ts_to_x[ts])
            else:
                x = float(i)
            tip = None
            if spec.tooltips and i < len(spec.tooltips):
                tip = spec.tooltips[i]
            elif use_tooltip:
                if signed_y:
                    tip = (
                        f"{format_time_label(ts, time_fmt)}\n"
                        f"{format_signed_value(v, decimals=y_decimals or 1, suffix=y_suffix)}"
                    )
                else:
                    tip = f"{format_time_label(ts, time_fmt)}\n${float(v):.2f}"
            points.append(
                ft.LineChartDataPoint(
                    x,
                    scale.to_chart_y(float(v)),
                    tooltip=tip or "",
                    show_tooltip=bool(tip) and use_tooltip,
                )
            )
        if not points:
            continue
        # Keep points ordered by X so the stroke never backtracks.
        points.sort(key=lambda p: p.x)
        line_series.append(
            ft.LineChartData(
                data_points=points,
                color=spec.color,
                stroke_width=spec.stroke_width if spec.stroke_width is not None else 3,
                dash_pattern=spec.dash_pattern,
                curved=False,
                point=spec.show_points if spec.show_points is not None else True,
            )
        )
        legend.append(legend_chip(spec.label, spec.color))

    baseline_y: float | None = None
    if show_zero_baseline and scale.y_min < 0 < scale.y_max:
        baseline_y = scale.to_chart_y(0.0)

    chart = build_line_chart(
        page,
        line_series,
        min_x=0.0,
        max_x=float(max(n - 1, 1)),
        min_y=0.0,
        max_y=scale.norm_max,
        left_axis=left_axis,
        bottom_axis=bottom_axis,
        x_interval=x_interval,
        y_interval=1.0,
        height=height,
        expand=expand,
        interactive=interactive,
        use_tooltip=use_tooltip,
        on_event=on_event,
        baseline_y=baseline_y,
    )

    primary = series_list[0].values
    insights = compute_trend_insights(primary, label=insights_label)
    if extra_insights:
        insights = [*insights, *extra_insights]
    return chart_plot_column(
        page,
        chart,
        legend=legend,
        insights=insights,
        subtitle=subtitle,
        expand=expand,
        plot_height=height,
    )


def build_price_line_chart(
    page: ft.Page | None,
    ticker: str,
    timestamps: list[str],
    closes: list[float],
    *,
    height: int = 360,
    time_fmt: TimeFormat = "intraday",
    x_title: str = "Time (UTC)",
    y_title: str = "Price ($)",
) -> ft.Control:
    """Single-series price line chart with tooltips and normalized axes."""
    if not closes:
        return chart_empty_state(page, f"No intraday data for {ticker}.")

    if len(closes) == 1:
        closes = [closes[0], closes[0]]
        timestamps = [timestamps[0], timestamps[0]]

    tooltips = [
        f"{format_time_label(ts, time_fmt)}\n${float(c):.2f}"
        for ts, c in zip(timestamps, closes)
    ]
    color = ThemeHelper.chart_named(page, "price")
    return build_multi_series_chart(
        page,
        [
            SeriesSpec(
                label=ticker,
                values=closes,
                color=color,
                tooltips=tooltips,
            )
        ],
        timestamps,
        y_title=y_title,
        x_title=x_title,
        time_fmt=time_fmt,
        signed_y=False,
        height=height,
        interactive=True,
        use_tooltip=True,
        insights_label=ticker,
    )


def build_ohlc_chart(
    page: ft.Page | None,
    ticker: str,
    timestamps: list[str],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    height: int = CHART_MIN_PLOT_HEIGHT,
    expand: bool = False,
) -> ft.Control:
    """Candlestick chart with normalized price axis and OHLC tooltips."""
    n = len(closes)
    if n == 0:
        return chart_empty_state(page, f"No OHLC data for {ticker}.")

    all_vals = list(opens) + list(highs) + list(lows) + list(closes)
    left_axis, scale = normalized_price_axis(page, all_vals, title="Price ($)")
    labels = [format_time_label(ts, "intraday") for ts in timestamps]
    bottom_axis, x_interval = normalized_time_axis(
        page, labels, title="Time (UTC)", fmt="intraday"
    )

    spots: list[fchart.CandlestickChartSpot] = []
    for i in range(n):
        tip = (
            f"{labels[i]}\n"
            f"O ${opens[i]:.2f}\n"
            f"H ${highs[i]:.2f}\n"
            f"L ${lows[i]:.2f}\n"
            f"C ${closes[i]:.2f}"
        )
        spots.append(
            fchart.CandlestickChartSpot(
                x=float(i),
                open=scale.to_chart_y(opens[i]),
                high=scale.to_chart_y(highs[i]),
                low=scale.to_chart_y(lows[i]),
                close=scale.to_chart_y(closes[i]),
                tooltip=tip,
                show_tooltip=True,
            )
        )

    chart = fchart.CandlestickChart(
        spots=spots,
        min_x=-0.5,
        max_x=float(max(n - 1, 1)) + 0.5,
        min_y=0.0,
        max_y=scale.norm_max,
        left_axis=left_axis,
        bottom_axis=bottom_axis,
        horizontal_grid_lines=fchart.ChartGridLines(
            interval=1.0, color=ThemeHelper.chart_grid_color(page)
        ),
        vertical_grid_lines=fchart.ChartGridLines(
            interval=x_interval, color=ThemeHelper.chart_grid_color(page)
        ),
        tooltip=fchart.CandlestickChartTooltip(
            bgcolor=ThemeHelper.chart_tooltip_bg(page),
            fit_inside_horizontally=True,
            fit_inside_vertically=True,
        ),
        interactive=True,
        height=height,
        expand=expand,
    )
    color = ThemeHelper.chart_named(page, "price")
    return ft.Column(
        [
            ft.Row([legend_chip(ticker, color)], spacing=12),
            ft.Container(
                content=chart,
                padding=ft.padding.only(left=4, right=12, top=4, bottom=8),
                clip_behavior=ft.ClipBehavior.NONE,
            ),
            trend_insight_panel(page, compute_trend_insights(closes, label=ticker)),
        ],
        spacing=8,
        tight=True,
    )
