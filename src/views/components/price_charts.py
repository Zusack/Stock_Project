"""
Native Flet price charts using real Y values and bounded plot containers.

Single-Ticker OHLC uses CandlestickChart with real dollar Y values.
Dashboard index comparison uses integer-normalized LineChart (stock_charts).
Live intraday charts remain in stock_charts.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft
import flet_charts as fchart
import pandas as pd

from src.views.components.chart_factory import (
    CHART_AXIS_LABEL_SIZE,
    chart_axis_title,
    chart_empty_state,
    compute_trend_insights,
    legend_chip,
    nice_y_axis_bounds,
    spaced_index_ticks,
    trend_insight_panel,
    y_axis_decimals,
)
from src.views.components.stock_charts import SeriesSpec, build_multi_series_chart
from src.views.theme import ThemeHelper

_LEFT_AXIS_LABEL_SIZE = 56
_BOTTOM_AXIS_LABEL_SIZE = 48
_DEFAULT_PLOT_HEIGHT = 320


def _interval_time_fmt(interval_key: str) -> str:
    if interval_key in ("week", "month"):
        return "daily"
    return "monthly"

def _bounded_plot(content: ft.Control, *, height: int) -> ft.Container:
    """Fixed-height plot shell so native charts receive stable layout constraints."""
    return ft.Container(
        content=content,
        height=height,
        padding=ft.padding.only(left=4, right=12, top=8, bottom=12),
        clip_behavior=ft.ClipBehavior.NONE,
    )


def _price_axis_labels(
    y_ticks: list[float],
    page: ft.Page | None,
    *,
    decimals: int,
) -> list[ft.ChartAxisLabel]:
    color = ThemeHelper.chart_axis_label(page)
    labels: list[ft.ChartAxisLabel] = []
    for value in y_ticks:
        if decimals <= 0 and abs(value - round(value)) < 1e-9:
            body = str(int(round(value)))
        else:
            body = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
        labels.append(
            ft.ChartAxisLabel(
                value=value,
                label=ft.Text(f"${body}", size=CHART_AXIS_LABEL_SIZE, color=color),
            )
        )
    return labels


def _date_axis_labels(
    page: ft.Page | None,
    dates: list[str],
    *,
    date_fmt: str = "%Y-%m",
    max_ticks: int = 6,
) -> list[ft.ChartAxisLabel]:
    n = len(dates)
    if n <= 0:
        return [ft.ChartAxisLabel(value=0, label=ft.Text("", size=CHART_AXIS_LABEL_SIZE))]
    color = ThemeHelper.chart_axis_label(page)
    indices = spaced_index_ticks(n, max_ticks=max_ticks)
    labels: list[ft.ChartAxisLabel] = []
    for i in indices:
        raw = dates[min(i, n - 1)]
        try:
            text = pd.Timestamp(raw).strftime(date_fmt)
        except (ValueError, TypeError):
            text = raw[:10] if len(raw) >= 10 else raw
        labels.append(
            ft.ChartAxisLabel(
                value=float(i),
                label=ft.Text(text, size=CHART_AXIS_LABEL_SIZE, color=color),
            )
        )
    return labels


@dataclass(frozen=True)
class IndexSeries:
    """One normalized % line for the dashboard index comparison chart."""

    label: str
    values: list[float]
    color: str
    timestamps: list[str] | None = None


def build_candlestick_chart(
    page: ft.Page | None,
    ticker: str,
    dates: list[str],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    height: int = 360,
) -> ft.Control:
    """OHLC candlestick chart with real dollar Y values and hover tooltips."""
    n = len(closes)
    if n == 0:
        return chart_empty_state(page, f"No OHLC data for {ticker}.")

    all_vals = [float(v) for v in lows + highs]
    min_y, max_y, y_ticks, y_interval = nice_y_axis_bounds(all_vals, tick_count=7)
    decimals = y_axis_decimals(y_ticks)
    x_interval = float(max(1, round((n - 1) / 5.0))) if n > 1 else 1.0

    spots: list[fchart.CandlestickChartSpot] = []
    for i in range(n):
        o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        tip = (
            f"{dates[i]}\n"
            f"O ${o:.2f}\n"
            f"H ${h:.2f}\n"
            f"L ${l:.2f}\n"
            f"C ${c:.2f}"
        )
        spots.append(
            fchart.CandlestickChartSpot(
                x=float(i),
                open=o,
                high=h,
                low=l,
                close=c,
                tooltip=tip,
                show_tooltip=True,
            )
        )

    chart = fchart.CandlestickChart(
        spots=spots,
        min_x=-0.5,
        max_x=float(max(n - 1, 1)) + 0.5,
        min_y=min_y,
        max_y=max_y,
        left_axis=ft.ChartAxis(
            title=chart_axis_title("Price ($)"),
            labels=_price_axis_labels(y_ticks, page, decimals=decimals),
            labels_size=_LEFT_AXIS_LABEL_SIZE,
        ),
        bottom_axis=ft.ChartAxis(
            title=chart_axis_title("Date"),
            labels=_date_axis_labels(page, dates, date_fmt="%Y-%m-%d", max_ticks=6),
            labels_size=_BOTTOM_AXIS_LABEL_SIZE,
        ),
        horizontal_grid_lines=fchart.ChartGridLines(
            interval=y_interval, color=ThemeHelper.chart_grid_color(page)
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
        expand=False,
    )

    gain = ThemeHelper.chart_named(page, "gain")
    loss = ThemeHelper.chart_named(page, "loss")
    return ft.Column(
        [
            ft.Row(
                [
                    legend_chip(ticker, ThemeHelper.chart_named(page, "price")),
                    legend_chip("Up day", gain),
                    legend_chip("Down day", loss),
                ],
                spacing=12,
                wrap=True,
            ),
            _bounded_plot(chart, height=height),
            trend_insight_panel(page, compute_trend_insights(closes, label=ticker)),
        ],
        spacing=8,
        tight=True,
    )


def build_index_comparison_chart(
    page: ft.Page | None,
    series_list: list[IndexSeries],
    dates: list[str] | None = None,
    *,
    interval_key: str = "week",
    height: int = _DEFAULT_PLOT_HEIGHT,
    subtitle: str | None = None,
    insights_label: str = "Index basket",
) -> ft.Control:
    """Multi-series normalized % LineChart (integer Y ticks, same pattern as Live)."""
    specs = [
        SeriesSpec(
            label=spec.label,
            values=spec.values,
            color=spec.color,
            timestamps=spec.timestamps,
        )
        for spec in series_list
        if len(spec.values) >= 2
    ]
    if not specs:
        return chart_empty_state(page, "Not enough index data for the selected interval.")

    return build_multi_series_chart(
        page,
        specs,
        dates or [],
        y_title="% change from interval start",
        x_title="Date",
        time_fmt=_interval_time_fmt(interval_key),
        signed_y=True,
        y_suffix="%",
        y_decimals=1,
        show_zero_baseline=True,
        height=height,
        expand=False,
        subtitle=subtitle,
        insights_label=insights_label,
    )
