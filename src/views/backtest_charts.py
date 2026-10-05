"""Chart builders for strategy backtest visualizations."""

from __future__ import annotations

import flet as ft
import flet_charts as fchart
from flet_charts.types import ChartEventType

from src.analysis.backtest_common import TickerChartSeries, TradeMarker
from src.analysis.chart_sampling import (
    CURVED_LINE_MAX_POINTS,
    chart_note_for_sampling,
    downsample_series,
    prepare_ticker_series_for_display,
)
from src.views.components.chart_factory import (
    build_line_chart,
    chart_empty_state,
    legend_chip,
)
from src.views.components.stock_charts import (
    NormalizedYScale,
    normalized_price_axis,
    normalized_signed_axis,
    normalized_time_axis,
)
from src.views.theme import ThemeHelper

_TICKER_COLORS = ("price", "ma_short", "ma_long", "volume", "gain")
_MARKER_RADIUS = 7
_PLOT_HEIGHT = 400
_TRADE_PANEL_MIN_HEIGHT = 72
_PORTFOLIO_PLOT_HEIGHT = 320
_LEFT_AXIS_LABELS_SIZE = 72
_BOTTOM_AXIS_LABELS_SIZE = 48
_Y_TICK_COUNT = 7


def _marker_points(
    markers: list[TradeMarker],
    *,
    side: str,
    color: str,
    scale: NormalizedYScale,
) -> list[ft.LineChartDataPoint]:
    shape = fchart.ChartCirclePoint(
        radius=_MARKER_RADIUS,
        color=color,
        stroke_color=color,
        stroke_width=1,
    )
    points: list[ft.LineChartDataPoint] = []
    for marker in markers:
        if marker.side != side:
            continue
        points.append(
            ft.LineChartDataPoint(
                marker.x,
                scale.to_chart_y(marker.y),
                tooltip=marker.tooltip,
                show_tooltip=False,
                point=shape,
            )
        )
    return points


def _line_points(values: list[float], scale: NormalizedYScale) -> list[ft.LineChartDataPoint]:
    return [
        ft.LineChartDataPoint(i, scale.to_chart_y(v), show_tooltip=False)
        for i, v in enumerate(values)
    ]


def _chart_plot_container(content: ft.Control, *, height: int) -> ft.Container:
    """Fixed-height plot area; no hard clip so axis labels stay visible."""
    return ft.Container(
        content=content,
        height=height,
        padding=ft.padding.only(left=4, right=12, top=8, bottom=12),
        clip_behavior=ft.ClipBehavior.NONE,
    )


def _trade_detail_panel(page: ft.Page, hint: str) -> tuple[ft.Container, ft.Text]:
    text = ft.Text(
        hint,
        size=11,
        color=ThemeHelper.text_primary(page),
        selectable=True,
    )
    panel = ft.Container(
        content=text,
        height=_TRADE_PANEL_MIN_HEIGHT,
        padding=10,
        border=ft.border.all(1, ThemeHelper.border_default(page)),
        border_radius=6,
        bgcolor=ThemeHelper.surface_dim(page),
    )
    return panel, text


def _make_marker_hover_handler(
    trade_text: ft.Text,
    marker_tooltips: dict[tuple[int, int], str],
    price_series_count: int,
    *,
    default_hint: str,
):
    last_shown = {"text": ""}

    def _handler(e: fchart.LineChartEvent) -> None:
        if e.type == ChartEventType.POINTER_EXIT:
            new_text = default_hint
        elif e.type == ChartEventType.POINTER_HOVER and e.spots:
            spot = e.spots[0]
            if spot.bar_index < price_series_count or spot.spot_index < 0:
                new_text = default_hint
            else:
                new_text = marker_tooltips.get(
                    (spot.bar_index, spot.spot_index), default_hint
                )
        else:
            return

        if new_text == last_shown["text"]:
            return
        last_shown["text"] = new_text
        trade_text.value = new_text
        try:
            trade_text.update()
        except RuntimeError:
            pass

    return _handler


def build_performance_chart(
    page: ft.Page,
    series_by_ticker: dict[str, TickerChartSeries],
    *,
    height: int = _PLOT_HEIGHT,
    title: str = "% change from period start",
) -> ft.Control:
    if not series_by_ticker:
        return chart_empty_state(page, "Enter ticker(s) above to view performance charts.")

    default_hint = (
        "Hover over a green (buy) or red (sell) marker on the chart for trade details."
    )
    trade_panel, trade_text = _trade_detail_panel(page, default_hint)

    prepared: list[tuple[str, TickerChartSeries, str]] = []
    marker_tooltips: dict[tuple[int, int], str] = {}
    legend: list[ft.Control] = []
    max_x = 0
    all_y: list[float] = []
    note_text = ""
    first_dates: list[str] = []
    has_markers = False

    for idx, (ticker, raw_series) in enumerate(series_by_ticker.items()):
        series, was_downsampled = prepare_ticker_series_for_display(raw_series)
        if not series.pct_returns:
            continue
        if was_downsampled and not note_text:
            note_text = chart_note_for_sampling(
                len(raw_series.pct_returns), len(series.pct_returns)
            )
        if not first_dates:
            first_dates = series.dates
        color_key = _TICKER_COLORS[idx % len(_TICKER_COLORS)]
        color = ThemeHelper.chart_named(page, color_key)
        max_x = max(max_x, len(series.pct_returns) - 1)
        all_y.extend(series.pct_returns)
        prepared.append((ticker, series, color))
        legend.append(legend_chip(ticker, color))

    if not prepared or not all_y:
        return chart_empty_state(page, "Not enough price data in the selected period.")

    left_axis, scale = normalized_signed_axis(
        page,
        all_y,
        title=title,
        tick_count=_Y_TICK_COUNT,
        suffix="%",
        label_size=_LEFT_AXIS_LABELS_SIZE,
        padding_frac=0.12,
        min_pad=1.0,
    )
    bottom_axis, x_interval = normalized_time_axis(
        page,
        first_dates,
        title="Date",
        fmt="monthly",
        label_size=_BOTTOM_AXIS_LABELS_SIZE,
    )

    line_series: list[ft.LineChartData] = []
    price_series_count = 0
    gain_color = ThemeHelper.chart_named(page, "gain")
    loss_color = ThemeHelper.chart_named(page, "loss")

    for _ticker, series, color in prepared:
        use_curve = len(series.pct_returns) <= CURVED_LINE_MAX_POINTS
        line_series.append(
            ft.LineChartData(
                data_points=_line_points(series.pct_returns, scale),
                color=color,
                stroke_width=2,
                curved=use_curve,
            )
        )
        price_series_count += 1

        buy_markers = [m for m in series.markers if m.side == "buy"]
        buy_pts = _marker_points(series.markers, side="buy", color=gain_color, scale=scale)
        if buy_pts:
            has_markers = True
            bar_idx = len(line_series)
            for spot_idx, marker in enumerate(buy_markers):
                marker_tooltips[(bar_idx, spot_idx)] = marker.tooltip
            line_series.append(
                ft.LineChartData(
                    data_points=buy_pts,
                    color=gain_color,
                    stroke_width=0,
                    point=True,
                    curved=False,
                )
            )
        sell_markers = [m for m in series.markers if m.side == "sell"]
        sell_pts = _marker_points(series.markers, side="sell", color=loss_color, scale=scale)
        if sell_pts:
            has_markers = True
            bar_idx = len(line_series)
            for spot_idx, marker in enumerate(sell_markers):
                marker_tooltips[(bar_idx, spot_idx)] = marker.tooltip
            line_series.append(
                ft.LineChartData(
                    data_points=sell_pts,
                    color=loss_color,
                    stroke_width=0,
                    point=True,
                    curved=False,
                )
            )

    on_event = None
    if has_markers:
        on_event = _make_marker_hover_handler(
            trade_text,
            marker_tooltips,
            price_series_count,
            default_hint=default_hint,
        )

    chart = build_line_chart(
        page,
        line_series,
        min_x=0,
        max_x=max(max_x, 1),
        min_y=0.0,
        max_y=scale.norm_max,
        y_interval=1.0,
        x_interval=x_interval,
        left_axis=left_axis,
        bottom_axis=bottom_axis,
        height=height,
        expand=True,
        interactive=has_markers,
        on_event=on_event,
        use_tooltip=False,
    )

    header: list[ft.Control] = [
        ft.Row(legend, spacing=12, wrap=True),
        ft.Row(
            [
                legend_chip("Buy", ThemeHelper.chart_named(page, "gain")),
                legend_chip("Sell", ThemeHelper.chart_named(page, "loss")),
            ],
            spacing=12,
        ),
    ]
    if note_text:
        header.append(
            ft.Text(note_text, size=10, italic=True, color=ThemeHelper.text_muted(page))
        )

    body: list[ft.Control] = [
        *header,
        _chart_plot_container(chart, height=height),
    ]
    if has_markers:
        body.append(trade_panel)

    return ft.Column(body, spacing=8, tight=True, expand=True)


def build_portfolio_chart(
    page: ft.Page,
    dates: list[str],
    values: list[float],
    *,
    height: int = _PORTFOLIO_PLOT_HEIGHT,
    initial_capital: float = 1.0,
) -> ft.Control:
    if not dates or not values:
        return chart_empty_state(page, "No portfolio curve for the selected period.")

    original_len = len(values)
    display_dates, display_values, _indices = downsample_series(dates, values)
    use_curve = len(display_values) <= CURVED_LINE_MAX_POINTS

    axis_values = list(display_values) + [float(initial_capital)]
    left_axis, scale = normalized_price_axis(
        page,
        axis_values,
        title="Portfolio value ($)",
        tick_count=_Y_TICK_COUNT,
        label_size=_LEFT_AXIS_LABELS_SIZE,
    )
    bottom_axis, x_interval = normalized_time_axis(
        page,
        display_dates,
        title="Date",
        fmt="monthly",
        label_size=_BOTTOM_AXIS_LABELS_SIZE,
    )

    line = ft.LineChartData(
        data_points=_line_points(display_values, scale),
        color=ThemeHelper.chart_named(page, "ma_long"),
        stroke_width=2,
        curved=use_curve,
    )
    chart = build_line_chart(
        page,
        [line],
        min_x=0,
        max_x=max(len(display_values) - 1, 1),
        min_y=0.0,
        max_y=scale.norm_max,
        y_interval=1.0,
        x_interval=x_interval,
        left_axis=left_axis,
        bottom_axis=bottom_axis,
        height=height,
        expand=True,
        interactive=False,
        use_tooltip=False,
    )

    note = chart_note_for_sampling(original_len, len(display_values))
    body: list[ft.Control] = [_chart_plot_container(chart, height=height)]
    if note:
        body.insert(
            0,
            ft.Text(note, size=10, italic=True, color=ThemeHelper.text_muted(page)),
        )
    return ft.Column(body, spacing=6, tight=True, expand=True)


def build_equity_comparison_chart(
    page: ft.Page,
    *,
    strategy_dates: list[str],
    strategy_values: list[float],
    benchmark_dates: list[str] | None = None,
    benchmark_values: list[float] | None = None,
    buy_hold_dates: list[str] | None = None,
    buy_hold_values: list[float] | None = None,
    height: int = _PORTFOLIO_PLOT_HEIGHT,
    initial_capital: float = 10000.0,
) -> ft.Control:
    """Strategy equity vs benchmark and buy-and-hold overlays."""
    if not strategy_dates or not strategy_values:
        return chart_empty_state(page, "Run a backtest to view the equity curve.")

    display_dates, display_values, _ = downsample_series(strategy_dates, strategy_values)
    series_list: list[tuple[str, list[float], str]] = [
        ("Strategy", display_values, "gain"),
    ]

    if benchmark_dates and benchmark_values:
        _, bench_disp, _ = downsample_series(benchmark_dates, benchmark_values)
        if len(bench_disp) == len(display_values):
            series_list.append(("Benchmark", bench_disp, "ma_short"))
    if buy_hold_dates and buy_hold_values:
        _, bh_disp, _ = downsample_series(buy_hold_dates, buy_hold_values)
        if len(bh_disp) == len(display_values):
            series_list.append(("Buy & Hold", bh_disp, "volume"))

    all_vals = [v for _, vals, _ in series_list for v in vals]
    all_vals.append(float(initial_capital))
    left_axis, scale = normalized_price_axis(
        page, all_vals, title="Portfolio ($)", tick_count=_Y_TICK_COUNT,
        label_size=_LEFT_AXIS_LABELS_SIZE,
    )
    bottom_axis, x_interval = normalized_time_axis(
        page, display_dates, title="Date", fmt="monthly", label_size=_BOTTOM_AXIS_LABELS_SIZE,
    )

    line_series: list[ft.LineChartData] = []
    legend: list[ft.Control] = []
    for label, vals, color_key in series_list:
        color = ThemeHelper.chart_named(page, color_key)
        use_curve = len(vals) <= CURVED_LINE_MAX_POINTS
        line_series.append(
            ft.LineChartData(
                data_points=_line_points(vals, scale),
                color=color,
                stroke_width=2,
                curved=use_curve,
            )
        )
        legend.append(legend_chip(label, color))

    chart = build_line_chart(
        page, line_series,
        min_x=0, max_x=max(len(display_values) - 1, 1),
        min_y=0.0, max_y=scale.norm_max, y_interval=1.0, x_interval=x_interval,
        left_axis=left_axis, bottom_axis=bottom_axis,
        height=height, expand=True, interactive=False, use_tooltip=False,
    )
    return ft.Column(
        [
            ft.Row(legend, spacing=12, wrap=True),
            _chart_plot_container(chart, height=height),
        ],
        spacing=8, tight=True, expand=True,
    )


def build_drawdown_chart(
    page: ft.Page,
    drawdown_pct: list[float],
    dates: list[str],
    *,
    height: int = 200,
) -> ft.Control:
    """Underwater / drawdown chart (negative % from peak)."""
    if not drawdown_pct or not dates:
        return chart_empty_state(page, "Drawdown chart appears after a backtest run.")
    display_dates, display_dd, _ = downsample_series(dates, drawdown_pct)
    left_axis, scale = normalized_signed_axis(
        page, display_dd, title="Drawdown (%)", tick_count=5,
        label_size=_LEFT_AXIS_LABELS_SIZE,
    )
    bottom_axis, x_interval = normalized_time_axis(
        page, display_dates, title="", fmt="monthly", label_size=32,
    )
    line = ft.LineChartData(
        data_points=_line_points(display_dd, scale),
        color=ThemeHelper.chart_named(page, "loss"),
        stroke_width=2,
        curved=len(display_dd) <= CURVED_LINE_MAX_POINTS,
    )
    chart = build_line_chart(
        page, [line],
        min_x=0, max_x=max(len(display_dd) - 1, 1),
        min_y=0.0, max_y=scale.norm_max, y_interval=1.0, x_interval=x_interval,
        left_axis=left_axis, bottom_axis=bottom_axis,
        height=height, expand=True, interactive=False, use_tooltip=False,
    )
    return ft.Column(
        [
            ft.Text("Drawdown from peak", size=11, color=ThemeHelper.text_muted(page)),
            _chart_plot_container(chart, height=height),
        ],
        spacing=4, tight=True,
    )


def build_compare_equity_chart(
    page: ft.Page,
    curves: list[tuple[str, list[str], list[float]]],
    *,
    height: int = _PORTFOLIO_PLOT_HEIGHT,
) -> ft.Control:
    """Overlay multiple strategy equity curves for compare mode."""
    if not curves:
        return chart_empty_state(page, "Run a comparison to view overlaid equity curves.")

    color_keys = ("gain", "ma_short", "ma_long", "volume", "price")
    prepared: list[tuple[str, list[str], list[float], str]] = []
    for i, (label, dates, values) in enumerate(curves):
        if not values:
            continue
        disp_dates, disp_vals, _ = downsample_series(
            dates or [str(j) for j in range(len(values))], values
        )
        if not disp_vals:
            continue
        prepared.append((label, disp_dates, disp_vals, color_keys[i % len(color_keys)]))

    if not prepared:
        return chart_empty_state(page, "No equity data to compare.")

    all_vals = [v for _, _, vals, _ in prepared for v in vals]
    axis_dates = max((dates for _, dates, _, _ in prepared), key=len)
    max_len = max(len(vals) for _, _, vals, _ in prepared)

    left_axis, scale = normalized_price_axis(
        page, all_vals, title="Portfolio ($)", tick_count=_Y_TICK_COUNT,
        label_size=_LEFT_AXIS_LABELS_SIZE,
    )
    bottom_axis, x_interval = normalized_time_axis(
        page, axis_dates, title="Date", fmt="monthly", label_size=_BOTTOM_AXIS_LABELS_SIZE,
    )

    line_series: list[ft.LineChartData] = []
    legend: list[ft.Control] = []
    for label, _dates, vals, color_key in prepared:
        color = ThemeHelper.chart_named(page, color_key)
        line_series.append(
            ft.LineChartData(
                data_points=_line_points(vals, scale),
                color=color,
                stroke_width=2,
                curved=len(vals) <= CURVED_LINE_MAX_POINTS,
            )
        )
        legend.append(legend_chip(label, color))

    chart = build_line_chart(
        page, line_series,
        min_x=0, max_x=max(max_len - 1, 1),
        min_y=0.0, max_y=scale.norm_max, y_interval=1.0, x_interval=x_interval,
        left_axis=left_axis, bottom_axis=bottom_axis,
        height=height, expand=True, interactive=False, use_tooltip=False,
    )
    return ft.Column(
        [ft.Row(legend, spacing=12, wrap=True), _chart_plot_container(chart, height=height)],
        spacing=8, tight=True, expand=True,
    )
