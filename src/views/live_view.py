"""Live intraday charts: Finnhub WebSocket streaming + Yahoo backfill."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import flet as ft
import pandas as pd

from src.analysis.db import load_intraday_bars
from src.analysis.intraday_ingest import backfill_intraday, backfill_intraday_today
from src.analysis.watchlist_schema import resolve_watchlist_symbols
from src.services.live_stream_service import LiveBarSnapshot, live_stream_service
from src.services.stock_config import stock_config
from src.utils.format_utils import format_change, format_currency
from src.views.base_view import BaseView
from src.views.components.chart_factory import (
    CHART_MIN_PLOT_HEIGHT,
    chart_empty_state,
    chart_loading_state,
    chart_panel_container,
    dynamic_content_slot,
)
from src.views.components.feedback import AsyncStatusRow, show_snackbar
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.stock_charts import SeriesSpec, build_multi_series_chart, build_price_line_chart
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import on_ticker_field_blur, parse_symbols

_CHART_COLOR_KEYS = ("price", "ma_short", "ma_long", "volume", "warning")


@dataclass(frozen=True)
class SessionDayStats:
    ticker: str
    last: float
    session_open: float
    change: float
    change_pct: float
    bar_count: int


def _bars_to_series(bars: list[LiveBarSnapshot]) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    closes: list[float] = []
    for bar in bars:
        labels.append(bar.timestamp)
        closes.append(float(bar.close))
    return labels, closes


def resolve_chart_symbols(chart_symbol: str, extra_symbols: list[str] | None = None) -> list[str]:
    """Deduped symbols shown on the Live plot: chart symbol first, then Also stream."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in [chart_symbol, *(extra_symbols or [])]:
        sym = str(raw or "").strip().upper()
        if not sym or sym.startswith("^") or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def session_day_stats(ticker: str, bars: list[LiveBarSnapshot]) -> SessionDayStats | None:
    """Day move from the session's first bar open to the latest close."""
    if not bars:
        return None
    open_px = float(bars[0].open)
    last = float(bars[-1].close)
    if open_px == 0:
        return None
    change = last - open_px
    return SessionDayStats(
        ticker=ticker.strip().upper(),
        last=last,
        session_open=open_px,
        change=change,
        change_pct=(last / open_px - 1.0) * 100.0,
        bar_count=len(bars),
    )


def _bars_to_session_pct_series(
    bars: list[LiveBarSnapshot],
) -> tuple[list[str], list[float]] | None:
    """Normalize closes to % change from the session open (first bar open)."""
    if not bars:
        return None
    base = float(bars[0].open)
    if base == 0:
        return None
    timestamps = [b.timestamp for b in bars]
    pct = [((float(b.close) / base) - 1.0) * 100.0 for b in bars]
    return timestamps, pct


class LiveView(BaseView):
    _tab_index = 8

    def __init__(self, page: ft.Page):
        super().__init__(page)
        self._stream = live_stream_service()
        self._backfill_running = False
        self._stream_starting = False
        self._start_in_flight = False
        self._waiting_for_live = False
        self._stream_started_at: float | None = None

        focus_symbols = self._load_focus_symbols()
        default_sym = focus_symbols[0] if focus_symbols else ""

        if focus_symbols:
            self.focus_pick_dropdown = InputStyles.dropdown(
                page,
                label="Watchlist",
                value=default_sym,
                width=180,
                options=[ft.dropdown.Option(s) for s in focus_symbols],
                on_select=self._on_focus_pick,
                tooltip="Quick-pick from your saved watchlist (Watchlists tab).",
            )
            self._focus_pick_control: ft.Control = self.focus_pick_dropdown
        else:
            self.focus_pick_dropdown = None
            self._focus_pick_control = ft.Text(
                "No watchlist symbols yet — add tickers on the Watchlists tab or type below.",
                size=12,
                color=ThemeHelper.text_muted(page),
            )
        self.chart_symbol_field = InputStyles.text_field(
            page,
            label="Chart symbol",
            value=default_sym,
            width=140,
            hint_text="e.g. AAPL",
            on_blur=on_ticker_field_blur(multi=False),
            tooltip="Primary symbol on the shared intraday chart.",
        )
        self.stream_symbols_field = InputStyles.text_field(
            page,
            label="Also stream",
            hint_text="MSFT, NVDA (optional)",
            width=360,
            on_blur=on_ticker_field_blur(multi=True),
            tooltip=(
                "Extra symbols plotted on the same chart and subscribed on the live feed "
                "(comma/space separated). Watchlist symbols are also subscribed "
                "unless Settings uses chart-only mode."
            ),
        )
        self.start_btn = ft.ElevatedButton(
            "Start stream",
            icon=ft.Icons.PLAY_ARROW,
            style=ButtonStyles.primary(page),
            on_click=self._on_start_stream,
        )
        self.stop_btn = ft.ElevatedButton(
            "Stop stream",
            icon=ft.Icons.STOP,
            style=ButtonStyles.secondary(page),
            disabled=True,
            on_click=self._on_stop_stream,
        )
        self.backfill_btn = ft.OutlinedButton(
            "Backfill history",
            icon=ft.Icons.CLOUD_DOWNLOAD,
            on_click=self._on_backfill,
            tooltip="Download recent minute bars from Yahoo Finance for chart + Also stream symbols.",
        )
        self.async_status = AsyncStatusRow(page)
        self.stream_status = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self._chart_placeholder = "Select a symbol and start streaming."
        self.chart_slot = dynamic_content_slot(page, self._chart_placeholder)
        self.session_summary = ft.Column(
            [
                ft.Text(
                    "Session summary appears after bars load.",
                    size=12,
                    color=ThemeHelper.text_muted(page),
                )
            ],
            spacing=4,
            tight=True,
        )

        self._setup_pubsub({"intraday_bar": self._on_intraday_bar_event})
        self.controls = [
            ViewTitleBar("Live Intraday"),
            ft.Divider(),
            ft.Text(
                "Live 1-minute bars via Finnhub (real-time trades aggregated), "
                "history backfilled from Yahoo Finance. "
                "Add your free Finnhub key under Settings → Intraday / Live Data. "
                "Chart symbol and Also stream share one plot; your watchlist "
                "is also subscribed for live data unless Settings uses chart-only mode.",
                size=12,
                color=ThemeHelper.text_muted(page),
            ),
            ft.Row(
                [
                    self._focus_pick_control,
                    self.chart_symbol_field,
                ],
                spacing=12,
                wrap=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Row(
                [
                    self.stream_symbols_field,
                    self.start_btn,
                    self.stop_btn,
                    self.backfill_btn,
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            self.async_status,
            self.stream_status,
            SectionHeader("Intraday chart", icon=ft.Icons.CANDLESTICK_CHART, page_ref=page),
            chart_panel_container(
                page,
                self.chart_slot,
                min_height=CHART_MIN_PLOT_HEIGHT + 200,
            ),
            SectionHeader("Session summary", icon=ft.Icons.SHOW_CHART, page_ref=page),
            self.session_summary,
        ]
        self._sync_stream_buttons()
        self._refresh_chart_from_sources()

    def _sync_stream_buttons(self) -> None:
        """Enable Start/Stop from stream state (matches other Start/Stop tabs)."""
        running = bool(self._stream.is_running)
        self.start_btn.disabled = running or self._start_in_flight
        self.stop_btn.disabled = not running
        try:
            self.start_btn.update()
            self.stop_btn.update()
        except RuntimeError:
            pass

    @staticmethod
    def _load_focus_symbols() -> list[str]:
        return [
            s
            for s in resolve_watchlist_symbols(stock_config().db_path)
            if s and not str(s).startswith("^")
        ]

    def _extra_stream_symbols(self) -> list[str]:
        return parse_symbols(self.stream_symbols_field.value or "", exclude_indices=True)

    def _current_symbol(self) -> str:
        return (self.chart_symbol_field.value or "").strip().upper()

    def _chart_symbols(self) -> list[str]:
        return resolve_chart_symbols(self._current_symbol(), self._extra_stream_symbols())

    def _on_focus_pick(self, e) -> None:
        if self.focus_pick_dropdown is None:
            return
        picked = (self.focus_pick_dropdown.value or "").strip().upper()
        if not picked:
            return
        self.chart_symbol_field.value = picked
        try:
            self.chart_symbol_field.update()
        except RuntimeError:
            pass

    def _sync_focus_dropdown(self) -> None:
        if self.focus_pick_dropdown is None:
            return
        focus_symbols = self._load_focus_symbols()
        self.focus_pick_dropdown.options = [ft.dropdown.Option(s) for s in focus_symbols]
        current = self._current_symbol()
        if current and current in focus_symbols:
            self.focus_pick_dropdown.value = current
        elif focus_symbols:
            self.focus_pick_dropdown.value = focus_symbols[0]
        try:
            self.focus_pick_dropdown.update()
        except RuntimeError:
            pass

    def _load_db_bars(self, sym: str) -> list[LiveBarSnapshot]:
        cfg = stock_config()
        df = load_intraday_bars(sym, cfg.db_path, interval=cfg.intraday_interval, max_points=390)
        if df is None or df.empty:
            return []
        out: list[LiveBarSnapshot] = []
        for ts, row in df.iterrows():
            out.append(
                LiveBarSnapshot(
                    ticker=sym,
                    timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    interval=cfg.intraday_interval,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row.get("Volume") or 0),
                    vwap=float(row["Vwap"]) if "Vwap" in row and pd.notna(row["Vwap"]) else None,
                    trade_count=int(row["Trade_Count"])
                    if "Trade_Count" in row and pd.notna(row["Trade_Count"])
                    else None,
                )
            )
        return out

    def _merge_bars(self, sym: str) -> list[LiveBarSnapshot]:
        """Combine persisted intraday history with the live ring buffer (buffer wins on ties)."""
        db_bars = self._load_db_bars(sym)
        buffer = self._stream.get_buffer(sym)
        if not buffer:
            return db_bars
        by_ts: dict[str, LiveBarSnapshot] = {b.timestamp: b for b in db_bars}
        for bar in buffer:
            by_ts[bar.timestamp] = bar
        return sorted(by_ts.values(), key=lambda b: b.timestamp)

    def _live_wait_submessage(self) -> str:
        cfg = stock_config()
        interval = cfg.intraday_interval or "1Min"
        elapsed = ""
        if self._stream_started_at is not None:
            secs = int(time.time() - self._stream_started_at)
            elapsed = f" ({secs}s elapsed)"
        return (
            f"Connected — aggregating trades into {interval} bars. "
            f"Each completed bar appears when the minute rolls over{elapsed}."
        )

    def _update_stream_status(self, *, bar_count: int | None = None) -> None:
        if self._stream_starting:
            self.stream_status.value = "Preparing live stream…"
            return
        if self._waiting_for_live:
            self.stream_status.value = self._live_wait_submessage()
            return
        running = self._stream.is_running
        if not running:
            self.stream_status.value = "Stream stopped."
            return
        syms = ", ".join(self._stream.subscribed_symbols[:8])
        suffix = "…" if len(self._stream.subscribed_symbols) > 8 else ""
        bar_note = ""
        if bar_count is not None:
            plotted = ", ".join(self._chart_symbols()[:6])
            bar_note = f" · {bar_count} bar(s) · chart: {plotted}"
        self.stream_status.value = f"Streaming: {syms}{suffix}{bar_note}"

    def _render_session_summary(self, stats: list[SessionDayStats]) -> None:
        page = self.page_ref
        if not stats:
            self.session_summary.controls = [
                ft.Text(
                    "No session bars yet for the chart symbols.",
                    size=12,
                    color=ThemeHelper.text_muted(page),
                )
            ]
            return

        rows: list[ft.Control] = [
            ft.Text(
                "Day move vs session open (first minute bar).",
                size=11,
                color=ThemeHelper.text_muted(page),
            )
        ]
        for item in stats:
            if item.change_pct > 0:
                chg_color = ThemeHelper.accent_green(page)
            elif item.change_pct < 0:
                chg_color = ThemeHelper.text_error(page)
            else:
                chg_color = ThemeHelper.text_primary(page)
            rows.append(
                ft.Row(
                    [
                        ft.Text(
                            item.ticker,
                            size=13,
                            weight=ft.FontWeight.W_600,
                            width=72,
                            color=ThemeHelper.text_primary(page),
                        ),
                        ft.Text(
                            format_currency(item.last),
                            size=13,
                            width=96,
                            color=ThemeHelper.text_primary(page),
                        ),
                        ft.Text(
                            format_change(item.change, item.change_pct),
                            size=13,
                            color=chg_color,
                        ),
                        ft.Text(
                            f"{item.bar_count} bars",
                            size=11,
                            color=ThemeHelper.text_muted(page),
                        ),
                    ],
                    spacing=16,
                    wrap=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )
        self.session_summary.controls = rows

    def _build_chart_content(
        self,
        symbols: list[str],
        bars_by_sym: dict[str, list[LiveBarSnapshot]],
    ) -> ft.Control:
        if len(symbols) == 1:
            sym = symbols[0]
            bars = bars_by_sym.get(sym) or []
            timestamps, closes = _bars_to_series(bars)
            return build_price_line_chart(
                self.page_ref,
                sym,
                timestamps,
                closes,
                height=360,
                time_fmt="intraday",
            )

        series_specs: list[SeriesSpec] = []
        for i, sym in enumerate(symbols):
            bars = bars_by_sym.get(sym) or []
            series = _bars_to_session_pct_series(bars)
            if series is None or len(series[1]) < 1:
                continue
            timestamps, pct = series
            if len(pct) == 1:
                timestamps = [timestamps[0], timestamps[0]]
                pct = [pct[0], pct[0]]
            color_key = _CHART_COLOR_KEYS[i % len(_CHART_COLOR_KEYS)]
            series_specs.append(
                SeriesSpec(
                    label=sym,
                    values=pct,
                    color=ThemeHelper.chart_named(self.page_ref, color_key),
                    timestamps=timestamps,
                )
            )

        if not series_specs:
            return chart_empty_state(
                self.page_ref,
                "No intraday data for the selected chart symbols.",
            )

        return build_multi_series_chart(
            self.page_ref,
            series_specs,
            [],
            y_title="% vs open",
            x_title="Time (UTC)",
            time_fmt="intraday",
            signed_y=True,
            show_zero_baseline=True,
            height=360,
            expand=False,
            subtitle="Normalized from each symbol's session open · shared timeline",
            insights_label=series_specs[0].label,
            extra_insights=[
                f"{s.ticker}: {format_change(s.change, s.change_pct)} from open"
                for s in (
                    session_day_stats(sym, bars_by_sym.get(sym) or [])
                    for sym in symbols
                )
                if s is not None
            ],
        )

    def _refresh_chart_from_sources(self) -> None:
        symbols = self._chart_symbols()
        if not symbols:
            self._render_session_summary([])
            return
        if self._stream_starting:
            self.chart_slot.content = chart_loading_state(
                self.page_ref,
                "Loading today's intraday data…",
                submessage="Fetching minute bars from Yahoo Finance before connecting live.",
                height=CHART_MIN_PLOT_HEIGHT,
            )
            self._update_stream_status()
            try:
                self.chart_slot.update()
                self.stream_status.update()
                self.session_summary.update()
            except RuntimeError:
                pass
            return

        bars_by_sym = {sym: self._merge_bars(sym) for sym in symbols}
        max_bars = max((len(b) for b in bars_by_sym.values()), default=0)
        stats = [
            s
            for sym in symbols
            if (s := session_day_stats(sym, bars_by_sym.get(sym) or [])) is not None
        ]

        if self._waiting_for_live and max_bars < 2:
            self.chart_slot.content = chart_loading_state(
                self.page_ref,
                "Waiting for live trades…",
                submessage=self._live_wait_submessage(),
                height=CHART_MIN_PLOT_HEIGHT,
            )
        else:
            if self._waiting_for_live and max_bars >= 2:
                self._waiting_for_live = False
            self.chart_slot.content = self._build_chart_content(symbols, bars_by_sym)

        self.chart_slot.height = None
        self.chart_slot.expand = False
        self._render_session_summary(stats)
        self._update_stream_status(bar_count=max_bars)
        try:
            self.chart_slot.update()
            self.stream_status.update()
            self.session_summary.update()
        except RuntimeError:
            pass

    def _on_intraday_bar_event(self, *, ticker: str, bar: LiveBarSnapshot | None = None, **_) -> None:
        chart_syms = set(self._chart_symbols())
        if not chart_syms or ticker.upper() not in chart_syms:
            return
        if self._waiting_for_live:
            self._waiting_for_live = False

        def _ui():
            if not self._is_active_tab():
                return
            self._refresh_chart_from_sources()

        self._safe_update(_ui)

    def _on_start_stream(self, e) -> None:
        sym = self._current_symbol()
        if not sym:
            show_snackbar(self.page_ref, "Select a symbol.", severity="warning")
            return
        if self._stream_starting or self._start_in_flight or self._stream.is_running:
            return
        chart_syms = self._chart_symbols()
        self._stream_starting = True
        self._start_in_flight = True
        self._waiting_for_live = False
        self._stream_started_at = None
        self._sync_stream_buttons()
        label = ", ".join(chart_syms[:4])
        if len(chart_syms) > 4:
            label += "…"
        self.async_status.set_running(
            f"Loading today's session for {label}…",
            progress=0.05,
        )
        self._refresh_chart_from_sources()
        try:
            self.page_ref.update()
        except RuntimeError:
            pass

        extra = self._extra_stream_symbols()
        stream_symbols = self._stream.resolve_stream_symbols(
            chart_symbol=sym,
            extra_symbols=extra,
        )

        def _work():
            def _report(pct: float, msg: str) -> None:
                self.async_status.set_running(msg, progress=pct)

            backfill_summary = backfill_intraday_today(
                tickers=stream_symbols,
                progress_callback=_report,
            )

            def _after_backfill():
                self._stream_starting = False
                self.async_status.set_running(
                    "Connecting to Finnhub live feed…",
                    progress=0.85,
                )
                self._sync_stream_buttons()
                self._refresh_chart_from_sources()
                try:
                    self.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_after_backfill)

            result = self._stream.start(
                chart_symbol=sym,
                extra_symbols=extra,
            )

            def _ui():
                self._stream_starting = False
                self._start_in_flight = False
                if result.get("ok"):
                    self._stream_started_at = time.time()
                    backfill_ok = backfill_summary.get("success", 0)
                    backfill_total = backfill_summary.get("total", 0)
                    max_bars = max(
                        (len(self._merge_bars(s)) for s in chart_syms),
                        default=0,
                    )
                    if max_bars >= 2:
                        self._waiting_for_live = False
                        self.async_status.set_success(
                            f"Live stream active ({len(result.get('symbols', []))} symbols). "
                            f"Session loaded: {backfill_ok}/{backfill_total}."
                        )
                    else:
                        self._waiting_for_live = True
                        self.async_status.set_running(
                            "Live feed connected — waiting for trades to build the chart…",
                        )
                    self._refresh_chart_from_sources()
                else:
                    self._waiting_for_live = False
                    self._stream_started_at = None
                    self.async_status.set_error(result.get("error", "Stream failed.")[:200])
                    show_snackbar(
                        self.page_ref,
                        result.get("error", "Stream failed."),
                        severity="error",
                    )
                    self._refresh_chart_from_sources()
                self._sync_stream_buttons()
                try:
                    self.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _on_stop_stream(self, e) -> None:
        if not self._stream.is_running:
            self._sync_stream_buttons()
            return
        self._stream.stop()
        self._stream_starting = False
        self._start_in_flight = False
        self._waiting_for_live = False
        self._stream_started_at = None
        self.async_status.set_idle("Stream stopped.")
        self._sync_stream_buttons()
        self._refresh_chart_from_sources()
        show_snackbar(self.page_ref, "Live stream stopped.", severity="info")

    def _on_backfill(self, e) -> None:
        if self._backfill_running:
            return
        self._backfill_running = True
        self.backfill_btn.disabled = True
        self.async_status.set_running("Backfilling intraday history…")
        try:
            self.page_ref.update()
        except RuntimeError:
            pass

        def _work():
            tickers = self._chart_symbols() or None

            def _progress(pct: float, msg: str) -> None:
                self.async_status.set_running(msg)

            summary = backfill_intraday(tickers=tickers, progress_callback=_progress)

            def _ui():
                self._backfill_running = False
                self.backfill_btn.disabled = False
                if summary.get("error"):
                    self.async_status.set_error(summary["error"][:200])
                    show_snackbar(self.page_ref, summary["error"], severity="error")
                else:
                    ok = summary.get("success", 0)
                    total = summary.get("total", 0)
                    self.async_status.set_success(f"Backfill complete ({ok}/{total} symbols).")
                    show_snackbar(
                        self.page_ref,
                        f"Intraday backfill: {ok}/{total} symbols with data.",
                        severity="success",
                    )
                    self._refresh_chart_from_sources()
                try:
                    self.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def refresh_data(self) -> None:
        self._refresh_capture = {"sym": self._current_symbol()}
        self.refresh_data_async(label="live_tab")

    def _fetch_data(self) -> dict:
        sym = getattr(self, "_refresh_capture", {}).get("sym") or ""
        return {"sym": sym, "focus": self._load_focus_symbols()}

    def _apply_data(self, data: dict) -> None:
        focus = data.get("focus") or []
        if self.focus_pick_dropdown is not None:
            self.focus_pick_dropdown.options = [ft.dropdown.Option(s) for s in focus]
            current = (data.get("sym") or "").strip().upper()
            if current and current in focus:
                self.focus_pick_dropdown.value = current
            elif focus:
                self.focus_pick_dropdown.value = focus[0]
        if not (data.get("sym") or "").strip() and focus:
            self.chart_symbol_field.value = focus[0]
        self._sync_stream_buttons()
        self._refresh_chart_from_sources()
        try:
            self.update()
        except RuntimeError:
            pass

    def refresh_theme(self) -> None:
        dropdowns = [self.focus_pick_dropdown] if self.focus_pick_dropdown is not None else []
        InputStyles.refresh_fields(
            self.page_ref,
            [self.chart_symbol_field, self.stream_symbols_field],
            dropdowns,
        )
        self.start_btn.style = ButtonStyles.primary(self.page_ref)
        self.stop_btn.style = ButtonStyles.secondary(self.page_ref)
        self.async_status.refresh_theme(self.page_ref)
        self._sync_stream_buttons()
        try:
            self.update()
        except RuntimeError:
            pass

    def on_tab_deactivated(self) -> None:
        pass

    def on_tab_activated(self) -> None:
        self._sync_stream_buttons()
        self.refresh_data()
