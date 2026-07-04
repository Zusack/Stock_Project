"""Live intraday charts: Finnhub WebSocket streaming + Yahoo backfill."""

from __future__ import annotations

import re
import threading
import time

import flet as ft
import pandas as pd

from src.analysis.db import load_intraday_bars
from src.analysis.intraday_ingest import backfill_intraday, backfill_intraday_today
from src.analysis.ticker_registry import list_focus_symbols
from src.services.live_stream_service import LiveBarSnapshot, live_stream_service
from src.services.stock_config import stock_config
from src.views.base_view import BaseView
from src.views.components.chart_factory import (
    CHART_MIN_PLOT_HEIGHT,
    chart_empty_state,
    chart_loading_state,
    chart_panel_container,
    dynamic_content_slot,
)
from src.views.components.stock_charts import build_price_line_chart
from src.views.components.feedback import AsyncStatusRow, show_snackbar
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import on_ticker_field_blur, parse_symbols



def _bars_to_series(bars: list[LiveBarSnapshot]) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    closes: list[float] = []
    for bar in bars:
        labels.append(bar.timestamp)
        closes.append(float(bar.close))
    return labels, closes


class LiveView(BaseView):
    _tab_index = 8

    def __init__(self, page: ft.Page):
        super().__init__(page)
        self._stream = live_stream_service()
        self._backfill_running = False
        self._stream_starting = False
        self._waiting_for_live = False
        self._stream_started_at: float | None = None

        focus_symbols = self._load_focus_symbols()
        default_sym = focus_symbols[0] if focus_symbols else ""

        if focus_symbols:
            self.focus_pick_dropdown = InputStyles.dropdown(
                page,
                label="Focus watchlist",
                value=default_sym,
                width=180,
                options=[ft.dropdown.Option(s) for s in focus_symbols],
                on_select=self._on_focus_pick,
                tooltip="Quick-pick from your focus watchlist (Watchlist tab).",
            )
            self._focus_pick_control: ft.Control = self.focus_pick_dropdown
        else:
            self.focus_pick_dropdown = None
            self._focus_pick_control = ft.Text(
                "No focus symbols yet — add tickers on the Watchlist tab or type below.",
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
            tooltip="Symbol shown on the intraday chart.",
        )
        self.stream_symbols_field = InputStyles.text_field(
            page,
            label="Also stream",
            hint_text="MSFT, NVDA (optional)",
            width=360,
            on_blur=on_ticker_field_blur(multi=True),
            tooltip=(
                "Extra symbols to subscribe on the live feed (comma/space separated). "
                "Combined with focus watchlist unless Settings uses chart-only mode."
            ),
        )
        self.start_btn = ft.ElevatedButton(
            "Start stream",
            icon=ft.Icons.PLAY_ARROW,
            style=ButtonStyles.primary(),
            on_click=self._on_start_stream,
        )
        self.stop_btn = ft.OutlinedButton(
            "Stop stream",
            icon=ft.Icons.STOP,
            on_click=self._on_stop_stream,
        )
        self.backfill_btn = ft.OutlinedButton(
            "Backfill history",
            icon=ft.Icons.CLOUD_DOWNLOAD,
            on_click=self._on_backfill,
            tooltip="Download recent minute bars from Yahoo Finance for focus watchlist.",
        )
        self.async_status = AsyncStatusRow(page)
        self.stream_status = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self._chart_placeholder = "Select a symbol and start streaming."
        self.chart_slot = dynamic_content_slot(page, self._chart_placeholder)

        self._setup_pubsub({"intraday_bar": self._on_intraday_bar_event})
        self.controls = [
            ViewTitleBar("Live Intraday"),
            ft.Divider(),
            ft.Text(
                "Live 1-minute bars via Finnhub (real-time trades aggregated), "
                "history backfilled from Yahoo Finance. "
                "Add your free Finnhub key under Settings → Intraday / Live Data. "
                "Symbols come from the focus watchlist, the chart symbol field, and "
                "optional 'Also stream' entries.",
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
        ]
        self._refresh_chart_from_sources()

    @staticmethod
    def _load_focus_symbols() -> list[str]:
        rows = list_focus_symbols(stock_config().db_path)
        return [r.symbol for r in rows if r.symbol and not str(r.symbol).startswith("^")]

    def _extra_stream_symbols(self) -> list[str]:
        return parse_symbols(self.stream_symbols_field.value or "", exclude_indices=True)

    def _current_symbol(self) -> str:
        return (self.chart_symbol_field.value or "").strip().upper()

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

    def _update_stream_status(self, *, bars: list[LiveBarSnapshot] | None = None) -> None:
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
        if bars is not None:
            bar_note = f" · {len(bars)} bar(s) on chart"
        self.stream_status.value = f"Streaming: {syms}{suffix}{bar_note}"

    def _refresh_chart_from_sources(self) -> None:
        sym = self._current_symbol()
        if not sym:
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
            except RuntimeError:
                pass
            return

        bars = self._merge_bars(sym)
        if self._waiting_for_live and len(bars) < 2:
            self.chart_slot.content = chart_loading_state(
                self.page_ref,
                "Waiting for live trades…",
                submessage=self._live_wait_submessage(),
                height=CHART_MIN_PLOT_HEIGHT,
            )
        else:
            if self._waiting_for_live and len(bars) >= 2:
                self._waiting_for_live = False
            timestamps, closes = _bars_to_series(bars)
            self.chart_slot.content = build_price_line_chart(
                self.page_ref,
                sym,
                timestamps,
                closes,
                height=360,
                time_fmt="intraday",
            )
        self.chart_slot.height = None
        self.chart_slot.expand = False
        self._update_stream_status(bars=bars)
        try:
            self.chart_slot.update()
            self.stream_status.update()
        except RuntimeError:
            pass

    def _on_intraday_bar_event(self, *, ticker: str, bar: LiveBarSnapshot | None = None, **_) -> None:
        sym = self._current_symbol()
        if not sym or ticker.upper() != sym:
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
        if self._stream_starting:
            return
        self._stream_starting = True
        self._waiting_for_live = False
        self._stream_started_at = None
        self.async_status.set_running(
            f"Loading today's session for {sym}…",
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
                if result.get("ok"):
                    self._stream_started_at = time.time()
                    backfill_ok = backfill_summary.get("success", 0)
                    backfill_total = backfill_summary.get("total", 0)
                    chart_bars = len(self._merge_bars(sym))
                    if chart_bars >= 2:
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
                try:
                    self.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _on_stop_stream(self, e) -> None:
        self._stream.stop()
        self._stream_starting = False
        self._waiting_for_live = False
        self._stream_started_at = None
        self.async_status.set_idle("Stream stopped.")
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
            sym = self._current_symbol()
            tickers = [sym] if sym else None

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
        self.async_status.refresh_theme(self.page_ref)
        try:
            self.update()
        except RuntimeError:
            pass

    def on_tab_deactivated(self) -> None:
        pass

    def on_tab_activated(self) -> None:
        self.refresh_data()
