"""Multi-ticker performance comparison."""

from __future__ import annotations

import threading

import flet as ft
import pandas as pd

from src.analysis.db import load_prices_bulk
from src.analysis.quote_snapshot import get_quotes_bulk
from src.analysis.ticker_registry import list_focus_symbols
from src.analysis.watchlist_schema import list_members, list_watchlists
from src.services.stock_config import stock_config
from src.utils.format_utils import format_change, format_currency, format_large_number, slice_price_df
from src.views.base_view import BaseView
from src.views.components.chart_factory import chart_empty_state, chart_panel_container
from src.views.components.chart_range import (
    DEFAULT_CHART_RANGE,
    chart_range_fetch_days,
    chart_range_label,
    chart_range_selector,
    selected_chart_range,
)
from src.views.components.feedback import AsyncStatusRow, show_snackbar
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.stock_charts import SeriesSpec, build_multi_series_chart
from src.views.components.tables import themed_data_table
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import navigate_to_stock_detail, on_ticker_field_blur


class CompareView(BaseView):
    _tab_index = 4

    def __init__(self, page: ft.Page):
        super().__init__(page)
        # Do not use expand=True on TextField inside a scrollable Column — Flet 0.85
        # inflates filled inputs to unbounded height (large grey box). Use a fixed width.
        self.ticker_field = InputStyles.text_field(
            page,
            label="Tickers to compare",
            hint_text="AAPL MSFT NVDA",
            width=360,
            on_blur=on_ticker_field_blur(multi=True),
        )
        self.watchlist_dropdown = InputStyles.dropdown(
            page,
            label="Load from watchlist",
            width=220,
            options=[],
            on_select=self._on_load_watchlist,
        )
        self.interval_selector = chart_range_selector(
            selected=DEFAULT_CHART_RANGE,
            on_change=self._on_interval_changed,
        )
        self.async_status = AsyncStatusRow(page)
        self.chart_slot = ft.Container(content=chart_empty_state(page, "Enter tickers and click Compare."))
        self.metrics_table = themed_data_table(
            page,
            columns=[
                ft.DataColumn(ft.Text("Symbol")),
                ft.DataColumn(ft.Text("Price")),
                ft.DataColumn(ft.Text("Change")),
                ft.DataColumn(ft.Text("Market Cap")),
                ft.DataColumn(ft.Text("P/E")),
                ft.DataColumn(ft.Text("Beta")),
            ],
            rows=[],
        )
        self._running = False
        self._latest_prices: dict[str, pd.DataFrame | None] = {}
        self._latest_tickers: list[str] = []

        self.controls = [
            ViewTitleBar("Compare"),
            ft.Divider(),
            ft.Text(
                "Normalized % performance comparison. Uses local price history + live quotes for metrics.",
                size=12,
                color=ThemeHelper.text_muted(page),
            ),
            ft.Row(
                [
                    self.ticker_field,
                    self.watchlist_dropdown,
                    ft.ElevatedButton(
                        "Compare",
                        icon=ft.Icons.COMPARE_ARROWS,
                        style=ButtonStyles.primary(),
                        on_click=self._on_compare,
                    ),
                ],
                spacing=8,
                wrap=True,
            ),
            self.async_status,
            SectionHeader("Performance chart", icon=ft.Icons.SHOW_CHART, page_ref=page),
            ft.Row([self.interval_selector]),
            # No fixed panel_height: legend may wrap, and the plot must stay fully visible.
            chart_panel_container(page, self.chart_slot),
            SectionHeader("Side-by-side metrics", icon=ft.Icons.TABLE_CHART, page_ref=page),
            ft.Container(content=self.metrics_table, padding=4),
        ]

    def refresh_data(self) -> None:
        self.refresh_data_async(label="compare_tab")

    def _selected_interval(self) -> str:
        return selected_chart_range(self.interval_selector)

    def _default_tickers(self) -> list[str]:
        """Symbols from the ticker field, or fall back to watchlist / focus pool."""
        tickers = self._parse_tickers()
        if tickers:
            return tickers
        cfg = stock_config()
        wls = list_watchlists(cfg.db_path)
        if wls:
            tickers = [m.ticker for m in list_members(cfg.db_path, wls[0].id)[:8]]
            if tickers:
                return tickers
        focus = list_focus_symbols(cfg.db_path)
        return [r.symbol for r in focus[:8]]

    def _fetch_data(self):
        cfg = stock_config()
        tickers = self._default_tickers()
        interval = self._selected_interval()
        # Load enough history for every range option so interval switches are instant.
        prices = (
            load_prices_bulk(tickers, cfg.db_path, max_days=chart_range_fetch_days("1y"))
            if tickers
            else {}
        )
        quotes = get_quotes_bulk(cfg.db_path, tickers) if tickers else {}
        return {
            "tickers": tickers,
            "prices": prices,
            "quotes": quotes,
            "interval": interval,
        }

    def _apply_data(self, data: dict) -> None:
        self._refresh_watchlist_options()
        tickers = data.get("tickers") or []
        if tickers and not (self.ticker_field.value or "").strip():
            self.ticker_field.value = " ".join(tickers)
        self._latest_tickers = list(tickers)
        self._latest_prices = data.get("prices") or {}
        interval = data.get("interval") or self._selected_interval()
        self._render_chart(tickers, self._latest_prices, interval)
        self._render_metrics_table(data.get("quotes") or {})
        if tickers:
            self.async_status.set_success(f"Compared {len(tickers)} symbol(s)")
        else:
            self.async_status.set_idle("Enter tickers or load a watchlist, then click Compare.")
        try:
            self.update()
        except RuntimeError:
            pass

    def _refresh_watchlist_options(self) -> None:
        cfg = stock_config()
        wls = list_watchlists(cfg.db_path)
        self.watchlist_dropdown.options = [
            ft.dropdown.Option(str(wl.id), wl.name) for wl in wls
        ]
        try:
            self.watchlist_dropdown.update()
        except RuntimeError:
            pass

    def _parse_tickers(self) -> list[str]:
        raw = (self.ticker_field.value or "").replace(",", " ")
        return [t.strip().upper() for t in raw.split() if t.strip()]

    def _on_load_watchlist(self, e) -> None:
        wl_id = self.watchlist_dropdown.value
        if not wl_id:
            return
        members = list_members(stock_config().db_path, int(wl_id))
        syms = [m.ticker for m in members]
        self.ticker_field.value = " ".join(syms)
        try:
            self.ticker_field.update()
        except RuntimeError:
            pass

    def _on_interval_changed(self, e) -> None:
        if not self._latest_prices:
            return
        interval = selected_chart_range(e.control if e is not None else self.interval_selector)
        self._render_chart(self._latest_tickers, self._latest_prices, interval)

    def _on_compare(self, e) -> None:
        if self._running:
            return
        tickers = self._parse_tickers()
        if len(tickers) < 1:
            show_snackbar(self.page_ref, "Enter at least one ticker.", severity="warning")
            return
        self._running = True
        self.async_status.set_running(f"Loading {len(tickers)} ticker(s)…")
        interval = self._selected_interval()

        def _work():
            cfg = stock_config()
            prices = load_prices_bulk(
                tickers, cfg.db_path, max_days=chart_range_fetch_days("1y")
            )
            quotes = get_quotes_bulk(cfg.db_path, tickers)
            payload = {
                "tickers": tickers,
                "prices": prices,
                "quotes": quotes,
                "interval": interval,
            }

            def _ui():
                self._running = False
                self._apply_data(payload)

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _assign_chart_content(self, content: ft.Control) -> None:
        """Swap chart_slot content. The outer chart_panel_container provides the border/height."""
        self.chart_slot.content = content

    def _slice_df(self, df: pd.DataFrame | None, interval: str) -> pd.DataFrame | None:
        return slice_price_df(df, interval, min_rows=2)

    def _render_chart(
        self,
        tickers: list[str],
        prices: dict[str, pd.DataFrame | None],
        interval: str,
    ) -> None:
        series_specs: list[SeriesSpec] = []
        colors = ["price", "ma_short", "ma_long", "volume", "neutral"]

        for i, sym in enumerate(tickers):
            df = self._slice_df(prices.get(sym), interval)
            if df is None or df.empty or "Adj Close" not in df.columns:
                continue
            close = df["Adj Close"].astype(float).dropna()
            if len(close) < 2:
                continue
            # Normalize from each series' first available bar (IPO start, not
            # chart-axis day 0). Timestamps keep points on their real dates.
            base = float(close.iloc[0])
            if base == 0:
                continue
            pct = [((float(p) / base) - 1.0) * 100.0 for p in close.tolist()]
            dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in close.index]
            color = ThemeHelper.chart_named(self.page_ref, colors[i % len(colors)])
            series_specs.append(
                SeriesSpec(label=sym, values=pct, color=color, timestamps=dates)
            )

        if not series_specs:
            self._assign_chart_content(
                chart_empty_state(
                    self.page_ref, "Not enough price history for comparison."
                )
            )
        else:
            self._assign_chart_content(
                build_multi_series_chart(
                    self.page_ref,
                    series_specs,
                    [],
                    y_title="% change",
                    signed_y=True,
                    show_zero_baseline=True,
                    height=340,
                    expand=False,
                    subtitle=(
                        f"Normalized from each series' first bar · "
                        f"{chart_range_label(interval)}"
                    ),
                )
            )
        try:
            self.chart_slot.update()
        except RuntimeError:
            pass

    def _render_metrics_table(self, quotes: dict) -> None:
        rows = []
        for sym, q in sorted(quotes.items()):
            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.TextButton(
                                sym,
                                on_click=lambda e, s=sym: navigate_to_stock_detail(s, run_analysis=False),
                            )
                        ),
                        ft.DataCell(ft.Text(format_currency(q.last_price))),
                        ft.DataCell(ft.Text(format_change(q.change, q.change_pct))),
                        ft.DataCell(ft.Text(format_large_number(q.market_cap))),
                        ft.DataCell(
                            ft.Text(f"{q.trailing_pe:.1f}" if q.trailing_pe else "—")
                        ),
                        ft.DataCell(ft.Text(f"{q.beta:.2f}" if q.beta is not None else "—")),
                    ]
                )
            )
        self.metrics_table.rows = rows or [
            ft.DataRow(cells=[ft.DataCell(ft.Text("—"))] * 6)
        ]
        try:
            self.metrics_table.update()
        except RuntimeError:
            pass

    def refresh_theme(self) -> None:
        InputStyles.refresh_fields(
            self.page_ref,
            [self.ticker_field],
            [self.watchlist_dropdown],
        )
        try:
            self.update()
        except RuntimeError:
            pass
