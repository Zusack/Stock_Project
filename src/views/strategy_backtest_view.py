"""Strategy backtests — general, hybrid, technical, CANSLIM."""

from __future__ import annotations

import threading

import flet as ft
import pandas as pd

from src.analysis.backtest_common import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PORTFOLIO_START,
    parse_lookback_days,
    parse_portfolio_start,
)
from src.analysis.symbol_universe import resolve_universe
from src.services.stock_config import stock_config
from src.views.backtest_charts import build_performance_chart, build_portfolio_chart
from src.views.base_view import BaseView
from src.views.components.chart_factory import chart_panel_container, dynamic_content_slot
from src.views.components.feedback import show_snackbar
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.tables import bordered_table_wrap, themed_data_table
from src.views.strategy_constants import STRATEGY_INFO
from src.views.strategy_runners import (
    run_canslim_strategy,
    run_general_strategy,
    run_hybrid_strategy,
    run_technical_strategy,
)
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import (
    dataframe_columns,
    dataframe_to_rows,
    on_ticker_field_blur,
    parse_ticker_filter,
)


class _BacktestPanel(ft.Container):
    """One strategy body: tables (left) and charts (right) in a single scroll region."""

    def __init__(
        self,
        page: ft.Page,
        parent: "StrategyBacktestView",
        title: str,
        *,
        extra_controls: list[ft.Control] | None = None,
        run_tooltip: str | None = None,
    ):
        super().__init__(expand=True)
        self.page_ref = page
        self.parent_view = parent
        self._title = title
        self.summary = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self.results_table = themed_data_table(page, columns=[], rows=[])
        self.trades_table = themed_data_table(page, columns=[], rows=[])
        self.performance_chart_slot = dynamic_content_slot(
            page,
            "Enter ticker(s) in the sidebar, then run a backtest to view charts.",
        )
        self.portfolio_chart_slot = dynamic_content_slot(
            page,
            "Portfolio value chart appears after a run with ticker(s) specified.",
        )
        self.run_btn = ft.ElevatedButton(
            f"Run {title}",
            icon=ft.Icons.PLAY_ARROW,
            style=ButtonStyles.primary(),
            on_click=self._on_run,
            tooltip=run_tooltip
            or f"Run the {title} backtest using the period, capital, and ticker filter in the sidebar.",
        )
        extras = ft.Row(extra_controls, spacing=12) if extra_controls else None

        tables_column = ft.Column(
            [
                *([extras] if extras is not None else []),
                ft.Row([self.run_btn]),
                self.summary,
                SectionHeader("Summary table", icon=ft.Icons.TABLE_CHART, page_ref=page),
                bordered_table_wrap(page, self.results_table, expand=True),
                SectionHeader("Trade log", icon=ft.Icons.SWAP_HORIZ, page_ref=page),
                bordered_table_wrap(page, self.trades_table, expand=True),
            ],
            spacing=8,
            tight=True,
            expand=2,
            scroll=ft.ScrollMode.AUTO,
        )

        charts_column = ft.Column(
            [
                SectionHeader(
                    "Price performance (% from period start)",
                    icon=ft.Icons.SHOW_CHART,
                    page_ref=page,
                ),
                chart_panel_container(
                    page,
                    self.performance_chart_slot,
                    min_height=460,
                    expand=True,
                ),
                SectionHeader("Portfolio value", icon=ft.Icons.ACCOUNT_BALANCE, page_ref=page),
                chart_panel_container(
                    page,
                    self.portfolio_chart_slot,
                    min_height=360,
                    expand=True,
                ),
            ],
            spacing=8,
            tight=True,
            expand=3,
        )

        self.content = ft.Row(
            [tables_column, charts_column],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=16,
        )

    def _on_run(self, e) -> None:
        self.parent_view.run_panel(self)

    def set_busy(self, busy: bool) -> None:
        self.run_btn.disabled = busy

    def show_result(
        self,
        summary: str,
        df: pd.DataFrame | None,
        *,
        trades_df: pd.DataFrame | None = None,
        chart_series: dict | None = None,
        portfolio_dates: list[str] | None = None,
        portfolio_values: list[float] | None = None,
        initial_capital: float = DEFAULT_PORTFOLIO_START,
    ) -> None:
        self.summary.value = summary
        if df is not None and not df.empty:
            self.results_table.columns = dataframe_columns(df)
            self.results_table.rows = dataframe_to_rows(df)
        else:
            self.results_table.columns = [ft.DataColumn(ft.Text("Info"))]
            self.results_table.rows = [
                ft.DataRow(cells=[ft.DataCell(ft.Text("No results"))])
            ]

        if trades_df is not None and not trades_df.empty:
            self.trades_table.columns = dataframe_columns(trades_df)
            max_trades = 200
            self.trades_table.rows = dataframe_to_rows(trades_df, max_rows=max_trades)
            if len(trades_df) > max_trades:
                self.summary.value = (
                    f"{summary} (trade log: first {max_trades} of {len(trades_df)} rows)"
                    if summary
                    else f"Trade log: first {max_trades} of {len(trades_df)} rows"
                )
        else:
            self.trades_table.columns = [ft.DataColumn(ft.Text("Info"))]
            self.trades_table.rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                "No trade log for this run "
                                "(enter ticker(s) for CANSLIM, or see summary table)."
                            )
                        )
                    ]
                )
            ]

        if chart_series:
            self.performance_chart_slot.content = build_performance_chart(
                self.page_ref, chart_series
            )
            self.portfolio_chart_slot.content = build_portfolio_chart(
                self.page_ref,
                portfolio_dates or [],
                portfolio_values or [],
                initial_capital=initial_capital,
            )
        else:
            self.performance_chart_slot.content = ft.Text(
                "Enter one or more tickers in the sidebar filter to view performance "
                "and buy/sell markers.",
                color=ThemeHelper.text_muted(self.page_ref),
                size=12,
            )
            self.portfolio_chart_slot.content = ft.Text(
                "Portfolio chart requires ticker filter (capital split equally per symbol).",
                color=ThemeHelper.text_muted(self.page_ref),
                size=12,
            )

        try:
            self.summary.update()
            self.results_table.update()
            self.trades_table.update()
            self.performance_chart_slot.update()
            self.portfolio_chart_slot.update()
        except RuntimeError:
            pass


class StrategyBacktestView(BaseView):
    _tab_index = 5
    def __init__(self, page: ft.Page):
        super().__init__(page)
        # Flet 0.85: no ScrollMode.DISABLED — None disables outer scroll (sidebar scrolls).
        self.scroll = None
        cfg = stock_config()

        self.hybrid_weeks = InputStyles.text_field(
            page,
            label="Market trend weeks",
            value=str(cfg.hybrid_market_trend_weeks),
            width=180,
            tooltip=(
                "Weeks of market index data used to decide if the broad market "
                "is in an uptrend for the hybrid filter."
            ),
        )
        self.canslim_stop = InputStyles.text_field(
            page,
            label="Stop loss",
            value=str(cfg.canslim_stop_loss),
            width=120,
            tooltip="Exit simulated trades when loss reaches this fraction (e.g. -0.08 = 8%).",
        )
        self.canslim_tp = InputStyles.text_field(
            page,
            label="Take profit",
            value=str(cfg.canslim_take_profit),
            width=120,
            tooltip="Exit simulated trades when gain reaches this fraction (e.g. 0.25 = 25%).",
        )

        self.panel_general = _BacktestPanel(
            page,
            self,
            "General Rules",
            run_tooltip=STRATEGY_INFO["general"]["run_tooltip"],
        )
        self.panel_hybrid = _BacktestPanel(
            page,
            self,
            "Hybrid",
            extra_controls=[self.hybrid_weeks],
            run_tooltip=STRATEGY_INFO["hybrid"]["run_tooltip"],
        )
        self.panel_technical = _BacktestPanel(
            page,
            self,
            "Technical",
            run_tooltip=STRATEGY_INFO["technical"]["run_tooltip"],
        )
        self.panel_canslim = _BacktestPanel(
            page,
            self,
            "CANSLIM Backtest",
            extra_controls=[self.canslim_stop, self.canslim_tp],
            run_tooltip=STRATEGY_INFO["canslim"]["run_tooltip"],
        )
        self._panels = [
            self.panel_general,
            self.panel_hybrid,
            self.panel_technical,
            self.panel_canslim,
        ]

        # Avoid nesting ft.Tabs inside the main app Tabs (Flet 0.85 layout bug
        # causes the inner Tabs to render blank). Use SegmentedButton +
        # AnimatedSwitcher for the same UX without the nesting.
        self._sub_keys = ["general", "hybrid", "technical", "canslim"]
        self._sub_segmented = ft.SegmentedButton(
            selected=["general"],
            allow_empty_selection=False,
            allow_multiple_selection=False,
            segments=[
                ft.Segment(
                    value="general",
                    label=ft.Text("General"),
                    icon=ft.Icon(ft.Icons.RULE),
                    tooltip=STRATEGY_INFO["general"]["tooltip"],
                ),
                ft.Segment(
                    value="hybrid",
                    label=ft.Text("Hybrid"),
                    icon=ft.Icon(ft.Icons.MERGE_TYPE),
                    tooltip=STRATEGY_INFO["hybrid"]["tooltip"],
                ),
                ft.Segment(
                    value="technical",
                    label=ft.Text("Technical"),
                    icon=ft.Icon(ft.Icons.CANDLESTICK_CHART),
                    tooltip=STRATEGY_INFO["technical"]["tooltip"],
                ),
                ft.Segment(
                    value="canslim",
                    label=ft.Text("CANSLIM BT"),
                    icon=ft.Icon(ft.Icons.TIMELINE),
                    tooltip=STRATEGY_INFO["canslim"]["tooltip"],
                ),
            ],
            on_change=self._on_sub_tab_change,
        )

        self.backtest_period_days = InputStyles.text_field(
            page,
            label="Backtest period (days)",
            value=str(DEFAULT_LOOKBACK_DAYS),
            width=180,
            tooltip="Calendar days ending at the latest date in your database (default: 1 year).",
        )
        self.portfolio_start = InputStyles.text_field(
            page,
            label="Starting portfolio ($)",
            value=str(DEFAULT_PORTFOLIO_START),
            width=160,
            tooltip="Starting cash for the simulation; partial shares allowed.",
        )
        self.ticker_filter = InputStyles.text_field(
            page,
            label="Ticker filter (optional)",
            hint_text="AAPL MSFT or comma-separated — shared across all strategies",
            expand=True,
            on_blur=on_ticker_field_blur(multi=True),
            tooltip=(
                "Limit the backtest to specific symbols. Leave empty to use the "
                "universe selector below. Charts and trade logs require at least one ticker."
            ),
        )
        self.backtest_universe = InputStyles.dropdown(
            page,
            label="Universe (when filter empty)",
            width=260,
            value=cfg.default_backtest_universe or "with_history",
            options=[
                ft.dropdown.Option("focus", "Focus watchlist"),
                ft.dropdown.Option("with_history", "All with price data"),
                ft.dropdown.Option("universe", "Full active universe"),
            ],
            tooltip="Symbol set used when the ticker filter is empty.",
        )
        self._form_text_fields = [
            self.hybrid_weeks,
            self.canslim_stop,
            self.canslim_tp,
            self.backtest_period_days,
            self.portfolio_start,
            self.ticker_filter,
        ]
        self._form_dropdowns = [self.backtest_universe]
        self.strategy_info_title = ft.Text(
            STRATEGY_INFO["general"]["title"],
            size=13,
            weight=ft.FontWeight.W_600,
        )
        self.strategy_info_body = ft.Text(
            self._format_strategy_info("general"),
            size=11,
            color=ThemeHelper.text_muted(page),
        )
        self._strategy_info_panel = ft.Container(
            content=ft.Column(
                [self.strategy_info_title, self.strategy_info_body],
                spacing=4,
                tight=True,
            ),
            padding=ft.padding.only(top=4, bottom=4),
            border=ft.border.all(1, ThemeHelper.border_default(page)),
            border_radius=6,
            bgcolor=ThemeHelper.surface_dim(page),
        )
        # No inner AnimatedSwitcher — Flet 0.85 leaves AnimatedSwitcher content
        # unbounded when its child Column uses expand/scroll, which makes the
        # whole tab paint blank. A plain Container with .content swaps works.
        self._sub_host = ft.Container(
            content=self.panel_general,
            expand=True,
            padding=ft.padding.only(left=4, top=4),
        )

        self.status_ring = ft.ProgressRing(width=22, height=22, stroke_width=2, visible=False)
        self.status_text = ft.Text("", size=12, expand=True)
        self.status_bar = ft.ProgressBar(value=0, visible=False)
        self._status_row = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [self.status_ring, self.status_text],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.status_bar,
                ],
                spacing=6,
                tight=True,
            ),
            visible=False,
            padding=ft.padding.only(bottom=8),
        )

        sidebar_width = max(260, min(400, int((page.width or 1100) * 0.28)))
        self._sidebar = ft.Container(
            width=sidebar_width,
            content=ft.Column(
                [
                    SectionHeader("Controls", icon=ft.Icons.TUNE, page_ref=page),
                    self._sub_segmented,
                    self._strategy_info_panel,
                    self.backtest_period_days,
                    self.portfolio_start,
                    self.ticker_filter,
                    self.backtest_universe,
                    self._status_row,
                ],
                spacing=10,
                tight=True,
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=ft.padding.only(right=16),
            border=ft.border.only(right=ft.BorderSide(1, ThemeHelper.border_default(page))),
        )

        self.controls = [
            ViewTitleBar("Strategy Backtests"),
            ft.Divider(),
            ft.Row(
                [self._sidebar, self._sub_host],
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
        ]
        self._active_panel = self.panel_general

    @staticmethod
    def _format_strategy_info(key: str) -> str:
        info = STRATEGY_INFO.get(key, STRATEGY_INFO["general"])
        return (
            f"{info['summary']}\n\n"
            f"Strengths: {info['strengths']}\n\n"
            f"Weaknesses: {info['weaknesses']}"
        )

    def _update_strategy_info(self, key: str) -> None:
        info = STRATEGY_INFO.get(key, STRATEGY_INFO["general"])
        self.strategy_info_title.value = info["title"]
        self.strategy_info_body.value = self._format_strategy_info(key)
        try:
            self.strategy_info_title.update()
            self.strategy_info_body.update()
        except RuntimeError:
            pass

    def _set_status(self, message: str, *, pct: float | None = None, busy: bool = False) -> None:
        """Update the shared progress strip (thread-safe via _safe_update)."""
        self._status_row.visible = busy or bool(message)
        self.status_ring.visible = busy
        self.status_text.value = message
        if pct is not None:
            self.status_bar.visible = True
            self.status_bar.value = max(0.0, min(1.0, pct))
        else:
            self.status_bar.visible = busy
            if busy:
                self.status_bar.value = None

        def _ui():
            try:
                self._status_row.update()
                self.status_ring.update()
                self.status_text.update()
                self.status_bar.update()
            except RuntimeError:
                pass

        self._safe_update(_ui, label="backtest_status")

    def _clear_status(self) -> None:
        self._set_status("", busy=False)
        self._status_row.visible = False

        def _ui():
            try:
                self._status_row.update()
            except RuntimeError:
                pass

        self._safe_update(_ui, label="backtest_status_clear")

    def _progress_callback(self, phase: str):
        """Build a callback matching analysis modules' ``ProgressCallback`` (0.0–1.0)."""

        def _cb(pct: float) -> None:
            pct_clamped = max(0.0, min(1.0, pct))
            pct_display = int(pct_clamped * 100)
            self._set_status(
                f"{phase} — {pct_display}% complete",
                pct=pct_clamped,
                busy=True,
            )

        return _cb

    def _on_sub_tab_change(self, e) -> None:
        selected = list(getattr(e.control, "selected", None) or [])
        if not selected:
            return
        key = selected[0]
        try:
            idx = self._sub_keys.index(key)
        except ValueError:
            return
        self._active_panel = self._panels[idx]
        self._sub_host.content = self._active_panel
        self._update_strategy_info(key)
        try:
            self._sub_host.update()
        except RuntimeError:
            pass

    def run_panel(self, panel: _BacktestPanel) -> None:
        if getattr(self, "_running", False):
            show_snackbar(self.page_ref, "A backtest is already running.", severity="warning")
            return
        self._running = True
        for p in self._panels:
            p.set_busy(True)

        cfg = stock_config()
        explicit = parse_ticker_filter(self.ticker_filter.value)
        if explicit:
            tickers = explicit
        else:
            scope = self.backtest_universe.value or cfg.default_backtest_universe
            tickers = resolve_universe(cfg.db_path, scope) or None
        lookback_days = parse_lookback_days(self.backtest_period_days.value)
        initial_capital = parse_portfolio_start(self.portfolio_start.value)
        run_title = panel._title
        self._set_status(f"Starting {run_title} backtest…", busy=True)

        def _work():
            def _progress(msg: str, pct: float | None, busy: bool) -> None:
                self._set_status(msg, pct=pct, busy=busy)

            summary = ""
            df = None
            trades_df = None
            chart_series = None
            portfolio_dates: list[str] | None = None
            portfolio_values: list[float] | None = None
            try:
                if panel is self.panel_general:
                    outcome = run_general_strategy(
                        db_path=cfg.db_path,
                        tickers=tickers,
                        lookback_days=lookback_days,
                        initial_capital=initial_capital,
                        use_parallel=cfg.use_parallel,
                        progress=_progress,
                    )
                elif panel is self.panel_hybrid:
                    try:
                        weeks = int(self.hybrid_weeks.value or cfg.hybrid_market_trend_weeks)
                    except ValueError:
                        weeks = cfg.hybrid_market_trend_weeks
                    outcome = run_hybrid_strategy(
                        db_path=cfg.db_path,
                        tickers=tickers,
                        lookback_days=lookback_days,
                        initial_capital=initial_capital,
                        use_parallel=cfg.use_parallel,
                        market_trend_weeks=weeks,
                        progress=_progress,
                    )
                elif panel is self.panel_technical:
                    outcome = run_technical_strategy(
                        db_path=cfg.db_path,
                        tickers=tickers,
                        lookback_days=lookback_days,
                        initial_capital=initial_capital,
                        use_parallel=cfg.use_parallel,
                        progress=_progress,
                    )
                else:
                    try:
                        stop = float(self.canslim_stop.value)
                        tp = float(self.canslim_tp.value)
                    except ValueError:
                        stop, tp = cfg.canslim_stop_loss, cfg.canslim_take_profit
                    outcome = run_canslim_strategy(
                        db_path=cfg.db_path,
                        tickers=tickers,
                        lookback_days=lookback_days,
                        initial_capital=initial_capital,
                        use_parallel=cfg.use_parallel,
                        stop_loss=stop,
                        take_profit=tp,
                        progress=_progress,
                    )
                summary = outcome.summary
                df = outcome.df
                trades_df = outcome.trades_df
                chart_series = outcome.chart_series
                portfolio_dates = outcome.portfolio_dates
                portfolio_values = outcome.portfolio_values
            except Exception as ex:
                summary = f"Error: {ex}"

            def _ui():
                self._running = False
                for p in self._panels:
                    p.set_busy(False)
                self._clear_status()
                panel.show_result(
                    summary,
                    df,
                    trades_df=trades_df,
                    chart_series=chart_series,
                    portfolio_dates=portfolio_dates,
                    portfolio_values=portfolio_values,
                    initial_capital=initial_capital,
                )
                severity = "error" if summary.startswith("Error:") else "info"
                show_snackbar(self.page_ref, summary[:100], severity=severity)
                try:
                    self.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def refresh_data(self) -> None:
        cfg = stock_config()
        self.hybrid_weeks.value = str(cfg.hybrid_market_trend_weeks)
        self.canslim_stop.value = str(cfg.canslim_stop_loss)
        self.canslim_tp.value = str(cfg.canslim_take_profit)

    def refresh_theme(self) -> None:
        InputStyles.refresh_fields(
            self.page_ref, self._form_text_fields, self._form_dropdowns
        )
        for p in self._panels:
            p.summary.color = ThemeHelper.text_muted(self.page_ref)
        self.status_text.color = ThemeHelper.text_muted(self.page_ref)
        self.strategy_info_body.color = ThemeHelper.text_muted(self.page_ref)
        self._strategy_info_panel.bgcolor = ThemeHelper.surface_dim(self.page_ref)
        self._strategy_info_panel.border = ft.border.all(
            1, ThemeHelper.border_default(self.page_ref)
        )
        try:
            self.update()
        except RuntimeError:
            pass
