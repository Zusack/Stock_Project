"""Stock Detail — quote metrics, chart, fundamentals, news, CANSLM, insights."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import flet as ft
import pandas as pd

from src.utils.format_utils import slice_price_df

from src.analysis.ai.advisor import SignalSuggestion, get_signal_advisor
from src.analysis.canslim import CanslimResult, analyze_canslim, canslim_result_from_bar
from src.analysis.db import browse_table, get_profile, load_ohlcv, load_price_data
from src.analysis.news_signals import load_recent_headlines
from src.analysis.ticker_evaluation import evaluate_ticker_snapshot
from src.analysis.quote_snapshot import get_quote
from src.services.stock_config import stock_config
from src.views.base_view import BaseView
from src.views.components.chart_factory import chart_empty_state, chart_panel_container, dynamic_content_slot
from src.views.components.chart_range import (
    DEFAULT_CHART_RANGE,
    chart_range_fetch_days,
    chart_range_selector,
    selected_chart_range,
)
from src.views.components.feedback import AsyncStatusRow, show_snackbar
from src.views.components.insights_panel import build_insights_panel
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.price_charts import build_candlestick_chart
from src.views.components.quote_panel import build_quote_header
from src.views.components.stock_charts import SeriesSpec, build_multi_series_chart
from src.views.components.tables import themed_data_table
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import BENCHMARK_OPTIONS, on_ticker_field_blur


@dataclass
class _TickerSessionSnapshot:
    ticker: str
    quote: Any
    profile: Any
    headlines: list
    sent: float | None
    tags: list
    catalysts: list
    ohlcv: pd.DataFrame | None
    fund_rows: list
    canslim_result: CanslimResult
    suggestions: list[SignalSuggestion]


class SingleTickerView(BaseView):
    _tab_index = 3

    def __init__(self, page: ft.Page):
        super().__init__(page)

        self.ticker_field = InputStyles.text_field(
            page,
            label="Ticker symbol",
            value="AAPL",
            width=160,
            on_submit=self._on_load,
            on_blur=on_ticker_field_blur(multi=False),
        )
        self.benchmark_dropdown = InputStyles.dropdown(
            page,
            label="Compare vs",
            value="^GSPC",
            width=180,
            options=[ft.dropdown.Option(k, v) for k, v in BENCHMARK_OPTIONS],
            on_select=lambda e: self._on_chart_options_changed(),
        )
        self.show_benchmark_switch = ft.Switch(
            label="Overlay benchmark %",
            value=False,
            on_change=lambda e: self._on_chart_options_changed(),
        )
        self.range_selector = chart_range_selector(
            selected=DEFAULT_CHART_RANGE,
            on_change=lambda e: self._on_chart_options_changed(),
        )
        self.async_status = AsyncStatusRow(page)
        self.quote_slot = ft.Container()
        self.chart_slot = dynamic_content_slot(page, "Enter a ticker and click Load.")
        self.fundamentals_table = themed_data_table(
            page,
            columns=[
                ft.DataColumn(ft.Text("Metric")),
                ft.DataColumn(ft.Text("Value")),
            ],
            rows=[],
        )
        self.news_list = ft.Column(spacing=6)
        self.criteria_table = themed_data_table(
            page,
            columns=[
                ft.DataColumn(ft.Text("Criterion")),
                ft.DataColumn(ft.Text("Result")),
            ],
            rows=[],
        )
        self.score_card = ft.Text("", size=14, weight=ft.FontWeight.W_600)
        self.metrics_card = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self.insights_slot = ft.Container()
        self._running = False
        self._chart_refreshing = False
        self._current_ticker = ""
        self._latest_ohlcv: pd.DataFrame | None = None
        self._session_by_ticker: dict[str, _TickerSessionSnapshot] = {}

        self.controls = [
            ViewTitleBar("Stock Detail"),
            ft.Divider(),
            ft.Row(
                [
                    self.ticker_field,
                    ft.ElevatedButton(
                        "Load",
                        icon=ft.Icons.REFRESH,
                        style=ButtonStyles.primary(),
                        on_click=self._on_load,
                    ),
                ],
                spacing=8,
                wrap=True,
            ),
            self.async_status,
            self.quote_slot,
            SectionHeader("Chart", icon=ft.Icons.CANDLESTICK_CHART, page_ref=page),
            ft.Row(
                [self.range_selector, self.benchmark_dropdown, self.show_benchmark_switch],
                spacing=12,
                wrap=True,
            ),
            chart_panel_container(page, self.chart_slot, panel_height=440),
            SectionHeader("Fundamentals", icon=ft.Icons.ACCOUNT_BALANCE, page_ref=page),
            ft.Container(content=self.fundamentals_table, padding=4),
            SectionHeader("News & sentiment", icon=ft.Icons.NEWSPAPER, page_ref=page),
            self.news_list,
            SectionHeader("CANSLM analysis", icon=ft.Icons.CHECKLIST, page_ref=page),
            self.score_card,
            self.metrics_card,
            ft.Container(content=self.criteria_table, padding=4),
            SectionHeader("Insights & suggestions", icon=ft.Icons.LIGHTBULB, page_ref=page),
            self.insights_slot,
        ]

    def _selected_range(self) -> str:
        return selected_chart_range(self.range_selector)

    def _on_chart_options_changed(self, _e=None) -> None:
        """Re-render the chart when range/benchmark display changes — no LLM or CANSLM."""
        if not self._current_ticker or self._running:
            return
        self._refresh_chart_only()

    def _refresh_chart_only(self) -> None:
        sym = self._current_ticker
        if not sym or self._chart_refreshing:
            return
        self._chart_refreshing = True
        range_key = self._selected_range()
        show_benchmark = bool(getattr(self.show_benchmark_switch, "value", False))
        benchmark = self.benchmark_dropdown.value or "^GSPC"
        ohlcv = self._latest_ohlcv

        def _work():
            cfg = stock_config()
            df = ohlcv
            if df is None or df.empty:
                df = load_ohlcv(sym, cfg.db_path)
            chart_data = self._build_chart_data(
                sym, df, range_key, cfg, show_benchmark, benchmark
            )

            def _ui():
                self._chart_refreshing = False
                self.chart_slot.content = chart_data
                try:
                    self.chart_slot.update()
                    if self.page_ref:
                        self.page_ref.update()
                except RuntimeError:
                    pass

            self._safe_update(_ui)

        threading.Thread(target=_work, daemon=True, name="stock-detail-chart").start()

    def _on_load(self, e) -> None:
        if self._running:
            return
        sym = (self.ticker_field.value or "").strip().upper()
        if not sym:
            show_snackbar(self.page_ref, "Enter a ticker.", severity="warning")
            return
        self._running = True
        self._current_ticker = sym
        self.async_status.set_running(f"Loading {sym}…")
        range_key = self._selected_range()
        show_benchmark = bool(getattr(self.show_benchmark_switch, "value", False))
        benchmark = self.benchmark_dropdown.value or "^GSPC"

        def _work():
            cfg = stock_config()
            quote = get_quote(cfg.db_path, sym)
            profile = get_profile(cfg.db_path, sym)
            headlines = load_recent_headlines(cfg.db_path, sym, limit=8)
            ohlcv = load_ohlcv(sym, cfg.db_path)
            self._latest_ohlcv = ohlcv
            chart_data = self._build_chart_data(sym, ohlcv, range_key, cfg, show_benchmark, benchmark)
            fund_rows = self._build_fundamentals_rows(cfg.db_path, sym, profile, quote)
            eval_snap = evaluate_ticker_snapshot(
                sym,
                cfg.db_path,
                cfg.market_ticker,
                ohlcv=ohlcv,
                headlines=headlines,
                fund_profile=profile,
                apply_scores=True,
            )
            canslim_result = (
                canslim_result_from_bar(eval_snap.bar)
                if eval_snap is not None
                else analyze_canslim(sym, cfg.db_path, market_ticker=cfg.market_ticker, ohlcv=ohlcv)
            )
            sent = eval_snap.features.news_norm if eval_snap else 0.5
            tags = list(eval_snap.features.news_tags) if eval_snap else []
            catalysts = list(eval_snap.features.catalyst_tags) if eval_snap else []
            lb_row = eval_snap.leaderboard_row if eval_snap else None
            self.insights_slot.content = ft.Text(
                "Generating insights…",
                size=12,
                color=ThemeHelper.text_muted(self.page_ref),
            )

            def _ui_loading():
                self.quote_slot.content = build_quote_header(self.page_ref, quote, sym)
                self.chart_slot.content = chart_data
                self.fundamentals_table.rows = fund_rows
                self._render_news(headlines, sent, tags, catalysts)
                try:
                    self.insights_slot.update()
                    self.update()
                    if self.page_ref:
                        self.page_ref.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_ui_loading)

            suggestions = self._insights_for_ticker(sym, cfg.db_path, canslim_result, lb_row)

            def _ui():
                self._running = False
                self._apply_canslim_result(canslim_result, show_errors=False)
                self.insights_slot.content = build_insights_panel(self.page_ref, suggestions)
                self._session_by_ticker[sym] = _TickerSessionSnapshot(
                    ticker=sym,
                    quote=quote,
                    profile=profile,
                    headlines=headlines,
                    sent=sent,
                    tags=tags,
                    catalysts=catalysts,
                    ohlcv=ohlcv,
                    fund_rows=fund_rows,
                    canslim_result=canslim_result,
                    suggestions=suggestions,
                )
                self.async_status.set_success(f"Loaded {sym}")
                try:
                    self.fundamentals_table.update()
                    self.news_list.update()
                    self.score_card.update()
                    self.metrics_card.update()
                    self.criteria_table.update()
                    self.insights_slot.update()
                    self.update()
                    if self.page_ref:
                        self.page_ref.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _build_chart_data(self, sym, ohlcv, range_key, cfg, show_benchmark=False, benchmark="^GSPC"):
        if ohlcv is None or ohlcv.empty:
            return chart_empty_state(self.page_ref, f"No price history for {sym}.")

        df = self._slice_ohlcv(ohlcv, range_key)
        if show_benchmark:
            bench = benchmark or "^GSPC"
            bench_df = load_price_data(
                bench, cfg.db_path, max_days=chart_range_fetch_days(range_key)
            )
            if bench_df is not None and not bench_df.empty:
                return self._build_overlay_chart(sym, df, bench, self._slice_ohlcv(bench_df, range_key))

        clean = df.dropna(subset=["Open", "High", "Low", "Close"]).tail(500)
        if len(clean) < 2:
            return chart_empty_state(self.page_ref, "Not enough data for chart.")
        dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in clean.index]
        return build_candlestick_chart(
            self.page_ref,
            sym,
            dates,
            clean["Open"].astype(float).tolist(),
            clean["High"].astype(float).tolist(),
            clean["Low"].astype(float).tolist(),
            clean["Close"].astype(float).tolist(),
            height=360,
        )

    def _slice_ohlcv(self, df: pd.DataFrame, range_key: str) -> pd.DataFrame:
        sliced = slice_price_df(df, range_key, min_rows=2)
        return sliced if sliced is not None else df

    def _build_overlay_chart(self, sym, stock_df, bench_sym, bench_df):
        if stock_df is None or bench_df is None or stock_df.empty or bench_df.empty:
            return chart_empty_state(self.page_ref, "Not enough data for overlay.")

        stock_close = stock_df["Close"].astype(float).dropna()
        bench_close = bench_df["Close"].astype(float).dropna() if "Close" in bench_df.columns else bench_df["Adj Close"].astype(float).dropna()

        common_idx = stock_close.index.intersection(bench_close.index)
        if len(common_idx) < 2:
            return chart_empty_state(self.page_ref, "Insufficient overlapping dates for benchmark.")

        stock_close = stock_close.loc[common_idx]
        bench_close = bench_close.loc[common_idx]
        base_s = float(stock_close.iloc[0])
        base_b = float(bench_close.iloc[0])
        pct_s = [((float(p) / base_s) - 1) * 100 for p in stock_close.tolist()]
        pct_b = [((float(p) / base_b) - 1) * 100 for p in bench_close.tolist()]
        dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in common_idx]

        return build_multi_series_chart(
            self.page_ref,
            [
                SeriesSpec(label=sym, values=pct_s, color=ThemeHelper.chart_named(self.page_ref, "price")),
                SeriesSpec(label=bench_sym, values=pct_b, color=ThemeHelper.chart_named(self.page_ref, "ma_short")),
            ],
            dates,
            y_title="% vs period start",
            signed_y=True,
            show_zero_baseline=True,
            height=360,
        )

    def _build_fundamentals_rows(self, db_path, sym, profile, quote) -> list[ft.DataRow]:
        rows: list[ft.DataRow] = []
        if profile:
            for key in (
                "Trailing_PE", "Forward_PE", "PEG_Ratio", "ROE",
                "Profit_Margins", "Debt_to_Equity", "Inst_Ownership", "Sector", "Industry",
            ):
                val = profile.get(key)
                if val is not None:
                    rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text(key.replace("_", " "))),
                                ft.DataCell(ft.Text(f"{val:.4f}" if isinstance(val, float) else str(val))),
                            ]
                        )
                    )
        try:
            fund_df = browse_table(db_path, "fundamentals", ticker=sym, limit=12)
            for _, row in fund_df.iterrows():
                rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(f"{row.get('Metric', '')} ({str(row.get('Report_Date', ''))[:10]})")),
                            ft.DataCell(ft.Text(f"{row.get('Value', '')}")),
                        ]
                    )
                )
        except Exception as ex:
            from src.utils.logger_utils import app_logger

            app_logger.log("STOCK_DETAIL", "Fundamentals browse failed.", level="DEBUG", ticker=sym, error=str(ex))
        if quote and quote.eps:
            rows.insert(
                0,
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text("EPS (quote)")),
                        ft.DataCell(ft.Text(f"{quote.eps:.2f}")),
                    ]
                ),
            )
        return rows or [ft.DataRow(cells=[ft.DataCell(ft.Text("No fundamentals")), ft.DataCell(ft.Text(""))])]

    def _render_news(self, headlines, sent, tags, catalysts) -> None:
        self.news_list.controls = []
        if sent is not None:
            self.news_list.controls.append(
                ft.Text(
                    f"Sentiment: {sent:.2f} · Tags: {', '.join(tags[:5]) or 'none'} · "
                    f"Catalysts: {', '.join(catalysts[:3]) or 'none'}",
                    size=12,
                    color=ThemeHelper.text_muted(self.page_ref),
                )
            )
        if not headlines:
            self.news_list.controls.append(
                ft.Text("No recent headlines in database.", size=12, color=ThemeHelper.text_muted(self.page_ref))
            )
            return
        for h in headlines:
            self.news_list.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(h.get("title", ""), size=13, weight=ft.FontWeight.W_500),
                            ft.Text(
                                f"{h.get('date', '')} · {h.get('publisher', '')}",
                                size=11,
                                color=ThemeHelper.text_muted(self.page_ref),
                            ),
                        ],
                        spacing=2,
                        tight=True,
                    ),
                    padding=8,
                    border=ft.border.all(1, ThemeHelper.border_subtle(self.page_ref)),
                    border_radius=6,
                )
            )

    def _insights_for_ticker(
        self,
        sym: str,
        db_path: str,
        canslim_result: CanslimResult | None,
        leaderboard_row=None,
    ):
        """Build insights using the same CANSLM score shown in Stock Detail."""
        kwargs: dict = {"db_path": db_path}
        if leaderboard_row is not None:
            kwargs["leaderboard_row"] = leaderboard_row
        if canslim_result is not None and not canslim_result.error:
            kwargs["canslim_score"] = canslim_result.score
            kwargs["canslim_max"] = canslim_result.max_score
        return get_signal_advisor().analyze_ticker(sym, **kwargs)

    def _format_leaderboard_metrics(self, result: CanslimResult) -> str:
        risk = result.risk_flag or "None"
        setup = "Yes" if result.pass_setup else "No"
        pattern = "Yes" if result.pass_pattern else "No"
        return (
            f"RS vs market {result.rs_pct:+.1f}% · "
            f"Vol {result.volume_ratio:.2f}x 50d avg · "
            f"{result.near_high_pct:.1f}% of 52w high · "
            f"Setup {setup} · Pattern {pattern} · Risk: {risk}"
        )

    def _apply_canslim_result(
        self, result: CanslimResult, *, show_errors: bool = True
    ) -> None:
        if result.error:
            if show_errors:
                show_snackbar(self.page_ref, result.error, severity="error")
            self.score_card.value = ""
            self.metrics_card.value = ""
            self.criteria_table.rows = []
            return
        cfg = stock_config()
        cfg.set_last_analysis_label(
            f"CANSLM {result.ticker} ({result.score}/{result.max_score})"
        )
        self.score_card.value = (
            f"Score: {result.score}/{result.max_score} — {result.verdict}"
        )
        self.metrics_card.value = self._format_leaderboard_metrics(result)
        self.criteria_table.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(line.split(":")[0] if ":" in line else line)),
                    ft.DataCell(
                        ft.Text(line.split(":", 1)[-1].strip() if ":" in line else "")
                    ),
                ]
            )
            for line in result.lines
        ]

    def _restore_session(self, sym: str) -> None:
        """Re-apply cached Stock Detail data for this session without re-running analysis."""
        snap = self._session_by_ticker.get(sym)
        if snap is None:
            self._on_load(None)
            return

        self._current_ticker = sym
        self._latest_ohlcv = snap.ohlcv
        range_key = self._selected_range()
        show_benchmark = bool(getattr(self.show_benchmark_switch, "value", False))
        benchmark = self.benchmark_dropdown.value or "^GSPC"
        cfg = stock_config()
        chart_data = self._build_chart_data(
            sym, snap.ohlcv, range_key, cfg, show_benchmark, benchmark
        )

        self.quote_slot.content = build_quote_header(self.page_ref, snap.quote, sym)
        self.chart_slot.content = chart_data
        self.fundamentals_table.rows = snap.fund_rows
        self._render_news(snap.headlines, snap.sent, snap.tags, snap.catalysts)
        self._apply_canslim_result(snap.canslim_result, show_errors=False)
        self.insights_slot.content = build_insights_panel(self.page_ref, snap.suggestions)
        self.async_status.set_success(f"Loaded {sym}")
        try:
            self.quote_slot.update()
            self.chart_slot.update()
            self.fundamentals_table.update()
            self.news_list.update()
            self.score_card.update()
            self.metrics_card.update()
            self.criteria_table.update()
            self.insights_slot.update()
            self.update()
            if self.page_ref:
                self.page_ref.update()
        except RuntimeError:
            pass

    def set_ticker(self, symbol: str, *, run_analysis: bool = False) -> None:
        """Set the ticker field and load quote, chart, CANSLM, and insights.

        ``run_analysis`` is kept for API compatibility; Load always runs CANSLM.
        """
        _ = run_analysis
        sym = (symbol or "").strip().upper()
        if not sym:
            return
        self.ticker_field.value = sym
        try:
            self.ticker_field.update()
        except RuntimeError:
            pass
        if sym in self._session_by_ticker and not run_analysis:
            self._restore_session(sym)
            return
        self._on_load(None)

    def refresh_data(self) -> None:
        if self._current_ticker and self._current_ticker in self._session_by_ticker:
            self._restore_session(self._current_ticker)
            return
        if self._current_ticker:
            self._on_load(None)

    def refresh_theme(self) -> None:
        InputStyles.refresh_fields(
            self.page_ref,
            [self.ticker_field],
            [self.benchmark_dropdown],
        )
        self.async_status.refresh_theme(self.page_ref)
        try:
            self.update()
        except RuntimeError:
            pass
