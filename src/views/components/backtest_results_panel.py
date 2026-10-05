"""Results presentation panel for strategy backtests."""

from __future__ import annotations

import csv
import os
import tempfile
from typing import Callable

import flet as ft
import pandas as pd

from src.analysis.backtest_engine import BacktestResult
from src.views.backtest_charts import (
    build_drawdown_chart,
    build_equity_comparison_chart,
    build_performance_chart,
    build_portfolio_chart,
)
from src.views.components.cards import SelectableMetricCard
from src.views.components.chart_factory import chart_panel_container, dynamic_content_slot
from src.views.components.insights_panel import build_insights_panel
from src.views.components.layouts import SectionHeader
from src.views.components.tables import (
    bordered_table_wrap,
    placeholder_data_columns,
    themed_data_table,
)
from src.views.strategy_constants import METRIC_TOOLTIPS, RESULTS_HELP_TEXT
from src.views.theme import ButtonStyles, ThemeHelper

_TRADES_PAGE_SIZE = 25


def _metric_color(page: ft.Page, key: str, value: float) -> str:
    good = ThemeHelper.chart_named(page, "gain")
    bad = ThemeHelper.chart_named(page, "loss")
    neutral = ThemeHelper.text_muted(page)
    if key in ("sharpe", "profit_factor"):
        return good if value >= 1.0 else (bad if value < 0 else neutral)
    if key in ("max_drawdown_pct",):
        return bad if value > 15 else good
    if key in ("total_return_pct", "cagr_pct", "alpha_pct"):
        return good if value > 0 else bad
    if key == "win_rate_pct":
        return good if value >= 50 else neutral
    return neutral


def _fmt_metric(key: str, value: float) -> str:
    if key in ("sharpe", "sortino", "beta", "profit_factor"):
        return f"{value:.2f}"
    if key == "trade_count":
        return str(int(value))
    return f"{value:.1f}%"


class BacktestResultsPanel(ft.Container):
    """KPI cards, charts, monthly grid, trade log, and AI insights slot."""

    def __init__(
        self,
        page: ft.Page,
        *,
        on_discuss: Callable[[], None] | None = None,
        on_analyze: Callable[[], None] | None = None,
    ):
        super().__init__(expand=True)
        self.page_ref = page
        self._on_discuss = on_discuss
        self._on_analyze = on_analyze
        self._trades_page = 0
        self._trades_data: list[dict] = []
        self._initial_capital = 10000.0

        self.summary_text = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self._kpi_row = ft.Row(wrap=True, spacing=8)
        self.equity_slot = dynamic_content_slot(
            page, "Run a backtest to see equity curves and metrics."
        )
        self.drawdown_slot = dynamic_content_slot(page, "")
        self.performance_slot = dynamic_content_slot(page, "")
        self.monthly_table = themed_data_table(page, columns=[], rows=[])
        self.trades_table = themed_data_table(page, columns=[], rows=[])
        self._trades_page_label = ft.Text("", size=11)
        self._table_section_title = ft.Text(
            "Monthly returns (%)",
            weight=ft.FontWeight.BOLD,
            size=14,
        )
        self._table_section_icon = ft.Icon(ft.Icons.CALENDAR_MONTH, size=18)
        self._ai_slot = ft.Container()
        self._help_expanded = False
        self._help_body = ft.Text(
            RESULTS_HELP_TEXT, size=11, color=ThemeHelper.text_muted(page), selectable=True,
        )
        self._help_container = ft.Container(
            content=self._help_body, visible=False, padding=8,
            border=ft.border.all(1, ThemeHelper.border_default(page)), border_radius=6,
        )

        self.content = ft.Column(
            [
                self.summary_text,
                self._kpi_row,
                SectionHeader("How to read these results", icon=ft.Icons.HELP_OUTLINE, page_ref=page),
                ft.TextButton(
                    "Show / hide guide",
                    icon=ft.Icons.EXPAND_MORE,
                    on_click=self._toggle_help,
                ),
                self._help_container,
                SectionHeader("Equity curve", icon=ft.Icons.TRENDING_UP, page_ref=page),
                chart_panel_container(page, self.equity_slot, expand=False),
                SectionHeader("Drawdown", icon=ft.Icons.WATER_DROP, page_ref=page),
                chart_panel_container(page, self.drawdown_slot, expand=False),
                SectionHeader("Per-ticker performance", icon=ft.Icons.SHOW_CHART, page_ref=page),
                chart_panel_container(page, self.performance_slot, expand=False),
                ft.Column(
                    [
                        ft.Row(
                            [self._table_section_icon, self._table_section_title],
                            spacing=6,
                        ),
                        ft.Divider(height=1, color=ThemeHelper.divider_color(page)),
                    ],
                    spacing=5,
                ),
                bordered_table_wrap(page, self.monthly_table, expand=False),
                ft.Row(
                    [
                        SectionHeader("Trade log", icon=ft.Icons.SWAP_HORIZ, page_ref=page),
                        ft.Container(expand=True),
                        ft.ElevatedButton(
                            "Export CSV",
                            icon=ft.Icons.DOWNLOAD,
                            style=ButtonStyles.secondary(),
                            on_click=self._export_trades,
                            tooltip="Download the full trade log as CSV.",
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CHEVRON_LEFT,
                            on_click=lambda e: self._page_trades(-1),
                            tooltip="Previous page",
                        ),
                        self._trades_page_label,
                        ft.IconButton(
                            icon=ft.Icons.CHEVRON_RIGHT,
                            on_click=lambda e: self._page_trades(1),
                            tooltip="Next page",
                        ),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                bordered_table_wrap(page, self.trades_table, expand=False),
                ft.Row(
                    [
                        ft.ElevatedButton(
                            "Analyze with AI",
                            icon=ft.Icons.AUTO_AWESOME,
                            style=ButtonStyles.primary(),
                            on_click=lambda e: self._on_analyze() if self._on_analyze else None,
                            tooltip="Generate an AI narrative of these backtest results (requires Local AI).",
                        ),
                        ft.OutlinedButton(
                            "Discuss with Assistant",
                            icon=ft.Icons.CHAT,
                            on_click=lambda e: self._on_discuss() if self._on_discuss else None,
                            tooltip="Open the Assistant tab with this backtest as context.",
                        ),
                    ],
                    spacing=12,
                ),
                self._ai_slot,
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            tight=True,
        )

    def _toggle_help(self, e) -> None:
        self._help_container.visible = not self._help_container.visible
        try:
            self._help_container.update()
        except RuntimeError:
            pass

    def _page_trades(self, delta: int) -> None:
        if not self._trades_data:
            return
        max_page = max(0, (len(self._trades_data) - 1) // _TRADES_PAGE_SIZE)
        self._trades_page = max(0, min(max_page, self._trades_page + delta))
        self._render_trades_page()
        try:
            self.trades_table.update()
            self._trades_page_label.update()
        except RuntimeError:
            pass

    def _render_trades_page(self) -> None:
        start = self._trades_page * _TRADES_PAGE_SIZE
        chunk = self._trades_data[start : start + _TRADES_PAGE_SIZE]
        max_page = max(0, (len(self._trades_data) - 1) // _TRADES_PAGE_SIZE)
        self._trades_page_label.value = f"Page {self._trades_page + 1} / {max_page + 1}"
        if not chunk:
            self.trades_table.columns = placeholder_data_columns()
            self.trades_table.rows = []
            return
        df = pd.DataFrame(chunk)
        self.trades_table.columns = [
            ft.DataColumn(ft.Text(str(c))) for c in df.columns
        ]
        self.trades_table.rows = [
            ft.DataRow(cells=[ft.DataCell(ft.Text(str(v))) for v in row])
            for row in df.values.tolist()
        ]

    def _export_trades(self, e) -> None:
        if not self._trades_data:
            return
        df = pd.DataFrame(self._trades_data)
        path = os.path.join(tempfile.gettempdir(), "backtest_trades.csv")
        df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        try:
            self.page_ref.launch_url(f"file://{path}")
        except Exception:
            pass

    def show_result(
        self,
        result: BacktestResult,
        *,
        initial_capital: float = 10000.0,
        ai_controls: list[ft.Control] | None = None,
    ) -> None:
        self._initial_capital = initial_capital
        m = result.metrics
        self.summary_text.value = result.summary
        self.summary_text.color = ThemeHelper.text_muted(self.page_ref)

        kpi_defs = [
            ("total_return_pct", "Total Return", m.total_return_pct),
            ("cagr_pct", "CAGR", m.cagr_pct),
            ("sharpe", "Sharpe", m.sharpe),
            ("max_drawdown_pct", "Max Drawdown", m.max_drawdown_pct),
            ("win_rate_pct", "Win Rate", m.win_rate_pct),
            ("profit_factor", "Profit Factor", m.profit_factor),
        ]
        self._kpi_row.controls.clear()
        for key, title, val in kpi_defs:
            tip = METRIC_TOOLTIPS.get(key, "")
            card = SelectableMetricCard(
                self.page_ref,
                title=title,
                value=_fmt_metric(key, val),
                subtitle=tip[:60] + "…" if len(tip) > 60 else tip,
                accent="primary",
                width=140,
            )
            card.tooltip = tip
            card._value_text.color = _metric_color(self.page_ref, key, val)
            self._kpi_row.controls.append(card)

        self.equity_slot.content = build_equity_comparison_chart(
            self.page_ref,
            strategy_dates=result.equity_dates,
            strategy_values=result.equity_values,
            benchmark_dates=result.benchmark_dates or None,
            benchmark_values=result.benchmark_values or None,
            buy_hold_dates=result.buy_hold_dates or None,
            buy_hold_values=result.buy_hold_values or None,
            initial_capital=initial_capital,
        )
        self.drawdown_slot.content = build_drawdown_chart(
            self.page_ref, result.drawdown_pct, result.equity_dates,
        )
        if result.chart_series:
            self.performance_slot.content = build_performance_chart(
                self.page_ref, result.chart_series,
            )
        else:
            self.performance_slot.content = ft.Text(
                "No per-ticker chart data.", color=ThemeHelper.text_muted(self.page_ref),
            )

        if result.monthly_returns is not None and not result.monthly_returns.empty:
            self._table_section_title.value = "Monthly returns (%)"
            self._table_section_icon.name = ft.Icons.CALENDAR_MONTH
            mr = result.monthly_returns.reset_index()
            mr.columns = ["Year"] + list(result.monthly_returns.columns)
            self.monthly_table.columns = [
                ft.DataColumn(ft.Text(str(c))) for c in mr.columns
            ]
            self.monthly_table.rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                f"{v:.1f}" if isinstance(v, float) else str(v),
                                color=ThemeHelper.chart_named(self.page_ref, "gain")
                                if isinstance(v, float) and v > 0
                                else (
                                    ThemeHelper.chart_named(self.page_ref, "loss")
                                    if isinstance(v, float) and v < 0
                                    else None
                                ),
                            )
                        )
                        for v in row
                    ]
                )
                for row in mr.values.tolist()
            ]
        else:
            self._table_section_title.value = "Monthly returns (%)"
            self._table_section_icon.name = ft.Icons.CALENDAR_MONTH
            self.monthly_table.columns = placeholder_data_columns()
            self.monthly_table.rows = []

        self._trades_data = [t.to_dict() for t in result.trades]
        self._trades_page = 0
        self._render_trades_page()

        hints = m.interpretation_hints or []
        if hints:
            self.summary_text.value += "\n" + " ".join(hints[:2])

        if ai_controls:
            self._ai_slot.content = ft.Column(ai_controls, spacing=8)
        else:
            self._ai_slot.content = None

        self.refresh_ui()

    def show_error(self, message: str) -> None:
        """Surface a durable error in the results area (not just a snackbar)."""
        text = (message or "Unknown error").strip()
        self.summary_text.value = f"Error: {text}"
        self.summary_text.color = ThemeHelper.text_error(self.page_ref)
        self._kpi_row.controls.clear()
        err = ft.Text(text, color=ThemeHelper.text_error(self.page_ref), selectable=True, size=12)
        self.equity_slot.content = err
        self.drawdown_slot.content = ft.Text("")
        self.performance_slot.content = ft.Text("")
        self.monthly_table.columns = placeholder_data_columns()
        self.monthly_table.rows = []
        self._trades_data = []
        self._render_trades_page()
        self.refresh_ui()

    def show_compare(
        self,
        results: list[BacktestResult],
        *,
        compare_chart: ft.Control | None = None,
        compare_columns: list[ft.DataColumn] | None = None,
        compare_rows: list[ft.DataRow] | None = None,
        compare_table: ft.DataTable | None = None,
    ) -> None:
        if not results:
            return

        # Restore summary color after a prior error display.
        self.summary_text.color = ThemeHelper.text_muted(self.page_ref)

        best = max(results, key=lambda r: r.metrics.total_return_pct)
        self.summary_text.value = (
            f"Compared {len(results)} strategies. "
            f"Best total return: {best.spec.name} "
            f"({best.metrics.total_return_pct:.1f}%).\n"
            + " | ".join(
                f"{r.spec.name}: {r.metrics.total_return_pct:.1f}% / Sharpe {r.metrics.sharpe:.2f}"
                for r in results
            )
        )

        self._kpi_row.controls.clear()
        for r in results:
            m = r.metrics
            tip = (
                f"{r.spec.name}: return {m.total_return_pct:.1f}%, "
                f"Sharpe {m.sharpe:.2f}, max DD {m.max_drawdown_pct:.1f}%, "
                f"{m.trade_count} trades"
            )
            card = SelectableMetricCard(
                self.page_ref,
                title=r.spec.name,
                value=_fmt_metric("total_return_pct", m.total_return_pct),
                subtitle=f"Sharpe {m.sharpe:.2f} · DD {m.max_drawdown_pct:.1f}%",
                accent="primary",
                width=160,
                selected=(r is best),
            )
            card.tooltip = tip
            card._value_text.color = _metric_color(
                self.page_ref, "total_return_pct", m.total_return_pct
            )
            self._kpi_row.controls.append(card)

        if compare_chart is not None:
            self.equity_slot.content = compare_chart
        else:
            self.equity_slot.content = ft.Text(
                "No comparison chart available.",
                color=ThemeHelper.text_muted(self.page_ref),
            )

        self.drawdown_slot.content = ft.Text(
            "Drawdown charts are shown for single-strategy runs. "
            "Use the overlay equity curve above to compare path risk.",
            color=ThemeHelper.text_muted(self.page_ref),
            size=12,
        )
        self.performance_slot.content = ft.Text(
            "Per-ticker charts are available after running a single strategy.",
            color=ThemeHelper.text_muted(self.page_ref),
            size=12,
        )

        self._table_section_title.value = "Strategy comparison"
        self._table_section_icon.name = ft.Icons.COMPARE_ARROWS
        if compare_columns is not None:
            self.monthly_table.columns = compare_columns or placeholder_data_columns()
            self.monthly_table.rows = compare_rows or []
        elif compare_table is not None:
            self.monthly_table.columns = compare_table.columns or placeholder_data_columns()
            self.monthly_table.rows = compare_table.rows or []
        else:
            self.monthly_table.columns = placeholder_data_columns()
            self.monthly_table.rows = []

        self._trades_data = []
        self._trades_page = 0
        self._render_trades_page()
        self._ai_slot.content = None
        self.refresh_ui()

    def refresh_ui(self) -> None:
        """Force Flet to patch all result surfaces after a data bind."""
        for c in (
            self.summary_text,
            self._kpi_row,
            self.equity_slot,
            self.drawdown_slot,
            self.performance_slot,
            self._table_section_title,
            self._table_section_icon,
            self.monthly_table,
            self.trades_table,
            self._trades_page_label,
            self._ai_slot,
            self,
        ):
            try:
                c.update()
            except RuntimeError:
                pass

    def clear(self) -> None:
        self.summary_text.value = ""
        self._kpi_row.controls.clear()
        self._trades_data = []
        self.trades_table.columns = placeholder_data_columns()
        self.trades_table.rows = []
        self.monthly_table.columns = placeholder_data_columns()
        self.monthly_table.rows = []
        self._table_section_title.value = "Monthly returns (%)"
        self._table_section_icon.name = ft.Icons.CALENDAR_MONTH

    def set_ai_insights(self, controls: list[ft.Control]) -> None:
        self._ai_slot.content = ft.Column(controls, spacing=8) if controls else None
        self.refresh_ui()
