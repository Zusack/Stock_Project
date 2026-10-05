"""Portfolio — accounts, holdings, trades, E-Trade statement import."""

from __future__ import annotations

import threading
from pathlib import Path

import flet as ft
import pandas as pd

from src.analysis.ai.advisor import get_signal_advisor
from src.analysis.portfolio_schema import (
    add_trade,
    delete_holding,
    import_statement_holdings,
    list_account_value_history,
    list_accounts,
    list_holdings,
    list_trades,
    record_account_value,
    upsert_holding,
)
from src.analysis.quote_snapshot import get_quotes_bulk
from src.analysis.statements.etrade_parser import parse_etrade_statement
from src.services.stock_config import stock_config
from src.utils.format_utils import format_change, format_currency, format_percent
from src.views.base_view import BaseView
from src.views.components.chart_factory import chart_empty_state, chart_panel_container
from src.views.components.feedback import AsyncStatusRow, show_snackbar
from src.views.components.insights_panel import build_insights_panel
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.stock_charts import SeriesSpec, build_multi_series_chart
from src.views.components.tables import themed_data_table
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import navigate_to_stock_detail, on_ticker_field_blur


class PortfolioView(BaseView):
    _tab_index = 1

    def __init__(self, page: ft.Page):
        super().__init__(page)
        self._selected_account_id: int | None = None
        self._parsed_statement = None
        self._refocus_holding_ticker = False

        self.account_dropdown = InputStyles.dropdown(
            page,
            label="Account",
            width=240,
            on_select=self._on_account_change,
        )
        self.summary_card = ft.Text("", size=13, color=ThemeHelper.text_muted(page))
        self.holdings_table = themed_data_table(
            page,
            columns=[
                ft.DataColumn(ft.Text("Symbol")),
                ft.DataColumn(ft.Text("Qty")),
                ft.DataColumn(ft.Text("Cost basis")),
                ft.DataColumn(ft.Text("Price")),
                ft.DataColumn(ft.Text("Market value")),
                ft.DataColumn(ft.Text("Gain/Loss")),
                ft.DataColumn(ft.Text("Alloc %")),
                ft.DataColumn(ft.Text("Actions")),
            ],
            rows=[],
        )
        self.trades_table = themed_data_table(
            page,
            columns=[
                ft.DataColumn(ft.Text("Date")),
                ft.DataColumn(ft.Text("Action")),
                ft.DataColumn(ft.Text("Symbol")),
                ft.DataColumn(ft.Text("Qty")),
                ft.DataColumn(ft.Text("Price")),
                ft.DataColumn(ft.Text("Amount")),
            ],
            rows=[],
        )
        self.performance_slot = ft.Container(
            content=chart_empty_state(page, "Record account values to see performance over time.")
        )
        self.insights_slot = ft.Container()
        self.async_status = AsyncStatusRow(page)

        # Manual holding entry
        self.h_ticker = InputStyles.text_field(
            page,
            label="Ticker",
            width=100,
            on_blur=on_ticker_field_blur(),
            on_submit=self._on_save_holding,
        )
        self.h_qty = InputStyles.text_field(page, label="Quantity", width=100, on_submit=self._on_save_holding)
        self.h_cost = InputStyles.text_field(page, label="Avg cost", width=100, on_submit=self._on_save_holding)

        # Manual trade entry
        self.t_date = InputStyles.text_field(page, label="Date (YYYY-MM-DD)", width=140)
        self.t_ticker = InputStyles.text_field(page, label="Ticker", width=100, on_blur=on_ticker_field_blur())
        self.t_action = InputStyles.dropdown(
            page,
            label="Action",
            width=120,
            value="buy",
            options=[
                ft.dropdown.Option("buy", "Buy"),
                ft.dropdown.Option("sell", "Sell"),
                ft.dropdown.Option("dividend", "Dividend"),
            ],
        )
        self.t_qty = InputStyles.text_field(page, label="Qty", width=80)
        self.t_price = InputStyles.text_field(page, label="Price", width=80)

        self.controls = [
            ViewTitleBar("Portfolio"),
            ft.Divider(),
            ft.Text(
                "Track holdings across Retirement and Wealth Building accounts. "
                "All data stays local.",
                size=12,
                color=ThemeHelper.text_muted(page),
            ),
            ft.Row([self.account_dropdown, self.summary_card], spacing=16, wrap=True),
            SectionHeader("Holdings", icon=ft.Icons.ACCOUNT_BALANCE, page_ref=page),
            ft.Container(content=self.holdings_table, padding=4),
            ft.Text("Add / update holding", size=13, weight=ft.FontWeight.W_600),
            ft.Row(
                [
                    self.h_ticker,
                    self.h_qty,
                    self.h_cost,
                    ft.ElevatedButton("Save holding", icon=ft.Icons.SAVE, on_click=self._on_save_holding),
                ],
                spacing=8,
                wrap=True,
            ),
            SectionHeader("Recent trades", icon=ft.Icons.SWAP_HORIZ, page_ref=page),
            ft.Container(content=self.trades_table, padding=4),
            ft.Text("Log trade", size=13, weight=ft.FontWeight.W_600),
            ft.Row(
                [
                    self.t_date,
                    self.t_ticker,
                    self.t_action,
                    self.t_qty,
                    self.t_price,
                    ft.ElevatedButton("Add trade", icon=ft.Icons.ADD, on_click=self._on_add_trade),
                ],
                spacing=8,
                wrap=True,
            ),
            SectionHeader("Performance", icon=ft.Icons.TRENDING_UP, page_ref=page),
            chart_panel_container(page, self.performance_slot, panel_height=280),
            SectionHeader("E-Trade statement import", icon=ft.Icons.UPLOAD_FILE, page_ref=page),
            ft.Row(
                [
                    ft.ElevatedButton(
                        "Import PDF statement",
                        icon=ft.Icons.PICTURE_AS_PDF,
                        style=ButtonStyles.primary(),
                        on_click=self._on_pick_pdf,
                    ),
                ],
            ),
            self.async_status,
            SectionHeader("Portfolio insights", icon=ft.Icons.LIGHTBULB, page_ref=page),
            self.insights_slot,
        ]

    def refresh_data(self) -> None:
        self.refresh_data_async(label="portfolio")

    def _fetch_data(self) -> dict:
        cfg = stock_config()
        accounts = list_accounts(cfg.db_path)
        acct_id = self._selected_account_id
        if acct_id is None and accounts:
            acct_id = accounts[0].id
        holdings = list_holdings(cfg.db_path, acct_id) if acct_id else []
        trades = list_trades(cfg.db_path, acct_id) if acct_id else []
        value_hist = list_account_value_history(cfg.db_path, acct_id) if acct_id else []
        tickers = [h.ticker for h in holdings]
        quotes = get_quotes_bulk(cfg.db_path, tickers) if tickers else {}
        return {
            "accounts": accounts,
            "account_id": acct_id,
            "holdings": holdings,
            "trades": trades,
            "value_hist": value_hist,
            "quotes": quotes,
        }

    def _apply_data(self, data: dict) -> None:
        accounts = data.get("accounts") or []
        self.account_dropdown.options = [
            ft.dropdown.Option(str(a.id), a.name) for a in accounts
        ]
        acct_id = data.get("account_id")
        if acct_id:
            self._selected_account_id = acct_id
            self.account_dropdown.value = str(acct_id)

        holdings = data.get("holdings") or []
        quotes = data.get("quotes") or {}
        total_mv = 0.0
        rows = []
        for h in holdings:
            q = quotes.get(h.ticker)
            price = q.last_price if q else None
            mv = (price or 0) * h.quantity
            total_mv += mv
            cost = h.quantity * h.avg_cost_basis
            gl = mv - cost if price else None
            gl_pct = (gl / cost * 100) if gl is not None and cost else None
            rows.append((h, price, mv, gl, gl_pct))

        table_rows = []
        for h, price, mv, gl, gl_pct in rows:
            alloc = (mv / total_mv * 100) if total_mv else 0
            gl_text = "—"
            if gl is not None:
                gl_text = f"{format_currency(gl)} ({format_percent(gl_pct, signed=True)})"
            table_rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.TextButton(
                                h.ticker,
                                on_click=lambda e, s=h.ticker: navigate_to_stock_detail(s),
                            )
                        ),
                        ft.DataCell(ft.Text(f"{h.quantity:.2f}")),
                        ft.DataCell(ft.Text(format_currency(h.avg_cost_basis))),
                        ft.DataCell(ft.Text(format_currency(price))),
                        ft.DataCell(ft.Text(format_currency(mv))),
                        ft.DataCell(ft.Text(gl_text)),
                        ft.DataCell(ft.Text(format_percent(alloc, decimals=1))),
                        ft.DataCell(
                            ft.IconButton(
                                icon=ft.Icons.DELETE_OUTLINE,
                                tooltip="Remove holding",
                                on_click=lambda e, hid=h.id: self._on_delete_holding(hid),
                            )
                        ),
                    ]
                )
            )
        self.holdings_table.rows = table_rows or [
            ft.DataRow(cells=[ft.DataCell(ft.Text("No holdings"))] + [ft.DataCell(ft.Text(""))] * 7)
        ]

        total_cost = sum(h.quantity * h.avg_cost_basis for h in holdings)
        self.summary_card.value = (
            f"Positions: {len(holdings)} · "
            f"Market value: {format_currency(total_mv)} · "
            f"Cost: {format_currency(total_cost)}"
        )

        trades = data.get("trades") or []
        self.trades_table.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(t.trade_date)),
                    ft.DataCell(ft.Text(t.action)),
                    ft.DataCell(ft.Text(t.ticker)),
                    ft.DataCell(ft.Text(f"{t.quantity:.2f}")),
                    ft.DataCell(ft.Text(format_currency(t.price))),
                    ft.DataCell(ft.Text(format_currency(t.amount))),
                ]
            )
            for t in trades[:50]
        ] or [ft.DataRow(cells=[ft.DataCell(ft.Text("No trades"))] * 6)]

        self._render_performance(data.get("value_hist") or [], total_mv)
        self._render_insights(holdings)

        try:
            self.update()
            if self._refocus_holding_ticker:
                self._refocus_holding_ticker = False
                self.h_ticker.focus()
        except RuntimeError:
            pass

    def _render_performance(self, history, current_mv: float) -> None:
        if not history and current_mv <= 0:
            self.performance_slot.content = chart_empty_state(
                self.page_ref, "Import a statement or record values to track performance."
            )
            return
        dates = [h.as_of_date for h in history]
        values = [h.total_value for h in history]
        if current_mv > 0 and (not dates or dates[-1] != pd.Timestamp.today().strftime("%Y-%m-%d")):
            dates.append(pd.Timestamp.today().strftime("%Y-%m-%d"))
            values.append(current_mv)
        if len(values) < 2:
            self.performance_slot.content = chart_empty_state(
                self.page_ref, "Need at least two value points for a chart."
            )
            return
        base = values[0]
        pct = [((v / base) - 1) * 100 if base else 0 for v in values]
        color = ThemeHelper.chart_named(self.page_ref, "price")
        self.performance_slot.content = build_multi_series_chart(
            self.page_ref,
            [SeriesSpec(label="Account", values=pct, color=color)],
            dates,
            y_title="% change",
            signed_y=True,
            height=220,
        )

    def _render_insights(self, holdings) -> None:
        advisor = get_signal_advisor()
        suggestions = []
        for h in holdings[:5]:
            suggestions.extend(advisor.analyze_ticker(h.ticker, db_path=stock_config().db_path)[:1])
        self.insights_slot.content = build_insights_panel(
            self.page_ref, suggestions[:8], title="Holdings signals (rule-based)"
        )

    def _on_account_change(self, e) -> None:
        if self.account_dropdown.value:
            self._selected_account_id = int(self.account_dropdown.value)
            self.refresh_data_async(label="portfolio_account")

    def _on_save_holding(self, e) -> None:
        if not self._selected_account_id:
            show_snackbar(self.page_ref, "Select an account.", severity="warning")
            return
        sym = (self.h_ticker.value or "").strip().upper()
        try:
            qty = float(self.h_qty.value or 0)
            cost = float(self.h_cost.value or 0)
        except ValueError:
            show_snackbar(self.page_ref, "Invalid quantity or cost.", severity="error")
            return
        if not sym or qty <= 0:
            show_snackbar(self.page_ref, "Ticker and quantity required.", severity="warning")
            return
        account_id = self._selected_account_id

        def _work():
            result = upsert_holding(stock_config().db_path, account_id, sym, qty, cost)

            def _ui():
                if result.get("ok"):
                    self.h_ticker.value = ""
                    self.h_qty.value = ""
                    self.h_cost.value = ""
                    self._refocus_holding_ticker = True
                    show_snackbar(self.page_ref, f"Saved {sym}", severity="success")
                    try:
                        self.h_ticker.update()
                        self.h_qty.update()
                        self.h_cost.update()
                        self.h_ticker.focus()
                    except RuntimeError:
                        pass
                    self.refresh_data_async(label="portfolio_save")
                else:
                    show_snackbar(self.page_ref, result.get("error", "Failed"), severity="error")

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _on_add_trade(self, e) -> None:
        if not self._selected_account_id:
            return
        sym = (self.t_ticker.value or "").strip().upper()
        try:
            qty = float(self.t_qty.value or 0)
            price = float(self.t_price.value or 0)
        except ValueError:
            show_snackbar(self.page_ref, "Invalid qty/price.", severity="error")
            return
        account_id = self._selected_account_id
        trade_date = (self.t_date.value or "").strip()
        action = self.t_action.value or "buy"

        def _work():
            result = add_trade(
                stock_config().db_path,
                account_id,
                sym,
                trade_date,
                action,
                qty,
                price,
            )

            def _ui():
                if result.get("ok"):
                    show_snackbar(self.page_ref, "Trade recorded.", severity="success")
                    self.refresh_data_async(label="portfolio_trade")
                else:
                    show_snackbar(self.page_ref, result.get("error", "Failed"), severity="error")

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _on_delete_holding(self, holding_id: int) -> None:
        def _work():
            delete_holding(stock_config().db_path, holding_id)

            def _ui():
                self.refresh_data_async(label="portfolio_delete")

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _on_pick_pdf(self, e) -> None:
        # Flet 0.85+: FilePicker is a Service; pick_files is async and returns files.
        async def _pick() -> None:
            picker = ft.FilePicker()
            files = await picker.pick_files(
                dialog_title="Select E-Trade statement PDF",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["pdf"],
                allow_multiple=False,
            )
            if not files:
                return
            path = files[0].path
            if not path:
                show_snackbar(self.page_ref, "Could not read file path.", severity="error")
                return
            parsed = parse_etrade_statement(path)
            self._parsed_statement = parsed
            self._show_import_review_dialog(path, parsed)

        self.page_ref.run_task(_pick)

    def _show_import_review_dialog(self, path: str, parsed) -> None:
        pos_rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(p.symbol)),
                    ft.DataCell(ft.Text(f"{p.quantity:.2f}")),
                    ft.DataCell(ft.Text(format_currency(p.price))),
                    ft.DataCell(ft.Text(format_currency(p.cost_basis))),
                    ft.DataCell(ft.Text(format_currency(p.market_value))),
                ]
            )
            for p in parsed.positions
        ]
        warn_text = "\n".join(parsed.warnings) if parsed.warnings else "Review positions before confirming."
        review_table = themed_data_table(
            self.page_ref,
            columns=[
                ft.DataColumn(ft.Text("Symbol")),
                ft.DataColumn(ft.Text("Qty")),
                ft.DataColumn(ft.Text("Price")),
                ft.DataColumn(ft.Text("Cost basis")),
                ft.DataColumn(ft.Text("Market value")),
            ],
            rows=pos_rows or [ft.DataRow(cells=[ft.DataCell(ft.Text("No positions parsed"))] * 5)],
        )

        def _confirm(_e):
            if not self._selected_account_id:
                show_snackbar(self.page_ref, "Select account first.", severity="warning")
                return
            positions = [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "price": p.price,
                    "cost_basis": p.cost_basis,
                }
                for p in parsed.positions
            ]
            activities = [
                {
                    "symbol": a.symbol,
                    "date": a.date,
                    "action": a.action,
                    "quantity": a.quantity,
                    "price": a.price,
                    "amount": a.amount,
                    "fees": a.fees,
                }
                for a in parsed.activities
            ]
            result = import_statement_holdings(
                stock_config().db_path,
                self._selected_account_id,
                file_name=Path(path).name,
                period_start=parsed.period_start,
                period_end=parsed.period_end,
                positions=positions,
                activities=activities,
                ending_value=parsed.ending_value,
            )
            self.page_ref.close(dlg)
            if result.get("ok"):
                show_snackbar(self.page_ref, "Statement imported.", severity="success")
                self.refresh_data_async(label="portfolio_import")
            else:
                show_snackbar(self.page_ref, result.get("error", "Import failed"), severity="error")

        def _cancel(_e):
            self.page_ref.close(dlg)

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Review statement import"),
            content=ft.Column(
                [
                    ft.Text(f"File: {Path(path).name} · Variant: {parsed.variant}", size=12),
                    ft.Text(warn_text, size=12, color=ThemeHelper.text_muted(self.page_ref)),
                    ft.Column(
                        [review_table],
                        height=240,
                        scroll=ft.ScrollMode.AUTO,
                    ),
                ],
                tight=True,
                spacing=8,
                width=560,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.ElevatedButton("Confirm import", on_click=_confirm),
            ],
        )
        self.page_ref.open(dlg)

    def refresh_theme(self) -> None:
        InputStyles.refresh_fields(
            self.page_ref,
            [self.h_ticker, self.h_qty, self.h_cost, self.t_date, self.t_ticker, self.t_qty, self.t_price],
            [self.account_dropdown, self.t_action],
        )
        try:
            self.update()
        except RuntimeError:
            pass
