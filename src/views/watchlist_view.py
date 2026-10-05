"""Watchlists — multiple saved lists with live quote metrics."""

from __future__ import annotations

import re
import threading

import flet as ft

from src.analysis import ticker_registry as registry
from src.analysis.ingest import ingest_stock_data
from src.analysis.quote_snapshot import get_quotes_bulk
from src.analysis.watchlist_schema import (
    WATCHLIST_KIND_HOLDINGS,
    add_member,
    create_watchlist,
    delete_watchlist,
    list_members,
    list_watchlists,
    remove_member,
    rename_watchlist,
)
from src.services.event_bus import event_bus
from src.services.stock_config import stock_config
from src.services.tab_indices import TAB_STOCK_DETAIL
from src.utils.format_utils import format_change, format_currency, format_range
from src.views.base_view import BaseView
from src.views.components.feedback import AsyncStatusRow, show_snackbar
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.tables import InteractiveTablePanel, data_column
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import navigate_to_stock_detail, on_ticker_field_blur, parse_symbols



class WatchlistView(BaseView):
    _tab_index = 2

    def __init__(self, page: ft.Page):
        super().__init__(page)
        self._selected_watchlist_id: int | None = None

        self.list_dropdown = InputStyles.dropdown(
            page,
            label="Watchlist",
            width=220,
            on_select=self._on_list_change,
        )
        self.new_list_field = InputStyles.text_field(page, label="New list name", width=180)
        self.add_field = InputStyles.text_field(
            page,
            label="Add symbol(s)",
            hint_text="AAPL MSFT",
            expand=True,
            on_blur=on_ticker_field_blur(multi=True),
        )
        self.async_status = AsyncStatusRow(page)
        self.cap_hint = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self._table_panel = InteractiveTablePanel(
            page,
            [
                data_column("Symbol"),
                data_column("Price"),
                data_column("Change"),
                data_column("Day range"),
                data_column("Actions"),
            ],
            search_hint="Filter symbols…",
            expand=True,
        )
        self.focus_table = self._table_panel.table
        self.progress = ft.ProgressBar(value=0, visible=False)
        self.status_text = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self._running = False
        self._cancel_event: threading.Event | None = None

        self.run_focus_btn = ft.ElevatedButton(
            "Update focus ingest",
            icon=ft.Icons.CLOUD_DOWNLOAD,
            style=ButtonStyles.primary(),
            on_click=self._on_run_focus_ingest,
            tooltip="Smart incremental ingest for registry focus symbols.",
        )
        self.stop_btn = ft.ElevatedButton(
            "Stop",
            icon=ft.Icons.STOP,
            style=ButtonStyles.secondary(),
            disabled=True,
            on_click=self._on_stop,
        )

        self.controls = [
            ViewTitleBar("Watchlists"),
            ft.Divider(),
            self.cap_hint,
            ft.Row(
                [
                    self.list_dropdown,
                    self.new_list_field,
                    ft.IconButton(
                        icon=ft.Icons.ADD,
                        tooltip="Create watchlist",
                        on_click=self._on_create_list,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.EDIT,
                        tooltip="Rename watchlist",
                        on_click=self._on_rename_list,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DELETE,
                        tooltip="Delete watchlist",
                        on_click=self._on_delete_list,
                    ),
                ],
                spacing=8,
                wrap=True,
            ),
            ft.Row(
                [
                    self.add_field,
                    ft.ElevatedButton("Add", icon=ft.Icons.ADD, on_click=self._on_add),
                ],
                spacing=8,
            ),
            SectionHeader("Symbols", icon=ft.Icons.BOOKMARK, page_ref=page),
            self._table_panel.control,
            self.async_status,
            SectionHeader("Focus ingest (registry)", icon=ft.Icons.CLOUD_DOWNLOAD, page_ref=page),
            ft.Text(
                "Focus symbols (≤20) drive daily scans. Add to focus via Data Management or Leaderboard.",
                size=12,
                color=ThemeHelper.text_muted(page),
            ),
            ft.Row([self.run_focus_btn, self.stop_btn], spacing=8),
            self.progress,
            self.status_text,
        ]

    def _db_path(self) -> str:
        return stock_config().db_path

    def refresh_data(self) -> None:
        self.refresh_data_async(label="watchlist")

    def _fetch_data(self) -> dict:
        cfg = stock_config()
        db = cfg.db_path
        from src.analysis.watchlist_schema import (
            sync_focus_to_default_watchlist,
            sync_watchlist_members_to_registry,
        )

        sync_focus_to_default_watchlist(db)
        sync_watchlist_members_to_registry(db)
        registry.ensure_registry(db)
        watchlists = list_watchlists(db)
        wl_id = self._selected_watchlist_id
        if wl_id is None and watchlists:
            wl_id = watchlists[0].id
        members = list_members(db, wl_id) if wl_id else []
        syms = [m.ticker for m in members]
        quotes = get_quotes_bulk(db, syms) if syms else {}
        summary = registry.count_summary(db)
        return {
            "watchlists": watchlists,
            "watchlist_id": wl_id,
            "members": members,
            "quotes": quotes,
            "summary": summary,
            "cap": cfg.focus_watchlist_max,
        }

    def _apply_data(self, data: dict) -> None:
        watchlists = data.get("watchlists") or []
        self.list_dropdown.options = [
            ft.dropdown.Option(str(wl.id), wl.name) for wl in watchlists
        ]
        wl_id = data.get("watchlist_id")
        if wl_id:
            self._selected_watchlist_id = wl_id
            self.list_dropdown.value = str(wl_id)

        cap = data.get("cap", 20)
        summary = data.get("summary") or {}
        self.cap_hint.value = (
            f"Registry focus: {summary.get('focus', 0)}/{cap} · "
            f"Showing saved list with {len(data.get('members') or [])} symbol(s)."
        )

        quotes = data.get("quotes") or {}
        page = self.page_ref
        table_rows: list[ft.DataRow] = []
        for m in data.get("members") or []:
            sym = m.ticker
            q = quotes.get(sym)
            price = format_currency(q.last_price) if q else "—"
            chg = format_change(q.change, q.change_pct) if q else "—"
            chg_color = ThemeHelper.text_primary(page)
            if q and q.change_pct is not None:
                if q.change_pct > 0:
                    chg_color = ThemeHelper.accent_green(page)
                elif q.change_pct < 0:
                    chg_color = ThemeHelper.text_error(page)
            day_rng = format_range(q.day_low, q.day_high) if q else "—"
            table_rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.TextButton(
                                sym,
                                on_click=lambda e, s=sym: self._open_ticker(s),
                            )
                        ),
                        ft.DataCell(ft.Text(price)),
                        ft.DataCell(ft.Text(chg, color=chg_color)),
                        ft.DataCell(ft.Text(day_rng)),
                        ft.DataCell(
                            ft.IconButton(
                                icon=ft.Icons.REMOVE_CIRCLE_OUTLINE,
                                tooltip="Remove from list",
                                on_click=lambda e, s=sym: self._on_remove(s),
                            )
                        ),
                    ]
                )
            )
        if not table_rows:
            table_rows.append(
                ft.DataRow(
                    cells=[ft.DataCell(ft.Text("—"))] * 5
                )
            )
        self._table_panel.set_rows(table_rows)
        try:
            self.update()
        except RuntimeError:
            pass

    def _on_list_change(self, e) -> None:
        if self.list_dropdown.value:
            self._selected_watchlist_id = int(self.list_dropdown.value)
            self.refresh_data_async(label="watchlist_switch")

    def _on_create_list(self, e) -> None:
        name = (self.new_list_field.value or "").strip()
        if not name:
            show_snackbar(self.page_ref, "Enter a list name.", severity="warning")
            return
        result = create_watchlist(self._db_path(), name)
        if result.get("ok"):
            self.new_list_field.value = ""
            self._selected_watchlist_id = result.get("id")
            show_snackbar(self.page_ref, f"Created '{name}'", severity="success")
            self.refresh_data_async(label="watchlist_create")
        else:
            show_snackbar(self.page_ref, result.get("error", "Failed"), severity="error")

    def _on_rename_list(self, e) -> None:
        if not self._selected_watchlist_id:
            return
        name = (self.new_list_field.value or "").strip()
        if not name:
            return
        result = rename_watchlist(self._db_path(), self._selected_watchlist_id, name)
        if result.get("ok"):
            show_snackbar(self.page_ref, "Renamed.", severity="success")
            self.refresh_data_async(label="watchlist_rename")

    def _on_delete_list(self, e) -> None:
        if not self._selected_watchlist_id:
            return
        result = delete_watchlist(self._db_path(), self._selected_watchlist_id)
        if result.get("ok"):
            self._selected_watchlist_id = None
            show_snackbar(self.page_ref, "Deleted.", severity="success")
            self.refresh_data_async(label="watchlist_delete")
        else:
            show_snackbar(self.page_ref, result.get("error", "Cannot delete"), severity="error")

    def _on_add(self, e) -> None:
        if not self._selected_watchlist_id:
            show_snackbar(self.page_ref, "Select or create a watchlist.", severity="warning")
            return
        wl = next((w for w in list_watchlists(self._db_path()) if w.id == self._selected_watchlist_id), None)
        if wl and wl.kind == WATCHLIST_KIND_HOLDINGS:
            show_snackbar(
                self.page_ref,
                "Holdings list is synced from Portfolio — edit holdings there.",
                severity="info",
            )
            return
        syms = parse_symbols(self.add_field.value or "")
        wl_id = self._selected_watchlist_id
        db = self._db_path()

        def _work():
            for sym in syms:
                add_member(db, wl_id, sym)

            def _ui():
                self.add_field.value = ""
                show_snackbar(self.page_ref, f"Added {len(syms)} symbol(s).", severity="success")
                self.refresh_data_async(label="watchlist_add")

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _on_remove(self, sym: str) -> None:
        if self._selected_watchlist_id:
            remove_member(self._db_path(), self._selected_watchlist_id, sym)
            self.refresh_data_async(label="watchlist_remove")

    def _open_ticker(self, sym: str) -> None:
        navigate_to_stock_detail(sym, run_analysis=False)

    def _on_stop(self, e) -> None:
        if self._cancel_event:
            self._cancel_event.set()

    def _on_run_focus_ingest(self, e) -> None:
        if self._running:
            return
        db = self._db_path()
        focus = registry.list_focus_symbols(db)
        if not focus:
            show_snackbar(self.page_ref, "No focus symbols in registry.", severity="warning")
            return
        self._running = True
        self._cancel_event = threading.Event()
        self.run_focus_btn.disabled = True
        self.stop_btn.disabled = False
        self.progress.visible = True
        self.progress.value = None

        def _work():
            self.global_control.push_db_activity("Focus ingest")

            def progress(pct, msg, *_):
                from src.views.base_view import schedule_ui_update

                def _ui():
                    self.progress.value = pct if pct >= 0 else None
                    self.status_text.value = msg
                    try:
                        self.progress.update()
                        self.status_text.update()
                    except RuntimeError:
                        pass

                schedule_ui_update(self.page_ref, _ui, label="focus_ingest")

            try:
                summary = ingest_stock_data(
                    db_path=db,
                    scope="focus",
                    use_parallel=stock_config().ingest_use_parallel,
                    progress_callback=progress,
                    cancel_event=self._cancel_event,
                )
                msg = f"Focus ingest: {summary.get('success', 0)}/{summary.get('total', 0)} OK"
            except Exception as ex:
                msg = f"Failed: {ex}"
            finally:
                self.global_control.pop_db_activity()

            def _done():
                self._running = False
                self.run_focus_btn.disabled = False
                self.stop_btn.disabled = True
                self.progress.visible = False
                self.status_text.value = msg
                self.refresh_data_async(label="focus_done")

            from src.views.base_view import schedule_ui_update

            schedule_ui_update(self.page_ref, _done, label="focus_done")

        threading.Thread(target=_work, daemon=True).start()

    def refresh_theme(self) -> None:
        InputStyles.refresh_fields(
            self.page_ref,
            [self.add_field, self.new_list_field],
            [self.list_dropdown],
        )
        try:
            self.update()
        except RuntimeError:
            pass
