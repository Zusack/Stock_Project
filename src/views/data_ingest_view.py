"""Research universe — bulk symbol registry and weekly Yahoo ingest into SQLite."""

from __future__ import annotations

import re
import threading
import time

import flet as ft

from src.analysis.db import get_db_summary
from src.analysis import ticker_registry as registry
from src.analysis.history_coverage import audit_watchlist
from src.analysis.ingest_run import get_resumable_run
from src.analysis.ticker_investigation import investigate_symbols
from src.analysis.ticker_validation import validate_symbols
from src.services.event_bus import event_bus
from src.services.ingest_runner import format_ingest_eta_label, run_universe_ingest
from src.services.stock_config import stock_config
from src.views.base_view import BaseView
from src.views.components.feedback import show_snackbar
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.tables import bordered_table_wrap, data_column, themed_data_table
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import navigate_to_stock_detail, on_ticker_field_blur, parse_symbols

_PAGE_SIZE = 50
from src.services.tab_indices import TAB_STOCK_DETAIL as _SINGLE_TICKER_TAB
from src.utils.progress_format import format_duration


_COVERAGE_HEADER_TOOLTIP = (
    "Coverage is based on stored price rows in the last 90 calendar days "
    "(weekdays only). gaps (N days) = missing trading days in that window. "
    "stale = latest bar is before the last completed trading day. "
    "empty = no price history. Use Audit gaps to refresh all symbols."
)

_VALIDATE_PAGE_TOOLTIP = (
    "Validates only the 50 symbols on this page against Yahoo "
    "(1-month bars + quote metadata) and your local DB. Updates validation "
    "status and reason; does not remove symbols or change tickerList.csv. "
    "Run after ingest to confirm delisted vs typo vs rate limit. "
    "Use Maintain watchlist for bulk dead-symbol actions."
)

_TIP_ADD = (
    "Add symbols to the research universe (not the focus watchlist). "
    "Enter one ticker or several separated by commas or spaces."
)
_TIP_SEARCH = "Filter the universe table by symbol substring (case-insensitive)."
_TIP_FILTER = (
    "Narrow the table: warnings, dead symbols (with or without stored history), "
    "or archived (updates paused)."
)
_TIP_PREV_PAGE = "Show the previous page of universe symbols (50 per page)."
_TIP_NEXT_PAGE = "Show the next page of universe symbols (50 per page)."
_TIP_INGEST_MODE = (
    "Smart update fetches only missing or new price bars. "
    "Full re-download replaces all history for each symbol (slower, use after fixes)."
)
_TIP_RUN_INGEST = (
    "Start downloading prices and fundamentals for active watchlist symbols into "
    "market_data.db. Required before backtests and the Leaderboard."
)
_TIP_STOP_INGEST = (
    "Request a graceful stop after the current symbol finishes. "
    "You can resume a paused run later if needed."
)
_TIP_RESUME_INGEST = (
    "Continue a previously paused ingest run from where it left off, "
    "without re-processing completed symbols."
)
_TIP_AUDIT_GAPS = (
    "Scan the watchlist for missing or stale price bars in the last 90 days and "
    "refresh coverage status for every symbol."
)
_TIP_REMOVE_CONFIRM = (
    "Permanently remove this symbol from the watchlist. "
    "Optional: delete stored price history and remove from tickerList.csv."
)
_TIP_MAINT_CLOSE = "Close this dialog without making changes."
_TIP_MAINT_VALIDATE = (
    "Run Yahoo validation on up to 50 removable symbols (no history) "
    "to confirm they are safe to delete."
)
_TIP_MAINT_INVESTIGATE = (
    "Look up up to 30 dead symbols via Yahoo and SEC to see delist or "
    "ticker-change reasons before archiving."
)
_TIP_MAINT_ARCHIVE = (
    "Pause ingest for dead symbols that still have history in the database "
    "(keeps data for research, stops failed downloads)."
)
_TIP_MAINT_REMOVE = (
    "Delete removable symbols (no price history) from the watchlist, database, "
    "and optionally tickerList.csv."
)


def format_coverage_label(row: registry.WatchlistRow) -> str:
    if row.skip_ingest:
        return "—"
    st = row.coverage_status or ("empty" if not row.has_history else "—")
    if st == "gaps" and row.gap_count:
        return f"gaps ({row.gap_count} days)"
    if st == "stale" and row.price_max_date:
        return f"stale (→{row.price_max_date[:10]})"
    return st


def format_coverage_tooltip(row: registry.WatchlistRow) -> str:
    if row.skip_ingest:
        return "Archived — coverage not shown."
    st = row.coverage_status or ("empty" if not row.has_history else "")
    if st == "empty":
        return "No rows in stock_history — full download on next smart ingest."
    if st == "gaps":
        base = (
            f"{row.gap_count} missing weekday(s) in the last 90-day window"
        )
        if row.gap_summary:
            return f"{base}: {row.gap_summary}"
        return base
    if st == "stale" and row.price_max_date:
        return (
            f"Latest bar {row.price_max_date[:10]} is before the last "
            "completed trading day."
        )
    if st == "current":
        return "Price history is up to date through the last trading day."
    return st or "Coverage unknown — run Audit gaps."


def format_reason_label(row: registry.WatchlistRow, *, max_len: int = 48) -> str:
    if row.skip_ingest and row.archive_reason:
        text = row.archive_reason
    elif row.validation_reason:
        text = row.validation_reason
    elif row.delist_category:
        text = row.delist_category
    else:
        return "—"
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


class DataIngestView(BaseView):
    _tab_index = 9

    def __init__(self, page: ft.Page):
        super().__init__(page)
        cfg = stock_config()

        self._page_offset = 0
        self._watchlist_filter: registry.WatchlistFilter | None = None
        self._running = False
        self._validating = False
        self._maintaining = False
        self._cancel_event: threading.Event | None = None
        self._resume_mode = False
        self._session_log_lines: list[str] = []

        self.summary_total = ft.Text("Total: —", size=12)
        self.summary_active = ft.Text("Active: —", size=12)
        self.summary_archived = ft.Text("Archived: —", size=12)
        self.summary_warnings = ft.Text("Warnings: —", size=12, color=ThemeHelper.text_error(page))
        self.summary_focus = ft.Text("Focus: —", size=12)
        self.universe_ingest_hint = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self.maint_status_ring = ft.ProgressRing(width=22, height=22, stroke_width=2, visible=False)
        self.maint_status_text = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self._maint_status_row = ft.Container(
            content=ft.Row(
                [self.maint_status_ring, self.maint_status_text],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            visible=False,
            padding=ft.padding.only(bottom=4),
        )
        self.summary_removable = ft.TextButton(
            "Removable: —",
            tooltip="Dead symbols with no price history — safe to remove from DB and CSV",
            on_click=lambda e: self._open_maintenance_dialog(focus="removable"),
        )
        self.summary_dead_history = ft.TextButton(
            "Dead (history): —",
            tooltip="Cannot pull new prices but local history exists — archive, do not delete",
            on_click=lambda e: self._open_maintenance_dialog(focus="history"),
        )

        self.add_field = InputStyles.text_field(
            page,
            label="Add symbol(s)",
            hint_text="AAPL or AAPL, MSFT",
            expand=True,
            on_submit=self._on_add,
            on_blur=on_ticker_field_blur(multi=True),
            tooltip=_TIP_ADD,
        )
        self.add_btn = ft.ElevatedButton(
            "Add",
            icon=ft.Icons.ADD,
            style=ButtonStyles.primary(),
            on_click=self._on_add,
            tooltip=_TIP_ADD,
        )
        self.search_field = InputStyles.text_field(
            page,
            label="Search",
            width=200,
            on_change=self._on_search_changed,
            on_blur=on_ticker_field_blur(multi=False),
            tooltip=_TIP_SEARCH,
        )
        self.filter_dropdown = InputStyles.dropdown(
            page,
            label="Show",
            width=200,
            value="all",
            tooltip=_TIP_FILTER,
            options=[
                ft.dropdown.Option("all", "All symbols"),
                ft.dropdown.Option("warnings", "Warnings"),
                ft.dropdown.Option("dead_no_history", "Dead — no history"),
                ft.dropdown.Option("dead_with_history", "Dead — has history"),
                ft.dropdown.Option("archived", "Archived"),
            ],
            on_select=self._on_filter_changed,
        )
        self.prev_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT,
            tooltip=_TIP_PREV_PAGE,
            on_click=self._on_prev_page,
        )
        self.next_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT,
            tooltip=_TIP_NEXT_PAGE,
            on_click=self._on_next_page,
        )
        self.page_label = ft.Text("Page 1", size=12)

        self.ticker_table = themed_data_table(
            page,
            columns=[
                data_column("Symbol", tooltip="Ticker symbol in the research universe."),
                data_column("Pool", tooltip="focus = daily watchlist; universe = research set."),
                data_column("Status", tooltip="Active, archived, or validation state."),
                data_column("Reason", tooltip="Why a symbol is paused or flagged."),
                data_column("History", tooltip="Whether price history exists in the database."),
                data_column("Coverage", tooltip=_COVERAGE_HEADER_TOOLTIP),
                data_column("Last validated", tooltip="Last validation timestamp."),
                data_column("Last ingest", tooltip="Last successful ingest timestamp."),
                data_column("Actions", tooltip="Validate, ingest, or remove symbol."),
            ],
            rows=[],
        )
        self._ticker_table_wrap = bordered_table_wrap(page, self.ticker_table, expand=True)

        self.validate_btn = ft.ElevatedButton(
            "Validate page",
            icon=ft.Icons.VERIFIED_USER,
            style=ButtonStyles.secondary(),
            tooltip=_VALIDATE_PAGE_TOOLTIP,
            on_click=self._on_validate_page,
        )
        self.maintain_btn = ft.ElevatedButton(
            "Maintain watchlist…",
            icon=ft.Icons.BUILD_CIRCLE,
            style=ButtonStyles.secondary(),
            tooltip="Review and remove dead symbols, investigate delistings",
            on_click=lambda e: self._open_maintenance_dialog(),
        )
        self.skip_dead_btn = ft.ElevatedButton(
            "Archive dead tickers",
            icon=ft.Icons.BLOCK,
            style=ButtonStyles.secondary(),
            tooltip="Pause updates for symbols with No Price Data or delisted/invalid validation",
            on_click=self._on_skip_dead,
        )
        self.prune_csv_btn = ft.ElevatedButton(
            "Remove dead from CSV",
            icon=ft.Icons.CONTENT_CUT,
            style=ButtonStyles.secondary(),
            tooltip=(
                "Remove removable dead symbols (no price history) from tickerList.csv; "
                "creates a timestamped backup"
            ),
            on_click=self._on_prune_csv,
        )
        self.suggest_hint = ft.Text(
            "Ticker suggestions — coming soon",
            size=11,
            italic=True,
            color=ThemeHelper.text_muted(page),
        )

        self.db_field = InputStyles.text_field(
            page,
            label="Database path",
            value=cfg.db_path,
            expand=True,
        )
        self.ingest_mode_dropdown = InputStyles.dropdown(
            page,
            label="Ingest mode",
            value=cfg.ingest_mode_default,
            width=180,
            tooltip=_TIP_INGEST_MODE,
            options=[
                ft.dropdown.Option("smart", "Smart update (incremental)"),
                ft.dropdown.Option("full", "Full re-download"),
            ],
        )
        self.retry_dead_switch = ft.Switch(
            label="Re-try skipped dead tickers",
            value=cfg.retry_dead_tickers,
            tooltip="When off, ingest skips archived, delisted, and No Price Data symbols",
        )
        self.progress = ft.ProgressBar(value=0, visible=False)
        self.eta_text = ft.Text(
            "",
            size=12,
            color=ThemeHelper.text_muted(page),
        )
        self.status_text = ft.Text("", size=12)
        self.log_text = InputStyles.text_field(
            page,
            read_only=True,
            multiline=True,
            expand=True,
            min_lines=14,
            text_size=11,
            hint_text="Ingest log (scroll for full session history)",
        )
        self.run_btn = ft.ElevatedButton(
            "Start universe ingest",
            icon=ft.Icons.CLOUD_DOWNLOAD,
            style=ButtonStyles.primary(),
            on_click=self._on_run,
            tooltip="Weekly bulk update for the full research symbol database (slow).",
        )
        self.stop_btn = ft.ElevatedButton(
            "Stop ingest",
            icon=ft.Icons.STOP,
            style=ButtonStyles.secondary(),
            disabled=True,
            on_click=self._on_stop,
            tooltip=_TIP_STOP_INGEST,
        )
        self.resume_btn = ft.ElevatedButton(
            "Resume paused run",
            icon=ft.Icons.PLAY_ARROW,
            style=ButtonStyles.secondary(),
            visible=False,
            on_click=self._on_resume,
            tooltip=_TIP_RESUME_INGEST,
        )
        self.audit_btn = ft.ElevatedButton(
            "Audit gaps",
            icon=ft.Icons.FACT_CHECK,
            style=ButtonStyles.secondary(),
            on_click=self._on_audit,
            tooltip=_TIP_AUDIT_GAPS,
        )

        self.db_status_text = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self.db_metric_focus = ft.Text("Focus: —", size=12)
        self.db_metric_universe = ft.Text("Universe: —", size=12)
        self.db_metric_tickers = ft.Text("With data: —", size=12)
        self.db_metric_dates = ft.Text("Range: —", size=12)

        log_panel = ft.Container(
            content=self.log_text,
            height=240,
            border=ft.border.all(1, ThemeHelper.border_default(page)),
            border_radius=8,
            padding=8,
        )

        self.controls = [
            ViewTitleBar("Data Management"),
            ft.Divider(),
            SectionHeader("Database health", icon=ft.Icons.STORAGE, page_ref=page),
            self.db_status_text,
            ft.Row(
                [
                    self.db_metric_focus,
                    self.db_metric_universe,
                    self.db_metric_tickers,
                    self.db_metric_dates,
                ],
                spacing=16,
                wrap=True,
            ),
            SectionHeader("Research universe symbols", icon=ft.Icons.LIST_ALT, page_ref=page),
            ft.Row(
                [
                    self.summary_total,
                    self.summary_active,
                    self.summary_focus,
                    self.summary_archived,
                    self.summary_warnings,
                    self.summary_removable,
                    self.summary_dead_history,
                ],
                spacing=16,
                wrap=True,
            ),
            self.universe_ingest_hint,
            self._maint_status_row,
            ft.Row([self.add_field, self.add_btn], spacing=8),
            ft.Row(
                [
                    self.search_field,
                    self.filter_dropdown,
                    self.validate_btn,
                    self.maintain_btn,
                    self.skip_dead_btn,
                    self.prune_csv_btn,
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                wrap=True,
            ),
            self.suggest_hint,
            ft.Row(
                [self.prev_btn, self.page_label, self.next_btn],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Container(content=self._ticker_table_wrap, padding=4, expand=True),
            SectionHeader("Ingest", icon=ft.Icons.CLOUD_DOWNLOAD, page_ref=page),
            ft.Row([self.db_field]),
            ft.Row(
                [
                    self.ingest_mode_dropdown,
                    self.retry_dead_switch,
                    self.run_btn,
                    self.stop_btn,
                    self.resume_btn,
                    self.audit_btn,
                ],
                spacing=12,
                wrap=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            self.progress,
            self.eta_text,
            self.status_text,
            SectionHeader("Log", icon=ft.Icons.TERMINAL, page_ref=page),
            log_panel,
        ]

    def _db_path(self) -> str:
        return (self.db_field.value or stock_config().db_path).strip()

    def _set_maint_status(self, message: str, *, busy: bool = False) -> None:
        self._maint_status_row.visible = busy or bool(message)
        self.maint_status_ring.visible = busy
        self.maint_status_text.value = message

        def _ui():
            try:
                self._maint_status_row.update()
                self.maint_status_ring.update()
                self.maint_status_text.update()
            except RuntimeError:
                pass

        self._safe_update_critical(_ui)

    def _clear_maint_status(self) -> None:
        self._set_maint_status("", busy=False)
        self._maint_status_row.visible = False

        def _ui():
            try:
                self._maint_status_row.update()
            except RuntimeError:
                pass

        self._safe_update_critical(_ui)

    def _set_maint_busy(self, busy: bool) -> None:
        blocked = busy or self._running or self._validating
        self.maintain_btn.disabled = blocked
        self.skip_dead_btn.disabled = blocked
        self.prune_csv_btn.disabled = blocked
        self.validate_btn.disabled = blocked or self._validating
        self.add_btn.disabled = blocked

        def _ui():
            try:
                self.maintain_btn.update()
                self.skip_dead_btn.update()
                self.prune_csv_btn.update()
                self.validate_btn.update()
                self.add_btn.update()
            except RuntimeError:
                pass

        self._safe_update_critical(_ui)

    def _run_db_maintenance(
        self,
        *,
        label: str,
        status_message: str,
        work,
        on_success,
        on_error=None,
    ) -> bool:
        """Run a blocking registry/DB task on a worker thread with loading UI."""
        if self._maintaining:
            show_snackbar(self.page_ref, "Database update already in progress.", severity="warning")
            return False
        if not self._check_permissions():
            return False
        self._maintaining = True
        self._set_maint_busy(True)
        self._set_maint_status(status_message, busy=True)

        def _work():
            err: Exception | None = None
            result = None
            activity_label = (label or "database maintenance").replace("_", " ").title()
            self.global_control.push_db_activity(activity_label)
            try:
                result = work()
            except Exception as ex:
                err = ex
            finally:
                self.global_control.pop_db_activity()

            def _finish():
                self._maintaining = False
                self._set_maint_busy(False)
                if err is not None:
                    msg = on_error(err) if on_error else f"Update failed: {err}"
                    self._set_maint_status(str(msg)[:160], busy=False)
                    show_snackbar(self.page_ref, str(msg)[:160], severity="error")
                else:
                    on_success(result)
                    run_summary = (result or {}).get("run_summary") if isinstance(result, dict) else None
                    if run_summary:
                        from src.analysis.db_perf import DbRunSummary

                        summary = DbRunSummary(
                            operation=run_summary.get("operation", label or "maintenance"),
                            duration_sec=float(run_summary.get("duration_sec", 0)),
                            items_processed=int(run_summary.get("items_processed", 0)),
                            items_total=int(run_summary.get("items_total", 0)),
                            success_count=int(run_summary.get("success_count", 0)),
                            notes=run_summary.get("notes", ""),
                        )
                        self._set_maint_status(summary.format_short(), busy=False)
                    else:
                        self._clear_maint_status()
                    try:
                        from src.services.event_bus import event_bus

                        event_bus.emit("ingest_finished", source=label)
                    except Exception as ex:
                        from src.utils.logger_utils import app_logger

                        app_logger.log(
                            "DATA_MGMT",
                            "ingest_finished emit failed.",
                            level="DEBUG",
                            error=str(ex),
                        )
                    self.refresh_data_async(label=label)

            self._safe_update_critical(_finish)

        threading.Thread(target=_work, daemon=True).start()
        return True

    def _search(self) -> str | None:
        s = (self.search_field.value or "").strip()
        return s if s else None

    def _clear_log(self) -> None:
        self._session_log_lines.clear()
        self.log_text.value = ""

    def _append_log(self, message: str) -> None:
        if not message:
            return
        self._session_log_lines.append(message)
        self.log_text.value = "\n".join(self._session_log_lines)
        try:
            self.log_text.update()
        except RuntimeError:
            pass

    def _flush_session_log_to_file(self) -> None:
        if not self._session_log_lines:
            return
        try:
            from src.utils.logger_utils import export_session_log_to_file

            export_session_log_to_file(self._session_log_lines, "ingest_session")
        except Exception as ex:
            from src.utils.logger_utils import app_logger

            app_logger.log("DATA_MGMT", "Session log export failed.", level="DEBUG", error=str(ex))

    def _coverage_label(self, row: registry.WatchlistRow) -> str:
        return format_coverage_label(row)

    def _status_label(self, row: registry.WatchlistRow) -> str:
        if row.skip_ingest:
            return "archived"
        if row.validation_status and row.validation_status not in ("ok", "archived"):
            return row.validation_status
        if row.last_ingest_status and row.last_ingest_status != "Success":
            return row.last_ingest_status[:40]
        return row.last_ingest_status or "—"

    def _labeled_cell(self, text: str, tooltip: str, *, size: int = 11) -> ft.Control:
        return ft.Text(text, size=size, tooltip=tooltip)

    def _build_action_row(self, row: registry.WatchlistRow) -> ft.Row:
        sym = row.symbol

        def _nav(_e):
            navigate_to_stock_detail(sym, run_analysis=True)

        def _archive(_e):
            registry.set_skip_ingest(self._db_path(), sym, True)
            show_snackbar(self.page_ref, f"{sym}: updates paused, history kept.", severity="info")
            self.refresh_data_async(label="archive")

        def _unarchive(_e):
            registry.set_skip_ingest(self._db_path(), sym, False)
            show_snackbar(self.page_ref, f"{sym}: updates resumed.", severity="success")
            self.refresh_data_async(label="unarchive")

        def _remove(_e):
            self._confirm_remove(row)

        def _promote_focus(_e):
            result = registry.add_to_focus(self._db_path(), sym)
            if result.get("ok"):
                show_snackbar(self.page_ref, f"{sym} added to focus watchlist.", severity="success")
                self.refresh_data_async(label="promote_focus")
            else:
                show_snackbar(self.page_ref, result.get("error", "Could not add"), severity="error")

        def _investigate(_e):
            db_path = self._db_path()

            def _work():
                try:
                    from src.analysis.ticker_investigation import investigate_and_record

                    inv = investigate_and_record(db_path, sym, use_sec=True)
                    msg = inv.summary_text()
                except Exception as ex:
                    msg = f"Investigation failed: {ex}"

                def _fin():
                    show_snackbar(self.page_ref, msg[:200], severity="info")
                    self.refresh_data_async(label="investigate")

                self._safe_update_critical(_fin)

            threading.Thread(target=_work, daemon=True).start()

        buttons = [
            ft.IconButton(
                icon=ft.Icons.QUERY_STATS,
                tooltip="Open in Single-Ticker",
                on_click=_nav,
            ),
        ]
        if row.pool != registry.POOL_FOCUS:
            buttons.append(
                ft.IconButton(
                    icon=ft.Icons.BOOKMARK_ADD,
                    tooltip="Promote to focus watchlist (max 20)",
                    on_click=_promote_focus,
                ),
            )
        buttons.append(
            ft.IconButton(
                icon=ft.Icons.DELETE_OUTLINE,
                tooltip="Remove from research universe",
                on_click=_remove,
            ),
        )
        if registry.is_dead_symbol(row) or row.skip_ingest:
            buttons.insert(
                2,
                ft.IconButton(
                    icon=ft.Icons.SEARCH,
                    tooltip="Investigate delist reason (Yahoo + SEC)",
                    on_click=_investigate,
                ),
            )
        if row.skip_ingest:
            buttons.insert(
                1,
                ft.IconButton(
                    icon=ft.Icons.PLAY_ARROW,
                    tooltip="Resume updates",
                    on_click=_unarchive,
                ),
            )
        else:
            buttons.insert(
                1,
                ft.IconButton(
                    icon=ft.Icons.PAUSE_CIRCLE_OUTLINE,
                    tooltip="Keep data, stop updates",
                    on_click=_archive,
                ),
            )
        return ft.Row(buttons, spacing=0)

    def _confirm_remove(self, row: registry.WatchlistRow) -> None:
        sym = row.symbol
        purge_check = ft.Checkbox(
            label="Also delete historical price data from database",
            value=False,
            visible=row.has_history,
        )
        csv_check = ft.Checkbox(
            label="Also remove from tickerList.csv",
            value=True,
        )

        def _do_remove(_e):
            purge = bool(purge_check.value) if row.has_history else False
            update_csv = bool(csv_check.value)

            def _work():
                result = registry.remove_symbol(self._db_path(), sym, purge_history=purge)
                if result.get("removed") and update_csv:
                    cfg = stock_config()
                    csv_path = (cfg.ticker_csv_path or "").strip()
                    if csv_path:
                        try:
                            from src.analysis.ticker_cleanup import apply_ticker_cleanup

                            apply_ticker_cleanup(csv_path, [sym], backup=True)
                        except Exception as ex:
                            from src.utils.logger_utils import app_logger

                            app_logger.log(
                                "DATA_MGMT",
                                "Ticker cleanup after remove failed.",
                                level="DEBUG",
                                symbol=sym,
                                error=str(ex),
                            )

            def _on_success(result):
                if result.get("removed"):
                    msg = f"Removed {sym}"
                    if result.get("purged"):
                        msg += " (historical data deleted)"
                    if update_csv:
                        msg += " (CSV updated)"
                    show_snackbar(self.page_ref, msg, severity="success")

            if self._run_db_maintenance(
                label="remove",
                status_message=f"Removing {sym} from database…",
                work=_work,
                on_success=_on_success,
            ):
                self.page_ref.close(dlg)

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Remove {sym}?"),
            content=ft.Column(
                [
                    ft.Text(
                        "Removes the symbol from your watchlist."
                        if row.has_history
                        else "No price history found; symbol and any metadata will be removed."
                    ),
                    purge_check,
                    csv_check,
                ],
                tight=True,
                spacing=8,
            ),
            actions=[
                ft.TextButton(
                    "Cancel",
                    tooltip="Keep this symbol on the watchlist.",
                    on_click=lambda e: self.page_ref.close(dlg),
                ),
                ft.ElevatedButton(
                    "Remove",
                    style=ButtonStyles.destructive(),
                    tooltip=_TIP_REMOVE_CONFIRM,
                    on_click=_do_remove,
                ),
            ],
        )
        self.page_ref.open(dlg)

    def _apply_table(self, rows: list[registry.WatchlistRow], total: int) -> None:
        page_num = self._page_offset // _PAGE_SIZE + 1
        max_page = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self.page_label.value = f"Page {page_num} of {max_page} ({total} symbols)"
        self.prev_btn.disabled = self._page_offset <= 0
        self.next_btn.disabled = self._page_offset + _PAGE_SIZE >= total

        self.ticker_table.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(r.symbol, weight=ft.FontWeight.W_500)),
                    ft.DataCell(
                        ft.Text(
                            "focus" if r.pool == registry.POOL_FOCUS else "universe",
                            size=11,
                            color=ThemeHelper.text_primary(self.page_ref)
                            if r.pool == registry.POOL_FOCUS
                            else None,
                        ),
                    ),
                    ft.DataCell(
                        self._labeled_cell(
                            self._status_label(r),
                            registry.compose_status_tooltip(r),
                        )
                    ),
                    ft.DataCell(
                        self._labeled_cell(
                            format_reason_label(r),
                            registry.compose_status_tooltip(r),
                        )
                    ),
                    ft.DataCell(ft.Text("Yes" if r.has_history else "No", size=11)),
                    ft.DataCell(
                        self._labeled_cell(
                            self._coverage_label(r),
                            format_coverage_tooltip(r),
                        )
                    ),
                    ft.DataCell(ft.Text((r.last_validated_at or "—")[:16], size=11)),
                    ft.DataCell(ft.Text(r.last_ingest_at or "—", size=11)),
                    ft.DataCell(self._build_action_row(r)),
                ],
            )
            for r in rows
        ]
        try:
            self.ticker_table.update()
            self.page_label.update()
            self.prev_btn.update()
            self.next_btn.update()
        except RuntimeError:
            pass

    def _apply_summary(self, summary: dict[str, int]) -> None:
        cfg = stock_config()
        cap = summary.get("focus_cap", cfg.focus_watchlist_max)
        self.summary_total.value = f"Universe: {summary.get('universe', summary.get('active', 0))}"
        self.summary_active.value = f"Active: {summary.get('active', 0)}"
        self.summary_focus.value = f"Focus: {summary.get('focus', 0)}/{cap}"
        self.summary_archived.value = f"Archived: {summary.get('archived', 0)}"
        self.summary_warnings.value = f"Warnings: {summary.get('warnings', 0)}"
        self.summary_removable.text = f"Removable: {summary.get('removable', 0)}"
        self.summary_dead_history.text = f"Dead (history): {summary.get('dead_with_history', 0)}"
        self._update_universe_ingest_hint(cfg)

    def _update_universe_ingest_hint(self, cfg) -> None:
        from datetime import datetime, timedelta, timezone

        last = cfg.last_universe_ingest_at
        interval = max(1, int(cfg.universe_ingest_interval_days or 7))
        if not last:
            self.universe_ingest_hint.value = (
                f"No universe ingest recorded yet — run weekly (every {interval} days)."
            )
            self.universe_ingest_hint.color = ThemeHelper.text_error(self.page_ref)
            return
        try:
            ts = datetime.fromisoformat(last.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - ts).days
        except ValueError:
            age_days = interval + 1
        if age_days >= interval:
            self.universe_ingest_hint.value = (
                f"Universe ingest is {age_days} days old (target: every {interval} days). "
                "Consider running Start universe ingest."
            )
            self.universe_ingest_hint.color = ThemeHelper.text_error(self.page_ref)
        else:
            self.universe_ingest_hint.value = (
                f"Last universe ingest: {last[:16]} ({age_days} day(s) ago)."
            )
            self.universe_ingest_hint.color = ThemeHelper.text_muted(self.page_ref)

    def _fetch_data(self) -> dict:
        cfg = stock_config()
        db_path = self._db_path()
        registry.auto_migrate_if_needed(db_path, cfg.ticker_csv_path)
        search = self._search()
        wf = self._watchlist_filter
        total = registry.count_symbols(
            db_path,
            include_archived=True,
            watchlist_filter=wf,
            search=search,
        )
        rows = registry.list_symbols(
            db_path,
            include_archived=True,
            watchlist_filter=wf,
            search=search,
            limit=_PAGE_SIZE,
            offset=self._page_offset,
        )
        summary = registry.count_summary(db_path)
        db_summary = get_db_summary(db_path, light=True)
        paused = get_resumable_run(db_path)
        self.resume_btn.visible = paused is not None and not self._running
        try:
            self.resume_btn.update()
        except RuntimeError:
            pass
        return {"rows": rows, "total": total, "summary": summary, "db_summary": db_summary}

    def _apply_db_health(self, db_summary: dict, summary: dict) -> None:
        if not db_summary.get("exists"):
            self.db_status_text.value = "No database found — run ingest below."
            return
        if db_summary.get("locked"):
            self.db_status_text.value = "Database busy — metrics may be stale."
        else:
            self.db_status_text.value = "Database ready."
        cap = summary.get("focus_cap", stock_config().focus_watchlist_max)
        self.db_metric_focus.value = f"Focus: {summary.get('focus', 0)}/{cap}"
        self.db_metric_universe.value = f"Universe: {summary.get('universe', summary.get('active', 0))}"
        self.db_metric_tickers.value = f"With data: {db_summary.get('ticker_count', 0)}"
        dmin = db_summary.get("date_min") or "?"
        dmax = db_summary.get("date_max") or "?"
        self.db_metric_dates.value = f"Range: {dmin} → {dmax}"

    def _apply_data(self, data: dict) -> None:
        summary = data.get("summary") or {}
        self._apply_db_health(data.get("db_summary") or {}, summary)
        self._apply_summary(summary)
        self._apply_table(data.get("rows") or [], data.get("total", 0))
        try:
            self.update()
        except RuntimeError:
            pass

    def _on_add(self, e) -> None:
        symbols = parse_symbols(self.add_field.value or "")
        if not symbols:
            show_snackbar(self.page_ref, "Enter at least one symbol.", severity="warning")
            return
        added = registry.add_symbols_bulk(self._db_path(), symbols)
        self.add_field.value = ""
        show_snackbar(
            self.page_ref,
            f"Added {added} symbol(s) to research universe.",
            severity="success",
        )
        self.refresh_data_async(label="add")

    def _on_search_changed(self, e) -> None:
        self._page_offset = 0
        self._safe_update_debounced("search", 0.35, lambda: self.refresh_data_async(label="search"))

    def _on_filter_changed(self, e) -> None:
        val = self.filter_dropdown.value or "all"
        self._watchlist_filter = None if val == "all" else val  # type: ignore[assignment]
        self._page_offset = 0
        self.refresh_data_async(label="filter_changed")

    def _open_maintenance_dialog(self, *, focus: str = "removable") -> None:
        db_path = self._db_path()
        removable = registry.list_dead_symbols(db_path, has_history=False, limit=200)
        with_hist = registry.list_dead_symbols(db_path, has_history=True, limit=200)

        def _sym_list(rows: list[registry.WatchlistRow], limit: int = 12) -> str:
            if not rows:
                return "(none)"
            syms = ", ".join(r.symbol for r in rows[:limit])
            if len(rows) > limit:
                syms += f", … (+{len(rows) - limit} more)"
            return syms

        rem_text = ft.Text(
            f"Removable ({len(removable)}): no Yahoo price and no DB history.\n"
            f"{_sym_list(removable)}",
            size=12,
            selectable=True,
        )
        hist_text = ft.Text(
            f"Keep history ({len(with_hist)}): dead but historical data remains.\n"
            f"{_sym_list(with_hist)}",
            size=12,
            selectable=True,
        )
        csv_remove = ft.Checkbox(
            label="Also remove from tickerList.csv when deleting removable symbols",
            value=True,
        )
        use_sec = ft.Checkbox(label="Use SEC EDGAR lookup when investigating", value=True)

        dlg_ref: list[ft.AlertDialog] = []

        def _close(_e=None):
            if dlg_ref:
                self.page_ref.close(dlg_ref[0])

        def _bulk_remove(_e):
            if not removable:
                show_snackbar(self.page_ref, "No removable symbols.", severity="info")
                return
            cfg = stock_config()
            csv_path = (cfg.ticker_csv_path or "").strip() if csv_remove.value else None
            remove_csv = bool(csv_remove.value and csv_path)

            def _progress(done: int, total: int, message: str) -> None:
                pct = (done / total) if total else 0.0

                def _ui():
                    self._set_maint_status(message[:160], busy=True)
                    if hasattr(self, "_maint_status_row") and self._maint_status_row.bar:
                        self._maint_status_row.bar.visible = total > 0
                        self._maint_status_row.bar.value = pct

                self._safe_update_critical(_ui)

            def _work():
                return registry.bulk_remove_dead_no_history(
                    db_path,
                    csv_path or "",
                    remove_from_csv=remove_csv,
                    progress_callback=_progress,
                )

            def _on_success(result):
                n = result.get("removed_db", 0)
                show_snackbar(
                    self.page_ref,
                    f"Removed {n} symbol(s) from database"
                    + (" and CSV" if remove_csv else ""),
                    severity="success",
                )

            if self._run_db_maintenance(
                label="bulk_remove",
                status_message=f"Removing {len(removable)} dead symbol(s) from database…",
                work=_work,
                on_success=_on_success,
            ):
                _close()

        def _bulk_archive(_e):
            def _work():
                return registry.bulk_skip_dead_tickers(db_path)

            def _on_success(result):
                show_snackbar(
                    self.page_ref,
                    f"Archived {result.get('skipped', 0)} dead ticker(s).",
                    severity="success",
                )

            if self._run_db_maintenance(
                label="bulk_archive",
                status_message="Archiving dead symbols in database…",
                work=_work,
                on_success=_on_success,
            ):
                _close()

        def _validate_removable(_e):
            syms = [r.symbol for r in removable]
            if not syms:
                return

            def _work():
                try:
                    validate_symbols(db_path, syms[:50], progress_callback=None)
                    msg = f"Validated {min(len(syms), 50)} removable symbol(s)."
                except Exception as ex:
                    msg = f"Validation failed: {ex}"

                def _fin():
                    show_snackbar(self.page_ref, msg[:120], severity="success")
                    self.refresh_data_async(label="maint_validate")

                self._safe_update_critical(_fin)

            threading.Thread(target=_work, daemon=True).start()

        def _investigate_dead(_e):
            syms = [r.symbol for r in (removable + with_hist)[:30]]
            if not syms:
                return

            def _work():
                try:
                    investigate_symbols(
                        db_path, syms, use_sec=bool(use_sec.value), progress_callback=None
                    )
                    msg = f"Investigated {len(syms)} symbol(s)."
                except Exception as ex:
                    msg = f"Investigation failed: {ex}"

                def _fin():
                    show_snackbar(self.page_ref, msg[:120], severity="success")
                    self.refresh_data_async(label="maint_investigate")

                self._safe_update_critical(_fin)

            threading.Thread(target=_work, daemon=True).start()

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Maintain watchlist"),
            content=ft.Column(
                [
                    ft.Text(
                        "Removable symbols can be deleted from the database and CSV. "
                        "Symbols with history should be archived, not deleted.",
                        size=12,
                    ),
                    rem_text,
                    hist_text,
                    csv_remove,
                    use_sec,
                ],
                tight=True,
                spacing=10,
                scroll=ft.ScrollMode.AUTO,
                height=320,
            ),
            actions=[
                ft.TextButton("Close", tooltip=_TIP_MAINT_CLOSE, on_click=_close),
                ft.TextButton(
                    "Validate removable (50 max)",
                    tooltip=_TIP_MAINT_VALIDATE,
                    on_click=_validate_removable,
                ),
                ft.TextButton(
                    "Investigate (30 max)",
                    tooltip=_TIP_MAINT_INVESTIGATE,
                    on_click=_investigate_dead,
                ),
                ft.ElevatedButton(
                    "Archive dead",
                    tooltip=_TIP_MAINT_ARCHIVE,
                    on_click=_bulk_archive,
                ),
                ft.ElevatedButton(
                    "Remove removable",
                    style=ButtonStyles.destructive(),
                    tooltip=_TIP_MAINT_REMOVE,
                    on_click=_bulk_remove,
                ),
            ],
        )
        dlg_ref.append(dlg)
        self.page_ref.open(dlg)

    def _on_prev_page(self, e) -> None:
        if self._page_offset >= _PAGE_SIZE:
            self._page_offset -= _PAGE_SIZE
            self.refresh_data_async(label="prev_page")

    def _on_next_page(self, e) -> None:
        self._page_offset += _PAGE_SIZE
        self.refresh_data_async(label="next_page")

    def _on_validate_page(self, e) -> None:
        if self._validating:
            return
        if not self._check_permissions():
            return
        data = self._fetch_data()
        symbols = [r.symbol for r in data.get("rows") or []]
        if not symbols:
            show_snackbar(self.page_ref, "No symbols on this page.", severity="warning")
            return
        self._validating = True
        self._set_maint_busy(True)
        total = len(symbols)
        self._set_maint_status(f"Validating {total} symbol(s) against Yahoo…", busy=True)

        def _progress(pct: float, sym: str) -> None:
            done = max(1, min(total, int(round(pct * total))))
            self._set_maint_status(f"Validating {sym} ({done}/{total})…", busy=True)

        def _work():
            success = False
            try:
                self.global_control.push_db_activity("Validate page")
                validate_symbols(self._db_path(), symbols, progress_callback=_progress)
                msg = f"Validated {total} symbol(s)."
                success = True
            except Exception as ex:
                msg = f"Validation failed: {ex}"
            finally:
                self.global_control.pop_db_activity()

            def _finish():
                self._validating = False
                self._set_maint_busy(False)
                self._set_maint_status(msg[:160], busy=False)
                show_snackbar(
                    self.page_ref,
                    msg[:120],
                    severity="success" if success else "error",
                )
                self.refresh_data_async(label="validate_done")

            self._safe_update_critical(_finish)

        threading.Thread(target=_work, daemon=True).start()

    def _on_skip_dead(self, e) -> None:
        db_path = self._db_path()

        def _work():
            return registry.bulk_skip_dead_tickers(db_path)

        def _on_success(result):
            n = result.get("skipped", 0)
            show_snackbar(
                self.page_ref,
                f"Archived {n} dead ticker(s) (no price / delisted / invalid).",
                severity="success",
            )

        self._run_db_maintenance(
            label="skip_dead",
            status_message="Archiving dead tickers in database…",
            work=_work,
            on_success=_on_success,
        )

    def _on_prune_csv(self, e) -> None:
        cfg = stock_config()
        csv_path = (cfg.ticker_csv_path or "").strip()
        if not csv_path:
            show_snackbar(self.page_ref, "Set ticker CSV path in Settings first.", severity="warning")
            return
        db_path = self._db_path()

        def _work():
            return registry.prune_ticker_csv_from_watchlist(db_path, csv_path, backup=True)

        def _on_success(result):
            msg = (
                f"CSV updated: removed {result.get('removed_count', 0)} removable dead "
                f"symbol(s) ({result.get('after_count', '?')} remain)."
            )
            if result.get("backup_path"):
                msg += f" Backup: {result['backup_path']}"
            show_snackbar(self.page_ref, msg[:200], severity="success")

        self._run_db_maintenance(
            label="prune_csv",
            status_message="Removing dead symbols from ticker CSV…",
            work=_work,
            on_success=_on_success,
        )

    def _on_stop(self, e) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.status_text.value = "Stopping after current ticker…"
            try:
                self.status_text.update()
            except RuntimeError:
                pass

    def _on_resume(self, e) -> None:
        self._on_run(e, resume=True)

    def _on_audit(self, e) -> None:
        if not self._check_permissions():
            return
        db_path = self._db_path()

        def _work():
            try:
                report = audit_watchlist(db_path)
                msg = (
                    f"Audit: {report.get('current', 0)} current, "
                    f"{report.get('stale', 0)} stale, "
                    f"{report.get('gaps', 0)} with gaps, "
                    f"{report.get('empty', 0)} empty"
                )
            except Exception as ex:
                msg = f"Audit failed: {ex}"

            def _finish():
                show_snackbar(self.page_ref, msg[:160], severity="success" if "Audit:" in msg else "error")
                self.refresh_data_async(label="audit_done")

            self._safe_update_critical(_finish)

        threading.Thread(target=_work, daemon=True).start()

    def _on_run(self, e, *, resume: bool = False) -> None:
        if self._running:
            show_snackbar(self.page_ref, "Ingest already running.", severity="warning")
            return
        if not self._check_permissions():
            return

        cfg = stock_config()
        cfg.db_path = self._db_path()
        cfg.retry_dead_tickers = bool(self.retry_dead_switch.value)
        ingest_mode = "resume" if resume else (self.ingest_mode_dropdown.value or cfg.ingest_mode_default)

        self._running = True
        self._resume_mode = resume
        self._cancel_event = threading.Event()
        self.run_btn.disabled = True
        self.stop_btn.disabled = False
        self.resume_btn.visible = False
        self.progress.visible = True
        self.progress.value = None
        self.eta_text.value = ""
        self.eta_text.visible = True
        self._clear_log()
        self._append_log("Starting ingest…")
        try:
            self.page_ref.update()
        except RuntimeError:
            pass

        def _work():
            self.global_control.push_db_activity("Universe ingest")
            summary_holder: dict = {}
            msg = "Ingest failed."
            started_at = time.perf_counter()
            last_completed = 0
            ingest_total = 0
            finished_baseline: int | None = None
            last_logged_msg: str | None = None

            def _eta_label(run_progress: dict | None) -> str:
                nonlocal finished_baseline
                prog = run_progress or {}
                total = int(prog.get("total") or ingest_total or 0)
                finished = int(prog.get("finished") or 0)
                remaining = int(
                    prog.get("remaining")
                    if prog.get("remaining") is not None
                    else prog.get("pending") or 0
                )
                if finished_baseline is None:
                    finished_baseline = finished
                return format_ingest_eta_label(
                    elapsed_sec=time.perf_counter() - started_at,
                    finished_total=finished,
                    finished_baseline=finished_baseline,
                    remaining=remaining,
                    total=total,
                )

            def progress(
                pct: float,
                status_msg: str,
                run_progress: dict | None = None,
                log_line: str | None = None,
            ) -> None:
                nonlocal last_completed, ingest_total, last_logged_msg
                prog = run_progress or summary_holder.get("run_progress") or {}
                if prog.get("total"):
                    ingest_total = int(prog["total"])
                elif status_msg:
                    m = re.search(r"(\d+)/(\d+)\s+finished", status_msg)
                    if m:
                        ingest_total = max(ingest_total, int(m.group(2)))
                    else:
                        m2 = re.search(r"(\d+)\s+tickers", status_msg, re.IGNORECASE)
                        if m2:
                            ingest_total = max(ingest_total, int(m2.group(1)))
                finished = int(prog.get("finished") or 0)
                if finished > last_completed:
                    last_completed = finished
                eta_label = _eta_label(prog)
                session_note: str | None = None
                if log_line:
                    session_note = log_line
                elif status_msg and (
                    status_msg.startswith("Processing ")
                    or status_msg.startswith("Phase:")
                ):
                    if status_msg != last_logged_msg:
                        last_logged_msg = status_msg
                        session_note = status_msg

                def _ui():
                    self.progress.value = pct if pct >= 0 else None
                    self.status_text.value = status_msg or self.status_text.value
                    self.eta_text.value = eta_label
                    if session_note:
                        self._append_log(session_note)
                    try:
                        self.progress.update()
                        self.eta_text.update()
                        self.status_text.update()
                    except RuntimeError:
                        pass

                self._safe_update_critical(_ui)

            try:
                summary = run_universe_ingest(
                    db_path=cfg.db_path,
                    use_parallel=cfg.ingest_use_parallel,
                    progress_callback=progress,
                    cancel_event=self._cancel_event,
                    mode=ingest_mode,
                    resume=resume,
                    scope="universe",
                )
                summary_holder.update(summary)
                from datetime import datetime, timezone

                if not summary.get("paused") and not summary.get("no_new_data"):
                    cfg.set_last_ingest_at(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
                if summary.get("no_new_data"):
                    msg = summary.get("message") or "No new data available since the last ingest."
                elif summary.get("paused"):
                    pending = summary.get("run_progress", {}).get("pending", "?")
                    msg = (
                        f"Paused — {pending} tickers remaining. "
                        "Use Resume to continue."
                    )
                else:
                    msg = (
                        f"Done: {summary.get('success', 0)}/{summary.get('total', 0)} succeeded, "
                        f"{summary.get('skipped', 0)} skipped. "
                        f"Failed: {summary.get('failed', 0)}. "
                        f"Warnings: {summary.get('warnings', 0)}."
                    )
            except Exception as ex:
                msg = f"Ingest failed: {ex}"
            finally:
                self.global_control.pop_db_activity()

            def _finish():
                self._running = False
                self._resume_mode = False
                self._cancel_event = None
                self.run_btn.disabled = False
                self.stop_btn.disabled = True
                self.progress.visible = False
                self.progress.value = 1.0
                self.eta_text.visible = False
                self.eta_text.value = ""
                elapsed_total = time.perf_counter() - started_at
                elapsed_note = ""
                if ingest_total > 0 and ("Done" in msg or "Paused" in msg):
                    elapsed_note = f" (elapsed {format_duration(elapsed_total)})"
                self.status_text.value = msg + elapsed_note
                self._append_log(msg + elapsed_note)
                self._flush_session_log_to_file()
                if summary_holder.get("no_new_data"):
                    sev = "success"
                elif summary_holder.get("paused"):
                    sev = "warning"
                elif "Done" in msg:
                    sev = "success"
                else:
                    sev = "error"
                show_snackbar(self.page_ref, msg[:120], severity=sev)
                try:
                    from src.services.event_bus import event_bus

                    event_bus.emit("ingest_finished", source="universe_ingest")
                except Exception as ex:
                    from src.utils.logger_utils import app_logger

                    app_logger.log(
                        "DATA_MGMT",
                        "ingest_finished emit failed.",
                        level="DEBUG",
                        error=str(ex),
                    )
                self.refresh_data_async(label="ingest_done")
                try:
                    self.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_finish)

        threading.Thread(target=_work, daemon=True).start()

    def refresh_data(self) -> None:
        self.refresh_data_async(label="data_ingest")

    def refresh_theme(self) -> None:
        self.summary_warnings.color = ThemeHelper.text_error(self.page_ref)
        self.suggest_hint.color = ThemeHelper.text_muted(self.page_ref)
        self.eta_text.color = ThemeHelper.text_muted(self.page_ref)
        InputStyles.refresh_fields(
            self.page_ref,
            [self.add_field, self.search_field, self.db_field, self.log_text],
            [self.filter_dropdown, self.ingest_mode_dropdown],
        )
        kwargs = ThemeHelper.results_summary_data_table_kwargs(self.page_ref)
        for key, val in kwargs.items():
            setattr(self.ticker_table, key, val)
        self._ticker_table_wrap.border = ft.border.all(
            1, ThemeHelper.border_default(self.page_ref)
        )
        try:
            self.update()
        except RuntimeError:
            pass


DataManagementView = DataIngestView
