"""Optimization — focused momentum grid and volatility / ATR."""

from __future__ import annotations

import threading
import time

import flet as ft
import pandas as pd

from src.analysis.db import count_tickers
from src.analysis.focused import run_focused_optimization
from src.analysis.volatility import run_volatility_analysis
from src.services.stock_config import stock_config
from src.utils.progress_format import (
    estimate_eta_seconds,
    format_elapsed,
    format_eta_remaining,
)
from src.views.base_view import BaseView
from src.views.components.feedback import AsyncStatusRow, show_snackbar
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.modals import ConfirmationDialog
from src.views.components.tables import bordered_table_wrap, themed_data_table
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import (
    dataframe_columns,
    dataframe_to_rows,
    on_ticker_field_blur,
    parse_ticker_filter,
)

_LARGE_UNIVERSE_THRESHOLD = 500
_DISPLAY_ROWS = 200


class _OptimizationPanel(ft.Column):
    def __init__(
        self,
        page: ft.Page,
        parent: "OptimizationView",
        title: str,
        *,
        extra_controls: list[ft.Control] | None = None,
        run_tooltip: str | None = None,
    ):
        super().__init__(spacing=10, tight=True)
        self.page_ref = page
        self.parent_view = parent
        self._title = title
        self.summary = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self.results_table = themed_data_table(page, columns=[], rows=[])
        self.ticker_filter = InputStyles.text_field(
            page,
            label="Ticker filter (optional)",
            hint_text="Leave empty for full universe",
            expand=True,
            on_blur=on_ticker_field_blur(multi=True),
            tooltip=(
                "Limit the optimization to specific symbols. "
                "Leave empty to scan the full database (slower)."
            ),
        )
        self.run_btn = ft.ElevatedButton(
            f"Run {title}",
            icon=ft.Icons.PLAY_ARROW,
            style=ButtonStyles.primary(),
            on_click=lambda e: parent.run_panel(self),
            tooltip=run_tooltip
            or f"Run {title} on your database using the parameters above.",
        )
        extras = ft.Row(extra_controls, spacing=12) if extra_controls else None
        self._table_wrap = bordered_table_wrap(page, self.results_table, expand=True)
        results_scroll = ft.Column(
            [self._table_wrap],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )
        self.controls = [
            ft.Row([self.ticker_filter]),
            *([extras] if extras is not None else []),
            ft.Row([self.run_btn]),
            self.summary,
            ft.Container(content=results_scroll, expand=True),
        ]
        self._partial_rows: list[dict] = []

    def set_busy(self, busy: bool) -> None:
        self.run_btn.disabled = busy

    def clear_results(self) -> None:
        self._partial_rows = []
        self.summary.value = ""
        self.results_table.columns = [ft.DataColumn(ft.Text("Info"))]
        self.results_table.rows = [
            ft.DataRow(cells=[ft.DataCell(ft.Text("Running…"))])
        ]
        try:
            self.summary.update()
            self.results_table.update()
        except RuntimeError:
            pass

    def update_partial_results(self, rows: list[dict], *, done: int, total: int) -> None:
        self._partial_rows = rows
        if not rows:
            return
        df = pd.DataFrame(rows)
        sort_col = "Alpha" if "Alpha" in df.columns else df.columns[-1]
        df = df.sort_values(by=sort_col, ascending=False)
        self.summary.value = f"{done:,} / {total:,} complete — top results shown"
        self.results_table.columns = dataframe_columns(df)
        self.results_table.rows = dataframe_to_rows(df, max_rows=_DISPLAY_ROWS)
        try:
            self.summary.update()
            self.results_table.update()
        except RuntimeError:
            pass

    def show_result(self, summary: str, df) -> None:
        self.summary.value = summary
        if df is not None and not df.empty:
            self.results_table.columns = dataframe_columns(df)
            self.results_table.rows = dataframe_to_rows(df, max_rows=_DISPLAY_ROWS)
        else:
            self.results_table.columns = [ft.DataColumn(ft.Text("Info"))]
            self.results_table.rows = [
                ft.DataRow(cells=[ft.DataCell(ft.Text("No results"))])
            ]
        try:
            self.summary.update()
            self.results_table.update()
        except RuntimeError:
            pass


class OptimizationView(BaseView):
    _tab_index = 7

    def __init__(self, page: ft.Page):
        super().__init__(page)

        self.atr_period = InputStyles.text_field(
            page,
            label="ATR period",
            value="14",
            width=100,
            tooltip="Number of bars used to compute Average True Range.",
        )
        self.entry_mult = InputStyles.text_field(
            page,
            label="Entry mult",
            value="2.0",
            width=100,
            tooltip="ATR multiple for simulated entry distance from the moving average.",
        )
        self.exit_mult = InputStyles.text_field(
            page,
            label="Exit mult",
            value="3.0",
            width=100,
            tooltip="ATR multiple for trailing stop or exit threshold.",
        )
        self.async_status = AsyncStatusRow(page)
        self.opt_progress = ft.ProgressBar(value=0, visible=False)
        self.opt_status = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self.opt_eta = ft.Text("", size=11, color=ThemeHelper.text_muted(page))
        self.opt_elapsed = ft.Text("", size=11, color=ThemeHelper.text_muted(page))
        self.stop_btn = ft.ElevatedButton(
            "Stop",
            icon=ft.Icons.STOP,
            style=ButtonStyles.destructive(),
            disabled=True,
            on_click=self._on_stop,
            tooltip="Stop the current optimization run and keep partial results.",
        )
        self._progress_block = ft.Column(
            [
                self.opt_progress,
                self.opt_status,
                ft.Row([self.opt_eta, self.opt_elapsed], spacing=16),
                ft.Row([self.stop_btn]),
            ],
            spacing=6,
            tight=True,
            visible=False,
        )

        self.panel_focused = _OptimizationPanel(
            page,
            self,
            "Focused Momentum",
            run_tooltip=(
                "Grid-search short lookback windows and buy/sell thresholds per ticker "
                "to find momentum parameters that beat buy-and-hold."
            ),
        )
        self.panel_volatility = _OptimizationPanel(
            page,
            self,
            "Volatility / ATR",
            extra_controls=[self.atr_period, self.entry_mult, self.exit_mult],
            run_tooltip=(
                "Rank tickers by ATR-based volatility metrics and simulated "
                "ATR channel entries using your period and multiplier settings."
            ),
        )
        self._panels = [self.panel_focused, self.panel_volatility]
        self._running = False
        self._cancel_event: threading.Event | None = None
        self._partial_results: list[dict] = []
        self._run_started_at: float = 0.0
        self._active_panel: _OptimizationPanel | None = None
        self._total_tickers: int = 0

        self._sub_keys = ["focused", "volatility"]
        self._sub_segmented = ft.SegmentedButton(
            selected=["focused"],
            allow_empty_selection=False,
            allow_multiple_selection=False,
            segments=[
                ft.Segment(
                    value="focused",
                    label=ft.Text("Focused"),
                    icon=ft.Icon(ft.Icons.TUNE),
                    tooltip="Momentum parameter grid search vs buy-and-hold.",
                ),
                ft.Segment(
                    value="volatility",
                    label=ft.Text("Volatility"),
                    icon=ft.Icon(ft.Icons.SPEED),
                    tooltip="ATR-based volatility ranking and channel simulation.",
                ),
            ],
            on_change=self._on_sub_tab_change,
        )
        self._sub_host = ft.Container(
            content=self.panel_focused,
            padding=ft.padding.only(top=8),
        )

        self.controls = [
            ViewTitleBar("Optimization"),
            ft.Divider(),
            SectionHeader("Parameter search", icon=ft.Icons.AUTO_GRAPH, page_ref=page),
            ft.Row([self._sub_segmented]),
            self._progress_block,
            self.async_status,
            self._sub_host,
        ]

    def _on_sub_tab_change(self, e) -> None:
        selected = list(getattr(e.control, "selected", None) or [])
        if not selected:
            return
        key = selected[0]
        try:
            idx = self._sub_keys.index(key)
        except ValueError:
            return
        self._sub_host.content = self._panels[idx]
        try:
            self._sub_host.update()
        except RuntimeError:
            pass

    def _on_stop(self, _e) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.stop_btn.disabled = True
        self.opt_status.value = "Stopping…"
        try:
            self.opt_status.update()
        except RuntimeError:
            pass

    def _begin_progress_ui(self, panel: _OptimizationPanel, total: int) -> None:
        self._active_panel = panel
        self._total_tickers = total
        self._partial_results = []
        self._run_started_at = time.perf_counter()
        panel.clear_results()
        self._progress_block.visible = True
        self.opt_progress.visible = True
        self.opt_progress.value = 0
        self.opt_status.value = f"Starting {panel._title}…"
        self.opt_status.color = ThemeHelper.text_primary(self.page_ref)
        self.opt_eta.value = "Estimating time remaining…"
        self.opt_eta.visible = True
        self.opt_elapsed.value = ""
        self.opt_elapsed.visible = True
        self.stop_btn.disabled = False
        self.async_status.set_running(f"Running {panel._title}…", progress=0.0)
        try:
            self._progress_block.update()
            self.async_status.update()
        except RuntimeError:
            pass

    def _update_progress_ui(
        self,
        *,
        phase: str,
        pct: float,
        done: int,
        total: int,
        detail: str,
    ) -> None:
        elapsed = time.perf_counter() - self._run_started_at
        eta_sec = estimate_eta_seconds(
            elapsed_sec=elapsed,
            done=done,
            total=total,
            min_samples=3,
        )
        eta_label = (
            format_eta_remaining(eta_sec)
            if eta_sec is not None
            else "Estimating time remaining…"
        )
        pct_display = int(max(0.0, min(1.0, pct)) * 100)
        status = f"{detail} — {done:,} / {total:,} ({pct_display}%)"
        self.opt_progress.value = pct
        self.opt_status.value = status
        self.opt_eta.value = eta_label
        self.opt_elapsed.value = format_elapsed(elapsed)
        self.async_status.set_running(status[:120], progress=pct)
        try:
            self.opt_progress.update()
            self.opt_status.update()
            self.opt_eta.update()
            self.opt_elapsed.update()
            self.async_status.update()
        except RuntimeError:
            pass

    def _finish_progress_ui(self, message: str, *, error: bool = False) -> None:
        self.opt_progress.visible = False
        self.opt_progress.value = 0
        self.opt_eta.visible = False
        self.opt_eta.value = ""
        self.opt_elapsed.visible = False
        self.opt_elapsed.value = ""
        self.opt_status.value = message
        self.opt_status.color = (
            ThemeHelper.text_error(self.page_ref)
            if error
            else ThemeHelper.text_muted(self.page_ref)
        )
        self.stop_btn.disabled = True
        if error:
            self.async_status.set_error(message[:200])
        else:
            self.async_status.set_success(message[:200])
        try:
            self._progress_block.update()
            self.async_status.update()
        except RuntimeError:
            pass

    def run_panel(self, panel: _OptimizationPanel) -> None:
        if self._running:
            show_snackbar(self.page_ref, "Optimization already running.", severity="warning")
            return
        cfg = stock_config()
        tickers = parse_ticker_filter(panel.ticker_filter.value)
        if not tickers:
            n = count_tickers(cfg.db_path)
            if n > _LARGE_UNIVERSE_THRESHOLD:
                ConfirmationDialog(
                    "Large optimization run",
                    (
                        f"This will optimize ~{n:,} tickers and may take many hours. "
                        "Consider using a ticker filter for faster results. Continue?"
                    ),
                    on_confirm=lambda _e: self._start_panel_run(panel, tickers, n),
                    confirm_text="Continue",
                ).show(self.page_ref)
                return
            self._start_panel_run(panel, tickers, n)
        else:
            self._start_panel_run(panel, tickers, len(tickers))

    def _start_panel_run(
        self,
        panel: _OptimizationPanel,
        tickers: list[str] | None,
        total_estimate: int,
    ) -> None:
        self._running = True
        self._cancel_event = threading.Event()
        for p in self._panels:
            p.set_busy(True)
        self._begin_progress_ui(panel, total_estimate)
        cfg = stock_config()

        def _work():
            self.global_control.push_db_activity("Optimization")
            summary = ""
            df = None
            cancelled = False
            try:
                def on_progress(phase, pct, done, total, detail):
                    def _ui():
                        self._update_progress_ui(
                            phase=phase,
                            pct=pct,
                            done=done,
                            total=total,
                            detail=detail,
                        )

                    self._safe_update_throttled("opt_progress", 0.4, _ui)

                def on_result(row: dict):
                    self._partial_results.append(row)
                    done = len(self._partial_results)
                    total = total_estimate

                    def _ui():
                        if self._active_panel is not None:
                            self._active_panel.update_partial_results(
                                list(self._partial_results),
                                done=done,
                                total=total,
                            )

                    self._safe_update_throttled("opt_results", 0.5, _ui)

                if panel is self.panel_focused:
                    res = run_focused_optimization(
                        cfg.db_path,
                        target_tickers=tickers or None,
                        use_parallel=cfg.use_parallel,
                        workers=cfg.worker_count,
                        progress_callback=on_progress,
                        result_callback=on_result,
                        cancel_event=self._cancel_event,
                    )
                    if res is None:
                        summary = "No data."
                    else:
                        cancelled = res.cancelled
                        df = res.results_df
                        if cancelled:
                            summary = (
                                f"Stopped — {res.stock_count:,} tickers completed "
                                "(top alpha shown)."
                            )
                        else:
                            summary = (
                                f"Optimized {res.stock_count:,} tickers "
                                "(top alpha shown)."
                            )
                        if not cancelled:
                            cfg.set_last_analysis_label("Focused optimization")
                else:
                    try:
                        atr = int(self.atr_period.value or 14)
                        ent = float(self.entry_mult.value or 2.0)
                        ex = float(self.exit_mult.value or 3.0)
                    except ValueError:
                        atr, ent, ex = 14, 2.0, 3.0
                    res = run_volatility_analysis(
                        cfg.db_path,
                        tickers=tickers or None,
                        atr_period=atr,
                        entry_mult=ent,
                        exit_mult=ex,
                        use_parallel=cfg.use_parallel,
                        workers=cfg.worker_count,
                        progress_callback=on_progress,
                        result_callback=on_result,
                        cancel_event=self._cancel_event,
                    )
                    if res is None:
                        summary = "No data."
                    else:
                        cancelled = res.cancelled
                        df = res.results_df
                        if cancelled:
                            summary = (
                                f"Stopped — {res.stock_count:,} tickers completed."
                            )
                        else:
                            summary = (
                                f"Volatility scan on {res.stock_count:,} tickers."
                            )
                        if not cancelled:
                            cfg.set_last_analysis_label("Volatility analysis")
            except Exception as ex:
                summary = f"Error: {ex}"
            finally:
                self.global_control.pop_db_activity()

            def _ui():
                self._running = False
                self._cancel_event = None
                for p in self._panels:
                    p.set_busy(False)
                panel.show_result(summary, df)
                if summary.lower().startswith("error"):
                    self._finish_progress_ui(summary[:200], error=True)
                else:
                    self._finish_progress_ui(summary[:200])
                show_snackbar(
                    self.page_ref,
                    summary[:100],
                    severity="warning" if cancelled else "info",
                )

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def refresh_data(self) -> None:
        pass

    def refresh_theme(self) -> None:
        InputStyles.refresh_fields(
            self.page_ref,
            [
                self.atr_period,
                self.entry_mult,
                self.exit_mult,
                self.panel_focused.ticker_filter,
                self.panel_volatility.ticker_filter,
            ],
        )
        self.async_status.refresh_theme(self.page_ref)
        for p in self._panels:
            p.summary.color = ThemeHelper.text_muted(self.page_ref)
        self.opt_status.color = ThemeHelper.text_muted(self.page_ref)
        self.opt_eta.color = ThemeHelper.text_muted(self.page_ref)
        self.opt_elapsed.color = ThemeHelper.text_muted(self.page_ref)
