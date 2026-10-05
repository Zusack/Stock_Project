"""Strategy backtests — presets, custom builder, and compare mode."""

from __future__ import annotations

import json
import threading

import flet as ft
import pandas as pd

from src.analysis.ai.backtest_advisor import (
    analyze_backtest_results,
    backtest_assistant_prefill,
    draft_strategy_from_nl,
)
from src.analysis.backtest_schema import list_custom_strategies, save_custom_strategy
from src.analysis.strategy_presets import PRESET_CATALOG, get_preset, list_preset_ids
from src.analysis.strategy_spec import StrategySpec, parse_spec_from_llm_json
from src.analysis.backtest_common import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PORTFOLIO_START,
    parse_lookback_days,
    parse_portfolio_start,
)
from src.analysis.symbol_universe import resolve_universe
from src.services.event_bus import event_bus
from src.services.stock_config import stock_config
from src.services.tab_indices import TAB_ASSISTANT
from src.utils.logger_utils import app_logger
from src.views.backtest_charts import build_compare_equity_chart
from src.views.base_view import BaseView
from src.views.components.backtest_results_panel import BacktestResultsPanel
from src.views.components.cards import SelectableMetricCard
from src.views.components.feedback import PersistentBanner, show_snackbar
from src.views.components.insights_panel import build_insights_panel
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.model_setup_bar import ModelSetupBar
from src.views.components.strategy_builder import StrategyBuilder
from src.views.strategy_constants import (
    CONTROL_TOOLTIPS,
    GUIDED_EMPTY_STATE,
    MODE_TOOLTIPS,
    STRATEGY_INFO,
)
from src.views.strategy_runners import run_compare_backtests, run_strategy_backtest
from src.analysis.backtest_engine import BacktestCancelled
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import on_ticker_field_blur, parse_ticker_filter


class StrategyBacktestView(BaseView):
    _tab_index = 5

    def __init__(self, page: ft.Page):
        super().__init__(page)
        self.scroll = None
        cfg = stock_config()
        self._running = False
        self._cancel_event = threading.Event()
        self._last_result = None
        self._selected_preset = "golden_cross"
        self._compare_specs: list[StrategySpec] = []
        self._preset_param_fields: dict[str, ft.TextField] = {}

        self.model_bar = ModelSetupBar(page)
        self.backtest_period_days = InputStyles.text_field(
            page, label="Backtest period (days)", value=str(DEFAULT_LOOKBACK_DAYS), width=180,
            tooltip=CONTROL_TOOLTIPS["period"],
        )
        self.portfolio_start = InputStyles.text_field(
            page, label="Starting portfolio ($)", value=str(DEFAULT_PORTFOLIO_START), width=160,
            tooltip=CONTROL_TOOLTIPS["capital"],
        )
        self.ticker_filter = InputStyles.text_field(
            page, label="Ticker filter (optional)", expand=True,
            hint_text="AAPL MSFT — leave empty for universe",
            on_blur=on_ticker_field_blur(multi=True),
            tooltip=CONTROL_TOOLTIPS["tickers"],
        )
        self.backtest_universe = InputStyles.dropdown(
            page, label="Universe (when filter empty)", width=260,
            value=cfg.default_backtest_universe or "with_history",
            options=[
                ft.dropdown.Option("focus", "Focus watchlist"),
                ft.dropdown.Option("with_history", "All with price data"),
                ft.dropdown.Option("universe", "Full active universe"),
            ],
            tooltip=CONTROL_TOOLTIPS["universe"],
        )
        self.benchmark_field = InputStyles.text_field(
            page, label="Benchmark ticker", value=cfg.market_ticker or "^GSPC", width=140,
            tooltip=CONTROL_TOOLTIPS["benchmark"],
        )
        self.apply_costs_switch = ft.Switch(label="Transaction costs", value=True, tooltip=CONTROL_TOOLTIPS["costs"])

        self.status_ring = ft.ProgressRing(width=22, height=22, stroke_width=2, visible=False)
        self.status_text = ft.Text("", size=12, expand=True)
        self.status_bar = ft.ProgressBar(value=0, visible=False)
        self._status_row = ft.Container(
            content=ft.Column(
                [
                    ft.Row([self.status_ring, self.status_text], spacing=10),
                    self.status_bar,
                ],
                spacing=6, tight=True,
            ),
            visible=False,
        )

        self._mode_segmented = ft.SegmentedButton(
            selected=["presets"],
            allow_empty_selection=False,
            allow_multiple_selection=False,
            segments=[
                ft.Segment(value="presets", label=ft.Text("Presets"), icon=ft.Icon(ft.Icons.GRID_VIEW),
                           tooltip=MODE_TOOLTIPS["presets"]),
                ft.Segment(value="builder", label=ft.Text("Builder"), icon=ft.Icon(ft.Icons.BUILD),
                           tooltip=MODE_TOOLTIPS["builder"]),
                ft.Segment(value="compare", label=ft.Text("Compare"), icon=ft.Icon(ft.Icons.COMPARE_ARROWS),
                           tooltip=MODE_TOOLTIPS["compare"]),
            ],
            on_change=self._on_mode_change,
        )

        self._preset_gallery = ft.Row(wrap=True, spacing=8)
        self._preset_params_host = ft.Column(spacing=8)
        self._preset_info = ft.Text("", size=11, color=ThemeHelper.text_muted(page))

        self.strategy_builder = StrategyBuilder(
            page, on_ai_request=self._on_ai_draft_strategy,
        )
        self._saved_strategies_dd = InputStyles.dropdown(
            page, label="Saved strategies", width=260, options=[],
        )
        self._compare_checks: dict[str, ft.Checkbox] = {}
        self._compare_host = ft.Column(spacing=8)

        self.results_panel = BacktestResultsPanel(
            page,
            on_discuss=self._discuss_with_assistant,
            on_analyze=self._analyze_results,
        )
        self.error_banner = PersistentBanner(page, visible=False, dismissible=True)
        self._empty_state = ft.Container(
            content=ft.Text(GUIDED_EMPTY_STATE, size=12, color=ThemeHelper.text_muted(page), selectable=True),
            padding=16,
            border=ft.border.all(1, ThemeHelper.border_default(page)),
            border_radius=8,
            bgcolor=ThemeHelper.surface_dim(page),
        )

        self.run_btn = ft.ElevatedButton(
            "Run Backtest", icon=ft.Icons.PLAY_ARROW, style=ButtonStyles.primary(),
            on_click=self._on_run,
        )
        self.cancel_btn = ft.OutlinedButton(
            "Cancel", icon=ft.Icons.STOP, visible=False, on_click=self._on_cancel,
        )
        self.save_strategy_btn = ft.OutlinedButton(
            "Save strategy", icon=ft.Icons.SAVE, on_click=self._on_save_strategy,
            tooltip="Save the current custom strategy to your library.",
        )

        self._presets_panel = ft.Column(
            [
                SectionHeader("Strategy presets", icon=ft.Icons.CATEGORY, page_ref=page),
                self._preset_gallery,
                self._preset_info,
                self._preset_params_host,
            ],
            spacing=8,
            tight=True,
        )
        self._builder_panel = ft.Column(
            [
                ft.Row([self._saved_strategies_dd,
                        ft.ElevatedButton("Load", on_click=self._on_load_saved)]),
                self.strategy_builder,
                self.save_strategy_btn,
            ],
            spacing=8,
            tight=True,
        )
        self._compare_panel = ft.Column(
            [SectionHeader("Select strategies to compare", page_ref=page), self._compare_host],
            spacing=8,
            tight=True,
        )

        # Size-to-content so results stay visible below the run controls.
        self._mode_host = ft.Container(content=self._presets_panel)
        self._main_stack = ft.Column(
            [
                self._empty_state,
                self._mode_host,
                ft.Row([self.run_btn, self.cancel_btn], spacing=12),
                self.error_banner,
                self.results_panel,
            ],
            spacing=10,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            tight=True,
        )

        sidebar_width = max(260, min(400, int((page.width or 1100) * 0.28)))
        self._sidebar = ft.Container(
            width=sidebar_width,
            content=ft.Column(
                [
                    SectionHeader("Setup", icon=ft.Icons.TUNE, page_ref=page),
                    self._mode_segmented,
                    self.backtest_period_days,
                    self.portfolio_start,
                    self.ticker_filter,
                    self.backtest_universe,
                    self.benchmark_field,
                    self.apply_costs_switch,
                    self._status_row,
                ],
                spacing=10,
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=ft.padding.only(right=16),
            border=ft.border.only(right=ft.BorderSide(1, ThemeHelper.border_default(page))),
        )

        self.controls = [
            ViewTitleBar("Strategy Backtests"),
            self.model_bar,
            ft.Divider(),
            ft.Row([self._sidebar, self._main_stack], expand=True, vertical_alignment=ft.CrossAxisAlignment.START),
        ]

        self._form_fields = [
            self.backtest_period_days, self.portfolio_start, self.ticker_filter, self.benchmark_field,
        ]
        self._form_dropdowns = [self.backtest_universe, self._saved_strategies_dd]
        self._build_preset_gallery()
        self._build_compare_checks()
        self._refresh_saved_strategies()

    def _build_preset_gallery(self) -> None:
        self._preset_gallery.controls.clear()
        for pid in list_preset_ids():
            meta = PRESET_CATALOG[pid]
            card = SelectableMetricCard(
                self.page_ref,
                title=meta["title"],
                value=meta.get("category", ""),
                subtitle=meta["summary"][:50] + "…" if len(meta["summary"]) > 50 else meta["summary"],
                selected=(pid == self._selected_preset),
                width=160,
                on_click=lambda e, p=pid: self._select_preset(p),
            )
            card.tooltip = meta["summary"]
            card.data = pid
            self._preset_gallery.controls.append(card)
        self._select_preset(self._selected_preset)

    def _select_preset(self, preset_id: str) -> None:
        self._selected_preset = preset_id
        for ctrl in self._preset_gallery.controls:
            if isinstance(ctrl, SelectableMetricCard):
                ctrl.set_selected(getattr(ctrl, "data", None) == preset_id)
        info = STRATEGY_INFO.get(preset_id, {})
        self._preset_info.value = info.get("summary", "")
        self._preset_params_host.controls.clear()
        self._preset_param_fields.clear()
        if preset_id == "golden_cross":
            self._add_param_field("fast", "Fast SMA", "50")
            self._add_param_field("slow", "Slow SMA", "200")
        elif preset_id == "trend_200":
            self._add_param_field("period", "SMA period", "200")
        elif preset_id == "rsi_dip":
            self._add_param_field("rsi_period", "RSI period", "14")
            self._add_param_field("buy_below", "Buy below", "30")
            self._add_param_field("sell_above", "Sell above", "50")
        elif preset_id == "macd_cross":
            self._add_param_field("fast", "Fast", "12")
            self._add_param_field("slow", "Slow", "26")
            self._add_param_field("signal", "Signal", "9")
        elif preset_id == "bollinger_reversion":
            self._add_param_field("period", "Period", "20")
            self._add_param_field("num_std", "Std dev", "2")
        elif preset_id == "market_filtered":
            self._add_param_field("stock_sma", "Stock SMA", "50")
            self._add_param_field("market_sma", "Market SMA days", "100")
        elif preset_id == "canslim":
            self._add_param_field("stop_loss_pct", "Stop loss %", "8")
            self._add_param_field("take_profit_pct", "Take profit %", "25")
        try:
            self._preset_info.update()
            self._preset_params_host.update()
        except RuntimeError:
            pass

    def _add_param_field(self, key: str, label: str, default: str) -> None:
        field = InputStyles.text_field(self.page_ref, label=label, value=default, width=120)
        self._preset_param_fields[key] = field
        self._preset_params_host.controls.append(field)

    def _build_compare_checks(self) -> None:
        self._compare_host.controls.clear()
        self._compare_checks.clear()
        options = [(pid, PRESET_CATALOG[pid]["title"]) for pid in list_preset_ids()]
        for pid, title in options:
            cb = ft.Checkbox(label=title, value=pid in ("golden_cross", "buy_hold", "macd_cross"))
            self._compare_checks[pid] = cb
            self._compare_host.controls.append(cb)

    def _current_preset_spec(self) -> StrategySpec:
        pid = self._selected_preset
        kwargs: dict = {}
        for key, field in self._preset_param_fields.items():
            try:
                val = float(field.value)
                if key in ("stop_loss_pct", "take_profit_pct"):
                    kwargs[key] = val / 100.0 if val > 1 else val
                elif key in ("fast", "slow", "signal", "period", "rsi_period", "stock_sma", "market_sma"):
                    kwargs[key.replace("_period", "_period").replace("rsi_period", "rsi_period")] = int(val)
                else:
                    kwargs[key] = val
            except ValueError:
                pass
        if pid == "golden_cross":
            spec = get_preset(pid, fast=int(kwargs.get("fast", 50)), slow=int(kwargs.get("slow", 200)))
        elif pid == "trend_200":
            spec = get_preset(pid, period=int(kwargs.get("period", 200)))
        elif pid == "rsi_dip":
            spec = get_preset(
                pid,
                rsi_period=int(kwargs.get("rsi_period", 14)),
                buy_below=float(kwargs.get("buy_below", 30)),
                sell_above=float(kwargs.get("sell_above", 50)),
            )
        elif pid == "macd_cross":
            spec = get_preset(
                pid,
                fast=int(kwargs.get("fast", 12)),
                slow=int(kwargs.get("slow", 26)),
                signal=int(kwargs.get("signal", 9)),
            )
        elif pid == "bollinger_reversion":
            spec = get_preset(pid, period=int(kwargs.get("period", 20)), num_std=float(kwargs.get("num_std", 2)))
        elif pid == "market_filtered":
            spec = get_preset(
                pid,
                stock_sma=int(kwargs.get("stock_sma", 50)),
                market_sma=int(kwargs.get("market_sma", 100)),
                market_ticker=self.benchmark_field.value,
            )
        elif pid == "canslim":
            spec = get_preset(
                pid,
                stop_loss_pct=float(kwargs.get("stop_loss_pct", 0.08)),
                take_profit_pct=float(kwargs.get("take_profit_pct", 0.25)),
            )
        else:
            spec = get_preset(pid)
        spec.apply_costs = self.apply_costs_switch.value
        spec.market_ticker = self.benchmark_field.value
        return spec

    def _active_mode(self) -> str:
        sel = list(self._mode_segmented.selected or ["presets"])
        return sel[0]

    def _on_mode_change(self, e) -> None:
        mode = self._active_mode()
        panels = {"presets": self._presets_panel, "builder": self._builder_panel, "compare": self._compare_panel}
        self._mode_host.content = panels.get(mode, self._presets_panel)
        try:
            self._mode_host.update()
        except RuntimeError:
            pass

    def _resolve_tickers(self) -> list[str] | None:
        cfg = stock_config()
        explicit = parse_ticker_filter(self.ticker_filter.value)
        if explicit:
            return explicit
        scope = self.backtest_universe.value or cfg.default_backtest_universe
        return resolve_universe(cfg.db_path, scope) or None

    def _clear_error(self) -> None:
        self.error_banner.hide()

    def _report_error(
        self,
        message: str,
        *,
        context: str = "backtest",
        mirror_in_panel: bool = True,
    ) -> None:
        """Log, keep a dismissible banner, and optionally mirror into the results panel."""
        text = (message or "Unknown error").strip()
        if len(text) > 600:
            text = text[:597] + "…"
        try:
            app_logger.log(
                "BACKTEST",
                f"Strategy backtest error ({context}).",
                level="ERROR",
                error=text,
            )
        except Exception:
            pass
        print(f"[BACKTEST] {context}: {text}")
        self.error_banner.show(text, severity="error")
        if mirror_in_panel:
            try:
                self.results_panel.show_error(text)
            except Exception:
                pass
        show_snackbar(
            self.page_ref,
            text[:180],
            severity="error",
            duration_ms=8000,
        )

    def _set_status(self, message: str, *, pct: float | None = None, busy: bool = False) -> None:
        self._status_row.visible = busy or bool(message)
        self.status_ring.visible = busy
        self.status_text.value = message
        if pct is not None:
            self.status_bar.visible = True
            self.status_bar.value = max(0.0, min(1.0, pct))
        else:
            self.status_bar.visible = busy
        self.cancel_btn.visible = busy

        def _ui():
            for c in (self._status_row, self.status_ring, self.status_text, self.status_bar, self.cancel_btn):
                try:
                    c.update()
                except RuntimeError:
                    pass

        self._safe_update(_ui, label="backtest_status")

    def _on_cancel(self, e) -> None:
        if not self._running:
            return
        self._cancel_event.set()
        self.cancel_btn.disabled = True
        self._set_status("Cancelling…", busy=True)
        try:
            self.cancel_btn.update()
        except RuntimeError:
            pass

    def _on_cancelled(self) -> None:
        show_snackbar(self.page_ref, "Backtest cancelled.", severity="warning", duration_ms=4000)

    def _on_run(self, e) -> None:
        if self._running:
            show_snackbar(self.page_ref, "A backtest is already running.", severity="warning")
            return
        self._running = True
        self._cancel_event.clear()
        self.run_btn.disabled = True
        self.cancel_btn.disabled = False
        self._empty_state.visible = False
        self._clear_error()
        try:
            self.run_btn.update()
            self.cancel_btn.update()
            self._empty_state.update()
        except RuntimeError:
            pass
        cfg = stock_config()
        tickers = self._resolve_tickers()
        lookback = parse_lookback_days(self.backtest_period_days.value)
        capital = parse_portfolio_start(self.portfolio_start.value)
        mode = self._active_mode()
        self._set_status("Starting backtest…", busy=True)

        def _work():
            try:
                if mode == "compare":
                    specs = [
                        get_preset(pid) for pid, cb in self._compare_checks.items() if cb.value
                    ][:5]
                    if len(specs) < 2:
                        raise ValueError("Select at least 2 strategies to compare.")
                    for s in specs:
                        s.apply_costs = self.apply_costs_switch.value
                    cmp_res = run_compare_backtests(
                        specs,
                        db_path=cfg.db_path,
                        tickers=tickers,
                        lookback_days=lookback,
                        initial_capital=capital,
                        benchmark_ticker=self.benchmark_field.value,
                        use_parallel=cfg.use_parallel,
                        progress=lambda m, p, b: self._set_status(m, pct=p, busy=b),
                        cancel_event=self._cancel_event,
                    )
                    if cmp_res.cancelled and not cmp_res.results:
                        raise BacktestCancelled("Cancelled by user.")
                    if self._cancel_event.is_set() and not cmp_res.results:
                        raise BacktestCancelled("Cancelled by user.")
                    compare_results = list(cmp_res.results)
                    if self._cancel_event.is_set():
                        self._safe_update_critical(self._on_cancelled)
                        if compare_results:
                            self._safe_update_critical(
                                lambda r=compare_results, c=capital: self._show_compare(r, c)
                            )
                        return
                    self._safe_update_critical(
                        lambda r=compare_results, c=capital: self._show_compare(r, c)
                    )
                else:
                    spec = (
                        self._current_preset_spec()
                        if mode == "presets"
                        else self.strategy_builder.build_spec()
                    )
                    spec.apply_costs = self.apply_costs_switch.value
                    result = run_strategy_backtest(
                        spec,
                        db_path=cfg.db_path,
                        tickers=tickers,
                        lookback_days=lookback,
                        initial_capital=capital,
                        benchmark_ticker=self.benchmark_field.value,
                        use_parallel=cfg.use_parallel,
                        progress=lambda m, p, b: self._set_status(m, pct=p, busy=b),
                        cancel_event=self._cancel_event,
                    )
                    if self._cancel_event.is_set():
                        raise BacktestCancelled("Cancelled by user.")
                    if result is None:
                        raise ValueError("No data available for backtest.")
                    self._last_result = result
                    self._safe_update_critical(
                        lambda r=result, c=capital: self._show_single(r, c)
                    )
            except BacktestCancelled:
                self._safe_update_critical(self._on_cancelled)
            except Exception as ex:
                err_msg = str(ex)
                self._safe_update_critical(
                    lambda msg=err_msg: self._report_error(msg, context=mode)
                )
            finally:
                self._safe_update_critical(self._finish_run)

        threading.Thread(target=_work, daemon=True).start()

    def _finish_run(self) -> None:
        """Re-enable controls and clear busy status after a backtest completes."""
        self._running = False
        self.run_btn.disabled = False
        self.cancel_btn.disabled = False
        self._status_row.visible = False
        self.status_ring.visible = False
        self.status_text.value = ""
        self.status_bar.visible = False
        self.status_bar.value = 0
        self.cancel_btn.visible = False
        # Explicit control updates — Flet does not always auto-patch disabled
        # state from a background-scheduled finish callback until a full redraw.
        for c in (
            self.run_btn,
            self.cancel_btn,
            self._status_row,
            self.status_ring,
            self.status_text,
            self.status_bar,
        ):
            try:
                c.update()
            except RuntimeError:
                pass

    def _show_single(self, result, capital: float) -> None:
        self._clear_error()
        self._empty_state.visible = False
        try:
            self.results_panel.show_result(result, initial_capital=capital)
            show_snackbar(self.page_ref, result.summary[:120], severity="info")
        except Exception as ex:
            self._report_error(f"Failed to display backtest results: {ex}", context="display")
            return
        try:
            self._empty_state.update()
            self._main_stack.update()
        except RuntimeError:
            pass

    def _show_compare(self, results, capital: float) -> None:
        if not results:
            self._report_error(
                "Comparison produced no results. Check that selected strategies "
                "have price history for the chosen tickers and period.",
                context="compare",
            )
            return
        self._clear_error()
        self._empty_state.visible = False
        chart: ft.Control | None = None
        chart_error: str | None = None
        try:
            curves = [(r.spec.name, r.equity_dates, r.equity_values) for r in results]
            chart = build_compare_equity_chart(self.page_ref, curves)
        except Exception as ex:
            chart_error = str(ex)
            chart = ft.Text(
                f"Equity overlay unavailable: {chart_error}",
                color=ThemeHelper.text_error(self.page_ref),
                size=12,
                selectable=True,
            )
        try:
            cols = ["Strategy", "Return %", "Sharpe", "Max DD %", "Trades"]
            best_ret = max(r.metrics.total_return_pct for r in results)
            columns = [ft.DataColumn(ft.Text(c)) for c in cols]
            rows = []
            for r in results:
                m = r.metrics
                cells = [
                    ft.DataCell(ft.Text(r.spec.name)),
                    ft.DataCell(ft.Text(
                        f"{m.total_return_pct:.1f}",
                        color=ThemeHelper.chart_named(self.page_ref, "gain")
                        if m.total_return_pct == best_ret else None,
                        weight=ft.FontWeight.BOLD if m.total_return_pct == best_ret else None,
                    )),
                    ft.DataCell(ft.Text(f"{m.sharpe:.2f}")),
                    ft.DataCell(ft.Text(f"{m.max_drawdown_pct:.1f}")),
                    ft.DataCell(ft.Text(str(m.trade_count))),
                ]
                rows.append(ft.DataRow(cells=cells))
            self.results_panel.show_compare(
                results,
                compare_chart=chart,
                compare_columns=columns,
                compare_rows=rows,
            )
            self._last_result = results[0]
            if chart_error:
                self._report_error(
                    f"Comparison metrics are shown, but the equity overlay failed: {chart_error}",
                    context="compare_chart",
                    mirror_in_panel=False,
                )
            else:
                show_snackbar(
                    self.page_ref,
                    f"Comparison complete — {len(results)} strategies.",
                    severity="info",
                )
            try:
                self._empty_state.update()
                self._main_stack.update()
            except RuntimeError:
                pass
        except Exception as ex:
            self._report_error(
                f"Failed to display comparison results: {ex}",
                context="compare_display",
            )

    def _analyze_results(self) -> None:
        if not self._last_result:
            show_snackbar(self.page_ref, "Run a backtest first.", severity="warning")
            return

        def _work():
            suggestions = analyze_backtest_results(self._last_result, db_path=stock_config().db_path)
            panel = build_insights_panel(self.page_ref, suggestions)

            def _ui():
                self.results_panel.set_ai_insights([panel])
                try:
                    self.results_panel.update()
                except RuntimeError:
                    pass

            self._safe_update(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _discuss_with_assistant(self) -> None:
        if not self._last_result:
            show_snackbar(self.page_ref, "Run a backtest first.", severity="warning")
            return
        prefill = backtest_assistant_prefill(self._last_result)
        event_bus.emit("navigate_tab", tab_index=TAB_ASSISTANT)
        event_bus.emit("navigate_assistant", prefill_message=prefill)

    def _on_ai_draft_strategy(self, description: str) -> None:
        def _work():
            data, err = draft_strategy_from_nl(description)
            if err:
                self._safe_update(lambda: show_snackbar(self.page_ref, err, severity="warning"))
                return
            spec, parse_err = parse_spec_from_llm_json(json.dumps(data) if data else "")
            if parse_err or spec is None:
                self._safe_update(
                    lambda: show_snackbar(self.page_ref, parse_err or "Parse failed", severity="error")
                )
                return

            def _ui():
                self.strategy_builder.load_spec(spec)
                show_snackbar(self.page_ref, "AI draft loaded — review rules before running.", severity="info")

            self._safe_update(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _on_save_strategy(self, e) -> None:
        spec = self.strategy_builder.build_spec()
        try:
            save_custom_strategy(stock_config().db_path, spec)
            self._refresh_saved_strategies()
            show_snackbar(self.page_ref, f"Saved '{spec.name}'.", severity="info")
        except Exception as ex:
            show_snackbar(self.page_ref, str(ex), severity="error")

    def _refresh_saved_strategies(self) -> None:
        saved = list_custom_strategies(stock_config().db_path)
        self._saved_strategies_dd.options = [
            ft.dropdown.Option(s.name, s.name) for s in saved
        ]
        try:
            self._saved_strategies_dd.update()
        except RuntimeError:
            pass

    def _on_load_saved(self, e) -> None:
        name = self._saved_strategies_dd.value
        if not name:
            return
        for s in list_custom_strategies(stock_config().db_path):
            if s.name == name:
                self.strategy_builder.load_spec(s.spec)
                show_snackbar(self.page_ref, f"Loaded '{name}'.", severity="info")
                break

    def refresh_data(self) -> None:
        self._refresh_saved_strategies()
        self.model_bar.refresh()

    def refresh_theme(self) -> None:
        InputStyles.refresh_fields(self.page_ref, self._form_fields, self._form_dropdowns)
        self._preset_info.color = ThemeHelper.text_muted(self.page_ref)
        try:
            self.update()
        except RuntimeError:
            pass
