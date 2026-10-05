"""Smoke tests for Strategy Backtests view."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.views.strategy_backtest_view import StrategyBacktestView


@pytest.fixture
def backtest_view():
    page = MagicMock()
    view = StrategyBacktestView.__new__(StrategyBacktestView)
    view.page_ref = page
    view._running = False
    view._selected_preset = "golden_cross"
    view._last_result = None
    view._cancel_event = MagicMock()
    view._cancel_event.is_set.return_value = False
    return view


def test_active_mode_defaults_presets(backtest_view):
    backtest_view._mode_segmented = MagicMock(selected=["presets"])
    assert backtest_view._active_mode() == "presets"


def test_current_preset_spec_golden_cross(backtest_view):
    backtest_view._preset_param_fields = {}
    backtest_view.apply_costs_switch = MagicMock(value=True)
    backtest_view.benchmark_field = MagicMock(value="^GSPC")
    spec = backtest_view._current_preset_spec()
    assert spec.preset_id == "golden_cross"
    assert spec.engine == "unified"


def test_cancel_sets_event(backtest_view):
    backtest_view._running = True
    backtest_view._cancel_event = MagicMock()
    backtest_view.cancel_btn = MagicMock()
    backtest_view._status_row = MagicMock()
    backtest_view.status_ring = MagicMock()
    backtest_view.status_text = MagicMock()
    backtest_view.status_bar = MagicMock()
    backtest_view._set_status = MagicMock()
    backtest_view._on_cancel(MagicMock())
    backtest_view._cancel_event.set.assert_called_once()
    assert backtest_view.cancel_btn.disabled is True


def test_finish_run_reenables_button_and_updates(backtest_view):
    backtest_view._running = True
    backtest_view.run_btn = MagicMock(disabled=True)
    backtest_view.cancel_btn = MagicMock(visible=True, disabled=True)
    backtest_view._status_row = MagicMock(visible=True)
    backtest_view.status_ring = MagicMock(visible=True)
    backtest_view.status_text = MagicMock(value="Running…")
    backtest_view.status_bar = MagicMock(visible=True, value=0.5)

    backtest_view._finish_run()

    assert backtest_view._running is False
    assert backtest_view.run_btn.disabled is False
    assert backtest_view.cancel_btn.visible is False
    assert backtest_view._status_row.visible is False
    backtest_view.run_btn.update.assert_called()
    backtest_view.cancel_btn.update.assert_called()
    backtest_view._status_row.update.assert_called()


def test_run_exception_captures_message_for_ui(backtest_view, monkeypatch):
    """Exception messages must be bound before scheduling UI (Python clears `ex`)."""
    from src.views import strategy_backtest_view as mod

    captured: list = []

    def _critical(func):
        captured.append(func)

    backtest_view._running = False
    backtest_view.run_btn = MagicMock()
    backtest_view.cancel_btn = MagicMock()
    backtest_view._empty_state = MagicMock()
    backtest_view.error_banner = MagicMock()
    backtest_view.results_panel = MagicMock()
    backtest_view._cancel_event = MagicMock()
    backtest_view.backtest_period_days = MagicMock(value="252")
    backtest_view.portfolio_start = MagicMock(value="10000")
    backtest_view._mode_segmented = MagicMock(selected=["compare"])
    backtest_view._compare_checks = {"a": MagicMock(value=True)}
    backtest_view._safe_update_critical = _critical
    backtest_view._set_status = MagicMock()
    backtest_view._resolve_tickers = MagicMock(return_value=["AAPL"])
    backtest_view._clear_error = MagicMock()
    reported: list[str] = []
    backtest_view._report_error = lambda msg, context="backtest": reported.append(msg)

    monkeypatch.setattr(
        mod, "stock_config", lambda: MagicMock(db_path=":memory:", use_parallel=False)
    )

    def _start(target=None, daemon=None):
        target()
        return MagicMock()

    monkeypatch.setattr(mod.threading, "Thread", _start)
    backtest_view._on_run(MagicMock())

    err_cbs = [f for f in captured if getattr(f, "__defaults__", None)]
    assert err_cbs, "expected error callback with bound message"
    err_cbs[0]()  # must not raise UnboundLocalError / free-variable error
    assert reported, "expected _report_error to be called"
    assert "preset" in reported[0].lower() or "compare" in reported[0].lower() or "select" in reported[0].lower()


def test_show_compare_populates_kpis_and_refreshes(backtest_view, monkeypatch):
    from src.analysis.backtest_engine import BacktestResult
    from src.analysis.backtest_metrics import MetricsReport
    from src.analysis.strategy_spec import StrategySpec
    from src.views.components.backtest_results_panel import BacktestResultsPanel

    panel = BacktestResultsPanel.__new__(BacktestResultsPanel)
    panel.page_ref = MagicMock()
    panel.summary_text = MagicMock()
    panel._kpi_row = MagicMock()
    panel._kpi_row.controls = []
    panel.equity_slot = MagicMock()
    panel.drawdown_slot = MagicMock()
    panel.performance_slot = MagicMock()
    panel.monthly_table = MagicMock()
    panel.trades_table = MagicMock()
    panel._trades_page_label = MagicMock()
    panel._table_section_title = MagicMock()
    panel._table_section_icon = MagicMock()
    panel._ai_slot = MagicMock()
    panel.refresh_ui = MagicMock()
    panel._render_trades_page = MagicMock()

    fake_card = MagicMock()
    fake_card._value_text = MagicMock()
    monkeypatch.setattr(
        "src.views.components.backtest_results_panel.SelectableMetricCard",
        lambda *a, **k: fake_card,
    )

    results = []
    for name, ret in (("Alpha", 12.0), ("Beta", 5.0)):
        results.append(
            BacktestResult(
                spec=StrategySpec(name=name),
                metrics=MetricsReport(
                    total_return_pct=ret, sharpe=1.1, max_drawdown_pct=8.0, trade_count=4
                ),
                equity_dates=["2020-01-01", "2020-06-01"],
                equity_values=[100.0, 100.0 + ret],
                summary=f"{name} summary",
            )
        )

    BacktestResultsPanel.show_compare(
        panel,
        results,
        compare_chart=MagicMock(),
        compare_columns=[MagicMock()],
        compare_rows=[MagicMock()],
    )

    assert len(panel._kpi_row.controls) == 2
    assert panel.equity_slot.content is not None
    assert panel._table_section_title.value == "Strategy comparison"
    panel.refresh_ui.assert_called_once()
