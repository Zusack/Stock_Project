"""Tests for Stock Detail chart vs full-load behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.views.single_ticker_view import SingleTickerView


@pytest.fixture
def stock_detail_view():
    page = MagicMock()
    view = SingleTickerView.__new__(SingleTickerView)
    view.page_ref = page
    view._running = False
    view._chart_refreshing = False
    view._current_ticker = "AAPL"
    view._latest_ohlcv = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [1_000_000, 1_100_000],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    view.show_benchmark_switch = MagicMock(value=False)
    view.benchmark_dropdown = MagicMock(value="^GSPC")
    view.range_selector = MagicMock(selected=["week"])
    view.chart_slot = MagicMock()
    view._session_by_ticker = {}
    return view


def test_chart_options_changed_skips_full_load(stock_detail_view):
    with (
        patch.object(stock_detail_view, "_on_load") as mock_load,
        patch.object(stock_detail_view, "_refresh_chart_only") as mock_chart,
    ):
        stock_detail_view._on_chart_options_changed()
        mock_chart.assert_called_once()
        mock_load.assert_not_called()


def test_chart_options_changed_ignored_without_ticker(stock_detail_view):
    stock_detail_view._current_ticker = ""
    with patch.object(stock_detail_view, "_refresh_chart_only") as mock_chart:
        stock_detail_view._on_chart_options_changed()
        mock_chart.assert_not_called()


def test_chart_options_changed_ignored_during_full_load(stock_detail_view):
    stock_detail_view._running = True
    with patch.object(stock_detail_view, "_refresh_chart_only") as mock_chart:
        stock_detail_view._on_chart_options_changed()
        mock_chart.assert_not_called()


def test_refresh_chart_only_does_not_invoke_advisor(stock_detail_view):
    stock_detail_view.show_benchmark_switch.value = True

    with (
        patch("src.views.single_ticker_view.get_signal_advisor") as mock_advisor,
        patch("src.views.single_ticker_view.analyze_canslim") as mock_canslim,
        patch.object(stock_detail_view, "_selected_range", return_value="week"),
        patch.object(stock_detail_view, "_build_chart_data", return_value=MagicMock()) as mock_build,
        patch.object(stock_detail_view, "_safe_update", side_effect=lambda fn: fn()),
    ):
        stock_detail_view._refresh_chart_only()

    mock_build.assert_called_once()
    mock_advisor.assert_not_called()
    mock_canslim.assert_not_called()


def test_refresh_data_restores_session_without_reload(stock_detail_view):
    from src.analysis.canslim import CanslimResult

    stock_detail_view._current_ticker = "AAPL"
    stock_detail_view._session_by_ticker["AAPL"] = MagicMock(
        ticker="AAPL",
        quote=None,
        profile=None,
        headlines=[],
        sent=None,
        tags=[],
        catalysts=[],
        ohlcv=stock_detail_view._latest_ohlcv,
        fund_rows=[],
        canslim_result=CanslimResult(ticker="AAPL", score=4, lines=[]),
        suggestions=[],
    )

    with (
        patch.object(stock_detail_view, "_restore_session") as mock_restore,
        patch.object(stock_detail_view, "_on_load") as mock_load,
    ):
        stock_detail_view.refresh_data()

    mock_restore.assert_called_once_with("AAPL")
    mock_load.assert_not_called()
