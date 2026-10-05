"""Unit tests for Live tab chart-symbol helpers and session day stats."""

from __future__ import annotations

from src.services.live_stream_service import LiveBarSnapshot
from src.views.live_view import resolve_chart_symbols, session_day_stats


def _bar(ticker: str, ts: str, open_: float, close: float) -> LiveBarSnapshot:
    return LiveBarSnapshot(
        ticker=ticker,
        timestamp=ts,
        interval="1Min",
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        volume=1000,
    )


def test_resolve_chart_symbols_includes_also_stream():
    assert resolve_chart_symbols("aapl", ["msft", "NVDA", "AAPL"]) == [
        "AAPL",
        "MSFT",
        "NVDA",
    ]


def test_resolve_chart_symbols_skips_indices_and_blanks():
    assert resolve_chart_symbols("AAPL", ["", "^GSPC", "msft"]) == ["AAPL", "MSFT"]
    assert resolve_chart_symbols("", ["msft"]) == ["MSFT"]
    assert resolve_chart_symbols("", []) == []


def test_session_day_stats_from_open():
    bars = [
        _bar("AAPL", "2026-07-16T13:30:00Z", 100.0, 101.0),
        _bar("AAPL", "2026-07-16T13:31:00Z", 101.0, 102.5),
    ]
    stats = session_day_stats("aapl", bars)
    assert stats is not None
    assert stats.ticker == "AAPL"
    assert stats.session_open == 100.0
    assert stats.last == 102.5
    assert abs(stats.change - 2.5) < 1e-9
    assert abs(stats.change_pct - 2.5) < 1e-9
    assert stats.bar_count == 2


def test_session_day_stats_empty_or_zero_open():
    assert session_day_stats("AAPL", []) is None
    assert (
        session_day_stats(
            "AAPL",
            [_bar("AAPL", "2026-07-16T13:30:00Z", 0.0, 1.0)],
        )
        is None
    )
