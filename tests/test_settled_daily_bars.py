"""Tests for in-progress daily bar filtering."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from src.analysis.market_calendar import filter_settled_daily_bars, last_completed_trading_day


def test_filter_drops_today_before_utc_close() -> None:
    # Monday 18:00 UTC -> last completed should be prior Friday
    as_of = datetime(2026, 5, 18, 18, 0, tzinfo=timezone.utc)  # Monday
    cutoff = last_completed_trading_day(as_of)
    assert cutoff == date(2026, 5, 15)  # Friday

    idx = pd.to_datetime(["2026-05-15", "2026-05-18"])
    hist = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    filtered, dropped = filter_settled_daily_bars(hist, as_of=as_of)
    assert dropped == 1
    assert len(filtered) == 1
    assert filtered.index[0].date() == date(2026, 5, 15)


def test_filter_handles_utc_aware_yfinance_index() -> None:
    """Regression: naive Timestamp vs datetime64[ns, UTC] comparison."""
    as_of = datetime(2026, 5, 18, 18, 0, tzinfo=timezone.utc)
    idx = pd.to_datetime(["2026-05-15", "2026-05-18"], utc=True)
    hist = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    filtered, dropped = filter_settled_daily_bars(hist, as_of=as_of)
    assert dropped == 1
    assert len(filtered) == 1


def test_filter_keeps_today_after_utc_close() -> None:
    as_of = datetime(2026, 5, 18, 22, 0, tzinfo=timezone.utc)  # Monday after 21:00 UTC
    cutoff = last_completed_trading_day(as_of)
    assert cutoff == date(2026, 5, 18)

    idx = pd.to_datetime(["2026-05-15", "2026-05-18"])
    hist = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    filtered, dropped = filter_settled_daily_bars(hist, as_of=as_of)
    assert dropped == 0
    assert len(filtered) == 2
