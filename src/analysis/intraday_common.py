"""Shared helpers for intraday bars (interval parsing, row/timestamp formatting)."""

from __future__ import annotations

from datetime import datetime, timezone

INTRADAY_COLUMNS = [
    "Ticker",
    "Timestamp",
    "Interval",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Vwap",
    "Trade_Count",
]

# Map app interval labels to (yfinance interval, window seconds).
_INTERVAL_SECONDS: dict[str, int] = {
    "1Min": 60,
    "5Min": 300,
    "15Min": 900,
    "1Hour": 3600,
}

_YFINANCE_INTERVAL: dict[str, str] = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "1Hour": "1h",
}


def interval_to_seconds(interval: str) -> int:
    return _INTERVAL_SECONDS.get((interval or "1Min").strip(), 60)


def interval_to_yfinance(interval: str) -> str:
    return _YFINANCE_INTERVAL.get((interval or "1Min").strip(), "1m")


def utc_timestamp_str(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
