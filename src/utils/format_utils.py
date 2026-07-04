"""Shared formatting helpers for UI and reports."""
from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import pandas as pd


def format_timestamp(ts_val: Any) -> str:
    """Format a timestamp for display. Handles None, datetime, or string."""
    if not ts_val:
        return "N/A"
    if hasattr(ts_val, "strftime"):
        return ts_val.strftime("%Y-%m-%d %H:%M")
    return str(ts_val)


def natural_sort_key(value: str) -> tuple:
    """
    Return a sort key tuple that orders strings like a human (e.g. AAPL10 after
    AAPL2 instead of before it). Useful for ticker lists, file names, etc.
    """
    text = (value or "").lower()
    parts = re.split(r"(\d+)", text)
    key: list[tuple] = []
    for part in parts:
        if part.isdigit():
            key.append((1, int(part)))
        else:
            key.append((0, part))
    return tuple(key)


def shorten(value: str, max_len: int = 24) -> str:
    """Truncate long strings with an ellipsis. Safe with None."""
    text = (value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def format_currency(value: float | int | None, *, symbol: str = "$", decimals: int = 2) -> str:
    """Format a number as currency. Returns '—' for None."""
    if value is None:
        return "—"
    try:
        return f"{symbol}{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: float | int | None, *, decimals: int = 2, signed: bool = False) -> str:
    """Format a number as a percentage. Returns '—' for None."""
    if value is None:
        return "—"
    try:
        v = float(value)
        sign = "+" if signed and v > 0 else ""
        return f"{sign}{v:.{decimals}f}%"
    except (TypeError, ValueError):
        return str(value)


def format_large_number(value: float | int | None, *, decimals: int = 2) -> str:
    """Format large numbers with K/M/B/T suffixes (market cap, volume)."""
    if value is None:
        return "—"
    try:
        v = abs(float(value))
        sign = "-" if float(value) < 0 else ""
        if v >= 1_000_000_000_000:
            return f"{sign}{v / 1_000_000_000_000:.{decimals}f}T"
        if v >= 1_000_000_000:
            return f"{sign}{v / 1_000_000_000:.{decimals}f}B"
        if v >= 1_000_000:
            return f"{sign}{v / 1_000_000:.{decimals}f}M"
        if v >= 1_000:
            return f"{sign}{v / 1_000:.{decimals}f}K"
        return f"{sign}{v:,.0f}"
    except (TypeError, ValueError):
        return str(value)


def format_volume(value: float | int | None) -> str:
    """Format share volume for display."""
    if value is None:
        return "—"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def format_range(low: float | None, high: float | None, *, currency: bool = True) -> str:
    """Format a low–high range."""
    if low is None and high is None:
        return "—"
    if currency:
        lo = format_currency(low) if low is not None else "—"
        hi = format_currency(high) if high is not None else "—"
    else:
        lo = str(low) if low is not None else "—"
        hi = str(high) if high is not None else "—"
    return f"{lo} – {hi}"


def format_change(value: float | None, pct: float | None) -> str:
    """Format dollar and percent change together."""
    if value is None and pct is None:
        return "—"
    parts = []
    if value is not None:
        sign = "+" if value > 0 else ""
        parts.append(f"{sign}{format_currency(value)}")
    if pct is not None:
        parts.append(format_percent(pct, signed=True))
    return " · ".join(parts) if parts else "—"


def format_number(
    val: Any,
    *,
    decimals: int = 2,
    null: str = "—",
) -> str:
    """Format a numeric table/chart cell value."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return null
    try:
        f = float(val)
        if decimals == 0:
            return str(int(round(f)))
        if abs(f) < 100:
            return f"{f:.{max(decimals, 4)}f}" if decimals >= 4 else f"{f:.{decimals}f}"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


_INTERVAL_DAYS: dict[str, int | None] = {
    "week": 7,
    "1w": 7,
    "month": 30,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "ytd": None,
}


def slice_price_df(
    df: pd.DataFrame | None,
    interval_key: str,
    *,
    min_rows: int = 1,
) -> pd.DataFrame | None:
    """Slice a date-indexed price DataFrame to a lookback window."""
    if df is None or df.empty:
        return None
    latest = df.index.max()
    if interval_key == "ytd":
        start = pd.Timestamp(year=latest.year, month=1, day=1)
    else:
        days = _INTERVAL_DAYS.get(interval_key, 90) or 90
        start = latest - timedelta(days=days)
    sliced = df[df.index >= start]
    if sliced.empty:
        return df.tail(max(min_rows, 1))
    return sliced
