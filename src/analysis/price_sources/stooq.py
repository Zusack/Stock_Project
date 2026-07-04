"""Stooq free daily OHLCV CSV (no API key). Used for initial backfill only."""

from __future__ import annotations

import io

import pandas as pd

_STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"


def _to_stooq_symbol(ticker: str) -> str | None:
    t = ticker.strip().upper()
    if not t or t.startswith("^"):
        return None
    if "." in t:
        return t.lower()
    return f"{t}.us"


def fetch_stooq_daily_history(ticker: str) -> pd.DataFrame:
    """
    Download daily bars from Stooq. Returns empty DataFrame on failure.
    Columns aligned with Yahoo history for ingest upsert.
    """
    sym = _to_stooq_symbol(ticker)
    if not sym:
        return pd.DataFrame()

    url = _STOOQ_DAILY_URL.format(symbol=sym)
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=45) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return pd.DataFrame()

    if not raw.strip() or "No data" in raw[:200]:
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception:
        return pd.DataFrame()

    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if df.empty:
        return pd.DataFrame()

    df = df.set_index("Date").sort_index()
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    for col in ("Dividends", "Stock Splits", "Capital Gains"):
        if col not in df.columns:
            df[col] = 0.0
    df.index.name = "Date"
    return df
