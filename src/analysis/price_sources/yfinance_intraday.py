"""Yahoo Finance intraday bars via yfinance (free, no API key).

Yahoo limits: 1-minute bars cover roughly the last 7 days; coarser intraday
intervals are available up to ~60 days.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.analysis.intraday_common import interval_to_yfinance, utc_timestamp_str

# Yahoo's hard caps on how far back intraday data is available per interval.
_MAX_DAYS_BY_INTERVAL = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "1h": 730,
}


def fetch_yfinance_intraday(
    ticker: str,
    *,
    days: int = 7,
    interval: str = "1Min",
    extended_hours: bool = False,
) -> pd.DataFrame:
    """Download intraday OHLCV bars from Yahoo. Empty DataFrame on failure."""
    sym = str(ticker).strip().upper()
    if not sym or sym.startswith("^"):
        return pd.DataFrame()

    yf_interval = interval_to_yfinance(interval)
    max_days = _MAX_DAYS_BY_INTERVAL.get(yf_interval, 60)
    period_days = max(1, min(int(days), max_days))

    try:
        stock = yf.Ticker(sym)
        hist = stock.history(
            period=f"{period_days}d",
            interval=yf_interval,
            prepost=bool(extended_hours),
            auto_adjust=False,
            actions=False,
        )
    except Exception:
        return pd.DataFrame()

    if hist is None or hist.empty:
        return pd.DataFrame()

    hist = hist.reset_index()
    ts_col = None
    for cand in ("Datetime", "Date", "index"):
        if cand in hist.columns:
            ts_col = cand
            break
    if ts_col is None:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in hist.iterrows():
        ts = row[ts_col]
        if pd.isnull(ts):
            continue
        ts = pd.to_datetime(ts, utc=True).to_pydatetime()
        try:
            rows.append(
                {
                    "Ticker": sym,
                    "Timestamp": utc_timestamp_str(ts),
                    "Interval": interval,
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": float(row["Close"]),
                    "Volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else 0,
                    "Vwap": None,
                    "Trade_Count": None,
                }
            )
        except (KeyError, ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Timestamp").reset_index(drop=True)
