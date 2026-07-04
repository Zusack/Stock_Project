"""Yahoo Finance fundamentals (short history, no extra API key)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.analysis.fundamentals_sources.types import FundamentalRow

_METRICS = ("Net Income", "Basic EPS", "Total Revenue")


def fetch_yahoo_fundamentals(ticker: str) -> list[FundamentalRow]:
    ticker = ticker.strip().upper()
    rows: list[FundamentalRow] = []
    stock = yf.Ticker(ticker)

    try:
        q_fin = stock.quarterly_financials
        if q_fin is not None and not q_fin.empty:
            for date, row in q_fin.T.items():
                date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]
                for metric in _METRICS:
                    if metric in row.index and pd.notna(row[metric]):
                        rows.append((ticker, date_str, metric, float(row[metric]), "Quarterly"))
    except Exception:
        pass

    try:
        a_fin = stock.financials
        if a_fin is not None and not a_fin.empty:
            for date, row in a_fin.T.items():
                date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]
                for metric in _METRICS:
                    if metric in row.index and pd.notna(row[metric]):
                        rows.append((ticker, date_str, metric, float(row[metric]), "Annual"))
    except Exception:
        pass

    return rows
