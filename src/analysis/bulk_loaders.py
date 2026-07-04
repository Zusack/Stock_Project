"""Batch SQLite loaders to avoid per-ticker query storms."""

from __future__ import annotations

import sqlite3
import pandas as pd

from src.analysis.db import db_connection, load_entire_database, load_market_data


def load_sector_map(db_path: str) -> dict[str, str]:
    """Ticker -> sector (backward compatible)."""
    return {t: p.get("sector", "") for t, p in load_profile_map(db_path).items()}


def load_profile_map(db_path: str) -> dict[str, dict[str, str]]:
    """Ticker -> {sector, industry} in one query."""
    try:
        with db_connection(db_path, readonly=True) as conn:
            return {
                str(r[0]).upper(): {
                    "sector": r[1] or "",
                    "industry": r[2] or "",
                }
                for r in conn.execute(
                    "SELECT Ticker, Sector, Industry FROM stock_profiles"
                ).fetchall()
            }
    except sqlite3.Error:
        return {}


def load_fundamentals_profile_map(db_path: str) -> dict[str, dict[str, float | None]]:
    """Ticker -> valuation/quality fields from stock_profiles."""
    cols = (
        "Ticker",
        "Inst_Ownership",
        "Trailing_PE",
        "PEG_Ratio",
        "ROE",
        "Profit_Margins",
        "Debt_to_Equity",
    )
    try:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                f"""
                SELECT {", ".join(cols)} FROM stock_profiles
                """
            ).fetchall()
    except sqlite3.Error:
        return {}
    out: dict[str, dict[str, float | None]] = {}
    keys = cols[1:]
    for row in rows:
        sym = str(row[0]).upper()
        out[sym] = {
            k: (float(row[i + 1]) if row[i + 1] is not None else None)
            for i, k in enumerate(keys)
        }
    return out


def load_fundamentals_eps_by_ticker(
    db_path: str,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load quarterly and annual Basic EPS for all tickers in two queries."""
    empty = pd.DataFrame()
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    try:
        with db_connection(db_path, readonly=True) as conn:
            q_all = pd.read_sql(
                """
                SELECT Ticker, Report_Date, Value FROM fundamentals
                WHERE Metric = 'Basic EPS' AND Period_Type = 'Quarterly'
                ORDER BY Ticker, Report_Date ASC
                """,
                conn,
            )
            a_all = pd.read_sql(
                """
                SELECT Ticker, Report_Date, Value FROM fundamentals
                WHERE Metric = 'Basic EPS' AND Period_Type = 'Annual'
                ORDER BY Ticker, Report_Date ASC
                """,
                conn,
            )
    except sqlite3.Error:
        return out

    if not q_all.empty:
        q_all["Report_Date"] = pd.to_datetime(q_all["Report_Date"])
        for ticker, grp in q_all.groupby("Ticker"):
            df = grp.drop(columns=["Ticker"]).copy()
            df["Growth_Q"] = df["Value"].pct_change(periods=4, fill_method=None)
            out[str(ticker)] = (df, empty.copy())

    if not a_all.empty:
        a_all["Report_Date"] = pd.to_datetime(a_all["Report_Date"])
        for ticker, grp in a_all.groupby("Ticker"):
            df = grp.drop(columns=["Ticker"]).copy()
            df["Growth_A"] = df["Value"].pct_change(periods=1, fill_method=None)
            key = str(ticker)
            if key in out:
                out[key] = (out[key][0], df)
            else:
                out[key] = (empty.copy(), df)

    return out


def load_history_grouped(
    db_path: str,
    *,
    tickers: list[str] | None = None,
    full_ohlcv: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load stock_history grouped by ticker (single query)."""
    df = load_entire_database(db_path, full_ohlcv=full_ohlcv, tickers=tickers)
    if df is None or df.empty:
        return {}
    if "Ticker" not in df.columns:
        return {}
    if tickers:
        allowed = {t.upper() for t in tickers}
        df = df[df["Ticker"].astype(str).str.upper().isin(allowed)]
    grouped: dict[str, pd.DataFrame] = {}
    for ticker, grp in df.groupby("Ticker"):
        g = grp.drop(columns=["Ticker"]).copy()
        if g.index.has_duplicates:
            g = g[~g.index.duplicated(keep="last")]
        grouped[str(ticker)] = g
    return grouped


def prepare_market_frame(db_path: str, market_ticker: str) -> pd.DataFrame | None:
    return load_market_data(db_path, market_ticker)
