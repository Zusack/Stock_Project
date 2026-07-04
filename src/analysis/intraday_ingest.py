"""Intraday bar backfill and persistence (Yahoo Finance -> SQLite)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pandas as pd

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.ingest import init_db
from src.analysis.intraday_common import INTRADAY_COLUMNS
from src.analysis.price_sources.yfinance_intraday import fetch_yfinance_intraday
from src.analysis.ticker_registry import list_focus_symbols
from src.services.stock_config import stock_config

ProgressCallback = Callable[[float, str], None]


def upsert_intraday_rows(db_path: str | os.PathLike, rows: list[tuple[Any, ...]]) -> int:
    """Insert or replace intraday bar rows under the ingest write lock."""
    if not rows:
        return 0
    placeholders = ", ".join(["?"] * len(INTRADAY_COLUMNS))
    col_names = ", ".join(INTRADAY_COLUMNS)
    sql = f"INSERT OR REPLACE INTO intraday_bars ({col_names}) VALUES ({placeholders})"
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.executemany(sql, rows)
            conn.commit()
    return len(rows)


def upsert_intraday_dataframe(db_path: str | os.PathLike, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    cols = [c for c in INTRADAY_COLUMNS if c in df.columns]
    if len(cols) != len(INTRADAY_COLUMNS):
        return 0
    sub = df[cols].copy()
    rows = [tuple(r) for r in sub.itertuples(index=False, name=None)]
    return upsert_intraday_rows(db_path, rows)


def _default_tickers(db_path: str | os.PathLike) -> list[str]:
    rows = list_focus_symbols(db_path)
    return [r.symbol for r in rows if r.symbol and not str(r.symbol).startswith("^")]


def backfill_intraday_today(
    tickers: list[str] | None = None,
    *,
    db_path: str | os.PathLike | None = None,
    interval: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Backfill the most recent session (1 day) of intraday bars for live chart startup."""
    return backfill_intraday(
        db_path=db_path,
        tickers=tickers,
        days=1,
        interval=interval,
        progress_callback=progress_callback,
    )


def backfill_intraday(
    db_path: str | os.PathLike | None = None,
    tickers: list[str] | None = None,
    *,
    days: int | None = None,
    interval: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Backfill intraday bars for focus (or custom) symbols via Yahoo Finance.

    Returns summary dict with per-ticker row counts and errors.
    """
    cfg = stock_config()
    db_path = str(db_path or cfg.db_path)
    init_db(db_path)

    symbols = tickers or _default_tickers(db_path)
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s and not s.startswith("^")))
    if not symbols:
        return {
            "error": "No symbols to backfill (add tickers to the focus watchlist).",
            "total": 0,
            "success": 0,
            "failed": 0,
            "results": [],
        }

    lookback_days = days if days is not None else cfg.intraday_backfill_days
    bar_interval = interval or cfg.intraday_interval
    extended = cfg.intraday_extended_hours

    results: list[dict[str, Any]] = []
    total = len(symbols)
    success = 0
    failed = 0

    for idx, sym in enumerate(symbols):
        if progress_callback:
            progress_callback(idx / max(total, 1), f"Backfilling {sym} ({idx + 1}/{total})…")
        try:
            df = fetch_yfinance_intraday(
                sym,
                days=int(lookback_days),
                interval=bar_interval,
                extended_hours=extended,
            )
            row_count = upsert_intraday_dataframe(db_path, df)
            status = "Success" if row_count > 0 else "No Data"
            if row_count > 0:
                success += 1
            else:
                failed += 1
            results.append(
                {"Ticker": sym, "Status": status, "Rows": row_count, "Interval": bar_interval}
            )
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "Ticker": sym,
                    "Status": f"Error: {exc}"[:120],
                    "Rows": 0,
                    "Interval": bar_interval,
                }
            )

    if progress_callback:
        progress_callback(1.0, f"Intraday backfill complete ({success}/{total} ok).")

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "interval": bar_interval,
        "days": lookback_days,
        "results": results,
    }
