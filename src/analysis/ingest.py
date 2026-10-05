"""Data ingestion from Yahoo Finance into SQLite."""

from __future__ import annotations

import logging
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.fundamentals_sources import fetch_fundamentals_for_ticker
from src.analysis.fundamentals_sources.rate_limits import (
    YahooRateLimiter,
    ingest_uses_sec,
    reset_ingest_budgets,
)
from src.analysis.history_coverage import (
    TickerIngestPlan,
    assess_ingest_preflight,
    build_ticker_plan,
    clear_ingest_skip_lookup,
    init_coverage_schema,
    refresh_ticker_coverage,
    set_fundamentals_last_at,
    set_ingest_skip_lookup,
)
from src.analysis.market_calendar import filter_settled_daily_bars, last_completed_trading_day
from src.analysis.ingest_run import (
    complete_run,
    count_items_by_state,
    create_run,
    get_resumable_run,
    get_run,
    init_ingest_run_schema,
    pause_run,
    resume_run,
    run_progress_summary,
)
from src.services.stock_config import stock_config

ProgressCallback = Callable[[float, str, dict[str, Any] | None, str | None], None]

_TERMINAL_SKIP_STATUSES = frozenset({"No Price Data"})
_DEFAULT_PRICE_FETCH_TIMEOUT_SEC = 180.0


@contextmanager
def _quiet_yfinance_logs():
    """Reduce yfinance ERROR spam on stderr during bulk ingest."""
    log = logging.getLogger("yfinance")
    prev = log.level
    log.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        log.setLevel(prev)


def _log_ingest(message: str, level: str = "INFO", **kwargs: Any) -> None:
    """Write ingest diagnostics to app log and ingest_detail.log."""
    try:
        from src.utils.logger_utils import append_ingest_detail_log, app_logger

        detail = message
        if kwargs:
            parts = [f"{k}={v}" for k, v in kwargs.items()]
            detail = f"{message} | " + ", ".join(parts)
        append_ingest_detail_log(f"[{level}] {detail}")
        app_logger.log("INGEST", message, level=level, **kwargs)
    except Exception:
        pass


def _ingest_item_skipped(status: str) -> bool:
    return status.startswith("Skipped") or status in _TERMINAL_SKIP_STATUSES


def _status_counts_line(prog: dict[str, Any]) -> str:
    total = int(prog.get("total") or 0)
    finished = int(prog.get("finished") or 0)
    remaining = int(prog.get("remaining") if prog.get("remaining") is not None else prog.get("pending") or 0)
    return (
        f"Progress: {finished}/{total} finished, {remaining} remaining "
        f"({prog.get('done', 0)} ok, {prog.get('skipped', 0)} skipped, {prog.get('failed', 0)} failed)"
    )


def _format_ticker_log_line(res: dict[str, Any]) -> str | None:
    """One-line UI log entry for a single ticker; None to omit routine skips."""
    ticker = res.get("Ticker", "")
    status = res.get("Status", "")
    price_rows = int(res.get("Price_Rows") or 0)
    fund_rows = int(res.get("Fund_Rows") or 0)
    price_action = res.get("Price_Action", "")

    if status.startswith("Skipped"):
        return None

    if status == "Success":
        parts = []
        if price_rows:
            parts.append(f"{price_rows} price rows")
        if fund_rows:
            parts.append(f"{fund_rows} fund rows")
        suffix = f" ({', '.join(parts)})" if parts else ""
        return f"OK  {ticker}{suffix}"

    if status == "No Price Data":
        note = "no Yahoo price history (likely delisted)"
        if price_action == "full":
            note += "; first-time full download attempted"
        return f"DEAD {ticker} — {note}"

    if status.startswith("Price Error") or status.startswith("General Error"):
        _log_ingest(
            f"Ticker error: {ticker}",
            level="WARN",
            status=status,
            price_action=price_action,
        )
        return f"FAIL {ticker} — {status}"

    if status != "Success":
        _log_ingest(
            f"Ticker issue: {ticker}",
            level="WARN",
            status=status,
            price_action=price_action,
        )
        return f"WARN {ticker} — {status}"

    return None


def _status_indicates_no_quote(status: str) -> bool:
    s = status.lower()
    return (
        status == "No Price Data"
        or "no price" in s
        or "not found" in s
        or "delisted" in s
        or "404" in s
        or "no timezone" in s
    )


def init_db(db_path: str | os.PathLike) -> None:
    with db_connection(db_path, readonly=False) as conn:
        _init_db_schema(conn)
    try:
        from src.analysis.intelligence_schema import ensure_intelligence_schema

        ensure_intelligence_schema(db_path)
    except Exception:
        pass


def _ensure_query_indexes(c: sqlite3.Cursor) -> None:
    """Indexes for hot read paths (ticker-scoped history, fundamentals, news)."""
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_stock_history_ticker_date "
        "ON stock_history(Ticker, Date)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_metric "
        "ON fundamentals(Ticker, Metric, Period_Type)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_stock_news_ticker_date ON stock_news(Ticker, Date)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_intraday_ticker_ts "
        "ON intraday_bars(Ticker, Interval, Timestamp)"
    )


def _init_db_schema(conn: sqlite3.Connection) -> None:
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_history (
            Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL,
            "Adj Close" REAL, Volume INTEGER, Dividends REAL, "Stock Splits" REAL,
            "Capital Gains" REAL, UNIQUE(Ticker, Date)
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS intraday_bars (
            Ticker TEXT, Timestamp TEXT, Interval TEXT,
            Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER,
            Vwap REAL, Trade_Count INTEGER,
            UNIQUE(Ticker, Timestamp, Interval)
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamentals (
            Ticker TEXT, Report_Date TEXT, Metric TEXT, Value REAL, Period_Type TEXT,
            UNIQUE(Ticker, Report_Date, Metric)
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_profiles (
            Ticker TEXT PRIMARY KEY, Inst_Ownership REAL, Trailing_PE REAL,
            Forward_PE REAL, PEG_Ratio REAL, Price_to_Book REAL, ROE REAL,
            Profit_Margins REAL, Debt_to_Equity REAL, Current_Ratio REAL,
            Short_Ratio REAL, Beta REAL, Sector TEXT, Industry TEXT, Last_Updated TEXT
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS insider_trading (
            Ticker TEXT, Date TEXT, Insider_Name TEXT, Position TEXT,
            Transaction_Type TEXT, Shares INTEGER, Value REAL,
            UNIQUE(Ticker, Date, Insider_Name, Transaction_Type)
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_news (
            Ticker TEXT, Date TEXT, Title TEXT, Publisher TEXT, Link TEXT,
            UNIQUE(Ticker, Link)
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist_tickers (
            symbol TEXT PRIMARY KEY,
            added_at TEXT,
            skip_ingest INTEGER NOT NULL DEFAULT 0,
            last_ingest_at TEXT,
            last_ingest_status TEXT,
            has_history INTEGER NOT NULL DEFAULT 0,
            validation_status TEXT DEFAULT 'ok',
            validation_reason TEXT,
            last_validated_at TEXT
        )
    """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_watchlist_skip_ingest ON watchlist_tickers(skip_ingest)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_watchlist_validation ON watchlist_tickers(validation_status)"
    )
    _ensure_query_indexes(c)

    current_cols_profile = [row[1] for row in c.execute("PRAGMA table_info(stock_profiles)")]
    new_cols_profile = {
        "Trailing_PE": "REAL",
        "Forward_PE": "REAL",
        "PEG_Ratio": "REAL",
        "Price_to_Book": "REAL",
        "ROE": "REAL",
        "Profit_Margins": "REAL",
        "Debt_to_Equity": "REAL",
        "Current_Ratio": "REAL",
        "Short_Ratio": "REAL",
        "Beta": "REAL",
        "Sector": "TEXT",
        "Industry": "TEXT",
    }
    for col, dtype in new_cols_profile.items():
        if col not in current_cols_profile:
            try:
                c.execute(f"ALTER TABLE stock_profiles ADD COLUMN {col} {dtype}")
            except sqlite3.OperationalError:
                pass

    current_cols_hist = [row[1] for row in c.execute("PRAGMA table_info(stock_history)")]
    if "Capital Gains" not in current_cols_hist:
        try:
            c.execute('ALTER TABLE stock_history ADD COLUMN "Capital Gains" REAL')
        except sqlite3.OperationalError:
            pass

    init_coverage_schema_via_conn(conn)
    init_ingest_run_schema(conn)
    conn.commit()


def init_coverage_schema_via_conn(conn: sqlite3.Connection) -> None:
    init_coverage_schema(":memory:")
    cols = {row[1] for row in conn.execute("PRAGMA table_info(watchlist_tickers)")}
    migrations = {
        "price_min_date": "TEXT",
        "price_max_date": "TEXT",
        "price_row_count": "INTEGER DEFAULT 0",
        "coverage_status": "TEXT",
        "gap_count": "INTEGER DEFAULT 0",
        "fundamentals_last_at": "TEXT",
    }
    for col, dtype in migrations.items():
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE watchlist_tickers ADD COLUMN {col} {dtype}")
            except sqlite3.OperationalError:
                pass


_EXPECTED_HIST_COLS = [
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "Capital Gains",
]


def _yf_end_exclusive(end: date) -> str:
    return (end + timedelta(days=1)).isoformat()


def _fetch_price_history(stock: yf.Ticker, plan: TickerIngestPlan) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if plan.price_action == "skip":
        return pd.DataFrame()
    if plan.price_action == "full":
        hist = stock.history(period=plan.period, auto_adjust=False, actions=True)
        if not hist.empty:
            frames.append(hist)
    elif plan.price_action in ("incremental", "gap_fill") and plan.price_ranges:
        for rng in plan.price_ranges:
            hist = stock.history(
                start=rng.start_str(),
                end=_yf_end_exclusive(rng.end),
                auto_adjust=False,
                actions=True,
            )
            if not hist.empty:
                frames.append(hist)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined, dropped = filter_settled_daily_bars(combined)
    if dropped:
        _log_ingest(
            "Dropped unsettled daily bars after fetch",
            ticker=plan.ticker,
            dropped=dropped,
            cutoff=last_completed_trading_day().isoformat(),
        )
    return combined


def _purge_unsettled_price_rows(db_path: str, ticker: str) -> int:
    """Remove in-progress session rows (e.g. today before close) from the database."""
    cutoff = last_completed_trading_day()
    sym = str(ticker).strip().upper()
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            cur = conn.execute(
                "DELETE FROM stock_history WHERE Ticker = ? AND Date > ?",
                (sym, cutoff.isoformat()),
            )
            conn.commit()
            return int(cur.rowcount or 0)


def _upsert_price_history(db_path: str, ticker: str, hist: pd.DataFrame) -> int:
    if hist.empty:
        _purge_unsettled_price_rows(db_path, ticker)
        return 0
    hist, dropped = filter_settled_daily_bars(hist)
    if dropped:
        _log_ingest(
            "Dropped unsettled daily bars before upsert",
            ticker=ticker,
            dropped=dropped,
            cutoff=last_completed_trading_day().isoformat(),
        )
    purged = _purge_unsettled_price_rows(db_path, ticker)
    if purged:
        _log_ingest(
            "Purged unsettled rows from database",
            ticker=ticker,
            purged=purged,
            cutoff=last_completed_trading_day().isoformat(),
        )
    if hist.empty:
        return 0
    hist = hist.copy()
    hist.reset_index(inplace=True)
    hist["Ticker"] = ticker
    hist["Date"] = pd.to_datetime(hist["Date"], utc=True).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    cols_to_keep = [c for c in _EXPECTED_HIST_COLS if c in hist.columns]
    hist = hist[cols_to_keep]
    rows = [tuple(r) for r in hist.itertuples(index=False, name=None)]
    if not rows:
        return 0
    placeholders = ", ".join(["?"] * len(cols_to_keep))
    col_names = ", ".join(f'"{c}"' if " " in c else c for c in cols_to_keep)
    sql = f"INSERT OR REPLACE INTO stock_history ({col_names}) VALUES ({placeholders})"
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.executemany(sql, rows)
            conn.commit()
    return len(rows)


def _new_result_log(plan: TickerIngestPlan) -> dict[str, Any]:
    return {
        "Ticker": plan.ticker,
        "Price_Rows": 0,
        "Fund_Rows": 0,
        "Fund_Source": "",
        "Fund_Note": "",
        "Insider": 0,
        "News": 0,
        "Status": "Success",
        "Price_Action": plan.price_action,
        "Fund_Action": plan.fund_action,
        "Skipped": plan.all_skip,
    }


def _yahoo_price_fetch(plan: TickerIngestPlan) -> pd.DataFrame:
    YahooRateLimiter.wait()
    stock = yf.Ticker(plan.ticker)
    return _fetch_price_history(stock, plan)


def _price_fetch_timeout_sec() -> float:
    try:
        return max(30.0, float(stock_config().ingest_price_fetch_timeout_sec))
    except (TypeError, ValueError):
        return _DEFAULT_PRICE_FETCH_TIMEOUT_SEC


def _fetch_price_for_plan(plan: TickerIngestPlan) -> pd.DataFrame:
    """Stooq backfill for empty coverage, then Yahoo (with timeout)."""
    cfg = stock_config()
    if (
        cfg.ingest_use_stooq_backfill
        and plan.price_action == "full"
        and plan.coverage_status == "empty"
    ):
        from src.analysis.price_sources import fetch_stooq_daily_history

        stooq_hist = fetch_stooq_daily_history(plan.ticker)
        if stooq_hist is not None and not stooq_hist.empty:
            stooq_hist, dropped = filter_settled_daily_bars(stooq_hist)
            if dropped:
                _log_ingest(
                    "Dropped unsettled Stooq bars",
                    ticker=plan.ticker,
                    dropped=dropped,
                    cutoff=last_completed_trading_day().isoformat(),
                )
            if not stooq_hist.empty:
                _log_ingest("Price from Stooq", ticker=plan.ticker, rows=len(stooq_hist))
                return stooq_hist

    timeout = _price_fetch_timeout_sec()
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_yahoo_price_fetch, plan)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeoutError as e:
            _log_ingest(
                "Yahoo price fetch timed out",
                level="WARN",
                ticker=plan.ticker,
                timeout_sec=timeout,
            )
            raise TimeoutError(
                f"Price fetch timed out after {int(timeout)}s for {plan.ticker}"
            ) from e


def _run_price_step(plan: TickerIngestPlan, result_log: dict[str, Any]) -> bool:
    """Returns True if ingest should stop further steps for this ticker."""
    ticker = plan.ticker
    db_path = plan.db_path
    if plan.price_action == "skip":
        return False
    try:
        hist = _fetch_price_for_plan(plan)
        if hist.empty:
            if plan.price_action == "full":
                result_log["Status"] = "No Price Data"
                return True
        else:
            result_log["Price_Rows"] = _upsert_price_history(db_path, ticker, hist)
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            raise
        result_log["Status"] = f"Price Error: {e}"
        return _status_indicates_no_quote(result_log["Status"])
    return _status_indicates_no_quote(result_log["Status"])


def _run_fundamentals_step(plan: TickerIngestPlan, result_log: dict[str, Any]) -> None:
    if plan.fund_action != "refresh":
        return
    ticker = plan.ticker
    cfg = stock_config()
    try:
        fund_data, fund_meta = fetch_fundamentals_for_ticker(
            ticker,
            source=cfg.fundamentals_source,
            sec_user_agent=cfg.sec_user_agent,
            fmp_api_key=cfg.fmp_api_key,
        )
        result_log["Fund_Rows"] = len(fund_data)
        result_log["Fund_Source"] = fund_meta.source
        result_log["Fund_Note"] = fund_meta.summary()
        if fund_data:
            with ingest_write_lock():
                with db_connection(plan.db_path, readonly=False) as conn:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO fundamentals
                        (Ticker, Report_Date, Metric, Value, Period_Type) VALUES (?, ?, ?, ?, ?)
                    """,
                        fund_data,
                    )
                    conn.commit()
            set_fundamentals_last_at(plan.db_path, ticker)
    except Exception as e:
        result_log["Fund_Note"] = f"fundamentals error: {e}"[:120]


def _run_enrich_step(plan: TickerIngestPlan, result_log: dict[str, Any]) -> None:
    if plan.enrich_action != "refresh":
        return
    cfg = stock_config()
    if not (cfg.ingest_fetch_profile or cfg.ingest_fetch_insider or cfg.ingest_fetch_news):
        return

    ticker = plan.ticker
    db_path = plan.db_path
    YahooRateLimiter.wait()
    stock = yf.Ticker(ticker)

    profile_data: dict = {}
    if cfg.ingest_fetch_profile:
        try:
            info = stock.info
            profile_data = {
                # Still ingested for the Fundamentals panel only — not used in CANSLIM.
                # Yahoo heldPercentInstitutions is stale/unreliable vs IBD proprietary "I".
                "Inst_Ownership": info.get("heldPercentInstitutions"),
                "Trailing_PE": info.get("trailingPE"),
                "Forward_PE": info.get("forwardPE"),
                "PEG_Ratio": info.get("pegRatio"),
                "Price_to_Book": info.get("priceToBook"),
                "ROE": info.get("returnOnEquity"),
                "Profit_Margins": info.get("profitMargins"),
                "Debt_to_Equity": info.get("debtToEquity"),
                "Current_Ratio": info.get("currentRatio"),
                "Short_Ratio": info.get("shortRatio"),
                "Beta": info.get("beta"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
            }
        except Exception:
            pass

    insider_rows: list[tuple] = []
    if cfg.ingest_fetch_insider:
        YahooRateLimiter.wait()
        try:
            insider = stock.insider_transactions
            if insider is not None and not insider.empty:
                for _, row in insider.iterrows():
                    d = row.get("Start Date")
                    if pd.isnull(d):
                        continue
                    date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                    insider_rows.append(
                        (
                            ticker,
                            date_str,
                            row.get("Insider", "Unknown"),
                            row.get("Position", "Unknown"),
                            row.get("Text", "Unknown"),
                            int(row.get("Shares", 0)),
                            float(row.get("Value", 0.0)),
                        )
                    )
        except Exception:
            pass

    news_count = 0
    if cfg.ingest_fetch_news:
        try:
            from src.analysis.news_sources.aggregator import (
                fetch_headlines_for_ticker,
                persist_headlines,
            )

            headlines = fetch_headlines_for_ticker(ticker, include_yahoo=True)
            news_count = persist_headlines(db_path, headlines)
        except Exception:
            YahooRateLimiter.wait()
            try:
                news = stock.news
                if news:
                    news_rows_legacy: list[tuple] = []
                    for item in news:
                        ts = item.get("providerPublishTime", 0)
                        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        news_rows_legacy.append(
                            (
                                ticker,
                                date_str,
                                item.get("title", ""),
                                item.get("publisher", ""),
                                item.get("link", ""),
                            )
                        )
                    with ingest_write_lock():
                        with db_connection(db_path, readonly=False) as conn:
                            conn.executemany(
                                """
                                INSERT OR IGNORE INTO stock_news
                                (Ticker, Date, Title, Publisher, Link) VALUES (?, ?, ?, ?, ?)
                            """,
                                news_rows_legacy,
                            )
                            conn.commit()
                    news_count = len(news_rows_legacy)
            except Exception:
                pass

    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            cursor = conn.cursor()
            if any(profile_data.values()):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO stock_profiles (
                        Ticker, Inst_Ownership, Trailing_PE, Forward_PE, PEG_Ratio,
                        Price_to_Book, ROE, Profit_Margins, Debt_to_Equity,
                        Current_Ratio, Short_Ratio, Beta, Sector, Industry, Last_Updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, DATE('now'))
                """,
                    (
                        ticker,
                        profile_data.get("Inst_Ownership"),
                        profile_data.get("Trailing_PE"),
                        profile_data.get("Forward_PE"),
                        profile_data.get("PEG_Ratio"),
                        profile_data.get("Price_to_Book"),
                        profile_data.get("ROE"),
                        profile_data.get("Profit_Margins"),
                        profile_data.get("Debt_to_Equity"),
                        profile_data.get("Current_Ratio"),
                        profile_data.get("Short_Ratio"),
                        profile_data.get("Beta"),
                        profile_data.get("Sector"),
                        profile_data.get("Industry"),
                    ),
                )
            if insider_rows:
                cursor.executemany(
                    """
                    INSERT OR IGNORE INTO insider_trading
                    (Ticker, Date, Insider_Name, Position, Transaction_Type, Shares, Value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    insider_rows,
                )
                result_log["Insider"] = len(insider_rows)
            if news_count:
                result_log["News"] = news_count
            conn.commit()


def process_ticker_from_plan(plan: TickerIngestPlan) -> dict | None:
    ticker = plan.ticker
    db_path = plan.db_path
    if not ticker or ticker == "NAN":
        return None

    result_log = _new_result_log(plan)

    if plan.all_skip:
        result_log["Status"] = "Skipped (up to date)"
        return result_log

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if _run_price_step(plan, result_log):
                try:
                    refresh_ticker_coverage(db_path, ticker)
                except Exception:
                    pass
                return result_log

            _run_fundamentals_step(plan, result_log)
            _run_enrich_step(plan, result_log)

            try:
                refresh_ticker_coverage(db_path, ticker)
            except Exception:
                pass
            break

        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(5, 15) * (attempt + 1))
                    _log_ingest("Yahoo rate limit backoff", ticker=ticker, attempt=attempt + 1)
                    continue
                result_log["Status"] = "Rate Limit (Skipped)"
                _log_ingest("Yahoo rate limit exhausted", level="WARN", ticker=ticker)
            else:
                result_log["Status"] = f"General Error: {e}"
                _log_ingest(f"General error: {ticker}", level="WARN", error=str(e)[:200])
                break

    try:
        from src.analysis.data_quality import record_ingest_result

        record_ingest_result(db_path, result_log)
    except Exception:
        pass
    return result_log


def process_ticker_price_phase(plan: TickerIngestPlan) -> dict | None:
    """Price phase only (parallel ingest)."""
    if plan.all_skip:
        r = _new_result_log(plan)
        r["Status"] = "Skipped (up to date)"
        return r
    result_log = _new_result_log(plan)
    try:
        if _run_price_step(plan, result_log):
            try:
                refresh_ticker_coverage(plan.db_path, plan.ticker)
            except Exception:
                pass
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            result_log["Status"] = "Rate Limit (Skipped)"
        elif isinstance(e, TimeoutError) or "timed out" in str(e).lower():
            result_log["Status"] = f"Price Timeout: {e}"[:120]
        else:
            result_log["Status"] = f"Price Error: {e}"
    return result_log


def process_ticker_fundamentals_phase(plan: TickerIngestPlan, partial: dict[str, Any]) -> dict[str, Any]:
    result_log = dict(partial)
    if plan.fund_action != "refresh":
        return result_log
    try:
        _run_fundamentals_step(plan, result_log)
    except Exception as e:
        result_log["Fund_Note"] = f"fundamentals error: {e}"[:120]
    return result_log


def process_ticker_enrich_phase(plan: TickerIngestPlan, partial: dict[str, Any]) -> dict[str, Any]:
    result_log = dict(partial)
    try:
        _run_enrich_step(plan, result_log)
        refresh_ticker_coverage(plan.db_path, plan.ticker)
    except Exception:
        pass
    return result_log


def process_ticker(args: tuple) -> dict | None:
    """Legacy entry point: (ticker, period, db_path) -> full smart plan."""
    ticker, period, db_path = args
    plan = build_ticker_plan(db_path, ticker, mode="smart")
    if period == "max" and plan.price_action != "full":
        pass
    plan.db_path = str(db_path)
    return process_ticker_from_plan(plan)


def load_tickers_from_csv(ticker_csv_path: str | os.PathLike) -> list[str]:
    df_tickers = pd.read_csv(ticker_csv_path)
    normalized_cols = [str(c).strip().lower() for c in df_tickers.columns]
    target_col = None
    if "symbol" in normalized_cols:
        target_col = df_tickers.columns[normalized_cols.index("symbol")]
    elif "ticker" in normalized_cols:
        target_col = df_tickers.columns[normalized_cols.index("ticker")]
    if target_col is None or "^" in str(df_tickers.columns[0]):
        df_tickers = pd.read_csv(ticker_csv_path, header=None)
        target_col = 0
    return df_tickers[target_col].dropna().astype(str).str.strip().str.upper().unique().tolist()


def ingest_stock_data(
    ticker_csv_path: str | os.PathLike | None = None,
    db_path: str | os.PathLike = "market_data.db",
    tickers: list[str] | None = None,
    period: str = "max",
    use_parallel: bool = True,
    progress_callback: ProgressCallback | None = None,
    *,
    run_id: str | None = None,
    cancel_event: threading.Event | None = None,
    mode: str | None = None,
    resume: bool = False,
    scope: str | None = None,
) -> dict[str, Any]:
    """Ingest price and fundamentals with resumable runs and incremental fetch."""
    from src.analysis import ticker_registry as registry

    db_path = str(db_path)
    init_db(db_path)
    init_coverage_schema(db_path)

    cfg = stock_config()
    ingest_mode = (mode or cfg.ingest_mode_default or "smart").strip().lower()
    if ingest_mode == "resume" or resume:
        ingest_mode = "resume"

    ingest_scope = (scope or "universe").strip().lower()
    if tickers is not None:
        ingest_scope = "custom"

    skipped_archived = 0
    if tickers is None:
        if ticker_csv_path is not None:
            import warnings

            warnings.warn(
                "ticker_csv_path is deprecated; import CSV via Settings then ingest universe scope.",
                DeprecationWarning,
                stacklevel=2,
            )
            tickers = load_tickers_from_csv(ticker_csv_path)
            ingest_scope = "custom"
        else:
            all_syms = registry.list_symbols(db_path, include_archived=True, limit=None)
            skipped_archived = len(all_syms) - sum(1 for r in all_syms if not r.skip_ingest)
            tickers = registry.list_symbols_for_ingest(db_path, ingest_scope)

    fund_source = cfg.fundamentals_source
    reset_ingest_budgets(fund_source)

    active_run_id = run_id
    if ingest_mode == "resume":
        existing = get_resumable_run(db_path) if not active_run_id else get_run(db_path, active_run_id)
        if existing:
            active_run_id = existing.run_id
            resume_run(db_path, active_run_id)
            ingest_mode = existing.mode
        elif not active_run_id:
            ingest_mode = cfg.ingest_mode_default or "smart"

    force_full_run = cfg.force_full_history or ingest_mode == "full"

    def report(pct: float, status_msg: str, *, log_line: str | None = None) -> None:
        if not progress_callback:
            return
        prog: dict[str, Any] | None = None
        if active_run_id:
            prog = run_progress_summary(db_path, active_run_id)
        try:
            progress_callback(pct, status_msg, prog, log_line)
        except TypeError:
            try:
                progress_callback(pct, status_msg, prog)  # type: ignore[misc]
            except TypeError:
                progress_callback(pct, status_msg)  # type: ignore[misc]

    if not tickers:
        label = "focus watchlist" if ingest_scope == "focus" else "research universe"
        report(0.0, f"No symbols in {label} (add symbols or check archived).")
        clear_ingest_skip_lookup()
        return {
            "total": 0,
            "processed": 0,
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "skipped_archived": skipped_archived,
            "warnings": 0,
            "db_path": os.path.abspath(db_path),
            "results": [],
            "run_id": None,
            "paused": False,
            "run_progress": {},
            "error": "No tickers to ingest",
        }

    if not active_run_id and ingest_mode == "smart" and not force_full_run:
        preflight = assess_ingest_preflight(
            db_path,
            scope=ingest_scope,
            mode=ingest_mode,
            force_full=force_full_run,
            retry_dead=cfg.retry_dead_tickers,
        )
        if not preflight.should_run:
            skip_msg = preflight.message or "No new data available since the last ingest."
            report(1.0, skip_msg)
            clear_ingest_skip_lookup()
            _log_ingest(
                "Ingest skipped — no new market data",
                scope=ingest_scope,
                calendar_required=preflight.calendar_required_day.isoformat(),
                effective_required=preflight.effective_required_day.isoformat(),
                benchmarks={
                    sym: (d.isoformat() if d else None)
                    for sym, d in preflight.benchmark_max_dates.items()
                },
            )
            return {
                "total": len(tickers),
                "processed": 0,
                "success": 0,
                "skipped": len(tickers),
                "failed": 0,
                "skipped_archived": skipped_archived,
                "warnings": 0,
                "db_path": os.path.abspath(db_path),
                "results": [],
                "run_id": None,
                "paused": False,
                "run_progress": {},
                "ingest_scope": ingest_scope,
                "no_new_data": True,
                "message": skip_msg,
                "preflight": {
                    "calendar_required_day": preflight.calendar_required_day.isoformat(),
                    "effective_required_day": preflight.effective_required_day.isoformat(),
                    "benchmark_max_dates": {
                        sym: (d.isoformat() if d else None)
                        for sym, d in preflight.benchmark_max_dates.items()
                    },
                },
            }

    if not active_run_id:
        report(0.0, f"Preparing ingest ({len(tickers)} tickers)…")
        focus_syms = {
            r.symbol for r in registry.list_focus_symbols(db_path)
        }
        active_run_id = create_run(
            db_path,
            tickers,
            ingest_mode,
            fund_source,
            plans=None,
            ingest_scope=ingest_scope,
            focus_symbols=focus_syms,
        )
        _log_ingest(
            "Created ingest run",
            run_id=active_run_id,
            mode=ingest_mode,
            scope=ingest_scope,
            total=len(tickers),
            fundamentals_source=fund_source,
        )
    else:
        _log_ingest(
            "Resuming ingest run",
            run_id=active_run_id,
            mode=ingest_mode,
        )

    run = get_run(db_path, active_run_id)
    if not run:
        return {"error": "Failed to create ingest run", "db_path": db_path}

    total_tasks = run.total_count
    results: list[dict] = []
    paused = False

    set_ingest_skip_lookup(
        registry.load_ingest_skip_lookup(
            db_path,
            retry_dead=cfg.retry_dead_tickers,
            force_retry=force_full_run,
        )
    )

    report(
        0.0,
        f"Starting ingest ({ingest_mode}, {total_tasks} tickers)…",
    )
    _log_ingest(
        "Ingest run started",
        run_id=active_run_id,
        mode=ingest_mode,
        total=total_tasks,
        fundamentals_source=fund_source,
        skipped_archived=skipped_archived,
    )

    use_parallel = use_parallel and cfg.ingest_use_parallel

    from src.analysis.ingest_workers import IngestWorkerContext, run_phased_ingest

    def worker_report(pct: float, status_msg: str, log_line: str | None) -> None:
        report(pct, status_msg, log_line=log_line)

    worker_ctx = IngestWorkerContext(
        db_path=db_path,
        run_id=active_run_id,
        ingest_mode=ingest_mode,
        force_full=cfg.force_full_history,
        total_tasks=total_tasks,
        cancel_event=cancel_event,
        report=worker_report,
        results=results,
    )

    try:
        with _quiet_yfinance_logs():
            if cancel_event is not None and cancel_event.is_set():
                pause_run(db_path, active_run_id, "Stopped by user")
                paused = True
            else:
                run_phased_ingest(worker_ctx, use_parallel=use_parallel)
                if worker_ctx.cancelled():
                    pause_run(db_path, active_run_id, "Stopped by user")
                    paused = True
                    pause_prog = run_progress_summary(db_path, active_run_id)
                    state_counts = count_items_by_state(db_path, active_run_id)
                    with worker_ctx.active_lock:
                        active = worker_ctx.active_ticker
                        phase = worker_ctx.active_phase
                    _log_ingest(
                        "Ingest run paused by user",
                        run_id=active_run_id,
                        pending=pause_prog.get("pending"),
                        done=pause_prog.get("done"),
                        skipped=pause_prog.get("skipped"),
                        failed=pause_prog.get("failed"),
                        states=state_counts,
                        active_ticker=active,
                        active_phase=phase,
                    )
                    report(
                        worker_ctx.processed / max(total_tasks, 1),
                        f"Paused — {pause_prog.get('pending', '?')} tickers remaining. "
                        "Use Resume to continue.",
                    )
                else:
                    complete_run(db_path, active_run_id)
                    if ingest_scope in ("universe", "all") and not paused:
                        registry.record_universe_ingest_complete(db_path)
                    report(1.0, "Ingest completed.")
    finally:
        clear_ingest_skip_lookup()

    prog = run_progress_summary(db_path, active_run_id)
    success = sum(1 for r in results if r.get("Status") == "Success")
    skipped_n = sum(1 for r in results if str(r.get("Status", "")).startswith("Skipped"))
    failed = len(results) - success - skipped_n
    warnings_count = sum(
        1
        for r in results
        if r.get("Status") not in ("Success",) and not str(r.get("Status", "")).startswith("Skipped")
        and registry.symbol_has_any_history(db_path, r.get("Ticker", ""))
    )
    _log_ingest(
        "Ingest run finished",
        run_id=active_run_id,
        paused=paused,
        total=total_tasks,
        success=success,
        skipped=skipped_n,
        failed=failed,
        warnings=warnings_count,
    )
    return {
        "total": total_tasks,
        "processed": len(results),
        "success": success,
        "skipped": skipped_n,
        "failed": failed,
        "skipped_archived": skipped_archived,
        "warnings": warnings_count,
        "db_path": os.path.abspath(db_path),
        "results": results,
        "run_id": active_run_id,
        "paused": paused,
        "run_progress": prog,
        "ingest_scope": ingest_scope,
    }
