"""Database access and browser APIs."""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from src.analysis.db_perf import (
    DbReadResult,
    execute_with_retry,
    is_db_locked_error,
    record_db_op,
    track_db_op,
)

_ingest_write_lock = threading.Lock()

BROWSER_TABLES = [
    "stock_history",
    "intraday_bars",
    "fundamentals",
    "stock_profiles",
    "insider_trading",
    "stock_news",
    "strategy_rankings",
    "volatility_metrics",
    "watchlist_tickers",
    "data_quality_events",
    "market_context_daily",
    "guidance_snapshots",
    "guidance_changes",
    "portfolio_suggestions",
    "alert_events",
    "signal_calibration",
    "leaderboard_runs",
    "leaderboard_snapshots",
    "quotes_snapshot",
    "watchlists",
    "watchlist_members",
    "accounts",
    "holdings",
    "trades",
    "statements",
    "account_value_history",
]

DEFAULT_BUSY_TIMEOUT_MS = 30_000
# Checkpoint WAL every N pages (default 1000). Lower = more frequent merges, less WAL growth.
WAL_AUTOCHECKPOINT_PAGES = 1000


def apply_sqlite_performance_pragmas(conn: sqlite3.Connection) -> None:
    """Tune SQLite for large RAM hosts (cache/mmap/temp store)."""
    try:
        from src.services.stock_config import stock_config

        cfg = stock_config()
        cache_mb = getattr(cfg, "sqlite_cache_mb", 0)
        mmap_mb = getattr(cfg, "sqlite_mmap_mb", 0)
        if cache_mb and cache_mb > 0:
            conn.execute(f"PRAGMA cache_size={-int(cache_mb * 1024)}")
        if mmap_mb and mmap_mb > 0:
            conn.execute(f"PRAGMA mmap_size={int(mmap_mb) * 1024 * 1024}")
        if getattr(cfg, "sqlite_temp_store_memory", True):
            conn.execute("PRAGMA temp_store=MEMORY")
    except Exception:
        pass


def configure_connection(conn: sqlite3.Connection, *, readonly: bool = False) -> None:
    """Improve concurrency: WAL for writers, long busy wait for readers during ingest."""
    conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
    apply_sqlite_performance_pragmas(conn)
    if readonly:
        conn.execute("PRAGMA query_only=ON")
    else:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA wal_autocheckpoint={WAL_AUTOCHECKPOINT_PAGES}")


def connect(db_path: str | Path, *, readonly: bool = False, timeout: float = 30.0) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if readonly:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=timeout,
            check_same_thread=False,
        )
    else:
        conn = sqlite3.connect(str(path), timeout=timeout, check_same_thread=False)
    configure_connection(conn, readonly=readonly)
    return conn


def _safe_close(conn: sqlite3.Connection | None) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except sqlite3.Error:
        pass


@contextmanager
def ingest_write_lock():
    """Serialize SQLite writes during parallel ingest workers and bulk maintenance."""
    with _ingest_write_lock:
        yield


# Alias for maintenance/registry bulk writes
db_write_lock = ingest_write_lock


@contextmanager
def db_connection(
    db_path: str | Path,
    *,
    readonly: bool = False,
    timeout: float = 30.0,
) -> Iterator[sqlite3.Connection]:
    """Open a connection and always close it, even on exceptions or process abort paths."""
    conn = connect(db_path, readonly=readonly, timeout=timeout)
    try:
        yield conn
    finally:
        _safe_close(conn)


def ensure_db_ready(db_path: str | Path) -> dict[str, Any]:
    """
    Run once at application startup.

    Ensures WAL mode is active and performs a passive WAL checkpoint so a prior
    crash does not leave readers blocked on a large -wal file. Does not force
    OS-level locks (those are released when the process exits); this addresses
    SQLite WAL recovery after abnormal termination.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        return {"exists": False, "ready": False}

    try:
        with db_connection(path, readonly=False) as conn:
            journal = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            conn.commit()
            return {
                "exists": True,
                "ready": True,
                "journal_mode": journal[0] if journal else "unknown",
            }
    except sqlite3.Error as exc:
        return {"exists": True, "ready": False, "error": str(exc)}


def checkpoint_db_on_shutdown(db_path: str | Path, *, truncate: bool = False) -> None:
    """
    Run on clean application exit. Merges WAL pages into the main DB so the
    next startup has less recovery work. PASSIVE is fast; TRUNCATE resets the
    WAL file but can take noticeable time on very large databases.
    """
    if not db_exists(db_path):
        return
    mode = "TRUNCATE" if truncate else "PASSIVE"
    try:
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(f"PRAGMA wal_checkpoint({mode})")
            conn.commit()
    except sqlite3.Error:
        pass


def db_exists(db_path: str | Path) -> bool:
    return Path(db_path).is_file()


def get_db_summary(
    db_path: str | Path,
    *,
    light: bool = True,
    retries: int = 3,
    retry_delay: float = 0.5,
) -> dict[str, Any]:
    if not db_exists(db_path):
        return {"exists": False}

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return _get_db_summary_once(db_path, light=light)
        except sqlite3.OperationalError as exc:
            last_error = exc
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    return {
        "exists": True,
        "locked": True,
        "error": str(last_error),
        "ticker_count": "?",
        "date_min": "?",
        "date_max": "?",
        "tables": {},
    }


def _get_db_summary_once(db_path: str | Path, *, light: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {"exists": True, "locked": False, "tables": {}}
    with db_connection(db_path, readonly=True) as conn:
        tickers = conn.execute("SELECT COUNT(DISTINCT Ticker) FROM stock_history").fetchone()
        summary["ticker_count"] = tickers[0] if tickers else 0

        date_range = conn.execute(
            'SELECT MIN(Date), MAX(Date) FROM stock_history'
        ).fetchone()
        summary["date_min"] = date_range[0]
        summary["date_max"] = date_range[1]

        if not light:
            for table in BROWSER_TABLES:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    summary["tables"][table] = count
                except sqlite3.OperationalError:
                    summary["tables"][table] = 0
    return summary


def ticker_has_recent_history(db_path: str | Path, ticker: str, within_days: int = 730) -> bool:
    if not db_exists(db_path):
        return False
    try:
        with db_connection(db_path, readonly=True) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM stock_history
                WHERE Ticker = ? AND Date >= date('now', ?)
            """,
                (ticker.upper(), f"-{within_days} days"),
            ).fetchone()
            return bool(row and row[0] > 0)
    except sqlite3.OperationalError:
        return False


def count_tickers(db_path: str | Path) -> int:
    if not db_exists(db_path):
        return 0
    try:
        with db_connection(db_path, readonly=True) as conn:
            row = conn.execute("SELECT COUNT(DISTINCT Ticker) FROM stock_history").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0


def list_tickers(db_path: str | Path, limit: int | None = None) -> list[str]:
    if not db_exists(db_path):
        return []
    try:
        with db_connection(db_path, readonly=True) as conn:
            query = "SELECT DISTINCT Ticker FROM stock_history ORDER BY Ticker"
            if limit:
                query += f" LIMIT {int(limit)}"
            return [r[0] for r in conn.execute(query).fetchall()]
    except sqlite3.OperationalError:
        return []


def _finalize_price_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def load_price_data(
    ticker: str,
    db_path: str | Path,
    columns: str | None = None,
    *,
    max_days: int | None = None,
) -> pd.DataFrame | None:
    result = load_price_data_checked(ticker, db_path, columns=columns, max_days=max_days)
    return result.data if result.ok else None


def load_price_data_checked(
    ticker: str,
    db_path: str | Path,
    columns: str | None = None,
    *,
    max_days: int | None = None,
) -> DbReadResult:
    """Load price history; distinguishes database lock from missing data."""
    if columns is None:
        columns = 'Date, "Adj Close", Volume'
    sym = ticker.upper()
    limit_clause = ""
    params: list[Any] = [sym]
    if max_days is not None and max_days > 0:
        limit_clause = " ORDER BY Date DESC LIMIT ?"
        params.append(int(max_days))

    def _query() -> pd.DataFrame:
        with db_connection(db_path, readonly=True) as conn:
            if max_days is not None and max_days > 0:
                df = pd.read_sql(
                    f'SELECT {columns} FROM stock_history WHERE Ticker = ?{limit_clause}',
                    conn,
                    params=params,
                )
                if not df.empty:
                    df = df.sort_values("Date")
            else:
                df = pd.read_sql(
                    f'SELECT {columns} FROM stock_history WHERE Ticker = ? ORDER BY Date ASC',
                    conn,
                    params=(sym,),
                )
            return df

    try:
        with track_db_op("load_price_data", caller=sym) as meta:
            df = execute_with_retry("load_price_data", _query, retries=2, caller=sym)
            meta["rows"] = len(df)
    except sqlite3.OperationalError as exc:
        if is_db_locked_error(exc):
            return DbReadResult(data=None, locked=True, error=str(exc))
        return DbReadResult(data=None, locked=False, error=str(exc))

    finalized = _finalize_price_frame(df)
    return DbReadResult(data=finalized, locked=False)


def load_prices_bulk(
    tickers: Iterable[str],
    db_path: str | Path,
    *,
    columns: str | None = None,
    max_days: int | None = 400,
) -> dict[str, pd.DataFrame | None]:
    """Load price history for multiple tickers in one query."""
    symbols = [_normalize_ticker(t) for t in tickers if t]
    symbols = list(dict.fromkeys(symbols))
    out: dict[str, pd.DataFrame | None] = {s: None for s in symbols}
    if not symbols or not db_exists(db_path):
        return out
    if columns is None:
        columns = 'Ticker, Date, "Adj Close", Volume'

    placeholders = ", ".join(["?"] * len(symbols))

    def _query() -> pd.DataFrame:
        with db_connection(db_path, readonly=True) as conn:
            return pd.read_sql(
                f"SELECT {columns} FROM stock_history WHERE Ticker IN ({placeholders}) ORDER BY Ticker, Date",
                conn,
                params=symbols,
            )

    try:
        with track_db_op("load_prices_bulk", caller=f"n={len(symbols)}") as meta:
            df = execute_with_retry("load_prices_bulk", _query, retries=2)
            meta["rows"] = len(df)
    except sqlite3.OperationalError:
        return out

    if df.empty:
        return out

    if "Ticker" not in df.columns:
        return out

    for sym, group in df.groupby("Ticker"):
        sym = str(sym).upper()
        sub = group.drop(columns=["Ticker"], errors="ignore").copy()
        finalized = _finalize_price_frame(sub)
        if finalized is not None and max_days and len(finalized) > max_days:
            finalized = finalized.iloc[-max_days:].copy()
        out[sym] = finalized
    return out


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def load_ohlcv(ticker: str, db_path: str | Path) -> pd.DataFrame | None:
    return load_price_data(
        ticker,
        db_path,
        'Date, Open, High, Low, Close, "Adj Close", Volume',
    )


def _finalize_intraday_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty:
        return None
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.set_index("Timestamp").sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def load_intraday_bars(
    ticker: str,
    db_path: str | Path,
    *,
    interval: str = "1Min",
    max_points: int | None = 390,
    columns: str | None = None,
) -> pd.DataFrame | None:
    """Load intraday OHLCV bars; returns timestamp-indexed frame."""
    if not db_exists(db_path):
        return None
    if columns is None:
        columns = "Timestamp, Open, High, Low, Close, Volume, Vwap, Trade_Count"
    sym = _normalize_ticker(ticker)
    limit_clause = ""
    params: list[Any] = [sym, interval]
    if max_points is not None and max_points > 0:
        limit_clause = " ORDER BY Timestamp DESC LIMIT ?"
        params.append(int(max_points))

    def _query() -> pd.DataFrame:
        with db_connection(db_path, readonly=True) as conn:
            if max_points is not None and max_points > 0:
                df = pd.read_sql(
                    f"SELECT {columns} FROM intraday_bars "
                    f"WHERE Ticker = ? AND Interval = ?{limit_clause}",
                    conn,
                    params=params,
                )
                if not df.empty:
                    df = df.sort_values("Timestamp")
            else:
                df = pd.read_sql(
                    f"SELECT {columns} FROM intraday_bars "
                    "WHERE Ticker = ? AND Interval = ? ORDER BY Timestamp ASC",
                    conn,
                    params=(sym, interval),
                )
            return df

    try:
        with track_db_op("load_intraday_bars", caller=sym) as meta:
            df = execute_with_retry("load_intraday_bars", _query, retries=2, caller=sym)
            meta["rows"] = len(df)
    except sqlite3.OperationalError:
        return None

    return _finalize_intraday_frame(df)


def load_market_data(db_path: str | Path, market_ticker: str) -> pd.DataFrame | None:
    """Load benchmark index with 50-day SMA and uptrend flag."""
    try:
        with db_connection(db_path, readonly=True) as conn:
            df = pd.read_sql(
                'SELECT Date, "Adj Close" FROM stock_history WHERE Ticker = ? ORDER BY Date',
                conn,
                params=(market_ticker,),
            )
    except sqlite3.OperationalError:
        return None

    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df["SMA50"] = df["Adj Close"].rolling(window=50).mean()
    df["Uptrend"] = df["Adj Close"] > df["SMA50"]
    return df[["Adj Close", "SMA50", "Uptrend"]]


def browse_table(
    db_path: str | Path,
    table: str,
    ticker: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    if table not in BROWSER_TABLES:
        raise ValueError(f"Unknown table: {table}")

    with db_connection(db_path, readonly=True) as conn:
        if ticker and table != "stock_profiles":
            df = pd.read_sql(
                f"SELECT * FROM {table} WHERE Ticker = ? ORDER BY 1 DESC LIMIT ?",
                conn,
                params=(ticker.upper(), limit),
            )
        elif ticker and table == "stock_profiles":
            df = pd.read_sql(
                f"SELECT * FROM {table} WHERE Ticker = ? LIMIT ?",
                conn,
                params=(ticker.upper(), limit),
            )
        else:
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT ?", conn, params=(limit,))
    return df


def load_entire_database(
    db_path: str | Path,
    columns: str = 'Ticker, Date, "Adj Close"',
    full_ohlcv: bool = False,
    tickers: list[str] | None = None,
) -> pd.DataFrame | None:
    if not db_exists(db_path):
        return None
    if full_ohlcv:
        query = "SELECT * FROM stock_history"
    else:
        query = f"SELECT {columns} FROM stock_history"

    params: tuple[Any, ...] | None = None
    if tickers:
        syms = [_normalize_ticker(t) for t in tickers if t]
        if syms:
            placeholders = ", ".join(["?"] * len(syms))
            query += f" WHERE Ticker IN ({placeholders})"
            params = tuple(syms)

    try:
        with db_connection(db_path, readonly=True) as conn:
            df = pd.read_sql(query, conn, params=params)
    except sqlite3.OperationalError:
        return None

    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.set_index("Date", inplace=True)
    return df


def resolve_market_ticker(db_path: str | Path, preferred: str | None = None) -> str:
    cfg_preferred = preferred or "^DJI"
    tickers = set(list_tickers(db_path))
    if cfg_preferred in tickers:
        return cfg_preferred
    if "^IXIC" in tickers:
        return "^IXIC"
    if tickers:
        return sorted(tickers)[0]
    return cfg_preferred


def get_profile(db_path: str | Path, ticker: str) -> dict | None:
    if not db_exists(db_path):
        return None
    try:
        with db_connection(db_path, readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM stock_profiles WHERE Ticker = ?", (ticker.upper(),)
            ).fetchone()
            if not row:
                return None
            cols = [d[0] for d in conn.execute("PRAGMA table_info(stock_profiles)")]
            return dict(zip(cols, row))
    except sqlite3.OperationalError:
        return None
