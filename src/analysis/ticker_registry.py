"""Watchlist ticker registry stored in SQLite."""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

WatchlistFilter = Literal[
    "warnings",
    "dead_no_history",
    "dead_with_history",
    "archived",
]

POOL_FOCUS = "focus"
POOL_UNIVERSE = "universe"
IngestScope = Literal["focus", "universe", "custom", "all"]

from src.analysis.db import db_connection, db_write_lock
from src.analysis.db_perf import DbRunSummary, execute_with_retry, track_db_op
from src.analysis.history_coverage import init_coverage_schema
from src.analysis.ingest import init_db, load_tickers_from_csv

_REGISTRY_READY: set[str] = set()
_BATCH_CHUNK = 500

WARNING_STATUSES = frozenset({"invalid", "delisted", "review"})

_DEAD_WHERE = """
    (
        validation_status IN ('delisted', 'invalid')
        OR last_ingest_status = 'No Price Data'
    )
"""

_TICKER_TABLES = (
    "stock_history",
    "fundamentals",
    "stock_profiles",
    "insider_trading",
    "stock_news",
    "strategy_rankings",
    "volatility_metrics",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _normalize(symbol: str) -> str:
    return str(symbol).strip().upper()


@dataclass
class WatchlistRow:
    symbol: str
    added_at: str | None
    skip_ingest: bool
    last_ingest_at: str | None
    last_ingest_status: str | None
    has_history: bool
    validation_status: str
    validation_reason: str | None
    last_validated_at: str | None
    coverage_status: str | None = None
    price_max_date: str | None = None
    gap_count: int = 0
    gap_summary: str | None = None
    archive_reason: str | None = None
    archived_at: str | None = None
    investigation_at: str | None = None
    delist_category: str | None = None
    pool: str = POOL_UNIVERSE
    focus_sort_order: int | None = None
    universe_last_ingest_at: str | None = None

    @property
    def is_focus(self) -> bool:
        return self.pool == POOL_FOCUS

    @classmethod
    def from_row(cls, row: sqlite3.Row | tuple) -> WatchlistRow:
        if hasattr(row, "keys"):
            d = dict(row)
            return cls(
                symbol=d["symbol"],
                added_at=d.get("added_at"),
                skip_ingest=bool(d.get("skip_ingest")),
                last_ingest_at=d.get("last_ingest_at"),
                last_ingest_status=d.get("last_ingest_status"),
                has_history=bool(d.get("has_history")),
                validation_status=d.get("validation_status") or "ok",
                validation_reason=d.get("validation_reason"),
                last_validated_at=d.get("last_validated_at"),
                coverage_status=d.get("coverage_status"),
                price_max_date=d.get("price_max_date"),
                gap_count=int(d.get("gap_count") or 0),
                gap_summary=d.get("gap_summary"),
                archive_reason=d.get("archive_reason"),
                archived_at=d.get("archived_at"),
                investigation_at=d.get("investigation_at"),
                delist_category=d.get("delist_category"),
                pool=d.get("pool") or POOL_UNIVERSE,
                focus_sort_order=(
                    int(d["focus_sort_order"])
                    if d.get("focus_sort_order") is not None
                    else None
                ),
                universe_last_ingest_at=d.get("universe_last_ingest_at"),
            )
        return cls(
            symbol=row[0],
            added_at=row[1],
            skip_ingest=bool(row[2]),
            last_ingest_at=row[3],
            last_ingest_status=row[4],
            has_history=bool(row[5]),
            validation_status=row[6] or "ok",
            validation_reason=row[7],
            last_validated_at=row[8],
        )


def is_dead_symbol(row: WatchlistRow) -> bool:
    """Symbol that cannot be pulled: delisted/invalid validation or no Yahoo price."""
    if row.validation_status in ("delisted", "invalid"):
        return True
    return row.last_ingest_status == "No Price Data"


def compose_status_tooltip(row: WatchlistRow) -> str:
    """Human-readable explanation for Status column tooltips."""
    parts: list[str] = []
    if row.skip_ingest:
        parts.append(row.archive_reason or "Updates paused (archived)")
        if row.archived_at:
            parts.append(f"Archived at {row.archived_at}")
    if row.validation_status not in ("ok", "archived"):
        parts.append(f"Validation: {row.validation_status}")
    if row.validation_reason:
        parts.append(row.validation_reason)
    if row.last_ingest_status and row.last_ingest_status != "Success":
        parts.append(f"Last ingest: {row.last_ingest_status}")
    if row.delist_category:
        parts.append(f"Category: {row.delist_category}")
    text = " — ".join(parts) if parts else "No issues recorded"
    return text[:240]


def _filter_clause(
    watchlist_filter: WatchlistFilter | None,
    *,
    warnings_only: bool,
) -> str | None:
    if watchlist_filter == "warnings" or (warnings_only and watchlist_filter is None):
        return (
            "(validation_status IN ('invalid','delisted','review') "
            "OR (last_ingest_status IS NOT NULL AND last_ingest_status != 'Success' "
            "AND has_history = 1))"
        )
    if watchlist_filter == "dead_no_history":
        return f"({_DEAD_WHERE.strip()} AND has_history = 0)"
    if watchlist_filter == "dead_with_history":
        return f"({_DEAD_WHERE.strip()} AND has_history = 1)"
    if watchlist_filter == "archived":
        return "skip_ingest = 1"
    return None


def _focus_max(db_path: str | os.PathLike | None = None) -> int:
    from src.services.stock_config import stock_config

    return stock_config().focus_watchlist_max


def init_pool_schema(db_path: str | os.PathLike) -> None:
    """Add pool columns and migrate existing rows to universe."""
    with db_connection(db_path, readonly=False) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(watchlist_tickers)")}
        migrations = {
            "pool": "TEXT NOT NULL DEFAULT 'universe'",
            "focus_sort_order": "INTEGER",
            "universe_last_ingest_at": "TEXT",
        }
        for col, dtype in migrations.items():
            if col not in cols:
                try:
                    conn.execute(f"ALTER TABLE watchlist_tickers ADD COLUMN {col} {dtype}")
                except sqlite3.OperationalError:
                    pass
        conn.execute(
            """
            UPDATE watchlist_tickers SET pool = 'universe'
            WHERE pool IS NULL OR pool = ''
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_watchlist_pool ON watchlist_tickers(pool, skip_ingest)"
        )
        conn.commit()


def ensure_registry(db_path: str | os.PathLike) -> None:
    key = str(Path(db_path).resolve())
    if key in _REGISTRY_READY:
        return
    init_db(db_path)
    init_coverage_schema(db_path)
    init_pool_schema(db_path)
    _REGISTRY_READY.add(key)


def invalidate_registry_cache(db_path: str | os.PathLike | None = None) -> None:
    if db_path is None:
        _REGISTRY_READY.clear()
    else:
        _REGISTRY_READY.discard(str(Path(db_path).resolve()))


def _chunked(items: list[str], size: int = _BATCH_CHUNK) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def symbol_has_any_history(db_path: str | os.PathLike, symbol: str) -> bool:
    sym = _normalize(symbol)
    try:
        with db_connection(db_path, readonly=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM stock_history WHERE Ticker = ?", (sym,)
            ).fetchone()
            return bool(row and row[0] > 0)
    except sqlite3.OperationalError:
        return False


def refresh_has_history(db_path: str | os.PathLike, symbol: str) -> bool:
    has = symbol_has_any_history(db_path, symbol)
    sym = _normalize(symbol)
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            "UPDATE watchlist_tickers SET has_history = ? WHERE symbol = ?",
            (1 if has else 0, sym),
        )
        conn.commit()
    return has


def purge_symbol_data(
    db_path: str | os.PathLike,
    symbol: str,
    *,
    conn: sqlite3.Connection | None = None,
) -> dict[str, int]:
    sym = _normalize(symbol)
    deleted: dict[str, int] = {}

    def _purge_on(c: sqlite3.Connection) -> None:
        for table in _TICKER_TABLES:
            try:
                cur = c.execute(f"DELETE FROM {table} WHERE Ticker = ?", (sym,))
                deleted[table] = cur.rowcount
            except sqlite3.OperationalError:
                deleted[table] = 0
        c.execute(
            """
            UPDATE watchlist_tickers SET
                price_min_date = NULL, price_max_date = NULL, price_row_count = 0,
                coverage_status = 'empty', gap_count = 0, fundamentals_last_at = NULL
            WHERE symbol = ?
        """,
            (sym,),
        )

    if conn is not None:
        _purge_on(conn)
        return deleted

    with db_connection(db_path, readonly=False) as c:
        _purge_on(c)
        c.commit()
    return deleted


def _purge_symbols_batch(db_path: str | os.PathLike, symbols: list[str]) -> dict[str, int]:
    """Delete market data for many tickers in one transaction."""
    totals: dict[str, int] = {t: 0 for t in _TICKER_TABLES}
    if not symbols:
        return totals
    with db_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            for chunk in _chunked(symbols):
                placeholders = ", ".join(["?"] * len(chunk))
                for table in _TICKER_TABLES:
                    try:
                        cur = conn.execute(
                            f"DELETE FROM {table} WHERE Ticker IN ({placeholders})",
                            chunk,
                        )
                        totals[table] = totals.get(table, 0) + int(cur.rowcount or 0)
                    except sqlite3.OperationalError:
                        pass
                conn.execute(
                    f"""
                    UPDATE watchlist_tickers SET
                        price_min_date = NULL, price_max_date = NULL, price_row_count = 0,
                        coverage_status = 'empty', gap_count = 0, fundamentals_last_at = NULL
                    WHERE symbol IN ({placeholders})
                """,
                    chunk,
                )
            conn.commit()
    return totals


def _delete_watchlist_symbols(db_path: str | os.PathLike, symbols: list[str]) -> int:
    if not symbols:
        return 0
    removed = 0
    with db_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            for chunk in _chunked(symbols):
                placeholders = ", ".join(["?"] * len(chunk))
                cur = conn.execute(
                    f"DELETE FROM watchlist_tickers WHERE symbol IN ({placeholders})",
                    chunk,
                )
                removed += int(cur.rowcount or 0)
            conn.commit()
    return removed


def add_symbol(
    db_path: str | os.PathLike,
    symbol: str,
    *,
    pool: str = POOL_UNIVERSE,
) -> bool:
    sym = _normalize(symbol)
    if not sym or sym == "NAN":
        return False
    if pool == POOL_FOCUS:
        return add_to_focus(db_path, sym).get("added", False)
    ensure_registry(db_path)
    has = symbol_has_any_history(db_path, sym)
    with db_connection(db_path, readonly=False) as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO watchlist_tickers
            (symbol, added_at, has_history, validation_status, pool)
            VALUES (?, ?, ?, 'ok', ?)
        """,
            (sym, _utc_now(), 1 if has else 0, POOL_UNIVERSE),
        )
        conn.commit()
    return cur.rowcount > 0


def count_focus_symbols(db_path: str | os.PathLike, *, active_only: bool = True) -> int:
    ensure_registry(db_path)
    clause = "pool = ?" + (" AND skip_ingest = 0" if active_only else "")
    params: tuple[Any, ...] = (POOL_FOCUS,)
    with db_connection(db_path, readonly=True) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM watchlist_tickers WHERE {clause}",
            params,
        ).fetchone()
    return int(row[0]) if row else 0


def list_focus_symbols(
    db_path: str | os.PathLike,
    *,
    include_archived: bool = False,
) -> list[WatchlistRow]:
    ensure_registry(db_path)
    clauses = ["pool = ?"]
    params: list[Any] = [POOL_FOCUS]
    if not include_archived:
        clauses.append("skip_ingest = 0")
    where = " WHERE " + " AND ".join(clauses)
    with db_connection(db_path, readonly=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM watchlist_tickers{where} "
            "ORDER BY COALESCE(focus_sort_order, 999), symbol",
            params,
        ).fetchall()
    return [WatchlistRow.from_row(r) for r in rows]


def add_to_focus(db_path: str | os.PathLike, symbol: str) -> dict[str, Any]:
    """Add symbol to focus pool (max 20). Ensures row exists in registry."""
    sym = _normalize(symbol)
    if not sym or sym == "NAN":
        return {"ok": False, "error": "invalid symbol"}
    ensure_registry(db_path)
    cap = _focus_max()
    existing = get_symbol(db_path, sym)
    if existing and existing.pool == POOL_FOCUS and not existing.skip_ingest:
        return {"ok": True, "added": False, "symbol": sym, "message": "already on focus watchlist"}

    active_focus = count_focus_symbols(db_path)
    if active_focus >= cap and not (existing and existing.pool == POOL_FOCUS):
        return {
            "ok": False,
            "error": f"Focus watchlist is full ({cap} symbols). Remove one before adding.",
            "cap": cap,
        }

    has = symbol_has_any_history(db_path, sym)
    now = _utc_now()
    with db_connection(db_path, readonly=False) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(focus_sort_order), -1) FROM watchlist_tickers WHERE pool = ?",
            (POOL_FOCUS,),
        ).fetchone()
        next_order = int(row[0] or -1) + 1
        conn.execute(
            """
            INSERT INTO watchlist_tickers
            (symbol, added_at, has_history, validation_status, pool, focus_sort_order, skip_ingest)
            VALUES (?, ?, ?, 'ok', ?, ?, 0)
            ON CONFLICT(symbol) DO UPDATE SET
                pool = excluded.pool,
                focus_sort_order = COALESCE(watchlist_tickers.focus_sort_order, excluded.focus_sort_order),
                skip_ingest = 0
            """,
            (sym, now, 1 if has else 0, POOL_FOCUS, next_order),
        )
        conn.commit()
    if not has:
        refresh_has_history(db_path, sym)
    return {"ok": True, "added": True, "symbol": sym, "focus_count": count_focus_symbols(db_path)}


def remove_from_focus(db_path: str | os.PathLike, symbol: str) -> dict[str, Any]:
    """Move symbol from focus to universe without deleting market data."""
    sym = _normalize(symbol)
    ensure_registry(db_path)
    removed = 0
    with db_connection(db_path, readonly=False) as conn:
        cur = conn.execute(
            """
            UPDATE watchlist_tickers
            SET pool = ?, focus_sort_order = NULL
            WHERE symbol = ? AND pool = ?
            """,
            (POOL_UNIVERSE, sym, POOL_FOCUS),
        )
        removed = int(cur.rowcount or 0)
        conn.commit()
    return {"ok": True, "removed": removed > 0, "symbol": sym}


def promote_to_focus(db_path: str | os.PathLike, symbol: str) -> dict[str, Any]:
    """Promote an existing universe symbol to focus."""
    return add_to_focus(db_path, symbol)


def add_symbols_bulk(db_path: str | os.PathLike, symbols: list[str]) -> int:
    ensure_registry(db_path)
    normalized = [_normalize(s) for s in symbols if s]
    normalized = [s for s in dict.fromkeys(normalized) if s and s != "NAN"]
    if not normalized:
        return 0
    now = _utc_now()
    added = 0
    with db_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            for sym in normalized:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO watchlist_tickers
                    (symbol, added_at, has_history, validation_status, pool)
                    VALUES (?, ?, 0, 'ok', ?)
                """,
                    (sym, now, POOL_UNIVERSE),
                )
                if cur.rowcount:
                    added += 1
            conn.commit()
    return added


def get_symbol(db_path: str | os.PathLike, symbol: str) -> WatchlistRow | None:
    sym = _normalize(symbol)
    try:
        with db_connection(db_path, readonly=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM watchlist_tickers WHERE symbol = ?", (sym,)
            ).fetchone()
        return WatchlistRow.from_row(row) if row else None
    except sqlite3.OperationalError:
        ensure_registry(db_path)
        with db_connection(db_path, readonly=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM watchlist_tickers WHERE symbol = ?", (sym,)
            ).fetchone()
        return WatchlistRow.from_row(row) if row else None


def load_ingest_skip_lookup(
    db_path: str | os.PathLike,
    *,
    retry_dead: bool = False,
    force_retry: bool = False,
) -> dict[str, str]:
    """Load symbol -> skip_reason for ingest planning in one query."""
    if retry_dead or force_retry:
        return {}
    ensure_registry(db_path)
    lookup: dict[str, str] = {}
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(
            """
            SELECT symbol, skip_ingest, validation_status, last_ingest_status
            FROM watchlist_tickers
        """
        ).fetchall()
    for sym, skip_ingest, validation_status, last_status in rows:
        sym = _normalize(sym)
        if skip_ingest:
            lookup[sym] = "archived"
        elif validation_status in ("delisted", "invalid"):
            lookup[sym] = validation_status
        elif last_status == "No Price Data":
            lookup[sym] = "no price data"
    return lookup


def bulk_skip_dead_tickers(db_path: str | os.PathLike) -> dict[str, int]:
    """Archive symbols already known dead (no Yahoo price, delisted, or invalid)."""
    ensure_registry(db_path)
    now = _utc_now()
    with track_db_op("bulk_skip_dead_tickers"):
        with db_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                cur = conn.execute(
                    f"""
                    UPDATE watchlist_tickers
                    SET skip_ingest = 1,
                        archive_reason = 'Bulk skip dead tickers',
                        archived_at = ?
                    WHERE skip_ingest = 0
                      AND {_DEAD_WHERE.strip()}
                """,
                    (now,),
                )
                conn.commit()
                return {"skipped": int(cur.rowcount)}


def prune_ticker_csv_from_watchlist(
    db_path: str | os.PathLike,
    csv_path: str | os.PathLike,
    *,
    backup: bool = True,
    scope: Literal["removable", "all_archived"] = "removable",
) -> dict[str, Any]:
    """Remove dead symbols from ticker CSV using watchlist state."""
    from src.analysis.ticker_cleanup import apply_ticker_cleanup

    ensure_registry(db_path)
    if scope == "removable":
        remove = [s.symbol for s in list_dead_symbols(db_path, has_history=False, limit=None)]
    else:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT symbol FROM watchlist_tickers
                WHERE skip_ingest = 1
                   OR validation_status IN ('delisted', 'invalid')
                   OR last_ingest_status = 'No Price Data'
            """
            ).fetchall()
        remove = [r[0] for r in rows]
    if not remove:
        return {"removed_count": 0, "removed": [], "csv_path": str(Path(csv_path).resolve())}
    return apply_ticker_cleanup(csv_path, remove, backup=backup)


def list_dead_symbols(
    db_path: str | os.PathLike,
    *,
    has_history: bool | None = None,
    limit: int | None = 50,
    offset: int = 0,
) -> list[WatchlistRow]:
    """List symbols matching the dead predicate, optionally filtered by history."""
    filt: WatchlistFilter
    if has_history is False:
        filt = "dead_no_history"
    elif has_history is True:
        filt = "dead_with_history"
    else:
        # both dead buckets — use warnings-style OR without history split
        ensure_registry(db_path)
        clauses = [_DEAD_WHERE.strip()]
        where = " WHERE " + " AND ".join(clauses)
        query = f"SELECT * FROM watchlist_tickers{where} ORDER BY symbol"
        params: list[Any] = []
        if limit is not None and limit >= 0:
            query += " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])
        with db_connection(db_path, readonly=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [WatchlistRow.from_row(r) for r in rows]

    return list_symbols(
        db_path,
        include_archived=True,
        watchlist_filter=filt,
        limit=limit,
        offset=offset,
    )


def bulk_remove_dead_no_history(
    db_path: str | os.PathLike,
    csv_path: str | os.PathLike | None = None,
    *,
    remove_from_csv: bool = True,
    backup: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Remove dead symbols with no price history from DB and optionally ticker CSV."""
    ensure_registry(db_path)
    started = time.perf_counter()
    symbols = [s.symbol for s in list_dead_symbols(db_path, has_history=False, limit=None)]
    total = len(symbols)
    removed_db = 0

    if progress_callback:
        progress_callback(0, total, f"Removing {total} dead symbols (no history)…")

    with track_db_op("bulk_remove_dead_no_history") as meta:
        # No history rows — watchlist delete only (no per-symbol purge loop).
        removed_db = _delete_watchlist_symbols(db_path, symbols)
        meta["rows"] = removed_db

    if progress_callback and total:
        progress_callback(total, total, f"Removed {removed_db} symbols from database.")

    csv_result: dict[str, Any] | None = None
    if remove_from_csv and csv_path and symbols:
        from src.analysis.ticker_cleanup import apply_ticker_cleanup

        csv_result = apply_ticker_cleanup(csv_path, symbols, backup=backup)

    duration = time.perf_counter() - started
    summary = DbRunSummary(
        operation="bulk_remove_dead_no_history",
        duration_sec=duration,
        items_processed=removed_db,
        items_total=total,
        success_count=removed_db,
        notes=f"batch delete watchlist ({total} candidates)",
    )
    return {
        "removed_db": removed_db,
        "symbols": symbols,
        "csv": csv_result,
        "run_summary": summary.to_dict(),
    }


def set_skip_ingest(
    db_path: str | os.PathLike,
    symbol: str,
    skip: bool,
    *,
    archive_reason: str | None = None,
) -> None:
    sym = _normalize(symbol)
    now = _utc_now()
    with db_connection(db_path, readonly=False) as conn:
        if skip:
            reason = archive_reason or "Paused by user"
            conn.execute(
                """
                UPDATE watchlist_tickers
                SET skip_ingest = 1,
                    archive_reason = ?,
                    archived_at = ?
                WHERE symbol = ?
            """,
                (reason, now, sym),
            )
        else:
            conn.execute(
                """
                UPDATE watchlist_tickers
                SET skip_ingest = 0,
                    archive_reason = NULL,
                    archived_at = NULL
                WHERE symbol = ?
            """,
                (sym,),
            )
        conn.commit()


def remove_symbol(
    db_path: str | os.PathLike,
    symbol: str,
    *,
    purge_history: bool = False,
) -> dict[str, Any]:
    sym = _normalize(symbol)
    row = get_symbol(db_path, sym)
    if not row:
        return {"removed": False, "symbol": sym}

    purged: dict[str, int] | None = None
    if purge_history or not row.has_history:
        purged = purge_symbol_data(db_path, sym)

    with db_connection(db_path, readonly=False) as conn:
        conn.execute("DELETE FROM watchlist_tickers WHERE symbol = ?", (sym,))
        conn.commit()

    return {
        "removed": True,
        "symbol": sym,
        "purged": purged,
        "had_history": row.has_history,
    }


def record_ingest_result(
    db_path: str | os.PathLike,
    symbol: str,
    status: str,
) -> None:
    sym = _normalize(symbol)
    has = refresh_has_history(db_path, sym)
    validation_status = None
    if status != "Success" and has:
        validation_status = "review"
    with db_connection(db_path, readonly=False) as conn:
        if validation_status:
            conn.execute(
                """
                UPDATE watchlist_tickers
                SET last_ingest_at = ?, last_ingest_status = ?, has_history = ?,
                    validation_status = ?
                WHERE symbol = ?
            """,
                (_utc_now(), status, 1 if has else 0, validation_status, sym),
            )
        else:
            conn.execute(
                """
                UPDATE watchlist_tickers
                SET last_ingest_at = ?, last_ingest_status = ?, has_history = ?
                WHERE symbol = ?
            """,
                (_utc_now(), status, 1 if has else 0, sym),
            )
        if status == "No Price Data":
            from src.services.stock_config import stock_config

            if stock_config().auto_skip_dead_tickers:
                conn.execute(
                    """
                    UPDATE watchlist_tickers
                    SET skip_ingest = 1,
                        archive_reason = 'No Yahoo price on ingest',
                        archived_at = ?,
                        validation_status = CASE
                            WHEN validation_status IN ('ok', 'archived', 'review')
                            THEN 'delisted'
                            ELSE validation_status
                        END
                    WHERE symbol = ?
                """,
                    (_utc_now(), sym),
                )
        conn.commit()


def record_validation(
    db_path: str | os.PathLike,
    symbol: str,
    validation_status: str,
    validation_reason: str,
) -> None:
    sym = _normalize(symbol)
    has = refresh_has_history(db_path, sym)
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE watchlist_tickers
            SET validation_status = ?, validation_reason = ?,
                last_validated_at = ?, has_history = ?
            WHERE symbol = ?
        """,
            (validation_status, validation_reason, _utc_now(), 1 if has else 0, sym),
        )
        conn.commit()


def record_investigation(
    db_path: str | os.PathLike,
    symbol: str,
    *,
    validation_status: str,
    validation_reason: str,
    delist_category: str | None = None,
) -> None:
    sym = _normalize(symbol)
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE watchlist_tickers
            SET validation_status = ?,
                validation_reason = ?,
                last_validated_at = ?,
                investigation_at = ?,
                delist_category = ?
            WHERE symbol = ?
        """,
            (
                validation_status,
                validation_reason,
                _utc_now(),
                _utc_now(),
                delist_category,
                sym,
            ),
        )
        conn.commit()


def _verdict_to_status(verdict: str) -> str:
    if verdict == "remove_invalid":
        return "invalid"
    if verdict == "remove_delisted":
        return "delisted"
    if verdict == "review":
        return "review"
    return "ok"


def list_symbols(
    db_path: str | os.PathLike,
    *,
    include_archived: bool = True,
    warnings_only: bool = False,
    watchlist_filter: WatchlistFilter | None = None,
    pool: str | None = None,
    search: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
) -> list[WatchlistRow]:
    ensure_registry(db_path)
    clauses: list[str] = []
    params: list[Any] = []

    if pool:
        clauses.append("pool = ?")
        params.append(pool)
    if not include_archived and watchlist_filter != "archived":
        clauses.append("skip_ingest = 0")
    filt = _filter_clause(watchlist_filter, warnings_only=warnings_only)
    if filt:
        clauses.append(filt)
    if search:
        clauses.append("symbol LIKE ?")
        params.append(f"%{_normalize(search)}%")

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT * FROM watchlist_tickers{where} ORDER BY symbol"
    if limit is not None and limit >= 0:
        query += " LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])

    with db_connection(db_path, readonly=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    return [WatchlistRow.from_row(r) for r in rows]


def count_symbols(
    db_path: str | os.PathLike,
    *,
    include_archived: bool = True,
    warnings_only: bool = False,
    watchlist_filter: WatchlistFilter | None = None,
    pool: str | None = None,
    search: str | None = None,
) -> int:
    ensure_registry(db_path)
    clauses: list[str] = []
    params: list[Any] = []

    if pool:
        clauses.append("pool = ?")
        params.append(pool)
    if not include_archived and watchlist_filter != "archived":
        clauses.append("skip_ingest = 0")
    filt = _filter_clause(watchlist_filter, warnings_only=warnings_only)
    if filt:
        clauses.append(filt)
    if search:
        clauses.append("symbol LIKE ?")
        params.append(f"%{_normalize(search)}%")

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with db_connection(db_path, readonly=True) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM watchlist_tickers{where}", params
        ).fetchone()
    return int(row[0]) if row else 0


def count_summary(db_path: str | os.PathLike) -> dict[str, int]:
    ensure_registry(db_path)
    cap = _focus_max()
    dead_where = _DEAD_WHERE.strip()

    def _query() -> dict[str, int]:
        with db_connection(db_path, readonly=True) as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN skip_ingest = 0 THEN 1 ELSE 0 END) AS active,
                    SUM(CASE WHEN pool = ? AND skip_ingest = 0 THEN 1 ELSE 0 END) AS focus,
                    SUM(CASE WHEN skip_ingest = 0 THEN 1 ELSE 0 END) AS universe,
                    SUM(CASE WHEN skip_ingest = 1 THEN 1 ELSE 0 END) AS archived,
                    SUM(CASE WHEN validation_status IN ('invalid','delisted','review')
                        OR (last_ingest_status IS NOT NULL AND last_ingest_status != 'Success'
                            AND has_history = 1) THEN 1 ELSE 0 END) AS warnings,
                    SUM(CASE WHEN {dead_where} AND has_history = 0 THEN 1 ELSE 0 END) AS removable,
                    SUM(CASE WHEN {dead_where} AND has_history = 1 THEN 1 ELSE 0 END) AS dead_with_history
                FROM watchlist_tickers
                """,
                (POOL_FOCUS,),
            ).fetchone()
        return {
            "total": int(row[0] or 0),
            "active": int(row[1] or 0),
            "focus": int(row[2] or 0),
            "focus_cap": cap,
            "universe": int(row[3] or 0),
            "archived": int(row[4] or 0),
            "warnings": int(row[5] or 0),
            "removable": int(row[6] or 0),
            "dead_with_history": int(row[7] or 0),
        }

    try:
        return execute_with_retry("count_summary", _query, caller="registry")
    except sqlite3.OperationalError:
        return {
            "total": 0,
            "active": 0,
            "focus": 0,
            "focus_cap": cap,
            "universe": 0,
            "archived": 0,
            "warnings": 0,
            "removable": 0,
            "dead_with_history": 0,
            "locked": True,
        }


def list_symbols_for_ingest(
    db_path: str | os.PathLike,
    scope: IngestScope | str = "all",
) -> list[str]:
    """Return symbol list for an ingest run ordered focus-first when scope is universe/all."""
    ensure_registry(db_path)
    scope = (scope or "all").strip().lower()
    if scope == "focus":
        sql = """
            SELECT symbol FROM watchlist_tickers
            WHERE pool = ? AND skip_ingest = 0
            ORDER BY COALESCE(focus_sort_order, 999), symbol
        """
        params = (POOL_FOCUS,)
    elif scope == "universe":
        sql = """
            SELECT symbol FROM watchlist_tickers
            WHERE skip_ingest = 0
            ORDER BY CASE WHEN pool = 'focus' THEN 0 ELSE 1 END,
                     COALESCE(focus_sort_order, 999), symbol
        """
        params = ()
    else:
        sql = """
            SELECT symbol FROM watchlist_tickers
            WHERE skip_ingest = 0
            ORDER BY CASE WHEN pool = 'focus' THEN 0 ELSE 1 END,
                     COALESCE(focus_sort_order, 999), symbol
        """
        params = ()
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [r[0] for r in rows]


def is_focus_symbol(db_path: str | os.PathLike, symbol: str) -> bool:
    row = get_symbol(db_path, symbol)
    return bool(row and row.pool == POOL_FOCUS and not row.skip_ingest)


def record_universe_ingest_complete(db_path: str | os.PathLike) -> None:
    """Stamp universe_last_ingest_at on all active symbols after a universe-scope run."""
    now = _utc_now()
    ensure_registry(db_path)
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE watchlist_tickers SET universe_last_ingest_at = ?
            WHERE skip_ingest = 0
            """,
            (now,),
        )
        conn.commit()
    from src.services.stock_config import stock_config

    stock_config().set_last_universe_ingest_at(now)


def watchlist_is_empty(db_path: str | os.PathLike) -> bool:
    ensure_registry(db_path)
    with db_connection(db_path, readonly=True) as conn:
        row = conn.execute("SELECT COUNT(*) FROM watchlist_tickers").fetchone()
    return not row or row[0] == 0


def migrate_csv_to_db(
    db_path: str | os.PathLike,
    csv_path: str | os.PathLike,
) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.is_file():
        return {"imported": 0, "error": f"CSV not found: {path}"}
    ensure_registry(db_path)
    symbols = load_tickers_from_csv(path)

    added = 0
    now = _utc_now()
    with db_connection(db_path, readonly=False) as conn:
        for sym in symbols:
            sym = _normalize(sym)
            if not sym or sym == "NAN":
                continue
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO watchlist_tickers
                (symbol, added_at, validation_status, pool)
                VALUES (?, ?, 'ok', ?)
            """,
                (sym, now, POOL_UNIVERSE),
            )
            if cur.rowcount:
                added += 1
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM watchlist_tickers").fetchone()[0]

    for sym in symbols:
        refresh_has_history(db_path, _normalize(sym))

    return {
        "csv_path": str(path.resolve()),
        "symbols_in_csv": len(symbols),
        "newly_added": added,
        "total_in_watchlist": int(after),
    }


def auto_migrate_if_needed(
    db_path: str | os.PathLike,
    csv_path: str | os.PathLike,
) -> dict[str, Any] | None:
    if not watchlist_is_empty(db_path):
        return None
    path = Path(csv_path)
    if not path.is_file():
        return None
    return migrate_csv_to_db(db_path, path)
