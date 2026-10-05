"""Named watchlists (saved views) separate from focus/universe registry."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src.analysis.db import db_connection, ingest_write_lock

WATCHLIST_KIND_USER = "user"
WATCHLIST_KIND_HOLDINGS = "holdings"
DEFAULT_WATCHLIST_NAME = "My Watchlist"


@dataclass
class NamedWatchlist:
    id: int
    name: str
    kind: str
    sort_order: int
    created_at: str


@dataclass
class WatchlistMember:
    watchlist_id: int
    ticker: str
    sort_order: int
    added_at: str


def ensure_watchlist_schema(db_path: str) -> None:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL DEFAULT 'user',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS watchlist_members (
                    watchlist_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (watchlist_id, ticker),
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_wlm_ticker ON watchlist_members(ticker);
                """
            )
            conn.commit()
    _seed_defaults(db_path)


def _seed_defaults(db_path: str) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with db_connection(db_path, readonly=False) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM watchlists").fetchone()
        if rows and rows[0] > 0:
            return
        defaults = [
            ("My Watchlist", WATCHLIST_KIND_USER, 0),
            ("Holdings", WATCHLIST_KIND_HOLDINGS, 1),
        ]
        for name, kind, order in defaults:
            conn.execute(
                "INSERT INTO watchlists (name, kind, sort_order, created_at) VALUES (?,?,?,?)",
                (name, kind, order, now),
            )
        conn.commit()


def _collect_saved_watchlist_symbols(
    db_path: str,
    *,
    watchlist_name: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Saved watchlist tickers; default list first, then other lists, deduped."""
    ensure_watchlist_schema(db_path)
    ordered: list[str] = []
    seen: set[str] = set()

    def _take(members: list[WatchlistMember]) -> None:
        for m in members:
            sym = str(m.ticker or "").strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                ordered.append(sym)

    if watchlist_name:
        wl = get_watchlist_by_name(db_path, watchlist_name)
        if wl:
            _take(list_members(db_path, wl.id))
        return ordered[:limit] if limit is not None else ordered

    default = get_watchlist_by_name(db_path, DEFAULT_WATCHLIST_NAME)
    if default:
        _take(list_members(db_path, default.id))
    for wl in list_watchlists(db_path):
        if default and wl.id == default.id:
            continue
        _take(list_members(db_path, wl.id))

    if limit is not None:
        return ordered[:limit]
    return ordered


def list_default_watchlist_symbols(
    db_path: str,
    *,
    watchlist_name: str = DEFAULT_WATCHLIST_NAME,
    limit: int | None = None,
) -> list[str]:
    """Symbols from one named watchlist, preserving member order."""
    return _collect_saved_watchlist_symbols(
        db_path,
        watchlist_name=watchlist_name,
        limit=limit,
    )


def resolve_watchlist_symbols(
    db_path: str,
    *,
    limit: int | None = None,
) -> list[str]:
    """User-facing watchlist symbols: all saved lists, then registry focus fallback."""
    syms = _collect_saved_watchlist_symbols(db_path, limit=limit)
    if syms:
        return syms
    try:
        from src.analysis.ticker_registry import list_focus_symbols

        focus = list_focus_symbols(db_path)
    except Exception:
        return []
    out = [r.symbol for r in focus if r.symbol]
    if limit is not None:
        return out[:limit]
    return out


def sync_watchlist_members_to_registry(db_path: str) -> int:
    """Ensure every saved watchlist symbol is in the research-universe registry."""
    from src.analysis.ticker_registry import POOL_UNIVERSE, add_symbol, ensure_registry

    ensure_registry(db_path)
    added = 0
    for sym in _collect_saved_watchlist_symbols(db_path):
        if add_symbol(db_path, sym, pool=POOL_UNIVERSE):
            added += 1
    return added


def sync_focus_to_default_watchlist(db_path: str) -> int:
    """Copy registry focus symbols into 'My Watchlist' when that list has no members."""
    ensure_watchlist_schema(db_path)
    try:
        with db_connection(db_path, readonly=True) as conn:
            row = conn.execute(
                "SELECT id FROM watchlists WHERE name = ?", ("My Watchlist",)
            ).fetchone()
            if not row:
                return 0
            wl_id = int(row[0])
            member_count = conn.execute(
                "SELECT COUNT(*) FROM watchlist_members WHERE watchlist_id = ?",
                (wl_id,),
            ).fetchone()[0]
            if member_count > 0:
                return 0
    except sqlite3.OperationalError:
        return 0

    try:
        from src.analysis.ticker_registry import list_focus_symbols

        focus = list_focus_symbols(db_path)
    except Exception:
        return 0

    added = 0
    for row in focus:
        result = add_member(db_path, wl_id, row.symbol)
        if result.get("ok"):
            added += 1
    return added


def list_watchlists(db_path: str) -> list[NamedWatchlist]:
    ensure_watchlist_schema(db_path)
    try:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                "SELECT id, name, kind, sort_order, created_at FROM watchlists ORDER BY sort_order, name"
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        NamedWatchlist(id=r[0], name=r[1], kind=r[2], sort_order=r[3], created_at=r[4])
        for r in rows
    ]


def get_watchlist_by_name(db_path: str, name: str) -> NamedWatchlist | None:
    for wl in list_watchlists(db_path):
        if wl.name == name:
            return wl
    return None


def create_watchlist(db_path: str, name: str, *, kind: str = WATCHLIST_KIND_USER) -> dict:
    from datetime import datetime, timezone

    ensure_watchlist_schema(db_path)
    nm = (name or "").strip()
    if not nm:
        return {"ok": False, "error": "Name required"}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        with ingest_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                max_order = conn.execute("SELECT COALESCE(MAX(sort_order), -1) FROM watchlists").fetchone()
                order = int(max_order[0]) + 1 if max_order else 0
                conn.execute(
                    "INSERT INTO watchlists (name, kind, sort_order, created_at) VALUES (?,?,?,?)",
                    (nm, kind, order, now),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT id FROM watchlists WHERE name = ?", (nm,)
                ).fetchone()
        return {"ok": True, "id": row[0] if row else None}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": f"Watchlist '{nm}' already exists"}
    except sqlite3.OperationalError as ex:
        return {"ok": False, "error": str(ex)}


def rename_watchlist(db_path: str, watchlist_id: int, new_name: str) -> dict:
    nm = (new_name or "").strip()
    if not nm:
        return {"ok": False, "error": "Name required"}
    try:
        with ingest_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                conn.execute("UPDATE watchlists SET name = ? WHERE id = ?", (nm, watchlist_id))
                conn.commit()
        return {"ok": True}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Name already in use"}
    except sqlite3.OperationalError as ex:
        return {"ok": False, "error": str(ex)}


def delete_watchlist(db_path: str, watchlist_id: int) -> dict:
    try:
        with ingest_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                row = conn.execute(
                    "SELECT kind FROM watchlists WHERE id = ?", (watchlist_id,)
                ).fetchone()
                if not row:
                    return {"ok": False, "error": "Not found"}
                if row[0] == WATCHLIST_KIND_HOLDINGS:
                    return {"ok": False, "error": "Cannot delete the Holdings smart list"}
                conn.execute("DELETE FROM watchlist_members WHERE watchlist_id = ?", (watchlist_id,))
                conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))
                conn.commit()
        return {"ok": True}
    except sqlite3.OperationalError as ex:
        return {"ok": False, "error": str(ex)}


def list_members(db_path: str, watchlist_id: int) -> list[WatchlistMember]:
    ensure_watchlist_schema(db_path)
    try:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT watchlist_id, ticker, sort_order, added_at
                FROM watchlist_members WHERE watchlist_id = ?
                ORDER BY sort_order, ticker
                """,
                (watchlist_id,),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        WatchlistMember(watchlist_id=r[0], ticker=r[1], sort_order=r[2], added_at=r[3])
        for r in rows
    ]


def add_member(db_path: str, watchlist_id: int, ticker: str) -> dict:
    from datetime import datetime, timezone

    sym = str(ticker).strip().upper()
    if not sym:
        return {"ok": False, "error": "Ticker required"}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        with ingest_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                conn.execute(
                    """
                    INSERT INTO watchlist_members (watchlist_id, ticker, sort_order, added_at)
                    VALUES (?, ?, 0, ?)
                    ON CONFLICT(watchlist_id, ticker) DO NOTHING
                    """,
                    (watchlist_id, sym, now),
                )
                conn.commit()
        _register_watchlist_symbol(db_path, sym)
        return {"ok": True}
    except sqlite3.OperationalError as ex:
        return {"ok": False, "error": str(ex)}


def _register_watchlist_symbol(db_path: str, symbol: str) -> None:
    """Add symbol to the research universe registry so ingest can fetch prices."""
    try:
        from src.analysis.ticker_registry import POOL_UNIVERSE, add_symbol

        add_symbol(db_path, symbol, pool=POOL_UNIVERSE)
    except Exception:
        pass


def remove_member(db_path: str, watchlist_id: int, ticker: str) -> dict:
    sym = str(ticker).strip().upper()
    try:
        with ingest_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                conn.execute(
                    "DELETE FROM watchlist_members WHERE watchlist_id = ? AND ticker = ?",
                    (watchlist_id, sym),
                )
                conn.commit()
        return {"ok": True}
    except sqlite3.OperationalError as ex:
        return {"ok": False, "error": str(ex)}


def sync_holdings_watchlist(db_path: str, tickers: list[str]) -> None:
    """Update the Holdings smart list from portfolio tickers."""
    wl = get_watchlist_by_name(db_path, "Holdings")
    if not wl:
        return
    current = {m.ticker for m in list_members(db_path, wl.id)}
    desired = {str(t).strip().upper() for t in tickers if t}
    for sym in desired - current:
        add_member(db_path, wl.id, sym)
    for sym in current - desired:
        remove_member(db_path, wl.id, sym)
