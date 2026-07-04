"""User portfolio schema: accounts, holdings, trades, statements, value history."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.watchlist_schema import sync_holdings_watchlist

ACCOUNT_TYPE_RETIREMENT = "retirement"
ACCOUNT_TYPE_WEALTH = "wealth_building"


@dataclass
class Account:
    id: int
    name: str
    broker: str
    account_type: str
    created_at: str


@dataclass
class Holding:
    id: int
    account_id: int
    ticker: str
    quantity: float
    avg_cost_basis: float
    source: str
    updated_at: str


@dataclass
class Trade:
    id: int
    account_id: int
    ticker: str
    trade_date: str
    action: str
    quantity: float
    price: float
    amount: float
    fees: float
    source: str
    statement_id: int | None


@dataclass
class Statement:
    id: int
    account_id: int
    period_start: str
    period_end: str
    file_name: str
    imported_at: str
    ending_value: float | None


@dataclass
class AccountValuePoint:
    account_id: int
    as_of_date: str
    total_value: float
    source: str


def ensure_portfolio_schema(db_path: str) -> None:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    broker TEXT NOT NULL DEFAULT 'E-Trade',
                    account_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_accounts_name ON accounts(name);

                CREATE TABLE IF NOT EXISTS holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_cost_basis REAL NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'manual',
                    updated_at TEXT NOT NULL,
                    UNIQUE(account_id, ticker),
                    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL DEFAULT 0,
                    amount REAL NOT NULL DEFAULT 0,
                    fees REAL NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'manual',
                    statement_id INTEGER,
                    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE,
                    FOREIGN KEY (statement_id) REFERENCES statements(id)
                );
                CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id, trade_date);

                CREATE TABLE IF NOT EXISTS statements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    period_start TEXT,
                    period_end TEXT,
                    file_name TEXT,
                    imported_at TEXT NOT NULL,
                    ending_value REAL,
                    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS account_value_history (
                    account_id INTEGER NOT NULL,
                    as_of_date TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    PRIMARY KEY (account_id, as_of_date),
                    FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
                );
                """
            )
            conn.commit()
    _seed_default_accounts(db_path)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _seed_default_accounts(db_path: str) -> None:
    with db_connection(db_path, readonly=False) as conn:
        count = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()
        if count and count[0] > 0:
            return
        now = _now()
        defaults = [
            ("Retirement", "E-Trade", ACCOUNT_TYPE_RETIREMENT),
            ("Wealth Building", "E-Trade", ACCOUNT_TYPE_WEALTH),
        ]
        for name, broker, acct_type in defaults:
            conn.execute(
                "INSERT INTO accounts (name, broker, account_type, created_at) VALUES (?,?,?,?)",
                (name, broker, acct_type, now),
            )
        conn.commit()


def list_accounts(db_path: str) -> list[Account]:
    ensure_portfolio_schema(db_path)
    try:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                "SELECT id, name, broker, account_type, created_at FROM accounts ORDER BY id"
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [Account(id=r[0], name=r[1], broker=r[2], account_type=r[3], created_at=r[4]) for r in rows]


def get_account(db_path: str, account_id: int) -> Account | None:
    for acct in list_accounts(db_path):
        if acct.id == account_id:
            return acct
    return None


def list_holdings(db_path: str, account_id: int | None = None) -> list[Holding]:
    ensure_portfolio_schema(db_path)
    try:
        with db_connection(db_path, readonly=True) as conn:
            if account_id is not None:
                rows = conn.execute(
                    """
                    SELECT id, account_id, ticker, quantity, avg_cost_basis, source, updated_at
                    FROM holdings WHERE account_id = ? ORDER BY ticker
                    """,
                    (account_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, account_id, ticker, quantity, avg_cost_basis, source, updated_at
                    FROM holdings ORDER BY account_id, ticker
                    """
                ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        Holding(
            id=r[0],
            account_id=r[1],
            ticker=r[2],
            quantity=r[3],
            avg_cost_basis=r[4],
            source=r[5],
            updated_at=r[6],
        )
        for r in rows
    ]


def upsert_holding(
    db_path: str,
    account_id: int,
    ticker: str,
    quantity: float,
    avg_cost_basis: float,
    *,
    source: str = "manual",
) -> dict:
    sym = str(ticker).strip().upper()
    now = _now()
    try:
        with ingest_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                conn.execute(
                    """
                    INSERT INTO holdings (account_id, ticker, quantity, avg_cost_basis, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_id, ticker) DO UPDATE SET
                        quantity=excluded.quantity,
                        avg_cost_basis=excluded.avg_cost_basis,
                        source=excluded.source,
                        updated_at=excluded.updated_at
                    """,
                    (account_id, sym, quantity, avg_cost_basis, source, now),
                )
                conn.commit()
        _sync_all_holdings_watchlist(db_path)
        return {"ok": True}
    except sqlite3.OperationalError as ex:
        return {"ok": False, "error": str(ex)}


def delete_holding(db_path: str, holding_id: int) -> dict:
    try:
        with ingest_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                conn.execute("DELETE FROM holdings WHERE id = ?", (holding_id,))
                conn.commit()
        _sync_all_holdings_watchlist(db_path)
        return {"ok": True}
    except sqlite3.OperationalError as ex:
        return {"ok": False, "error": str(ex)}


def add_trade(
    db_path: str,
    account_id: int,
    ticker: str,
    trade_date: str,
    action: str,
    quantity: float,
    price: float,
    *,
    amount: float | None = None,
    fees: float = 0.0,
    source: str = "manual",
    statement_id: int | None = None,
) -> dict:
    sym = str(ticker).strip().upper()
    amt = amount if amount is not None else quantity * price
    try:
        with ingest_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                conn.execute(
                    """
                    INSERT INTO trades (
                        account_id, ticker, trade_date, action, quantity, price,
                        amount, fees, source, statement_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (account_id, sym, trade_date, action, quantity, price, amt, fees, source, statement_id),
                )
                conn.commit()
        return {"ok": True}
    except sqlite3.OperationalError as ex:
        return {"ok": False, "error": str(ex)}


def list_trades(db_path: str, account_id: int, *, limit: int = 200) -> list[Trade]:
    ensure_portfolio_schema(db_path)
    try:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT id, account_id, ticker, trade_date, action, quantity, price,
                       amount, fees, source, statement_id
                FROM trades WHERE account_id = ?
                ORDER BY trade_date DESC, id DESC LIMIT ?
                """,
                (account_id, limit),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        Trade(
            id=r[0],
            account_id=r[1],
            ticker=r[2],
            trade_date=r[3],
            action=r[4],
            quantity=r[5],
            price=r[6],
            amount=r[7],
            fees=r[8],
            source=r[9],
            statement_id=r[10],
        )
        for r in rows
    ]


def record_account_value(
    db_path: str,
    account_id: int,
    as_of_date: str,
    total_value: float,
    *,
    source: str = "manual",
) -> None:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                INSERT INTO account_value_history (account_id, as_of_date, total_value, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(account_id, as_of_date) DO UPDATE SET
                    total_value=excluded.total_value,
                    source=excluded.source
                """,
                (account_id, as_of_date, total_value, source),
            )
            conn.commit()


def list_account_value_history(db_path: str, account_id: int) -> list[AccountValuePoint]:
    ensure_portfolio_schema(db_path)
    try:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT account_id, as_of_date, total_value, source
                FROM account_value_history WHERE account_id = ?
                ORDER BY as_of_date
                """,
                (account_id,),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [AccountValuePoint(account_id=r[0], as_of_date=r[1], total_value=r[2], source=r[3]) for r in rows]


def portfolio_summary(db_path: str, account_id: int | None = None) -> dict:
    """Aggregate holdings count and cost basis by account."""
    holdings = list_holdings(db_path, account_id)
    by_account: dict[int, dict] = {}
    for h in holdings:
        bucket = by_account.setdefault(
            h.account_id,
            {"positions": 0, "cost_basis": 0.0, "tickers": []},
        )
        bucket["positions"] += 1
        bucket["cost_basis"] += h.quantity * h.avg_cost_basis
        bucket["tickers"].append(h.ticker)
    return by_account


def _sync_all_holdings_watchlist(db_path: str) -> None:
    tickers = list({h.ticker for h in list_holdings(db_path)})
    sync_holdings_watchlist(db_path, tickers)


def import_statement_holdings(
    db_path: str,
    account_id: int,
    *,
    file_name: str,
    period_start: str,
    period_end: str,
    positions: list[dict],
    activities: list[dict],
    ending_value: float | None = None,
) -> dict:
    """Persist confirmed statement import."""
    now = _now()
    try:
        with ingest_write_lock():
            with db_connection(db_path, readonly=False) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO statements (account_id, period_start, period_end, file_name, imported_at, ending_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (account_id, period_start, period_end, file_name, now, ending_value),
                )
                statement_id = cur.lastrowid
                for pos in positions:
                    sym = str(pos.get("symbol", "")).strip().upper()
                    if not sym:
                        continue
                    qty = float(pos.get("quantity", 0))
                    cost = float(pos.get("cost_basis", 0) or 0)
                    price = float(pos.get("price", 0) or 0)
                    avg = cost / qty if qty else price
                    conn.execute(
                        """
                        INSERT INTO holdings (account_id, ticker, quantity, avg_cost_basis, source, updated_at)
                        VALUES (?, ?, ?, ?, 'statement', ?)
                        ON CONFLICT(account_id, ticker) DO UPDATE SET
                            quantity=excluded.quantity,
                            avg_cost_basis=excluded.avg_cost_basis,
                            source='statement',
                            updated_at=excluded.updated_at
                        """,
                        (account_id, sym, qty, avg, now),
                    )
                for act in activities:
                    sym = str(act.get("symbol", "")).strip().upper()
                    if not sym:
                        continue
                    conn.execute(
                        """
                        INSERT INTO trades (
                            account_id, ticker, trade_date, action, quantity, price,
                            amount, fees, source, statement_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'statement', ?)
                        """,
                        (
                            account_id,
                            sym,
                            act.get("date", period_end),
                            act.get("action", "unknown"),
                            float(act.get("quantity", 0)),
                            float(act.get("price", 0) or 0),
                            float(act.get("amount", 0) or 0),
                            float(act.get("fees", 0) or 0),
                            statement_id,
                        ),
                    )
                if ending_value is not None and period_end:
                    conn.execute(
                        """
                        INSERT INTO account_value_history (account_id, as_of_date, total_value, source)
                        VALUES (?, ?, ?, 'statement')
                        ON CONFLICT(account_id, as_of_date) DO UPDATE SET
                            total_value=excluded.total_value,
                            source='statement'
                        """,
                        (account_id, period_end, ending_value),
                    )
                conn.commit()
        _sync_all_holdings_watchlist(db_path)
        return {"ok": True, "statement_id": statement_id}
    except sqlite3.OperationalError as ex:
        return {"ok": False, "error": str(ex)}
