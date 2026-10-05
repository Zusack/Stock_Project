"""SQLite persistence for custom strategies and backtest run history."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.strategy_spec import StrategySpec


@dataclass
class SavedStrategy:
    id: int
    name: str
    spec: StrategySpec
    created_at: str
    updated_at: str


@dataclass
class SavedBacktestRun:
    id: int
    strategy_name: str
    spec_json: str
    metrics_json: str
    tickers_json: str
    lookback_days: int
    initial_capital: float
    created_at: str


def ensure_backtest_schema(db_path: str) -> None:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS custom_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    spec_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    metrics_json TEXT,
                    tickers_json TEXT,
                    lookback_days INTEGER,
                    initial_capital REAL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_backtest_runs_created
                    ON backtest_runs(created_at);
                """
            )
            conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def save_custom_strategy(db_path: str, spec: StrategySpec) -> int:
    ensure_backtest_schema(db_path)
    now = _now()
    payload = spec.to_json()
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            existing = conn.execute(
                "SELECT id FROM custom_strategies WHERE name = ?", (spec.name,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE custom_strategies SET spec_json = ?, updated_at = ? WHERE name = ?",
                    (payload, now, spec.name),
                )
                conn.commit()
                return int(existing[0])
            cur = conn.execute(
                """
                INSERT INTO custom_strategies (name, spec_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (spec.name, payload, now, now),
            )
            conn.commit()
            return int(cur.lastrowid)


def delete_custom_strategy(db_path: str, name: str) -> bool:
    ensure_backtest_schema(db_path)
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            cur = conn.execute("DELETE FROM custom_strategies WHERE name = ?", (name,))
            conn.commit()
            return cur.rowcount > 0


def list_custom_strategies(db_path: str) -> list[SavedStrategy]:
    ensure_backtest_schema(db_path)
    try:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                "SELECT id, name, spec_json, created_at, updated_at FROM custom_strategies ORDER BY name"
            ).fetchall()
    except sqlite3.Error:
        return []
    out: list[SavedStrategy] = []
    for row in rows:
        try:
            spec = StrategySpec.from_json(row[2])
        except (json.JSONDecodeError, TypeError):
            continue
        out.append(
            SavedStrategy(
                id=int(row[0]),
                name=str(row[1]),
                spec=spec,
                created_at=str(row[3]),
                updated_at=str(row[4]),
            )
        )
    return out


def save_backtest_run(
    db_path: str,
    *,
    spec: StrategySpec,
    metrics: dict[str, Any],
    tickers: list[str],
    lookback_days: int,
    initial_capital: float,
) -> int:
    ensure_backtest_schema(db_path)
    now = _now()
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            cur = conn.execute(
                """
                INSERT INTO backtest_runs
                (strategy_name, spec_json, metrics_json, tickers_json, lookback_days,
                 initial_capital, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.name,
                    spec.to_json(),
                    json.dumps(metrics),
                    json.dumps(tickers),
                    lookback_days,
                    initial_capital,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)


def list_backtest_runs(db_path: str, limit: int = 20) -> list[SavedBacktestRun]:
    ensure_backtest_schema(db_path)
    try:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT id, strategy_name, spec_json, metrics_json, tickers_json,
                       lookback_days, initial_capital, created_at
                FROM backtest_runs ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [
        SavedBacktestRun(
            id=int(r[0]),
            strategy_name=str(r[1]),
            spec_json=str(r[2]),
            metrics_json=str(r[3] or "{}"),
            tickers_json=str(r[4] or "[]"),
            lookback_days=int(r[5] or 365),
            initial_capital=float(r[6] or 10000),
            created_at=str(r[7]),
        )
        for r in rows
    ]
