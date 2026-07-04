"""SQLite schema for cached quote snapshots (yfinance fast_info)."""

from __future__ import annotations

from src.analysis.db import db_connection, ingest_write_lock

QUOTES_TABLE = "quotes_snapshot"


def ensure_quotes_schema(db_path: str) -> None:
    """Create quotes_snapshot table if missing."""
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS quotes_snapshot (
                    Ticker TEXT PRIMARY KEY,
                    Last_Price REAL,
                    Prev_Close REAL,
                    Open REAL,
                    Day_High REAL,
                    Day_Low REAL,
                    Volume INTEGER,
                    Avg_Volume_10d REAL,
                    Avg_Volume_3mo REAL,
                    Market_Cap REAL,
                    Shares_Outstanding REAL,
                    FiftyTwo_Week_High REAL,
                    FiftyTwo_Week_Low REAL,
                    Trailing_PE REAL,
                    Forward_PE REAL,
                    EPS REAL,
                    Beta REAL,
                    Sector TEXT,
                    Industry TEXT,
                    Fifty_Day_Avg REAL,
                    Two_Hundred_Day_Avg REAL,
                    Fetched_At TEXT,
                    Extra_Json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_quotes_fetched ON quotes_snapshot(Fetched_At);
                """
            )
            conn.commit()
