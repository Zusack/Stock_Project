"""Concurrent SQLite writes under ingest_write_lock (WAL)."""

from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path

from src.analysis.db import configure_connection, ingest_write_lock


def _writer(db_path: Path, worker_id: int, n: int, errors: list[str]) -> None:
    try:
        for i in range(n):
            with ingest_write_lock():
                conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
                configure_connection(conn, readonly=False)
                conn.execute(
                    "INSERT INTO stock_history (Ticker, Date, Open, High, Low, Close, Volume) "
                    "VALUES (?, ?, 1, 1, 1, 1, 100)",
                    (f"W{worker_id}", f"2020-01-{i + 1:02d}"),
                )
                conn.commit()
                conn.close()
    except Exception as e:
        errors.append(str(e))


def test_parallel_writes_with_ingest_lock() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn, readonly=False)
        conn.execute(
            """
            CREATE TABLE stock_history (
                Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL,
                Close REAL, Volume REAL,
                PRIMARY KEY (Ticker, Date)
            )
        """
        )
        conn.commit()
        conn.close()

        errors: list[str] = []
        threads = [
            threading.Thread(target=_writer, args=(db_path, wid, 20, errors), daemon=True)
            for wid in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive(), "writer thread hung"

        assert not errors, errors
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT COUNT(*) FROM stock_history").fetchone()
        conn.close()
        assert row is not None and row[0] == 80
