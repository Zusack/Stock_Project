"""Ensure load_price_data deduplicates duplicate dates."""

import pandas as pd

from src.analysis.db import load_price_data


def test_load_price_data_dedupes_index(tmp_path):
    import sqlite3

    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE stock_history (
            Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL,
            "Adj Close" REAL, Volume INTEGER
        )
    """
    )
    conn.executemany(
        'INSERT INTO stock_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        [
            ("AAPL", "2024-01-02", 1, 1, 1, 1, 100.0, 1000),
            ("AAPL", "2024-01-02", 1, 1, 1, 1, 101.0, 1100),
            ("AAPL", "2024-01-03", 1, 1, 1, 1, 102.0, 1200),
        ],
    )
    conn.commit()
    conn.close()

    df = load_price_data("AAPL", str(db))
    assert df is not None
    assert not df.index.has_duplicates
    assert len(df) == 2
    assert float(df.loc[pd.Timestamp("2024-01-02"), "Adj Close"]) == 101.0
