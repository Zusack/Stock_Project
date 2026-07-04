"""Bulk loader helpers."""

from __future__ import annotations

import sqlite3

from src.analysis.bulk_loaders import load_profile_map, load_sector_map


def test_load_profile_map_returns_sector_and_industry(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE stock_profiles (
                Ticker TEXT PRIMARY KEY, Sector TEXT, Industry TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO stock_profiles VALUES ('AAPL', 'Technology', 'Hardware')"
        )
        conn.commit()
    profiles = load_profile_map(str(db))
    assert profiles["AAPL"]["sector"] == "Technology"
    assert profiles["AAPL"]["industry"] == "Hardware"
    assert load_sector_map(str(db))["AAPL"] == "Technology"
