"""Tests for ai_insights persistence."""

from __future__ import annotations

import sqlite3
import tempfile

from src.analysis.intelligence_schema import (
    ensure_intelligence_schema,
    load_latest_ai_insight,
    save_ai_insight,
)


def test_ai_insights_round_trip():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_path = tmp.name
        ensure_intelligence_schema(db_path)
        save_ai_insight(
            db_path,
            ticker="MSFT",
            provider="lm_studio",
            model="local-model",
            kind="ticker_narrative",
            payload={"headline": "Watch"},
            source_context_hash="abc123",
        )
        row = load_latest_ai_insight(db_path, "MSFT", kind="ticker_narrative")
        assert row is not None
        assert row["ticker"] == "MSFT"
        assert row["payload"]["headline"] == "Watch"
        assert row["source_context_hash"] == "abc123"

        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM ai_insights").fetchone()[0]
        assert count == 1
