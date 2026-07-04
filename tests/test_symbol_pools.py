"""Tests for focus watchlist vs research universe pools."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.analysis import ticker_registry as registry
from src.analysis.symbol_universe import resolve_universe
from src.services.stock_config import stock_config


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    registry.ensure_registry(path)
    registry.init_pool_schema(path)
    return path


def test_csv_import_sets_universe_pool(db_path: str, tmp_path: Path) -> None:
    csv_path = tmp_path / "tickers.csv"
    csv_path.write_text("AAA\nBBB\n", encoding="utf-8")
    result = registry.migrate_csv_to_db(db_path, str(csv_path))
    assert result.get("newly_added", 0) >= 2
    with sqlite3.connect(db_path) as conn:
        pools = {
            r[0]
            for r in conn.execute(
                "SELECT pool FROM watchlist_tickers WHERE symbol IN ('AAA','BBB')"
            ).fetchall()
        }
    assert pools == {"universe"}


def test_focus_cap_enforced(db_path: str) -> None:
    cap = stock_config().focus_watchlist_max
    for i in range(cap):
        result = registry.add_to_focus(db_path, f"T{i:02d}")
        assert result.get("ok"), result
    overflow = registry.add_to_focus(db_path, "OVER")
    assert not overflow.get("ok")
    assert "cap" in (overflow.get("error") or "").lower() or "20" in str(overflow.get("error", ""))


def test_remove_from_focus_keeps_universe(db_path: str) -> None:
    registry.add_symbols_bulk(db_path, ["KEEP"])
    registry.add_to_focus(db_path, "KEEP")
    registry.remove_from_focus(db_path, "KEEP")
    row = registry.get_symbol(db_path, "KEEP")
    assert row is not None
    assert row.pool == registry.POOL_UNIVERSE


def test_list_symbols_for_ingest_scopes(db_path: str) -> None:
    registry.add_symbols_bulk(db_path, ["U1", "U2"])
    registry.add_to_focus(db_path, "F1")
    focus = registry.list_symbols_for_ingest(db_path, "focus")
    universe = registry.list_symbols_for_ingest(db_path, "universe")
    assert focus == ["F1"]
    assert "F1" in universe
    assert "U1" in universe
    assert focus[0] == universe[0]  # focus-first ordering


def test_resolve_universe_focus(db_path: str) -> None:
    registry.add_to_focus(db_path, "ZX")
    tickers = resolve_universe(db_path, "focus")
    assert tickers == ["ZX"]


def test_count_summary_includes_focus(db_path: str) -> None:
    registry.add_to_focus(db_path, "A1")
    summary = registry.count_summary(db_path)
    assert summary["focus"] == 1
    assert summary["focus_cap"] == stock_config().focus_watchlist_max
