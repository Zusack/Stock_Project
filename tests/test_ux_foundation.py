"""Tests for quote snapshot and portfolio schema."""

from src.analysis.portfolio_schema import ensure_portfolio_schema, list_accounts, upsert_holding, list_holdings
from src.analysis.quotes_schema import ensure_quotes_schema
from src.analysis.quote_snapshot import QuoteSnapshot, get_quotes_bulk
from src.analysis.watchlist_schema import ensure_watchlist_schema, list_watchlists
from src.utils.format_utils import format_large_number, format_range


def test_quote_snapshot_change_pct():
    q = QuoteSnapshot(ticker="AAPL", last_price=110.0, prev_close=100.0)
    assert q.change == 10.0
    assert abs(q.change_pct - 10.0) < 0.01


def test_format_large_number():
    assert format_large_number(1_500_000_000) == "1.50B"
    assert format_large_number(None) == "—"


def test_format_range():
    assert "–" in format_range(10.0, 20.0)


def test_portfolio_schema_seeds_accounts(tmp_path):
    db = str(tmp_path / "test.db")
    ensure_portfolio_schema(db)
    accounts = list_accounts(db)
    names = {a.name for a in accounts}
    assert "Retirement" in names
    assert "Wealth Building" in names


def test_watchlist_schema_seeds_defaults(tmp_path):
    db = str(tmp_path / "test.db")
    ensure_watchlist_schema(db)
    names = {w.name for w in list_watchlists(db)}
    assert "My Watchlist" in names
    assert "Holdings" in names


def test_upsert_holding(tmp_path):
    db = str(tmp_path / "test.db")
    ensure_portfolio_schema(db)
    acct = list_accounts(db)[0]
    upsert_holding(db, acct.id, "AAPL", 10, 150.0)
    holdings = list_holdings(db, acct.id)
    assert any(h.ticker == "AAPL" and h.quantity == 10 for h in holdings)


def test_sync_focus_to_default_watchlist(tmp_path):
    db = str(tmp_path / "test.db")
    from src.analysis.ticker_registry import ensure_registry, add_to_focus
    from src.analysis.watchlist_schema import (
        ensure_watchlist_schema,
        list_members,
        get_watchlist_by_name,
    )

    ensure_registry(db)
    add_to_focus(db, "AAPL")
    add_to_focus(db, "MSFT")
    ensure_watchlist_schema(db)
    from src.analysis.watchlist_schema import sync_focus_to_default_watchlist

    sync_focus_to_default_watchlist(db)
    wl = get_watchlist_by_name(db, "My Watchlist")
    assert wl is not None
    members = list_members(db, wl.id)
    assert len(members) == 2
    assert {m.ticker for m in members} == {"AAPL", "MSFT"}


def test_get_quotes_bulk_argument_order(tmp_path):
    """db_path must be first arg — reversed args raise TypeError."""
    import pytest

    db = str(tmp_path / "test.db")
    ensure_quotes_schema(db)
    with pytest.raises(TypeError):
        get_quotes_bulk(["AAPL"], db)  # wrong order


def test_quotes_schema_creates_table(tmp_path):
    db = str(tmp_path / "test.db")
    ensure_quotes_schema(db)
    import sqlite3

    conn = sqlite3.connect(db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='quotes_snapshot'"
    ).fetchone()
    conn.close()
    assert tables is not None
