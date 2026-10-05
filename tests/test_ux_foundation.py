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


def test_quote_snapshot_change_vs_prev_close_not_open():
    """Gap-down then rally: up from open, still down vs previous close."""
    q = QuoteSnapshot(
        ticker="TEST",
        last_price=582.69,
        open=563.99,
        prev_close=587.97,
    )
    assert abs(q.change - (-5.28)) < 0.01
    assert abs(q.change_pct - (-0.90)) < 0.01
    assert abs(q.change_from_open - 18.70) < 0.01
    assert abs(q.change_from_open_pct - 3.32) < 0.01


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


def test_resolve_watchlist_symbols_prefers_saved_list(tmp_path):
    db = str(tmp_path / "test.db")
    from src.analysis.ticker_registry import ensure_registry, add_to_focus
    from src.analysis.watchlist_schema import (
        add_member,
        ensure_watchlist_schema,
        get_watchlist_by_name,
        resolve_watchlist_symbols,
    )

    ensure_registry(db)
    add_to_focus(db, "FOCUS1")
    ensure_watchlist_schema(db)
    wl = get_watchlist_by_name(db, "My Watchlist")
    assert wl is not None
    add_member(db, wl.id, "SAVED1")
    add_member(db, wl.id, "SAVED2")

    syms = resolve_watchlist_symbols(db)
    assert syms == ["SAVED1", "SAVED2"]


def test_resolve_watchlist_symbols_falls_back_to_focus(tmp_path):
    db = str(tmp_path / "test.db")
    from src.analysis.ticker_registry import ensure_registry, add_to_focus
    from src.analysis.watchlist_schema import ensure_watchlist_schema, resolve_watchlist_symbols

    ensure_registry(db)
    add_to_focus(db, "AAPL")
    add_to_focus(db, "MSFT")
    ensure_watchlist_schema(db)

    syms = resolve_watchlist_symbols(db)
    assert syms == ["AAPL", "MSFT"]


def test_resolve_watchlist_symbols_unions_named_lists(tmp_path):
    db = str(tmp_path / "test.db")
    from src.analysis.watchlist_schema import (
        add_member,
        create_watchlist,
        ensure_watchlist_schema,
        get_watchlist_by_name,
        resolve_watchlist_symbols,
    )

    ensure_watchlist_schema(db)
    my_wl = get_watchlist_by_name(db, "My Watchlist")
    assert my_wl is not None
    add_member(db, my_wl.id, "AAA")
    custom = create_watchlist(db, "Growth")
    assert custom.get("ok")
    add_member(db, int(custom["id"]), "BBB")
    add_member(db, my_wl.id, "CCC")

    syms = resolve_watchlist_symbols(db)
    assert syms == ["AAA", "CCC", "BBB"]


def test_add_member_registers_in_universe(tmp_path):
    db = str(tmp_path / "test.db")
    from src.analysis.ticker_registry import ensure_registry, get_symbol
    from src.analysis.watchlist_schema import (
        add_member,
        ensure_watchlist_schema,
        get_watchlist_by_name,
    )

    ensure_registry(db)
    ensure_watchlist_schema(db)
    wl = get_watchlist_by_name(db, "My Watchlist")
    assert wl is not None
    add_member(db, wl.id, "ZZZZ")
    row = get_symbol(db, "ZZZZ")
    assert row is not None
    assert row.pool == "universe"


def test_fill_unscored_watchlist_gaps():
    import pandas as pd

    from src.analysis.leaderboard import fill_unscored_watchlist_gaps

    df = pd.DataFrame([{"ticker": "AAPL", "composite_score": 80.0}])
    out = fill_unscored_watchlist_gaps(df, ["AAPL", "MSFT", "NVDA"])
    assert len(out) == 3
    assert set(out["ticker"]) == {"AAPL", "MSFT", "NVDA"}
    msft = out[out["ticker"] == "MSFT"].iloc[0]
    assert msft["notes"] == "Not scored"


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
