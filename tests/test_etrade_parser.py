"""Tests for E-Trade / Morgan Stanley CLIENT STATEMENT PDF parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.statements.etrade_parser import (
    _parse_positions_from_lines,
    parse_etrade_statement,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ms_client_statement_sample.pdf"


def test_paren_ticker_positions_from_holdings_section():
    lines = [
        "Account Summary",
        "Stocks 6,725.00 1,870.00",  # summary noise — must not become a position
        "HOLDINGS",
        "STOCKS",
        "COMMON STOCKS",
        "Security Description Quantity Share Price Total Cost Market Value",
        "MIND MEDICINE MINDMED INC NEW (MNMD) 2,000.000 $3.660 $7,447.69 $7,320.00 $(127.69) — —",
        "Asset Class: Equities",
        "EXCHANGE-TRADED & CLOSED-END FUNDS",
        "ADVISORSHARES PSYCHEDELICS (PSIL) 2,000.000 $1.390 $11,462.05 $2,780.00 $(8,682.05) $6.72 0.24",
        "TOTAL VALUE 100.00% $17,774.69 $72,623.04",
        "ACTIVITY",
        "12/1 12/4 Sold CALL AMZN 12/01/23 148.000 ACTED AS AGENT 50.000 0.0500 241.80",
    ]
    positions = _parse_positions_from_lines(lines)
    by_sym = {p.symbol: p for p in positions}
    assert set(by_sym) == {"MNMD", "PSIL"}
    assert by_sym["MNMD"].quantity == 2000.0
    assert by_sym["MNMD"].price == pytest.approx(3.66)
    assert by_sym["MNMD"].cost_basis == pytest.approx(7447.69)
    assert by_sym["MNMD"].market_value == pytest.approx(7320.0)
    assert by_sym["MNMD"].asset_type == "stock"
    assert by_sym["PSIL"].quantity == 2000.0
    assert by_sym["PSIL"].market_value == pytest.approx(2780.0)
    assert by_sym["PSIL"].asset_type == "etf"


def test_summary_page_alone_yields_no_positions():
    lines = [
        "CLIENT STATEMENT",
        "ASSET ALLOCATION",
        "Cash $67,973.04 93.60",
        "Equities 4,650.00 6.40",
        "BALANCE SHEET",
        "Stocks 6,725.00 1,870.00",
        "ETFs & CEFs 2,570.00 2,780.00",
        "TOTAL VALUE $42,747.06 $72,623.04",
    ]
    assert _parse_positions_from_lines(lines) == []


def test_legacy_leading_ticker_inside_holdings():
    lines = [
        "ACCOUNT HOLDINGS",
        "SYMBOL DESCRIPTION QTY PRICE COST MARKET VALUE",
        "AAPL Apple Inc 10.000 190.00 1800.00 1900.00",
        "ACCOUNT ACTIVITY",
    ]
    positions = _parse_positions_from_lines(lines)
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == 10.0
    assert positions[0].market_value == pytest.approx(1900.0)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="sample statement fixture missing")
def test_sample_ms_client_statement_pdf():
    result = parse_etrade_statement(FIXTURE)
    assert result.variant == "client_statement"
    assert result.period_start == "12/1/23"
    assert result.period_end == "12/31/23"
    assert result.ending_value == pytest.approx(72623.04)

    by_sym = {p.symbol: p for p in result.positions}
    assert "MNMD" in by_sym
    assert "PSIL" in by_sym
    # Must not invent symbols from summary totals.
    assert "STOCKS" not in by_sym
    assert not any(s[0].isdigit() for s in by_sym)

    assert by_sym["MNMD"].quantity == pytest.approx(2000.0)
    assert by_sym["MNMD"].price == pytest.approx(3.66)
    assert by_sym["MNMD"].cost_basis == pytest.approx(7447.69)
    assert by_sym["MNMD"].market_value == pytest.approx(7320.0)

    assert by_sym["PSIL"].quantity == pytest.approx(2000.0)
    assert by_sym["PSIL"].market_value == pytest.approx(2780.0)

    assert len(result.activities) > 0
    assert any(a.action in ("bought", "sold") for a in result.activities)
    assert not result.warnings
