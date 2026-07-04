"""Market regime, breadth, and sector context for guidance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from src.analysis.db import db_connection, ingest_write_lock, load_price_data, load_prices_bulk
from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.analysis.market_calendar import last_completed_trading_day
from src.services.stock_config import stock_config

# Sector ETF proxies (US equities + ETFs scope)
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial": "XLF",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

MARKET_INDICES: tuple[str, ...] = ("^GSPC", "^DJI", "^IXIC")
VIX_PROXY = "^VIX"


@dataclass
class MarketContextSnapshot:
    as_of_date: str
    market_ticker: str
    regime: str  # risk_on | neutral | risk_off
    trend_score: float
    breadth_pct: float
    vix_proxy_ticker: str
    vix_level: float | None
    sector_leaders: list[str]
    sector_laggards: list[str]
    payload: dict[str, Any]

    def to_row(self) -> tuple:
        return (
            self.as_of_date,
            self.market_ticker,
            self.regime,
            self.trend_score,
            self.breadth_pct,
            self.vix_proxy_ticker,
            self.vix_level,
            json.dumps(self.sector_leaders),
            json.dumps(self.sector_laggards),
            json.dumps(self.payload),
        )


def _trend_score_from_prices(df: pd.DataFrame, *, sma_days: int = 50) -> float:
    if df is None or len(df) < sma_days + 5:
        return 0.5
    close = df["Adj Close"].astype(float)
    sma = close.rolling(sma_days).mean()
    last = float(close.iloc[-1])
    sma_last = float(sma.iloc[-1])
    if sma_last <= 0:
        return 0.5
    dist = (last / sma_last) - 1.0
    ret_3m = float(close.iloc[-1] / close.iloc[-min(63, len(close) - 1)] - 1) if len(close) > 63 else 0.0
    score = 0.5 + min(0.25, max(-0.25, dist * 2)) + min(0.15, max(-0.15, ret_3m))
    return max(0.0, min(1.0, score))


def _compute_breadth(db_path: str, *, lookback_days: int = 126, max_symbols: int = 500) -> float:
    """Fraction of research-universe sample above 50-day SMA (proxy advance/decline)."""
    from src.analysis import ticker_registry as registry

    rows = registry.list_symbols(
        db_path, include_archived=False, pool=registry.POOL_UNIVERSE, limit=max_symbols
    )
    symbols = [r.symbol for r in rows if not r.skip_ingest][:max_symbols]
    if not symbols:
        return 0.5
    bulk = load_prices_bulk(
        symbols,
        db_path,
        columns='Ticker, Date, "Adj Close"',
        max_days=lookback_days + 60,
    )
    above = 0
    total = 0
    for sym in symbols:
        df = bulk.get(sym)
        if df is None or len(df) < 55:
            continue
        close = df["Adj Close"].astype(float)
        sma50 = close.rolling(50).mean().iloc[-1]
        if pd.isna(sma50):
            continue
        total += 1
        if float(close.iloc[-1]) > float(sma50):
            above += 1
    return above / total if total else 0.5


def _sector_etf_returns(db_path: str, *, days: int = 63) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for sector, etf in SECTOR_ETF_MAP.items():
        df = load_price_data(etf, db_path)
        if df is None or len(df) < days + 2:
            continue
        close = df["Adj Close"].astype(float)
        ret = float(close.iloc[-1] / close.iloc[-days] - 1)
        out.append((sector, ret))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def build_market_context(
    db_path: str,
    *,
    market_ticker: str | None = None,
    as_of: date | None = None,
) -> MarketContextSnapshot:
    cfg = stock_config()
    mkt = market_ticker or cfg.market_ticker
    as_of = as_of or last_completed_trading_day()
    as_of_str = as_of.isoformat()

    mdf = load_price_data(mkt, db_path)
    trend = _trend_score_from_prices(mdf)
    breadth = _compute_breadth(db_path)

    vix_df = load_price_data(VIX_PROXY, db_path)
    vix_level = float(vix_df["Adj Close"].iloc[-1]) if vix_df is not None and not vix_df.empty else None

    sector_rets = _sector_etf_returns(db_path)
    leaders = [s for s, _ in sector_rets[:3]]
    laggards = [s for s, _ in sector_rets[-3:]] if len(sector_rets) >= 3 else []

    if trend >= 0.58 and breadth >= 0.52:
        regime = "risk_on"
    elif trend <= 0.42 or breadth <= 0.38 or (vix_level is not None and vix_level > 28):
        regime = "risk_off"
    else:
        regime = "neutral"

    index_snap: dict[str, float] = {}
    for idx in MARKET_INDICES:
        idf = load_price_data(idx, db_path)
        if idf is not None and len(idf) >= 2:
            index_snap[idx] = round(float(idf["Adj Close"].iloc[-1]), 2)

    payload = {
        "indices": index_snap,
        "sector_returns_63d": {s: round(r, 4) for s, r in sector_rets},
        "vix_note": "High VIX favors defensive posture" if vix_level and vix_level > 25 else "",
    }

    return MarketContextSnapshot(
        as_of_date=as_of_str,
        market_ticker=mkt,
        regime=regime,
        trend_score=round(trend, 4),
        breadth_pct=round(breadth, 4),
        vix_proxy_ticker=VIX_PROXY,
        vix_level=round(vix_level, 2) if vix_level is not None else None,
        sector_leaders=leaders,
        sector_laggards=laggards,
        payload=payload,
    )


def persist_market_context(db_path: str, snap: MarketContextSnapshot) -> None:
    ensure_intelligence_schema(db_path)
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO market_context_daily
                (as_of_date, market_ticker, regime, trend_score, breadth_pct,
                 vix_proxy_ticker, vix_level, sector_leaders, sector_laggards, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                snap.to_row(),
            )
            conn.commit()


def load_latest_market_context(db_path: str) -> MarketContextSnapshot | None:
    ensure_intelligence_schema(db_path)
    with db_connection(db_path, readonly=True) as conn:
        row = conn.execute(
            """
            SELECT as_of_date, market_ticker, regime, trend_score, breadth_pct,
                   vix_proxy_ticker, vix_level, sector_leaders, sector_laggards, payload_json
            FROM market_context_daily ORDER BY as_of_date DESC LIMIT 1
            """
        ).fetchone()
    if not row:
        return None
    return MarketContextSnapshot(
        as_of_date=row[0],
        market_ticker=row[1],
        regime=row[2],
        trend_score=float(row[3]),
        breadth_pct=float(row[4]),
        vix_proxy_ticker=row[5],
        vix_level=float(row[6]) if row[6] is not None else None,
        sector_leaders=json.loads(row[7] or "[]"),
        sector_laggards=json.loads(row[8] or "[]"),
        payload=json.loads(row[9] or "{}"),
    )


def regime_alignment_score(regime: str, *, pass_setup: bool, risk_flag: str) -> float:
    """How well a ticker setup aligns with current market regime (0-1)."""
    risk_off = regime == "risk_off"
    risk_on = regime == "risk_on"
    if risk_off:
        return 0.25 if risk_flag else 0.4
    if risk_on and pass_setup and not risk_flag:
        return 0.9
    if risk_on:
        return 0.65
    return 0.55 if pass_setup else 0.45
