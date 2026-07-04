"""Assemble structured ticker context for LLM analysis."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from src.analysis.db import get_profile, load_ohlcv
from src.analysis.leaderboard import score_ticker
from src.analysis.market_context import load_latest_market_context
from src.analysis.quote_snapshot import get_quote, quote_to_dict
from src.services.stock_config import stock_config

if TYPE_CHECKING:
    from src.analysis.guidance import GuidanceRow


def _row_to_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if hasattr(row, "__dict__"):
        return {k: v for k, v in row.__dict__.items() if not k.startswith("_")}
    return None


def assemble_ticker_context(
    ticker: str,
    db_path: str | None = None,
    *,
    market_ticker: str | None = None,
    headline_limit: int = 8,
    canslim_score: int | None = None,
    canslim_max: int | None = None,
) -> dict[str, Any]:
    """Build a single JSON-serializable payload for LLM/advisor consumption."""
    cfg = stock_config()
    db = db_path or cfg.db_path
    sym = str(ticker).strip().upper()
    mkt = market_ticker or cfg.market_ticker

    market_ctx = load_latest_market_context(db)
    regime = market_ctx.regime if market_ctx else "neutral"

    from src.analysis.guidance import score_guidance_row
    from src.analysis.news_signals import load_recent_headlines, score_ticker_news

    lb_row = score_ticker(sym, db, mkt, market_regime=regime)
    guidance_row: GuidanceRow | None = None
    if lb_row is not None:
        guidance_row = score_guidance_row(
            lb_row,
            db,
            market_regime=regime,
            canslim_score=canslim_score,
            canslim_max=canslim_max,
        )

    headlines = load_recent_headlines(db, sym, limit=headline_limit)
    sent, risk_tags, conf, catalysts = score_ticker_news(db, sym, headlines=headlines)
    quote = get_quote(db, sym)
    profile = get_profile(db, sym)
    ohlcv = load_ohlcv(sym, db)
    ohlcv_summary: dict[str, Any] = {}
    if ohlcv is not None and not ohlcv.empty:
        latest = ohlcv.iloc[-1]
        ohlcv_summary = {
            "bars": len(ohlcv),
            "latest_date": str(ohlcv.index[-1])[:10],
            "close": float(latest.get("Close", 0) or 0),
            "volume": float(latest.get("Volume", 0) or 0),
        }

    payload: dict[str, Any] = {
        "ticker": sym,
        "market_regime": regime,
        "market_context": _row_to_dict(market_ctx),
        "leaderboard": _row_to_dict(lb_row),
        "guidance": _row_to_dict(guidance_row),
        "quote": quote_to_dict(quote) if quote else None,
        "profile": profile,
        "ohlcv_summary": ohlcv_summary,
        "news": {
            "headlines": headlines,
            "sentiment": sent,
            "risk_tags": risk_tags,
            "confidence": conf,
            "catalyst_tags": catalysts,
        },
        "canslim_note": (
            f"CANSLIM score is out of {canslim_max} (interactive Stock Detail uses 7 "
            "letters including Institutions; leaderboard rankings use 6: C,A,N,S,L,M)."
            if canslim_max is not None
            else (
                "Leaderboard CANSLIM uses 6 pass columns (C,A,N,S,L,M); "
                "interactive analyze_canslim scores 7 letters including Institutions (I)."
            )
        ),
    }
    payload["context_hash"] = context_hash(payload)
    return payload


def context_hash(payload: dict[str, Any]) -> str:
    """Stable hash for cache invalidation of AI insights."""
    stable = {k: v for k, v in payload.items() if k != "context_hash"}
    raw = json.dumps(stable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_guidance_for_ticker(db_path: str, ticker: str) -> GuidanceRow | None:
    """Load latest guidance row for a ticker from the most recent scan."""
    from src.analysis.guidance import GuidanceRow, load_guidance_scan

    df = load_guidance_scan(db_path)
    if df is None or df.empty:
        return None
    sym = str(ticker).strip().upper()
    rows = df[df["ticker"].astype(str).str.upper() == sym]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return GuidanceRow(
        ticker=sym,
        composite_score=float(row.get("composite_score", 0) or 0),
        confidence=float(row.get("confidence", 0) or 0),
        recommendation_band=str(row.get("recommendation_band", "monitor")),
        canslim_score=int(row.get("canslim_score", 0) or 0),
        news_sentiment=float(row.get("news_sentiment", 0.5) or 0.5),
        regime_alignment=float(row.get("regime_alignment", 0) or 0),
        positive_drivers=[],
        negative_drivers=[],
        risk_notes=str(row.get("risk_notes", "") or ""),
    )
