"""Conservative ranked guidance with explainability and change detection."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.analysis.leaderboard import LeaderboardRow, build_leaderboard, score_ticker
from src.analysis.canslim_core import CANSLM_RULE_COUNT, CANSLIM_RULE_COUNT
from src.analysis.market_context import (
    build_market_context,
    load_latest_market_context,
    persist_market_context,
    regime_alignment_score,
)
from src.analysis.news_signals import score_ticker_news
from src.services.stock_config import stock_config


class RecommendationBand(str, Enum):
    MONITOR = "monitor"
    CANDIDATE = "candidate"
    HIGH_PRIORITY = "high_priority"
    RISK_REVIEW = "risk_review"
    EXIT_WATCH = "exit_watch"


BAND_ORDER = {
    RecommendationBand.EXIT_WATCH: 0,
    RecommendationBand.RISK_REVIEW: 1,
    RecommendationBand.MONITOR: 2,
    RecommendationBand.CANDIDATE: 3,
    RecommendationBand.HIGH_PRIORITY: 4,
}


@dataclass
class GuidanceRow:
    ticker: str
    composite_score: float
    confidence: float
    recommendation_band: str
    canslim_score: int
    news_sentiment: float
    regime_alignment: float
    positive_drivers: list[str] = field(default_factory=list)
    negative_drivers: list[str] = field(default_factory=list)
    risk_notes: str = ""
    prior_band: str | None = None
    band_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "composite_score": self.composite_score,
            "confidence": self.confidence,
            "recommendation_band": self.recommendation_band,
            "canslim_score": self.canslim_score,
            "news_sentiment": self.news_sentiment,
            "regime_alignment": self.regime_alignment,
            "positive_drivers": self.positive_drivers,
            "negative_drivers": self.negative_drivers,
            "risk_notes": self.risk_notes,
            "prior_band": self.prior_band,
            "band_changed": self.band_changed,
        }


def _assign_band(
    lb: LeaderboardRow,
    *,
    news_sent: float,
    regime_align: float,
    market_regime: str,
) -> RecommendationBand:
    sell_p = getattr(lb, "sell_pressure", 0.0) or 0.0
    buy_r = getattr(lb, "buy_readiness", 0.0) or 0.0
    buy_signal = buy_r if buy_r > 0 else lb.composite_score
    if sell_p >= 65 or (
        lb.risk_flag and "Below 50-day simple moving average" in lb.risk_flag
    ):
        return RecommendationBand.EXIT_WATCH
    if lb.risk_flag and market_regime == "risk_off":
        return RecommendationBand.RISK_REVIEW
    if lb.risk_flag or sell_p >= 45:
        return RecommendationBand.RISK_REVIEW
    if lb.pass_setup and lb.pass_pattern and buy_signal >= 72 and regime_align >= 0.6:
        return RecommendationBand.HIGH_PRIORITY
    if lb.pass_setup and (buy_signal >= 58 or lb.composite_score >= 58):
        return RecommendationBand.CANDIDATE
    if news_sent < 0.35:
        return RecommendationBand.RISK_REVIEW
    return RecommendationBand.MONITOR


def _build_confidence(
    lb: LeaderboardRow,
    *,
    news_sent: float,
    regime_align: float,
    canslim_score: int,
    canslim_max: int = CANSLM_RULE_COUNT,
) -> float:
    base = lb.composite_score / 100.0
    denom = float(canslim_max) if canslim_max > 0 else float(CANSLM_RULE_COUNT)
    canslim_boost = canslim_score / denom * 0.15
    news_adj = (news_sent - 0.5) * 0.1
    regime_adj = (regime_align - 0.5) * 0.12
    conf = base * 0.65 + canslim_boost + news_adj + regime_adj
    if lb.risk_flag:
        conf *= 0.75
    return round(max(0.05, min(0.95, conf)), 3)


def _drivers(
    lb: LeaderboardRow,
    *,
    news_sent: float,
    news_tags: list[str],
    regime_align: float,
    market_regime: str,
    canslim_score: int,
    canslim_max: int = CANSLM_RULE_COUNT,
) -> tuple[list[str], list[str], str]:
    pos: list[str] = []
    neg: list[str] = []
    if lb.pass_setup:
        pos.append("CANSLM setup passes")
    if lb.pass_pattern:
        pos.append("Cup-with-handle pattern")
    if lb.rs_pct > 5:
        pos.append(f"Relative strength +{lb.rs_pct:.1f}%")
    if lb.volume_ratio >= 1.2:
        pos.append(f"Volume {lb.volume_ratio:.1f}x 50-day avg")
    if regime_align >= 0.65:
        pos.append(f"Aligned with {market_regime} regime")
    if news_sent >= 0.6:
        pos.append("News sentiment supportive")

    if lb.risk_flag:
        neg.append(lb.risk_flag)
    if canslim_score < 4:
        neg.append(f"Low CANSLM score ({canslim_score}/{canslim_max})")
    if news_sent < 0.4:
        neg.append("Negative news tone")
    if lb.near_high_pct < 75 and lb.pass_setup:
        neg.append(f"Below 75% of 52-week high ({lb.near_high_pct:.0f}%)")
    neg.extend(news_tags[:3])

    risk_note = (
        "Decision support only — confirm with your own research before trading."
    )
    if lb.risk_flag:
        risk_note = f"{lb.risk_flag}. {risk_note}"
    return pos, neg, risk_note


def score_guidance_row(
    lb: LeaderboardRow,
    db_path: str,
    *,
    market_regime: str = "neutral",
    prior_band: str | None = None,
    canslim_score: int | None = None,
    canslim_max: int | None = None,
) -> GuidanceRow:
    """Score guidance for a leaderboard row.

    Optional ``canslim_score`` / ``canslim_max`` override the leaderboard row when
    Stock Detail has a fresher evaluation (same 6-letter CANSLM rules).
    """
    score = lb.canslim_score if canslim_score is None else int(canslim_score)
    max_score = CANSLM_RULE_COUNT if canslim_max is None else int(canslim_max)
    if lb.news_tags is not None:
        news_sent = float(lb.news_sentiment)
        news_tags = list(lb.news_tags)
    else:
        news_sent, news_tags, _, _ = score_ticker_news(db_path, lb.ticker)
    regime_align = regime_alignment_score(
        market_regime, pass_setup=lb.pass_setup, risk_flag=lb.risk_flag
    )
    band = _assign_band(lb, news_sent=news_sent, regime_align=regime_align, market_regime=market_regime)
    pos, neg, risk_note = _drivers(
        lb,
        news_sent=news_sent,
        news_tags=news_tags,
        regime_align=regime_align,
        market_regime=market_regime,
        canslim_score=score,
        canslim_max=max_score,
    )
    conf = _build_confidence(
        lb,
        news_sent=news_sent,
        regime_align=regime_align,
        canslim_score=score,
        canslim_max=max_score,
    )
    changed = prior_band is not None and prior_band != band.value
    return GuidanceRow(
        ticker=lb.ticker,
        composite_score=lb.composite_score,
        confidence=conf,
        recommendation_band=band.value,
        canslim_score=score,
        news_sentiment=news_sent,
        regime_alignment=round(regime_align, 3),
        positive_drivers=pos,
        negative_drivers=neg,
        risk_notes=risk_note,
        prior_band=prior_band,
        band_changed=changed,
    )


def _prior_bands(db_path: str, prior_scan_id: str | None) -> dict[str, str]:
    if not prior_scan_id:
        return {}
    ensure_intelligence_schema(db_path)
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(
            "SELECT ticker, recommendation_band FROM guidance_snapshots WHERE scan_id = ?",
            (prior_scan_id,),
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def _latest_scan_id(db_path: str) -> str | None:
    ensure_intelligence_schema(db_path)
    with db_connection(db_path, readonly=True) as conn:
        row = conn.execute(
            "SELECT scan_id FROM guidance_snapshots ORDER BY scanned_at DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None


def run_guidance_scan(
    db_path: str,
    *,
    tickers: list[str] | None = None,
    market_ticker: str | None = None,
    limit: int = 200,
    use_parallel: bool = True,
) -> tuple[str, list[GuidanceRow]]:
    """Score universe, persist snapshots, return (scan_id, rows)."""
    cfg = stock_config()
    mkt = market_ticker or cfg.market_ticker
    ensure_intelligence_schema(db_path)

    ctx_snap = build_market_context(db_path, market_ticker=mkt)
    persist_market_context(db_path, ctx_snap)
    regime = ctx_snap.regime

    prior_scan = _latest_scan_id(db_path)
    prior_bands = _prior_bands(db_path, prior_scan)

    if tickers is None and cfg.daily_scan_scope == "focus":
        from src.analysis.ticker_registry import list_focus_symbols

        tickers = [r.symbol for r in list_focus_symbols(db_path)]
        if not tickers:
            return str(uuid.uuid4())[:12], []

    lb_df = build_leaderboard(
        db_path,
        mkt,
        tickers=tickers,
        segment=None,
        limit=limit * 3,
        use_parallel=use_parallel,
    )
    if lb_df.empty:
        return str(uuid.uuid4())[:12], []

    rows: list[GuidanceRow] = []
    for _, r in lb_df.iterrows():
        lb = LeaderboardRow(
            ticker=str(r["ticker"]),
            composite_score=float(r["composite_score"]),
            canslim_score=int(r["canslim_score"]),
            pattern_quality=float(r["pattern_quality"]),
            rs_pct=float(r["rs_pct"]),
            volume_ratio=float(r["volume_ratio"]),
            near_high_pct=float(r["near_high_pct"]),
            pass_setup=bool(r["pass_setup"]),
            pass_pattern=bool(r["pass_pattern"]),
            risk_flag=str(r.get("risk_flag") or ""),
            latest_price=float(r["latest_price"]),
            sector=str(r.get("sector") or ""),
            notes=str(r.get("notes") or ""),
            buy_readiness=float(r.get("buy_readiness") or 0),
            sell_pressure=float(r.get("sell_pressure") or 0),
            news_sentiment=float(r.get("news_sentiment") or 0.5),
            news_tags=r.get("news_tags"),
        )
        g = score_guidance_row(lb, db_path, market_regime=regime, prior_band=prior_bands.get(lb.ticker))
        rows.append(g)

    rows.sort(
        key=lambda x: (
            BAND_ORDER.get(RecommendationBand(x.recommendation_band), 2),
            x.confidence,
            x.composite_score,
        ),
        reverse=True,
    )
    rows = rows[:limit]

    scan_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    scanned_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            for g in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO guidance_snapshots
                    (scan_id, scanned_at, ticker, composite_score, confidence,
                     recommendation_band, canslim_score, news_sentiment, regime_alignment,
                     drivers_json, risk_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scan_id,
                        scanned_at,
                        g.ticker,
                        g.composite_score,
                        g.confidence,
                        g.recommendation_band,
                        g.canslim_score,
                        g.news_sentiment,
                        g.regime_alignment,
                        json.dumps(
                            {
                                "positive": g.positive_drivers,
                                "negative": g.negative_drivers,
                                "prior_band": g.prior_band,
                            }
                        ),
                        g.risk_notes[:500],
                    ),
                )
                if g.band_changed and g.prior_band:
                    conn.execute(
                        """
                        INSERT INTO guidance_changes
                        (scan_id, prior_scan_id, ticker, prior_band, new_band, change_type, detected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            scan_id,
                            prior_scan or "",
                            g.ticker,
                            g.prior_band,
                            g.recommendation_band,
                            _change_type(g.prior_band, g.recommendation_band),
                            scanned_at,
                        ),
                    )
            conn.commit()

    cfg.set_last_analysis_label(f"Guidance scan {scan_id}")
    return scan_id, rows


def _change_type(prior: str, new: str) -> str:
    p = BAND_ORDER.get(RecommendationBand(prior), 2)
    n = BAND_ORDER.get(RecommendationBand(new), 2)
    if n > p:
        return "upgrade"
    if n < p:
        return "downgrade"
    return "lateral"


def load_guidance_scan(db_path: str, scan_id: str | None = None) -> pd.DataFrame:
    ensure_intelligence_schema(db_path)
    with db_connection(db_path, readonly=True) as conn:
        if scan_id:
            df = pd.read_sql(
                "SELECT * FROM guidance_snapshots WHERE scan_id = ? ORDER BY confidence DESC",
                conn,
                params=(scan_id,),
            )
        else:
            latest = conn.execute(
                "SELECT scan_id FROM guidance_snapshots ORDER BY scanned_at DESC LIMIT 1"
            ).fetchone()
            if not latest:
                return pd.DataFrame()
            df = pd.read_sql(
                "SELECT * FROM guidance_snapshots WHERE scan_id = ? ORDER BY confidence DESC",
                conn,
                params=(latest[0],),
            )
    return df


def guidance_summary(db_path: str, scan_id: str | None = None) -> dict[str, Any]:
    df = load_guidance_scan(db_path, scan_id)
    ctx = load_latest_market_context(db_path)
    if df.empty:
        return {"scan_id": None, "count": 0, "by_band": {}, "market_regime": ctx.regime if ctx else "unknown"}
    by_band = df["recommendation_band"].value_counts().to_dict()
    return {
        "scan_id": df["scan_id"].iloc[0] if "scan_id" in df.columns else scan_id,
        "scanned_at": df["scanned_at"].iloc[0] if "scanned_at" in df.columns else "",
        "count": len(df),
        "by_band": by_band,
        "market_regime": ctx.regime if ctx else "unknown",
        "top_candidates": df[df["recommendation_band"] == "high_priority"]["ticker"].head(10).tolist(),
    }
