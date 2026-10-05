"""Interactive CANSLM scoring for Stock Detail (aligned with Leaderboard).

Core score uses 6 letters (C, A, N, S, L, M) via the same pipeline as the
Leaderboard. The IBD \"I\" (Institutional sponsorship) letter is omitted — real
IBD accumulation metrics are proprietary and not available from our data sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analysis.canslim_core import (
    CANSLM_RULE_COUNT,
    CANSLIM_RULE_COUNT,
    CanslimBarEvaluation,
    evaluate_canslim_bar,
    market_direction_line,
    price_vs_sma50,
)
from src.analysis.db import load_price_data
from src.analysis.ticker_evaluation import evaluate_ticker_snapshot

__all__ = [
    "CANSLM_RULE_COUNT",
    "CANSLIM_RULE_COUNT",
    "CanslimResult",
    "analyze_canslim",
    "canslim_result_from_bar",
    "market_direction_line",
    "price_vs_sma50",
]


@dataclass
class CanslimResult:
    ticker: str
    score: int
    max_score: int = CANSLM_RULE_COUNT
    lines: list[str] = field(default_factory=list)
    verdict: str = "PASS"
    error: str | None = None
    risk_flag: str = ""
    rs_pct: float = 0.0
    volume_ratio: float = 0.0
    near_high_pct: float = 0.0
    pass_setup: bool = False
    pass_pattern: bool = False
    pattern_quality: float = 0.0
    latest_price: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "score": self.score,
            "max_score": self.max_score,
            "lines": self.lines,
            "verdict": self.verdict,
            "error": self.error,
            "risk_flag": self.risk_flag,
            "rs_pct": self.rs_pct,
            "volume_ratio": self.volume_ratio,
            "near_high_pct": self.near_high_pct,
            "pass_setup": self.pass_setup,
            "pass_pattern": self.pass_pattern,
            "pattern_quality": self.pattern_quality,
            "latest_price": self.latest_price,
        }


def canslim_result_from_bar(bar: CanslimBarEvaluation) -> CanslimResult:
    return CanslimResult(
        ticker=bar.ticker,
        score=bar.score,
        max_score=bar.max_score,
        lines=list(bar.lines),
        verdict=bar.verdict,
        risk_flag=bar.risk_flag,
        rs_pct=round(bar.rs_pct, 2),
        volume_ratio=round(bar.vol_ratio, 2),
        near_high_pct=round(bar.near_high * 100, 1),
        pass_setup=bar.pass_setup,
        pass_pattern=bar.pass_pattern,
        pattern_quality=round(bar.pattern_quality, 3),
        latest_price=round(bar.price, 2),
    )


def analyze_canslim(
    ticker: str,
    db_path: str,
    market_ticker: str = "^DJI",
    *,
    use_live_fundamentals: bool = True,
    ohlcv=None,
    headlines=None,
    fund_profile: dict | None = None,
) -> CanslimResult:
    """Score a single ticker using the shared Leaderboard CANSLM pipeline."""
    _ = use_live_fundamentals  # retained for API compatibility
    sym = ticker.upper().strip()
    df = load_price_data(sym, db_path)
    if df is None or len(df) < 60:
        return CanslimResult(
            ticker=sym,
            score=0,
            lines=[],
            verdict="PASS",
            error="Insufficient price data in database (need 60+ days).",
        )

    snap = evaluate_ticker_snapshot(
        sym,
        db_path,
        market_ticker,
        ohlcv=ohlcv,
        headlines=headlines,
        fund_profile=fund_profile,
        apply_scores=False,
    )
    if snap is None:
        return CanslimResult(
            ticker=sym,
            score=0,
            lines=[],
            verdict="PASS",
            error="Could not evaluate CANSLM criteria for this ticker.",
        )
    return canslim_result_from_bar(snap.bar)
