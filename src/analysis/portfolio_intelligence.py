"""Portfolio construction guardrails and model portfolio templates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.analysis.db import db_connection, ingest_write_lock, load_price_data
from src.analysis.guidance import RecommendationBand, load_guidance_scan
from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.analysis.market_context import load_latest_market_context
from src.services.stock_config import stock_config

MAX_POSITION_PCT = 0.12
MAX_SECTOR_PCT = 0.30
MIN_CASH_RISK_OFF = 0.25
MIN_CASH_NEUTRAL = 0.10


@dataclass
class PositionSuggestion:
    ticker: str
    target_weight_pct: float
    conviction: float
    sector: str
    rationale: str


@dataclass
class PortfolioTemplate:
    name: str
    description: str
    max_positions: int
    min_band: str  # minimum recommendation band to include


TEMPLATES: dict[str, PortfolioTemplate] = {
    "growth_canslim": PortfolioTemplate(
        name="growth_canslim",
        description="CANSLIM-heavy growth: high-priority setups only",
        max_positions=8,
        min_band=RecommendationBand.HIGH_PRIORITY.value,
    ),
    "balanced": PortfolioTemplate(
        name="balanced",
        description="Candidates + high priority with sector diversification",
        max_positions=12,
        min_band=RecommendationBand.CANDIDATE.value,
    ),
    "defensive": PortfolioTemplate(
        name="defensive",
        description="Lower volatility, reduced size in risk-off regimes",
        max_positions=10,
        min_band=RecommendationBand.MONITOR.value,
    ),
}


@dataclass
class PortfolioPlan:
    template_name: str
    generated_at: str
    cash_pct: float
    positions: list[PositionSuggestion] = field(default_factory=list)
    sector_weights: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "template": self.template_name,
            "generated_at": self.generated_at,
            "cash_pct": self.cash_pct,
            "positions": [
                {
                    "ticker": p.ticker,
                    "target_weight_pct": p.target_weight_pct,
                    "conviction": p.conviction,
                    "sector": p.sector,
                    "rationale": p.rationale,
                }
                for p in self.positions
            ],
            "sector_weights": self.sector_weights,
            "warnings": self.warnings,
        }


def _volatility_scale(db_path: str, ticker: str, *, lookback: int = 60) -> float:
    df = load_price_data(ticker, db_path)
    if df is None or len(df) < lookback + 5:
        return 1.0
    ret = df["Adj Close"].astype(float).pct_change().dropna().tail(lookback)
    vol = float(ret.std()) if len(ret) else 0.02
    if vol <= 0:
        return 1.0
    # Higher vol -> smaller weight multiplier
    return max(0.5, min(1.2, 0.02 / vol))


def _guidance_rows_from_scan(db_path: str, scan_id: str | None) -> list[dict]:
    df = load_guidance_scan(db_path, scan_id)
    if df.empty:
        return []
    return df.to_dict("records")


def _band_rank(band: str) -> int:
    order = {
        RecommendationBand.EXIT_WATCH.value: 0,
        RecommendationBand.RISK_REVIEW.value: 1,
        RecommendationBand.MONITOR.value: 2,
        RecommendationBand.CANDIDATE.value: 3,
        RecommendationBand.HIGH_PRIORITY.value: 4,
    }
    return order.get(band, 2)


def build_portfolio_plan(
    db_path: str,
    template_name: str = "balanced",
    *,
    scan_id: str | None = None,
    guidance_rows: list[dict] | None = None,
) -> PortfolioPlan:
    tpl = TEMPLATES.get(template_name, TEMPLATES["balanced"])
    ctx = load_latest_market_context(db_path)
    regime = ctx.regime if ctx else "neutral"

    if guidance_rows is None:
        guidance_rows = _guidance_rows_from_scan(db_path, scan_id)

    min_rank = _band_rank(tpl.min_band)
    eligible = [
        g for g in guidance_rows
        if _band_rank(str(g.get("recommendation_band", ""))) >= min_rank
        and str(g.get("recommendation_band", "")) not in (
            RecommendationBand.EXIT_WATCH.value,
            RecommendationBand.RISK_REVIEW.value,
        )
    ]
    eligible.sort(key=lambda g: float(g.get("confidence", 0)), reverse=True)
    eligible = eligible[: tpl.max_positions * 2]

    if regime == "risk_off":
        cash_pct = MIN_CASH_RISK_OFF
    elif regime == "neutral":
        cash_pct = MIN_CASH_NEUTRAL
    else:
        cash_pct = 0.05

    investable = 1.0 - cash_pct
    positions: list[PositionSuggestion] = []
    sector_totals: dict[str, float] = {}
    warnings: list[str] = []

    if not eligible:
        warnings.append("No eligible guidance rows for this template — run a guidance scan first.")
        return PortfolioPlan(
            template_name=tpl.name,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            cash_pct=cash_pct,
            positions=[],
            sector_weights={},
            warnings=warnings,
        )

    raw_weights: list[tuple[str, float, str, str]] = []
    for g in eligible[: tpl.max_positions]:
        ticker = str(g["ticker"])
        conf = float(g.get("confidence", 0.5))
        vol_scale = _volatility_scale(db_path, ticker)
        w = conf * vol_scale
        sector = _load_sector(db_path, ticker)
        raw_weights.append((ticker, w, sector, str(g.get("recommendation_band", ""))))

    total_w = sum(x[1] for x in raw_weights) or 1.0
    for ticker, w, sector, band in raw_weights:
        pct = (w / total_w) * investable
        pct = min(pct, MAX_POSITION_PCT)
        sec_total = sector_totals.get(sector, 0.0) + pct
        if sec_total > MAX_SECTOR_PCT:
            pct = max(0.0, MAX_SECTOR_PCT - sector_totals.get(sector, 0.0))
            warnings.append(f"Sector cap applied for {sector}")
        if pct < 0.01:
            continue
        sector_totals[sector] = sector_totals.get(sector, 0.0) + pct
        positions.append(
            PositionSuggestion(
                ticker=ticker,
                target_weight_pct=round(pct * 100, 2),
                conviction=round(w / total_w, 3),
                sector=sector,
                rationale=f"{band} band; volatility-adjusted sizing",
            )
        )

    # Renormalize if we capped
    pos_sum = sum(p.target_weight_pct for p in positions) / 100.0
    if pos_sum > investable + 0.01:
        scale = investable / pos_sum if pos_sum else 1.0
        for p in positions:
            p.target_weight_pct = round(p.target_weight_pct * scale, 2)

    if ctx and ctx.sector_laggards:
        warnings.append(f"Weak sectors recently: {', '.join(ctx.sector_laggards[:3])}")

    return PortfolioPlan(
        template_name=tpl.name,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        cash_pct=round(cash_pct * 100, 1),
        positions=positions,
        sector_weights={k: round(v * 100, 1) for k, v in sector_totals.items()},
        warnings=warnings,
    )


def _load_sector(db_path: str, ticker: str) -> str:
    try:
        with db_connection(db_path, readonly=True) as conn:
            row = conn.execute(
                "SELECT Sector FROM stock_profiles WHERE Ticker = ?", (ticker.upper(),)
            ).fetchone()
        return row[0] if row and row[0] else "Unknown"
    except Exception:
        return "Unknown"


def persist_portfolio_plan(db_path: str, scan_id: str, plan: PortfolioPlan) -> None:
    ensure_intelligence_schema(db_path)
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO portfolio_suggestions
                (scan_id, template_name, generated_at, total_positions, cash_pct, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_id,
                    plan.template_name,
                    plan.generated_at,
                    len(plan.positions),
                    plan.cash_pct,
                    json.dumps(plan.to_payload()),
                ),
            )
            conn.commit()


def build_all_template_plans(db_path: str, scan_id: str) -> dict[str, PortfolioPlan]:
    plans = {}
    for name in TEMPLATES:
        plans[name] = build_portfolio_plan(db_path, name, scan_id=scan_id)
        persist_portfolio_plan(db_path, scan_id, plans[name])
    return plans
