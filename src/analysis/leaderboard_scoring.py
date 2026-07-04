"""Leaderboard composite scoring v1/v2, factor buckets, and action scores."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.canslim_rulebook import (
    LeaderboardWeights,
    LeaderboardWeightsV2,
    get_rule_set,
)

CANSLIM_RULE_COUNT = 6
CANSLIM_PASS_COLUMNS = ("Pass_C", "Pass_A", "Pass_N", "Pass_S", "Pass_L", "Pass_M")

SCORE_VERSION_V1 = "v1"
SCORE_VERSION_V2 = "v2"
DEFAULT_SCORE_VERSION = SCORE_VERSION_V2


class ScoreVersion(str, Enum):
    V1 = SCORE_VERSION_V1
    V2 = SCORE_VERSION_V2


@dataclass
class ScoreComponent:
    name: str
    raw: float
    normalized: float
    weight: float
    contribution: float
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TickerFeatures:
    """Per-ticker raw inputs before cross-universe normalization."""

    ticker: str
    canslim_n: int
    pattern_q: float
    rs_pct: float
    vol_ratio: float
    near_high: float
    news_norm: float
    news_confidence: float = 0.5
    news_tags: list[str] = field(default_factory=list)
    pass_setup: bool = False
    pass_pattern: bool = False
    risk_flag: str = ""
    price: float = 0.0
    sector: str = ""
    industry: str = ""
    roe: float | None = None
    profit_margins: float | None = None
    debt_to_equity: float | None = None
    peg_ratio: float | None = None
    trailing_pe: float | None = None
    inst_ownership: float | None = None
    vol_annual_pct: float | None = None
    regime_align: float = 0.5
    has_eps_data: bool = False
    has_news: bool = False
    history_bars: int = 0
    catalyst_tags: list[str] = field(default_factory=list)


@dataclass
class ScoredTicker:
    """Fully scored ticker ready for LeaderboardRow conversion."""

    features: TickerFeatures
    composite_score: float
    composite_score_v1: float
    score_version: str
    momentum_score: float
    quality_score: float
    value_score: float
    risk_score: float
    sentiment_score: float
    buy_readiness: float
    sell_pressure: float
    rating_value: str
    rating_quality: str
    rating_momentum: str
    rating_risk: str
    data_confidence: float
    components: list[ScoreComponent] = field(default_factory=list)

    def components_json(self) -> str:
        return json.dumps([c.to_dict() for c in self.components], separators=(",", ":"))


def canslim_score_from_row(row: pd.Series) -> int:
    return sum(
        1 for c in CANSLIM_PASS_COLUMNS if c in row.index and bool(row[c])
    )


def compute_v1_composite(
    *,
    canslim_n: int,
    pattern_q: float,
    rs_pct: float,
    vol_ratio: float,
    news_norm: float,
    weights: LeaderboardWeights | None = None,
) -> tuple[float, list[ScoreComponent]]:
    lw = weights or get_rule_set().leaderboard
    canslim_norm = canslim_n / float(CANSLIM_RULE_COUNT)
    vol_norm = min(1.0, vol_ratio / 2.0)
    rs_norm = max(0.0, min(1.0, (rs_pct + 20) / 60))
    components = [
        ScoreComponent("canslim_setup", float(canslim_n), canslim_norm, lw.canslim_setup, lw.canslim_setup * canslim_norm),
        ScoreComponent("pattern_quality", pattern_q, pattern_q, lw.pattern_quality, lw.pattern_quality * pattern_q),
        ScoreComponent("relative_strength", rs_pct, rs_norm, lw.relative_strength, lw.relative_strength * rs_norm),
        ScoreComponent("volume_profile", vol_ratio, vol_norm, lw.volume_profile, lw.volume_profile * vol_norm),
        ScoreComponent("news_sentiment", news_norm, news_norm, lw.news_sentiment, lw.news_sentiment * news_norm),
    ]
    composite = sum(c.contribution for c in components) * 100
    return round(composite, 2), components


def annualized_volatility_pct(price_df: pd.DataFrame, *, window: int = 60) -> float | None:
    if price_df is None or len(price_df) < window + 5:
        return None
    close = price_df["Adj Close"].astype(float)
    rets = close.pct_change(fill_method=None).dropna().tail(window)
    if rets.empty:
        return None
    return float(rets.std() * np.sqrt(252) * 100)


def compute_data_confidence(f: TickerFeatures) -> float:
    """0-1 confidence based on data completeness and freshness proxies."""
    score = 0.0
    if f.history_bars >= 126:
        score += 0.35
    elif f.history_bars >= 60:
        score += 0.20
    if f.has_eps_data:
        score += 0.25
    elif f.canslim_n > 0:
        score += 0.10
    if f.roe is not None or f.profit_margins is not None:
        score += 0.15
    if f.has_news:
        score += 0.10
    if f.trailing_pe is not None or f.peg_ratio is not None:
        score += 0.10
    if f.vol_annual_pct is not None:
        score += 0.05
    return round(max(0.15, min(1.0, score)), 3)


def _rank_pct(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    """Vectorized percentile rank in [0, 1]; NaN inputs -> 0.5."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.5, index=s.index, dtype=float)
    ranked = s.rank(pct=True, method="average")
    if not higher_is_better:
        ranked = 1.0 - ranked
    return ranked.fillna(0.5).clip(0.0, 1.0)


def _sector_pe_ranks(df: pd.DataFrame) -> pd.Series:
    """P/E percentile within sector when enough peers; else universe-wide."""
    pe = pd.to_numeric(df["trailing_pe"], errors="coerce")
    out = pd.Series(0.5, index=df.index, dtype=float)
    valid = pe.notna() & (pe > 0)
    if valid.sum() >= 2:
        out.loc[valid] = (1.0 - pe.loc[valid].rank(pct=True, method="average")).clip(0.0, 1.0)
    for sector, idx in df.groupby("sector", dropna=False).groups.items():
        if not str(sector).strip():
            continue
        mask = df.index.isin(idx) & valid
        if mask.sum() >= 3:
            out.loc[mask] = (1.0 - pe.loc[mask].rank(pct=True, method="average")).clip(0.0, 1.0)
    return out


def _precompute_v2_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized rank columns for v2 scoring (O(n) vs per-ticker scans)."""
    out = df.copy()
    out["_mom_rs"] = _rank_pct(out["rs_pct"])
    out["_mom_pat"] = _rank_pct(out["pattern_q"])
    out["_mom_vol"] = _rank_pct(out["vol_ratio"])
    out["_mom_near"] = _rank_pct(out["near_high"])
    out["_mom_canslim"] = _rank_pct(out["canslim_n"])
    out["_qual_roe"] = _rank_pct(out["roe"])
    out["_qual_margin"] = _rank_pct(out["profit_margins"])
    out["_qual_de"] = _rank_pct(out["debt_to_equity"], higher_is_better=False)
    out["_val_peg"] = _rank_pct(out["peg_ratio"], higher_is_better=False)
    out["_val_pe"] = _sector_pe_ranks(out)
    out["_risk_vol"] = _rank_pct(out["vol_annual_pct"], higher_is_better=False)
    return out


def _score_to_letter(score_0_100: float) -> str:
    if score_0_100 >= 85:
        return "A"
    if score_0_100 >= 70:
        return "B"
    if score_0_100 >= 55:
        return "C"
    if score_0_100 >= 40:
        return "D"
    return "F"


def _risk_flag_penalty(risk_flag: str) -> float:
    if not risk_flag:
        return 0.0
    if "Below 50-day simple moving average" in risk_flag:
        return 0.35
    if "Market weak" in risk_flag:
        return 0.25
    if "Extended below highs" in risk_flag:
        return 0.15
    return 0.10


def compute_buy_readiness(
    *,
    momentum: float,
    quality: float,
    value: float,
    risk: float,
    sentiment: float,
    pass_setup: bool,
    pass_pattern: bool,
    risk_flag: str,
    regime_align: float,
) -> float:
    base = (
        0.30 * momentum
        + 0.20 * quality
        + 0.15 * value
        + 0.20 * risk
        + 0.15 * sentiment
    )
    if pass_setup:
        base += 0.08
    if pass_pattern:
        base += 0.05
    base += (regime_align - 0.5) * 0.12
    base -= _risk_flag_penalty(risk_flag)
    return round(max(0.0, min(100.0, base * 100)), 2)


def compute_sell_pressure(
    *,
    momentum: float,
    quality: float,
    risk: float,
    sentiment: float,
    rs_pct: float,
    near_high: float,
    risk_flag: str,
    news_tags: list[str],
    regime_align: float,
) -> float:
    pressure = 0.0
    pressure += _risk_flag_penalty(risk_flag) * 100
    if rs_pct < 0:
        pressure += min(25.0, abs(rs_pct) * 0.5)
    if near_high < 0.75:
        pressure += (0.75 - near_high) * 40
    pressure += (1.0 - risk) * 35
    pressure += (1.0 - momentum) * 25
    pressure += max(0.0, 0.5 - sentiment) * 30
    if regime_align < 0.45:
        pressure += (0.45 - regime_align) * 40
    bearish_tags = {"lawsuit", "investigation", "downgrade", "miss", "recall", "fraud", "bankruptcy"}
    for tag in news_tags:
        if any(b in str(tag).lower() for b in bearish_tags):
            pressure += 8
    return round(max(0.0, min(100.0, pressure)), 2)


def _weighted_bucket_composite(
    buckets: dict[str, tuple[float, bool]],
    weights: LeaderboardWeightsV2,
) -> tuple[float, list[ScoreComponent]]:
    available = {k: v for k, v in buckets.items() if v[1]}
    if not available:
        return 50.0, []
    total_w = sum(getattr(weights, k) for k in available)
    components: list[ScoreComponent] = []
    composite = 0.0
    for key, (norm, _) in available.items():
        w = getattr(weights, key)
        eff_w = w / total_w if total_w else 0.0
        contrib = eff_w * norm
        composite += contrib
        components.append(
            ScoreComponent(
                name=key,
                raw=norm,
                normalized=norm,
                weight=eff_w,
                contribution=contrib,
            )
        )
    return round(composite * 100, 2), components


def apply_universe_scoring(
    features: list[TickerFeatures],
    *,
    score_version: str = DEFAULT_SCORE_VERSION,
    market_regime: str = "neutral",
) -> list[ScoredTicker]:
    """Cross-ticker normalization and final composite/action scores."""
    if not features:
        return []

    v1_weights = get_rule_set().leaderboard
    v2_weights = get_rule_set().leaderboard_v2

    rows: list[dict[str, Any]] = []
    for f in features:
        f.data_confidence = compute_data_confidence(f)
        rows.append(
            {
                "ticker": f.ticker,
                "sector": f.sector,
                "canslim_n": f.canslim_n,
                "pattern_q": f.pattern_q,
                "rs_pct": f.rs_pct,
                "vol_ratio": f.vol_ratio,
                "near_high": f.near_high,
                "news_norm": f.news_norm,
                "roe": f.roe,
                "profit_margins": f.profit_margins,
                "debt_to_equity": f.debt_to_equity,
                "peg_ratio": f.peg_ratio,
                "trailing_pe": f.trailing_pe,
                "inst_ownership": f.inst_ownership,
                "vol_annual_pct": f.vol_annual_pct,
                "regime_align": f.regime_align,
                "features": f,
            }
        )
    df = pd.DataFrame(rows)
    ranked_df = _precompute_v2_ranks(df) if score_version != SCORE_VERSION_V1 else df

    scored: list[ScoredTicker] = []
    for idx, row in df.iterrows():
        f: TickerFeatures = row["features"]
        v1_composite, v1_components = compute_v1_composite(
            canslim_n=f.canslim_n,
            pattern_q=f.pattern_q,
            rs_pct=f.rs_pct,
            vol_ratio=f.vol_ratio,
            news_norm=f.news_norm,
            weights=v1_weights,
        )

        if score_version == SCORE_VERSION_V1:
            momentum = v1_components[1].normalized * 0.5 + v1_components[2].normalized * 0.5
            composite = v1_composite
            components = v1_components
            quality = 0.5
            value = 0.5
            risk = 1.0 - _risk_flag_penalty(f.risk_flag)
            sentiment = f.news_norm
        else:
            r = ranked_df.loc[idx]
            momentum = (
                0.30 * float(r["_mom_rs"])
                + 0.25 * float(r["_mom_pat"])
                + 0.20 * float(r["_mom_vol"])
                + 0.15 * float(r["_mom_near"])
                + 0.10 * float(r["_mom_canslim"])
            )

            qual_parts: list[float] = []
            if f.roe is not None and not pd.isna(f.roe):
                qual_parts.append(float(r["_qual_roe"]))
            if f.profit_margins is not None and not pd.isna(f.profit_margins):
                qual_parts.append(float(r["_qual_margin"]))
            if f.debt_to_equity is not None and not pd.isna(f.debt_to_equity):
                qual_parts.append(float(r["_qual_de"]))
            quality = sum(qual_parts) / len(qual_parts) if qual_parts else 0.5

            val_parts: list[float] = []
            if f.peg_ratio is not None and f.peg_ratio > 0 and not pd.isna(f.peg_ratio):
                val_parts.append(float(r["_val_peg"]))
            if f.trailing_pe is not None and f.trailing_pe > 0 and not pd.isna(f.trailing_pe):
                val_parts.append(float(r["_val_pe"]))
            value = sum(val_parts) / len(val_parts) if val_parts else 0.5

            risk_parts: list[float] = []
            if f.vol_annual_pct is not None and not pd.isna(f.vol_annual_pct):
                risk_parts.append(float(r["_risk_vol"]))
            risk_parts.append(float(f.regime_align))
            risk_parts.append(1.0 - _risk_flag_penalty(f.risk_flag))
            risk = sum(risk_parts) / len(risk_parts) if risk_parts else 0.5

            sentiment = f.news_norm
            if f.news_confidence < 0.4:
                sentiment = 0.5 + (sentiment - 0.5) * f.news_confidence

            buckets = {
                "momentum_technical": (momentum, True),
                "quality": (quality, bool(qual_parts)),
                "value": (value, bool(val_parts)),
                "risk_regime": (risk, True),
                "sentiment_catalyst": (sentiment, f.has_news or f.news_confidence >= 0.4),
            }
            composite, components = _weighted_bucket_composite(buckets, v2_weights)
            conf = f.data_confidence
            composite = round(composite * conf + 50.0 * (1.0 - conf), 2)

        buy = compute_buy_readiness(
            momentum=momentum,
            quality=quality,
            value=value,
            risk=risk,
            sentiment=sentiment,
            pass_setup=f.pass_setup,
            pass_pattern=f.pass_pattern,
            risk_flag=f.risk_flag,
            regime_align=f.regime_align,
        )
        sell = compute_sell_pressure(
            momentum=momentum,
            quality=quality,
            risk=risk,
            sentiment=sentiment,
            rs_pct=f.rs_pct,
            near_high=f.near_high,
            risk_flag=f.risk_flag,
            news_tags=f.news_tags,
            regime_align=f.regime_align,
        )

        scored.append(
            ScoredTicker(
                features=f,
                composite_score=composite,
                composite_score_v1=v1_composite,
                score_version=score_version,
                momentum_score=round(momentum * 100, 2),
                quality_score=round(quality * 100, 2),
                value_score=round(value * 100, 2),
                risk_score=round(risk * 100, 2),
                sentiment_score=round(sentiment * 100, 2),
                buy_readiness=buy,
                sell_pressure=sell,
                rating_momentum=_score_to_letter(momentum * 100),
                rating_quality=_score_to_letter(quality * 100),
                rating_value=_score_to_letter(value * 100),
                rating_risk=_score_to_letter(risk * 100),
                data_confidence=f.data_confidence,
                components=components,
            )
        )
    return scored
