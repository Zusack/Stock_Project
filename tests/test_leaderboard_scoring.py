"""Leaderboard composite scoring tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.leaderboard import (
    LeaderboardSegment,
    filter_leaderboard_segment,
    rows_to_display_df,
)
from src.analysis.leaderboard_scoring import (
    CANSLIM_RULE_COUNT,
    TickerFeatures,
    apply_universe_scoring,
    compute_buy_readiness,
    compute_data_confidence,
    compute_sell_pressure,
    compute_v1_composite,
)
from src.analysis.news_signals import apply_bounded_sentiment


def test_canslim_rule_count_is_six():
    assert CANSLIM_RULE_COUNT == 6


def test_v1_composite_perfect_score():
    composite, components = compute_v1_composite(
        canslim_n=6,
        pattern_q=1.0,
        rs_pct=40.0,
        vol_ratio=2.0,
        news_norm=1.0,
    )
    assert composite == 100.0
    assert len(components) == 5
    assert sum(c.contribution for c in components) == pytest.approx(1.0)


def test_v1_composite_canslim_normalization_uses_six():
    composite, _ = compute_v1_composite(
        canslim_n=6,
        pattern_q=0.0,
        rs_pct=-20.0,
        vol_ratio=0.0,
        news_norm=0.5,
    )
    # 6/6 * 0.35 + 0 + 0 + 0 + 0.5*0.10 = 35 + 5 = 40
    assert composite == pytest.approx(40.0)


def test_bounded_sentiment_caps_delta():
    assert apply_bounded_sentiment(0.9, 0.2) == 0.5
    assert apply_bounded_sentiment(0.9, 1.0) == pytest.approx(0.65, abs=0.01)


def test_data_confidence_sparse():
    f = TickerFeatures(
        ticker="X",
        canslim_n=3,
        pattern_q=0.2,
        rs_pct=5,
        vol_ratio=1.0,
        near_high=0.8,
        news_norm=0.5,
        history_bars=60,
    )
    conf = compute_data_confidence(f)
    assert 0.15 <= conf <= 0.5


def test_universe_scoring_v2_ranks_stronger_rs_higher():
    features = [
        TickerFeatures(
            ticker="WEAK",
            canslim_n=2,
            pattern_q=0.1,
            rs_pct=-10,
            vol_ratio=0.8,
            near_high=0.6,
            news_norm=0.5,
            pass_setup=False,
            risk_flag="Extended below highs",
            history_bars=130,
            has_eps_data=True,
            roe=0.05,
            trailing_pe=40,
        ),
        TickerFeatures(
            ticker="STRONG",
            canslim_n=5,
            pattern_q=0.8,
            rs_pct=25,
            vol_ratio=1.5,
            near_high=0.95,
            news_norm=0.7,
            pass_setup=True,
            pass_pattern=True,
            history_bars=200,
            has_eps_data=True,
            roe=0.25,
            trailing_pe=18,
            peg_ratio=1.1,
        ),
    ]
    scored = apply_universe_scoring(features, score_version="v2")
    by_ticker = {s.features.ticker: s for s in scored}
    assert by_ticker["STRONG"].composite_score > by_ticker["WEAK"].composite_score
    assert by_ticker["STRONG"].buy_readiness > by_ticker["WEAK"].buy_readiness
    assert by_ticker["WEAK"].sell_pressure >= by_ticker["STRONG"].sell_pressure


def test_buy_and_sell_pressure_bounds():
    buy = compute_buy_readiness(
        momentum=0.8,
        quality=0.7,
        value=0.6,
        risk=0.75,
        sentiment=0.6,
        pass_setup=True,
        pass_pattern=True,
        risk_flag="",
        regime_align=0.8,
    )
    sell = compute_sell_pressure(
        momentum=0.2,
        quality=0.3,
        risk=0.2,
        sentiment=0.25,
        rs_pct=-15,
        near_high=0.5,
        risk_flag="Below 50-day simple moving average",
        news_tags=["downgrade"],
        regime_align=0.3,
    )
    assert 0 <= buy <= 100
    assert 0 <= sell <= 100
    assert sell > buy


def test_rows_to_display_canslim_six():
    df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "composite_score": 80.0,
                "canslim_score": 6,
                "buy_readiness": 75.0,
                "sell_pressure": 10.0,
                "rating_value": "B",
                "pass_setup": True,
                "pass_pattern": False,
                "risk_flag": "",
                "latest_price": 150.0,
                "sector": "Tech",
            }
        ]
    )
    out = rows_to_display_df(df)
    assert out.iloc[0]["CANSLM"] == "6/6"


def test_filter_buy_ready_segment():
    df = pd.DataFrame(
        [
            {"ticker": "A", "composite_score": 90, "buy_readiness": 80, "sell_pressure": 5, "pass_setup": True, "pass_pattern": False, "pattern_quality": 0.1, "risk_flag": ""},
            {"ticker": "B", "composite_score": 85, "buy_readiness": 60, "sell_pressure": 10, "pass_setup": True, "pass_pattern": True, "pattern_quality": 0.9, "risk_flag": ""},
        ]
    )
    out = filter_leaderboard_segment(df, LeaderboardSegment.BUY_READY, ":memory:", limit=10)
    assert out.iloc[0]["ticker"] == "A"


def test_filter_sell_pressure_segment():
    df = pd.DataFrame(
        [
            {"ticker": "A", "composite_score": 90, "buy_readiness": 80, "sell_pressure": 20, "pass_setup": True, "pass_pattern": False, "pattern_quality": 0.1, "risk_flag": ""},
            {"ticker": "B", "composite_score": 85, "buy_readiness": 60, "sell_pressure": 70, "pass_setup": False, "pass_pattern": True, "pattern_quality": 0.9, "risk_flag": "Below 50-day simple moving average"},
        ]
    )
    out = filter_leaderboard_segment(df, LeaderboardSegment.SELL_PRESSURE, ":memory:", limit=10)
    assert out.iloc[0]["ticker"] == "B"
