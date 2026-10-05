"""Tests for unified ticker evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.analysis.canslim import canslim_result_from_bar
from src.analysis.canslim_core import CanslimBarEvaluation
from src.analysis.ticker_evaluation import (
    evaluate_ticker_snapshot,
    features_from_bar,
    refresh_canslm_metrics_df,
    score_news_for_ticker,
)
from src.analysis.canslim_core import format_canslm_tooltip


def test_canslim_result_from_bar_includes_leaderboard_metrics():
    bar = CanslimBarEvaluation(
        ticker="AMD",
        score=4,
        max_score=6,
        lines=["[M] Market Direction: PASS (Nasdaq above 50-day simple moving average)"],
        risk_flag="",
        price=150.0,
        near_high=0.92,
        rs_pct=12.5,
        vol_ratio=1.35,
        pass_setup=True,
        pass_pattern=False,
        pattern_quality=0.4,
        history_bars=200,
    )
    result = canslim_result_from_bar(bar)
    assert result.rs_pct == 12.5
    assert result.volume_ratio == 1.35
    assert result.near_high_pct == 92.0
    assert result.pass_setup is True


def test_features_from_bar_maps_shared_metrics():
    bar = CanslimBarEvaluation(
        ticker="AMD",
        score=3,
        max_score=6,
        risk_flag="Extended below highs",
        price=100.0,
        near_high=0.7,
        rs_pct=-2.0,
        vol_ratio=0.9,
        pass_setup=False,
        pass_pattern=False,
        history_bars=120,
        has_eps_data=True,
    )
    features = features_from_bar(
        bar,
        ticker="AMD",
        sector="Technology",
        industry="Semiconductors",
        fund_profile={"ROE": 0.2},
        news_norm=0.6,
        news_tags=["earnings"],
        news_confidence=0.7,
        catalyst_tags=[],
        has_news=True,
        market_regime="neutral",
        ohlcv=None,
        price_df=pd.DataFrame({"Adj Close": [1.0, 2.0]}),
    )
    assert features.canslim_n == 3
    assert features.rs_pct == -2.0
    assert features.risk_flag == "Extended below highs"
    assert features.history_bars == 120


def test_score_news_for_ticker_with_headlines():
    headlines = [{"title": "Beat", "date": "2026-01-01"}]
    with patch(
        "src.analysis.news_signals.score_headlines_list",
        return_value=(0.7, ["earnings"], 0.8, ["growth"]),
    ):
        sent, tags, conf, catalysts, has_news = score_news_for_ticker(
            "/tmp/db", "AMD", headlines=headlines
        )
    assert sent == 0.7
    assert has_news is True
    assert "earnings" in tags


def test_evaluate_ticker_snapshot_returns_features_and_row():
    bar = CanslimBarEvaluation(
        ticker="AMD",
        score=2,
        max_score=6,
        risk_flag="",
        price=100.0,
        near_high=0.8,
        rs_pct=1.0,
        vol_ratio=1.1,
        history_bars=100,
    )
    mock_features = MagicMock()
    mock_row = MagicMock()
    with (
        patch("src.analysis.ticker_evaluation.evaluate_canslim_bar", return_value=bar),
        patch("src.analysis.ticker_evaluation.load_price_data", return_value=pd.DataFrame({"Adj Close": [1.0]})),
        patch("src.analysis.ticker_evaluation.load_ohlcv", return_value=None),
        patch("src.analysis.ticker_evaluation.score_news_for_ticker", return_value=(0.5, [], 0.5, [], False)),
        patch("src.analysis.ticker_evaluation.features_from_bar", return_value=mock_features) as mock_map,
        patch("src.analysis.ticker_evaluation.apply_universe_scoring", return_value=[MagicMock()]),
        patch("src.analysis.leaderboard._scored_to_row", return_value=mock_row),
    ):
        snap = evaluate_ticker_snapshot("AMD", "/tmp/db", "^IXIC", apply_scores=True)
    assert snap is not None
    assert snap.bar is bar
    assert snap.features is mock_features
    assert snap.leaderboard_row is mock_row
    mock_map.assert_called_once()


def test_format_canslm_tooltip_joins_lines():
    lines = [
        "[C] Current Earnings: FAIL",
        "[M] Market Direction: PASS (Nasdaq above 50-day simple moving average)",
    ]
    text = format_canslm_tooltip(lines)
    assert "[C]" in text
    assert "[M]" in text


def test_refresh_canslm_metrics_df_overlays_stale_cache():
    bar = CanslimBarEvaluation(
        ticker="AAPL",
        score=2,
        max_score=6,
        lines=[
            "[L] Leader: PASS",
            "[M] Market Direction: PASS (Dow Jones above 50-day simple moving average)",
        ],
        risk_flag="",
        price=200.0,
        near_high=0.8,
        rs_pct=5.0,
        vol_ratio=1.1,
        pass_setup=False,
        pass_pattern=False,
        pattern_quality=0.2,
        history_bars=200,
    )
    stale = pd.DataFrame(
        [{"ticker": "AAPL", "canslim_score": 1, "rs_pct": 1.0, "volume_ratio": 0.5, "near_high_pct": 70.0}]
    )
    with (
        patch("src.analysis.ticker_evaluation.evaluate_canslim_bar", return_value=bar),
        patch("src.analysis.bulk_loaders.prepare_market_frame", return_value=pd.DataFrame()),
        patch("src.analysis.bulk_loaders.load_history_grouped", return_value={"AAPL": pd.DataFrame()}),
    ):
        out = refresh_canslm_metrics_df(stale, "/tmp/db", "^DJI")
    assert int(out.iloc[0]["canslim_score"]) == 2
    assert "[L]" in str(out.iloc[0]["canslm_tooltip"])
