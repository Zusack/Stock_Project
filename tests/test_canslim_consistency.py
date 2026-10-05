"""Tests that Stock Detail and Leaderboard share CANSLM evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.analysis.canslim import analyze_canslim, canslim_result_from_bar
from src.analysis.canslim_core import CANSLM_RULE_COUNT, CanslimBarEvaluation, evaluate_canslim_bar
from src.analysis.leaderboard import extract_ticker_features


@pytest.fixture
def sample_price_df():
    dates = pd.date_range("2024-01-01", periods=280, freq="B")
    close = pd.Series([100 + i * 0.2 for i in range(len(dates))], index=dates)
    return pd.DataFrame(
        {
            "Adj Close": close,
            "Volume": 1_000_000,
        }
    )


@pytest.fixture
def sample_market_df():
    dates = pd.date_range("2024-01-01", periods=280, freq="B")
    close = pd.Series([200 + i * 0.1 for i in range(len(dates))], index=dates)
    return pd.DataFrame({"Adj Close": close})


def _sample_bar() -> CanslimBarEvaluation:
    return CanslimBarEvaluation(
        ticker="AMD",
        score=4,
        max_score=CANSLM_RULE_COUNT,
        lines=["[M] Market Direction: PASS (Nasdaq above 50-day simple moving average)"],
        risk_flag="",
        price=150.0,
        near_high=0.9,
        rs_pct=5.0,
        vol_ratio=1.2,
        history_bars=280,
    )


def test_analyze_canslim_uses_six_letter_score(sample_price_df):
    snap = MagicMock(bar=_sample_bar(), features=MagicMock(), leaderboard_row=None)
    with (
        patch("src.analysis.canslim.load_price_data", return_value=sample_price_df),
        patch("src.analysis.canslim.evaluate_ticker_snapshot", return_value=snap),
    ):
        result = analyze_canslim("AMD", "/tmp/db.db", market_ticker="^IXIC")

    assert result.max_score == CANSLM_RULE_COUNT
    assert result.score == 4
    assert not any(line.startswith("[I]") for line in result.lines)


def test_stock_detail_and_leaderboard_share_evaluator(sample_price_df, sample_market_df):
    with (
        patch("src.analysis.canslim_core.load_price_data") as mock_price,
        patch("src.analysis.canslim_core.load_ohlcv", return_value=None),
        patch("src.analysis.ticker_evaluation.load_ohlcv", return_value=None),
        patch(
            "src.analysis.canslim_core._load_eps_frames",
            return_value=(pd.DataFrame(), pd.DataFrame(), False),
        ),
        patch("src.analysis.ticker_evaluation.score_news_for_ticker", return_value=(0.5, [], 0.5, [], False)),
    ):
        mock_price.side_effect = lambda sym, db, cols=None, max_days=None: (
            sample_market_df if sym == "^IXIC" else sample_price_df
        )
        bar = evaluate_canslim_bar("AMD", "/tmp/db.db", "^IXIC")
        features = extract_ticker_features(
            "AMD",
            "/tmp/db.db",
            "^IXIC",
            market_df=sample_market_df,
            price_df=sample_price_df,
            ohlcv=None,
            q_df=pd.DataFrame(),
            a_df=pd.DataFrame(),
        )

    assert bar is not None
    assert features is not None
    assert bar.score == features.canslim_n
    assert bar.risk_flag == features.risk_flag
