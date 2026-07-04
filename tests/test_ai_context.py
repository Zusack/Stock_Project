"""Tests for AI context assembly and advisor wiring."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.ai.advisor import RuleBasedSignalAdvisor, resolve_advisor_rows
from src.analysis.ai.context import assemble_ticker_context, context_hash
from src.analysis.ai.lmstudio_client import LMStudioAdapter, _validate_analysis_json
from src.analysis.guidance import GuidanceRow


def test_context_hash_stable():
    payload = {"ticker": "AAPL", "news": {"sentiment": 0.5}}
    h1 = context_hash(payload)
    h2 = context_hash(payload)
    assert h1 == h2
    assert len(h1) == 16


def test_validate_analysis_json():
    assert _validate_analysis_json({"sentiment": "positive", "confidence": 0.8})
    assert not _validate_analysis_json({"sentiment": "maybe", "confidence": 0.8})
    assert not _validate_analysis_json({"sentiment": "positive", "confidence": 1.5})


def test_lmstudio_parse_catalyst_type():
    adapter = LMStudioAdapter(max_retries=0)
    content = json.dumps(
        {
            "sentiment": "positive",
            "risk_tags": [],
            "summary": "Beat earnings",
            "thesis_flags": [],
            "catalyst_type": "earnings",
            "confidence": 0.9,
        }
    )
    result = adapter._parse_response(content, "AAPL")
    assert result.catalyst_type == "earnings"
    assert "earnings" in result.thesis_flags
    assert result.provider == "lm_studio"


def test_advisor_with_guidance_row():
    advisor = RuleBasedSignalAdvisor()
    guidance = GuidanceRow(
        ticker="AAPL",
        composite_score=72.0,
        confidence=0.7,
        recommendation_band="candidate",
        canslim_score=4,
        news_sentiment=0.6,
        regime_alignment=0.7,
        positive_drivers=["Strong RS"],
        negative_drivers=[],
    )
    suggestions = advisor.analyze_ticker("AAPL", guidance_row=guidance)
    assert suggestions
    assert suggestions[0].headline != "No guidance data"
    assert suggestions[0].band == "candidate"


@patch("src.analysis.ai.context.score_ticker")
@patch("src.analysis.ai.context.load_latest_market_context")
@patch("src.analysis.news_signals.load_recent_headlines")
@patch("src.analysis.news_signals.score_ticker_news")
@patch("src.analysis.ai.context.get_quote")
@patch("src.analysis.ai.context.get_profile")
@patch("src.analysis.ai.context.load_ohlcv")
def test_assemble_ticker_context_shape(
    mock_ohlcv,
    mock_profile,
    mock_quote,
    mock_news_score,
    mock_headlines,
    mock_market,
    mock_score,
):
    mock_market.return_value = None
    mock_score.return_value = None
    mock_headlines.return_value = [{"title": "Test", "date": "2026-01-01"}]
    mock_news_score.return_value = (0.55, [], 0.5, [])
    mock_quote.return_value = None
    mock_profile.return_value = {"Sector": "Tech"}
    mock_ohlcv.return_value = None

    payload = assemble_ticker_context("AAPL", "/tmp/test.db")
    assert payload["ticker"] == "AAPL"
    assert "context_hash" in payload
    assert "canslim_note" in payload
    assert payload["news"]["headlines"]
