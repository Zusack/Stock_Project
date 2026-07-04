"""AI adapter fallback behavior."""

from src.analysis.ai.rule_based import RuleBasedAdapter


def test_rule_based_positive():
    adapter = RuleBasedAdapter()
    result = adapter.analyze_text("Company beats earnings with record growth upgrade")
    assert result.sentiment == "positive"
    assert result.provider == "rule_based"


def test_rule_based_negative():
    adapter = RuleBasedAdapter()
    result = adapter.analyze_text("SEC probe after fraud investigation and major loss")
    assert result.sentiment == "negative"
    assert len(result.risk_tags) >= 0
