"""Tests for narrative JSON parsing."""

from src.analysis.ai.narrative import parse_narrative_json


def test_parse_narrative_json_nested():
    raw = 'Here is the result:\n{"headline":"Buy watch", "detail":"Strong setup.", "severity":"watch", "drivers":["momentum"]}'
    data = parse_narrative_json(raw)
    assert data is not None
    assert data["headline"] == "Buy watch"
    assert data["severity"] == "watch"


def test_parse_narrative_json_rejects_headline_format():
    raw = '{"sentiment":"positive","summary":"ok","confidence":0.8}'
    assert parse_narrative_json(raw) is None
