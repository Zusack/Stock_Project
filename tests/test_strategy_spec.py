"""Tests for StrategySpec serialization and validation."""

from __future__ import annotations

import json

import pytest

from src.analysis.strategy_spec import (
    Rule,
    RuleGroup,
    RuleTarget,
    StrategySpec,
    parse_spec_from_llm_json,
    spec_to_plain_english,
    validate_spec,
)
from src.analysis.strategy_presets import get_preset, list_preset_ids


def test_preset_round_trip():
    for pid in list_preset_ids():
        spec = get_preset(pid)
        restored = StrategySpec.from_dict(spec.to_dict())
        assert restored.name == spec.name
        assert restored.preset_id == spec.preset_id
        assert restored.engine == spec.engine


def test_spec_json_round_trip():
    spec = get_preset("rsi_dip")
    text = spec.to_json()
    restored = StrategySpec.from_json(text)
    assert restored.entry.rules[0].indicator == "RSI"
    assert restored.preset_id == "rsi_dip"


def test_validate_requires_entry_rules():
    spec = StrategySpec(name="Empty", entry=RuleGroup(rules=[]))
    errors = validate_spec(spec)
    assert any("entry rule" in e.lower() for e in errors)


def test_plain_english_preview():
    spec = get_preset("golden_cross")
    text = spec_to_plain_english(spec)
    assert "Buy when" in text
    assert "SMA" in text


def test_parse_llm_json_valid():
    payload = {
        "name": "AI RSI",
        "entry": {
            "logic": "and",
            "rules": [
                {
                    "indicator": "RSI",
                    "params": {"period": 14},
                    "comparator": "crosses_below",
                    "target": {"kind": "value", "value": 30},
                }
            ],
        },
        "engine": "unified",
    }
    spec, err = parse_spec_from_llm_json(json.dumps(payload))
    assert err is None
    assert spec is not None
    assert spec.name == "AI RSI"


def test_parse_llm_json_invalid():
    spec, err = parse_spec_from_llm_json("not json at all")
    assert spec is None
    assert err is not None


def test_buy_hold_skips_entry_validation():
    spec = get_preset("buy_hold")
    assert validate_spec(spec) == []
