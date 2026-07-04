"""Tests for versioned CANSLIM rulebook."""

from src.analysis.canslim_rulebook import (
    RULE_SET_VERSION,
    get_rule_set,
    rule_set_from_config,
)


def test_default_rule_set_version():
    rs = get_rule_set()
    assert rs.version == RULE_SET_VERSION
    assert rs.exit.stop_loss_pct == -0.08
    assert rs.exit.take_profit_pct == 0.25


def test_rule_set_spec_hash_stable():
    h1 = get_rule_set().spec_hash()
    h2 = get_rule_set().spec_hash()
    assert h1 == h2
    assert len(h1) == 16


def test_rule_set_from_config_overrides():
    rs = rule_set_from_config(stop_loss=-0.07, take_profit=0.22, require_pattern=True)
    assert rs.exit.stop_loss_pct == -0.07
    assert rs.exit.take_profit_pct == 0.22
    assert rs.entry.require_pattern is True


def test_legacy_rule_set():
    rs = get_rule_set("legacy")
    assert rs.version == "legacy"
    assert rs.entry.require_pattern is False
