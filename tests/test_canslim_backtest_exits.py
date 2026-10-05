"""Tests for rulebook-driven exit evaluation."""

from src.analysis.canslim_backtest import evaluate_exit
from src.analysis.canslim_rulebook import (
    ExitRules,
    get_rule_set,
    normalize_stop_loss_pct,
    rule_set_from_config,
)


def test_stop_loss_first():
    rules = get_rule_set()
    should, reason = evaluate_exit(-0.09, 90.0, 100.0, 0.0, rules)
    assert should is True
    assert reason == "Stop Loss"


def test_positive_stop_magnitude_from_ui_does_not_false_trigger():
    """UI passes +0.08; must NOT treat +5% as a stop-loss."""
    rules = rule_set_from_config(stop_loss=0.08, take_profit=0.25)
    assert rules.exit.stop_loss_pct == -0.08
    should, reason = evaluate_exit(0.05, 105.0, 100.0, 0.05, rules)
    assert should is False
    assert reason == ""
    should, reason = evaluate_exit(-0.09, 91.0, 100.0, 0.0, rules)
    assert should is True
    assert reason == "Stop Loss"


def test_normalize_stop_accepts_either_sign():
    assert normalize_stop_loss_pct(0.08) == -0.08
    assert normalize_stop_loss_pct(-0.08) == -0.08


def test_take_profit():
    rules = get_rule_set()
    should, reason = evaluate_exit(0.26, 126.0, 100.0, 0.2, rules)
    assert should is True
    assert reason == "Take Profit"


def test_eight_week_hold_defers_take_profit():
    rules = get_rule_set()
    # +26% after 10 bars with early surge — still inside 8-week lock.
    should, reason = evaluate_exit(
        0.26, 126.0, 100.0, 0.26, rules, bars_held=10, hit_early_surge=True
    )
    assert should is False
    assert reason == ""
    # After 40 bars, profit-taking resumes.
    should, reason = evaluate_exit(
        0.26, 126.0, 100.0, 0.26, rules, bars_held=40, hit_early_surge=True
    )
    assert should is True
    assert reason == "Take Profit"


def test_eight_week_hold_still_honors_stop():
    rules = get_rule_set()
    should, reason = evaluate_exit(
        -0.09, 91.0, 100.0, 0.22, rules, bars_held=5, hit_early_surge=True
    )
    assert should is True
    assert reason == "Stop Loss"


def test_round_trip_sell():
    rules = get_rule_set()
    should, reason = evaluate_exit(-0.02, 98.0, 100.0, 0.15, rules)
    assert should is True
    assert reason == "Round-Trip Sell"


def test_no_exit_in_range():
    rules = get_rule_set()
    should, reason = evaluate_exit(0.05, 105.0, 100.0, 0.05, rules)
    assert should is False
    assert reason == ""
