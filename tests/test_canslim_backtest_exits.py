"""Tests for rulebook-driven exit evaluation."""

from src.analysis.canslim_backtest import evaluate_exit
from src.analysis.canslim_rulebook import get_rule_set


def test_stop_loss_first():
    rules = get_rule_set()
    should, reason = evaluate_exit(-0.09, 90.0, 100.0, 0.0, rules)
    assert should is True
    assert reason == "Stop Loss"


def test_take_profit():
    rules = get_rule_set()
    should, reason = evaluate_exit(0.26, 126.0, 100.0, 0.2, rules)
    assert should is True
    assert reason == "Take Profit"


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
