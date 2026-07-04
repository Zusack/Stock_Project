"""Calibration gate tests for leaderboard score versions."""

from src.analysis.leaderboard_calibration import rollout_gate_passes


def test_rollout_gate_passes_with_stable_correlation():
    assert rollout_gate_passes({"ok": True, "correlation_v1_v2": 0.75, "count": 20})


def test_rollout_gate_fails_low_correlation():
    assert not rollout_gate_passes({"ok": True, "correlation_v1_v2": 0.3, "count": 20})


def test_rollout_gate_fails_empty():
    assert not rollout_gate_passes({"ok": False, "count": 0})
