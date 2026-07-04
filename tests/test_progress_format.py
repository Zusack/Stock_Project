"""Tests for shared progress formatting helpers."""

from src.utils.progress_format import (
    estimate_eta_seconds,
    format_duration,
    format_elapsed,
    format_eta_remaining,
)


def test_format_duration_seconds():
    assert format_duration(45) == "45 sec"


def test_format_eta_remaining():
    assert format_eta_remaining(90) == "About 1 min 30 sec remaining"


def test_estimate_eta_seconds():
    eta = estimate_eta_seconds(elapsed_sec=100.0, done=10, total=100)
    assert eta == 900.0


def test_format_elapsed():
    assert format_elapsed(120) == "Elapsed 2 min"
