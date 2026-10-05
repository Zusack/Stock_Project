"""Tests for shared chart range / toggle SegmentedButton helpers."""

from __future__ import annotations

from src.flet_v1_compat import apply_flet_v1_compat  # noqa: F401
from src.views.components.chart_range import (
    chart_toggle_selector,
    selected_chart_toggles,
)


def test_chart_toggle_selector_multi_select_reads_selection():
    selector = chart_toggle_selector(
        (("a", "A"), ("b", "B")),
        selected=["a", "b"],
        allow_multiple=True,
        allow_empty=True,
    )
    assert selected_chart_toggles(selector) == ["a", "b"]
    selector.selected = ["b"]
    assert selected_chart_toggles(selector) == ["b"]


def test_chart_toggle_selector_single_select_allows_empty():
    selector = chart_toggle_selector(
        (("sma50", "50 SMA"),),
        selected=[],
        allow_multiple=False,
        allow_empty=True,
    )
    assert selected_chart_toggles(selector, default=[]) == []
    selector.selected = ["sma50"]
    assert selected_chart_toggles(selector, default=[]) == ["sma50"]
