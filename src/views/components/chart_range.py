"""Shared chart time-range SegmentedButton (Dashboard-style UX)."""

from __future__ import annotations

from typing import Callable

import flet as ft

# Canonical daily-chart ranges used by Dashboard, Compare, and Stock Detail.
CHART_RANGE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("week", "Week"),
    ("month", "Month"),
    ("ytd", "YTD"),
    ("1y", "1Y"),
)
DEFAULT_CHART_RANGE = "week"

_LABELS = dict(CHART_RANGE_OPTIONS)

# Days of history to request from the DB for each range (YTD needs a full year buffer).
_FETCH_DAYS: dict[str, int] = {
    "week": 14,
    "1w": 14,
    "month": 45,
    "1m": 45,
    "3m": 120,
    "6m": 200,
    "ytd": 400,
    "1y": 400,
    "5y": 365 * 5 + 30,
}


def chart_range_label(key: str) -> str:
    """Human label for a range key (e.g. ``week`` → ``Week``)."""
    return _LABELS.get(key, (key or "").upper())


def chart_range_fetch_days(key: str) -> int:
    """How many calendar days of prices to load for *key*."""
    return _FETCH_DAYS.get(key, 400)


def selected_chart_range(
    selector: ft.SegmentedButton | None,
    *,
    default: str = DEFAULT_CHART_RANGE,
) -> str:
    """Read the selected range key from a SegmentedButton."""
    if selector is None:
        return default
    sel = list(getattr(selector, "selected", None) or [default])
    return sel[0] if sel else default


def chart_range_selector(
    *,
    selected: str = DEFAULT_CHART_RANGE,
    on_change: Callable | None = None,
) -> ft.SegmentedButton:
    """Dashboard-style Week / Month / YTD / 1Y segmented control."""
    initial = selected if selected in _LABELS else DEFAULT_CHART_RANGE
    return ft.SegmentedButton(
        selected=[initial],
        allow_empty_selection=False,
        segments=[
            ft.Segment(value=key, label=ft.Text(label))
            for key, label in CHART_RANGE_OPTIONS
        ],
        on_change=on_change,
    )
