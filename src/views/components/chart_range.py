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
    return chart_toggle_selector(
        CHART_RANGE_OPTIONS,
        selected=[initial],
        on_change=on_change,
        allow_multiple=False,
        allow_empty=False,
    )


def chart_toggle_selector(
    options: tuple[tuple[str, str], ...],
    *,
    selected: list[str] | None = None,
    on_change: Callable | None = None,
    allow_multiple: bool = True,
    allow_empty: bool = False,
) -> ft.SegmentedButton:
    """Multi- or single-select SegmentedButton (same UX shell as ``chart_range_selector``)."""
    valid = {key for key, _ in options}
    initial = [key for key in (selected or []) if key in valid]
    if not initial and not allow_empty and options:
        initial = [options[0][0]]
    return ft.SegmentedButton(
        selected=initial,
        allow_empty_selection=allow_empty,
        allow_multiple_selection=allow_multiple,
        segments=[
            ft.Segment(value=key, label=ft.Text(label))
            for key, label in options
        ],
        on_change=on_change,
    )


def selected_chart_toggles(
    selector: ft.SegmentedButton | None,
    *,
    default: list[str] | None = None,
) -> list[str]:
    """Read selected segment values from a toggle SegmentedButton."""
    if selector is None:
        return list(default or [])
    return list(getattr(selector, "selected", None) or default or [])
