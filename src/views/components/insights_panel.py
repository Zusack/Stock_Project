"""Insights / suggestions panel (rule-based now, LM Studio-ready)."""

from __future__ import annotations

import flet as ft

from src.analysis.ai.advisor import SignalSuggestion
from src.views.theme import Palette, ThemeHelper


def _severity_color(page: ft.Page, severity: str) -> str:
    if severity == "action":
        return ThemeHelper.accent_green(page)
    if severity == "risk":
        return ThemeHelper.text_error(page)
    if severity == "watch":
        return Palette.amber.s600 if ThemeHelper.is_dark(page) else Palette.amber.s800
    return ThemeHelper.text_muted(page)


def build_insights_panel(
    page: ft.Page,
    suggestions: list[SignalSuggestion],
    *,
    title: str = "Insights & Suggestions",
) -> ft.Control:
    if not suggestions:
        return ft.Text(
            "No suggestions available. Run a guidance scan or refresh Leaderboard.",
            size=12,
            color=ThemeHelper.text_muted(page),
        )

    items: list[ft.Control] = []
    for s in suggestions:
        items.append(
            ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(
                                    ft.Icons.LIGHTBULB_OUTLINE,
                                    size=18,
                                    color=_severity_color(page, s.severity),
                                ),
                                ft.Text(s.headline, weight=ft.FontWeight.W_600, size=14),
                            ],
                            spacing=8,
                        ),
                        ft.Text(s.detail, size=12, color=ThemeHelper.text_muted(page)),
                        ft.Text(
                            f"Provider: {s.provider}",
                            size=10,
                            color=ThemeHelper.text_muted(page),
                            italic=True,
                        ),
                    ],
                    spacing=4,
                    tight=True,
                ),
                padding=10,
                border=ft.border.all(1, ThemeHelper.border_subtle(page)),
                border_radius=8,
            )
        )

    return ft.Column(
        [
            ft.Text(title, size=16, weight=ft.FontWeight.W_600),
            *items,
        ],
        spacing=8,
    )
