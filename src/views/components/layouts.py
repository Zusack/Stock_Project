"""Layout primitives: title bars and section headers."""
from __future__ import annotations

import os
from typing import Union

import flet as ft

from src.utils.logger_utils import app_logger
from src.views.theme import ThemeHelper


def ViewTitleBar(title: str, right_control: Union[ft.Control, list, None] = None) -> ft.Container:
    """
    Standardized title bar for tab views.

    right_control: single control (e.g. progress bar), list of controls
    (e.g. buttons), or None.
    """
    if os.environ.get("FLET_DIAG_HIDE_VIEW_TITLE_BAR") == "1":
        container = ft.Container(height=1)
        container.data = "view_title_bar"
        return container

    if right_control is None:
        right = ft.Container()
    elif isinstance(right_control, list):
        right = ft.Row(right_control, spacing=10)
    else:
        right = right_control

    # NOTE: Flet 0.84 Linux can incorrectly inflate TextThemeStyle-based title
    # bars to enormous heights (e.g. 100k px). Use explicit size/weight by
    # default and keep an escape hatch for comparing with themed headline style.
    use_headline_style = os.environ.get("FLET_DIAG_USE_HEADLINE_TITLE_BAR") == "1"
    title_text = (
        ft.Text(title, style=ft.TextThemeStyle.HEADLINE_MEDIUM)
        if use_headline_style
        else ft.Text(title, size=20, weight=ft.FontWeight.BOLD)
    )

    container = ft.Container(
        content=ft.Row(
            [
                title_text,
                right,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        height=56,
        padding=ft.padding.symmetric(horizontal=4),
    )
    if os.environ.get("FLET_DIAG_LOG_TITLE_BAR_SIZE") == "1":
        def _log_size(e):
            app_logger.log(
                "DIAG",
                "ViewTitleBar size changed.",
                level="DEBUG",
                title=title,
                width=getattr(e, "width", None),
                height=getattr(e, "height", None),
            )
        container.on_size_change = _log_size
    container.data = "view_title_bar"  # Marker so theme refreshes can skip it.
    return container


class SectionHeader(ft.Column):
    """Bold title with optional leading icon, optional right control, and a divider."""

    def __init__(self, title: str, right_control: ft.Control = None, page_ref=None, icon=None):
        left_controls = []
        if icon is not None:
            left_controls.append(ft.Icon(icon, size=18))
        left_controls.append(ft.Text(title, weight=ft.FontWeight.BOLD))
        row_controls = [ft.Row(left_controls, spacing=6)]
        if right_control:
            row_controls.append(right_control)
        divider_color = ThemeHelper.divider_color(page_ref)

        super().__init__(
            controls=[
                ft.Row(row_controls, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(height=1, color=divider_color),
            ],
            spacing=5,
        )
