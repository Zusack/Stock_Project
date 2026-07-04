"""Display components: StatusBanner, ReadOnlyModeBanner, InspectorField."""
from __future__ import annotations

import flet as ft

from src.views.components.layouts import SectionHeader
from src.views.theme import MotionSpec, ThemeHelper, TRANSPARENT


class StatusBanner(ft.Container):
    """
    Displays the current status of the application (e.g., 'Status: Idle').
    Animates the background color when state changes.
    """

    def __init__(self):
        super().__init__(
            padding=10,
            border_radius=5,
            bgcolor=TRANSPARENT,
            animate=MotionSpec.animation(MotionSpec.STATUS_MS),
        )
        # Avoid TextThemeStyle inflation bug on Linux/Flet 0.84.
        self.label = ft.Text("Status: Idle.", size=14, weight=ft.FontWeight.W_500)
        self.content = self.label

    def set_status(self, text: str, state: str = "IDLE") -> None:
        self.label.value = text
        state_map = {
            "IDLE": "idle",
            "RUNNING": "running",
            "ERROR": "error",
            "SUCCESS": "success",
            "PAUSED": "warning",
        }
        sev = state_map.get((state or "").upper(), "idle")
        self.bgcolor = ThemeHelper.status_bg(None, sev)
        self.label.color = ThemeHelper.status_text(None, sev)
        try:
            self.update()
        except RuntimeError:
            pass


class ReadOnlyModeBanner(ft.Container):
    """Standardized banner shown when viewing read-only data (e.g. an archived snapshot)."""

    def __init__(self, visible: bool = False, page=None, message: str = "Read-Only Mode"):
        super().__init__(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.LOCK, color=ThemeHelper.status_text(page, "warning")),
                    ft.Text(
                        message,
                        color=ThemeHelper.status_text(page, "warning"),
                        weight=ft.FontWeight.BOLD,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
            ),
            bgcolor=ThemeHelper.status_bg(page, "warning"),
            padding=8,
            border_radius=5,
            visible=visible,
        )
        self._page = page

    def refresh_theme(self, page=None) -> None:
        page = page or self._page
        self.bgcolor = ThemeHelper.status_bg(page, "warning")
        accent = ThemeHelper.status_text(page, "warning")
        row = self.content
        if isinstance(row, ft.Row) and len(row.controls) >= 2:
            row.controls[0].color = accent
            row.controls[1].color = accent


class InspectorField(ft.Column):
    """
    Standardized read-only text field with a header and optional expand button.
    Useful for showing long-form details (descriptions, JSON dumps, news excerpts).
    """

    def __init__(self, title: str, value: str = "", on_expand=None, expand: bool = False):
        super().__init__(expand=expand)
        self.title = title
        self.on_expand = on_expand

        self.text_control = ft.Text(
            value,
            selectable=True,
            no_wrap=False,
            overflow=ft.TextOverflow.ELLIPSIS,
            font_family="monospace",
            size=12,
        )

        expand_btn = None
        if on_expand:
            expand_btn = ft.IconButton(
                ft.Icons.OPEN_IN_FULL,
                tooltip=f"Expand {title}",
                icon_size=18,
                on_click=lambda e: self.on_expand(self.title, self.text_control.value),
            )

        self.header = SectionHeader(title, right_control=expand_btn)

        self.container = ft.Container(
            content=self.text_control,
            padding=10,
            bgcolor=ThemeHelper.surface_dim(None),
            border_radius=5,
            border=ft.border.all(1, ThemeHelper.border_default(None)),
            expand=True,
        )

        self.controls = [self.header, self.container]

    def set_value(self, new_value: str) -> None:
        self.text_control.value = new_value if new_value else "(Empty)"
        try:
            self.update()
        except RuntimeError:
            pass
