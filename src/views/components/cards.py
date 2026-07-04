"""Reusable card components: StatusCard, SelectableMetricCard."""
from __future__ import annotations

import flet as ft

from src.views.theme import ColorPalette, MotionSpec, ThemeHelper


class StatusCard(ft.Container):
    """
    A standardized card used in grids to represent a row of selectable items
    (e.g. a stock in a watchlist, a chart preset, a category).

    Status drives the background fill; selection adds an outline and a subtle
    pop. When clickable (on_click provided), the cursor automatically switches
    to a pointer.
    """

    def __init__(
        self,
        text: str,
        status: str = "TODO",
        sub_text: str | None = None,
        on_click_data=None,
        on_click=None,
        is_selected: bool = False,
        tooltip: str = "",
        outline_color: str | None = None,
        mouse_cursor: ft.MouseCursor | None = None,
        page_ref=None,
    ):
        super().__init__()
        self.text = text
        self.status = status
        self.on_click_data = on_click_data
        self.is_selected = is_selected
        self.tooltip = tooltip or text
        self.outline_color = outline_color
        self.page_ref = page_ref
        self._on_click_handler = on_click
        if mouse_cursor is not None:
            self.mouse_cursor = mouse_cursor

        self.border_radius = 5
        self.animate = MotionSpec.animation(MotionSpec.STATUS_MS)
        self._base_padding = ft.padding.only(left=10, right=5)
        self._base_height = 40
        self._highlighted_height = 46
        self._highlighted_padding = ft.padding.only(left=12, right=7)
        self.on_click = self._handle_click
        self.data = on_click_data

        self.update_visuals()

    def _handle_click(self, e):
        if self._on_click_handler:
            self._on_click_handler(self.on_click_data)

    def update_status(self, new_status: str, is_selected: bool | None = None):
        self.status = new_status
        if is_selected is not None:
            self.is_selected = is_selected
        self.update_visuals()
        try:
            self.update()
        except RuntimeError:
            # Card may not be attached yet when status updates happen during startup.
            pass

    def update_visuals(self):
        if self.status == "IDLE" and self.outline_color:
            self.bgcolor = ColorPalette.IDLE
            self.border = ft.border.all(2, self.outline_color)
            self.shadow = None
            self.height = self._base_height
            self.padding = self._base_padding
        else:
            self.bgcolor = ColorPalette.get_color_for_status(
                self.status, page=self.page_ref, is_selected=False
            )
            is_highlighted = self.is_selected or self.status == "ACTIVE"
            if is_highlighted:
                outline_w = ThemeHelper.outline_selection_width()
                self.border = ft.border.all(outline_w, ThemeHelper.outline_selection(self.page_ref))
                self.shadow = ThemeHelper.shadow_pop(self.page_ref)
                self.height = self._highlighted_height
                self.padding = self._highlighted_padding
            else:
                self.border = None
                self.shadow = None
                self.height = self._base_height
                self.padding = self._base_padding
        icon = ColorPalette.get_icon_for_status(self.status, self.is_selected)
        if self.status in ["ACTIVE", "DONE", "ERROR", "PARTIAL", "INCOMPATIBLE"] or self.is_selected:
            text_color = ThemeHelper.text_on_status(self.page_ref)
        else:
            text_color = ThemeHelper.text_on_status_dim(self.page_ref)

        self.content = ft.Row(
            [
                ft.Icon(icon, size=16, color=text_color),
                ft.Text(
                    self.text,
                    size=12,
                    color=text_color,
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    expand=True,
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=5,
        )


class SelectableMetricCard(ft.Container):
    """Metric summary card with theme-aware selected state."""

    def __init__(
        self,
        page,
        *,
        title: str,
        value: str,
        subtitle: str = "",
        selected: bool = False,
        accent: str = "primary",
        on_click=None,
        width: int | None = None,
    ):
        super().__init__()
        self._page = page
        self._accent = accent
        self._selected = selected
        self._on_click = on_click
        self.width = width
        self.padding = 12
        self.border_radius = 8
        self.on_click = self._handle_click if on_click else None
        self._title_text = ft.Text(title, size=11, weight=ft.FontWeight.W_600)
        self._value_text = ft.Text(value, size=22, weight=ft.FontWeight.BOLD)
        self._subtitle_text = ft.Text(subtitle, size=10) if subtitle else None
        body = [self._title_text, self._value_text]
        if self._subtitle_text:
            body.append(self._subtitle_text)
        self.content = ft.Column(body, spacing=4, tight=True)
        self._apply_style()

    def _handle_click(self, e):
        if self._on_click:
            self._on_click(e)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()
        try:
            self.update()
        except RuntimeError:
            pass

    def set_value(self, value: str, subtitle: str | None = None):
        """Update the displayed value and optional subtitle in place."""
        self._value_text.value = value
        if subtitle is not None and self._subtitle_text is not None:
            self._subtitle_text.value = subtitle
        try:
            self.update()
        except RuntimeError:
            pass

    def _apply_style(self):
        border_c = (
            ThemeHelper.border_metric_selected(self._page, self._accent)
            if self._selected
            else ThemeHelper.border_card(self._page)
        )
        self.border = ft.border.all(2 if self._selected else 1, border_c)
        bg = ThemeHelper.metric_card_bg(self._page, self._selected, self._accent)
        self.bgcolor = bg if bg else None
        primary = ThemeHelper.text_primary(self._page)
        muted = ThemeHelper.text_muted(self._page)
        self._title_text.color = muted
        self._value_text.color = primary
        if self._subtitle_text:
            self._subtitle_text.color = muted
