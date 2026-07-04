"""Lightweight UI feedback: snackbars, async status rows, persistent banners."""
from __future__ import annotations

import flet as ft

from src.views.theme import MotionSpec, ThemeHelper, TRANSPARENT


def show_snackbar(
    page: ft.Page | None,
    message: str,
    *,
    severity: str = "info",
    duration_ms: int = 3200,
) -> None:
    """Display a snackbar at the bottom of the page with severity-driven colors."""
    if page is None or not message:
        return
    bg = ThemeHelper.status_bg(page, severity)
    fg = ThemeHelper.status_text(page, severity)
    snack = ft.SnackBar(
        content=ft.Text(message, color=fg),
        bgcolor=bg if bg != TRANSPARENT else ThemeHelper.surface_raised(page),
        duration=ft.Duration(milliseconds=max(1500, int(duration_ms))),
    )
    try:
        page.show_dialog(snack)
        page.update()
    except Exception:
        try:
            page.update()
        except Exception:
            pass


class AsyncStatusRow(ft.Container):
    """Spinner + status text + optional progress bar for long-running work."""

    def __init__(self, page=None, *, visible: bool = False):
        self._page = page
        self.ring = ft.ProgressRing(width=22, height=22, stroke_width=2, visible=False)
        self.label = ft.Text("", size=12, expand=True, color=ThemeHelper.text_muted(page))
        self.bar = ft.ProgressBar(value=0, visible=False)
        super().__init__(
            content=ft.Column(
                [
                    ft.Row(
                        [self.ring, self.label],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.bar,
                ],
                spacing=6,
                tight=True,
            ),
            visible=visible,
            padding=ft.Padding(0, 0, 0, 4),
            animate=MotionSpec.animation(MotionSpec.STATUS_MS),
        )

    def set_idle(self, message: str = "") -> None:
        self.visible = bool(message)
        self.ring.visible = False
        self.bar.visible = False
        self.label.value = message
        self.label.color = ThemeHelper.text_muted(self._page)
        self._touch()

    def set_running(self, message: str, *, progress: float | None = None) -> None:
        self.visible = True
        self.ring.visible = True
        self.label.value = message
        self.label.color = ThemeHelper.text_primary(self._page)
        if progress is not None:
            self.bar.visible = True
            self.bar.value = max(0.0, min(1.0, progress))
        else:
            self.bar.visible = False
        self._touch()

    def set_success(self, message: str) -> None:
        self.visible = bool(message)
        self.ring.visible = False
        self.bar.visible = False
        self.label.value = message
        self.label.color = ThemeHelper.accent_green(self._page)
        self._touch()

    def set_error(self, message: str) -> None:
        self.visible = bool(message)
        self.ring.visible = False
        self.bar.visible = False
        self.label.value = message
        self.label.color = ThemeHelper.text_error(self._page)
        self._touch()

    def refresh_theme(self, page=None) -> None:
        if page is not None:
            self._page = page
        self.label.color = ThemeHelper.text_muted(self._page)

    def _touch(self) -> None:
        try:
            self.update()
        except RuntimeError:
            pass


class PersistentBanner(ft.Container):
    """Dismissible or sticky banner for warnings that should not be snackbars only."""

    def __init__(
        self,
        page=None,
        *,
        message: str = "",
        severity: str = "warning",
        visible: bool = False,
        dismissible: bool = True,
    ):
        self._page = page
        self._severity = severity
        self._message_text = ft.Text(message, size=12, expand=True)
        close_btn = ft.IconButton(
            ft.Icons.CLOSE,
            icon_size=18,
            tooltip="Dismiss",
            visible=dismissible,
            on_click=self._on_dismiss,
        )
        self._icon = ft.Icon(ft.Icons.INFO_OUTLINE, size=18)
        super().__init__(
            content=ft.Row(
                [self._icon, self._message_text, close_btn],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            visible=visible and bool(message),
            padding=ft.Padding(10, 8, 10, 8),
            border_radius=6,
            border=ft.border.all(1, ThemeHelper.border_default(page)),
        )
        self._apply_colors(severity)

    def show(self, message: str, *, severity: str | None = None) -> None:
        if severity:
            self._severity = severity
        self._message_text.value = message
        self.visible = bool(message)
        self._apply_colors(self._severity)
        self._touch()

    def hide(self) -> None:
        self.visible = False
        self._touch()

    def _on_dismiss(self, _e) -> None:
        self.hide()

    def _apply_colors(self, severity: str) -> None:
        self.bgcolor = ThemeHelper.status_bg(self._page, severity)
        accent = ThemeHelper.status_text(self._page, severity)
        self._message_text.color = accent
        self._icon.color = accent
        icon_map = {
            "error": ft.Icons.ERROR_OUTLINE,
            "warning": ft.Icons.WARNING_AMBER,
            "success": ft.Icons.CHECK_CIRCLE_OUTLINE,
            "info": ft.Icons.INFO_OUTLINE,
        }
        self._icon.name = icon_map.get(severity, ft.Icons.INFO_OUTLINE)

    def refresh_theme(self, page=None) -> None:
        if page is not None:
            self._page = page
        self._apply_colors(self._severity)
        self.border = ft.border.all(1, ThemeHelper.border_default(self._page))

    def _touch(self) -> None:
        try:
            self.update()
        except RuntimeError:
            pass
