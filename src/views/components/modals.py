"""
Reusable modal and dialog components.

Generic confirm dialog, expandable text inspector, and an export-success
dialog. Add app-specific modals beside these as needed.
"""
from __future__ import annotations

import flet as ft

from src.views.theme import ButtonStyles, InputStyles, ThemeHelper


class ConfirmationDialog:
    """Reusable confirmation dialog."""

    def __init__(
        self,
        title: str,
        message: str,
        on_confirm,
        on_cancel=None,
        confirm_text: str = "Yes",
        cancel_text: str = "Cancel",
        destructive: bool = False,
    ):
        self.title = title
        self.message = message
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.confirm_text = confirm_text
        self.cancel_text = cancel_text
        self.destructive = destructive

        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Text(message, color=ThemeHelper.text_error(None) if destructive else None),
            actions=[
                ft.TextButton(cancel_text, on_click=self._handle_cancel),
                ft.ElevatedButton(
                    confirm_text,
                    style=ButtonStyles.destructive() if destructive else ButtonStyles.primary(),
                    on_click=self._handle_confirm,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

    def _handle_confirm(self, e):
        page_ref = e.page if hasattr(e, "page") else None
        if self.on_confirm:
            self.on_confirm(e)
        if page_ref:
            page_ref.close(self.dialog)
            page_ref.update()

    def _handle_cancel(self, e):
        page_ref = e.page if hasattr(e, "page") else None
        if self.on_cancel:
            self.on_cancel(e)
        if page_ref:
            page_ref.close(self.dialog)
            page_ref.update()

    def show(self, page_ref: ft.Page) -> None:
        page_ref.open(self.dialog)
        page_ref.update()

    def close(self, page_ref: ft.Page) -> None:
        page_ref.close(self.dialog)
        page_ref.update()


def show_expandable_text(
    page: ft.Page | None,
    *,
    title: str,
    text: str,
    max_width: int = 1400,
    max_height: int = 900,
    on_close=None,
) -> None:
    """Shared large-text inspector dialog used across views."""
    if page is None:
        return
    field = InputStyles.text_field(
        page,
        value=text or "",
        multiline=True,
        read_only=True,
        text_size=14,
        expand=True,
        text_style=ft.TextStyle(font_family="monospace"),
    )
    dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text(title),
        content=ft.Container(content=field, width=max_width, height=max_height),
        actions=[
            ft.TextButton(
                "Close",
                on_click=lambda e: _close_expand_dialog(page, dlg, on_close),
            )
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    page.open(dlg)
    page.update()


def _close_expand_dialog(page: ft.Page, dlg: ft.AlertDialog, on_close=None):
    try:
        page.close(dlg)
    except Exception:
        pass
    if on_close:
        on_close()
    try:
        page.update()
    except Exception:
        pass


class ExportSuccessDialog:
    """Dialog shown after a successful file export."""

    def __init__(self, export_path: str, on_open_folder=None):
        self.export_path = export_path
        self.on_open_folder = on_open_folder
        self.export_path_label = ft.Text("", italic=True, size=12, selectable=True)

        actions = [ft.TextButton("Close", on_click=self._handle_close)]
        if on_open_folder:
            actions.append(
                ft.ElevatedButton(
                    "Open Folder",
                    icon=ft.Icons.FOLDER_OPEN,
                    on_click=self._handle_open_folder,
                    tooltip="Open the folder containing the exported file.",
                )
            )

        self.dialog = ft.AlertDialog(
            title=ft.Text("Export Complete"),
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.CHECK_CIRCLE, color=ThemeHelper.accent_green(None), size=40),
                    ft.Text("File exported successfully."),
                    self.export_path_label,
                ],
                height=120,
            ),
            actions=actions,
        )

        self.export_path_label.value = export_path

    def _handle_open_folder(self, e):
        page_ref = e.page if hasattr(e, "page") else None
        if self.on_open_folder:
            self.on_open_folder(e)
        if page_ref:
            page_ref.close(self.dialog)
            page_ref.update()

    def _handle_close(self, e):
        page_ref = e.page if hasattr(e, "page") else None
        if page_ref:
            page_ref.close(self.dialog)
            page_ref.update()

    def show(self, page_ref: ft.Page) -> None:
        page_ref.open(self.dialog)
        page_ref.update()
