"""
Reusable button/control patterns. Generic ActionPanel for views that need a
Start/Pause/Resume/Stop button row.
"""
from __future__ import annotations

import flet as ft

from src.views.theme import ButtonStyles, InputStyles


class ActionPanel(ft.Row):
    """
    A unified panel of action buttons. Generic alternative to a
    benchmark-specific control panel; pass in only the callbacks you need.

    Usage:
        panel = ActionPanel(
            start_text="Fetch Quotes",
            on_start=self._on_fetch,
            on_stop=self._on_cancel,
        )
        panel.set_state("IDLE")
    """

    def __init__(
        self,
        *,
        start_text: str = "Start",
        start_icon=ft.Icons.PLAY_ARROW,
        on_start=None,
        on_pause=None,
        on_resume=None,
        on_skip=None,
        on_stop=None,
        disabled: bool = False,
    ):
        super().__init__(
            wrap=True,
            spacing=10,
            run_spacing=10,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        self.btn_start = ft.ElevatedButton(
            content=start_text,
            icon=start_icon,
            style=ButtonStyles.primary(),
            on_click=on_start,
            disabled=disabled,
            tooltip="Start the action.",
        ) if on_start else None

        self.btn_pause = ft.ElevatedButton(
            content="Pause",
            icon=ft.Icons.PAUSE,
            style=ButtonStyles.primary(),
            on_click=on_pause,
            disabled=True,
            tooltip="Pause the current action.",
        ) if on_pause else None

        self.btn_resume = ft.ElevatedButton(
            content="Resume",
            icon=ft.Icons.PLAY_ARROW_OUTLINED,
            style=ButtonStyles.primary(),
            on_click=on_resume,
            disabled=True,
            tooltip="Resume the action.",
        ) if on_resume else None

        self.btn_skip = ft.ElevatedButton(
            content="Skip Current",
            icon=ft.Icons.SKIP_NEXT,
            style=ButtonStyles.primary(),
            on_click=on_skip,
            disabled=True,
            tooltip="Skip the current item.",
        ) if on_skip else None

        self.btn_stop = ft.ElevatedButton(
            content="Stop",
            icon=ft.Icons.STOP,
            style=ButtonStyles.destructive(),
            on_click=on_stop,
            disabled=True,
            tooltip="Stop the action.",
        ) if on_stop else None

        self.controls = [
            btn
            for btn in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_skip, self.btn_stop)
            if btn is not None
        ]

    def set_state(self, state: str) -> None:
        """
        Update button states based on high-level process state.

        Options: 'IDLE', 'RUNNING', 'PAUSED'.
        """
        state = (state or "").upper()
        if self.btn_start:
            self.btn_start.disabled = state != "IDLE"
        if self.btn_pause:
            self.btn_pause.disabled = state != "RUNNING"
        if self.btn_resume:
            self.btn_resume.disabled = state != "PAUSED"
        if self.btn_skip:
            self.btn_skip.disabled = state not in ("RUNNING", "PAUSED")
        if self.btn_stop:
            self.btn_stop.disabled = state == "IDLE"
        try:
            self.update()
        except RuntimeError:
            pass


class ContextSelector(ft.Row):
    """
    A small button that opens a modal to change a numeric setting in-place.

    Reused from the parent project — useful for "Chart Window: 30 days" /
    "Top N: 25" / "Lookback: 90" style affordances.
    """

    def __init__(self, page: ft.Page, current_value: int, on_change, *, label: str = "Value"):
        super().__init__(alignment=ft.MainAxisAlignment.CENTER)
        self.page_ref = page
        self.current_value = current_value
        self.on_change = on_change
        self.label = label

        self.btn = ft.OutlinedButton(
            content=f"{label}: {self.current_value}",
            icon=ft.Icons.SETTINGS,
            on_click=self.open_modal,
            tooltip=f"Change the {label.lower()}.",
        )

        self.input_field = InputStyles.text_field(
            page, value=str(self.current_value), label=label
        )

        self.dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Edit {label}"),
            content=ft.Column(
                [
                    ft.Text(f"Enter a new {label.lower()}."),
                    self.input_field,
                ],
                height=120,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=self.close_modal),
                ft.ElevatedButton("Apply", on_click=self.apply_change),
            ],
        )

        self.controls = [self.btn]

    def open_modal(self, e):
        self.input_field.value = str(self.current_value)
        self.page_ref.open(self.dlg)
        self.page_ref.update()

    def close_modal(self, e):
        self.page_ref.close(self.dlg)
        self.page_ref.update()

    def apply_change(self, e):
        try:
            val = int(self.input_field.value)
            self.current_value = val
            self.btn.content = f"{self.label}: {val}"
            self.btn.update()
            if self.on_change:
                self.on_change(val)
            self.close_modal(None)
        except ValueError:
            self.input_field.error_text = "Must be an integer"
            self.input_field.update()
