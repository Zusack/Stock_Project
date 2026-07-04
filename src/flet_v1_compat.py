"""
Compatibility helpers for Flet 0.85+ / v1.

Restores legacy module-level helpers (ft.padding.only, ft.border.all),
re-exports chart classes from flet_charts under ft.* names, and adds a
few legacy properties (Button.text, Icon.name) so older view code
continues to work. Import this module once at startup — apply_flet_v1_compat()
runs automatically.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import flet as ft
import flet_charts as fchart


def _patch_page_legacy_dialogs() -> None:
    if not hasattr(ft.Page, "open"):
        def _open(self: ft.Page, dialog: ft.Control) -> None:
            self.show_dialog(dialog)

        setattr(ft.Page, "open", _open)

    if not hasattr(ft.Page, "close"):
        def _close(self: ft.Page, _dialog: Optional[ft.Control] = None) -> None:
            try:
                self.pop_dialog()
            except Exception:
                pass

        setattr(ft.Page, "close", _close)

    if not hasattr(ft.Page, "on_resized"):
        def _get_on_resized(self: ft.Page) -> Any:
            return self.on_resize

        def _set_on_resized(self: ft.Page, handler: Any) -> None:
            self.on_resize = handler

        setattr(ft.Page, "on_resized", property(_get_on_resized, _set_on_resized))


def _patch_module_helpers() -> None:
    if hasattr(ft, "padding") and not hasattr(ft.padding, "only"):
        ft.padding.only = ft.Padding.only  # type: ignore[attr-defined]
    if hasattr(ft, "padding") and not hasattr(ft.padding, "symmetric"):
        ft.padding.symmetric = ft.Padding.symmetric  # type: ignore[attr-defined]
    if hasattr(ft, "border") and not hasattr(ft.border, "all"):
        ft.border.all = ft.Border.all  # type: ignore[attr-defined]
    if hasattr(ft, "border") and not hasattr(ft.border, "only"):
        ft.border.only = ft.Border.only  # type: ignore[attr-defined]


def _patch_alignment_namespace() -> None:
    ft.alignment = SimpleNamespace(
        center=ft.Alignment.CENTER,
        center_left=ft.Alignment.CENTER_LEFT,
        center_right=ft.Alignment.CENTER_RIGHT,
        top_left=ft.Alignment.TOP_LEFT,
        top_center=ft.Alignment.TOP_CENTER,
        top_right=ft.Alignment.TOP_RIGHT,
        bottom_left=ft.Alignment.BOTTOM_LEFT,
        bottom_center=ft.Alignment.BOTTOM_CENTER,
        bottom_right=ft.Alignment.BOTTOM_RIGHT,
    )


def _patch_chart_exports() -> None:
    chart_symbols = [
        "BarChart",
        "BarChartGroup",
        "BarChartRod",
        "BarChartRodStackItem",
        "BarChartRodTooltip",
        "ChartAxis",
        "ChartAxisLabel",
        "ChartGridLines",
        "LineChart",
        "LineChartData",
        "LineChartDataPoint",
        "LineChartDataPoints",
        "LineChartEvent",
        "LineChartTooltipData",
        "LineChartTooltipItem",
        "PieChart",
        "PieChartSection",
        "ScatterChart",
        "ScatterChartData",
        "ScatterChartEvent",
        "ScatterChartSpot",
    ]
    for symbol in chart_symbols:
        if not hasattr(ft, symbol) and hasattr(fchart, symbol):
            setattr(ft, symbol, getattr(fchart, symbol))

    def _compat_line_chart_data(*args: Any, **kwargs: Any):
        if "data_points" in kwargs and "points" not in kwargs:
            kwargs["points"] = kwargs.pop("data_points")
        return fchart.LineChartData(*args, **kwargs)

    def _compat_bar_chart_group(*args: Any, **kwargs: Any):
        if "bar_rods" in kwargs and "rods" not in kwargs:
            kwargs["rods"] = kwargs.pop("bar_rods")
        return fchart.BarChartGroup(*args, **kwargs)

    def _compat_bar_chart(*args: Any, **kwargs: Any):
        if "bar_groups" in kwargs and "groups" not in kwargs:
            kwargs["groups"] = kwargs.pop("bar_groups")
        try:
            return fchart.BarChart(*args, **kwargs)
        except TypeError:
            kwargs.pop("tooltip_bgcolor", None)
            return fchart.BarChart(*args, **kwargs)

    def _compat_line_chart(*args: Any, **kwargs: Any):
        try:
            return fchart.LineChart(*args, **kwargs)
        except TypeError:
            kwargs.pop("tooltip_bgcolor", None)
            return fchart.LineChart(*args, **kwargs)

    def _compat_chart_axis(*args: Any, **kwargs: Any):
        if "labels_size" in kwargs and "label_size" not in kwargs:
            kwargs["label_size"] = kwargs.pop("labels_size")
        if "labels_interval" in kwargs and "label_spacing" not in kwargs:
            kwargs["label_spacing"] = kwargs.pop("labels_interval")
        else:
            kwargs.pop("labels_interval", None)
        return fchart.ChartAxis(*args, **kwargs)

    setattr(ft, "LineChartData", _compat_line_chart_data)
    setattr(ft, "BarChartGroup", _compat_bar_chart_group)
    setattr(ft, "BarChart", _compat_bar_chart)
    setattr(ft, "LineChart", _compat_line_chart)
    setattr(ft, "ChartAxis", _compat_chart_axis)


def _patch_legacy_control_properties() -> None:
    button_types = [
        ft.ElevatedButton,
        ft.TextButton,
        ft.OutlinedButton,
        ft.FilledButton,
        ft.FilledTonalButton,
        ft.CupertinoButton,
    ]

    def _text_getter(self: Any) -> str:
        content = getattr(self, "content", None)
        return content if isinstance(content, str) else ""

    def _text_setter(self: Any, value: str) -> None:
        self.content = value

    for button_type in button_types:
        if not hasattr(button_type, "text"):
            setattr(button_type, "text", property(_text_getter, _text_setter))

    if not hasattr(ft.Icon, "name"):
        def _name_getter(self: ft.Icon) -> Any:
            return self.icon

        def _name_setter(self: ft.Icon, value: Any) -> None:
            self.icon = value

        setattr(ft.Icon, "name", property(_name_getter, _name_setter))


class FilePickerResultEvent:
    """Backwards-compatible event shape used by older Flet callbacks."""

    def __init__(self, files: Optional[list[Any]] = None, path: Optional[str] = None):
        self.files = files
        self.path = path


def _patch_file_picker() -> None:
    if not hasattr(ft, "FilePickerResultEvent"):
        setattr(ft, "FilePickerResultEvent", FilePickerResultEvent)


def _patch_legacy_event_aliases() -> None:
    if not hasattr(ft, "WindowResizeEvent"):
        class WindowResizeEvent:
            def __init__(self, **kwargs: Any):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        setattr(ft, "WindowResizeEvent", WindowResizeEvent)


def apply_flet_v1_compat() -> None:
    _patch_page_legacy_dialogs()
    _patch_module_helpers()
    _patch_alignment_namespace()
    _patch_chart_exports()
    _patch_legacy_control_properties()
    _patch_file_picker()
    _patch_legacy_event_aliases()


apply_flet_v1_compat()
