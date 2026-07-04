"""
Central theme tokens (Tailwind-inspired) and ThemeHelper accessors.

All view code should use ThemeHelper / ColorPalette — not raw ft.Colors —
except inside this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flet as ft

from src.services.theme_settings import app_theme


# Tailwind v3 hex stops (subset used by the app)
_SHADES = ("50", "100", "200", "400", "600", "800", "900")


@dataclass(frozen=True)
class ColorFamily:
    s50: str
    s100: str
    s200: str
    s400: str
    s600: str
    s800: str
    s900: str

    def pick(self, dark: bool, dark_stop: str = "800", light_stop: str = "200") -> str:
        stop = dark_stop if dark else light_stop
        return getattr(self, f"s{stop}")


class Palette:
    slate = ColorFamily(
        s50="#f8fafc", s100="#f1f5f9", s200="#e2e8f0", s400="#94a3b8",
        s600="#475569", s800="#1e293b", s900="#0f172a",
    )
    indigo = ColorFamily(
        s50="#eef2ff", s100="#e0e7ff", s200="#c7d2fe", s400="#818cf8",
        s600="#4f46e5", s800="#3730a3", s900="#312e81",
    )
    emerald = ColorFamily(
        s50="#ecfdf5", s100="#d1fae5", s200="#a7f3d0", s400="#34d399",
        s600="#059669", s800="#065f46", s900="#064e3b",
    )
    amber = ColorFamily(
        s50="#fffbeb", s100="#fef3c7", s200="#fde68a", s400="#fbbf24",
        s600="#d97706", s800="#92400e", s900="#78350f",
    )
    rose = ColorFamily(
        s50="#fff1f2", s100="#ffe4e6", s200="#fecdd3", s400="#fb7185",
        s600="#e11d48", s800="#9f1239", s900="#881337",
    )
    sky = ColorFamily(
        s50="#f0f9ff", s100="#e0f2fe", s200="#bae6fd", s400="#38bdf8",
        s600="#0284c7", s800="#075985", s900="#0c4a6e",
    )
    violet = ColorFamily(
        s50="#f5f3ff", s100="#ede9fe", s200="#ddd6fe", s400="#a78bfa",
        s600="#7c3aed", s800="#5b21b6", s900="#4c1d95",
    )
    teal = ColorFamily(
        s50="#f0fdfa", s100="#ccfbf1", s200="#99f6e4", s400="#2dd4bf",
        s600="#0d9488", s800="#115e59", s900="#134e4a",
    )
    cyan = ColorFamily(
        s50="#ecfeff", s100="#cffafe", s200="#a5f3fc", s400="#22d3ee",
        s600="#0891b2", s800="#155e75", s900="#164e63",
    )


# Re-export for views (only theme.py may reference ft.Colors directly in new code).
TRANSPARENT = ft.Colors.TRANSPARENT


class MotionSpec:
    """Shared timing/easing values for subtle UI motion."""

    BUTTON_MS = 180
    TAB_MS = 260
    STATUS_MS = 500
    CONTENT_MS = 220

    @staticmethod
    def animation(duration_ms: int, curve=ft.AnimationCurve.EASE_IN_OUT):
        return ft.Animation(duration_ms, curve)


def _dark(page=None) -> bool:
    return ThemeHelper.is_dark(page)


class ColorPalette:
    """Generic status fills and chart palette."""

    # Legacy constants (dark-theme defaults; prefer get_color_for_status(page=...) in new code)
    TODO = Palette.slate.s800
    ACTIVE = Palette.indigo.s900
    DONE = Palette.emerald.s900
    FAIL = Palette.rose.s900
    IDLE = Palette.slate.s900
    WARNING = Palette.amber.s900
    INCOMPATIBLE = Palette.violet.s900
    TEXT_ON_STATUS = Palette.slate.s50
    TEXT_ON_STATUS_DIM = ft.Colors.with_opacity(0.78, Palette.slate.s50)
    TEXT_PRIMARY = Palette.slate.s50
    TEXT_SECONDARY = ft.Colors.with_opacity(0.78, Palette.slate.s50)
    TEXT_ERROR = Palette.rose.s400

    CHARTS = [
        Palette.rose.s400, Palette.sky.s400, Palette.emerald.s400,
        Palette.amber.s400, Palette.violet.s400, Palette.teal.s400,
        Palette.indigo.s400, Palette.cyan.s400,
    ]

    @staticmethod
    def get_color_for_status(status: str, page=None, is_selected: bool = False):
        status = (status or "").upper()
        dark = _dark(page)
        if status == "ACTIVE":
            return Palette.indigo.s900 if dark else Palette.indigo.s100
        if status == "DONE":
            return Palette.emerald.s900 if dark else Palette.emerald.s100
        if status in ("ERROR", "FAIL"):
            return Palette.rose.s900 if dark else Palette.rose.s100
        if status == "PARTIAL":
            return Palette.amber.s900 if dark else Palette.amber.s100
        if status == "IDLE":
            return Palette.slate.s900 if dark else Palette.slate.s200
        if status == "INCOMPATIBLE":
            return Palette.violet.s900 if dark else Palette.violet.s100
        return Palette.slate.s800 if dark else Palette.slate.s100

    @staticmethod
    def get_icon_for_status(status: str, is_selected: bool = False):
        if is_selected:
            return ft.Icons.CHECK_CIRCLE
        status = (status or "").upper()
        if status == "ACTIVE":
            return ft.Icons.PLAY_CIRCLE_OUTLINE
        if status == "DONE":
            return ft.Icons.CHECK_CIRCLE_OUTLINE
        if status in ("ERROR", "FAIL"):
            return ft.Icons.ERROR_OUTLINE
        if status == "PARTIAL":
            return ft.Icons.PIE_CHART_OUTLINE
        if status == "IDLE":
            return ft.Icons.DO_NOT_DISTURB
        if status == "INCOMPATIBLE":
            return ft.Icons.BLOCK
        return ft.Icons.CIRCLE_OUTLINED

    @staticmethod
    def text_for_surface(page=None):
        if page is None:
            return None
        return ThemeHelper.text_primary(page)

    @staticmethod
    def chart_palette(page=None):
        dark = _dark(page)
        if dark:
            return [
                Palette.sky.s200, Palette.teal.s200, Palette.amber.s200,
                Palette.violet.s200, Palette.cyan.s200, Palette.emerald.s200,
                Palette.amber.s400, Palette.rose.s200,
            ]
        return [
            Palette.indigo.s800, Palette.teal.s800, Palette.amber.s800,
            Palette.violet.s800, Palette.cyan.s800, Palette.emerald.s800,
            Palette.amber.s600, Palette.rose.s800,
        ]


class ThemeHelper:
    """Theme-aware colors for surfaces, charts, status, and selection."""

    @staticmethod
    def is_dark(page=None) -> bool:
        return app_theme().is_dark_effective()

    @staticmethod
    def text_primary(page=None) -> str:
        return Palette.slate.s50 if _dark(page) else Palette.slate.s900

    @staticmethod
    def text_muted(page=None) -> str:
        return Palette.slate.s200 if _dark(page) else Palette.slate.s600

    @staticmethod
    def text_on_status(page=None) -> str:
        return Palette.slate.s50

    @staticmethod
    def text_on_status_dim(page=None) -> str:
        """Secondary on status blocks — ≥4.5:1 on amber/rose dark fills."""
        return ft.Colors.with_opacity(0.78, Palette.slate.s50)

    @staticmethod
    def text_on_dark_box(page=None) -> str:
        return Palette.slate.s50

    @staticmethod
    def chart_grid_color(page=None) -> str:
        base = Palette.slate.s50 if _dark(page) else Palette.slate.s900
        return ft.Colors.with_opacity(0.2 if _dark(page) else 0.35, base)

    @staticmethod
    def chart_axis_label(page=None) -> str:
        return Palette.slate.s200 if _dark(page) else Palette.slate.s800

    @staticmethod
    def chart_series(page, idx: int) -> str:
        palette = ColorPalette.chart_palette(page)
        return palette[idx % len(palette)]

    @staticmethod
    def chart_named(page, key: str) -> str:
        """Named accents for common chart roles. Add more entries here as your
        app grows (e.g. "price", "volume", "ma_50", "ma_200")."""
        dark = _dark(page)
        mapping_dark = {
            "price": Palette.sky.s200,
            "volume": Palette.amber.s200,
            "gain": Palette.emerald.s200,
            "loss": Palette.rose.s200,
            "ma_short": Palette.violet.s200,
            "ma_long": Palette.cyan.s200,
            "warning": Palette.amber.s400,
        }
        mapping_light = {
            "price": Palette.indigo.s800,
            "volume": Palette.amber.s800,
            "gain": Palette.emerald.s800,
            "loss": Palette.rose.s800,
            "ma_short": Palette.violet.s800,
            "ma_long": Palette.cyan.s800,
            "warning": Palette.amber.s800,
        }
        mapping = mapping_dark if dark else mapping_light
        return mapping.get(key, ThemeHelper.chart_series(page, 0))

    @staticmethod
    def chart_empty_text(page=None) -> str:
        return ThemeHelper.text_muted(page)

    @staticmethod
    def accent_blue(page=None) -> str:
        return Palette.sky.s200 if _dark(page) else Palette.indigo.s800

    @staticmethod
    def accent_green(page=None) -> str:
        return Palette.emerald.s400 if _dark(page) else Palette.emerald.s800

    @staticmethod
    def accent_yellow(page=None) -> str:
        return Palette.amber.s400 if _dark(page) else Palette.amber.s800

    @staticmethod
    def accent_red(page=None) -> str:
        return Palette.rose.s400 if _dark(page) else Palette.rose.s800

    @staticmethod
    def rating_display(page=None) -> str:
        """Large numeric displays."""
        return Palette.sky.s200 if _dark(page) else Palette.indigo.s800

    @staticmethod
    def progress_primary(page=None) -> str:
        return Palette.indigo.s600

    @staticmethod
    def progress_color(role: str, page=None) -> str:
        roles = {
            "blue": Palette.indigo.s600,
            "green": Palette.emerald.s600,
            "purple": Palette.violet.s600,
            "teal": Palette.teal.s600,
            "cyan": Palette.cyan.s600,
        }
        return roles.get(role, Palette.indigo.s600)

    @staticmethod
    def surface_emphasis(page=None) -> str:
        """Stronger overlay for emphasized selection cards."""
        return (
            ft.Colors.with_opacity(0.38, Palette.slate.s900)
            if _dark(page)
            else Palette.slate.s200
        )

    @staticmethod
    def success_icon(page=None) -> str:
        return Palette.emerald.s400

    @staticmethod
    def overlay_scrim(page=None) -> str:
        return ft.Colors.with_opacity(0.8, Palette.slate.s900)

    @staticmethod
    def overlay_panel_bg(page=None) -> str:
        return Palette.slate.s900 if _dark(page) else Palette.slate.s50

    @staticmethod
    def row_highlight(base: str, page=None) -> str:
        return ft.Colors.with_opacity(0.15, base)

    @staticmethod
    def heatmap_green_bg(page=None) -> str:
        return Palette.emerald.s900 if _dark(page) else Palette.emerald.s100

    @staticmethod
    def heatmap_text_on_green(page=None) -> str:
        return Palette.slate.s50 if _dark(page) else Palette.slate.s900

    @staticmethod
    def border_default(page=None) -> str:
        return Palette.slate.s800 if _dark(page) else Palette.slate.s400

    @staticmethod
    def border_subtle(page=None) -> str:
        """Grid lines / inner borders — lighter than border_default in dark mode."""
        return Palette.slate.s600 if _dark(page) else Palette.slate.s200

    @staticmethod
    def border_strong(page=None) -> str:
        """Sidebar / panel dividers."""
        return Palette.slate.s800 if _dark(page) else Palette.slate.s300

    @staticmethod
    def border_card(page=None) -> str:
        return Palette.slate.s800 if _dark(page) else Palette.slate.s300

    @staticmethod
    def border_metric_selected(page=None, accent: str = "primary") -> str:
        accents = {
            "primary": Palette.indigo,
            "teal": Palette.teal,
            "amber": Palette.amber,
            "blue": Palette.sky,
            "indigo": Palette.indigo,
        }
        fam = accents.get(accent, Palette.indigo)
        return fam.s400 if _dark(page) else fam.s600

    @staticmethod
    def surface_dim(page=None) -> str:
        return ft.Colors.with_opacity(0.12, ft.Colors.BLACK) if _dark(page) else Palette.slate.s100

    @staticmethod
    def surface_raised(page=None) -> str:
        return ft.Colors.with_opacity(0.26, ft.Colors.BLACK) if _dark(page) else Palette.slate.s200

    @staticmethod
    def heading_row_color(page=None) -> str:
        return ThemeHelper.surface_dim(page)

    @staticmethod
    def results_summary_data_table_kwargs(page=None) -> dict:
        border_color = ThemeHelper.border_default(page)
        line_color = ThemeHelper.border_subtle(page)
        line = ft.BorderSide(1, line_color)
        primary = ThemeHelper.text_primary(page)
        return {
            "heading_row_color": ThemeHelper.heading_row_color(page),
            "border": ft.Border.all(1, border_color),
            "border_radius": 6,
            "horizontal_lines": line,
            "vertical_lines": line,
            "heading_row_height": 40,
            "data_row_min_height": 40,
            "data_row_max_height": 56,
            "heading_text_style": ft.TextStyle(
                size=12,
                weight=ft.FontWeight.W_600,
                color=primary,
            ),
            "data_text_style": ft.TextStyle(size=12, color=primary),
            "column_spacing": 20,
            "show_bottom_border": True,
        }

    @staticmethod
    def text_error(page=None) -> str:
        return Palette.rose.s400 if _dark(page) else Palette.rose.s800

    @staticmethod
    def divider_color(page=None) -> str:
        return getattr(ft.Colors, "OUTLINE", None) or ThemeHelper.border_default(page)

    @staticmethod
    def icon_toolbar(page=None) -> str:
        return ThemeHelper.text_primary(page)

    @staticmethod
    def icon_disabled(page=None) -> str:
        if _dark(page):
            return ft.Colors.with_opacity(0.30, Palette.slate.s50)
        return Palette.slate.s600

    @staticmethod
    def icon_toolbar_button_style(page=None) -> ft.ButtonStyle:
        dark = _dark(page)
        hover_base = Palette.slate.s50 if dark else Palette.slate.s900
        return ft.ButtonStyle(
            color={
                ft.ControlState.DEFAULT: ThemeHelper.icon_toolbar(page),
                ft.ControlState.DISABLED: ThemeHelper.icon_disabled(page),
            },
            bgcolor={
                ft.ControlState.DEFAULT: ft.Colors.TRANSPARENT,
                ft.ControlState.DISABLED: (
                    ft.Colors.with_opacity(0.14, Palette.slate.s50)
                    if dark
                    else ft.Colors.with_opacity(0.10, Palette.slate.s900)
                ),
            },
            overlay_color=ft.Colors.with_opacity(0.10, hover_base),
            animation_duration=MotionSpec.BUTTON_MS,
        )

    @staticmethod
    def chart_tooltip_bg(page=None) -> str:
        return Palette.slate.s900 if _dark(page) else Palette.slate.s800

    @staticmethod
    def status_bg(page=None, severity: str = "info") -> str:
        sev = (severity or "info").lower().strip()
        dark = _dark(page)
        if sev in ("error", "danger", "fail"):
            return Palette.rose.s900 if dark else Palette.rose.s100
        if sev in ("success", "done", "ok"):
            return Palette.emerald.s900 if dark else Palette.emerald.s100
        if sev in ("warning", "warn"):
            return Palette.amber.s900 if dark else Palette.amber.s100
        if sev in ("active", "running"):
            return Palette.indigo.s900 if dark else Palette.indigo.s100
        return ft.Colors.TRANSPARENT

    @staticmethod
    def status_text(page=None, severity: str = "info") -> str:
        sev = (severity or "info").lower().strip()
        if sev in ("idle", "neutral"):
            return ThemeHelper.text_primary(page)
        if _dark(page):
            return Palette.slate.s50
        if sev in ("warning", "success", "active", "running"):
            return Palette.slate.s900
        return Palette.rose.s900

    @staticmethod
    def card_feature_bg(page=None) -> str:
        return Palette.indigo.s900 if _dark(page) else Palette.indigo.s100

    @staticmethod
    def card_selected_bg(page=None) -> str:
        return Palette.indigo.s900 if _dark(page) else Palette.indigo.s100

    @staticmethod
    def metric_card_bg(page=None, selected: bool = False, accent: str = "primary") -> Optional[str]:
        if not selected:
            return None
        accents = {
            "teal": Palette.teal,
            "amber": Palette.amber,
            "blue": Palette.sky,
            "indigo": Palette.indigo,
            "primary": Palette.indigo,
        }
        fam = accents.get(accent, Palette.indigo)
        return ft.Colors.with_opacity(0.12, fam.s400) if _dark(page) else fam.s100

    @staticmethod
    def whisker_color(page=None) -> str:
        return Palette.sky.s400 if _dark(page) else Palette.indigo.s600

    @staticmethod
    def outline_selection(page=None) -> str:
        return Palette.sky.s400 if _dark(page) else Palette.indigo.s600

    @staticmethod
    def input_border(page=None) -> str:
        """Unfocused form control outline — stronger than panel borders in dark mode."""
        return Palette.slate.s600 if _dark(page) else Palette.slate.s400

    @staticmethod
    def input_fill(page=None) -> str:
        return ThemeHelper.surface_raised(page)

    @staticmethod
    def outline_selection_width() -> int:
        return 3

    @staticmethod
    def shadow_pop(page=None):
        opacity = 0.35 if _dark(page) else 0.45
        return ft.BoxShadow(
            spread_radius=2,
            blur_radius=12,
            color=ft.Colors.with_opacity(opacity, ft.Colors.BLACK),
            offset=ft.Offset(0, 4),
        )


class InputStyles:
    """Centralized TextField / Dropdown styling for readable dark-mode affordance."""

    @staticmethod
    def apply_text_field(page, field: ft.TextField, *, read_only: bool | None = None) -> None:
        ro = field.read_only if read_only is None else read_only
        field.border_color = ThemeHelper.input_border(page)
        field.focused_border_color = ThemeHelper.outline_selection(page)
        field.focused_border_width = ThemeHelper.outline_selection_width()
        field.color = ThemeHelper.text_primary(page)
        field.hint_style = ft.TextStyle(color=ThemeHelper.text_muted(page))
        field.label_style = ft.TextStyle(color=ThemeHelper.text_muted(page))
        field.filled = True
        field.fill_color = (
            ThemeHelper.surface_dim(page) if ro else ThemeHelper.input_fill(page)
        )

    @staticmethod
    def apply_dropdown(page, dropdown: ft.Dropdown) -> None:
        dropdown.border_color = ThemeHelper.input_border(page)
        dropdown.focused_border_color = ThemeHelper.outline_selection(page)
        dropdown.focused_border_width = ThemeHelper.outline_selection_width()
        dropdown.fill_color = ThemeHelper.input_fill(page)
        dropdown.filled = True
        dropdown.text_style = ft.TextStyle(color=ThemeHelper.text_primary(page))
        dropdown.label_style = ft.TextStyle(color=ThemeHelper.text_muted(page))

    @staticmethod
    def text_field(page, /, **kwargs) -> ft.TextField:
        read_only = bool(kwargs.get("read_only", False))
        field = ft.TextField(**kwargs)
        InputStyles.apply_text_field(page, field, read_only=read_only)
        return field

    @staticmethod
    def dropdown(page, /, **kwargs) -> ft.Dropdown:
        control = ft.Dropdown(**kwargs)
        InputStyles.apply_dropdown(page, control)
        return control

    @staticmethod
    def refresh_fields(
        page,
        text_fields: list[ft.TextField] | None = None,
        dropdowns: list[ft.Dropdown] | None = None,
    ) -> None:
        for field in text_fields or []:
            InputStyles.apply_text_field(page, field)
        for dd in dropdowns or []:
            InputStyles.apply_dropdown(page, dd)


class ButtonStyles:
    """Modern button styles with disabled states."""

    @staticmethod
    def _base_style(bg_color, fg_color, *, page=None):
        dark = _dark(page)
        disabled_bg = Palette.slate.s900 if dark else Palette.slate.s200
        disabled_fg = Palette.slate.s400 if dark else Palette.slate.s500
        return ft.ButtonStyle(
            color={
                ft.ControlState.DEFAULT: fg_color,
                ft.ControlState.DISABLED: disabled_fg,
            },
            bgcolor={
                ft.ControlState.DEFAULT: bg_color,
                ft.ControlState.DISABLED: disabled_bg,
            },
            shape=ft.RoundedRectangleBorder(radius=8),
            elevation={"pressed": 0, "": 4},
            animation_duration=MotionSpec.BUTTON_MS,
            padding=15,
        )

    @staticmethod
    def primary(page=None):
        return ButtonStyles._base_style(Palette.indigo.s600, Palette.slate.s50, page=page)

    @staticmethod
    def secondary(page=None):
        bg = Palette.slate.s800 if _dark(page) else Palette.slate.s600
        return ButtonStyles._base_style(bg, Palette.slate.s50, page=page)

    @staticmethod
    def destructive(page=None):
        return ButtonStyles._base_style(Palette.rose.s900, Palette.slate.s50, page=page)

    @staticmethod
    def success(page=None):
        return ButtonStyles._base_style(Palette.emerald.s800, Palette.slate.s50, page=page)
