"""
App theme settings: persists dark/light/system preference and provides effective mode.

Backed by settings_store (JSON) instead of a database. Default is dark mode.
"""
from __future__ import annotations

import flet as ft

from src.services.settings_store import settings_store

THEME_KEY = "theme_mode"
THEME_DARK = "dark"
THEME_LIGHT = "light"
THEME_SYSTEM = "system"
DEFAULT_THEME = THEME_DARK


class AppThemeService:
    """Singleton service for app theme (dark/light/system)."""

    _instance = None
    _cached_mode = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_from_store(self) -> str:
        try:
            return settings_store().get_setting(THEME_KEY, DEFAULT_THEME) or DEFAULT_THEME
        except Exception:
            return DEFAULT_THEME

    def get_theme_mode(self) -> str:
        """Returns 'dark', 'light', or 'system'."""
        if self._cached_mode is None:
            self._cached_mode = self._load_from_store()
        return self._cached_mode

    def set_theme_mode(self, mode: str, page: ft.Page | None = None) -> None:
        mode = (mode or "").lower().strip()
        if mode not in (THEME_DARK, THEME_LIGHT, THEME_SYSTEM):
            mode = THEME_DARK
        self._cached_mode = mode
        try:
            settings_store().set_setting(THEME_KEY, mode)
        except Exception:
            pass
        if page:
            self.apply_to_page(page)
        try:
            if page and hasattr(page, "pubsub"):
                page.pubsub.send_all("theme_changed")
        except Exception:
            pass

    def apply_to_page(self, page: ft.Page) -> None:
        m = self.get_theme_mode()
        if m == THEME_DARK:
            page.theme_mode = ft.ThemeMode.DARK
        elif m == THEME_LIGHT:
            page.theme_mode = ft.ThemeMode.LIGHT
        else:
            page.theme_mode = ft.ThemeMode.SYSTEM
        try:
            page.update()
        except Exception:
            pass

    def is_dark_effective(self) -> bool:
        """True when the effective UI should use dark backgrounds."""
        mode = self.get_theme_mode()
        if mode == THEME_DARK:
            return True
        if mode == THEME_LIGHT:
            return False
        if mode == THEME_SYSTEM:
            try:
                import darkdetect  # type: ignore
                return bool(darkdetect.isDark())
            except Exception:
                return True
        return True


def app_theme() -> AppThemeService:
    """Singleton accessor."""
    return AppThemeService()
