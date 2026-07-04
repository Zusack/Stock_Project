"""
App theme settings: persists dark/light/system preference and provides effective mode.
Defaults to dark mode. Wired for future settings_view.py toggle.
"""
import flet as ft
from src.database.manager import DatabaseManager, DEFAULT_DB_FILE

THEME_KEY = "theme_mode"
THEME_DARK = "dark"
THEME_LIGHT = "light"
THEME_SYSTEM = "system"
# Default: dark mode (user preference)
DEFAULT_THEME = THEME_DARK


class AppThemeService:
    """
    Singleton service for app theme (dark/light/system).
    Persists to database. Default is dark mode.
    Call set_theme_mode() from settings_view when toggle is built.
    """

    _instance = None
    _cached_mode = None
    _db_path = DEFAULT_DB_FILE

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_from_db(self) -> str:
        try:
            with DatabaseManager(self._db_path) as db:
                return db.get_setting(THEME_KEY, DEFAULT_THEME) or DEFAULT_THEME
        except Exception:
            return DEFAULT_THEME

    def get_theme_mode(self) -> str:
        """Returns 'dark', 'light', or 'system'."""
        if self._cached_mode is None:
            self._cached_mode = self._load_from_db()
        return self._cached_mode

    def set_theme_mode(self, mode: str, page: ft.Page = None):
        """
        Set theme mode and persist. Call from settings_view when toggle is built.
        mode: 'dark' | 'light' | 'system'
        page: optional, to apply immediately
        """
        mode = (mode or "").lower().strip()
        if mode not in (THEME_DARK, THEME_LIGHT, THEME_SYSTEM):
            mode = THEME_DARK
        self._cached_mode = mode
        try:
            with DatabaseManager(self._db_path) as db:
                db.set_setting(THEME_KEY, mode)
        except Exception:
            pass
        if page:
            self.apply_to_page(page)
        # Broadcast so views can refresh colors
        try:
            if page and hasattr(page, "pubsub"):
                page.pubsub.send_all("theme_changed")
        except Exception:
            pass

    def apply_to_page(self, page: ft.Page):
        """Apply stored mode to Flet page.theme_mode. Call at app startup."""
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
        """
        True when the effective UI should use dark backgrounds.
        Resolves 'system' via OS (darkdetect) when available.
        """
        mode = self.get_theme_mode()
        if mode == THEME_DARK:
            return True
        if mode == THEME_LIGHT:
            return False
        if mode == THEME_SYSTEM:
            try:
                import darkdetect
                return darkdetect.isDark()
            except Exception:
                return True  # fallback
        return True


# Singleton accessor
def app_theme() -> AppThemeService:
    return AppThemeService()
