"""
GlobalControlService — singleton that tracks page reference, the active tab
index (used by BaseView for tab-aware throttling), and a reference-counted
top-of-screen loading spinner.

This is a slimmed-down port of the parent project's GlobalControlService. The
benchmark-specific process-control fields (run_event, stop_event, paused
flags, fleet IDs, etc.) have been removed since the new app won't run
multi-step processes that need a unified Stop/Pause control.

If you later add long-running jobs, layer them on top of this service so the
top tab spinner and active-tab logic stay consistent.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import flet as ft


class GlobalControlService:
    """Singleton wiring for app-wide UI state."""

    _instance: Optional["GlobalControlService"] = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.page_ref: Optional[ft.Page] = None
        self.active_tab_index: int = 0

        # Hint flag for BaseView _check_permissions() — keep True unless your
        # app surfaces a read-only mode (e.g. demo / locked / remote views).
        self.is_interaction_allowed: bool = True

        # Used by BaseView._check_permissions(allow_during_run=False). Set this
        # to a non-empty label while a long-running job is active so views can
        # gate their interactive controls.
        self.active_process_name: Optional[str] = None

        # Database write activity (ingest, maintenance) for cross-tab banners.
        self._db_activity_depth = 0
        self._db_activity_label: str = ""
        self._db_banner_callback: Optional[Callable[[], None]] = None
        self._last_lock_notice_at: float = 0.0
        self._lock_notice_interval_sec = 60.0

        # Top tab strip loading indicator (registered from app.py).
        self._tab_strip_busy_depth = 0
        self._tab_strip_loading_message = ""
        self._tab_strip_ring: Optional[ft.ProgressRing] = None
        self._tab_strip_label: Optional[ft.Text] = None

    def register_page(self, page: ft.Page) -> None:
        self.page_ref = page

    @property
    def is_db_busy(self) -> bool:
        return self._db_activity_depth > 0

    @property
    def db_busy_message(self) -> str:
        label = (self._db_activity_label or self.active_process_name or "").strip()
        if label:
            return f"Database busy: {label} — reads may be slow until complete."
        return "Database busy — reads may be slow until complete."

    def register_db_banner_callback(self, callback: Callable[[], None]) -> None:
        self._db_banner_callback = callback

    def _sync_db_banner(self) -> None:
        if self._db_banner_callback is not None:
            try:
                self._db_banner_callback()
            except Exception:
                pass

    def push_db_activity(self, label: str) -> None:
        self._db_activity_depth += 1
        if label:
            self._db_activity_label = label
            self.active_process_name = label
        self._sync_db_banner()

    def pop_db_activity(self) -> None:
        self._db_activity_depth = max(0, self._db_activity_depth - 1)
        if self._db_activity_depth == 0:
            self._db_activity_label = ""
            self.active_process_name = None
        self._sync_db_banner()

    def notify_db_locked(self, source: str) -> bool:
        """Rate-limited lock notice; returns True if a notice was emitted."""
        now = time.time()
        if now - self._last_lock_notice_at < self._lock_notice_interval_sec:
            return False
        self._last_lock_notice_at = now
        try:
            from src.utils.logger_utils import app_logger

            app_logger.log(
                "DB",
                "Database lock contended during UI read.",
                level="WARN",
                source=source,
                db_busy=self.is_db_busy,
                active_process=self.active_process_name,
            )
        except Exception:
            pass
        self._sync_db_banner()
        return True

    # ------------------------------------------------------------------
    # Top tab-strip loading indicator
    # ------------------------------------------------------------------
    def register_tab_strip_loading_controls(
        self,
        ring: ft.ProgressRing,
        label: Optional[ft.Text] = None,
    ) -> None:
        """Wire the main-window tab row spinner (called once from app.py)."""
        self._tab_strip_ring = ring
        self._tab_strip_label = label
        self._apply_tab_strip_loading()

    def _apply_tab_strip_loading(self) -> None:
        busy = self._tab_strip_busy_depth > 0
        msg = (self._tab_strip_loading_message or "").strip()
        if self._tab_strip_ring is not None:
            self._tab_strip_ring.visible = busy
        if self._tab_strip_label is not None:
            self._tab_strip_label.visible = busy
            self._tab_strip_label.value = msg if msg else "Loading…"

    def push_tab_strip_loading(self, message: str = "", *, flush_page: bool = True) -> None:
        """Mark global tab-row loading busy (reference counted)."""
        self._tab_strip_busy_depth += 1
        if message:
            self._tab_strip_loading_message = message
        self._apply_tab_strip_loading()
        if flush_page and self.page_ref:
            try:
                self.page_ref.update()
            except Exception:
                pass

    def pop_tab_strip_loading(self, *, flush_page: bool = True) -> None:
        """Release one tab-row loading level."""
        self._tab_strip_busy_depth = max(0, self._tab_strip_busy_depth - 1)
        if self._tab_strip_busy_depth == 0:
            self._tab_strip_loading_message = ""
        self._apply_tab_strip_loading()
        if flush_page and self.page_ref:
            try:
                self.page_ref.update()
            except Exception:
                pass
