"""
Base view class with common functionality for all views.

Provides:
- Thread-safe UI update scheduling (always routed onto Flet's event loop)
- Back-pressure protection (drops non-critical updates when the queue is deep)
- Active-tab gating (so background views don't repaint while hidden)
- Debounce / throttle helpers
- Event-bus subscription lifecycle
"""
from __future__ import annotations

import threading
import time

import flet as ft

from src.services.event_bus import event_bus
from src.services.global_control_service import GlobalControlService


# ---------------------------------------------------------------------------
# Centralized UI update scheduler
# ---------------------------------------------------------------------------
# All UI updates MUST go through this function (or BaseView._safe_update which
# calls it). Using page.run_thread() for UI updates creates a threading
# hazard: run_thread dispatches onto a ThreadPoolExecutor while run_task
# dispatches onto the asyncio event loop. If both paths call control.update()
# or page.update() concurrent WebSocket writes can corrupt Flet's internal
# messaging, eventually freezing all UI updates while backend work continues.
#
# This function ALWAYS uses page.run_task() so every update is serialized on
# the single-threaded event loop.
# ---------------------------------------------------------------------------
_ui_task_pending = 0
_ui_task_completed = 0
_ui_task_dropped = 0
_ui_task_lock = threading.Lock()
_BACKPRESSURE_THRESHOLD = 200  # Drop non-critical updates when this many tasks are queued
_ui_update_slow_count = 0      # Updates that took > 1s to execute
_ui_update_max_duration = 0.0  # Longest single update execution time observed


def schedule_ui_update(page_ref, func, *, critical: bool = False, label: str | None = None) -> None:
    """
    Schedule *func* to run on Flet's main event loop.

    Safe to call from any thread. Must be used instead of page.run_thread()
    whenever the callback touches Flet controls or calls control.update().

    Args:
        critical: If True, bypasses back-pressure (use for stop/finish events).
        label: Optional source label for slow-update diagnostics.
    """
    global _ui_task_pending, _ui_task_completed, _ui_task_dropped
    if page_ref is None:
        return

    with _ui_task_lock:
        pending = _ui_task_pending - _ui_task_completed
        if not critical and pending > _BACKPRESSURE_THRESHOLD:
            _ui_task_dropped += 1
            return
        _ui_task_pending += 1

    scheduled_at = time.time()

    def _do_update():
        global _ui_task_completed, _ui_update_slow_count, _ui_update_max_duration
        start = time.time()
        try:
            func()
        except Exception as e:
            err = str(e)
            if (
                "Event loop is closed" not in err
                and "loop is closed" not in err
                and "cannot schedule new futures" not in err
            ):
                print(f"UI Update Failed: {e}")
        finally:
            elapsed = time.time() - start
            with _ui_task_lock:
                _ui_task_completed += 1
                if elapsed > 1.0:
                    _ui_update_slow_count += 1
                if elapsed > _ui_update_max_duration:
                    _ui_update_max_duration = elapsed
            if elapsed > 2.0 and label:
                try:
                    from src.utils.logger_utils import app_logger
                    app_logger.log(
                        "UI_HEALTH",
                        f"Slow update [{label}]: {elapsed:.2f}s (wait={start - scheduled_at:.2f}s)",
                        level="WARN",
                    )
                except Exception:
                    pass

    try:
        if hasattr(page_ref, "run_task"):
            async def _wrapper():
                _do_update()
            page_ref.run_task(_wrapper)
        else:
            _do_update()
    except Exception as e:
        err = str(e)
        if (
            "Event loop is closed" not in err
            and "loop is closed" not in err
            and "cannot schedule new futures" not in err
        ):
            print(f"UI Update Failed (schedule): {e}")


def get_ui_task_stats():
    """Return (pending, completed, dropped, slow_count, max_duration) for diagnostics."""
    with _ui_task_lock:
        return _ui_task_pending, _ui_task_completed, _ui_task_dropped, _ui_update_slow_count, _ui_update_max_duration


class BaseView(ft.Column):
    """Base class for all views with common functionality."""

    _tab_index = None  # Override in subclass to enable active-tab optimization.

    def __init__(self, page: ft.Page):
        super().__init__()
        self.expand = True
        self.scroll = ft.ScrollMode.AUTO
        self.spacing = 12
        self.page_ref = page
        self.global_control = GlobalControlService()
        self._event_handlers: dict[str, list] = {}
        self._safe_update_fail_count = 0
        self._throttle_last_update: dict[str, float] = {}
        self._throttle_lock = threading.Lock()
        self._debounce_timers: dict[str, threading.Timer] = {}

    # ------------------------------------------------------------------
    # Tab visibility
    # ------------------------------------------------------------------
    def _is_active_tab(self) -> bool:
        """True if this view's tab is currently selected (or tab tracking isn't configured)."""
        if self._tab_index is None:
            return True
        return self.global_control.active_tab_index == self._tab_index

    # ------------------------------------------------------------------
    # UI update wrappers
    # ------------------------------------------------------------------
    def _safe_update(self, func, label: str | None = None) -> None:
        """Schedule a UI update on the event loop. Safe to call from any thread."""
        if hasattr(self, "page") and not self.page:
            return
        if not self.page_ref:
            return

        def _guarded():
            try:
                func()
                self._safe_update_fail_count = 0
            except Exception as e:
                self._safe_update_fail_count = getattr(self, "_safe_update_fail_count", 0) + 1
                if self._safe_update_fail_count <= 3:
                    try:
                        from src.utils.logger_utils import app_logger
                        app_logger.log("UI", f"_safe_update handler failed: {e}", level="DEBUG")
                    except Exception:
                        pass
                err = str(e)
                if (
                    "Event loop is closed" not in err
                    and "loop is closed" not in err
                    and "cannot schedule new futures" not in err
                ):
                    if self._safe_update_fail_count <= 5:
                        print(f"UI Update Failed: {e}")

        schedule_ui_update(self.page_ref, _guarded, label=label)

    def _safe_update_critical(self, func) -> None:
        """Like _safe_update but bypasses back-pressure. Use for finish/stop events."""
        if hasattr(self, "page") and not self.page:
            return
        if not self.page_ref:
            return
        schedule_ui_update(self.page_ref, func, critical=True)

    def _safe_update_throttled(self, key: str, interval_sec: float, func) -> None:
        """Only schedule if interval_sec has passed since the last update for this key."""
        if not self._is_active_tab():
            return
        now = time.time()
        with self._throttle_lock:
            last = self._throttle_last_update.get(key, 0)
            if now - last < interval_sec:
                return
            self._throttle_last_update[key] = now
        self._safe_update(func, label=key)

    def _safe_update_debounced(self, key: str, delay_sec: float, func) -> None:
        """Run *func* after delay_sec, resetting the timer if called again sooner."""
        timer = self._debounce_timers.get(key)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

        def run_later():
            self._debounce_timers.pop(key, None)
            self._safe_update(func)

        t = threading.Timer(delay_sec, run_later)
        self._debounce_timers[key] = t
        t.start()

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------
    def _check_permissions(self, allow_during_run: bool = False) -> bool:
        """
        Check if the user can interact with this view.

        Args:
            allow_during_run: If True, allows interaction even during active processes.

        Returns:
            True if interaction is allowed.
        """
        if not self.global_control.is_interaction_allowed:
            return False
        if not allow_during_run and self.global_control.active_process_name is not None:
            return False
        return True

    # ------------------------------------------------------------------
    # Event bus
    # ------------------------------------------------------------------
    def _setup_pubsub(self, handlers: dict) -> None:
        """Subscribe to event bus events. Pass {event_type: handler_callable}."""
        for event_type, handler in handlers.items():
            event_bus.subscribe(event_type, handler)
            self._event_handlers.setdefault(event_type, []).append(handler)

    def _cleanup_pubsub(self) -> None:
        """Unsubscribe from all event bus subscriptions."""
        for event_type, handlers in self._event_handlers.items():
            for handler in handlers:
                event_bus.unsubscribe(event_type, handler)
        self._event_handlers.clear()

    # ------------------------------------------------------------------
    # Async refresh hook
    # ------------------------------------------------------------------
    def _view_name(self) -> str:
        return type(self).__name__

    def _apply_fetch_error(self, ex: Exception) -> None:
        """Surface fetch failures in the UI (lock-aware, rate-limited)."""
        from src.analysis.db_perf import is_db_locked_error
        from src.views.components.feedback import show_snackbar

        view = self._view_name()
        if is_db_locked_error(ex):
            self.global_control.notify_db_locked(view)
            if self.page_ref:
                show_snackbar(
                    self.page_ref,
                    "Database is busy — data refresh will retry when the write finishes.",
                    severity="warning",
                )
            if hasattr(self, "_apply_locked_fetch"):
                self._apply_locked_fetch(ex)
            return
        if self.page_ref:
            show_snackbar(
                self.page_ref,
                f"Could not refresh data: {ex}"[:140],
                severity="error",
            )

    def refresh_data_async(self, label: str | None = None) -> None:
        """
        Run _fetch_data on a worker thread, then _apply_data on the UI loop.
        Falls back to refresh_data() when those hooks are not implemented.
        """
        if not self.page_ref:
            return

        # Defer non-critical refresh while another tab holds a DB write lock.
        if self.global_control.is_db_busy and not self._is_active_tab():
            return

        def _work():
            data = None
            fetch_error: Exception | None = None
            view = self._view_name()
            try:
                if hasattr(self, "_fetch_data"):
                    data = self._fetch_data()
            except Exception as ex:
                fetch_error = ex
                try:
                    from src.utils.logger_utils import app_logger
                    app_logger.log(
                        "UI",
                        f"_fetch_data failed ({view}): {ex}",
                        level="ERROR",
                    )
                except Exception:
                    pass

            def _ui():
                try:
                    if data is not None and hasattr(self, "_apply_data"):
                        self._apply_data(data)
                    elif fetch_error is not None:
                        self._apply_fetch_error(fetch_error)
                    elif hasattr(self, "refresh_data"):
                        self.refresh_data()
                except Exception as ex:
                    try:
                        from src.utils.logger_utils import app_logger
                        app_logger.log(
                            "UI",
                            f"_apply_data failed ({view}): {ex}",
                            level="ERROR",
                        )
                    except Exception:
                        pass

            schedule_ui_update(self.page_ref, _ui, label=label or "refresh_data_async")

        threading.Thread(target=_work, daemon=True).start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def will_unmount(self):
        """Called when the view is about to be removed — cleanup subscriptions and timers."""
        for timer in getattr(self, "_debounce_timers", {}).values():
            try:
                timer.cancel()
            except Exception:
                pass
        self._debounce_timers = {}
        self._cleanup_pubsub()
        super().will_unmount()
