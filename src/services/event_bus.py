"""
Simple app-wide pub/sub bus. Subscribers are stored by event type and called
synchronously on emit. Includes self-healing: handlers that raise fatal UI
errors (detached controls, closed event loops) are automatically removed.

This is a direct port of the parent project's event bus.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List

# Emit-rate tracking is gated on DEBUG-level logging via app_logger.
_emit_counts: Dict[str, int] = {}
_emit_last_log = 0.0
_EMIT_LOG_INTERVAL = 10.0


class EventBus:
    """Singleton event bus."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._subscribers: Dict[str, List[Callable]] = {}
                cls._instance._is_shutting_down = False
        return cls._instance

    def subscribe(self, event_type: str, handler: Callable) -> None:
        if self._is_shutting_down:
            return
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        with self._lock:
            if event_type in self._subscribers:
                if handler in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(handler)

    def emit(self, event_type: str, **kwargs) -> None:
        if self._is_shutting_down:
            return
        # If the main thread has died (e.g. during Flet teardown), bail out.
        if not threading.main_thread().is_alive():
            return

        try:
            from src.utils.logger_utils import app_logger
            if app_logger.get_level() == "DEBUG":
                global _emit_counts, _emit_last_log
                _emit_counts[event_type] = _emit_counts.get(event_type, 0) + 1
                now = time.time()
                if now - _emit_last_log >= _EMIT_LOG_INTERVAL:
                    _emit_last_log = now
                    app_logger.log("EVENT_BUS", f"Emit stats: {dict(_emit_counts)}", level="DEBUG")
                    _emit_counts.clear()
        except Exception:
            pass

        with self._lock:
            handlers = self._subscribers.get(event_type, [])[:]

        for handler in handlers:
            try:
                handler(**kwargs)
            except Exception as e:
                self._handle_handler_error(e, event_type, handler)

    def _handle_handler_error(self, e: Exception, event_type: str, handler: Callable) -> None:
        """Decide whether to purge the subscriber (self-healing) or just log."""
        error_str = str(e)

        fatal_triggers = [
            "Event loop is closed",
            "cannot schedule new futures after shutdown",
            "assert self.__uid is not None",
            "object has no attribute 'page'",
            "RuntimeError: loop is closed",
        ]

        if any(trigger in error_str for trigger in fatal_triggers):
            msg = f"SELF-HEALING: Purging broken handler for '{event_type}' due to fatal error: {e}"
            print(f"[EventBus] {msg}")
            try:
                from src.utils.logger_utils import app_logger
                app_logger.log("EVENT_BUS", msg, level="WARN")
            except Exception:
                pass
            self.unsubscribe(event_type, handler)
            return

        if "SystemExit" in error_str:
            self._shutdown()
            return

        print(f"[EventBus] Warning: Handler for '{event_type}' failed: {e}")

    def shutdown(self) -> None:
        self._shutdown()

    def _shutdown(self) -> None:
        self._is_shutting_down = True
        with self._lock:
            self._subscribers.clear()


event_bus = EventBus()
