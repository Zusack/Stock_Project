import threading
import time
import traceback
from typing import Callable, Dict, List

# Debug: track emit rates for freeze diagnosis (only when DEBUG logging enabled)
_emit_counts = {}
_emit_last_log = 0.0
_EMIT_LOG_INTERVAL = 10.0  # Log emit stats every N seconds when DEBUG


class EventBus:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventBus, cls).__new__(cls)
                cls._instance._subscribers: Dict[str, List[Callable]] = {}
                cls._instance._is_shutting_down = False
        return cls._instance

    def subscribe(self, event_type: str, handler: Callable):
        """Register a handler. Prevents duplicates."""
        if self._is_shutting_down: return
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        """Unregister a handler safely."""
        with self._lock:
            if event_type in self._subscribers:
                if handler in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(handler)

    def emit(self, event_type: str, **kwargs):
        """
        Trigger an event. 
        Includes 'Self-Healing' logic: If a handler fails with a fatal UI error,
        it is automatically removed to prevent error loops.
        """
        if self._is_shutting_down:
            return

        # 1. Zombie Check: If Main Thread is dead, stop all bus activity immediately.
        if not threading.main_thread().is_alive():
            return

        # Debug: track emit counts for freeze diagnosis
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

        # 2. Snapshot: Copy list to allow modification (unsubscribing) during iteration
        with self._lock:
            handlers = self._subscribers.get(event_type, [])[:]
        
        for handler in handlers:
            try:
                handler(**kwargs)
            except Exception as e:
                self._handle_handler_error(e, event_type, handler)

    def _handle_handler_error(self, e: Exception, event_type: str, handler: Callable):
        """
        Intelligent error handler. Decides whether to log the error or 
        purge the subscriber (Self-Healing).
        """
        error_str = str(e)
        
        # 1. Fatal UI Errors - These mean the View is dead/detached.
        # We must UNSUBSCRIBE immediately to stop the loop.
        fatal_triggers = [
            "Event loop is closed",
            "cannot schedule new futures after shutdown",
            "assert self.__uid is not None", # Flet detached control error
            "object has no attribute 'page'",
            "RuntimeError: loop is closed"
        ]
        
        if any(trigger in error_str for trigger in fatal_triggers):
            # Log once, then purge - also log to app_logger for debug-level capture
            msg = f"SELF-HEALING: Purging broken handler for '{event_type}' due to fatal error: {e}"
            print(f"[EventBus] 🛡️ {msg}")
            try:
                from src.utils.logger_utils import app_logger
                app_logger.log("EVENT_BUS", msg, level="WARN")
            except Exception:
                pass
            self.unsubscribe(event_type, handler)
            return

        # 2. Shutdown Detection
        if "SystemExit" in error_str:
            self._shutdown()
            return

        # 3. Standard Logic Errors - Just Log
        # We silence 'new_token' errors to avoid console spam during generation
        if event_type != "new_token": 
            print(f"[EventBus] Warning: Handler for '{event_type}' failed: {e}")
            # traceback.print_exc() # Uncomment for deep debugging

    def shutdown(self):
        """Call from app close handler to stop all emissions before Flet teardown."""
        self._shutdown()

    def _shutdown(self):
        """Permanently disables the bus."""
        self._is_shutting_down = True
        with self._lock:
            self._subscribers.clear()

# Global Singleton Instance
event_bus = EventBus()