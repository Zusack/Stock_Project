"""
UI Health Monitor – detects event-loop and client-side freezes.

Start via ``UIHealthMonitor(page).start()`` once the Flet page is ready.
The monitor schedules a lightweight heartbeat coroutine on the page's asyncio
event loop every ``HEARTBEAT_INTERVAL`` seconds.  If the *actual* elapsed
time since the last successful heartbeat exceeds ``WARN_LATENCY`` or
``CRITICAL_LATENCY``, it logs detailed diagnostics to ``app_logger``.

Additionally tracks UI update throughput and execution timing to detect
client-side degradation (Flutter rendering stalls).
"""
import os
import threading
import time
import gc

from src.utils.logger_utils import app_logger


HEARTBEAT_INTERVAL = 5.0   # seconds between heartbeat checks
WARN_LATENCY = 3.0         # log WARN when heartbeat is this late
CRITICAL_LATENCY = 10.0    # log ERROR when heartbeat is this late
STATS_LOG_INTERVAL = 60.0  # log task stats every N seconds (DEBUG only)


def _get_process_rss_mb():
    """Return process RSS (resident set size) in MB. Linux-only, returns 0 elsewhere."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


class UIHealthMonitor:
    """Periodically probes the Flet event loop from a background thread."""

    def __init__(self, page_ref):
        self._page = page_ref
        self._last_heartbeat = time.time()
        self._running = False
        self._thread = None
        self._consecutive_misses = 0
        self._last_stats_log = 0.0
        self._prev_completed = 0
        self._prev_stats_time = time.time()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="UIHealthMonitor")
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            scheduled_at = time.time()

            try:
                if hasattr(self._page, "run_task"):
                    async def _heartbeat():
                        self._on_heartbeat(scheduled_at)
                    self._page.run_task(_heartbeat)
                else:
                    self._on_heartbeat(scheduled_at)
            except Exception:
                pass

            self._maybe_log_stats()
            time.sleep(HEARTBEAT_INTERVAL)

    def _on_heartbeat(self, scheduled_at: float):
        """Runs on the event loop – measures latency and logs if degraded."""
        now = time.time()
        latency = now - scheduled_at
        self._last_heartbeat = now

        if latency >= CRITICAL_LATENCY:
            self._consecutive_misses += 1
            self._log_diagnostics(latency, level="ERROR")
        elif latency >= WARN_LATENCY:
            self._consecutive_misses += 1
            self._log_diagnostics(latency, level="WARN")
        else:
            if self._consecutive_misses > 0:
                app_logger.log("UI_HEALTH", f"Event loop recovered after {self._consecutive_misses} slow heartbeat(s). Latency: {latency:.2f}s", level="INFO")
            self._consecutive_misses = 0

    def _log_diagnostics(self, latency: float, level: str = "WARN"):
        from src.views.base_view import get_ui_task_stats
        pending, completed, dropped, slow, max_dur = get_ui_task_stats()
        queued = pending - completed
        rss = _get_process_rss_mb()

        msg = (
            f"Event loop latency: {latency:.2f}s "
            f"(threshold: {'CRITICAL' if level == 'ERROR' else 'WARN'}). "
            f"UI tasks: queued={queued} completed={completed} dropped={dropped} "
            f"slow(>1s)={slow} max_duration={max_dur:.2f}s. "
            f"Consecutive slow heartbeats: {self._consecutive_misses}. "
            f"Threads: {threading.active_count()}. RSS: {rss:.0f}MB."
        )
        app_logger.log("UI_HEALTH", msg, level=level)
        print(f"[UI_HEALTH] {msg}")

        if level == "ERROR" and self._consecutive_misses <= 3:
            self._log_thread_dump()
            self._log_gc_stats()

    def _log_thread_dump(self):
        """Log active thread names for deadlock diagnosis."""
        threads = [f"{t.name}({'alive' if t.is_alive() else 'dead'})" for t in threading.enumerate()]
        app_logger.log("UI_HEALTH", f"Active threads ({len(threads)}): {', '.join(threads)}", level="ERROR")

    def _log_gc_stats(self):
        """Log GC generation counts."""
        counts = gc.get_count()
        app_logger.log("UI_HEALTH", f"GC counts: gen0={counts[0]} gen1={counts[1]} gen2={counts[2]}", level="DEBUG")

    def _maybe_log_stats(self):
        """Periodically log UI task statistics at DEBUG level."""
        now = time.time()
        if now - self._last_stats_log < STATS_LOG_INTERVAL:
            return
        elapsed_since_last = now - self._prev_stats_time
        self._last_stats_log = now

        try:
            if app_logger.get_level() != "DEBUG":
                return
        except Exception:
            return

        from src.views.base_view import get_ui_task_stats
        pending, completed, dropped, slow, max_dur = get_ui_task_stats()
        queued = pending - completed
        rate = (completed - self._prev_completed) / elapsed_since_last if elapsed_since_last > 0 else 0
        self._prev_completed = completed
        self._prev_stats_time = now
        rss = _get_process_rss_mb()
        app_logger.log(
            "UI_HEALTH",
            f"Stats: queued={queued} completed={completed} dropped={dropped} "
            f"rate={rate:.1f}/s slow={slow} max_dur={max_dur:.2f}s "
            f"threads={threading.active_count()} rss={rss:.0f}MB",
            level="DEBUG",
        )
