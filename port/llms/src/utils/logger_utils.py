import os
import json
import datetime
import glob
from typing import Optional

from src.database.manager import ROOT_DIR

LOG_DIR = os.path.join(ROOT_DIR, "exports", "logs")

# Level order for filtering: DEBUG < INFO < WARN < ERROR
_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}


class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.enabled = False
            cls._instance.current_log_file = None
            cls._instance._min_level = "INFO"
            cls._instance._echo_to_stdout = False
            cls._instance._ensure_log_dir()
        return cls._instance

    def _ensure_log_dir(self):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

    def set_level(self, level: str) -> None:
        """Set minimum log level. One of DEBUG, INFO, WARN, ERROR. Default INFO."""
        u = (level or "INFO").upper()
        if u in _LEVEL_ORDER:
            self._min_level = u

    def get_level(self) -> str:
        return getattr(self, "_min_level", "INFO")

    def set_echo_to_stdout(self, enabled: bool) -> None:
        self._echo_to_stdout = bool(enabled)

    def enable_logging(self, level: Optional[str] = None) -> None:
        self.enabled = True
        if level is not None:
            self.set_level(level)
        self._rotate_logs()
        self._start_new_log_file()

    def disable_logging(self) -> None:
        self.enabled = False
        self.current_log_file = None

    def _start_new_log_file(self) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"benchmark_log_{timestamp}.log"
        self.current_log_file = os.path.join(LOG_DIR, filename)
        self.log("SYSTEM", "Logging started.", level="INFO")

    def _rotate_logs(self) -> None:
        """Keep only the 10 most recent log files."""
        log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
        log_files.sort(key=os.path.getmtime) 
        
        while len(log_files) >= 10:
            oldest_file = log_files.pop(0)
            try:
                os.remove(oldest_file)
                print(f"Purged old log file: {oldest_file}")
            except OSError as e:
                print(f"Error purging log file {oldest_file}: {e}")

    def _sanitize_for_log(self, message: str, max_chars: int = 1000, max_lines: int = 3) -> str:
        """
        Prevent benchmark/evaluation output dumps in system logs. Truncate long or
        multi-line messages; full outputs belong in the DB or Inspector, not the log.
        """
        if not message:
            return message
        s = str(message)
        if len(s) <= max_chars and s.count("\n") <= max_lines:
            return s
        first = s[:max_chars].split("\n")[: max_lines + 1]
        out = "\n".join(first).rstrip()
        return f"{out} [truncated, {len(s)} chars]"

    def log(self, source: str, message: str, level: str = "INFO", **kwargs) -> None:
        """
        Write a log entry. Optional level (DEBUG, INFO, WARN, ERROR) and structured kwargs.
        Entries are formatted as: [timestamp] [LEVEL] [source] message | {"k":"v",...}
        DEBUG is dropped when min level is INFO. kwargs are JSON-encoded when present.
        """
        if not self.enabled or not self.current_log_file:
            return
        lvl = (level or "INFO").upper()
        if lvl not in _LEVEL_ORDER:
            lvl = "INFO"
        min_lvl = _LEVEL_ORDER.get(getattr(self, "_min_level", "INFO"), 1)
        if _LEVEL_ORDER.get(lvl, 1) < min_lvl:
            return

        sanitized = self._sanitize_for_log(message)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{lvl}] [{source}] {sanitized}"
        if kwargs:
            # Ensure JSON-serializable (str for non-primitives)
            clean = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            try:
                entry += " | " + json.dumps(clean, ensure_ascii=False)
            except (TypeError, ValueError):
                entry += " | {}"
        entry += "\n"

        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(entry)
            if getattr(self, "_echo_to_stdout", False):
                print(entry.rstrip("\n"))
        except Exception as e:
            print(f"Logging failed: {e}")

    def get_latest_log_content(self):
        """Returns the content of the most recent log file."""
        log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
        if not log_files:
            return "No log files found."
        
        latest_file = max(log_files, key=os.path.getmtime)
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading log file: {e}"

    def get_log_files(self):
        return sorted(glob.glob(os.path.join(LOG_DIR, "*.log")), key=os.path.getmtime, reverse=True)


def export_session_log_to_file(log_lines: list, file_prefix: str) -> str | None:
    """Write session log lines to exports/logs/{file_prefix}_{timestamp}.log.
    Returns the file path on success, None on failure."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{file_prefix}_{timestamp}.log"
    path = os.path.join(LOG_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + ("\n" if log_lines else ""))
        return path
    except OSError:
        return None


# Global instance
app_logger = Logger()