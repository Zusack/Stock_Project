"""
Tiny JSON-backed key/value store. Drop-in replacement for the parent
project's DatabaseManager.get_setting / set_setting API so any logic copied
across keeps working.

File: <project root>/data/settings.json
Format: a flat {"key": "value"} dict (values are coerced to strings).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional


def _project_root() -> str:
    """Resolve the directory containing main.py (two levels above this file)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


ROOT_DIR = _project_root()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEFAULT_SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")


class SettingsStore:
    """Thread-safe JSON settings store (singleton)."""

    _instance: Optional["SettingsStore"] = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "SettingsStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, path: str = DEFAULT_SETTINGS_FILE) -> None:
        if getattr(self, "_initialized", False):
            return
        self._path = path
        self._cache: Dict[str, str] = {}
        self._file_lock = threading.Lock()
        self._ensure_dir()
        self._load()
        self._initialized = True

    def _ensure_dir(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
        except OSError:
            pass

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._cache = {str(k): str(v) for k, v in data.items()}
        except (OSError, json.JSONDecodeError):
            # Corrupted file -> start fresh; old file is preserved by next save attempt.
            self._cache = {}

    def _save(self) -> None:
        tmp = f"{self._path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._file_lock:
            return self._cache.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        with self._file_lock:
            self._cache[key] = "" if value is None else str(value)
            self._save()

    def delete_setting(self, key: str) -> None:
        with self._file_lock:
            if key in self._cache:
                del self._cache[key]
                self._save()

    def all_settings(self) -> Dict[str, str]:
        with self._file_lock:
            return dict(self._cache)


def settings_store() -> SettingsStore:
    """Singleton accessor."""
    return SettingsStore()
