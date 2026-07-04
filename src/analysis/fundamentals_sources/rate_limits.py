"""Conservative rate limits for free-tier fundamentals APIs."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

from src.services.settings_store import settings_store

# SEC fair-access: max 10 requests/second. Stay well below.
SEC_MIN_INTERVAL_SEC = 0.15  # ~6.7 req/s

# Yahoo (yfinance): unofficial; stay conservative to reduce 429s.
DEFAULT_YAHOO_MIN_INTERVAL_SEC = 0.75

# Default caps (overridable via settings).
DEFAULT_SEC_MAX_PER_RUN = 300
DEFAULT_SEC_MAX_PER_DAY = 2000
DEFAULT_FMP_DAILY_BUDGET = 200  # FMP free tier is 250/day; keep margin.


def _project_data_dir() -> Path:
    here = Path(__file__).resolve()
    root = here.parent.parent.parent.parent
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    return data


def _quota_path() -> Path:
    return _project_data_dir() / "fundamentals_api_quota.json"


class YahooRateLimiter:
    """Process-wide spacing between Yahoo Finance requests (yfinance)."""

    _lock = threading.Lock()
    _last_at = 0.0
    _interval = DEFAULT_YAHOO_MIN_INTERVAL_SEC

    @classmethod
    def configure(cls, min_interval_sec: float) -> None:
        with cls._lock:
            cls._interval = max(0.1, float(min_interval_sec))

    @classmethod
    def wait(cls) -> None:
        with cls._lock:
            now = time.monotonic()
            gap = cls._interval - (now - cls._last_at)
            if gap > 0:
                time.sleep(gap)
            cls._last_at = time.monotonic()


class SecRateLimiter:
    """Process-wide spacing between SEC HTTP calls."""

    _lock = threading.Lock()
    _last_at = 0.0

    @classmethod
    def wait(cls) -> None:
        with cls._lock:
            now = time.monotonic()
            gap = SEC_MIN_INTERVAL_SEC - (now - cls._last_at)
            if gap > 0:
                time.sleep(gap)
            cls._last_at = time.monotonic()


class SecRunBudget:
    """Cap SEC companyfacts calls per ingest run."""

    _lock = threading.Lock()
    _count = 0
    _max = DEFAULT_SEC_MAX_PER_RUN

    @classmethod
    def configure(cls, max_per_run: int) -> None:
        with cls._lock:
            cls._max = max(1, int(max_per_run))
            cls._count = 0

    @classmethod
    def reset(cls, max_per_run: int | None = None) -> None:
        with cls._lock:
            if max_per_run is not None:
                cls._max = max(1, int(max_per_run))
            cls._count = 0

    @classmethod
    def try_acquire(cls) -> bool:
        with cls._lock:
            if cls._count >= cls._max:
                return False
            cls._count += 1
            return True

    @classmethod
    def remaining(cls) -> int:
        with cls._lock:
            return max(0, cls._max - cls._count)


class SecDailyQuota:
    """Persisted daily SEC request counter (UTC date)."""

    _lock = threading.Lock()

    @classmethod
    def _load(cls) -> dict:
        path = _quota_path()
        if not path.is_file():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @classmethod
    def _save(cls, data: dict) -> None:
        path = _quota_path()
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    @classmethod
    def _max_daily(cls) -> int:
        raw = settings_store().get_setting("sec_max_requests_per_day", str(DEFAULT_SEC_MAX_PER_DAY))
        try:
            return max(1, int(raw or DEFAULT_SEC_MAX_PER_DAY))
        except ValueError:
            return DEFAULT_SEC_MAX_PER_DAY

    @classmethod
    def try_acquire(cls) -> bool:
        today = date.today().isoformat()
        with cls._lock:
            data = cls._load()
            sec = data.setdefault("sec", {})
            if sec.get("date") != today:
                sec = {"date": today, "count": 0}
                data["sec"] = sec
            limit = cls._max_daily()
            if int(sec.get("count", 0)) >= limit:
                return False
            sec["count"] = int(sec.get("count", 0)) + 1
            cls._save(data)
            return True

    @classmethod
    def count_today(cls) -> int:
        today = date.today().isoformat()
        sec = cls._load().get("sec", {})
        if sec.get("date") != today:
            return 0
        return int(sec.get("count", 0))


class FmpDailyQuota:
    """Persisted daily FMP call counter with conservative budget."""

    _lock = threading.Lock()

    @classmethod
    def _budget(cls) -> int:
        raw = settings_store().get_setting("fmp_daily_budget", str(DEFAULT_FMP_DAILY_BUDGET))
        try:
            return max(1, min(250, int(raw or DEFAULT_FMP_DAILY_BUDGET)))
        except ValueError:
            return DEFAULT_FMP_DAILY_BUDGET

    @classmethod
    def try_acquire(cls) -> bool:
        today = date.today().isoformat()
        with cls._lock:
            data = SecDailyQuota._load()
            fmp = data.setdefault("fmp", {})
            if fmp.get("date") != today:
                fmp = {"date": today, "count": 0}
                data["fmp"] = fmp
            limit = cls._budget()
            if int(fmp.get("count", 0)) >= limit:
                return False
            fmp["count"] = int(fmp.get("count", 0)) + 1
            SecDailyQuota._save(data)
            return True

    @classmethod
    def remaining_today(cls) -> int:
        today = date.today().isoformat()
        fmp = SecDailyQuota._load().get("fmp", {})
        if fmp.get("date") != today:
            used = 0
        else:
            used = int(fmp.get("count", 0))
        return max(0, cls._budget() - used)


def ingest_uses_sec(source: str) -> bool:
    s = (source or "sec+yahoo").strip().lower()
    return "sec" in s


def reset_ingest_budgets(source: str) -> None:
    """Call once at the start of an ingest batch."""
    raw = settings_store().get_setting("sec_max_requests_per_run", str(DEFAULT_SEC_MAX_PER_RUN))
    try:
        per_run = max(1, int(raw or DEFAULT_SEC_MAX_PER_RUN))
    except ValueError:
        per_run = DEFAULT_SEC_MAX_PER_RUN
    if ingest_uses_sec(source):
        SecRunBudget.reset(per_run)
    yahoo_raw = settings_store().get_setting(
        "yahoo_min_interval_sec", str(DEFAULT_YAHOO_MIN_INTERVAL_SEC)
    )
    try:
        YahooRateLimiter.configure(float(yahoo_raw or DEFAULT_YAHOO_MIN_INTERVAL_SEC))
    except ValueError:
        YahooRateLimiter.configure(DEFAULT_YAHOO_MIN_INTERVAL_SEC)
