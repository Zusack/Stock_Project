"""
Stock application configuration backed by SettingsStore.

Mirrors paths and analysis defaults used by the analysis layer; persists to
data/settings.json alongside theme/logging keys.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from src.services.settings_store import settings_store

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "market_data.db"
_DEFAULT_TICKER_CSV = _PROJECT_ROOT / "tickerList.csv"


class StockConfig:
    """Singleton configuration for database paths and analysis defaults."""

    _instance: StockConfig | None = None
    _lock = threading.Lock()

    def __new__(cls) -> StockConfig:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    def _get(self, key: str, default: str) -> str:
        val = settings_store().get_setting(key, default)
        return val if val is not None else default

    def _set(self, key: str, value: str) -> None:
        settings_store().set_setting(key, value)

    @property
    def db_path(self) -> str:
        return self._get("db_path", str(_DEFAULT_DB))

    @db_path.setter
    def db_path(self, value: str) -> None:
        self._set("db_path", value)

    @property
    def ticker_csv_path(self) -> str:
        return self._get("ticker_csv_path", str(_DEFAULT_TICKER_CSV))

    @ticker_csv_path.setter
    def ticker_csv_path(self, value: str) -> None:
        self._set("ticker_csv_path", value)

    @property
    def market_ticker(self) -> str:
        return self._get("market_ticker", "^DJI")

    @market_ticker.setter
    def market_ticker(self, value: str) -> None:
        self._set("market_ticker", value.upper().strip())

    @property
    def use_parallel(self) -> bool:
        return self._get("use_parallel", "1") == "1"

    @use_parallel.setter
    def use_parallel(self, value: bool) -> None:
        self._set("use_parallel", "1" if value else "0")

    @property
    def worker_count(self) -> int:
        try:
            configured = max(1, int(self._get("worker_count", "8")))
        except ValueError:
            configured = 8
        cpus = os.cpu_count() or 4
        return max(1, min(configured, cpus))

    @worker_count.setter
    def worker_count(self, value: int) -> None:
        self._set("worker_count", str(max(1, int(value))))

    @property
    def sqlite_cache_mb(self) -> int:
        try:
            return max(0, int(self._get("sqlite_cache_mb", "256")))
        except ValueError:
            return 256

    @sqlite_cache_mb.setter
    def sqlite_cache_mb(self, value: int) -> None:
        self._set("sqlite_cache_mb", str(max(0, int(value))))

    @property
    def sqlite_mmap_mb(self) -> int:
        try:
            return max(0, int(self._get("sqlite_mmap_mb", "256")))
        except ValueError:
            return 256

    @sqlite_mmap_mb.setter
    def sqlite_mmap_mb(self, value: int) -> None:
        self._set("sqlite_mmap_mb", str(max(0, int(value))))

    @property
    def sqlite_temp_store_memory(self) -> bool:
        return self._get("sqlite_temp_store_memory", "1") == "1"

    @sqlite_temp_store_memory.setter
    def sqlite_temp_store_memory(self, value: bool) -> None:
        self._set("sqlite_temp_store_memory", "1" if value else "0")

    @property
    def leaderboard_auto_refresh(self) -> bool:
        return self._get("leaderboard_auto_refresh", "0") == "1"

    @leaderboard_auto_refresh.setter
    def leaderboard_auto_refresh(self, value: bool) -> None:
        self._set("leaderboard_auto_refresh", "1" if value else "0")

    @property
    def dashboard_cache_sec(self) -> int:
        try:
            return max(0, int(self._get("dashboard_cache_sec", "45")))
        except ValueError:
            return 45

    @dashboard_cache_sec.setter
    def dashboard_cache_sec(self, value: int) -> None:
        self._set("dashboard_cache_sec", str(max(0, int(value))))

    @property
    def canslim_stop_loss(self) -> float:
        try:
            return float(self._get("canslim_stop_loss", "-0.08"))
        except ValueError:
            return -0.08

    @canslim_stop_loss.setter
    def canslim_stop_loss(self, value: float) -> None:
        self._set("canslim_stop_loss", str(value))

    @property
    def canslim_take_profit(self) -> float:
        try:
            return float(self._get("canslim_take_profit", "0.25"))
        except ValueError:
            return 0.25

    @canslim_take_profit.setter
    def canslim_take_profit(self, value: float) -> None:
        self._set("canslim_take_profit", str(value))

    @property
    def hybrid_market_trend_weeks(self) -> int:
        try:
            return max(1, int(self._get("hybrid_market_trend_weeks", "20")))
        except ValueError:
            return 20

    @hybrid_market_trend_weeks.setter
    def hybrid_market_trend_weeks(self, value: int) -> None:
        self._set("hybrid_market_trend_weeks", str(max(1, int(value))))

    @property
    def last_ingest_at(self) -> str | None:
        return settings_store().get_setting("last_ingest_at")

    def set_last_ingest_at(self, iso_timestamp: str) -> None:
        self._set("last_ingest_at", iso_timestamp)

    @property
    def last_analysis_label(self) -> str | None:
        return settings_store().get_setting("last_analysis_label")

    def set_last_analysis_label(self, label: str) -> None:
        self._set("last_analysis_label", label)

    def db_exists(self) -> bool:
        return os.path.isfile(self.db_path)

    @property
    def fundamentals_source(self) -> str:
        return self._get("fundamentals_source", "sec+yahoo")

    @fundamentals_source.setter
    def fundamentals_source(self, value: str) -> None:
        self._set("fundamentals_source", (value or "sec+yahoo").strip().lower())

    @property
    def sec_user_agent(self) -> str:
        return self._get("sec_user_agent", "")

    @sec_user_agent.setter
    def sec_user_agent(self, value: str) -> None:
        self._set("sec_user_agent", (value or "").strip())

    @property
    def fmp_api_key(self) -> str:
        return self._get("fmp_api_key", "")

    @fmp_api_key.setter
    def fmp_api_key(self, value: str) -> None:
        self._set("fmp_api_key", (value or "").strip())

    @property
    def finnhub_api_key(self) -> str:
        return self._get("finnhub_api_key", "")

    @finnhub_api_key.setter
    def finnhub_api_key(self, value: str) -> None:
        self._set("finnhub_api_key", (value or "").strip())

    @property
    def intraday_interval(self) -> str:
        return self._get("intraday_interval", "1Min")

    @intraday_interval.setter
    def intraday_interval(self, value: str) -> None:
        self._set("intraday_interval", (value or "1Min").strip())

    @property
    def intraday_backfill_days(self) -> int:
        try:
            return max(1, min(365, int(self._get("intraday_backfill_days", "30"))))
        except ValueError:
            return 30

    @intraday_backfill_days.setter
    def intraday_backfill_days(self, value: int) -> None:
        self._set("intraday_backfill_days", str(max(1, min(365, int(value)))))

    @property
    def intraday_extended_hours(self) -> bool:
        return self._get("intraday_extended_hours", "0") == "1"

    @intraday_extended_hours.setter
    def intraday_extended_hours(self, value: bool) -> None:
        self._set("intraday_extended_hours", "1" if value else "0")

    @property
    def live_stream_symbols_source(self) -> str:
        return self._get("live_stream_symbols_source", "focus").strip().lower() or "focus"

    @live_stream_symbols_source.setter
    def live_stream_symbols_source(self, value: str) -> None:
        self._set("live_stream_symbols_source", (value or "focus").strip().lower())

    @property
    def sec_max_requests_per_run(self) -> int:
        try:
            return max(1, int(self._get("sec_max_requests_per_run", "300")))
        except ValueError:
            return 300

    @property
    def sec_max_requests_per_day(self) -> int:
        try:
            return max(1, int(self._get("sec_max_requests_per_day", "2000")))
        except ValueError:
            return 2000

    @property
    def fmp_daily_budget(self) -> int:
        try:
            return max(1, min(250, int(self._get("fmp_daily_budget", "200"))))
        except ValueError:
            return 200

    @property
    def ingest_mode_default(self) -> str:
        return self._get("ingest_mode_default", "smart")

    @ingest_mode_default.setter
    def ingest_mode_default(self, value: str) -> None:
        self._set("ingest_mode_default", (value or "smart").strip().lower())

    @property
    def fundamentals_refresh_days(self) -> int:
        try:
            return max(1, int(self._get("fundamentals_refresh_days", "30")))
        except ValueError:
            return 30

    @fundamentals_refresh_days.setter
    def fundamentals_refresh_days(self, value: int) -> None:
        self._set("fundamentals_refresh_days", str(max(1, int(value))))

    @property
    def force_full_history(self) -> bool:
        return self._get("force_full_history", "0") == "1"

    @force_full_history.setter
    def force_full_history(self, value: bool) -> None:
        self._set("force_full_history", "1" if value else "0")

    @property
    def retry_dead_tickers(self) -> bool:
        return self._get("retry_dead_tickers", "0") == "1"

    @retry_dead_tickers.setter
    def retry_dead_tickers(self, value: bool) -> None:
        self._set("retry_dead_tickers", "1" if value else "0")

    @property
    def auto_skip_dead_tickers(self) -> bool:
        return self._get("auto_skip_dead_tickers", "1") == "1"

    @auto_skip_dead_tickers.setter
    def auto_skip_dead_tickers(self, value: bool) -> None:
        self._set("auto_skip_dead_tickers", "1" if value else "0")

    @property
    def ingest_fetch_profile(self) -> bool:
        return self._get("ingest_fetch_profile", "1") == "1"

    @ingest_fetch_profile.setter
    def ingest_fetch_profile(self, value: bool) -> None:
        self._set("ingest_fetch_profile", "1" if value else "0")

    @property
    def ingest_fetch_insider(self) -> bool:
        return self._get("ingest_fetch_insider", "0") == "1"

    @ingest_fetch_insider.setter
    def ingest_fetch_insider(self, value: bool) -> None:
        self._set("ingest_fetch_insider", "1" if value else "0")

    @property
    def ingest_fetch_news(self) -> bool:
        return self._get("ingest_fetch_news", "0") == "1"

    @ingest_fetch_news.setter
    def ingest_fetch_news(self, value: bool) -> None:
        self._set("ingest_fetch_news", "1" if value else "0")

    @property
    def ingest_use_parallel(self) -> bool:
        return self._get("ingest_use_parallel", "0") == "1"

    @ingest_use_parallel.setter
    def ingest_use_parallel(self, value: bool) -> None:
        self._set("ingest_use_parallel", "1" if value else "0")

    @property
    def ingest_worker_count(self) -> int:
        try:
            configured = max(1, int(self._get("ingest_worker_count", "4")))
        except ValueError:
            configured = 4
        cpus = os.cpu_count() or 4
        cap = max(4, min(32, cpus // 2 or 4))
        return max(1, min(cap, configured))

    @ingest_worker_count.setter
    def ingest_worker_count(self, value: int) -> None:
        cpus = os.cpu_count() or 4
        cap = max(4, min(32, cpus // 2 or 4))
        self._set("ingest_worker_count", str(max(1, min(cap, int(value)))))

    @property
    def ingest_fund_worker_count(self) -> int:
        try:
            return max(1, min(8, int(self._get("ingest_fund_worker_count", "2"))))
        except ValueError:
            return 2

    @ingest_fund_worker_count.setter
    def ingest_fund_worker_count(self, value: int) -> None:
        self._set("ingest_fund_worker_count", str(max(1, min(8, int(value)))))

    @property
    def yahoo_min_interval_sec(self) -> float:
        try:
            return max(0.1, float(self._get("yahoo_min_interval_sec", "0.75")))
        except ValueError:
            return 0.75

    @yahoo_min_interval_sec.setter
    def yahoo_min_interval_sec(self, value: float) -> None:
        self._set("yahoo_min_interval_sec", str(max(0.1, float(value))))

    @property
    def ingest_use_stooq_backfill(self) -> bool:
        return self._get("ingest_use_stooq_backfill", "1") == "1"

    @ingest_use_stooq_backfill.setter
    def ingest_use_stooq_backfill(self, value: bool) -> None:
        self._set("ingest_use_stooq_backfill", "1" if value else "0")

    @property
    def ingest_price_fetch_timeout_sec(self) -> float:
        try:
            return max(30.0, float(self._get("ingest_price_fetch_timeout_sec", "180")))
        except ValueError:
            return 180.0

    @ingest_price_fetch_timeout_sec.setter
    def ingest_price_fetch_timeout_sec(self, value: float) -> None:
        self._set("ingest_price_fetch_timeout_sec", str(max(30.0, float(value))))

    @property
    def canslim_rule_set_version(self) -> str:
        return self._get("canslim_rule_set_version", "oneil_v1")

    @canslim_rule_set_version.setter
    def canslim_rule_set_version(self, value: str) -> None:
        self._set("canslim_rule_set_version", (value or "oneil_v1").strip())

    @property
    def canslim_require_pattern(self) -> bool:
        return self._get("canslim_require_pattern", "0") == "1"

    @canslim_require_pattern.setter
    def canslim_require_pattern(self, value: bool) -> None:
        self._set("canslim_require_pattern", "1" if value else "0")

    @property
    def lm_studio_enabled(self) -> bool:
        return self._get("lm_studio_enabled", "0") == "1"

    @lm_studio_enabled.setter
    def lm_studio_enabled(self, value: bool) -> None:
        self._set("lm_studio_enabled", "1" if value else "0")

    @property
    def lm_studio_base_url(self) -> str:
        return self._get("lm_studio_base_url", "http://localhost:1234/v1")

    @lm_studio_base_url.setter
    def lm_studio_base_url(self, value: str) -> None:
        self._set("lm_studio_base_url", (value or "http://localhost:1234/v1").strip())

    @property
    def lm_studio_model(self) -> str:
        return self._get("lm_studio_model", "")

    @lm_studio_model.setter
    def lm_studio_model(self, value: str) -> None:
        self._set("lm_studio_model", (value or "").strip())

    @property
    def lm_studio_timeout_sec(self) -> float:
        try:
            return max(5.0, float(self._get("lm_studio_timeout_sec", "60")))
        except ValueError:
            return 60.0

    @lm_studio_timeout_sec.setter
    def lm_studio_timeout_sec(self, value: float) -> None:
        self._set("lm_studio_timeout_sec", str(max(5.0, float(value))))

    @property
    def llm_backend_type(self) -> str:
        v = self._get("llm_backend_type", "lmstudio").strip().lower()
        return v if v in ("lmstudio", "ollama", "vllm") else "lmstudio"

    @llm_backend_type.setter
    def llm_backend_type(self, value: str) -> None:
        v = (value or "lmstudio").strip().lower()
        self._set("llm_backend_type", v if v in ("lmstudio", "ollama", "vllm") else "lmstudio")

    @property
    def ollama_host(self) -> str:
        return self._get("ollama_host", "http://localhost:11434")

    @ollama_host.setter
    def ollama_host(self, value: str) -> None:
        self._set("ollama_host", (value or "http://localhost:11434").strip())

    @property
    def vllm_base_url(self) -> str:
        return self._get("vllm_base_url", "http://localhost:8000")

    @vllm_base_url.setter
    def vllm_base_url(self, value: str) -> None:
        self._set("vllm_base_url", (value or "http://localhost:8000").strip())

    @property
    def vllm_api_key(self) -> str:
        return self._get("vllm_api_key", "")

    @vllm_api_key.setter
    def vllm_api_key(self, value: str) -> None:
        self._set("vllm_api_key", (value or "").strip())

    @property
    def llm_chat_model(self) -> str:
        return self._get("llm_chat_model", "")

    @llm_chat_model.setter
    def llm_chat_model(self, value: str) -> None:
        self._set("llm_chat_model", (value or "").strip())

    @property
    def llm_temperature(self) -> float:
        try:
            return max(0.0, min(2.0, float(self._get("llm_temperature", "0.7"))))
        except ValueError:
            return 0.7

    @llm_temperature.setter
    def llm_temperature(self, value: float) -> None:
        self._set("llm_temperature", str(max(0.0, min(2.0, float(value)))))

    @property
    def llm_max_tokens(self) -> int:
        try:
            return max(256, int(self._get("llm_max_tokens", "4096")))
        except ValueError:
            return 4096

    @llm_max_tokens.setter
    def llm_max_tokens(self, value: int) -> None:
        self._set("llm_max_tokens", str(max(256, int(value))))

    @property
    def llm_context_length(self) -> int:
        try:
            return max(2048, int(self._get("llm_context_length", "8192")))
        except ValueError:
            return 8192

    @llm_context_length.setter
    def llm_context_length(self, value: int) -> None:
        self._set("llm_context_length", str(max(2048, int(value))))

    @property
    def llm_agents_enabled(self) -> bool:
        return self._get("llm_agents_enabled", "1") == "1"

    @llm_agents_enabled.setter
    def llm_agents_enabled(self, value: bool) -> None:
        self._set("llm_agents_enabled", "1" if value else "0")

    @property
    def llm_autonomy_level(self) -> str:
        v = self._get("llm_autonomy_level", "confirm").strip().lower()
        return v if v in ("manual", "confirm", "auto") else "confirm"

    @llm_autonomy_level.setter
    def llm_autonomy_level(self, value: str) -> None:
        v = (value or "confirm").strip().lower()
        self._set("llm_autonomy_level", v if v in ("manual", "confirm", "auto") else "confirm")

    @property
    def llm_web_research_enabled(self) -> bool:
        return self._get("llm_web_research_enabled", "1") == "1"

    @llm_web_research_enabled.setter
    def llm_web_research_enabled(self, value: bool) -> None:
        self._set("llm_web_research_enabled", "1" if value else "0")

    @property
    def llm_web_domain_allowlist(self) -> str:
        return self._get(
            "llm_web_domain_allowlist",
            "finance.yahoo.com,sec.gov,marketwatch.com,reuters.com,bloomberg.com",
        )

    @llm_web_domain_allowlist.setter
    def llm_web_domain_allowlist(self, value: str) -> None:
        self._set("llm_web_domain_allowlist", (value or "").strip())

    @property
    def llm_web_max_pages(self) -> int:
        try:
            return max(1, min(10, int(self._get("llm_web_max_pages", "3"))))
        except ValueError:
            return 3

    @llm_web_max_pages.setter
    def llm_web_max_pages(self, value: int) -> None:
        self._set("llm_web_max_pages", str(max(1, min(10, int(value)))))

    @property
    def llm_web_max_bytes(self) -> int:
        try:
            return max(50_000, int(self._get("llm_web_max_bytes", "500000")))
        except ValueError:
            return 500_000

    @llm_web_max_bytes.setter
    def llm_web_max_bytes(self, value: int) -> None:
        self._set("llm_web_max_bytes", str(max(50_000, int(value))))

    def backend_settings(self) -> dict:
        """Settings dict for BackendFactory (LM Studio SDK host has no /v1 suffix)."""
        lm_url = self.lm_studio_base_url.rstrip("/")
        if lm_url.endswith("/v1"):
            lm_url = lm_url[:-3]
        return {
            "lmstudio_base_url": lm_url or "http://localhost:1234",
            "ollama_host": self.ollama_host,
            "vllm_base_url": self.vllm_base_url,
            "vllm_api_key": self.vllm_api_key,
        }

    @property
    def leaderboard_score_version(self) -> str:
        v = self._get("leaderboard_score_version", "v2")
        return v if v in ("v1", "v2") else "v2"

    @leaderboard_score_version.setter
    def leaderboard_score_version(self, value: str) -> None:
        v = (value or "v2").strip().lower()
        self._set("leaderboard_score_version", v if v in ("v1", "v2") else "v2")

    @property
    def backtest_slippage_bps(self) -> float:
        try:
            return max(0.0, float(self._get("backtest_slippage_bps", "5")))
        except ValueError:
            return 5.0

    @backtest_slippage_bps.setter
    def backtest_slippage_bps(self, value: float) -> None:
        self._set("backtest_slippage_bps", str(max(0.0, float(value))))

    @property
    def backtest_spread_bps(self) -> float:
        try:
            return max(0.0, float(self._get("backtest_spread_bps", "2")))
        except ValueError:
            return 2.0

    @backtest_spread_bps.setter
    def backtest_spread_bps(self, value: float) -> None:
        self._set("backtest_spread_bps", str(max(0.0, float(value))))

    @property
    def backtest_fee_per_trade(self) -> float:
        try:
            return max(0.0, float(self._get("backtest_fee_per_trade", "0")))
        except ValueError:
            return 0.0

    @backtest_fee_per_trade.setter
    def backtest_fee_per_trade(self, value: float) -> None:
        self._set("backtest_fee_per_trade", str(max(0.0, float(value))))

    @property
    def daily_scan_enabled(self) -> bool:
        return self._get("daily_scan_enabled", "1") == "1"

    @daily_scan_enabled.setter
    def daily_scan_enabled(self, value: bool) -> None:
        self._set("daily_scan_enabled", "1" if value else "0")

    @property
    def daily_scan_ticker_limit(self) -> int:
        try:
            return max(50, int(self._get("daily_scan_ticker_limit", "200")))
        except ValueError:
            return 200

    @daily_scan_ticker_limit.setter
    def daily_scan_ticker_limit(self, value: int) -> None:
        self._set("daily_scan_ticker_limit", str(max(50, int(value))))

    @property
    def guidance_auto_on_startup(self) -> bool:
        return self._get("guidance_auto_on_startup", "0") == "1"

    @guidance_auto_on_startup.setter
    def guidance_auto_on_startup(self, value: bool) -> None:
        self._set("guidance_auto_on_startup", "1" if value else "0")

    @property
    def focus_watchlist_max(self) -> int:
        try:
            return max(1, min(20, int(self._get("focus_watchlist_max", "20"))))
        except ValueError:
            return 20

    @focus_watchlist_max.setter
    def focus_watchlist_max(self, value: int) -> None:
        self._set("focus_watchlist_max", str(max(1, min(20, int(value)))))

    @property
    def universe_ingest_interval_days(self) -> int:
        try:
            return max(1, int(self._get("universe_ingest_interval_days", "7")))
        except ValueError:
            return 7

    @universe_ingest_interval_days.setter
    def universe_ingest_interval_days(self, value: int) -> None:
        self._set("universe_ingest_interval_days", str(max(1, int(value))))

    @property
    def last_universe_ingest_at(self) -> str | None:
        return settings_store().get_setting("last_universe_ingest_at")

    def set_last_universe_ingest_at(self, iso_timestamp: str) -> None:
        self._set("last_universe_ingest_at", iso_timestamp)

    @property
    def default_backtest_universe(self) -> str:
        return self._get("default_backtest_universe", "with_history")

    @default_backtest_universe.setter
    def default_backtest_universe(self, value: str) -> None:
        self._set("default_backtest_universe", (value or "with_history").strip().lower())

    @property
    def daily_scan_scope(self) -> str:
        return self._get("daily_scan_scope", "focus")

    @daily_scan_scope.setter
    def daily_scan_scope(self, value: str) -> None:
        self._set("daily_scan_scope", (value or "focus").strip().lower())


def stock_config() -> StockConfig:
    return StockConfig()
