"""Settings key groups for StockConfig — documents domain boundaries."""

from __future__ import annotations

# Paths & universe
DATA_KEYS = frozenset({"db_path", "ticker_csv_path", "market_ticker"})

# Ingest throughput & sources
INGEST_KEYS = frozenset({
    "worker_count",
    "ingest_use_parallel",
    "ingest_worker_count",
    "ingest_fund_worker_count",
    "ingest_mode_default",
    "ingest_yahoo_interval_sec",
    "ingest_profile",
    "ingest_insider",
    "ingest_news",
    "ingest_stooq",
    "auto_skip_dead",
    "retry_dead_default",
})

# Local AI / LM Studio
AI_KEYS = frozenset({
    "lm_studio_enabled",
    "lm_studio_base_url",
    "lm_studio_model",
    "lm_studio_timeout_sec",
    "llm_backend_type",
    "ollama_host",
    "vllm_base_url",
    "vllm_api_key",
    "llm_chat_model",
    "llm_temperature",
    "llm_max_tokens",
    "llm_context_length",
    "llm_agents_enabled",
    "llm_autonomy_level",
    "llm_web_research_enabled",
    "llm_web_domain_allowlist",
    "llm_web_max_pages",
    "llm_web_max_bytes",
})

# Backtest & friction
BACKTEST_KEYS = frozenset({
    "canslim_stop_loss",
    "canslim_take_profit",
    "canslim_require_pattern",
    "canslim_rule_set_version",
    "hybrid_market_trend_weeks",
    "backtest_slippage_bps",
    "backtest_spread_bps",
    "backtest_fee_per_trade",
    "default_backtest_universe",
})

# Intraday / live streaming
INTRADAY_KEYS = frozenset({
    "finnhub_api_key",
    "intraday_interval",
    "intraday_backfill_days",
    "intraday_extended_hours",
    "live_stream_symbols_source",
})

# Leaderboard & daily scan
INTELLIGENCE_KEYS = frozenset({
    "leaderboard_auto_refresh",
    "leaderboard_score_version",
    "daily_scan_enabled",
    "daily_scan_ticker_limit",
    "dashboard_cache_sec",
})
