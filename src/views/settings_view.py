"""
Settings view — Appearance, Logging, and About.

Mirrors the parent project's settings structure (ExpansionTile per section,
themed containers) but only includes UI-relevant options. The LLM backend,
thermal/power, and benchmark configuration sections have been removed since
they don't apply to a stock analyzer.

Add new sections by appending to ``section_blocks`` in __init__.
"""
from __future__ import annotations

import os
import subprocess
import sys

import flet as ft

from src.analysis import ticker_registry as registry
from src.services.settings_store import settings_store
from src.services.stock_config import stock_config
from src.services.tab_indices import TAB_SETTINGS
from src.services.theme_settings import (
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    app_theme,
)
from src.utils.logger_utils import app_logger
from src.views.base_view import BaseView
from src.views.components.feedback import AsyncStatusRow, show_snackbar
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.settings_sections import build_ai_settings_panel, build_backtest_settings_panel
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import on_ticker_field_blur


class SettingsView(BaseView):
    _tab_index = TAB_SETTINGS

    def __init__(self, page: ft.Page):
        super().__init__(page)
        cfg = stock_config()

        # --- Data paths ---
        self.db_path_field = ft.TextField(label="Database path", value=cfg.db_path, expand=True)
        self.csv_path_field = ft.TextField(
            label="Universe import file (CSV)",
            value=cfg.ticker_csv_path,
            expand=True,
            hint_text="Seeds research universe only — not the focus watchlist",
        )
        self.import_csv_btn = ft.ElevatedButton(
            "Import universe from CSV",
            icon=ft.Icons.UPLOAD_FILE,
            style=ButtonStyles.secondary(),
            on_click=self._import_csv_to_watchlist,
            tooltip=(
                "Import symbols from tickerList.csv into the research universe "
                "(pool=universe). Does not add to the focus watchlist."
            ),
        )
        self.market_ticker_field = ft.TextField(
            label="Market proxy ticker",
            value=cfg.market_ticker,
            width=160,
            on_blur=on_ticker_field_blur(multi=False),
            tooltip=(
                "Index used for market direction (M) and relative strength "
                "(e.g. ^DJI, ^GSPC). Must exist in your database."
            ),
        )
        self.parallel_switch = ft.Switch(
            label="Use parallel processing",
            value=cfg.use_parallel,
            tooltip=(
                "Run backtests, leaderboard scoring, and optimizations across "
                "multiple CPU cores (recommended on multi-core machines)."
            ),
        )
        self.worker_count_field = ft.TextField(
            label="Analysis worker count",
            value=str(cfg.worker_count),
            width=120,
            tooltip="Number of parallel workers for analysis jobs (typically ≤ CPU core count).",
        )
        self.sqlite_cache_field = ft.TextField(
            label="SQLite cache (MB)",
            value=str(cfg.sqlite_cache_mb),
            width=140,
            tooltip="Page cache for large databases on high-RAM machines.",
        )
        self.sqlite_mmap_field = ft.TextField(
            label="SQLite mmap (MB)",
            value=str(cfg.sqlite_mmap_mb),
            width=140,
            tooltip="Memory-map database file pages for faster reads on large databases.",
        )
        self.dashboard_cache_field = ft.TextField(
            label="Dashboard cache (sec)",
            value=str(cfg.dashboard_cache_sec),
            width=160,
            tooltip="Seconds to reuse dashboard market data before reloading from SQLite.",
        )
        self.leaderboard_auto_refresh_switch = ft.Switch(
            label="Auto-refresh Leaderboard on tab open",
            value=cfg.leaderboard_auto_refresh,
            tooltip=(
                "When on, opening the Leaderboard tab starts a full scan automatically. "
                "Off is faster — use Refresh rankings manually."
            ),
        )
        self.leaderboard_score_version_dropdown = ft.Dropdown(
            label="Leaderboard composite version",
            value=cfg.leaderboard_score_version,
            width=220,
            options=[
                ft.dropdown.Option("v2", "v2 — Balanced multi-factor (default)"),
                ft.dropdown.Option("v1", "v1 — Legacy CANSLM weights"),
            ],
            tooltip=(
                "v2 uses momentum, quality, value, risk/regime, and sentiment buckets. "
                "v1 keeps the original five-factor CANSLM proxy."
            ),
        )
        self.fundamentals_source_dropdown = ft.Dropdown(
            label="Fundamentals source",
            value=cfg.fundamentals_source,
            width=220,
            options=[
                ft.dropdown.Option("sec+yahoo", "SEC + Yahoo (recommended)"),
                ft.dropdown.Option("sec", "SEC only"),
                ft.dropdown.Option("yahoo", "Yahoo only (~5 quarters)"),
                ft.dropdown.Option("fmp", "FMP API (key required)"),
                ft.dropdown.Option("fmp+yahoo", "FMP + Yahoo"),
            ],
            tooltip="SEC is free for US stocks with multi-year history. Yahoo is a short fallback.",
        )
        self.sec_user_agent_field = ft.TextField(
            label="SEC User-Agent (required for SEC)",
            value=cfg.sec_user_agent,
            expand=True,
            hint_text="StockProject/1.0 your@email.com",
            tooltip="SEC requires an identifying User-Agent with contact email.",
        )
        self.fmp_api_key_field = ft.TextField(
            label="FMP API key (optional)",
            value=cfg.fmp_api_key,
            password=True,
            can_reveal_password=True,
            width=280,
            tooltip="Free tier ~250 calls/day; app defaults to 200/day budget.",
        )
        self.ingest_mode_default_dropdown = ft.Dropdown(
            label="Default ingest mode",
            value=cfg.ingest_mode_default,
            width=220,
            tooltip="Default for new ingest runs: incremental smart update vs full history replace.",
            options=[
                ft.dropdown.Option("smart", "Smart update (incremental)"),
                ft.dropdown.Option("full", "Full re-download"),
            ],
        )
        self.fundamentals_refresh_field = ft.TextField(
            label="Fundamentals refresh (days)",
            value=str(cfg.fundamentals_refresh_days),
            width=200,
        )
        self.force_full_switch = ft.Switch(
            label="Force full price history on next ingest",
            value=cfg.force_full_history,
            tooltip="Next ingest run re-downloads complete history for every symbol.",
        )
        self.auto_skip_dead_switch = ft.Switch(
            label="Auto-archive tickers with no Yahoo price data",
            value=cfg.auto_skip_dead_tickers,
            tooltip=(
                "Automatically pause ingest for symbols Yahoo cannot price "
                "(avoids repeated failed downloads)."
            ),
        )
        self.retry_dead_default_switch = ft.Switch(
            label="Default: re-try skipped dead tickers on ingest",
            value=cfg.retry_dead_tickers,
            tooltip="When on, ingest attempts archived or previously skipped symbols again.",
        )
        self.ingest_parallel_switch = ft.Switch(
            label="Parallel ingest (price workers + phased fundamentals/enrich)",
            value=cfg.ingest_use_parallel,
            tooltip="Uses separate Yahoo/SEC throttles. Keep Yahoo workers at 2–3.",
        )
        self.ingest_worker_count_field = ft.TextField(
            label="Ingest Yahoo price workers",
            value=str(cfg.ingest_worker_count),
            width=160,
            tooltip="Parallel threads fetching Yahoo prices (balance vs rate limits; try 3–6).",
        )
        self.ingest_fund_worker_count_field = ft.TextField(
            label="Ingest fundamentals workers",
            value=str(cfg.ingest_fund_worker_count),
            width=180,
            tooltip="Parallel threads for SEC/Yahoo fundamentals phase after prices.",
        )
        self.yahoo_interval_field = ft.TextField(
            label="Yahoo min interval (sec)",
            value=str(cfg.yahoo_min_interval_sec),
            width=160,
            tooltip="Spacing between Yahoo requests (yfinance).",
        )
        self.ingest_profile_switch = ft.Switch(
            label="Ingest: fetch company profile (Yahoo)",
            value=cfg.ingest_fetch_profile,
            tooltip="Store sector, industry, and company metadata in stock_profiles.",
        )
        self.ingest_insider_switch = ft.Switch(
            label="Ingest: fetch insider transactions",
            value=cfg.ingest_fetch_insider,
            tooltip="Download recent insider buys/sells (extra Yahoo calls per symbol).",
        )
        self.ingest_news_switch = ft.Switch(
            label="Ingest: fetch news headlines",
            value=cfg.ingest_fetch_news,
            tooltip="Store recent headlines for sentiment and research (slower enrich phase).",
        )
        self.ingest_stooq_switch = ft.Switch(
            label="Stooq backfill for empty price history (US)",
            value=cfg.ingest_use_stooq_backfill,
            tooltip="Uses free Stooq CSV for initial OHLCV; Yahoo for incremental updates.",
        )
        self.finnhub_api_key_field = ft.TextField(
            label="Finnhub API key",
            value=cfg.finnhub_api_key,
            password=True,
            can_reveal_password=True,
            width=280,
            tooltip=(
                "Free Finnhub key (email signup, no SSN) for live trade streaming. "
                "Get one at finnhub.io/register."
            ),
        )
        self.intraday_interval_dropdown = ft.Dropdown(
            label="Intraday bar interval",
            value=cfg.intraday_interval,
            width=160,
            options=[
                ft.dropdown.Option("1Min", "1 minute"),
                ft.dropdown.Option("5Min", "5 minutes"),
                ft.dropdown.Option("15Min", "15 minutes"),
                ft.dropdown.Option("1Hour", "1 hour"),
            ],
        )
        self.intraday_backfill_days_field = ft.TextField(
            label="Intraday backfill (days)",
            value=str(cfg.intraday_backfill_days),
            width=180,
            tooltip="Days of minute bars to fetch per focus symbol on backfill.",
        )
        self.intraday_extended_hours_switch = ft.Switch(
            label="Include extended hours (pre/post market)",
            value=cfg.intraday_extended_hours,
        )
        self.live_stream_source_dropdown = ft.Dropdown(
            label="Live stream symbol source",
            value=cfg.live_stream_symbols_source,
            width=280,
            options=[
                ft.dropdown.Option(
                    "focus",
                    "Focus watchlist + chart + Also stream",
                ),
                ft.dropdown.Option(
                    "custom",
                    "Chart symbol + Also stream only",
                ),
            ],
            tooltip=(
                "Focus mode streams your Watchlist tab symbols plus anything you "
                "enter on the Live tab. Chart-only mode ignores the watchlist."
            ),
        )
        self.save_intraday_btn = ft.ElevatedButton(
            "Save intraday / live settings",
            icon=ft.Icons.SAVE,
            style=ButtonStyles.primary(),
            on_click=self._save_intraday_settings,
        )
        self.sec_limits_hint = ft.Text(
            "Rate limits: SEC ~6.7 req/s, 300/run, 2000/day; Yahoo spacing configurable. "
            "Ingest: price → fundamentals → optional enrich; parallel pools optional.",
            size=11,
            color=ThemeHelper.text_muted(page),
        )
        self.save_data_btn = ft.ElevatedButton(
            "Save data settings",
            icon=ft.Icons.SAVE,
            style=ButtonStyles.primary(),
            on_click=self._save_data_settings,
            tooltip=(
                "Persist database paths, ingest defaults, API keys, and performance "
                "options to data/settings.json."
            ),
        )

        # --- Backtest defaults ---
        self.canslim_stop_field = ft.TextField(
            label="CANSLM stop loss", value=str(cfg.canslim_stop_loss), width=140
        )
        self.canslim_tp_field = ft.TextField(
            label="CANSLM take profit", value=str(cfg.canslim_take_profit), width=140
        )
        self.canslim_require_pattern_switch = ft.Switch(
            label="CANSLM: require cup-with-handle pattern for entries",
            value=cfg.canslim_require_pattern,
            tooltip="Backtest only enters when cup-with-handle pattern passes (stricter).",
        )
        self.canslim_rule_version_field = ft.TextField(
            label="CANSLM rule set version",
            value=cfg.canslim_rule_set_version,
            width=180,
        )
        self.lm_studio_switch = ft.Switch(
            label="Enable LM Studio (local AI)",
            value=cfg.lm_studio_enabled,
            tooltip=(
                "Master switch for local AI features (Assistant tab, agents, headline analysis). "
                "GPU and model loading in the LM Studio app are still configured in LM Studio."
            ),
        )
        self.lm_studio_url_field = ft.TextField(
            label="LM Studio base URL",
            value=cfg.lm_studio_base_url,
            width=400,
            tooltip=(
                "OpenAI-compatible HTTP base URL for legacy headline analysis (usually "
                "http://localhost:1234/v1). The SDK backend strips /v1 automatically."
            ),
        )
        self.lm_studio_model_field = ft.TextField(
            label="Headline model (HTTP)",
            value=cfg.lm_studio_model,
            width=220,
            hint_text="local-model",
            tooltip=(
                "Model name for quick headline sentiment via the HTTP API fallback. "
                "Leave blank to use the server's default loaded model."
            ),
        )
        self.lm_studio_timeout_field = ft.TextField(
            label="Request timeout (sec)",
            value=str(cfg.lm_studio_timeout_sec),
            width=140,
            tooltip=(
                "Maximum seconds to wait for a backend connection, model listing, or a single "
                "LLM response before the app cancels with an error. This is not an idle "
                "unload timer and does not control how long LM Studio keeps a model in memory."
            ),
        )
        self.lm_studio_test_btn = ft.ElevatedButton(
            "Test connection",
            icon=ft.Icons.LAN,
            style=ButtonStyles.secondary(),
            on_click=self._test_lm_studio,
            tooltip=(
                "Ping the selected backend and list available models. Does not load a model. "
                "Results appear in the status line below and as a snackbar."
            ),
        )
        self.ai_test_status = AsyncStatusRow(page)
        self.llm_backend_dropdown = ft.Dropdown(
            label="Backend",
            value=cfg.llm_backend_type,
            width=160,
            tooltip="LLM provider for the Assistant tab and SDK-based features.",
            options=[
                ft.dropdown.Option("lmstudio", "LM Studio"),
                ft.dropdown.Option("ollama", "Ollama"),
                ft.dropdown.Option("vllm", "vLLM"),
            ],
        )
        self.ollama_host_field = ft.TextField(
            label="Ollama host",
            value=cfg.ollama_host,
            width=260,
            tooltip="Base URL of your Ollama server (default http://localhost:11434).",
        )
        self.vllm_url_field = ft.TextField(
            label="vLLM base URL",
            value=cfg.vllm_base_url,
            width=260,
            tooltip="OpenAI-compatible base URL for a vLLM or compatible inference server.",
        )
        self.vllm_key_field = ft.TextField(
            label="vLLM API key",
            value=cfg.vllm_api_key,
            password=True,
            can_reveal_password=True,
            width=200,
            tooltip="Optional API key if your vLLM server requires authentication.",
        )
        self.llm_chat_model_field = ft.TextField(
            label="Default chat model",
            value=cfg.llm_chat_model,
            width=220,
            hint_text="model identifier",
            tooltip=(
                "Model identifier the Assistant tab loads via the SDK (must match a model "
                "available on the backend). Use Test connection to see identifiers."
            ),
        )
        self.llm_temperature_field = ft.TextField(
            label="Temperature",
            value=str(cfg.llm_temperature),
            width=100,
            tooltip=(
                "Randomness for each generation request sent by this app (0=focused, higher=more "
                "creative). Applied per API call; may override LM Studio defaults for app-driven "
                "chat and agents only."
            ),
        )
        self.llm_max_tokens_field = ft.TextField(
            label="Max tokens",
            value=str(cfg.llm_max_tokens),
            width=120,
            tooltip=(
                "Maximum number of new tokens the model may generate in one response. "
                "Caps output length, not the input context window."
            ),
        )
        self.llm_context_length_field = ft.TextField(
            label="Context length",
            value=str(cfg.llm_context_length),
            width=140,
            tooltip=(
                "Context window (tokens) used when this app loads a model via the SDK and when "
                "trimming chat history. Should be ≤ the model's supported context. Distinct from "
                "Max tokens, which limits output size per reply."
            ),
        )
        self.llm_agents_switch = ft.Switch(
            label="Enable tool-calling agents",
            value=cfg.llm_agents_enabled,
            tooltip="Allow agents to call app tools (quotes, news, web research, etc.).",
        )
        self.llm_autonomy_dropdown = ft.Dropdown(
            label="Agent autonomy",
            value=cfg.llm_autonomy_level,
            width=180,
            tooltip=(
                "Manual: chat only, no tools. Confirm: tools with user oversight. "
                "Auto: agents may call tools without pausing."
            ),
            options=[
                ft.dropdown.Option("manual", "Manual (no tools)"),
                ft.dropdown.Option("confirm", "Confirm tool calls"),
                ft.dropdown.Option("auto", "Auto"),
            ],
        )
        self.llm_web_research_switch = ft.Switch(
            label="Enable web research (scrape)",
            value=cfg.llm_web_research_enabled,
            tooltip="Allow agents to fetch allowed financial news URLs for research.",
        )
        self.llm_web_allowlist_field = ft.TextField(
            label="Allowed domains (comma-separated)",
            value=cfg.llm_web_domain_allowlist,
            width=520,
            multiline=True,
            min_lines=2,
            max_lines=4,
            tooltip="Only these domains may be fetched by web research tools.",
        )
        self.llm_web_max_pages_field = ft.TextField(
            label="Max pages per search",
            value=str(cfg.llm_web_max_pages),
            width=160,
            tooltip="Maximum pages to fetch per web_search_lite query.",
        )
        self.llm_web_max_bytes_field = ft.TextField(
            label="Max bytes per page",
            value=str(cfg.llm_web_max_bytes),
            width=160,
            tooltip="Maximum downloaded bytes per page (HTML truncated after this).",
        )
        self.hybrid_weeks_field = ft.TextField(
            label="Hybrid market trend (weeks)",
            value=str(cfg.hybrid_market_trend_weeks),
            width=180,
        )
        self.slippage_bps_field = ft.TextField(
            label="Backtest slippage (bps)",
            value=str(cfg.backtest_slippage_bps),
            width=160,
            tooltip="Basis points added to simulated trade friction.",
        )
        self.spread_bps_field = ft.TextField(
            label="Backtest spread (bps)",
            value=str(cfg.backtest_spread_bps),
            width=160,
        )
        self.fee_per_trade_field = ft.TextField(
            label="Fee per trade ($)",
            value=str(cfg.backtest_fee_per_trade),
            width=160,
        )
        self.daily_scan_switch = ft.Switch(
            label="Enable daily scan jobs",
            value=cfg.daily_scan_enabled,
        )
        self.daily_scan_limit_field = ft.TextField(
            label="Daily scan ticker limit",
            value=str(cfg.daily_scan_ticker_limit),
            width=180,
        )
        self.save_backtest_btn = ft.ElevatedButton(
            "Save backtest defaults",
            icon=ft.Icons.SAVE,
            style=ButtonStyles.secondary(),
            on_click=self._save_backtest_settings,
            tooltip="Save CANSLM, hybrid, friction, and daily scan settings.",
        )
        self.save_ai_btn = ft.ElevatedButton(
            "Save AI settings",
            icon=ft.Icons.SAVE,
            style=ButtonStyles.secondary(),
            on_click=self._save_ai_settings,
            tooltip="Save LM Studio connection settings.",
        )

        self.open_data_folder_btn = ft.ElevatedButton(
            "Open data folder",
            icon=ft.Icons.FOLDER_OPEN,
            style=ButtonStyles.secondary(),
            on_click=self._open_data_folder,
            tooltip="Open the project data directory in your file manager (settings, logs).",
        )

        # --- Appearance ---
        current = app_theme().get_theme_mode()
        self.theme_segmented = ft.SegmentedButton(
            selected=[current],
            allow_empty_selection=False,
            segments=[
                ft.Segment(value=THEME_LIGHT, label=ft.Text("Light"), icon=ft.Icon(ft.Icons.LIGHT_MODE)),
                ft.Segment(value=THEME_DARK, label=ft.Text("Dark"), icon=ft.Icon(ft.Icons.DARK_MODE)),
                ft.Segment(value=THEME_SYSTEM, label=ft.Text("System"), icon=ft.Icon(ft.Icons.COMPUTER)),
            ],
            on_change=self._on_theme_segment_change,
            tooltip="Application color theme.",
        )

        # --- Logging ---
        self.logging_switch = ft.Switch(
            label="Enable diagnostic logging",
            value=app_logger.enabled,
            on_change=self._on_toggle_logging,
            tooltip="Write detailed diagnostic logs to exports/logs for troubleshooting.",
        )
        self.log_level_dropdown = ft.Dropdown(
            label="Log level",
            value=app_logger.get_level(),
            tooltip="Minimum severity written to the log file (DEBUG is most verbose).",
            options=[
                ft.DropdownOption(key="DEBUG", text="DEBUG"),
                ft.DropdownOption(key="INFO", text="INFO"),
                ft.DropdownOption(key="WARN", text="WARN"),
                ft.DropdownOption(key="ERROR", text="ERROR"),
            ],
            width=180,
            on_select=self._on_log_level_change,
        )
        self.view_logs_button = ft.ElevatedButton(
            "View Latest Log",
            icon=ft.Icons.VISIBILITY,
            on_click=self._open_logs_modal,
            style=ButtonStyles.secondary(),
            tooltip="Show the most recent log file in a dialog.",
        )

        self.logs_content = ft.TextField(
            multiline=True,
            read_only=True,
            text_size=12,
            text_style=ft.TextStyle(font_family="monospace"),
            expand=True,
        )
        self._form_text_fields = [
            self.db_path_field,
            self.csv_path_field,
            self.market_ticker_field,
            self.worker_count_field,
            self.sqlite_cache_field,
            self.sqlite_mmap_field,
            self.dashboard_cache_field,
            self.sec_user_agent_field,
            self.fmp_api_key_field,
            self.fundamentals_refresh_field,
            self.ingest_worker_count_field,
            self.ingest_fund_worker_count_field,
            self.yahoo_interval_field,
            self.canslim_stop_field,
            self.canslim_tp_field,
            self.canslim_rule_version_field,
            self.lm_studio_url_field,
            self.hybrid_weeks_field,
            self.slippage_bps_field,
            self.spread_bps_field,
            self.fee_per_trade_field,
            self.daily_scan_limit_field,
            self.logs_content,
        ]
        self._form_dropdowns = [
            self.fundamentals_source_dropdown,
            self.ingest_mode_default_dropdown,
            self.log_level_dropdown,
        ]
        InputStyles.refresh_fields(page, self._form_text_fields, self._form_dropdowns)

        self.logs_dlg = ft.AlertDialog(
            title=ft.Text("Latest Log"),
            content=ft.Container(content=self.logs_content, width=800, height=550, expand=True),
            actions=[
                ft.TextButton(
                    "Close",
                    tooltip="Close the log viewer.",
                    on_click=lambda e: self.page_ref.close(self.logs_dlg),
                )
            ],
        )

        # --- Settings sections ---
        section_blocks = {
            "data": [
                SectionHeader("Data", icon=ft.Icons.STORAGE, page_ref=page),
                self._themed_panel(
                    ft.Column(
                        [
                            self.db_path_field,
                            self.csv_path_field,
                            self.import_csv_btn,
                            ft.Row(
                                [
                                    self.market_ticker_field,
                                    self.parallel_switch,
                                    self.worker_count_field,
                                ],
                                wrap=True,
                                spacing=12,
                            ),
                            ft.Text("Performance", weight=ft.FontWeight.W_600, size=13),
                            ft.Row(
                                [
                                    self.sqlite_cache_field,
                                    self.sqlite_mmap_field,
                                    self.dashboard_cache_field,
                                ],
                                wrap=True,
                                spacing=12,
                            ),
                            self.leaderboard_auto_refresh_switch,
                            self.leaderboard_score_version_dropdown,
                            self.fundamentals_source_dropdown,
                            self.sec_user_agent_field,
                            self.fmp_api_key_field,
                            self.ingest_mode_default_dropdown,
                            self.fundamentals_refresh_field,
                            self.force_full_switch,
                            self.auto_skip_dead_switch,
                            self.retry_dead_default_switch,
                            ft.Text("Ingest throughput", weight=ft.FontWeight.W_600, size=13),
                            ft.Row(
                                [
                                    self.ingest_parallel_switch,
                                    self.ingest_worker_count_field,
                                    self.ingest_fund_worker_count_field,
                                    self.yahoo_interval_field,
                                ],
                                wrap=True,
                                spacing=12,
                            ),
                            ft.Row(
                                [
                                    self.ingest_profile_switch,
                                    self.ingest_insider_switch,
                                    self.ingest_news_switch,
                                ],
                                wrap=True,
                                spacing=12,
                            ),
                            self.ingest_stooq_switch,
                            self.sec_limits_hint,
                            self.save_data_btn,
                        ],
                        spacing=10,
                    )
                ),
            ],
            "intraday": [
                SectionHeader("Intraday / Live Data", icon=ft.Icons.CANDLESTICK_CHART, page_ref=page),
                self._themed_panel(
                    ft.Column(
                        [
                            ft.Text(
                                "Live streaming uses Finnhub (free, email signup — no SSN). "
                                "Intraday history is backfilled from Yahoo Finance (no signup; "
                                "1-minute bars cover the last ~7 days, other intervals up to 60 days).",
                                size=12,
                                color=ThemeHelper.text_muted(page),
                            ),
                            ft.Row(
                                [self.finnhub_api_key_field],
                                wrap=True,
                                spacing=12,
                            ),
                            ft.Row(
                                [
                                    self.intraday_interval_dropdown,
                                    self.intraday_backfill_days_field,
                                ],
                                wrap=True,
                                spacing=12,
                            ),
                            ft.Row(
                                [
                                    self.intraday_extended_hours_switch,
                                    self.live_stream_source_dropdown,
                                ],
                                wrap=True,
                                spacing=12,
                            ),
                            self.save_intraday_btn,
                        ],
                        spacing=10,
                    )
                ),
            ],
            "backtest": build_backtest_settings_panel(
                page,
                canslim_stop_field=self.canslim_stop_field,
                canslim_tp_field=self.canslim_tp_field,
                canslim_rule_version_field=self.canslim_rule_version_field,
                hybrid_weeks_field=self.hybrid_weeks_field,
                canslim_require_pattern_switch=self.canslim_require_pattern_switch,
                slippage_bps_field=self.slippage_bps_field,
                spread_bps_field=self.spread_bps_field,
                fee_per_trade_field=self.fee_per_trade_field,
                daily_scan_switch=self.daily_scan_switch,
                daily_scan_limit_field=self.daily_scan_limit_field,
                save_backtest_btn=self.save_backtest_btn,
                themed_panel=self._themed_panel,
            ),
            "ai": build_ai_settings_panel(
                page,
                lm_studio_switch=self.lm_studio_switch,
                lm_studio_url_field=self.lm_studio_url_field,
                lm_studio_model_field=self.lm_studio_model_field,
                lm_studio_timeout_field=self.lm_studio_timeout_field,
                llm_backend_dropdown=self.llm_backend_dropdown,
                ollama_host_field=self.ollama_host_field,
                vllm_url_field=self.vllm_url_field,
                vllm_key_field=self.vllm_key_field,
                llm_chat_model_field=self.llm_chat_model_field,
                llm_temperature_field=self.llm_temperature_field,
                llm_max_tokens_field=self.llm_max_tokens_field,
                llm_context_length_field=self.llm_context_length_field,
                llm_agents_switch=self.llm_agents_switch,
                llm_autonomy_dropdown=self.llm_autonomy_dropdown,
                llm_web_research_switch=self.llm_web_research_switch,
                llm_web_allowlist_field=self.llm_web_allowlist_field,
                llm_web_max_pages_field=self.llm_web_max_pages_field,
                llm_web_max_bytes_field=self.llm_web_max_bytes_field,
                lm_studio_test_btn=self.lm_studio_test_btn,
                ai_test_status=self.ai_test_status,
                save_ai_btn=self.save_ai_btn,
                themed_panel=self._themed_panel,
            ),
            "appearance": [
                SectionHeader("Appearance", icon=ft.Icons.PALETTE, page_ref=page),
                self._themed_panel(
                    ft.Column([self.theme_segmented], spacing=15)
                ),
            ],
            "logging": [
                SectionHeader("Logging", icon=ft.Icons.DESCRIPTION, page_ref=page),
                self._themed_panel(
                    ft.Column(
                        [
                            ft.Row([self.logging_switch, self.log_level_dropdown, self.view_logs_button], spacing=15, wrap=True),
                            ft.Text(
                                "Logs are written to exports/logs/. The 10 newest files are kept.",
                                size=11,
                                color=ThemeHelper.text_muted(page),
                            ),
                        ],
                        spacing=8,
                    )
                ),
            ],
            "diagnostics": [
                SectionHeader("Diagnostics", icon=ft.Icons.BUG_REPORT, page_ref=page),
                self._themed_panel(
                    ft.Column(
                        [
                            self.open_data_folder_btn,
                            ft.Text(
                                "Settings file: data/settings.json · Logs: exports/logs/",
                                size=11,
                                color=ThemeHelper.text_muted(page),
                            ),
                        ],
                        spacing=8,
                    )
                ),
            ],
            "about": [
                SectionHeader("About", icon=ft.Icons.INFO_OUTLINE, page_ref=page),
                self._themed_panel(
                    ft.Column(
                        [
                            ft.Text("Stock Analyzer", weight=ft.FontWeight.BOLD, size=16),
                            ft.Text(
                                "Local stock research app: ingest Yahoo Finance data into SQLite, "
                                "run CANSLM and strategy backtests, and explore optimization tools. "
                                "Not financial advice.",
                                size=12,
                                color=ThemeHelper.text_muted(page),
                            ),
                            ft.Text("Flet 0.85.1 · Version 0.2.0", size=12, color=ThemeHelper.text_muted(page)),
                        ],
                        spacing=6,
                        horizontal_alignment=ft.CrossAxisAlignment.START,
                    )
                ),
            ],
        }

        expansion_meta = {
            "data": ("Data & paths", ft.Icons.STORAGE),
            "intraday": ("Intraday / Live Data", ft.Icons.CANDLESTICK_CHART),
            "backtest": ("Backtest defaults", ft.Icons.TIMELINE),
            "ai": ("Local AI", ft.Icons.PSYCHOLOGY),
            "appearance": ("Appearance", ft.Icons.PALETTE),
            "logging": ("Diagnostics & Logging", ft.Icons.DESCRIPTION),
            "diagnostics": ("Folders", ft.Icons.FOLDER),
            "about": ("About", ft.Icons.INFO_OUTLINE),
        }
        section_tiles: list[ft.Control] = []
        for key in section_blocks.keys():
            title, icon = expansion_meta.get(key, (key.title(), ft.Icons.SETTINGS))
            section_tiles.append(
                ft.ExpansionTile(
                    title=ft.Text(title, weight=ft.FontWeight.W_600),
                    leading=ft.Icon(icon),
                    controls=[ft.Column(section_blocks[key], spacing=8, tight=True)],
                )
            )
        self._settings_body = ft.Column(
            controls=section_tiles,
            spacing=8,
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        self.controls = [
            ViewTitleBar("Settings"),
            ft.Divider(),
            self._settings_body,
        ]

    # ------------------------------------------------------------------
    # Themed section container helper
    # ------------------------------------------------------------------
    def _themed_panel(self, content: ft.Control) -> ft.Container:
        return ft.Container(
            content=content,
            padding=15,
            bgcolor=ThemeHelper.surface_dim(self.page_ref),
            border_radius=10,
            border=ft.border.all(1, ThemeHelper.border_default(self.page_ref)),
        )

    # ------------------------------------------------------------------
    # Appearance handlers
    # ------------------------------------------------------------------
    def _import_csv_to_watchlist(self, e) -> None:
        cfg = stock_config()
        csv_path = (self.csv_path_field.value or cfg.ticker_csv_path).strip()
        cfg.ticker_csv_path = csv_path
        db_path = cfg.db_path

        def _work():
            try:
                result = registry.migrate_csv_to_db(db_path, csv_path)
            except Exception as ex:
                result = {"error": str(ex)}

            def _ui():
                if result.get("error"):
                    show_snackbar(self.page_ref, result["error"], severity="error")
                    return
                show_snackbar(
                    self.page_ref,
                    f"Imported {result.get('newly_added', 0)} new symbols "
                    f"({result.get('total_in_watchlist', 0)} total in watchlist).",
                    severity="success",
                )

            from src.views.base_view import schedule_ui_update

            schedule_ui_update(self.page_ref, _ui, label="csv_import_done")

        threading.Thread(target=_work, daemon=True).start()

    def _save_data_settings(self, e) -> None:
        cfg = stock_config()
        cfg.db_path = (self.db_path_field.value or "").strip()
        cfg.ticker_csv_path = (self.csv_path_field.value or "").strip()
        cfg.market_ticker = (self.market_ticker_field.value or "^DJI").strip()
        cfg.use_parallel = bool(self.parallel_switch.value)
        try:
            cfg.worker_count = int(self.worker_count_field.value or "8")
        except ValueError:
            pass
        try:
            cfg.sqlite_cache_mb = int(self.sqlite_cache_field.value or "256")
            cfg.sqlite_mmap_mb = int(self.sqlite_mmap_field.value or "256")
            cfg.dashboard_cache_sec = int(self.dashboard_cache_field.value or "45")
        except ValueError:
            pass
        cfg.leaderboard_auto_refresh = bool(self.leaderboard_auto_refresh_switch.value)
        cfg.leaderboard_score_version = (
            self.leaderboard_score_version_dropdown.value or "v2"
        ).strip()
        cfg.fundamentals_source = (self.fundamentals_source_dropdown.value or "sec+yahoo").strip()
        cfg.sec_user_agent = (self.sec_user_agent_field.value or "").strip()
        cfg.fmp_api_key = (self.fmp_api_key_field.value or "").strip()
        cfg.ingest_mode_default = (self.ingest_mode_default_dropdown.value or "smart").strip()
        try:
            cfg.fundamentals_refresh_days = int(self.fundamentals_refresh_field.value or "30")
        except ValueError:
            pass
        cfg.force_full_history = bool(self.force_full_switch.value)
        cfg.auto_skip_dead_tickers = bool(self.auto_skip_dead_switch.value)
        cfg.retry_dead_tickers = bool(self.retry_dead_default_switch.value)
        cfg.ingest_use_parallel = bool(self.ingest_parallel_switch.value)
        try:
            cfg.ingest_worker_count = int(self.ingest_worker_count_field.value or "4")
            cfg.ingest_fund_worker_count = int(self.ingest_fund_worker_count_field.value or "2")
        except ValueError:
            pass
        try:
            cfg.yahoo_min_interval_sec = float(self.yahoo_interval_field.value or "0.75")
        except ValueError:
            pass
        cfg.ingest_fetch_profile = bool(self.ingest_profile_switch.value)
        cfg.ingest_fetch_insider = bool(self.ingest_insider_switch.value)
        cfg.ingest_fetch_news = bool(self.ingest_news_switch.value)
        cfg.ingest_use_stooq_backfill = bool(self.ingest_stooq_switch.value)
        show_snackbar(self.page_ref, "Data settings saved.", severity="success")

    def _save_intraday_settings(self, e) -> None:
        cfg = stock_config()
        cfg.finnhub_api_key = (self.finnhub_api_key_field.value or "").strip()
        cfg.intraday_interval = (self.intraday_interval_dropdown.value or "1Min").strip()
        try:
            cfg.intraday_backfill_days = int(self.intraday_backfill_days_field.value or "30")
        except ValueError:
            pass
        cfg.intraday_extended_hours = bool(self.intraday_extended_hours_switch.value)
        cfg.live_stream_symbols_source = (
            self.live_stream_source_dropdown.value or "focus"
        ).strip()
        show_snackbar(self.page_ref, "Intraday / live settings saved.", severity="success")

    def _save_backtest_settings(self, e) -> None:
        cfg = stock_config()
        try:
            cfg.canslim_stop_loss = float(self.canslim_stop_field.value)
            cfg.canslim_take_profit = float(self.canslim_tp_field.value)
            cfg.hybrid_market_trend_weeks = int(self.hybrid_weeks_field.value)
            cfg.canslim_require_pattern = bool(self.canslim_require_pattern_switch.value)
            cfg.canslim_rule_set_version = (self.canslim_rule_version_field.value or "oneil_v1").strip()
            cfg.backtest_slippage_bps = float(self.slippage_bps_field.value)
            cfg.backtest_spread_bps = float(self.spread_bps_field.value)
            cfg.backtest_fee_per_trade = float(self.fee_per_trade_field.value)
            cfg.daily_scan_enabled = bool(self.daily_scan_switch.value)
            cfg.daily_scan_ticker_limit = int(self.daily_scan_limit_field.value)
        except ValueError:
            show_snackbar(self.page_ref, "Invalid numeric value.", severity="error")
            return
        show_snackbar(self.page_ref, "Backtest defaults saved.", severity="success")

    def _apply_ai_settings_from_ui(self) -> None:
        cfg = stock_config()
        cfg.lm_studio_enabled = bool(self.lm_studio_switch.value)
        cfg.lm_studio_base_url = (self.lm_studio_url_field.value or "").strip()
        cfg.lm_studio_model = (self.lm_studio_model_field.value or "").strip()
        cfg.lm_studio_timeout_sec = float(self.lm_studio_timeout_field.value or 60)
        cfg.llm_backend_type = self.llm_backend_dropdown.value or "lmstudio"
        cfg.ollama_host = (self.ollama_host_field.value or "").strip()
        cfg.vllm_base_url = (self.vllm_url_field.value or "").strip()
        cfg.vllm_api_key = (self.vllm_key_field.value or "").strip()
        cfg.llm_chat_model = (self.llm_chat_model_field.value or "").strip()
        cfg.llm_temperature = float(self.llm_temperature_field.value or 0.7)
        cfg.llm_max_tokens = int(self.llm_max_tokens_field.value or 4096)
        cfg.llm_context_length = int(self.llm_context_length_field.value or 8192)
        cfg.llm_agents_enabled = bool(self.llm_agents_switch.value)
        cfg.llm_autonomy_level = self.llm_autonomy_dropdown.value or "confirm"
        cfg.llm_web_research_enabled = bool(self.llm_web_research_switch.value)
        cfg.llm_web_domain_allowlist = (self.llm_web_allowlist_field.value or "").strip()
        cfg.llm_web_max_pages = int(self.llm_web_max_pages_field.value or 3)
        cfg.llm_web_max_bytes = int(self.llm_web_max_bytes_field.value or 500000)

    def _save_ai_settings(self, e) -> None:
        try:
            self._apply_ai_settings_from_ui()
        except ValueError:
            show_snackbar(self.page_ref, "Invalid numeric value in AI settings.", severity="error")
            return
        from src.llm.manager import llm_manager
        from src.views.components.model_setup_bar import invalidate_backend_connection_cache

        llm_manager().reset()
        invalidate_backend_connection_cache()
        show_snackbar(self.page_ref, "AI settings saved.", severity="success")

    def _test_lm_studio(self, e) -> None:
        import threading

        if not self.page_ref:
            return

        try:
            self._apply_ai_settings_from_ui()
        except ValueError:
            show_snackbar(self.page_ref, "Invalid numeric value in AI settings.", severity="error")
            return

        self.lm_studio_test_btn.disabled = True
        self.ai_test_status.set_running("Testing backend connection…")
        try:
            self.lm_studio_test_btn.update()
            self.ai_test_status.update()
            self.page_ref.update()
        except RuntimeError:
            pass

        def _work():
            ok = False
            msg = ""
            sev = "error"
            try:
                from src.llm.manager import llm_manager
                from src.utils.logger_utils import app_logger

                ok, msg = llm_manager().test_connection()
                sev = "success" if ok else "error"
                app_logger.log(
                    "SETTINGS",
                    f"LLM connection test: {msg[:200]}",
                    level="INFO" if ok else "WARN",
                    ok=ok,
                )
            except Exception as ex:
                msg = f"Connection failed: {ex}"
                sev = "error"

            def _ui():
                from src.views.components.model_setup_bar import note_backend_connection

                note_backend_connection(ok)
                self.lm_studio_test_btn.disabled = False
                if ok:
                    self.ai_test_status.set_success(msg[:240])
                else:
                    self.ai_test_status.set_error(msg[:240])
                show_snackbar(self.page_ref, msg[:200], severity=sev, duration_ms=5000)
                try:
                    self.lm_studio_test_btn.update()
                    self.ai_test_status.update()
                    if self.page_ref:
                        self.page_ref.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _open_data_folder(self, e) -> None:
        folder = os.path.join(stock_config().project_root, "data")
        os.makedirs(folder, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", folder], check=False)
            else:
                subprocess.run(["xdg-open", folder], check=False)
        except Exception as ex:
            show_snackbar(self.page_ref, f"Could not open folder: {ex}", severity="error")

    def _on_theme_segment_change(self, e):
        selected = getattr(e.control, "selected", None) or []
        if not selected:
            return
        try:
            mode = next(iter(selected))
        except StopIteration:
            return
        app_theme().set_theme_mode(mode, self.page_ref)
        self._refresh_appearance_ui()

    # ------------------------------------------------------------------
    # Logging handlers
    # ------------------------------------------------------------------
    def _on_toggle_logging(self, e):
        if self.logging_switch.value:
            app_logger.enable_logging(level=self.log_level_dropdown.value or "INFO")
            settings_store().set_setting("logging_enabled", "1")
            show_snackbar(self.page_ref, "Logging enabled.", severity="success")
        else:
            app_logger.disable_logging()
            settings_store().set_setting("logging_enabled", "0")
            show_snackbar(self.page_ref, "Logging disabled.", severity="info")

    def _on_log_level_change(self, e):
        level = self.log_level_dropdown.value or "INFO"
        app_logger.set_level(level)
        settings_store().set_setting("log_level", level)

    def _open_logs_modal(self, e):
        self.logs_content.value = app_logger.get_latest_log_content()
        self.page_ref.open(self.logs_dlg)
        try:
            self.page_ref.update()
        except (AssertionError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Lifecycle / theme refresh
    # ------------------------------------------------------------------
    def _iter_section_containers(self):
        body = getattr(self, "_settings_body", None)
        if body is None:
            return
        for tile in body.controls:
            if not isinstance(tile, ft.ExpansionTile):
                continue
            for block in tile.controls or []:
                if not isinstance(block, ft.Column):
                    continue
                for ctrl in block.controls or []:
                    if isinstance(ctrl, ft.Container):
                        yield ctrl

    def _refresh_appearance_ui(self):
        current = app_theme().get_theme_mode()
        self.theme_segmented.selected = [current]
        muted = ThemeHelper.text_muted(self.page_ref)
        for ctrl in self._iter_section_containers():
            ctrl.bgcolor = ThemeHelper.surface_dim(self.page_ref)
            ctrl.border = ft.border.all(1, ThemeHelper.border_default(self.page_ref))
            if isinstance(ctrl.content, ft.Column):
                for child in ctrl.content.controls:
                    if isinstance(child, ft.Text) and getattr(child, "size", None) in (11, 12):
                        child.color = muted

    def refresh_theme(self) -> None:
        self._refresh_appearance_ui()
        self.ai_test_status.refresh_theme(self.page_ref)
        InputStyles.refresh_fields(
            self.page_ref, self._form_text_fields, self._form_dropdowns
        )
        try:
            self.update()
        except RuntimeError:
            pass
        if self.page_ref:
            try:
                self.page_ref.update()
            except RuntimeError:
                pass

    def refresh_data(self) -> None:
        cfg = stock_config()
        self.db_path_field.value = cfg.db_path
        self.csv_path_field.value = cfg.ticker_csv_path
        self.market_ticker_field.value = cfg.market_ticker
        self.parallel_switch.value = cfg.use_parallel
        self.worker_count_field.value = str(cfg.worker_count)
        self.sqlite_cache_field.value = str(cfg.sqlite_cache_mb)
        self.sqlite_mmap_field.value = str(cfg.sqlite_mmap_mb)
        self.dashboard_cache_field.value = str(cfg.dashboard_cache_sec)
        self.leaderboard_auto_refresh_switch.value = cfg.leaderboard_auto_refresh
        self.leaderboard_score_version_dropdown.value = cfg.leaderboard_score_version
        self.fundamentals_source_dropdown.value = cfg.fundamentals_source
        self.sec_user_agent_field.value = cfg.sec_user_agent
        self.ingest_mode_default_dropdown.value = cfg.ingest_mode_default
        self.fundamentals_refresh_field.value = str(cfg.fundamentals_refresh_days)
        self.force_full_switch.value = cfg.force_full_history
        self.auto_skip_dead_switch.value = cfg.auto_skip_dead_tickers
        self.retry_dead_default_switch.value = cfg.retry_dead_tickers
        self.ingest_parallel_switch.value = cfg.ingest_use_parallel
        self.ingest_worker_count_field.value = str(cfg.ingest_worker_count)
        self.ingest_fund_worker_count_field.value = str(cfg.ingest_fund_worker_count)
        self.yahoo_interval_field.value = str(cfg.yahoo_min_interval_sec)
        self.ingest_profile_switch.value = cfg.ingest_fetch_profile
        self.ingest_insider_switch.value = cfg.ingest_fetch_insider
        self.ingest_news_switch.value = cfg.ingest_fetch_news
        self.ingest_stooq_switch.value = cfg.ingest_use_stooq_backfill
        self.canslim_stop_field.value = str(cfg.canslim_stop_loss)
        self.canslim_tp_field.value = str(cfg.canslim_take_profit)
        self.canslim_require_pattern_switch.value = cfg.canslim_require_pattern
        self.canslim_rule_version_field.value = cfg.canslim_rule_set_version
        self.lm_studio_switch.value = cfg.lm_studio_enabled
        self.lm_studio_url_field.value = cfg.lm_studio_base_url
        self.lm_studio_model_field.value = cfg.lm_studio_model
        self.lm_studio_timeout_field.value = str(cfg.lm_studio_timeout_sec)
        self.hybrid_weeks_field.value = str(cfg.hybrid_market_trend_weeks)
        current = app_theme().get_theme_mode()
        self.theme_segmented.selected = [current]
        self.logging_switch.value = app_logger.enabled
        self.log_level_dropdown.value = app_logger.get_level()
