"""Reusable Settings tab section builders."""

from __future__ import annotations

import flet as ft

from src.views.components.layouts import SectionHeader
from src.views.theme import ThemeHelper


def build_ai_settings_panel(
    page: ft.Page,
    *,
    lm_studio_switch: ft.Control,
    lm_studio_url_field: ft.Control,
    lm_studio_model_field: ft.Control,
    lm_studio_timeout_field: ft.Control,
    llm_backend_dropdown: ft.Control,
    ollama_host_field: ft.Control,
    vllm_url_field: ft.Control,
    vllm_key_field: ft.Control,
    llm_chat_model_field: ft.Control,
    llm_temperature_field: ft.Control,
    llm_max_tokens_field: ft.Control,
    llm_context_length_field: ft.Control,
    llm_agents_switch: ft.Control,
    llm_autonomy_dropdown: ft.Control,
    llm_web_research_switch: ft.Control,
    llm_web_allowlist_field: ft.Control,
    llm_web_max_pages_field: ft.Control,
    llm_web_max_bytes_field: ft.Control,
    lm_studio_test_btn: ft.Control,
    ai_test_status: ft.Control,
    save_ai_btn: ft.Control,
    themed_panel,
) -> list[ft.Control]:
    """Dedicated local AI / LLM configuration section."""
    return [
        SectionHeader("Local AI", icon=ft.Icons.PSYCHOLOGY, page_ref=page),
        themed_panel(
            ft.Column(
                [
                    ft.Text(
                        "Configure local LLM backends (LM Studio primary) for the Assistant tab, "
                        "headline analysis, and tool-calling agents.",
                        size=12,
                        color=ThemeHelper.text_muted(page),
                    ),
                    ft.Row([lm_studio_switch, llm_backend_dropdown], spacing=12, wrap=True),
                    lm_studio_url_field,
                    ft.Row([ollama_host_field, vllm_url_field, vllm_key_field], spacing=12, wrap=True),
                    ft.Row(
                        [
                            lm_studio_model_field,
                            llm_chat_model_field,
                            lm_studio_timeout_field,
                        ],
                        spacing=12,
                        wrap=True,
                    ),
                    ft.Row(
                        [
                            llm_temperature_field,
                            llm_max_tokens_field,
                            llm_context_length_field,
                        ],
                        spacing=12,
                        wrap=True,
                    ),
                    ft.Text(
                        "Inference settings below are sent with each app request (Assistant, agents). "
                        "They apply when this app loads or calls a model — they do not change LM Studio's "
                        "global UI defaults for manually loaded models.",
                        size=11,
                        italic=True,
                        color=ThemeHelper.text_muted(page),
                    ),
                    ft.Row([llm_agents_switch, llm_autonomy_dropdown], spacing=12, wrap=True),
                    ft.Text("Web research", weight=ft.FontWeight.W_600, size=13),
                    ft.Row([llm_web_research_switch], spacing=12),
                    llm_web_allowlist_field,
                    ft.Row([llm_web_max_pages_field, llm_web_max_bytes_field], spacing=12, wrap=True),
                    ft.Row([lm_studio_test_btn, save_ai_btn], spacing=12),
                    ai_test_status,
                ],
                spacing=10,
                tight=True,
            )
        ),
    ]


def build_backtest_settings_panel(
    page: ft.Page,
    *,
    canslim_stop_field: ft.Control,
    canslim_tp_field: ft.Control,
    canslim_rule_version_field: ft.Control,
    hybrid_weeks_field: ft.Control,
    canslim_require_pattern_switch: ft.Control,
    slippage_bps_field: ft.Control,
    spread_bps_field: ft.Control,
    fee_per_trade_field: ft.Control,
    daily_scan_switch: ft.Control,
    daily_scan_limit_field: ft.Control,
    save_backtest_btn: ft.Control,
    themed_panel,
) -> list[ft.Control]:
    """Backtest defaults without AI settings."""
    return [
        SectionHeader("Backtest defaults", icon=ft.Icons.TIMELINE, page_ref=page),
        themed_panel(
            ft.Column(
                [
                    ft.Row(
                        [
                            canslim_stop_field,
                            canslim_tp_field,
                            canslim_rule_version_field,
                            hybrid_weeks_field,
                        ],
                        wrap=True,
                        spacing=12,
                    ),
                    canslim_require_pattern_switch,
                    ft.Row(
                        [slippage_bps_field, spread_bps_field, fee_per_trade_field],
                        wrap=True,
                        spacing=12,
                    ),
                    ft.Row([daily_scan_switch, daily_scan_limit_field], spacing=12),
                    save_backtest_btn,
                ],
                spacing=10,
                tight=True,
            )
        ),
    ]
