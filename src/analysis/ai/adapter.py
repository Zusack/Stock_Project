"""Provider-agnostic AI adapter for news and research text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.services.stock_config import stock_config


@dataclass
class AIAnalysisResult:
    sentiment: str
    risk_tags: list[str]
    summary: str
    thesis_flags: list[str]
    confidence: float
    provider: str
    catalyst_type: str = ""


class AIAdapter(ABC):
    @abstractmethod
    def analyze_text(self, text: str, *, ticker: str = "") -> AIAnalysisResult:
        raise NotImplementedError


def get_ai_adapter() -> AIAdapter:
    """Return LLM adapter when enabled, else rule-based fallback."""
    cfg = stock_config()
    if cfg.lm_studio_enabled:
        model = cfg.llm_chat_model or cfg.lm_studio_model
        try:
            from src.analysis.ai.sdk_adapter import SDKLMStudioAdapter

            return SDKLMStudioAdapter(model=model)
        except Exception:
            from src.analysis.ai.lmstudio_client import LMStudioAdapter

            return LMStudioAdapter(
                base_url=cfg.lm_studio_base_url,
                model=cfg.lm_studio_model,
                timeout_sec=cfg.lm_studio_timeout_sec,
            )
    from src.analysis.ai.rule_based import RuleBasedAdapter

    return RuleBasedAdapter()


def analyze_headline(title: str, summary: str = "", *, ticker: str = "") -> AIAnalysisResult:
    text = f"{title}\n{summary}".strip()
    return get_ai_adapter().analyze_text(text, ticker=ticker)
