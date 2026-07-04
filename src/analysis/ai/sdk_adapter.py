"""LLM SDK-backed adapter for headline analysis."""

from __future__ import annotations

import json
import re

from src.analysis.ai.adapter import AIAdapter, AIAnalysisResult
from src.analysis.ai.rule_based import RuleBasedAdapter
from src.llm.manager import llm_manager


class SDKLMStudioAdapter(AIAdapter):
    """Uses LLMManager (port SDK backend) for structured headline analysis."""

    def __init__(self, *, model: str = "") -> None:
        self.model = model
        self._fallback = RuleBasedAdapter()

    def analyze_text(self, text: str, *, ticker: str = "") -> AIAnalysisResult:
        prompt = (
            f"Analyze this stock headline for {ticker or 'unknown ticker'}. "
            "Reply with JSON only: "
            '{"sentiment":"positive|neutral|negative","risk_tags":[],"summary":"...",'
            '"thesis_flags":[],"catalyst_type":"earnings|product|macro|legal|other|none",'
            '"confidence":0.0}\n\n'
            f"{text[:2000]}"
        )
        try:
            mgr = llm_manager()
            content = mgr.complete(prompt, model=self.model or None)
            return self._parse_response(content, ticker)
        except Exception:
            result = self._fallback.analyze_text(text, ticker=ticker)
            return AIAnalysisResult(
                sentiment=result.sentiment,
                risk_tags=result.risk_tags,
                summary=result.summary,
                thesis_flags=result.thesis_flags + ["llm_sdk_unavailable"],
                confidence=result.confidence * 0.8,
                provider="rule_based_fallback",
                catalyst_type=result.catalyst_type,
            )

    def _parse_response(self, content: str, ticker: str) -> AIAnalysisResult:
        data = _extract_json_object(content)
        if not data or not _validate_analysis_json(data):
            return self._fallback.analyze_text(content, ticker=ticker)
        catalyst = str(data.get("catalyst_type", "") or "").strip().lower()
        thesis_flags = list(data.get("thesis_flags") or [])
        if catalyst and catalyst != "none" and catalyst not in thesis_flags:
            thesis_flags.append(catalyst)
        return AIAnalysisResult(
            sentiment=str(data.get("sentiment", "neutral")),
            risk_tags=[str(t) for t in (data.get("risk_tags") or [])],
            summary=str(data.get("summary", ""))[:500],
            thesis_flags=thesis_flags,
            confidence=float(data.get("confidence", 0.7)),
            provider="llm_sdk",
            catalyst_type=catalyst,
        )


def _extract_json_object(content: str) -> dict | None:
    match = re.search(r"\{.*\}", content or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _validate_analysis_json(data: dict) -> bool:
    sentiment = str(data.get("sentiment", "")).lower()
    if sentiment not in {"positive", "neutral", "negative", "bullish", "bearish"}:
        return False
    try:
        conf = float(data.get("confidence", 0.7))
    except (TypeError, ValueError):
        return False
    return 0.0 <= conf <= 1.0
