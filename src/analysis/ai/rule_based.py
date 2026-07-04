"""Offline keyword-based fallback when LM Studio is unavailable."""

from __future__ import annotations

import re

from src.analysis.ai.adapter import AIAdapter, AIAnalysisResult

_BULLISH = re.compile(
    r"\b(beat|surge|growth|upgrade|record|profit|strong|raised guidance)\b",
    re.I,
)
_BEARISH = re.compile(
    r"\b(miss|cut|downgrade|loss|weak|probe|investigation|recall|layoff)\b",
    re.I,
)
_RISK = re.compile(
    r"\b(sec|lawsuit|fraud|delist|bankruptcy|warning|subpoena)\b",
    re.I,
)


class RuleBasedAdapter(AIAdapter):
    def analyze_text(self, text: str, *, ticker: str = "") -> AIAnalysisResult:
        bull = len(_BULLISH.findall(text))
        bear = len(_BEARISH.findall(text))
        risks = _RISK.findall(text)
        if bull > bear + 1:
            sentiment = "positive"
        elif bear > bull + 1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        risk_tags = list(dict.fromkeys(risks))[:5]
        flags = []
        if "sec" in text.lower():
            flags.append("regulatory")
        if bear > 2:
            flags.append("negative_headlines")
        summary = text[:200] + ("…" if len(text) > 200 else "")
        return AIAnalysisResult(
            sentiment=sentiment,
            risk_tags=risk_tags,
            summary=summary,
            thesis_flags=flags,
            confidence=0.45,
            provider="rule_based",
        )
