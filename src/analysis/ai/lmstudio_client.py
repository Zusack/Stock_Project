"""LM Studio OpenAI-compatible local inference."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

from src.analysis.ai.adapter import AIAdapter, AIAnalysisResult
from src.analysis.ai.rule_based import RuleBasedAdapter

_DEFAULT_BASE = "http://localhost:1234/v1"
_MAX_RETRIES = 2
_RETRY_BACKOFF_SEC = 0.75


class LMStudioAdapter(AIAdapter):
    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE,
        model: str = "",
        timeout_sec: float = 60.0,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_retries = max(0, int(max_retries))
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
            content = self._chat_completion(prompt)
            return self._parse_response(content, ticker)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            result = self._fallback.analyze_text(text, ticker=ticker)
            return AIAnalysisResult(
                sentiment=result.sentiment,
                risk_tags=result.risk_tags,
                summary=result.summary,
                thesis_flags=result.thesis_flags + ["lm_studio_unavailable"],
                confidence=result.confidence * 0.8,
                provider="rule_based_fallback",
                catalyst_type=result.catalyst_type,
            )

    def stream_text(self, text: str, *, ticker: str = "") -> Iterator[str]:
        """Yield streamed token chunks when the server supports SSE; else full response."""
        payload = {
            "model": self.model or "local-model",
            "messages": [{"role": "user", "content": text[:6000]}],
            "temperature": 0.2,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (KeyError, json.JSONDecodeError):
                        continue
        except Exception:
            yield self.analyze_text(text, ticker=ticker).summary

    def _chat_completion(self, prompt: str) -> str:
        payload = {
            "model": self.model or "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode())
                return str(body["choices"][0]["message"]["content"])
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(_RETRY_BACKOFF_SEC * (attempt + 1))
        raise last_error or RuntimeError("LM Studio request failed")

    def _parse_response(self, content: str, ticker: str) -> AIAnalysisResult:
        data = _extract_json_object(content)
        if not data:
            return self._fallback.analyze_text(content, ticker=ticker)
        if not _validate_analysis_json(data):
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
            provider="lm_studio",
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
