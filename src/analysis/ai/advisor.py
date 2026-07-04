"""Signal advisor seam — rule-based now, LM Studio-ready."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

from src.services.stock_config import stock_config

if TYPE_CHECKING:
    from src.analysis.guidance import GuidanceRow
    from src.analysis.leaderboard import LeaderboardRow


@dataclass
class SignalSuggestion:
    ticker: str
    headline: str
    detail: str
    severity: str = "info"  # info | watch | action | risk
    band: str = ""
    drivers: list[str] = field(default_factory=list)
    provider: str = "rule_based"


class SignalAdvisor(ABC):
    @abstractmethod
    def analyze_ticker(
        self,
        ticker: str,
        *,
        leaderboard_row: LeaderboardRow | None = None,
        guidance_row: GuidanceRow | None = None,
        db_path: str | None = None,
        canslim_score: int | None = None,
        canslim_max: int | None = None,
    ) -> list[SignalSuggestion]:
        ...


def resolve_advisor_rows(
    ticker: str,
    *,
    leaderboard_row: "LeaderboardRow | None" = None,
    guidance_row: "GuidanceRow | None" = None,
    db_path: str | None = None,
    canslim_score: int | None = None,
    canslim_max: int | None = None,
) -> tuple["LeaderboardRow | None", "GuidanceRow | None"]:
    """Load leaderboard/guidance rows when callers pass ticker only.

    Optional ``canslim_score`` / ``canslim_max`` align guidance text with an
    interactive CANSLIM analysis (7 letters) instead of the leaderboard's 6.
    """
    from src.analysis.ai.context import load_guidance_for_ticker
    from src.analysis.guidance import score_guidance_row
    from src.analysis.leaderboard import score_ticker

    cfg = stock_config()
    db = db_path or cfg.db_path
    sym = str(ticker).strip().upper()

    if leaderboard_row is None:
        leaderboard_row = score_ticker(sym, db, cfg.market_ticker)
    if guidance_row is None and leaderboard_row is not None:
        guidance_row = score_guidance_row(
            leaderboard_row,
            db,
            canslim_score=canslim_score,
            canslim_max=canslim_max,
        )
    elif guidance_row is None:
        guidance_row = load_guidance_for_ticker(db, sym)
    return leaderboard_row, guidance_row


class RuleBasedSignalAdvisor(SignalAdvisor):
    """Wraps existing guidance / leaderboard logic into user-facing suggestions."""

    def analyze_ticker(
        self,
        ticker: str,
        *,
        leaderboard_row: LeaderboardRow | None = None,
        guidance_row: GuidanceRow | None = None,
        db_path: str | None = None,
        canslim_score: int | None = None,
        canslim_max: int | None = None,
    ) -> list[SignalSuggestion]:
        sym = str(ticker).strip().upper()
        leaderboard_row, guidance_row = resolve_advisor_rows(
            sym,
            leaderboard_row=leaderboard_row,
            guidance_row=guidance_row,
            db_path=db_path,
            canslim_score=canslim_score,
            canslim_max=canslim_max,
        )
        suggestions: list[SignalSuggestion] = []

        if guidance_row is None:
            return [
                SignalSuggestion(
                    ticker=sym,
                    headline="No guidance data",
                    detail="Run a daily scan or refresh the Leaderboard to generate signals.",
                    severity="info",
                    provider="rule_based",
                )
            ]

        from src.analysis.guidance import RecommendationBand

        band = guidance_row.recommendation_band
        drivers = list(guidance_row.positive_drivers or []) + list(guidance_row.negative_drivers or [])

        severity_map = {
            RecommendationBand.HIGH_PRIORITY.value: "action",
            RecommendationBand.CANDIDATE.value: "watch",
            RecommendationBand.EXIT_WATCH.value: "risk",
            RecommendationBand.RISK_REVIEW.value: "risk",
            RecommendationBand.MONITOR.value: "info",
        }
        severity = severity_map.get(band, "info")

        headline = band.replace("_", " ").title()
        detail_parts = []
        if guidance_row.positive_drivers:
            detail_parts.append("Strengths: " + "; ".join(guidance_row.positive_drivers[:3]))
        if guidance_row.negative_drivers:
            detail_parts.append("Concerns: " + "; ".join(guidance_row.negative_drivers[:3]))
        if guidance_row.risk_notes:
            detail_parts.append(guidance_row.risk_notes)

        suggestions.append(
            SignalSuggestion(
                ticker=sym,
                headline=headline,
                detail=" · ".join(detail_parts) if detail_parts else f"Composite {guidance_row.composite_score:.0f}",
                severity=severity,
                band=band,
                drivers=drivers[:6],
                provider="rule_based",
            )
        )

        if leaderboard_row:
            if getattr(leaderboard_row, "pass_setup", False) and getattr(leaderboard_row, "pass_pattern", False):
                suggestions.append(
                    SignalSuggestion(
                        ticker=sym,
                        headline="Setup + pattern aligned",
                        detail="CANSLIM setup and cup-with-handle pattern both pass.",
                        severity="watch",
                        provider="rule_based",
                    )
                )
            risk = getattr(leaderboard_row, "risk_flag", "") or ""
            if risk:
                suggestions.append(
                    SignalSuggestion(
                        ticker=sym,
                        headline="Risk flag",
                        detail=str(risk),
                        severity="risk",
                        provider="rule_based",
                    )
                )

        return suggestions


class LMStudioSignalAdvisor(SignalAdvisor):
    """Local LLM narrative advisor using LLMManager (same path as Assistant chat)."""

    def __init__(self) -> None:
        self._fallback = RuleBasedSignalAdvisor()

    def analyze_ticker(
        self,
        ticker: str,
        *,
        leaderboard_row: LeaderboardRow | None = None,
        guidance_row: GuidanceRow | None = None,
        db_path: str | None = None,
        canslim_score: int | None = None,
        canslim_max: int | None = None,
    ) -> list[SignalSuggestion]:
        sym = str(ticker).strip().upper()
        cfg = stock_config()
        db = db_path or cfg.db_path
        leaderboard_row, guidance_row = resolve_advisor_rows(
            sym,
            leaderboard_row=leaderboard_row,
            guidance_row=guidance_row,
            db_path=db,
            canslim_score=canslim_score,
            canslim_max=canslim_max,
        )

        base = self._fallback.analyze_ticker(
            sym,
            leaderboard_row=leaderboard_row,
            guidance_row=guidance_row,
            db_path=db,
            canslim_score=canslim_score,
            canslim_max=canslim_max,
        )

        if guidance_row is None and leaderboard_row is None:
            return base

        if not cfg.lm_studio_enabled:
            return base

        try:
            from src.analysis.ai.context import assemble_ticker_context
            from src.analysis.ai.narrative import parse_narrative_json
            from src.analysis.intelligence_schema import save_ai_insight
            from src.llm.manager import llm_manager
            from src.utils.logger_utils import app_logger

            mgr = llm_manager()
            ok, model_or_err = mgr.ensure_model_loaded()
            if not ok:
                app_logger.log(
                    "ADVISOR",
                    f"AI insight skipped for {sym}: {model_or_err}",
                    level="WARN",
                    ticker=sym,
                )
                hint = SignalSuggestion(
                    ticker=sym,
                    headline="AI insight unavailable",
                    detail=model_or_err,
                    severity="info",
                    provider="llm_sdk",
                )
                return [hint] + base

            model_id = model_or_err
            context = assemble_ticker_context(
                sym,
                db,
                canslim_score=canslim_score,
                canslim_max=canslim_max,
            )
            prompt = (
                f"You are a stock research assistant. Analyze ticker {sym} using the JSON context below.\n"
                "Reply with JSON only (no markdown fences):\n"
                '{"headline":"short title", "detail":"2-4 sentences", '
                '"severity":"info|watch|action|risk", "drivers":["reason1","reason2"]}\n\n'
                f"{json.dumps(context, default=str)[:6000]}"
            )
            raw = mgr.complete(prompt, model=model_id)
            data = parse_narrative_json(raw)
            if not data:
                app_logger.log(
                    "ADVISOR",
                    f"AI narrative parse failed for {sym}.",
                    level="WARN",
                    ticker=sym,
                    raw_preview=(raw or "")[:200],
                )
                fallback_detail = (raw or "").strip()[:800] or "Model returned an empty response."
                suggestion = SignalSuggestion(
                    ticker=sym,
                    headline="AI insight",
                    detail=fallback_detail,
                    severity="info",
                    band=guidance_row.recommendation_band if guidance_row else "",
                    provider="llm_sdk",
                )
            else:
                suggestion = SignalSuggestion(
                    ticker=sym,
                    headline=str(data.get("headline", "AI insight")),
                    detail=str(data.get("detail", raw or ""))[:800],
                    severity=str(data.get("severity", "info")),
                    band=guidance_row.recommendation_band if guidance_row else "",
                    drivers=[str(d) for d in (data.get("drivers") or [])][:6],
                    provider="llm_sdk",
                )

            save_ai_insight(
                db,
                ticker=sym,
                provider="llm_sdk",
                model=model_id,
                kind="ticker_narrative",
                payload={"suggestion": suggestion.__dict__, "summary": raw},
                source_context_hash=context.get("context_hash", ""),
            )
            app_logger.log(
                "ADVISOR",
                f"AI insight generated for {sym}.",
                level="INFO",
                ticker=sym,
                model=model_id,
            )
            if base and base[0].headline != "No guidance data":
                return [suggestion] + base[1:]
            return [suggestion] + base
        except Exception as ex:
            from src.utils.logger_utils import app_logger

            app_logger.log(
                "ADVISOR",
                f"AI insight failed for {sym}: {ex}",
                level="ERROR",
                ticker=sym,
            )
            hint = SignalSuggestion(
                ticker=sym,
                headline="AI insight unavailable",
                detail=str(ex)[:500],
                severity="info",
                provider="llm_sdk",
            )
            return [hint] + base


def _parse_narrative_json(text: str) -> dict | None:
    from src.analysis.ai.narrative import parse_narrative_json

    return parse_narrative_json(text)


def get_signal_advisor() -> SignalAdvisor:
    """Return signal advisor — LM Studio variant when enabled, else rule-based."""
    cfg = stock_config()
    if cfg.lm_studio_enabled:
        return LMStudioSignalAdvisor()
    return RuleBasedSignalAdvisor()
