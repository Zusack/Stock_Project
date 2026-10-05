"""AI advisor for backtest results — LM Studio with rule-based fallback."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.analysis.ai.advisor import SignalSuggestion
from src.analysis.ai.narrative import extract_json_object
from src.analysis.backtest_engine import BacktestResult
from src.analysis.intelligence_schema import load_latest_ai_insight, save_ai_insight
from src.analysis.strategy_spec import spec_to_plain_english
from src.llm.manager import llm_manager
from src.services.stock_config import stock_config


def _context_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def assemble_backtest_context(result: BacktestResult) -> dict[str, Any]:
    m = result.metrics
    top_wins = sorted(result.trades, key=lambda t: t.return_pct, reverse=True)[:3]
    top_losses = sorted(result.trades, key=lambda t: t.return_pct)[:3]
    exit_reasons: dict[str, int] = {}
    for t in result.trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    return {
        "strategy": spec_to_plain_english(result.spec),
        "period": result.period_label,
        "metrics": m.to_dict(),
        "top_winning_trades": [t.to_dict() for t in top_wins],
        "top_losing_trades": [t.to_dict() for t in top_losses],
        "exit_reason_counts": exit_reasons,
        "ticker_count": len(result.tickers_run),
    }


def rule_based_backtest_insights(result: BacktestResult) -> list[SignalSuggestion]:
    """Heuristic narrative when LLM is disabled."""
    m = result.metrics
    suggestions: list[SignalSuggestion] = []
    suggestions.append(
        SignalSuggestion(
            ticker="BACKTEST",
            headline=f"{result.spec.name}: {m.total_return_pct:.1f}% total return",
            detail=(
                f"Over {m.period_days} days with {m.trade_count} trades. "
                f"Sharpe {m.sharpe:.2f}, max drawdown {m.max_drawdown_pct:.1f}%, "
                f"win rate {m.win_rate_pct:.1f}%."
            ),
            severity="info" if m.total_return_pct >= 0 else "watch",
            provider="rule_based",
        )
    )
    if m.sharpe < 1.0 and m.period_days > 60:
        suggestions.append(
            SignalSuggestion(
                ticker="BACKTEST",
                headline="Risk-adjusted returns are modest",
                detail=METRIC_HINT_SHARPE,
                severity="watch",
                provider="rule_based",
            )
        )
    if m.max_drawdown_pct > 20:
        suggestions.append(
            SignalSuggestion(
                ticker="BACKTEST",
                headline="Significant drawdown observed",
                detail=(
                    f"Peak-to-trough decline reached {m.max_drawdown_pct:.1f}%. "
                    "Consider tighter stop losses or a market filter."
                ),
                severity="risk",
                provider="rule_based",
            )
        )
    exit_counts: dict[str, int] = {}
    for t in result.trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    if exit_counts:
        dominant = max(exit_counts, key=exit_counts.get)
        suggestions.append(
            SignalSuggestion(
                ticker="BACKTEST",
                headline=f"Most exits via: {dominant}",
                detail=f"Exit reason breakdown: {exit_counts}",
                severity="info",
                provider="rule_based",
            )
        )
    for hint in m.interpretation_hints[:2]:
        suggestions.append(
            SignalSuggestion(
                ticker="BACKTEST",
                headline="Interpretation",
                detail=hint,
                severity="info",
                provider="rule_based",
            )
        )
    return suggestions


METRIC_HINT_SHARPE = (
    "Sharpe below 1.0 suggests returns did not fully compensate for volatility. "
    "Try a different period or tighter risk management."
)


def analyze_backtest_results(
    result: BacktestResult,
    *,
    db_path: str | None = None,
) -> list[SignalSuggestion]:
    cfg = stock_config()
    db = db_path or cfg.db_path
    context = assemble_backtest_context(result)
    ctx_hash = _context_hash(context)

    if cfg.lm_studio_enabled:
        cached = load_latest_ai_insight(db, "BACKTEST", kind="backtest_narrative")
        if cached and cached.get("source_context_hash") == ctx_hash:
            payload = cached.get("payload") or {}
            if payload.get("suggestions"):
                return [_suggestion_from_dict(s) for s in payload["suggestions"]]

        mgr = llm_manager()
        ok, model_or_err = mgr.ensure_model_loaded()
        if ok:
            prompt = (
                "You are a trading strategy analyst helping a non-professional investor "
                "understand backtest results. Use the JSON context below.\n"
                "Reply with JSON only (no markdown):\n"
                '{"suggestions":[{"headline":"...","detail":"...","severity":"info|watch|action|risk"}]}\n\n'
                f"{json.dumps(context, default=str)[:5000]}"
            )
            try:
                raw = mgr.complete(prompt, model=model_or_err)
                data = extract_json_object(raw) or {}
                items = data.get("suggestions") or []
                suggestions = [_suggestion_from_dict(s) for s in items if isinstance(s, dict)]
                if suggestions:
                    save_ai_insight(
                        db,
                        ticker="BACKTEST",
                        provider="llm_sdk",
                        model=model_or_err,
                        kind="backtest_narrative",
                        payload={"suggestions": [s.__dict__ for s in suggestions]},
                        source_context_hash=ctx_hash,
                    )
                    return suggestions
            except Exception:
                pass

    return rule_based_backtest_insights(result)


def _suggestion_from_dict(data: dict) -> SignalSuggestion:
    return SignalSuggestion(
        ticker="BACKTEST",
        headline=str(data.get("headline") or "Insight"),
        detail=str(data.get("detail") or ""),
        severity=str(data.get("severity") or "info"),
        provider="llm_sdk",
    )


def draft_strategy_from_nl(description: str) -> tuple[dict | None, str | None]:
    """Use LLM to draft StrategySpec JSON from natural language."""
    cfg = stock_config()
    if not cfg.lm_studio_enabled:
        return None, "Enable Local AI in Settings to draft strategies from text."
    mgr = llm_manager()
    ok, model_or_err = mgr.ensure_model_loaded()
    if not ok:
        return None, model_or_err
    from src.views.strategy_constants import LLM_STRATEGY_SCHEMA_HINT

    prompt = (
        "Convert the user's trading strategy description into a StrategySpec JSON object.\n"
        f"{LLM_STRATEGY_SCHEMA_HINT}\n\n"
        f"User description: {description}"
    )
    try:
        raw = mgr.complete(prompt, model=model_or_err)
        data = extract_json_object(raw)
        if not data:
            return None, "Model did not return valid JSON. Try rephrasing your strategy."
        return data, None
    except Exception as ex:
        return None, str(ex)


def backtest_assistant_prefill(result: BacktestResult) -> str:
    """Context string for Assistant tab handoff."""
    ctx = assemble_backtest_context(result)
    return (
        "Please help me understand this backtest result:\n"
        f"{json.dumps(ctx, indent=2, default=str)[:4000]}"
    )
