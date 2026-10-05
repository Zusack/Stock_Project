"""Domain and web tools for LLM agents."""

from __future__ import annotations

import json
from typing import Any, Callable

from src.analysis.intelligence_schema import save_ai_insight
from src.analysis.ticker_evaluation import score_ticker_with_cache
from src.analysis.llm_store import (
    list_open_questions,
    record_finding,
    record_open_question,
    resolve_question,
)
from src.analysis.news_sources.aggregator import fetch_headlines_for_ticker
from src.analysis.quote_snapshot import get_quote, quote_to_dict
from src.llm.research import get_web_research
from src.services.stock_config import stock_config


def get_quote_tool(ticker: str) -> str:
    """Get the latest quote snapshot for a stock ticker symbol."""
    cfg = stock_config()
    q = get_quote(cfg.db_path, ticker.upper())
    if q is None:
        return json.dumps({"error": f"No quote for {ticker.upper()}"})
    return json.dumps(quote_to_dict(q), default=str)


def fetch_news(ticker: str, limit: int = 10) -> str:
    """Fetch recent news headlines for a stock ticker."""
    headlines = fetch_headlines_for_ticker(ticker.upper(), yahoo_limit=limit)
    items = [
        {
            "title": h.title,
            "date": h.date,
            "publisher": h.publisher,
            "link": h.link,
            "summary": h.summary,
        }
        for h in headlines[:limit]
    ]
    return json.dumps({"ticker": ticker.upper(), "headlines": items}, default=str)


def get_canslim(ticker: str) -> str:
    """Get CANSLM / leaderboard scoring metrics for a ticker."""
    from src.analysis.canslim_core import CANSLM_RULE_COUNT

    cfg = stock_config()
    row = score_ticker_with_cache(ticker.upper(), cfg.db_path, cfg.market_ticker)
    if row is None:
        return json.dumps({"error": f"No CANSLM data for {ticker.upper()}"})
    data = {
        "ticker": row.ticker,
        "composite_score": row.composite_score,
        "canslim_score": row.canslim_score,
        "canslim_max": CANSLM_RULE_COUNT,
        "canslm_scale": (
            f"Leaderboard score is n/{CANSLM_RULE_COUNT} on letters C, A, N, S, L, M. "
            "IBD institutional sponsorship (I) is not scored (proprietary data)."
        ),
        "canslm_lines": list(getattr(row, "canslm_lines", []) or []),
        "pattern_quality": row.pattern_quality,
        "rs_pct": row.rs_pct,
        "volume_ratio": row.volume_ratio,
        "near_high_pct": row.near_high_pct,
        "pass_setup": row.pass_setup,
        "pass_pattern": row.pass_pattern,
        "risk_flag": row.risk_flag,
        "latest_price": row.latest_price,
        "sector": row.sector,
        "industry": row.industry,
        "news_sentiment": row.news_sentiment,
        "notes": row.notes,
    }
    return json.dumps(data, default=str)


def get_leaderboard_rank(ticker: str) -> str:
    """Get leaderboard rank and composite score for a ticker."""
    cfg = stock_config()
    with __import__("src.analysis.db", fromlist=["db_connection"]).db_connection(
        cfg.db_path, readonly=True
    ) as conn:
        row = conn.execute(
            """
            SELECT ticker, composite_score, canslim_score, rs_pct, risk_flag, notes
            FROM leaderboard_snapshots
            WHERE ticker = ? AND run_id = 'current'
            """,
            (ticker.upper(),),
        ).fetchone()
    if not row:
        return json.dumps({"error": f"{ticker.upper()} not in current leaderboard"})
    return json.dumps(
        {
            "ticker": row[0],
            "composite_score": row[1],
            "canslim_score": row[2],
            "rs_pct": row[3],
            "risk_flag": row[4],
            "notes": row[5],
        },
        default=str,
    )


def get_financials(ticker: str) -> str:
    """Get fundamental profile data for a ticker from the database."""
    from src.analysis.db import get_profile

    cfg = stock_config()
    profile = get_profile(cfg.db_path, ticker.upper())
    if not profile:
        return json.dumps({"error": f"No fundamentals for {ticker.upper()}"})
    return json.dumps(profile, default=str)


def web_fetch(url: str) -> str:
    """Fetch and extract text from an allowed financial news URL."""
    result = get_web_research().fetch_url(url)
    return json.dumps(result, default=str)


def web_search_lite(query: str) -> str:
    """Search the web for financial information (allowed domains only)."""
    result = get_web_research().search_lite(query)
    return json.dumps(result, default=str)


def save_analysis(ticker: str, kind: str, payload_json: str) -> str:
    """Save an LLM analysis insight to the database. payload_json must be valid JSON."""
    cfg = stock_config()
    try:
        payload = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
    except json.JSONDecodeError:
        payload = {"summary": str(payload_json)}
    save_ai_insight(
        cfg.db_path,
        ticker=ticker.upper(),
        provider=cfg.llm_backend_type,
        model=cfg.llm_chat_model or "local",
        kind=kind,
        payload=payload if isinstance(payload, dict) else {"text": str(payload)},
    )
    return json.dumps({"saved": True, "ticker": ticker.upper(), "kind": kind})


def record_open_question_tool(ticker: str, question: str, source_ref: str = "") -> str:
    """Record an open research question about a ticker."""
    cfg = stock_config()
    qid = record_open_question(
        cfg.db_path,
        ticker=ticker.upper(),
        question=question,
        source_ref=source_ref,
    )
    return json.dumps({"question_id": qid, "ticker": ticker.upper(), "question": question})


def list_open_questions_tool(ticker: str = "", status: str = "open") -> str:
    """List open research questions, optionally filtered by ticker and status."""
    cfg = stock_config()
    items = list_open_questions(
        cfg.db_path,
        ticker=ticker.upper() if ticker else None,
        status=status or None,
    )
    return json.dumps({"questions": items}, default=str)


def resolve_question_tool(question_id: int) -> str:
    """Mark an open question as resolved."""
    cfg = stock_config()
    resolve_question(cfg.db_path, question_id)
    return json.dumps({"resolved": True, "question_id": question_id})


def record_finding_tool(
    ticker: str,
    url: str,
    snippet: str,
    answer: str,
    question_id: int = 0,
) -> str:
    """Record a research finding for a ticker and optional question."""
    cfg = stock_config()
    fid = record_finding(
        cfg.db_path,
        question_id=question_id or None,
        ticker=ticker.upper(),
        url=url,
        snippet=snippet,
        answer=answer,
    )
    return json.dumps({"finding_id": fid, "ticker": ticker.upper()})


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "get_quote": get_quote_tool,
    "fetch_news": fetch_news,
    "get_canslim": get_canslim,
    "get_leaderboard_rank": get_leaderboard_rank,
    "get_financials": get_financials,
    "web_fetch": web_fetch,
    "web_search_lite": web_search_lite,
    "save_analysis": save_analysis,
    "record_open_question": record_open_question_tool,
    "list_open_questions": list_open_questions_tool,
    "resolve_question": resolve_question_tool,
    "record_finding": record_finding_tool,
}


def resolve_tools(names: list[str]) -> list[Callable]:
    """Return callables for the given tool names."""
    out: list[Callable] = []
    for name in names:
        fn = TOOL_REGISTRY.get(name)
        if fn is not None:
            out.append(fn)
    return out


def all_tool_names() -> list[str]:
    return sorted(TOOL_REGISTRY.keys())
