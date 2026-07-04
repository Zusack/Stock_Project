"""News sentiment scoring from stored headlines."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable

from src.analysis.ai.adapter import analyze_headline
from src.analysis.db import db_connection
from src.analysis.db_perf import execute_with_retry

# Max deviation from neutral when LLM confidence is low (bounded contribution).
MAX_SENTIMENT_DELTA = 0.15
MIN_LLM_CONFIDENCE = 0.35


def load_recent_headlines(
    db_path: str,
    ticker: str,
    *,
    days: int = 14,
    limit: int = 20,
) -> list[dict]:
    sym = str(ticker).strip().upper()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        with db_connection(db_path, readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT Date, Title, Summary, Publisher, Source
                FROM stock_news
                WHERE Ticker = ? AND Date >= ?
                  AND Title IS NOT NULL AND TRIM(Title) != ''
                ORDER BY Date DESC
                LIMIT ?
                """,
                (sym, cutoff, limit),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "date": r[0],
            "title": r[1] or "",
            "summary": r[2] or "",
            "publisher": r[3] or "",
            "source": r[4] or "",
        }
        for r in rows
    ]


def load_headlines_bulk(
    db_path: str,
    tickers: Iterable[str],
    *,
    days: int = 14,
    limit_per_ticker: int = 20,
) -> dict[str, list[dict]]:
    """Load recent headlines for many tickers in one query."""
    symbols = [str(t).strip().upper() for t in tickers if t]
    symbols = list(dict.fromkeys(symbols))
    out: dict[str, list[dict]] = {s: [] for s in symbols}
    if not symbols:
        return out
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    placeholders = ", ".join(["?"] * len(symbols))

    def _query() -> list[tuple]:
        with db_connection(db_path, readonly=True) as conn:
            return conn.execute(
                f"""
                SELECT Ticker, Date, Title, Summary, Publisher, Source
                FROM stock_news
                WHERE Ticker IN ({placeholders}) AND Date >= ?
                  AND Title IS NOT NULL AND TRIM(Title) != ''
                ORDER BY Ticker, Date DESC
                """,
                (*symbols, cutoff),
            ).fetchall()

    try:
        rows = execute_with_retry("load_headlines_bulk", _query, caller=f"n={len(symbols)}")
    except sqlite3.OperationalError:
        return out

    counts: dict[str, int] = {s: 0 for s in symbols}
    for ticker, date_val, title, summary, publisher, source in rows:
        sym = str(ticker).upper()
        if sym not in out or counts[sym] >= limit_per_ticker:
            continue
        out[sym].append(
            {
                "date": date_val,
                "title": title or "",
                "summary": summary or "",
                "publisher": publisher or "",
                "source": source or "",
            }
        )
        counts[sym] += 1
    return out


def _sentiment_to_score(sentiment: str) -> float:
    s = sentiment.lower()
    if s in ("bullish", "positive"):
        return 0.75
    if s in ("bearish", "negative"):
        return 0.25
    return 0.5


def apply_bounded_sentiment(raw_score: float, confidence: float) -> float:
    """Cap news impact when confidence is low; pull toward neutral."""
    conf = max(0.0, min(1.0, confidence))
    if conf < MIN_LLM_CONFIDENCE:
        return 0.5
    delta = (raw_score - 0.5) * conf
    delta = max(-MAX_SENTIMENT_DELTA, min(MAX_SENTIMENT_DELTA, delta))
    return round(0.5 + delta, 3)


def score_headlines_list(
    headlines: list[dict],
    ticker: str,
    *,
    max_headlines: int = 5,
) -> tuple[float, list[str], float, list[str]]:
    """
    Return (sentiment 0-1, risk_tags, confidence 0-1, catalyst_tags).

    Uses stored headlines; falls back to neutral when none.
    """
    if not headlines:
        return 0.5, [], 0.5, []
    scores: list[float] = []
    confidences: list[float] = []
    all_tags: list[str] = []
    catalyst_tags: list[str] = []
    for h in headlines[:max_headlines]:
        result = analyze_headline(h["title"], h.get("summary", ""), ticker=ticker)
        scores.append(_sentiment_to_score(result.sentiment))
        confidences.append(float(result.confidence or 0.5))
        all_tags.extend(result.risk_tags or [])
        catalyst_tags.extend(result.thesis_flags or [])
        if result.catalyst_type and result.catalyst_type not in catalyst_tags:
            catalyst_tags.append(result.catalyst_type)
    raw_avg = sum(scores) / len(scores) if scores else 0.5
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
    bounded = apply_bounded_sentiment(raw_avg, avg_conf)
    unique_tags = list(dict.fromkeys(all_tags))[:8]
    unique_catalyst = list(dict.fromkeys(catalyst_tags))[:8]
    return round(bounded, 3), unique_tags, round(avg_conf, 3), unique_catalyst


def score_ticker_news(
    db_path: str,
    ticker: str,
    *,
    days: int = 14,
    max_headlines: int = 5,
    headlines: list[dict] | None = None,
) -> tuple[float, list[str], float, list[str]]:
    """
    Return (sentiment 0-1, risk_tags, confidence, catalyst_tags).

    Uses stored headlines; falls back to neutral 0.5 when none.
    """
    if headlines is None:
        headlines = load_recent_headlines(db_path, ticker, days=days, limit=max_headlines)
    return score_headlines_list(headlines, ticker, max_headlines=max_headlines)
