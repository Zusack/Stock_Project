"""Aggregate and persist headlines from free providers."""

from __future__ import annotations

import sqlite3

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.news_sources.rss_fetcher import fetch_sec_press_releases
from src.analysis.news_sources.types import NewsHeadline
from src.analysis.news_sources.yahoo_headlines import fetch_yahoo_news


def _dedupe_key(h: NewsHeadline) -> str:
    return f"{h.link}|{h.title[:80]}"


def fetch_headlines_for_ticker(
    ticker: str,
    *,
    include_yahoo: bool = True,
    include_sec: bool = False,
    yahoo_limit: int = 15,
) -> list[NewsHeadline]:
    """Collect headlines for one symbol."""
    seen: set[str] = set()
    out: list[NewsHeadline] = []

    if include_yahoo:
        for h in fetch_yahoo_news(ticker, limit=yahoo_limit):
            k = _dedupe_key(h)
            if k not in seen:
                seen.add(k)
                out.append(h)

    if include_sec:
        sym = ticker.upper()
        for h in fetch_sec_press_releases(limit=50):
            if sym and sym in h.title.upper():
                k = _dedupe_key(h)
                if k not in seen:
                    seen.add(k)
                    h = NewsHeadline(
                        ticker=sym,
                        date=h.date,
                        title=h.title,
                        publisher=h.publisher,
                        link=h.link,
                        source=h.source,
                        summary=h.summary,
                    )
                    out.append(h)
    return out


def ensure_news_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_news (
            Ticker TEXT, Date TEXT, Title TEXT, Publisher TEXT, Link TEXT,
            UNIQUE(Ticker, Link)
        )
    """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(stock_news)")}
    for col, dtype in (
        ("Source", "TEXT"),
        ("Summary", "TEXT"),
        ("Fetched_At", "TEXT"),
    ):
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE stock_news ADD COLUMN {col} {dtype}")
            except sqlite3.OperationalError:
                pass


def persist_headlines(db_path: str, headlines: list[NewsHeadline]) -> int:
    if not headlines:
        return 0
    from datetime import datetime, timezone

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = [
        (
            h.ticker,
            h.date,
            h.title,
            h.publisher,
            h.link,
            h.source,
            h.summary,
            fetched_at,
        )
        for h in headlines
    ]
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            ensure_news_schema(conn)
            conn.executemany(
                """
                INSERT OR IGNORE INTO stock_news
                (Ticker, Date, Title, Publisher, Link, Source, Summary, Fetched_At)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                rows,
            )
            conn.commit()
    return len(rows)


def ingest_headlines_batch(
    db_path: str,
    tickers: list[str],
    *,
    include_yahoo: bool = True,
) -> dict[str, int]:
    """Fetch and store headlines for multiple tickers."""
    counts: dict[str, int] = {}
    for sym in tickers:
        headlines = fetch_headlines_for_ticker(sym, include_yahoo=include_yahoo)
        counts[sym] = persist_headlines(db_path, headlines)
    return counts
