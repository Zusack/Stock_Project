"""Dashboard market news feed — live headlines with TTL cache."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from src.analysis.news_sources.aggregator import fetch_headlines_for_ticker, persist_headlines
from src.analysis.news_sources.rss_fetcher import fetch_sec_press_releases
from src.analysis.news_sources.types import NewsHeadline
from src.analysis.quote_snapshot import QuoteSnapshot, get_quotes_bulk
from src.analysis.watchlist_schema import resolve_watchlist_symbols

# Market-wide and international proxies (free via Yahoo)
MARKET_TICKERS: tuple[str, ...] = ("^GSPC", "SPY")
INTERNATIONAL_TICKERS: tuple[str, ...] = ("EFA", "EEM", "FXI")  # dev ex-US, emerging, China

DEFAULT_TTL_SEC = 900  # 15 minutes
MAX_HEADLINES_PER_SOURCE = 4
MAX_TOTAL_HEADLINES = 48

_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, list["FeedHeadline"]]] = {}


@dataclass(frozen=True)
class FeedHeadline:
    category: str  # watchlist | market | movers | international | regulatory
    ticker: str
    date: str
    title: str
    publisher: str
    link: str
    source: str = "yahoo"

    @property
    def sort_key(self) -> tuple:
        return (self.date or "", self.title)


def _headline_to_feed(h: NewsHeadline, *, category: str) -> FeedHeadline:
    return FeedHeadline(
        category=category,
        ticker=h.ticker or category.upper(),
        date=h.date,
        title=h.title,
        publisher=h.publisher,
        link=h.link,
        source=h.source,
    )


def _dedupe_feed(items: list[FeedHeadline]) -> list[FeedHeadline]:
    seen: set[str] = set()
    out: list[FeedHeadline] = []
    for item in items:
        key = f"{item.link}|{item.title[:100]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _fetch_for_symbol(sym: str, limit: int) -> list[NewsHeadline]:
    return fetch_headlines_for_ticker(sym, include_yahoo=True, include_sec=False, yahoo_limit=limit)


def _top_movers(symbols: list[str], quotes: dict[str, QuoteSnapshot], *, limit: int = 5) -> list[str]:
    ranked = sorted(
        symbols,
        key=lambda s: abs(quotes[s].change_pct or 0) if s in quotes else 0,
        reverse=True,
    )
    movers: list[str] = []
    for sym in ranked:
        q = quotes.get(sym)
        if q and q.change_pct is not None and abs(q.change_pct) > 0.01:
            movers.append(sym)
        if len(movers) >= limit:
            break
    return movers


def build_dashboard_news_feed(
    db_path: str,
    *,
    watchlist_symbols: list[str] | None = None,
    ttl_sec: int = DEFAULT_TTL_SEC,
    max_total: int = MAX_TOTAL_HEADLINES,
    force_refresh: bool = False,
) -> list[FeedHeadline]:
    """
    Build a merged headline feed for the Dashboard.

    Categories: watchlist, market, movers, international, regulatory (SEC RSS).
    Results are cached briefly to avoid hammering Yahoo on every tab switch.
    """
    cache_key = db_path
    now = time.time()
    if not force_refresh:
        with _cache_lock:
            cached = _cache.get(cache_key)
            if cached and (now - cached[0]) < ttl_sec:
                return list(cached[1])

    if watchlist_symbols is None:
        watchlist_symbols = resolve_watchlist_symbols(db_path)

    watchlist_symbols = [str(s).strip().upper() for s in watchlist_symbols if s]
    quotes = get_quotes_bulk(db_path, watchlist_symbols) if watchlist_symbols else {}
    movers = _top_movers(watchlist_symbols, quotes, limit=5)

    # (symbol, category, per-symbol limit)
    fetch_plan: list[tuple[str, str, int]] = []
    for sym in watchlist_symbols[:12]:
        fetch_plan.append((sym, "watchlist", MAX_HEADLINES_PER_SOURCE))
    for sym in movers:
        if sym not in watchlist_symbols[:12]:
            fetch_plan.append((sym, "movers", MAX_HEADLINES_PER_SOURCE))
    for sym in MARKET_TICKERS:
        fetch_plan.append((sym, "market", MAX_HEADLINES_PER_SOURCE))
    for sym in INTERNATIONAL_TICKERS:
        fetch_plan.append((sym, "international", MAX_HEADLINES_PER_SOURCE))

    # Deduplicate fetch plan by symbol (keep first category)
    seen_syms: set[str] = set()
    unique_plan: list[tuple[str, str, int]] = []
    for sym, cat, lim in fetch_plan:
        if sym in seen_syms:
            continue
        seen_syms.add(sym)
        unique_plan.append((sym, cat, lim))

    collected: list[FeedHeadline] = []
    persist_batch: list[NewsHeadline] = []

    def _work(plan_item: tuple[str, str, int]) -> list[FeedHeadline]:
        sym, category, lim = plan_item
        headlines = _fetch_for_symbol(sym, lim)
        return [_headline_to_feed(h, category=category) for h in headlines]

    max_workers = min(8, max(1, len(unique_plan)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_work, p): p for p in unique_plan}
        for fut in as_completed(futures):
            try:
                collected.extend(fut.result())
            except Exception:
                pass

    # SEC regulatory headlines (market-wide)
    try:
        for h in fetch_sec_press_releases(limit=6):
            if h.title and h.link:
                collected.append(_headline_to_feed(h, category="regulatory"))
                persist_batch.append(h)
    except Exception:
        pass

    # Persist fetched Yahoo headlines for other views / future LLM use
    for item in collected:
        if item.source == "yahoo" and item.ticker:
            persist_batch.append(
                NewsHeadline(
                    ticker=item.ticker,
                    date=item.date,
                    title=item.title,
                    publisher=item.publisher,
                    link=item.link,
                    source=item.source,
                    summary="",
                )
            )
    if persist_batch:
        try:
            persist_headlines(db_path, persist_batch)
        except Exception:
            pass

    merged = _dedupe_feed(collected)
    merged.sort(key=lambda x: x.sort_key, reverse=True)
    merged = merged[:max_total]

    with _cache_lock:
        _cache[cache_key] = (now, merged)
    return merged


def clear_news_cache(db_path: str | None = None) -> None:
    with _cache_lock:
        if db_path is None:
            _cache.clear()
        else:
            _cache.pop(db_path, None)
