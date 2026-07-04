"""Tests for Yahoo news parsing and dashboard news feed."""

from src.analysis.market_news import FeedHeadline, _dedupe_feed, _headline_to_feed
from src.analysis.news_sources.types import NewsHeadline
from src.analysis.news_sources.yahoo_headlines import _parse_yahoo_item, fetch_yahoo_news


def test_parse_yahoo_nested_content_format():
    item = {
        "id": "abc",
        "content": {
            "title": "Apple raises MacBook prices",
            "summary": "Price increase amid component costs.",
            "pubDate": "2026-06-28T13:31:10Z",
            "provider": {"displayName": "Benzinga"},
            "canonicalUrl": {
                "url": "https://finance.yahoo.com/news/apple-raises-prices.html",
            },
        },
    }
    parsed = _parse_yahoo_item(item, "AAPL")
    assert parsed is not None
    assert parsed.title == "Apple raises MacBook prices"
    assert parsed.date == "2026-06-28"
    assert parsed.publisher == "Benzinga"
    assert "yahoo.com" in parsed.link


def test_parse_yahoo_legacy_flat_format():
    item = {
        "title": "Legacy headline",
        "link": "https://finance.yahoo.com/news/legacy.html",
        "providerPublishTime": 1719590400,
        "publisher": "Reuters",
        "summary": "Old API shape",
    }
    parsed = _parse_yahoo_item(item, "MSFT")
    assert parsed is not None
    assert parsed.title == "Legacy headline"
    assert parsed.publisher == "Reuters"


def test_parse_yahoo_skips_empty_title():
    assert _parse_yahoo_item({"content": {"title": "", "canonicalUrl": {"url": "http://x"}}}, "X") is None


def test_dedupe_feed_by_link():
    a = FeedHeadline("market", "SPY", "2026-06-28", "Same story", "Yahoo", "http://x/1")
    b = FeedHeadline("watchlist", "AAPL", "2026-06-28", "Same story", "Yahoo", "http://x/1")
    out = _dedupe_feed([a, b])
    assert len(out) == 1


def test_headline_to_feed_category():
    h = NewsHeadline(
        ticker="NVDA",
        date="2026-06-27",
        title="Chip demand strong",
        publisher="Yahoo",
        link="https://example.com/nvda",
        source="yahoo",
    )
    feed = _headline_to_feed(h, category="movers")
    assert feed.category == "movers"
    assert feed.ticker == "NVDA"


def test_fetch_yahoo_news_live_smoke():
    """Smoke test against Yahoo — skip if offline."""
    rows = fetch_yahoo_news("AAPL", limit=3)
    if not rows:
        return
    assert rows[0].title
    assert rows[0].link
    assert rows[0].date >= "2020-01-01"
