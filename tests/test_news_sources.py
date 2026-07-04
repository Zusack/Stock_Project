"""News source deduplication and types."""

from src.analysis.news_sources.aggregator import _dedupe_key
from src.analysis.news_sources.types import NewsHeadline


def test_dedupe_key_stable():
    h = NewsHeadline(
        ticker="AAPL",
        date="2024-01-01",
        title="Apple beats estimates",
        publisher="Test",
        link="https://example.com/a",
        source="test",
    )
    assert _dedupe_key(h) == _dedupe_key(h)
