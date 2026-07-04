"""Fetch headlines from free RSS feeds (SEC press releases, etc.)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.request import Request, urlopen

from src.analysis.news_sources.types import NewsHeadline

_USER_AGENT = "StockAnalyzer/1.0 (local research; contact: local)"

# Free public RSS endpoints
SEC_PRESS_RELEASES_RSS = "https://www.sec.gov/news/pressreleases.rss"


def _fetch_rss(url: str, *, timeout: float = 15.0) -> str:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_rss_items(xml_text: str, source: str, default_ticker: str = "") -> list[NewsHeadline]:
    items: list[NewsHeadline] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    for item in root.iter("item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate") or item.find("published")
        desc_el = item.find("description")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        if not title or not link:
            continue
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        if pub_el is not None and pub_el.text:
            try:
                from email.utils import parsedate_to_datetime

                date_str = parsedate_to_datetime(pub_el.text).strftime("%Y-%m-%d")
            except (TypeError, ValueError, OverflowError):
                pass
        summary = ""
        if desc_el is not None and desc_el.text:
            summary = re.sub(r"<[^>]+>", "", desc_el.text).strip()[:500]
        items.append(
            NewsHeadline(
                ticker=default_ticker,
                date=date_str,
                title=title,
                publisher=source,
                link=link,
                source=source,
                summary=summary,
            )
        )
    return items


def fetch_sec_press_releases(limit: int = 30) -> list[NewsHeadline]:
    """SEC press release RSS (market-wide, ticker empty)."""
    try:
        xml_text = _fetch_rss(SEC_PRESS_RELEASES_RSS)
        return _parse_rss_items(xml_text, "SEC Press", "")[:limit]
    except OSError:
        return []
