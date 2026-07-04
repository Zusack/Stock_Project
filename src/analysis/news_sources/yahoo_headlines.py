"""Yahoo Finance headlines via yfinance (already a project dependency)."""

from __future__ import annotations

from datetime import datetime, timezone

import yfinance as yf

from src.analysis.news_sources.types import NewsHeadline


def _parse_pub_date(raw) -> str:
    if raw is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(int(raw), tz=timezone.utc).strftime("%Y-%m-%d")
        except (TypeError, ValueError, OSError):
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = str(raw).strip()
    if not text:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if "T" in text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return text[:10] if len(text) >= 10 else datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _extract_link(content: dict, item: dict) -> str:
    for blob in (content, item):
        for key in ("canonicalUrl", "clickThroughUrl"):
            cu = blob.get(key)
            if isinstance(cu, dict) and cu.get("url"):
                return str(cu["url"]).strip()
        for key in ("link", "url"):
            val = blob.get(key)
            if val:
                return str(val).strip()
    return ""


def _extract_publisher(content: dict, item: dict) -> str:
    prov = content.get("provider")
    if isinstance(prov, dict) and prov.get("displayName"):
        return str(prov["displayName"])
    if item.get("publisher"):
        return str(item["publisher"])
    return "Yahoo Finance"


def _parse_yahoo_item(item: dict, sym: str) -> NewsHeadline | None:
    """Parse one yfinance news item (legacy flat or 2024+ nested content)."""
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = (content.get("title") or item.get("title") or "").strip()
    link = _extract_link(content, item)
    if not title or not link:
        return None
    pub_raw = (
        content.get("pubDate")
        or content.get("displayTime")
        or item.get("providerPublishTime")
        or item.get("pubDate")
    )
    summary = (content.get("summary") or content.get("description") or item.get("summary") or "")[:500]
    return NewsHeadline(
        ticker=sym,
        date=_parse_pub_date(pub_raw),
        title=title,
        publisher=_extract_publisher(content, item),
        link=link,
        source="yahoo",
        summary=summary,
    )


def fetch_yahoo_news(ticker: str, limit: int = 20) -> list[NewsHeadline]:
    sym = ticker.upper().strip()
    if not sym:
        return []
    try:
        t = yf.Ticker(sym)
        raw = t.news or []
        if not raw:
            raw = t.get_news(count=limit) or []
    except Exception:
        return []

    rows: list[NewsHeadline] = []
    seen: set[str] = set()
    for item in raw[: max(limit, len(raw))]:
        parsed = _parse_yahoo_item(item, sym)
        if parsed is None:
            continue
        key = f"{parsed.link}|{parsed.title[:80]}"
        if key in seen:
            continue
        seen.add(key)
        rows.append(parsed)
        if len(rows) >= limit:
            break
    return rows
