"""Lightweight web fetch and search for LLM research tools."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

from src.services.stock_config import stock_config

_USER_AGENT = (
    "Mozilla/5.0 (compatible; StockAnalyzer/1.0; +https://localhost/research)"
)


class WebResearch:
    """HTTP fetch + HTML text extraction with domain allowlist."""

    def __init__(self) -> None:
        cfg = stock_config()
        self.max_bytes = cfg.llm_web_max_bytes
        self.max_pages = cfg.llm_web_max_pages
        self.enabled = cfg.llm_web_research_enabled
        self.allowlist = self._parse_allowlist(cfg.llm_web_domain_allowlist)

    @staticmethod
    def _parse_allowlist(raw: str) -> set[str]:
        parts = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
        return set(parts)

    def is_allowed(self, url: str) -> bool:
        if not self.allowlist:
            return False
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            return False
        for domain in self.allowlist:
            if host == domain or host.endswith("." + domain):
                return True
        return False

    def fetch_url(self, url: str) -> dict:
        """Fetch and extract text from a URL."""
        if not self.enabled:
            return {"error": "Web research is disabled in settings."}
        if not self.is_allowed(url):
            return {"error": f"Domain not in allowlist: {url}"}
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=20,
                stream=True,
            )
            resp.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                chunks.append(chunk)
                size += len(chunk)
                if size > self.max_bytes:
                    break
            html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="ignore")
            text = self._extract_text(html)
            return {
                "url": url,
                "title": self._extract_title(html),
                "text": text[:8000],
                "bytes_read": size,
            }
        except Exception as ex:
            return {"error": str(ex), "url": url}

    def search_lite(self, query: str) -> dict:
        """Scrape DuckDuckGo HTML results and fetch top allowed pages."""
        if not self.enabled:
            return {"error": "Web research is disabled in settings."}
        q = (query or "").strip()
        if not q:
            return {"error": "Empty query."}
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
            resp = requests.get(
                search_url,
                headers={"User-Agent": _USER_AGENT},
                timeout=20,
            )
            resp.raise_for_status()
            links = self._extract_search_links(resp.text)
            results: list[dict] = []
            for link in links[: self.max_pages * 3]:
                if len(results) >= self.max_pages:
                    break
                if not self.is_allowed(link):
                    continue
                page = self.fetch_url(link)
                if "error" not in page:
                    results.append(page)
            return {"query": q, "results": results, "count": len(results)}
        except Exception as ex:
            return {"error": str(ex), "query": q}

    @staticmethod
    def _extract_search_links(html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for a in soup.select("a.result__a"):
            href = a.get("href", "")
            if href.startswith("http"):
                links.append(href)
        if not links:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and "duckduckgo" not in href:
                    links.append(href)
        return links

    @staticmethod
    def _extract_title(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        return ""

    @staticmethod
    def _extract_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


_web_research: Optional[WebResearch] = None


def get_web_research() -> WebResearch:
    global _web_research
    if _web_research is None:
        _web_research = WebResearch()
    else:
        _web_research = WebResearch()
    return _web_research
