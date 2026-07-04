"""Investigate why a ticker failed ingest or validation (Yahoo, news, SEC EDGAR)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.analysis.ticker_cleanup import _fetch_yahoo_signals, is_plausible_symbol

from src.analysis.ticker_registry import record_investigation, symbol_has_any_history

Outcome = Literal[
    "active",
    "likely_delisted",
    "invalid_symbol",
    "transient_error",
    "inconclusive",
]
DelistCategory = Literal[
    "unknown",
    "not_found",
    "merger_acquisition",
    "bankruptcy",
    "exchange_delisting",
    "symbol_changed",
]
Confidence = Literal["high", "medium", "low"]

_SEC_UA = "StockProject/1.0 (data-ingest maintenance; contact@local)"
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_NEWS_KEYWORDS: dict[str, DelistCategory] = {
    "acquired": "merger_acquisition",
    "merger": "merger_acquisition",
    "bankruptcy": "bankruptcy",
    "chapter 11": "bankruptcy",
    "delist": "exchange_delisting",
    "delisted": "exchange_delisting",
    "symbol change": "symbol_changed",
    "ticker change": "symbol_changed",
}

_cik_cache: dict[str, str] | None = None


@dataclass
class TickerInvestigation:
    symbol: str
    outcome: Outcome
    delist_category: DelistCategory | None = None
    confidence: Confidence = "low"
    reasons: list[str] = field(default_factory=list)
    yahoo_signals: dict[str, Any] = field(default_factory=dict)
    sec_signals: dict[str, Any] | None = None
    sec_filing_url: str | None = None

    def summary_text(self) -> str:
        lines = [f"{self.symbol}: {self.outcome}"]
        if self.delist_category:
            lines.append(f"Category: {self.delist_category} ({self.confidence})")
        lines.extend(self.reasons[:8])
        return "\n".join(lines)

    def validation_status(self) -> str:
        if self.outcome == "active":
            return "ok"
        if self.outcome == "invalid_symbol":
            return "invalid"
        if self.outcome == "likely_delisted":
            return "delisted"
        if self.outcome == "transient_error":
            return "review"
        return "review"

    def reason_text(self) -> str:
        return "; ".join(self.reasons)[:500]


def _load_sec_cik_map() -> dict[str, str]:
    global _cik_cache
    if _cik_cache is not None:
        return _cik_cache
    req = urllib.request.Request(
        _SEC_TICKERS_URL,
        headers={"User-Agent": _SEC_UA, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        _cik_cache = {}
        return _cik_cache
    mapping: dict[str, str] = {}
    for entry in raw.values():
        ticker = str(entry.get("ticker", "")).upper()
        cik = str(entry.get("cik_str", ""))
        if ticker and cik:
            mapping[ticker] = cik.zfill(10)
    _cik_cache = mapping
    return mapping


def _sec_recent_filings(cik: str, *, max_filings: int = 40) -> list[dict[str, str]]:
    """Fetch recent filing metadata from SEC submissions API."""
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _SEC_UA, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    descs = recent.get("primaryDocDescription", []) or recent.get("description", [])
    accessions = recent.get("accessionNumber", [])
    out: list[dict[str, str]] = []
    for i in range(min(len(forms), max_filings)):
        out.append(
            {
                "form": str(forms[i]),
                "date": str(dates[i]) if i < len(dates) else "",
                "description": str(descs[i]) if i < len(descs) else "",
                "accession": str(accessions[i]) if i < len(accessions) else "",
            }
        )
    return out


def _sec_filing_url(cik: str, accession: str, primary_doc: str = "") -> str:
    cik_num = str(int(cik))  # strip leading zeros for URL path
    acc = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc}"
    if primary_doc:
        return f"{base}/{primary_doc}"
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=40"


def _scan_sec_filings(cik: str) -> tuple[DelistCategory | None, list[str], str | None]:
    filings = _sec_recent_filings(cik)
    if not filings:
        return None, [], None

    keywords: dict[str, DelistCategory] = {
        "delist": "exchange_delisting",
        "merger": "merger_acquisition",
        "acquisition": "merger_acquisition",
        "bankruptcy": "bankruptcy",
        "dissolution": "bankruptcy",
    }
    target_forms = {"25", "8-K", "15", "25-NSE"}

    for f in filings:
        form = f.get("form", "")
        desc = (f.get("description") or "").lower()
        if form not in target_forms and not any(k in desc for k in keywords):
            continue
        category: DelistCategory = "unknown"
        for kw, cat in keywords.items():
            if kw in desc or form.startswith("25"):
                category = cat
                break
        if form.startswith("25"):
            category = "exchange_delisting"
        reason = f"SEC {form} ({f.get('date', '')}): {f.get('description', '')[:80]}"
        url = _sec_filing_url(cik, f.get("accession", ""))
        return category, [reason], url

    return None, [], None


def _scan_yahoo_news(ticker: str) -> tuple[DelistCategory | None, list[str]]:
    try:
        import yfinance as yf

        news = yf.Ticker(ticker).get_news(count=5) or []
    except Exception:
        return None, []

    reasons: list[str] = []
    category: DelistCategory | None = None
    for item in news:
        title = str(item.get("title", ""))
        summary = str(item.get("summary", ""))
        blob = f"{title} {summary}".lower()
        for kw, cat in _NEWS_KEYWORDS.items():
            if kw in blob:
                category = cat
                reasons.append(f"News hint ({cat}): {title[:100]}")
                break
        if category:
            break
    return category, reasons


def investigate_symbol(
    symbol: str,
    db_path: str | None = None,
    *,
    use_sec: bool = True,
    use_yahoo: bool = True,
) -> TickerInvestigation:
    """Classify ticker failure using Yahoo, optional news, and SEC EDGAR."""
    sym = symbol.strip().upper()
    reasons: list[str] = []
    inv = TickerInvestigation(symbol=sym, outcome="inconclusive")

    if not is_plausible_symbol(sym):
        inv.outcome = "invalid_symbol"
        inv.delist_category = "not_found"
        inv.confidence = "high"
        inv.reasons = ["Symbol format is not a valid ticker pattern"]
        return inv

    if sym.startswith("^"):
        inv.outcome = "active"
        inv.confidence = "high"
        inv.reasons = ["Market/index symbol"]
        return inv

    has_db = bool(db_path and symbol_has_any_history(db_path, sym))
    if has_db:
        reasons.append("Has price history in local database")

    if not use_yahoo:
        inv.reasons = reasons or ["Yahoo check skipped"]
        inv.outcome = "active" if has_db else "inconclusive"
        return inv

    time.sleep(0.15)
    yahoo = _fetch_yahoo_signals(sym)
    inv.yahoo_signals = yahoo
    yerr = str(yahoo.get("error") or "")
    hist_rows = int(yahoo.get("history_rows") or 0)

    if yerr and ("429" in yerr or "rate" in yerr.lower()):
        inv.outcome = "transient_error"
        inv.confidence = "low"
        inv.reasons = ["Yahoo rate limit — retry later", yerr]
        return inv

    if "possibly delisted" in yerr.lower():
        reasons.append("Yahoo reports possibly delisted (may be false positive)")

    if hist_rows > 0:
        inv.outcome = "active"
        inv.confidence = "high"
        inv.reasons = reasons + [f"Yahoo returned {hist_rows} recent daily bars"]
        return inv

    quote_type = yahoo.get("quote_type")
    if quote_type and quote_type not in (
        "EQUITY",
        "ETF",
        "MUTUALFUND",
        "INDEX",
        "CURRENCY",
        "CRYPTOCURRENCY",
    ):
        reasons.append(f"quoteType={quote_type}")

    if has_db and hist_rows == 0:
        inv.outcome = "likely_delisted"
        inv.confidence = "medium"
        reasons.append("No recent Yahoo prices; local history retained")
    elif hist_rows == 0 and not has_db:
        inv.outcome = "likely_delisted"
        inv.delist_category = "not_found"
        inv.confidence = "high"
        reasons.append("No Yahoo prices and no local history")
    else:
        inv.outcome = "inconclusive"
        reasons.append("Could not confirm active listing")

    news_cat, news_reasons = _scan_yahoo_news(sym)
    if news_cat:
        inv.delist_category = news_cat
        reasons.extend(news_reasons)
        if inv.confidence == "high":
            inv.confidence = "medium"

    if use_sec and not sym.startswith("^"):
        cik_map = _load_sec_cik_map()
        cik = cik_map.get(sym)
        if cik:
            time.sleep(0.2)
            sec_cat, sec_reasons, sec_url = _scan_sec_filings(cik)
            inv.sec_signals = {"cik": cik}
            if sec_cat:
                inv.delist_category = sec_cat
                reasons.extend(sec_reasons)
                inv.sec_filing_url = sec_url
                inv.confidence = "high" if inv.outcome == "likely_delisted" else "medium"
            elif cik:
                reasons.append(f"SEC CIK {cik}: no delist/merger filings in recent index")
        else:
            reasons.append("No SEC CIK mapping (non-US or unknown issuer)")

    if inv.delist_category is None and inv.outcome == "likely_delisted":
        inv.delist_category = "unknown"

    inv.reasons = reasons
    return inv


def investigate_and_record(
    db_path: str,
    symbol: str,
    *,
    use_sec: bool = True,
) -> TickerInvestigation:
    inv = investigate_symbol(symbol, db_path, use_sec=use_sec)
    record_investigation(
        db_path,
        symbol,
        validation_status=inv.validation_status(),
        validation_reason=inv.reason_text(),
        delist_category=inv.delist_category,
    )
    return inv


def investigate_symbols(
    db_path: str,
    symbols: list[str],
    *,
    use_sec: bool = True,
    progress_callback: Any | None = None,
) -> list[TickerInvestigation]:
    results: list[TickerInvestigation] = []
    total = len(symbols)
    for i, sym in enumerate(symbols):
        inv = investigate_and_record(db_path, sym, use_sec=use_sec)
        results.append(inv)
        if progress_callback:
            progress_callback((i + 1) / total if total else 1.0, sym)
    return results


def investigation_to_dict(inv: TickerInvestigation) -> dict[str, Any]:
    d = asdict(inv)
    return d
