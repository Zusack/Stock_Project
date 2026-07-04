"""SEC EDGAR Company Facts API — free, multi-year US fundamentals."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.analysis.fundamentals_sources.rate_limits import (
    SecDailyQuota,
    SecRateLimiter,
    SecRunBudget,
)
from src.analysis.fundamentals_sources.types import FundamentalRow

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# CIK ticker map cache TTL (SEC file changes infrequently).
CIK_CACHE_MAX_AGE_DAYS = 7

_METRIC_SPECS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("Basic EPS", ("EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"), ("USD/shares", "USD")),
    (
        "Net Income",
        ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
        ("USD",),
    ),
    (
        "Total Revenue",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ),
        ("USD",),
    ),
]

_QUARTERLY_FORMS = frozenset({"10-Q", "10-Q/A"})
_ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "20-F", "20-F/A"})


def _data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent.parent.parent
    d = root / "data" / "sec"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _http_get(url: str, user_agent: str, timeout: float = 30.0) -> bytes:
    if not user_agent or len(user_agent.strip()) < 10:
        raise ValueError("SEC User-Agent required (app name + contact email)")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent.strip(),
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _load_ticker_cik_map(user_agent: str) -> dict[str, str]:
    cache_path = _data_dir() / "company_tickers.json"
    stale = True
    if cache_path.is_file():
        age = time.time() - cache_path.stat().st_mtime
        stale = age > CIK_CACHE_MAX_AGE_DAYS * 86400

    if stale:
        SecRateLimiter.wait()
        try:
            raw = _http_get(SEC_TICKERS_URL, user_agent)
            cache_path.write_bytes(raw)
        except urllib.error.HTTPError as e:
            if cache_path.is_file():
                pass  # use stale cache on failure
            else:
                raise RuntimeError(f"SEC ticker list download failed: {e}") from e

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    if isinstance(data, dict):
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            t = str(entry.get("ticker", "")).upper().strip()
            cik = entry.get("cik_str") or entry.get("cik")
            if t and cik is not None:
                mapping[t] = str(int(cik)).zfill(10)
    return mapping


_CIK_CACHE: dict[str, str] | None = None
_CIK_CACHE_UA: str | None = None


def resolve_cik(ticker: str, user_agent: str) -> str | None:
    global _CIK_CACHE, _CIK_CACHE_UA
    ticker = ticker.strip().upper()
    if not ticker or "." in ticker and ticker.endswith(("X", "Y")):
        # Some tickers need normalization; try bare symbol first.
        pass
    ua = user_agent.strip()
    if _CIK_CACHE is None or _CIK_CACHE_UA != ua:
        _CIK_CACHE = _load_ticker_cik_map(ua)
        _CIK_CACHE_UA = ua
    cik = _CIK_CACHE.get(ticker)
    if cik:
        return cik
    # Class shares: BRK.B -> BRK-B in SEC list sometimes
    alt = ticker.replace(".", "-")
    return _CIK_CACHE.get(alt)


def _period_type(form: str, fp: str) -> str | None:
    form = (form or "").upper()
    fp = (fp or "").upper()
    if form in _QUARTERLY_FORMS or fp in ("Q1", "Q2", "Q3", "Q4"):
        return "Quarterly"
    if form in _ANNUAL_FORMS or fp == "FY":
        return "Annual"
    return None


def _parse_end_date(end: str) -> str | None:
    if not end:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(end))
    return m.group(1) if m else None


def _extract_metric_facts(
    us_gaap: dict[str, Any],
    concept_names: tuple[str, ...],
    unit_prefs: tuple[str, ...],
) -> list[dict[str, Any]]:
    for concept in concept_names:
        block = us_gaap.get(concept)
        if not isinstance(block, dict):
            continue
        units = block.get("units")
        if not isinstance(units, dict):
            continue
        for unit_key in unit_prefs:
            items = units.get(unit_key)
            if isinstance(items, list) and items:
                return items
        for items in units.values():
            if isinstance(items, list) and items:
                return items
    return []


def _facts_to_rows(ticker: str, facts_json: dict[str, Any]) -> list[FundamentalRow]:
    facts = facts_json.get("facts") or {}
    us_gaap = facts.get("us-gaap") or {}
    if not isinstance(us_gaap, dict):
        return []

    # Dedupe: (metric, period_type, end) -> best by latest filed
    dedupe: dict[tuple[str, str, str], tuple[str, float, str]] = {}

    for metric_name, concepts, unit_prefs in _METRIC_SPECS:
        items = _extract_metric_facts(us_gaap, concepts, unit_prefs)
        for item in items:
            if not isinstance(item, dict):
                continue
            val = item.get("val")
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval != fval:  # NaN
                continue

            end = _parse_end_date(str(item.get("end", "")))
            if not end:
                continue
            period = _period_type(str(item.get("form", "")), str(item.get("fp", "")))
            if not period:
                continue

            filed = str(item.get("filed", ""))[:10]
            key = (metric_name, period, end)
            prev = dedupe.get(key)
            if prev is None or filed > prev[2]:
                dedupe[key] = (ticker, fval, filed)

    rows: list[FundamentalRow] = []
    for (metric_name, period, end), (sym, fval, _) in dedupe.items():
        rows.append((sym, end, metric_name, fval, period))
    return rows


def fetch_sec_fundamentals(ticker: str, user_agent: str) -> tuple[list[FundamentalRow], str]:
    """
    Fetch fundamentals from SEC. Returns (rows, warning).
    Respects run budget, daily quota, and rate limiter.
    """
    if not SecRunBudget.try_acquire():
        return [], "SEC per-run limit reached"
    if not SecDailyQuota.try_acquire():
        return [], "SEC daily limit reached"

    cik = resolve_cik(ticker, user_agent)
    if not cik:
        return [], "no SEC CIK"

    url = SEC_COMPANY_FACTS_URL.format(cik=cik)
    SecRateLimiter.wait()
    try:
        raw = _http_get(url, user_agent)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return [], "SEC 404"
        if e.code == 429:
            return [], "SEC rate limited"
        return [], f"SEC HTTP {e.code}"
    except Exception as ex:
        return [], f"SEC error: {ex}"

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return [], "SEC invalid JSON"

    rows = _facts_to_rows(ticker.strip().upper(), payload)
    return rows, ""
