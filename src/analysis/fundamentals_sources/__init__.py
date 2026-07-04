"""Unified fundamentals fetch for ingest."""

from __future__ import annotations

from src.analysis.fundamentals_sources.fmp_fundamentals import fetch_fmp_fundamentals
from src.analysis.fundamentals_sources.merge import merge_fundamental_rows
from src.analysis.fundamentals_sources.sec_edgar import fetch_sec_fundamentals
from src.analysis.fundamentals_sources.types import FundamentalRow, FundamentalsMeta
from src.analysis.fundamentals_sources.yahoo_fundamentals import fetch_yahoo_fundamentals
from src.services.settings_store import settings_store


def _setting(key: str, default: str) -> str:
    val = settings_store().get_setting(key, default)
    return val if val is not None else default


def fetch_fundamentals_for_ticker(
    ticker: str,
    *,
    source: str | None = None,
    sec_user_agent: str | None = None,
    fmp_api_key: str | None = None,
) -> tuple[list[FundamentalRow], FundamentalsMeta]:
    """
    Fetch fundamentals rows for one ticker according to settings.

    Sources: sec+yahoo (default), sec, yahoo, fmp, fmp+yahoo
    """
    mode = (source or _setting("fundamentals_source", "sec+yahoo")).strip().lower()
    ua = (sec_user_agent or _setting("sec_user_agent", "")).strip()
    fmp_key = (fmp_api_key or _setting("fmp_api_key", "")).strip()

    meta = FundamentalsMeta()
    batches: list[tuple[list[FundamentalRow], str]] = []
    warnings: list[str] = []

    use_sec = mode in ("sec", "sec+yahoo")
    use_yahoo = mode in ("yahoo", "sec+yahoo", "fmp+yahoo")
    use_fmp = mode in ("fmp", "fmp+yahoo")

    if use_sec:
        if not ua:
            warnings.append("SEC User-Agent missing in Settings")
        else:
            sec_rows, sec_warn = fetch_sec_fundamentals(ticker, ua)
            if sec_rows:
                batches.append((sec_rows, "sec"))
                meta.sec_used = True
            if sec_warn:
                warnings.append(sec_warn)

    if use_fmp:
        fmp_annual = _setting("fmp_fetch_annual", "0") == "1"
        fmp_rows, fmp_warn = fetch_fmp_fundamentals(
            ticker, fmp_key, include_annual=fmp_annual
        )
        if fmp_rows:
            batches.append((fmp_rows, "fmp"))
            meta.fmp_used = True
        if fmp_warn:
            warnings.append(fmp_warn)

    if use_yahoo:
        yahoo_rows = fetch_yahoo_fundamentals(ticker)
        if yahoo_rows:
            batches.append((yahoo_rows, "yahoo"))
            meta.yahoo_used = True

    if not batches and mode != "yahoo":
        # Last resort if SEC/FMP failed
        yahoo_rows = fetch_yahoo_fundamentals(ticker)
        if yahoo_rows:
            batches.append((yahoo_rows, "yahoo"))
            meta.yahoo_used = True
            warnings.append("fallback yahoo")

    merged = merge_fundamental_rows(*batches) if batches else []

    tags = []
    if meta.sec_used:
        tags.append("sec")
    if meta.fmp_used:
        tags.append("fmp")
    if meta.yahoo_used:
        tags.append("yahoo")
    meta.source = "+".join(tags) if tags else "none"
    meta.row_count = len(merged)
    meta.warning = "; ".join(warnings)[:200]
    return merged, meta
