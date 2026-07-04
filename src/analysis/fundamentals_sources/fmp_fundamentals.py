"""Financial Modeling Prep income statement (optional API key, daily quota)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from src.analysis.fundamentals_sources.rate_limits import FmpDailyQuota
from src.analysis.fundamentals_sources.types import FundamentalRow

FMP_INCOME_URL = "https://financialmodelingprep.com/api/v3/income-statement/{symbol}"

# One call returns both quarterly and annual when limit is sufficient.
DEFAULT_LIMIT = 40


def fetch_fmp_fundamentals(
    ticker: str,
    api_key: str,
    limit: int = DEFAULT_LIMIT,
    *,
    include_annual: bool = False,
) -> tuple[list[FundamentalRow], str]:
    """
    One FMP call per period (quarterly by default). Set include_annual=True
    for a second call — uses two daily quota slots per ticker.
    """
    key = (api_key or "").strip()
    if not key:
        return [], "FMP API key not set"

    ticker = ticker.strip().upper()
    rows: list[FundamentalRow] = []

    periods = [("quarter", "Quarterly")]
    if include_annual:
        periods.append(("annual", "Annual"))

    for period, period_type in periods:
        if not FmpDailyQuota.try_acquire():
            return rows, "FMP daily quota exhausted"
        q = urllib.parse.urlencode(
            {
                "period": period,
                "limit": max(4, min(80, int(limit))),
                "apikey": key,
            }
        )
        url = f"{FMP_INCOME_URL.format(symbol=ticker)}?{q}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                return rows, "FMP rate limited"
            return rows, f"FMP HTTP {e.code}"
        except Exception as ex:
            return rows, f"FMP error: {ex}"

        if not isinstance(data, list):
            if isinstance(data, dict) and "Error Message" in data:
                return rows, str(data.get("Error Message", "FMP error"))[:80]
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            date_str = str(item.get("date") or item.get("fillingDate") or "")[:10]
            if len(date_str) < 10:
                continue
            eps = item.get("eps")
            if eps is not None:
                try:
                    rows.append((ticker, date_str, "Basic EPS", float(eps), period_type))
                except (TypeError, ValueError):
                    pass
            rev = item.get("revenue")
            if rev is not None:
                try:
                    rows.append((ticker, date_str, "Total Revenue", float(rev), period_type))
                except (TypeError, ValueError):
                    pass
            ni = item.get("netIncome")
            if ni is not None:
                try:
                    rows.append((ticker, date_str, "Net Income", float(ni), period_type))
                except (TypeError, ValueError):
                    pass
            for metric_key, label in (
                ("grossProfit", "Gross Profit"),
                ("operatingIncome", "Operating Income"),
                ("ebitda", "EBITDA"),
                ("totalAssets", "Total Assets"),
                ("totalLiabilities", "Total Liabilities"),
                ("totalStockholdersEquity", "Stockholders Equity"),
            ):
                val = item.get(metric_key)
                if val is not None:
                    try:
                        rows.append((ticker, date_str, label, float(val), period_type))
                    except (TypeError, ValueError):
                        pass

    return rows, ""
