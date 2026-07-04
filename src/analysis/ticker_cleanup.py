"""Identify delisted or invalid tickers and clean ticker CSV files."""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.analysis.db import ticker_has_recent_history
from src.analysis.ingest import load_tickers_from_csv

ProgressCallback = Callable[[float, str], None]

# Yahoo quote types we treat as valid listed instruments.
_VALID_QUOTE_TYPES = frozenset(
    {
        "EQUITY",
        "ETF",
        "MUTUALFUND",
        "INDEX",
        "CURRENCY",
        "CRYPTOCURRENCY",
    }
)

_INDEX_PATTERN = re.compile(r"^\^[A-Z0-9.\-^=_]+$")
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,12}$")


def is_plausible_symbol(ticker: str) -> bool:
    t = ticker.strip().upper()
    if not t or t == "NAN":
        return False
    if t.startswith("^"):
        return bool(_INDEX_PATTERN.match(t))
    return bool(_SYMBOL_PATTERN.match(t))


def _fetch_yahoo_signals(ticker: str) -> dict[str, Any]:
    """Live Yahoo checks (requires network)."""
    signals: dict[str, Any] = {
        "history_rows": 0,
        "quote_type": None,
        "short_name": None,
        "regular_market_price": None,
        "error": None,
    }
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo", auto_adjust=False)
        signals["history_rows"] = 0 if hist is None or hist.empty else len(hist)

        try:
            info = stock.info or {}
        except Exception as info_exc:
            info = {}
            signals["error"] = str(info_exc)

        signals["quote_type"] = info.get("quoteType")
        signals["short_name"] = info.get("shortName") or info.get("longName")
        signals["regular_market_price"] = info.get("regularMarketPrice")
        signals["symbol"] = info.get("symbol")
    except Exception as exc:
        signals["error"] = str(exc)
    return signals


def classify_ticker(
    ticker: str,
    db_path: str | Path | None = None,
    *,
    use_yahoo: bool = True,
) -> dict[str, Any]:
    """
    Classify a ticker for list cleaning.

    verdict: keep | remove_delisted | remove_invalid | review
    """
    t = ticker.strip().upper()
    reasons: list[str] = []

    if not is_plausible_symbol(t):
        return {
            "Ticker": t,
            "verdict": "remove_invalid",
            "confidence": "high",
            "reasons": ["Symbol format is not a valid ticker pattern"],
        }

    if t.startswith("^"):
        return {
            "Ticker": t,
            "verdict": "keep",
            "confidence": "high",
            "reasons": ["Market/index symbol"],
        }

    has_db = bool(db_path and ticker_has_recent_history(db_path, t))
    if has_db:
        reasons.append("Has price history in local DB (last ~2 years)")

    if not use_yahoo:
        verdict = "keep" if has_db else "review"
        return {
            "Ticker": t,
            "verdict": verdict,
            "confidence": "low",
            "reasons": reasons or ["No local data; Yahoo check skipped"],
        }

    time.sleep(0.15)
    yahoo = _fetch_yahoo_signals(t)
    hist_rows = yahoo["history_rows"]
    quote_type = yahoo.get("quote_type")
    price = yahoo.get("regular_market_price")
    yerr = yahoo.get("error")

    if yerr and "429" in str(yerr):
        return {
            "Ticker": t,
            "verdict": "review",
            "confidence": "low",
            "reasons": ["Yahoo rate limit — retry later", str(yerr)],
            "yahoo": yahoo,
        }

    delisted_signals = []
    if hist_rows == 0:
        delisted_signals.append("No recent price history on Yahoo (1mo)")
    if quote_type and quote_type not in _VALID_QUOTE_TYPES:
        delisted_signals.append(f"quoteType={quote_type}")
    if quote_type is None and hist_rows == 0 and price is None and not has_db:
        delisted_signals.append("No quote metadata and no prices")

    if hist_rows == 0 and not has_db:
        if quote_type and quote_type not in _VALID_QUOTE_TYPES:
            return {
                "Ticker": t,
                "verdict": "remove_delisted",
                "confidence": "high",
                "reasons": delisted_signals,
                "yahoo": yahoo,
            }
        if quote_type is None and price is None:
            return {
                "Ticker": t,
                "verdict": "remove_delisted",
                "confidence": "high",
                "reasons": delisted_signals + ["Symbol not found as active listing"],
                "yahoo": yahoo,
            }
        if not delisted_signals:
            return {
                "Ticker": t,
                "verdict": "remove_delisted",
                "confidence": "medium",
                "reasons": ["No recent Yahoo prices and not in local DB"],
                "yahoo": yahoo,
            }

    if has_db or hist_rows > 0 or quote_type in _VALID_QUOTE_TYPES:
        if has_db:
            reasons.append("Confirmed via local database")
        if hist_rows > 0:
            reasons.append(f"Yahoo returned {hist_rows} recent daily bars")
        if quote_type:
            reasons.append(f"quoteType={quote_type}")
        return {
            "Ticker": t,
            "verdict": "keep",
            "confidence": "high" if has_db or hist_rows > 0 else "medium",
            "reasons": reasons,
            "yahoo": yahoo,
        }

    return {
        "Ticker": t,
        "verdict": "review",
        "confidence": "low",
        "reasons": reasons or ["Inconclusive — manual review suggested"],
        "yahoo": yahoo,
    }


def scan_ticker_list(
    tickers: list[str],
    db_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    rows = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        rows.append(classify_ticker(ticker, db_path))
        if progress_callback:
            progress_callback((i + 1) / total, ticker)
    df = pd.DataFrame(rows)
    if not df.empty and "reasons" in df.columns:
        df["reasons"] = df["reasons"].apply(
            lambda r: "; ".join(r) if isinstance(r, list) else str(r)
        )
    return df


def scan_ticker_csv(
    csv_path: str | Path,
    db_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    tickers = load_tickers_from_csv(csv_path)
    return scan_ticker_list(tickers, db_path, progress_callback)


def apply_ticker_cleanup(
    csv_path: str | Path,
    tickers_to_remove: list[str],
    *,
    backup: bool = True,
) -> dict[str, Any]:
    """Remove tickers from CSV; writes backup copy first."""
    path = Path(csv_path)
    tickers = load_tickers_from_csv(path)
    remove_set = {t.strip().upper() for t in tickers_to_remove}
    kept = [t for t in tickers if t not in remove_set]
    removed = [t for t in tickers if t in remove_set]

    backup_path = None
    if backup and path.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
        shutil.copy2(path, backup_path)

    out = pd.DataFrame({"Symbol": kept})
    out.to_csv(path, index=False)

    return {
        "csv_path": str(path.resolve()),
        "backup_path": str(backup_path) if backup_path else None,
        "before_count": len(tickers),
        "after_count": len(kept),
        "removed_count": len(removed),
        "removed": removed,
    }
