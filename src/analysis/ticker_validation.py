"""Ticker validation wrappers around src.analysis.ticker_cleanup."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from src.analysis.ticker_cleanup import classify_ticker, scan_ticker_list

from src.analysis.ticker_registry import _verdict_to_status, record_validation

ProgressCallback = Callable[[float, str], None]


def validate_symbol(
    db_path: str | os.PathLike,
    symbol: str,
    *,
    use_yahoo: bool = True,
) -> dict[str, Any]:
    """Classify one symbol and persist validation columns."""
    result = classify_ticker(symbol, db_path, use_yahoo=use_yahoo)
    verdict = result.get("verdict", "review")
    reasons = result.get("reasons", [])
    if isinstance(reasons, list):
        reason_text = "; ".join(str(r) for r in reasons)
    else:
        reason_text = str(reasons)

    status = _verdict_to_status(verdict)
    record_validation(db_path, symbol, status, reason_text)
    result["validation_status"] = status
    result["validation_reason"] = reason_text
    return result


def validate_symbols(
    db_path: str | os.PathLike,
    symbols: list[str],
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Validate multiple symbols and persist results."""
    results: list[dict[str, Any]] = []
    total = len(symbols)
    for i, sym in enumerate(symbols):
        results.append(validate_symbol(db_path, sym))
        if progress_callback:
            progress_callback((i + 1) / total if total else 1.0, sym)
    return results


def scan_and_record(
    db_path: str | os.PathLike,
    symbols: list[str],
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Run batch scan (classify only) and record each result."""
    rows = scan_ticker_list(symbols, db_path, progress_callback=progress_callback)
    out: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        sym = str(row.get("Ticker", ""))
        verdict = str(row.get("verdict", "review"))
        reasons = row.get("reasons", "")
        status = _verdict_to_status(verdict)
        record_validation(db_path, sym, status, str(reasons))
        out.append(
            {
                "Ticker": sym,
                "verdict": verdict,
                "validation_status": status,
                "validation_reason": reasons,
            }
        )
    return out
