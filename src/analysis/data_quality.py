"""Data quality tracking for ingest and daily freshness audits."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.history_coverage import assess_coverage, audit_watchlist
from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.analysis.market_calendar import last_completed_trading_day


def classify_ingest_error(status: str) -> str:
    s = str(status or "")
    if "429" in s or "Rate Limit" in s:
        return "rate_limit"
    if "timed out" in s.lower() or "Timeout" in s:
        return "timeout"
    if "No Price Data" in s:
        return "no_price"
    if s.startswith("Price Error"):
        if "Invalid comparison" in s or "datetime64" in s:
            return "datetime_compare"
        return "price_fetch"
    if s.startswith("Skipped"):
        return "skipped"
    if s.startswith("General Error"):
        return "general"
    return "ok" if s == "Success" else "other"


def record_ingest_event(
    db_path: str,
    symbol: str,
    *,
    phase: str,
    status: str,
    price_rows: int = 0,
    fund_rows: int = 0,
    notes: str = "",
) -> None:
    ensure_intelligence_schema(db_path)
    sym = str(symbol).strip().upper()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    category = classify_ingest_error(status)
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                INSERT INTO data_quality_events
                (symbol, event_at, phase, status, error_category, price_rows, fund_rows, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sym, now, phase, status[:200], category, price_rows, fund_rows, notes[:500]),
            )
            conn.execute(
                """
                UPDATE watchlist_tickers SET
                    last_ingest_status = ?,
                    last_ingest_at = ?
                WHERE symbol = ?
                """,
                (status[:120], now, sym),
            )
            conn.commit()


def record_ingest_result(db_path: str, result: dict[str, Any]) -> None:
    """Persist quality row from a single-ticker ingest result dict."""
    ticker = result.get("Ticker", "")
    status = str(result.get("Status", "Success"))
    record_ingest_event(
        db_path,
        ticker,
        phase="ingest",
        status=status,
        price_rows=int(result.get("Price_Rows") or 0),
        fund_rows=int(result.get("Fund_Rows") or 0),
        notes=str(result.get("Fund_Note") or "")[:500],
    )


def freshness_summary(db_path: str, *, sample_limit: int = 500) -> dict[str, Any]:
    """Summarize coverage and recent ingest error rates."""
    ensure_intelligence_schema(db_path)
    audit = audit_watchlist(db_path)
    last_td = last_completed_trading_day()

    with db_connection(db_path, readonly=True) as conn:
        err_rows = conn.execute(
            """
            SELECT error_category, COUNT(*) FROM data_quality_events
            WHERE event_at >= datetime('now', '-7 days')
            GROUP BY error_category
            """
        ).fetchall()
        recent_fail = conn.execute(
            """
            SELECT COUNT(*) FROM data_quality_events
            WHERE event_at >= datetime('now', '-7 days')
              AND error_category NOT IN ('ok', 'skipped')
            """
        ).fetchone()

    return {
        "required_trading_day": last_td.isoformat(),
        "coverage_audit": audit,
        "errors_last_7d_by_category": {r[0]: r[1] for r in err_rows},
        "failures_last_7d": int(recent_fail[0]) if recent_fail else 0,
    }


def symbols_with_recent_errors(
    db_path: str,
    *,
    category: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    ensure_intelligence_schema(db_path)
    sql = """
        SELECT symbol, status, error_category, event_at, notes
        FROM data_quality_events
        WHERE error_category NOT IN ('ok', 'skipped')
    """
    params: list[Any] = []
    if category:
        sql += " AND error_category = ?"
        params.append(category)
    sql += " ORDER BY event_at DESC LIMIT ?"
    params.append(limit)
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "symbol": r[0],
            "status": r[1],
            "error_category": r[2],
            "event_at": r[3],
            "notes": r[4],
        }
        for r in rows
    ]


def assess_symbol_freshness(db_path: str, symbol: str) -> dict[str, Any]:
    sym = str(symbol).strip().upper()
    status, gap_count, bounds = assess_coverage(db_path, sym)
    last_td = last_completed_trading_day()
    stale = bool(bounds.max_date and bounds.max_date < last_td)
    return {
        "symbol": sym,
        "coverage_status": status,
        "gap_count": gap_count,
        "price_max_date": bounds.max_date.isoformat() if bounds.max_date else None,
        "required_day": last_td.isoformat(),
        "is_stale": stale,
        "row_count": bounds.row_count,
    }
