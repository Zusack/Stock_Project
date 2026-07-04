"""Price history coverage, gap detection, and per-ticker ingest planning."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.analysis.db import db_connection
from src.analysis.market_calendar import (
    day_after,
    iter_weekdays,
    last_completed_trading_day,
    next_weekday,
    previous_weekday,
)
from src.services.stock_config import stock_config

COVERAGE_EMPTY = "empty"
COVERAGE_CURRENT = "current"
COVERAGE_STALE = "stale"
COVERAGE_GAPS = "gaps"

_ingest_skip_lookup: dict[str, str] | None = None


def set_ingest_skip_lookup(lookup: dict[str, str] | None) -> None:
    """Set per-run cache for dead-ticker skip checks (avoids per-symbol init_db)."""
    global _ingest_skip_lookup
    _ingest_skip_lookup = lookup


def clear_ingest_skip_lookup() -> None:
    set_ingest_skip_lookup(None)


@dataclass
class PriceBounds:
    min_date: date | None
    max_date: date | None
    row_count: int
    distinct_dates: int


@dataclass
class DateRange:
    start: date
    end: date

    def start_str(self) -> str:
        return self.start.isoformat()

    def end_str(self) -> str:
        return self.end.isoformat()


@dataclass
class TickerIngestPlan:
    ticker: str
    db_path: str
    price_action: str = "skip"  # skip | incremental | full | gap_fill
    fund_action: str = "skip"  # skip | refresh
    enrich_action: str = "skip"  # skip | refresh (profile / insider / news)
    price_ranges: list[DateRange] = field(default_factory=list)
    period: str = "max"
    coverage_status: str = COVERAGE_EMPTY
    gap_count: int = 0
    force_full: bool = False
    skip_reason: str | None = None

    @property
    def all_skip(self) -> bool:
        return (
            self.price_action == "skip"
            and self.fund_action == "skip"
            and self.enrich_action == "skip"
        )


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def get_price_bounds(db_path: str | os.PathLike, ticker: str) -> PriceBounds:
    sym = str(ticker).strip().upper()
    with db_connection(db_path, readonly=True) as conn:
        row = conn.execute(
            """
            SELECT MIN(Date), MAX(Date), COUNT(*), COUNT(DISTINCT Date)
            FROM stock_history WHERE Ticker = ?
        """,
            (sym,),
        ).fetchone()
    if not row or row[2] == 0:
        return PriceBounds(None, None, 0, 0)
    return PriceBounds(
        _parse_date(row[0]),
        _parse_date(row[1]),
        int(row[2]),
        int(row[3]),
    )


def get_existing_dates(db_path: str | os.PathLike, ticker: str) -> set[date]:
    sym = str(ticker).strip().upper()
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(
            "SELECT Date FROM stock_history WHERE Ticker = ?",
            (sym,),
        ).fetchall()
    out: set[date] = set()
    for (d,) in rows:
        parsed = _parse_date(d)
        if parsed:
            out.add(parsed)
    return out


def find_missing_ranges(
    db_path: str | os.PathLike,
    ticker: str,
    *,
    bounds: PriceBounds | None = None,
    lookback_days: int = 90,
    max_ranges: int = 12,
) -> list[DateRange]:
    """Missing weekdays in a recent window (avoids holiday backfill for full history)."""
    bounds = bounds or get_price_bounds(db_path, ticker)
    if bounds.min_date is None or bounds.max_date is None:
        return []
    window_start = bounds.max_date - timedelta(days=lookback_days)
    range_start = max(bounds.min_date, window_start)
    existing = get_existing_dates(db_path, ticker)
    missing: list[date] = []
    for d in iter_weekdays(range_start, bounds.max_date):
        if d not in existing:
            missing.append(d)
    if not missing:
        return []
    ranges: list[DateRange] = []
    start = missing[0]
    prev = missing[0]
    for d in missing[1:]:
        if (d - prev).days <= 3:
            prev = d
            continue
        ranges.append(DateRange(start, prev))
        start = d
        prev = d
    ranges.append(DateRange(start, prev))
    return ranges[:max_ranges]


def _merge_ranges(ranges: list[DateRange]) -> list[DateRange]:
    if not ranges:
        return []
    sorted_r = sorted(ranges, key=lambda x: x.start)
    merged = [sorted_r[0]]
    for r in sorted_r[1:]:
        last = merged[-1]
        if r.start <= last.end + timedelta(days=3):
            merged[-1] = DateRange(last.start, max(last.end, r.end))
        else:
            merged.append(r)
    return merged


def get_fundamentals_last_at(db_path: str | os.PathLike, ticker: str) -> str | None:
    sym = str(ticker).strip().upper()
    try:
        with db_connection(db_path, readonly=True) as conn:
            row = conn.execute(
                "SELECT fundamentals_last_at FROM watchlist_tickers WHERE symbol = ?",
                (sym,),
            ).fetchone()
        return row[0] if row and row[0] else None
    except sqlite3.OperationalError:
        return None


def assess_coverage(
    db_path: str | os.PathLike,
    ticker: str,
    *,
    as_of: date | None = None,
) -> tuple[str, int, PriceBounds]:
    """Return (coverage_status, gap_count, bounds)."""
    bounds = get_price_bounds(db_path, ticker)
    if bounds.row_count == 0:
        return COVERAGE_EMPTY, 0, bounds

    last_td = last_completed_trading_day(as_of)
    gaps = find_missing_ranges(db_path, ticker, bounds=bounds)
    gap_count = sum(
        max(1, (r.end - r.start).days + 1) for r in gaps
    ) if gaps else 0
    gap_summary: str | None = None
    if gaps:
        parts = [
            f"{r.start.isoformat()}..{r.end.isoformat()}"
            for r in gaps[:4]
        ]
        gap_summary = "; ".join(parts)
        if len(gaps) > 4:
            gap_summary += f" (+{len(gaps) - 4} more)"

    if bounds.max_date and bounds.max_date < last_td:
        status = COVERAGE_STALE
    elif gaps and gap_count > 0:
        status = COVERAGE_GAPS
    else:
        status = COVERAGE_CURRENT
    return status, gap_count, bounds


def refresh_ticker_coverage(db_path: str | os.PathLike, ticker: str) -> dict[str, Any]:
    sym = str(ticker).strip().upper()
    status, gap_count, bounds = assess_coverage(db_path, sym)
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE watchlist_tickers SET
                price_min_date = ?,
                price_max_date = ?,
                price_row_count = ?,
                coverage_status = ?,
                gap_count = ?,
                gap_summary = ?
            WHERE symbol = ?
        """,
            (
                bounds.min_date.isoformat() if bounds.min_date else None,
                bounds.max_date.isoformat() if bounds.max_date else None,
                bounds.row_count,
                status,
                gap_count,
                gap_summary,
                sym,
            ),
        )
        conn.commit()
    return {
        "symbol": sym,
        "coverage_status": status,
        "gap_count": gap_count,
        "gap_summary": gap_summary,
        "price_min_date": bounds.min_date,
        "price_max_date": bounds.max_date,
    }


def _dead_ticker_skip_reason(
    db_path: str | os.PathLike,
    sym: str,
    *,
    retry_dead: bool,
    force_retry: bool,
) -> str | None:
    """Return skip reason for known-dead symbols, or None to continue planning."""
    if retry_dead or force_retry:
        return None
    if _ingest_skip_lookup is not None:
        return _ingest_skip_lookup.get(sym)

    from src.analysis.ticker_registry import get_symbol

    row = get_symbol(db_path, sym)
    if not row:
        return None
    if row.skip_ingest:
        return "archived"
    if row.validation_status in ("delisted", "invalid"):
        return row.validation_status
    if row.last_ingest_status == "No Price Data":
        return "no price data"
    return None


def build_ticker_plan(
    db_path: str | os.PathLike,
    ticker: str,
    *,
    mode: str = "smart",
    paused_at: str | None = None,
    force_full: bool | None = None,
) -> TickerIngestPlan:
    """Build ingest plan for one symbol."""
    sym = str(ticker).strip().upper()
    cfg = stock_config()
    force = force_full if force_full is not None else cfg.force_full_history
    plan = TickerIngestPlan(ticker=sym, db_path=str(db_path))

    skip_reason = _dead_ticker_skip_reason(
        db_path,
        sym,
        retry_dead=cfg.retry_dead_tickers,
        force_retry=bool(force or mode == "full"),
    )
    if skip_reason:
        plan.price_action = "skip"
        plan.fund_action = "skip"
        plan.skip_reason = skip_reason
        return plan

    if force or mode == "full":
        plan.price_action = "full"
        plan.period = "max"
        plan.fund_action = "refresh"
        plan.coverage_status = COVERAGE_EMPTY
        return plan

    status, gap_count, bounds = assess_coverage(db_path, sym)
    plan.coverage_status = status
    plan.gap_count = gap_count
    last_td = last_completed_trading_day()

    if status == COVERAGE_EMPTY:
        plan.price_action = "full"
        plan.period = "max"
        plan.fund_action = "refresh"
        return plan

    incremental_range: DateRange | None = None
    if bounds.max_date and bounds.max_date < last_td:
        incremental_range = DateRange(day_after(bounds.max_date), last_td)

    # Only repair gaps near the trailing edge when price is not already current.
    recent_gaps: list[DateRange] = []
    if bounds.max_date and bounds.max_date < last_td:
        recent_gaps = find_missing_ranges(db_path, sym, bounds=bounds, lookback_days=30)

    if recent_gaps and incremental_range:
        plan.price_action = "gap_fill"
        plan.price_ranges = _merge_ranges(recent_gaps + [incremental_range])
    elif recent_gaps:
        plan.price_action = "gap_fill"
        plan.price_ranges = recent_gaps
    elif incremental_range:
        plan.price_action = "incremental"
        plan.price_ranges = [incremental_range]
    else:
        plan.price_action = "skip"

    fund_days = cfg.fundamentals_refresh_days
    fund_last = get_fundamentals_last_at(db_path, sym)
    fund_stale = True
    if fund_last:
        try:
            dt = datetime.strptime(fund_last[:19], "%Y-%m-%d %H:%M:%S")
            fund_stale = (datetime.now(timezone.utc) - dt.replace(tzinfo=timezone.utc)).days >= fund_days
        except ValueError:
            fund_stale = True

    if plan.price_action != "skip":
        plan.fund_action = "refresh"
    elif fund_stale or not fund_last:
        plan.fund_action = "refresh"
    else:
        plan.fund_action = "skip"

    if cfg.ingest_fetch_profile or cfg.ingest_fetch_insider or cfg.ingest_fetch_news:
        plan.enrich_action = "refresh"
    else:
        plan.enrich_action = "skip"

    return plan


def audit_watchlist(db_path: str | os.PathLike) -> dict[str, Any]:
    """Scan all active watchlist symbols and refresh coverage cache."""
    from src.analysis import ticker_registry as registry

    init_coverage_schema(db_path)
    rows = registry.list_symbols(db_path, include_archived=False, limit=None)
    counts: dict[str, int] = {COVERAGE_EMPTY: 0, COVERAGE_CURRENT: 0, COVERAGE_STALE: 0, COVERAGE_GAPS: 0}
    for row in rows:
        if row.skip_ingest:
            continue
        info = refresh_ticker_coverage(db_path, row.symbol)
        st = info.get("coverage_status", COVERAGE_EMPTY)
        counts[st] = counts.get(st, 0) + 1
    return {
        "total": sum(counts.values()),
        "counts": counts,
        "stale": counts.get(COVERAGE_STALE, 0),
        "gaps": counts.get(COVERAGE_GAPS, 0),
        "empty": counts.get(COVERAGE_EMPTY, 0),
        "current": counts.get(COVERAGE_CURRENT, 0),
    }


def init_coverage_schema(db_path: str | os.PathLike) -> None:
    """Add coverage columns to watchlist_tickers if missing."""
    with db_connection(db_path, readonly=False) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(watchlist_tickers)")}
        migrations = {
            "price_min_date": "TEXT",
            "price_max_date": "TEXT",
            "price_row_count": "INTEGER DEFAULT 0",
            "coverage_status": "TEXT",
            "gap_count": "INTEGER DEFAULT 0",
            "gap_summary": "TEXT",
            "fundamentals_last_at": "TEXT",
            "archive_reason": "TEXT",
            "archived_at": "TEXT",
            "investigation_at": "TEXT",
            "delist_category": "TEXT",
        }
        for col, dtype in migrations.items():
            if col not in cols:
                try:
                    conn.execute(f"ALTER TABLE watchlist_tickers ADD COLUMN {col} {dtype}")
                except sqlite3.OperationalError:
                    pass
        conn.commit()


def set_fundamentals_last_at(db_path: str | os.PathLike, ticker: str) -> None:
    sym = str(ticker).strip().upper()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    init_coverage_schema(db_path)
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            "UPDATE watchlist_tickers SET fundamentals_last_at = ? WHERE symbol = ?",
            (now, sym),
        )
        conn.commit()
