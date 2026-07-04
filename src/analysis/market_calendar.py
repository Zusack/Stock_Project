"""US equity trading-day helpers (weekday-based v1)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


def _as_date(d: date | datetime | str | None) -> date | None:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return date.fromisoformat(d[:10])
    return None


def is_weekday(d: date) -> bool:
    return d.weekday() < 5


def previous_weekday(d: date) -> date:
    cur = d
    while not is_weekday(cur):
        cur -= timedelta(days=1)
    return cur


def next_weekday(d: date) -> date:
    cur = d
    while not is_weekday(cur):
        cur += timedelta(days=1)
    return cur


def last_completed_trading_day(as_of: date | datetime | None = None) -> date:
    """
    Last US session that should have settled daily bars available.
    Before ~21:00 UTC on a weekday, use the previous weekday.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    d = _as_date(as_of) or date.today()
    if isinstance(as_of, datetime) and as_of.tzinfo is not None:
        hour = as_of.astimezone(timezone.utc).hour
    else:
        hour = datetime.now(timezone.utc).hour

    d = previous_weekday(d)
    if is_weekday(d) and hour < 21:
        d -= timedelta(days=1)
        d = previous_weekday(d)
    return d


def trading_days_between(start: date | str, end: date | str, *, inclusive_end: bool = False) -> int:
    """Count weekdays in (start, end] or [start, end) depending on inclusive_end."""
    s = _as_date(start)
    e = _as_date(end)
    if s is None or e is None or e <= s:
        return 0
    count = 0
    cur = next_weekday(s + timedelta(days=1))
    while cur <= e:
        if is_weekday(cur):
            count += 1
        if cur >= e:
            break
        cur += timedelta(days=1)
    if inclusive_end and is_weekday(e) and e > s:
        if count == 0 or cur != e:
            if e not in {s + timedelta(days=i) for i in range((e - s).days + 1)}:
                pass
    return count


def iter_weekdays(start: date, end: date):
    """Yield weekdays from start through end inclusive."""
    cur = max(start, previous_weekday(start))
    end = previous_weekday(end) if not is_weekday(end) else end
    while cur <= end:
        if is_weekday(cur):
            yield cur
        cur += timedelta(days=1)


def day_after(d: date | str) -> date:
    n = _as_date(d)
    if n is None:
        return date.today()
    return next_weekday(n + timedelta(days=1))


def filter_settled_daily_bars(hist, *, as_of: date | datetime | None = None):
    """
    Remove daily OHLCV rows after the last completed US session.

    Yahoo/Stooq daily bars for "today" during market hours use the current price as
    Close; this drops those (and any later calendar dates) until after the ~21:00 UTC
    settle cutoff used by last_completed_trading_day().

    Returns (filtered_hist, dropped_row_count).
    """
    import pandas as pd

    if hist is None or hist.empty:
        return hist, 0

    cutoff = last_completed_trading_day(as_of)
    out = hist.copy()
    # yfinance uses tz-aware UTC indices; compare calendar days in UTC consistently.
    idx = pd.to_datetime(out.index, utc=True)
    bar_days = idx.normalize()
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    mask = bar_days <= cutoff_ts
    dropped = int((~mask).sum())
    if dropped:
        out = out.loc[mask]
    return out, dropped
