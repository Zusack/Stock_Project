"""On-demand quote snapshots via yfinance with SQLite TTL cache."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.db_perf import execute_with_retry
from src.analysis.quotes_schema import ensure_quotes_schema

DEFAULT_TTL_SEC = 900  # 15 minutes


@dataclass
class QuoteSnapshot:
    ticker: str
    last_price: float | None = None
    prev_close: float | None = None
    open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: int | None = None
    avg_volume_10d: float | None = None
    avg_volume_3mo: float | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    eps: float | None = None
    beta: float | None = None
    sector: str = ""
    industry: str = ""
    fifty_day_avg: float | None = None
    two_hundred_day_avg: float | None = None
    fetched_at: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def change(self) -> float | None:
        if self.last_price is None or self.prev_close is None or self.prev_close == 0:
            return None
        return self.last_price - self.prev_close

    @property
    def change_pct(self) -> float | None:
        if self.last_price is None or self.prev_close is None or self.prev_close == 0:
            return None
        return (self.last_price / self.prev_close - 1.0) * 100.0

    @property
    def volume_vs_avg(self) -> float | None:
        avg = self.avg_volume_10d or self.avg_volume_3mo
        if self.volume is None or avg is None or avg == 0:
            return None
        return self.volume / avg


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    f = _safe_float(val)
    if f is None:
        return None
    return int(f)


def _fetch_yfinance_quote(ticker: str) -> QuoteSnapshot:
    """Fetch a single quote from yfinance (network call)."""
    import yfinance as yf

    sym = str(ticker).strip().upper()
    t = yf.Ticker(sym)
    fi = t.fast_info
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    def _fi(key: str, snake: str | None = None):
        for k in (key, snake or key):
            try:
                v = fi.get(k) if hasattr(fi, "get") else getattr(fi, k, None)
                if v is not None:
                    return v
            except Exception:
                pass
        if snake:
            try:
                return getattr(fi, snake, None)
            except Exception:
                pass
        return None

    last = _safe_float(_fi("lastPrice", "last_price") or _fi("regularMarketPrice"))
    prev = _safe_float(_fi("previousClose", "previous_close") or _fi("regularMarketPreviousClose"))
    open_px = _safe_float(_fi("open", "open") or _fi("regularMarketOpen"))
    day_hi = _safe_float(_fi("dayHigh", "day_high") or _fi("regularMarketDayHigh"))
    day_lo = _safe_float(_fi("dayLow", "day_low") or _fi("regularMarketDayLow"))
    vol = _safe_int(_fi("lastVolume", "last_volume") or _fi("volume"))
    avg10 = _safe_float(_fi("tenDayAverageVolume", "ten_day_average_volume"))
    avg3mo = _safe_float(_fi("threeMonthAverageVolume", "three_month_average_volume"))
    mcap = _safe_float(_fi("marketCap", "market_cap"))
    shares = _safe_float(_fi("shares", "shares") or info.get("sharesOutstanding"))
    hi52 = _safe_float(_fi("yearHigh", "year_high") or _fi("fiftyTwoWeekHigh"))
    lo52 = _safe_float(_fi("yearLow", "year_low") or _fi("fiftyTwoWeekLow"))
    trail_pe = _safe_float(info.get("trailingPE") or info.get("trailingPe"))
    fwd_pe = _safe_float(info.get("forwardPE") or info.get("forwardPe"))
    eps = _safe_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"))
    beta = _safe_float(info.get("beta"))
    sector = str(info.get("sector") or "")
    industry = str(info.get("industry") or "")
    sma50 = _safe_float(_fi("fiftyDayAverage", "fifty_day_average"))
    sma200 = _safe_float(_fi("twoHundredDayAverage", "two_hundred_day_average"))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return QuoteSnapshot(
        ticker=sym,
        last_price=last,
        prev_close=prev,
        open=open_px,
        day_high=day_hi,
        day_low=day_lo,
        volume=vol,
        avg_volume_10d=avg10,
        avg_volume_3mo=avg3mo,
        market_cap=mcap,
        shares_outstanding=shares,
        fifty_two_week_high=hi52,
        fifty_two_week_low=lo52,
        trailing_pe=trail_pe,
        forward_pe=fwd_pe,
        eps=eps,
        beta=beta,
        sector=sector,
        industry=industry,
        fifty_day_avg=sma50,
        two_hundred_day_avg=sma200,
        fetched_at=now,
    )


def _row_to_quote(row: tuple, cols: list[str]) -> QuoteSnapshot:
    d = dict(zip(cols, row))
    extra = {}
    if d.get("Extra_Json"):
        try:
            extra = json.loads(d["Extra_Json"])
        except json.JSONDecodeError:
            pass
    return QuoteSnapshot(
        ticker=str(d.get("Ticker", "")),
        last_price=d.get("Last_Price"),
        prev_close=d.get("Prev_Close"),
        open=d.get("Open"),
        day_high=d.get("Day_High"),
        day_low=d.get("Day_Low"),
        volume=d.get("Volume"),
        avg_volume_10d=d.get("Avg_Volume_10d"),
        avg_volume_3mo=d.get("Avg_Volume_3mo"),
        market_cap=d.get("Market_Cap"),
        shares_outstanding=d.get("Shares_Outstanding"),
        fifty_two_week_high=d.get("FiftyTwo_Week_High"),
        fifty_two_week_low=d.get("FiftyTwo_Week_Low"),
        trailing_pe=d.get("Trailing_PE"),
        forward_pe=d.get("Forward_PE"),
        eps=d.get("EPS"),
        beta=d.get("Beta"),
        sector=str(d.get("Sector") or ""),
        industry=str(d.get("Industry") or ""),
        fifty_day_avg=d.get("Fifty_Day_Avg"),
        two_hundred_day_avg=d.get("Two_Hundred_Day_Avg"),
        fetched_at=str(d.get("Fetched_At") or ""),
        extra=extra,
    )


def _persist_quote(db_path: str, q: QuoteSnapshot) -> None:
    ensure_quotes_schema(db_path)
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                INSERT INTO quotes_snapshot (
                    Ticker, Last_Price, Prev_Close, Open, Day_High, Day_Low,
                    Volume, Avg_Volume_10d, Avg_Volume_3mo, Market_Cap,
                    Shares_Outstanding, FiftyTwo_Week_High, FiftyTwo_Week_Low,
                    Trailing_PE, Forward_PE, EPS, Beta, Sector, Industry,
                    Fifty_Day_Avg, Two_Hundred_Day_Avg, Fetched_At, Extra_Json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(Ticker) DO UPDATE SET
                    Last_Price=excluded.Last_Price,
                    Prev_Close=excluded.Prev_Close,
                    Open=excluded.Open,
                    Day_High=excluded.Day_High,
                    Day_Low=excluded.Day_Low,
                    Volume=excluded.Volume,
                    Avg_Volume_10d=excluded.Avg_Volume_10d,
                    Avg_Volume_3mo=excluded.Avg_Volume_3mo,
                    Market_Cap=excluded.Market_Cap,
                    Shares_Outstanding=excluded.Shares_Outstanding,
                    FiftyTwo_Week_High=excluded.FiftyTwo_Week_High,
                    FiftyTwo_Week_Low=excluded.FiftyTwo_Week_Low,
                    Trailing_PE=excluded.Trailing_PE,
                    Forward_PE=excluded.Forward_PE,
                    EPS=excluded.EPS,
                    Beta=excluded.Beta,
                    Sector=excluded.Sector,
                    Industry=excluded.Industry,
                    Fifty_Day_Avg=excluded.Fifty_Day_Avg,
                    Two_Hundred_Day_Avg=excluded.Two_Hundred_Day_Avg,
                    Fetched_At=excluded.Fetched_At,
                    Extra_Json=excluded.Extra_Json
                """,
                (
                    q.ticker,
                    q.last_price,
                    q.prev_close,
                    q.open,
                    q.day_high,
                    q.day_low,
                    q.volume,
                    q.avg_volume_10d,
                    q.avg_volume_3mo,
                    q.market_cap,
                    q.shares_outstanding,
                    q.fifty_two_week_high,
                    q.fifty_two_week_low,
                    q.trailing_pe,
                    q.forward_pe,
                    q.eps,
                    q.beta,
                    q.sector,
                    q.industry,
                    q.fifty_day_avg,
                    q.two_hundred_day_avg,
                    q.fetched_at,
                    json.dumps(q.extra) if q.extra else None,
                ),
            )
            conn.commit()


def _is_fresh(fetched_at: str, ttl_sec: int) -> bool:
    if not fetched_at:
        return False
    try:
        ts = fetched_at.replace(" UTC", "").strip()
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return age < ttl_sec
    except ValueError:
        return False


def load_cached_quotes(
    db_path: str,
    tickers: Iterable[str],
) -> dict[str, QuoteSnapshot]:
    """Load cached quotes without network refresh."""
    symbols = [str(t).strip().upper() for t in tickers if t]
    symbols = list(dict.fromkeys(symbols))
    out: dict[str, QuoteSnapshot] = {}
    if not symbols:
        return out
    ensure_quotes_schema(db_path)
    placeholders = ", ".join(["?"] * len(symbols))

    def _query():
        with db_connection(db_path, readonly=True) as conn:
            cols = [d[1] for d in conn.execute("PRAGMA table_info(quotes_snapshot)").fetchall()]
            rows = conn.execute(
                f"SELECT * FROM quotes_snapshot WHERE Ticker IN ({placeholders})",
                symbols,
            ).fetchall()
            return cols, rows

    try:
        cols, rows = execute_with_retry("load_cached_quotes", _query)
    except sqlite3.OperationalError:
        return out

    for row in rows:
        q = _row_to_quote(row, cols)
        out[q.ticker] = q
    return out


def get_quote(
    db_path: str,
    ticker: str,
    *,
    ttl_sec: int = DEFAULT_TTL_SEC,
    force_refresh: bool = False,
) -> QuoteSnapshot | None:
    """Return quote snapshot, refreshing from yfinance if stale."""
    sym = str(ticker).strip().upper()
    if not sym:
        return None
    cached = load_cached_quotes(db_path, [sym]).get(sym)
    if cached and not force_refresh and _is_fresh(cached.fetched_at, ttl_sec):
        return cached
    try:
        q = _fetch_yfinance_quote(sym)
        _persist_quote(db_path, q)
        return q
    except Exception:
        return cached


def get_quotes_bulk(
    db_path: str,
    tickers: Iterable[str],
    *,
    ttl_sec: int = DEFAULT_TTL_SEC,
    force_refresh: bool = False,
) -> dict[str, QuoteSnapshot]:
    """Batch fetch quotes with cache; refreshes stale/missing only."""
    symbols = [str(t).strip().upper() for t in tickers if t]
    symbols = list(dict.fromkeys(symbols))
    out: dict[str, QuoteSnapshot] = {}
    if not symbols:
        return out

    cached = load_cached_quotes(db_path, symbols)
    to_refresh: list[str] = []
    for sym in symbols:
        q = cached.get(sym)
        if q and not force_refresh and _is_fresh(q.fetched_at, ttl_sec):
            out[sym] = q
        else:
            to_refresh.append(sym)

    for sym in to_refresh:
        try:
            q = _fetch_yfinance_quote(sym)
            _persist_quote(db_path, q)
            out[sym] = q
        except Exception:
            if sym in cached:
                out[sym] = cached[sym]
    return out


def quote_to_dict(q: QuoteSnapshot) -> dict:
    return asdict(q)
