"""Shared helpers for strategy backtests (period, portfolio, chart payloads)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_PORTFOLIO_START = 1.0
MIN_WARMUP_DAYS = 252


@dataclass
class TradeMarker:
    """A buy/sell marker on a performance chart."""

    x: int
    y: float
    side: str  # "buy" | "sell"
    tooltip: str


@dataclass
class TickerChartSeries:
    ticker: str
    dates: list[str] = field(default_factory=list)
    pct_returns: list[float] = field(default_factory=list)
    portfolio_values: list[float] = field(default_factory=list)
    markers: list[TradeMarker] = field(default_factory=list)


def parse_lookback_days(value: str | int | None, default: int = DEFAULT_LOOKBACK_DAYS) -> int:
    try:
        days = int(value) if value is not None else default
    except (TypeError, ValueError):
        days = default
    return max(30, days)


def parse_portfolio_start(value: str | float | None, default: float = DEFAULT_PORTFOLIO_START) -> float:
    try:
        start = float(value) if value is not None else default
    except (TypeError, ValueError):
        start = default
    return max(0.01, start)


def period_bounds(df: pd.DataFrame, lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = df.index.max()
    start = end - pd.Timedelta(days=lookback_days)
    return start, end


def pct_returns_from_start(prices: pd.Series) -> pd.Series:
    if prices.empty:
        return prices
    base = float(prices.iloc[0])
    if base == 0:
        return prices * 0.0
    return (prices / base - 1.0) * 100.0


def format_period(start: pd.Timestamp | None, end: pd.Timestamp | None) -> str:
    if start is None or end is None:
        return "—"
    return f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"


def simulate_portfolio_values(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    trades: list[dict],
    initial_capital: float,
) -> list[float]:
    """Replay trades with fractional shares; one position at a time."""
    if len(dates) == 0:
        return []

    entry_map: dict[pd.Timestamp, dict] = {}
    exit_map: dict[pd.Timestamp, dict] = {}
    for trade in trades:
        entry_map[pd.Timestamp(trade["Entry_Date"])] = trade
        exit_map[pd.Timestamp(trade["Exit_Date"])] = trade

    price_list = prices.astype(float).tolist()
    cash = float(initial_capital)
    shares = 0.0
    values: list[float] = []
    for i, date in enumerate(dates):
        ts = pd.Timestamp(date)
        price = price_list[i]
        if ts in exit_map and shares > 0:
            cash = shares * price
            shares = 0.0
        if ts in entry_map and shares == 0 and cash > 0 and price > 0:
            shares = cash / price
            cash = 0.0
        values.append(cash + shares * price)
    return values


def combine_portfolio_series(
    series_by_ticker: dict[str, TickerChartSeries],
) -> tuple[list[str], list[float]]:
    """Sum per-ticker portfolio curves on a shared calendar."""
    if not series_by_ticker:
        return [], []

    frames = []
    for ticker, series in series_by_ticker.items():
        if not series.dates:
            continue
        idx = pd.to_datetime(series.dates)
        frames.append(
            pd.DataFrame({ticker: series.portfolio_values}, index=idx).sort_index()
        )
    if not frames:
        return [], []

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.join(frame, how="outer")
    combined = combined.sort_index().ffill().fillna(0.0)
    total = combined.sum(axis=1)
    return [d.strftime("%Y-%m-%d") for d in total.index], total.tolist()


def flags_to_trigger_string(flags: dict[str, bool]) -> str:
    active = [key for key, ok in flags.items() if ok]
    return ", ".join(active) if active else "none"


def build_marker_tooltip(
    *,
    side: str,
    date: pd.Timestamp,
    price: float,
    triggers: str | None = None,
    reason: str | None = None,
    return_pct: float | None = None,
) -> str:
    lines = [f"{side.upper()} {date.strftime('%Y-%m-%d')}", f"Price: ${price:.2f}"]
    if triggers:
        lines.append(f"Triggers: {triggers}")
    if reason:
        lines.append(f"Reason: {reason}")
    if return_pct is not None:
        lines.append(f"Return: {return_pct * 100:.2f}%")
    return "\n".join(lines)
