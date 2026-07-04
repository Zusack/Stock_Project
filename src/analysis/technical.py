"""MACD vs Bollinger technical strategy analysis."""

from __future__ import annotations

from src.analysis.parallel_exec import run_parallel_map
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.backtest_common import (
    DEFAULT_LOOKBACK_DAYS,
    TickerChartSeries,
    TradeMarker,
    build_marker_tooltip,
    combine_portfolio_series,
    format_period,
    pct_returns_from_start,
    period_bounds,
)
from src.analysis.db import db_connection, load_entire_database

ProgressCallback = Callable[[float], None]


@dataclass
class TechnicalResult:
    results_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    avg_baseline: float = 0.0
    avg_macd: float = 0.0
    avg_bollinger: float = 0.0
    stock_count: int = 0
    period_label: str = ""
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    chart_series: dict[str, TickerChartSeries] = field(default_factory=dict)
    combined_portfolio_dates: list[str] = field(default_factory=list)
    combined_portfolio_values: list[float] = field(default_factory=list)


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + (std * num_std), sma - (std * num_std)


def strat_macd(df: pd.DataFrame) -> float:
    data = df.copy()
    data["MACD"], data["Signal_Line"] = calculate_macd(data["Adj Close"])
    data.dropna(inplace=True)
    if data.empty:
        return 0.0
    data["Signal"] = np.where(data["MACD"] > data["Signal_Line"], 1, 0)
    data["Strategy_Return"] = data["Signal"].shift(1) * data["Adj Close"].pct_change(fill_method=None)
    return float((1 + data["Strategy_Return"]).prod())


def strat_bollinger(df: pd.DataFrame) -> float:
    data = df.copy()
    data["Upper"], data["Lower"] = calculate_bollinger_bands(data["Adj Close"])
    data.dropna(inplace=True)
    if data.empty:
        return 0.0
    prices = data["Adj Close"].values
    uppers = data["Upper"].values
    lowers = data["Lower"].values
    position = 0
    cash = 1.0
    shares = 0
    for i in range(len(prices)):
        price = prices[i]
        if position == 0 and price < lowers[i]:
            shares = cash / price
            cash = 0
            position = 1
        elif position == 1 and price > uppers[i]:
            cash = shares * price
            shares = 0
            position = 0
    final_value = cash if position == 0 else shares * prices[-1]
    return float(final_value)


def _bollinger_trades(prices, lowers, uppers) -> list[tuple[int, str, float]]:
    events: list[tuple[int, str, float]] = []
    position = 0
    for i in range(len(prices)):
        price = prices[i]
        if position == 0 and price < lowers[i]:
            events.append((i, "buy", price))
            position = 1
        elif position == 1 and price > uppers[i]:
            events.append((i, "sell", price))
            position = 0
    return events


def _macd_crossovers(macd, signal_line) -> list[tuple[int, str]]:
    events: list[tuple[int, str]] = []
    prev = 0
    for i in range(len(macd)):
        cur = 1 if macd[i] > signal_line[i] else 0
        if cur != prev:
            events.append((i, "buy" if cur == 1 else "sell"))
        prev = cur
    return events


def _build_technical_chart(
    ticker: str,
    df: pd.DataFrame,
    lookback_days: int,
    initial_capital: float,
) -> TickerChartSeries | None:
    period_start, _ = period_bounds(df, lookback_days)
    window = df[df.index >= period_start].copy()
    if len(window) < 30:
        return None

    data = window.copy()
    data["MACD"], data["Signal_Line"] = calculate_macd(data["Adj Close"])
    data["Upper"], data["Lower"] = calculate_bollinger_bands(data["Adj Close"])
    data.dropna(inplace=True)
    if data.empty:
        return None

    pct = pct_returns_from_start(data["Adj Close"])
    macd_sig = pd.Series(
        np.where(data["MACD"] > data["Signal_Line"], 1, 0), index=data.index
    )
    equity = (1 + macd_sig.shift(1) * data["Adj Close"].pct_change(fill_method=None)).cumprod()
    portfolio = (equity * initial_capital).round(4).tolist()

    markers: list[TradeMarker] = []
    bb_events = _bollinger_trades(
        data["Adj Close"].values, data["Lower"].values, data["Upper"].values
    )
    for idx, side, price in bb_events:
        markers.append(
            TradeMarker(
                x=idx,
                y=float(pct.iloc[idx]),
                side=side,
                tooltip=build_marker_tooltip(
                    side="Buy" if side == "buy" else "Sell",
                    date=data.index[idx],
                    price=float(price),
                    triggers="Bollinger Band",
                ),
            )
        )

    macd_events = _macd_crossovers(data["MACD"].values, data["Signal_Line"].values)
    for idx, side in macd_events:
        markers.append(
            TradeMarker(
                x=idx,
                y=float(pct.iloc[idx]),
                side=side,
                tooltip=build_marker_tooltip(
                    side="Buy" if side == "buy" else "Sell",
                    date=data.index[idx],
                    price=float(data["Adj Close"].iloc[idx]),
                    triggers="MACD crossover",
                ),
            )
        )

    return TickerChartSeries(
        ticker=ticker,
        dates=[d.strftime("%Y-%m-%d") for d in data.index],
        pct_returns=pct.round(4).tolist(),
        portfolio_values=portfolio,
        markers=markers,
    )


def worker_technical_analysis(args):
    ticker, df, lookback_days = args
    period_start, period_end = period_bounds(df, lookback_days)
    window = df[df.index >= period_start]
    if len(window) < 100:
        return None
    try:
        start = window["Adj Close"].iloc[0]
        end = window["Adj Close"].iloc[-1]
        baseline = end / start if start != 0 else 0
        macd_res = strat_macd(window)
        bb_res = strat_bollinger(window)
        return {
            "Ticker": ticker,
            "Baseline": baseline,
            "MACD": macd_res,
            "Bollinger": bb_res,
            "Best_Tech": max(macd_res, bb_res),
            "Period_Start": period_start,
            "Period_End": period_end,
            "Full_DF": df,
        }
    except Exception:
        return None


def run_technical_analysis(
    db_path: str,
    tickers: list[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    initial_capital: float = 1.0,
    progress_callback: ProgressCallback | None = None,
    use_parallel: bool = True,
    save_to_db: bool = True,
) -> TechnicalResult | None:
    all_data = load_entire_database(db_path)
    if all_data is None:
        return None

    tasks = []
    for ticker, df in all_data.groupby("Ticker"):
        if tickers and ticker not in tickers:
            continue
        tasks.append((ticker, df[["Adj Close"]], lookback_days))

    results = []
    raw_results: list[dict] = []
    if progress_callback:
        progress_callback(0.0 if tasks else 1.0)
    if use_parallel and len(tasks) > 1:
        for i, res in enumerate(
            run_parallel_map(worker_technical_analysis, tasks, use_parallel=True)
        ):
            if res:
                raw_results.append(res)
            if progress_callback:
                progress_callback((i + 1) / len(tasks))
    else:
        for i, task in enumerate(tasks):
            res = worker_technical_analysis(task)
            if res:
                raw_results.append(res)
            if progress_callback:
                progress_callback((i + 1) / len(tasks))

    if not raw_results:
        return TechnicalResult(lookback_days=lookback_days)

    period_start = raw_results[0].get("Period_Start")
    period_end = raw_results[0].get("Period_End")
    results = [
        {k: v for k, v in r.items() if k not in ("Period_Start", "Period_End", "Full_DF")}
        for r in raw_results
    ]

    res_df = pd.DataFrame(results)
    res_df["Winner"] = res_df.apply(
        lambda row: "MACD" if row["MACD"] > row["Bollinger"] else "Bollinger", axis=1
    )
    res_df["Alpha"] = res_df["Best_Tech"] - res_df["Baseline"]

    if save_to_db:
        try:
            with db_connection(db_path, readonly=False) as conn:
                res_df.to_sql("strategy_rankings", conn, if_exists="replace", index=False)
                conn.commit()
        except sqlite3.Error:
            pass

    chart_series: dict[str, TickerChartSeries] = {}
    if tickers:
        capital_each = initial_capital / len(raw_results) if raw_results else initial_capital
        for row in raw_results:
            full_df = row.get("Full_DF")
            if full_df is None:
                continue
            series = _build_technical_chart(
                row["Ticker"], full_df, lookback_days, capital_each
            )
            if series:
                chart_series[row["Ticker"]] = series

    combined_dates, combined_values = combine_portfolio_series(chart_series)

    return TechnicalResult(
        results_df=res_df,
        avg_baseline=float(res_df["Baseline"].mean()),
        avg_macd=float(res_df["MACD"].mean()),
        avg_bollinger=float(res_df["Bollinger"].mean()),
        stock_count=len(res_df),
        period_label=format_period(period_start, period_end),
        lookback_days=lookback_days,
        chart_series=chart_series,
        combined_portfolio_dates=combined_dates,
        combined_portfolio_values=combined_values,
    )
