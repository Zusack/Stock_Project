"""General strategy comparison across the universe."""

from __future__ import annotations

from src.analysis.parallel_exec import run_parallel_map
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.backtest_common import (
    DEFAULT_LOOKBACK_DAYS,
    TickerChartSeries,
    format_period,
    pct_returns_from_start,
    period_bounds,
)
from src.analysis.db import load_entire_database

ProgressCallback = Callable[[float], None]


@dataclass
class GeneralResult:
    strategy_averages: dict[str, float] = field(default_factory=dict)
    winner: str | None = None
    stock_count: int = 0
    per_ticker: pd.DataFrame | None = None
    period_label: str = ""
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    chart_series: dict[str, TickerChartSeries] = field(default_factory=dict)
    combined_portfolio_dates: list[str] = field(default_factory=list)
    combined_portfolio_values: list[float] = field(default_factory=list)
    best_strategy: str | None = None


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def strat_golden_cross(df: pd.DataFrame) -> float:
    data = df.copy()
    data["SMA50"] = data["Adj Close"].rolling(window=50).mean()
    data["SMA200"] = data["Adj Close"].rolling(window=200).mean()
    data.dropna(inplace=True)
    if data.empty:
        return 0.0
    data["Signal"] = np.where(data["SMA50"] > data["SMA200"], 1, 0)
    data["Strategy_Return"] = data["Signal"].shift(1) * data["Adj Close"].pct_change(fill_method=None)
    return float((1 + data["Strategy_Return"]).prod())


def strat_200_trend(df: pd.DataFrame) -> float:
    data = df.copy()
    data["SMA200"] = data["Adj Close"].rolling(window=200).mean()
    data.dropna(inplace=True)
    if data.empty:
        return 0.0
    data["Signal"] = np.where(data["Adj Close"] > data["SMA200"], 1, 0)
    data["Strategy_Return"] = data["Signal"].shift(1) * data["Adj Close"].pct_change(fill_method=None)
    return float((1 + data["Strategy_Return"]).prod())


def strat_rsi_dip(df: pd.DataFrame) -> float:
    data = df.copy()
    data["RSI"] = calculate_rsi(data["Adj Close"])
    data.dropna(inplace=True)
    if data.empty:
        return 0.0
    signal = 0
    signals = []
    for rsi in data["RSI"]:
        if rsi < 30:
            signal = 1
        elif rsi > 50:
            signal = 0
        signals.append(signal)
    data["Signal"] = signals
    data["Strategy_Return"] = data["Signal"].shift(1) * data["Adj Close"].pct_change(fill_method=None)
    return float((1 + data["Strategy_Return"]).prod())


def _equity_curve(df: pd.DataFrame, signal: pd.Series) -> pd.Series:
    returns = signal.shift(1) * df["Adj Close"].pct_change(fill_method=None)
    return (1 + returns.fillna(0)).cumprod()


def _build_general_chart(
    ticker: str,
    df: pd.DataFrame,
    lookback_days: int,
    initial_capital: float,
    strategy: str,
) -> TickerChartSeries | None:
    period_start, _ = period_bounds(df, lookback_days)
    window = df[df.index >= period_start]
    if window.empty:
        return None

    data = window.copy()
    if strategy == "Golden Cross":
        data["SMA50"] = data["Adj Close"].rolling(window=50).mean()
        data["SMA200"] = data["Adj Close"].rolling(window=200).mean()
        data.dropna(inplace=True)
        if data.empty:
            return None
        signal = pd.Series(np.where(data["SMA50"] > data["SMA200"], 1, 0), index=data.index)
    elif strategy == "200-Day Trend":
        data["SMA200"] = data["Adj Close"].rolling(window=200).mean()
        data.dropna(inplace=True)
        if data.empty:
            return None
        signal = pd.Series(np.where(data["Adj Close"] > data["SMA200"], 1, 0), index=data.index)
    elif strategy == "RSI Dip Buy":
        data["RSI"] = calculate_rsi(data["Adj Close"])
        data.dropna(inplace=True)
        if data.empty:
            return None
        pos = 0
        signals = []
        for rsi in data["RSI"]:
            if rsi < 30:
                pos = 1
            elif rsi > 50:
                pos = 0
            signals.append(pos)
        signal = pd.Series(signals, index=data.index)
    else:
        signal = pd.Series(1, index=data.index)

    pct = pct_returns_from_start(data["Adj Close"])
    equity = _equity_curve(data, signal)
    portfolio = (equity * initial_capital).round(4).tolist()

    return TickerChartSeries(
        ticker=ticker,
        dates=[d.strftime("%Y-%m-%d") for d in data.index],
        pct_returns=pct.round(4).tolist(),
        portfolio_values=[round(v, 4) for v in portfolio],
        markers=[],
    )


def process_strategies_wrapper(args):
    ticker, df, lookback_days = args
    period_start, period_end = period_bounds(df, lookback_days)
    window = df[df.index >= period_start]
    if len(window) < 50:
        return None
    try:
        start_price = window["Adj Close"].iloc[0]
        end_price = window["Adj Close"].iloc[-1]
        bh_ret = end_price / start_price if start_price != 0 else 0
        return {
            "Ticker": ticker,
            "Buy & Hold": bh_ret,
            "Golden Cross": strat_golden_cross(window),
            "200-Day Trend": strat_200_trend(window),
            "RSI Dip Buy": strat_rsi_dip(window),
            "Period_Start": period_start,
            "Period_End": period_end,
            "Full_DF": df,
        }
    except Exception:
        return None


def evaluate_general_rules(
    db_path: str,
    tickers: list[str] | None = None,
    min_history: int = 250,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    initial_capital: float = 1.0,
    progress_callback: ProgressCallback | None = None,
    use_parallel: bool = True,
) -> GeneralResult | None:
    all_data = load_entire_database(db_path)
    if all_data is None:
        return None

    grouped = list(all_data.groupby("Ticker"))
    tasks = []
    for ticker, group_df in grouped:
        if tickers and ticker not in tickers:
            continue
        if len(group_df) < min_history:
            continue
        tasks.append((ticker, group_df[["Adj Close"]], lookback_days))

    per_ticker_rows = []
    raw_rows: list[dict] = []
    if progress_callback:
        progress_callback(0.0 if tasks else 1.0)
    if use_parallel and len(tasks) > 1:
        for i, res in enumerate(
            run_parallel_map(process_strategies_wrapper, tasks, use_parallel=True)
        ):
            if res:
                raw_rows.append(res)
            if progress_callback:
                progress_callback((i + 1) / len(tasks))
    else:
        for i, task in enumerate(tasks):
            res = process_strategies_wrapper(task)
            if res:
                raw_rows.append(res)
            if progress_callback:
                progress_callback((i + 1) / len(tasks))

    if not raw_rows:
        return GeneralResult(stock_count=0, lookback_days=lookback_days)

    period_start = raw_rows[0].get("Period_Start")
    period_end = raw_rows[0].get("Period_End")
    per_ticker_rows = [
        {k: v for k, v in row.items() if k not in ("Period_Start", "Period_End", "Full_DF")}
        for row in raw_rows
    ]

    per_df = pd.DataFrame(per_ticker_rows)
    strategy_cols = ["Buy & Hold", "Golden Cross", "200-Day Trend", "RSI Dip Buy"]
    averages = {s: float(per_df[s].mean()) for s in strategy_cols if s in per_df.columns}

    best_strat = None
    best_perf = -1.0
    for strategy, avg in averages.items():
        if strategy != "Buy & Hold" and avg > best_perf:
            best_perf = avg
            best_strat = strategy

    chart_series: dict[str, TickerChartSeries] = {}
    if tickers:
        chart_strategy = best_strat or "Golden Cross"
        capital_each = initial_capital / len(raw_rows) if raw_rows else initial_capital
        for row in raw_rows:
            ticker = row["Ticker"]
            full_df = row.get("Full_DF")
            if full_df is None:
                continue
            series = _build_general_chart(
                ticker, full_df, lookback_days, capital_each, chart_strategy
            )
            if series:
                chart_series[ticker] = series

    from src.analysis.backtest_common import combine_portfolio_series

    combined_dates, combined_values = combine_portfolio_series(chart_series)

    return GeneralResult(
        strategy_averages=averages,
        winner=best_strat,
        stock_count=len(per_df),
        per_ticker=per_df,
        period_label=format_period(period_start, period_end),
        lookback_days=lookback_days,
        chart_series=chart_series,
        combined_portfolio_dates=combined_dates,
        combined_portfolio_values=combined_values,
        best_strategy=best_strat,
    )
