"""Hybrid market-filtered strategy analysis."""

from __future__ import annotations

from src.analysis.parallel_exec import run_parallel_map
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
from src.analysis.db import load_entire_database

ProgressCallback = Callable[[float], None]


@dataclass
class HybridResult:
    results_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    win_rate: float = 0.0
    avg_standalone: float = 0.0
    avg_hybrid: float = 0.0
    stock_count: int = 0
    market_ticker: str = ""
    period_label: str = ""
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    chart_series: dict[str, TickerChartSeries] = field(default_factory=dict)
    combined_portfolio_dates: list[str] = field(default_factory=list)
    combined_portfolio_values: list[float] = field(default_factory=list)


def _signal_markers(
    dates: pd.DatetimeIndex,
    pct: pd.Series,
    signal: pd.Series,
    label: str,
) -> list[TradeMarker]:
    markers: list[TradeMarker] = []
    prev = 0
    date_to_x = {d: i for i, d in enumerate(dates)}
    for date, val in signal.items():
        cur = int(val)
        if cur != prev:
            side = "buy" if cur == 1 else "sell"
            x = date_to_x.get(date)
            if x is not None:
                markers.append(
                    TradeMarker(
                        x=x,
                        y=float(pct.iloc[x]),
                        side=side,
                        tooltip=build_marker_tooltip(
                            side="Enter" if cur == 1 else "Exit",
                            date=pd.Timestamp(date),
                            price=0.0,
                            triggers=f"{label} signal {'on' if cur == 1 else 'off'}",
                        ),
                    )
                )
        prev = cur
    return markers


def _build_hybrid_chart(
    ticker: str,
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    market_trend_weeks: int,
    stock_sma_window: int,
    lookback_days: int,
    initial_capital: float,
) -> TickerChartSeries | None:
    s = stock_df.rename(columns={"Adj Close": "Price_Stock"})
    m = market_df.rename(columns={"Adj Close": "Price_Market"})
    combined = s.join(m, how="inner")
    period_start, _ = period_bounds(combined, lookback_days)
    combined = combined[combined.index >= period_start]
    df = combined.copy()
    df["SMA_Stock"] = df["Price_Stock"].rolling(window=stock_sma_window).mean()
    df["SMA_Market"] = df["Price_Market"].rolling(window=market_trend_weeks * 5).mean()
    df.dropna(inplace=True)
    if df.empty:
        return None

    df["Signal_Hybrid"] = np.where(
        (df["Price_Market"] > df["SMA_Market"]) & (df["Price_Stock"] > df["SMA_Stock"]),
        1,
        0,
    )
    pct = pct_returns_from_start(df["Price_Stock"])
    equity = (1 + df["Signal_Hybrid"].shift(1) * df["Price_Stock"].pct_change(fill_method=None)).cumprod()
    portfolio = (equity * initial_capital).round(4).tolist()
    markers = _signal_markers(df.index, pct, df["Signal_Hybrid"], "Hybrid")

    return TickerChartSeries(
        ticker=ticker,
        dates=[d.strftime("%Y-%m-%d") for d in df.index],
        pct_returns=pct.round(4).tolist(),
        portfolio_values=portfolio,
        markers=markers,
    )


def process_stock_wrapper(args):
    ticker, stock_df, market_df, market_trend_weeks, stock_sma_window, lookback_days = args
    s = stock_df.rename(columns={"Adj Close": "Price_Stock"})
    m = market_df.rename(columns={"Adj Close": "Price_Market"})
    combined = s.join(m, how="inner")
    period_start, period_end = period_bounds(combined, lookback_days)
    combined = combined[combined.index >= period_start]
    df = combined.copy()
    df["SMA_Stock"] = df["Price_Stock"].rolling(window=stock_sma_window).mean()
    market_lookback_days = market_trend_weeks * 5
    df["SMA_Market"] = df["Price_Market"].rolling(window=market_lookback_days).mean()
    df.dropna(inplace=True)
    if df.empty:
        return None

    df["Signal_Standalone"] = np.where(df["Price_Stock"] > df["SMA_Stock"], 1, 0)
    market_condition = df["Price_Market"] > df["SMA_Market"]
    stock_condition = df["Price_Stock"] > df["SMA_Stock"]
    df["Signal_Hybrid"] = np.where(market_condition & stock_condition, 1, 0)
    df["Pct_Change"] = df["Price_Stock"].pct_change(fill_method=None)
    df["Ret_Standalone"] = df["Signal_Standalone"].shift(1) * df["Pct_Change"]
    df["Ret_Hybrid"] = df["Signal_Hybrid"].shift(1) * df["Pct_Change"]

    return {
        "Ticker": ticker,
        "Standalone": float((1 + df["Ret_Standalone"]).prod()),
        "Hybrid": float((1 + df["Ret_Hybrid"]).prod()),
        "Diff": float((1 + df["Ret_Hybrid"]).prod() - (1 + df["Ret_Standalone"]).prod()),
        "Period_Start": period_start,
        "Period_End": period_end,
        "Stock_DF": stock_df,
        "Market_DF": market_df,
    }


def run_hybrid_analysis(
    db_path: str,
    market_ticker: str,
    market_trend_weeks: int = 20,
    stock_sma_window: int = 50,
    tickers: list[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    initial_capital: float = 1.0,
    progress_callback: ProgressCallback | None = None,
    use_parallel: bool = True,
) -> HybridResult | None:
    all_data = load_entire_database(db_path)
    if all_data is None:
        return None

    market_df = all_data[all_data["Ticker"] == market_ticker][["Adj Close"]]
    if market_df.empty:
        return None

    grouped = list(all_data.groupby("Ticker"))
    tasks = []
    for ticker, group_df in grouped:
        if ticker == market_ticker:
            continue
        if tickers and ticker not in tickers:
            continue
        tasks.append(
            (
                ticker,
                group_df[["Adj Close"]],
                market_df,
                market_trend_weeks,
                stock_sma_window,
                lookback_days,
            )
        )

    results = []
    raw_results: list[dict] = []
    if progress_callback:
        progress_callback(0.0 if tasks else 1.0)
    if use_parallel and len(tasks) > 1:
        for i, res in enumerate(
            run_parallel_map(process_stock_wrapper, tasks, use_parallel=True)
        ):
            if res:
                raw_results.append(res)
            if progress_callback:
                progress_callback((i + 1) / len(tasks))
    else:
        for i, task in enumerate(tasks):
            res = process_stock_wrapper(task)
            if res:
                raw_results.append(res)
            if progress_callback:
                progress_callback((i + 1) / len(tasks))

    if not raw_results:
        return HybridResult(market_ticker=market_ticker, lookback_days=lookback_days)

    period_start = raw_results[0].get("Period_Start")
    period_end = raw_results[0].get("Period_End")
    results = [
        {k: v for k, v in r.items() if k not in ("Period_Start", "Period_End", "Stock_DF", "Market_DF")}
        for r in raw_results
    ]

    res_df = pd.DataFrame(results)
    wins = res_df[res_df["Hybrid"] >= res_df["Standalone"]]
    win_rate = len(wins) / len(res_df) * 100 if len(res_df) else 0.0

    chart_series: dict[str, TickerChartSeries] = {}
    if tickers:
        capital_each = initial_capital / len(raw_results) if raw_results else initial_capital
        for row in raw_results:
            series = _build_hybrid_chart(
                row["Ticker"],
                row["Stock_DF"],
                row["Market_DF"],
                market_trend_weeks,
                stock_sma_window,
                lookback_days,
                capital_each,
            )
            if series:
                chart_series[row["Ticker"]] = series

    combined_dates, combined_values = combine_portfolio_series(chart_series)

    return HybridResult(
        results_df=res_df,
        win_rate=win_rate,
        avg_standalone=float(res_df["Standalone"].mean()),
        avg_hybrid=float(res_df["Hybrid"].mean()),
        stock_count=len(res_df),
        market_ticker=market_ticker,
        period_label=format_period(period_start, period_end),
        lookback_days=lookback_days,
        chart_series=chart_series,
        combined_portfolio_dates=combined_dates,
        combined_portfolio_values=combined_values,
    )


def run_hybrid_single_ticker(
    db_path: str,
    ticker: str,
    market_ticker: str,
    market_trend_weeks: int = 20,
    stock_sma_window: int = 50,
) -> dict | None:
    """Run hybrid backtest for one ticker; returns equity curve series."""
    stock_df = load_entire_database(db_path)
    if stock_df is None:
        return None
    stock = stock_df[stock_df["Ticker"] == ticker.upper()][["Adj Close"]]
    market = stock_df[stock_df["Ticker"] == market_ticker][["Adj Close"]]
    if stock.empty or market.empty:
        return None

    res = process_stock_wrapper(
        (
            ticker.upper(),
            stock,
            market,
            market_trend_weeks,
            stock_sma_window,
            DEFAULT_LOOKBACK_DAYS,
        )
    )
    if not res:
        return None

    s = stock.rename(columns={"Adj Close": "Price_Stock"})
    m = market.rename(columns={"Adj Close": "Price_Market"})
    combined = s.join(m, how="inner")
    df = combined.copy()
    df["SMA_Stock"] = df["Price_Stock"].rolling(window=stock_sma_window).mean()
    df["SMA_Market"] = df["Price_Market"].rolling(
        window=market_trend_weeks * 5
    ).mean()
    df.dropna(inplace=True)
    df["Signal_Standalone"] = np.where(df["Price_Stock"] > df["SMA_Stock"], 1, 0)
    df["Signal_Hybrid"] = np.where(
        (df["Price_Market"] > df["SMA_Market"]) & (df["Price_Stock"] > df["SMA_Stock"]),
        1,
        0,
    )
    df["Pct_Change"] = df["Price_Stock"].pct_change(fill_method=None)
    df["Equity_Standalone"] = (1 + df["Signal_Standalone"].shift(1) * df["Pct_Change"]).cumprod()
    df["Equity_Hybrid"] = (1 + df["Signal_Hybrid"].shift(1) * df["Pct_Change"]).cumprod()

    return {
        "summary": res,
        "equity": df[["Equity_Standalone", "Equity_Hybrid", "Price_Stock"]].dropna(),
    }
