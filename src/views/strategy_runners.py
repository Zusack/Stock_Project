"""Per-strategy backtest runners (extracted from StrategyBacktestView)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.analysis.canslim_backtest import run_canslim_backtest
from src.analysis.db import resolve_market_ticker
from src.analysis.general import evaluate_general_rules
from src.analysis.hybrid import run_hybrid_analysis
from src.analysis.technical import run_technical_analysis
from src.services.stock_config import stock_config
from src.utils.format_utils import format_percent


@dataclass
class StrategyRunOutcome:
    summary: str
    df: pd.DataFrame | None = None
    trades_df: pd.DataFrame | None = None
    chart_series: dict | None = None
    portfolio_dates: list[str] | None = None
    portfolio_values: list[float] | None = None


ProgressFn = Callable[[str, float | None, bool], None]


def run_general_strategy(
    *,
    db_path: str,
    tickers: list[str] | None,
    lookback_days: int,
    initial_capital: float,
    use_parallel: bool,
    progress: ProgressFn,
) -> StrategyRunOutcome:
    progress("Step 1/2: Loading database and evaluating general rules per ticker…", None, True)
    res = evaluate_general_rules(
        db_path,
        tickers=tickers,
        lookback_days=lookback_days,
        initial_capital=initial_capital,
        use_parallel=use_parallel,
        progress_callback=lambda p, *_: progress(f"Step 1/2: {int(p * 100)}% complete", p, True),
    )
    progress("Step 2/2: Aggregating strategy averages…", 1.0, True)
    if res is None:
        return StrategyRunOutcome(summary="No data in database.")
    avgs = ", ".join(f"{k}: {v:.2f}x" for k, v in res.strategy_averages.items())
    chart_note = f" Chart: {res.best_strategy} signals." if res.chart_series else ""
    summary = (
        f"Period: {res.period_label} ({lookback_days}d). "
        f"Start ${initial_capital:.2f}. "
        f"Stocks: {res.stock_count}. Winner: {res.winner}. {avgs}.{chart_note}"
    )
    stock_config().set_last_analysis_label("General rules backtest")
    return StrategyRunOutcome(
        summary=summary,
        df=res.per_ticker,
        chart_series=res.chart_series or None,
        portfolio_dates=res.combined_portfolio_dates,
        portfolio_values=res.combined_portfolio_values,
    )


def run_hybrid_strategy(
    *,
    db_path: str,
    tickers: list[str] | None,
    lookback_days: int,
    initial_capital: float,
    use_parallel: bool,
    market_trend_weeks: int,
    progress: ProgressFn,
) -> StrategyRunOutcome:
    cfg = stock_config()
    market = resolve_market_ticker(db_path, cfg.market_ticker)
    progress(
        f"Step 1/2: Loading database and running hybrid analysis (market: {market})…",
        None,
        True,
    )
    res = run_hybrid_analysis(
        db_path,
        market,
        market_trend_weeks=market_trend_weeks,
        tickers=tickers,
        lookback_days=lookback_days,
        initial_capital=initial_capital,
        use_parallel=use_parallel,
        progress_callback=lambda p, *_: progress(f"Step 1/2: {int(p * 100)}% complete", p, True),
    )
    progress("Step 2/2: Computing win rate and averages…", 1.0, True)
    if res is None:
        return StrategyRunOutcome(summary="No data or market ticker missing.")
    summary = (
        f"Period: {res.period_label} ({lookback_days}d). "
        f"Start ${initial_capital:.2f}. "
        f"Market: {res.market_ticker}. Stocks: {res.stock_count}. "
        f"Win rate {format_percent(res.win_rate, decimals=1)}. "
        f"Avg standalone: {res.avg_standalone:.2f}x, hybrid: {res.avg_hybrid:.2f}x. "
        "Markers: hybrid signal on/off."
    )
    stock_config().set_last_analysis_label("Hybrid backtest")
    return StrategyRunOutcome(
        summary=summary,
        df=res.results_df,
        chart_series=res.chart_series or None,
        portfolio_dates=res.combined_portfolio_dates,
        portfolio_values=res.combined_portfolio_values,
    )


def run_technical_strategy(
    *,
    db_path: str,
    tickers: list[str] | None,
    lookback_days: int,
    initial_capital: float,
    use_parallel: bool,
    progress: ProgressFn,
) -> StrategyRunOutcome:
    progress("Step 1/2: Loading database and running MACD / Bollinger per ticker…", None, True)
    res = run_technical_analysis(
        db_path,
        tickers=tickers,
        lookback_days=lookback_days,
        initial_capital=initial_capital,
        use_parallel=use_parallel,
        save_to_db=False,
        progress_callback=lambda p, *_: progress(f"Step 1/2: {int(p * 100)}% complete", p, True),
    )
    progress("Step 2/2: Ranking results by alpha…", 1.0, True)
    if res is None:
        return StrategyRunOutcome(summary="No data.")
    summary = (
        f"Period: {res.period_label} ({lookback_days}d). "
        f"Start ${initial_capital:.2f}. "
        f"Stocks: {res.stock_count}. "
        f"Avg baseline {res.avg_baseline:.2f}x, "
        f"MACD {res.avg_macd:.2f}x, Bollinger {res.avg_bollinger:.2f}x. "
        "Markers: MACD crossovers and Bollinger trades."
    )
    stock_config().set_last_analysis_label("Technical backtest")
    return StrategyRunOutcome(
        summary=summary,
        df=res.results_df.sort_values(by="Alpha", ascending=False),
        chart_series=res.chart_series or None,
        portfolio_dates=res.combined_portfolio_dates,
        portfolio_values=res.combined_portfolio_values,
    )


def run_canslim_strategy(
    *,
    db_path: str,
    tickers: list[str] | None,
    lookback_days: int,
    initial_capital: float,
    use_parallel: bool,
    stop_loss: float,
    take_profit: float,
    progress: ProgressFn,
) -> StrategyRunOutcome:
    cfg = stock_config()
    progress(
        f"Step 1/2: Loading market ({cfg.market_ticker}) and simulating CANSLIM trades…",
        None,
        True,
    )
    res = run_canslim_backtest(
        db_path,
        market_ticker=cfg.market_ticker,
        stop_loss=stop_loss,
        take_profit=take_profit,
        tickers=tickers,
        lookback_days=lookback_days,
        initial_capital=initial_capital,
        use_parallel=use_parallel,
        rule_set_version=cfg.canslim_rule_set_version,
        require_pattern=cfg.canslim_require_pattern,
        progress_callback=lambda p, *_: progress(f"Step 1/2: {int(p * 100)}% complete", p, True),
    )
    progress("Step 2/2: Computing trade statistics…", 1.0, True)
    if res is None:
        return StrategyRunOutcome(summary="No market data for backtest.")
    if res.total_trades == 0:
        summary = (
            f"Period: {res.period_label} ({lookback_days}d). "
            f"Start ${initial_capital:.2f}. No trades generated in this window."
        )
        return StrategyRunOutcome(
            summary=summary,
            chart_series=res.chart_series or None,
            portfolio_dates=res.combined_portfolio_dates,
            portfolio_values=res.combined_portfolio_values,
        )
    fund = res.fundamentals_note or ""
    summary = (
        f"Period: {res.period_label} ({lookback_days}d). "
        f"Start ${initial_capital:.2f} (partial shares). "
        f"Stop {stop_loss:.0%}, take profit {take_profit:.0%}. "
        f"Trades: {res.total_trades}, win rate {format_percent(res.win_rate, decimals=1)}, "
        f"avg return {format_percent(res.avg_return_pct, decimals=2)}. "
        "Entry on breakout (setup + volume surge); exits: stop, "
        "take profit, below 50-day simple moving average, or market downtrend. "
        f"{fund}"
    )
    stock_config().set_last_analysis_label("CANSLIM backtest")
    return StrategyRunOutcome(
        summary=summary,
        df=res.ticker_stats,
        trades_df=res.trades_df,
        chart_series=res.chart_series or None,
        portfolio_dates=res.combined_portfolio_dates,
        portfolio_values=res.combined_portfolio_values,
    )
