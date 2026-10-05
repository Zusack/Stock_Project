"""Adapt CANSLIM backtest output to unified BacktestResult contract."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.analysis.backtest_engine import BacktestResult, BacktestTrade
from src.analysis.backtest_metrics import compute_metrics, monthly_returns_grid
from src.analysis.canslim_backtest import run_canslim_backtest
from src.analysis.strategy_spec import StrategySpec
from src.services.stock_config import stock_config

ProgressCallback = Callable[[float], None]


def canslim_to_backtest_result(
    spec: StrategySpec,
    canslim_res,
    *,
    benchmark_dates: list[str] | None = None,
    benchmark_values: list[float] | None = None,
) -> BacktestResult:
    """Convert CanslimBacktestResult into BacktestResult."""
    trades: list[BacktestTrade] = []
    if canslim_res.trades_df is not None and not canslim_res.trades_df.empty:
        for _, row in canslim_res.trades_df.iterrows():
            ret_str = str(row.get("Return_Pct", "0")).replace("%", "")
            try:
                ret_pct = float(ret_str)
            except ValueError:
                ret_pct = 0.0
            entry_d = str(row.get("Entry_Date", ""))
            exit_d = str(row.get("Exit_Date", ""))
            holding = 0
            try:
                holding = (pd.Timestamp(exit_d) - pd.Timestamp(entry_d)).days
            except (TypeError, ValueError):
                pass
            trades.append(
                BacktestTrade(
                    ticker=str(row.get("Ticker", "")),
                    entry_date=entry_d,
                    exit_date=exit_d,
                    entry_price=float(row.get("Entry_Price") or 0),
                    exit_price=float(row.get("Exit_Price") or 0),
                    return_pct=ret_pct,
                    holding_days=holding,
                    exit_reason=str(row.get("Reason") or row.get("Sell_Rule") or ""),
                )
            )

    equity_dates = canslim_res.combined_portfolio_dates
    equity_values = canslim_res.combined_portfolio_values

    dd_pct: list[float] = []
    if equity_values:
        eq = pd.Series(equity_values)
        peak = eq.cummax()
        dd_pct = ((eq - peak) / peak.replace(0, float("nan")) * 100.0).fillna(0).tolist()

    metrics = compute_metrics(
        equity_dates,
        equity_values,
        trades=[t.to_dict() for t in trades],
        benchmark_dates=benchmark_dates,
        benchmark_values=benchmark_values,
    )
    if canslim_res.total_trades > 0:
        metrics.win_rate_pct = canslim_res.win_rate
        metrics.trade_count = canslim_res.total_trades

    summary = (
        f"CANSLIM: {canslim_res.total_trades} trades, "
        f"win rate {canslim_res.win_rate:.1f}%, "
        f"avg return {canslim_res.avg_return_pct:.2f}%. "
        f"{canslim_res.fundamentals_note or ''}"
    ).strip()

    return BacktestResult(
        spec=spec,
        metrics=metrics,
        trades=trades,
        equity_dates=equity_dates,
        equity_values=equity_values,
        benchmark_dates=benchmark_dates or [],
        benchmark_values=benchmark_values or [],
        drawdown_pct=dd_pct,
        chart_series=canslim_res.chart_series,
        monthly_returns=monthly_returns_grid(equity_dates, equity_values),
        period_label=canslim_res.period_label,
        summary=summary,
    )


def run_canslim_backtest_unified(
    spec: StrategySpec,
    *,
    db_path: str,
    tickers: list[str] | None,
    lookback_days: int,
    initial_capital: float,
    use_parallel: bool,
    progress_callback: ProgressCallback | None = None,
    cancel_event=None,
) -> BacktestResult | None:
    cfg = stock_config()
    # StrategySpec / UI store stop & take-profit as positive magnitudes.
    # CANSLIM ExitRules use a signed stop (negative); rule_set_from_config
    # normalizes either sign.
    raw_stop = spec.risk.stop_loss_pct if spec.risk.stop_loss_pct is not None else 0.08
    raw_tp = spec.risk.take_profit_pct if spec.risk.take_profit_pct is not None else 0.25
    res = run_canslim_backtest(
        db_path,
        market_ticker=spec.market_ticker or cfg.market_ticker,
        stop_loss=raw_stop,
        take_profit=raw_tp,
        tickers=tickers,
        lookback_days=lookback_days,
        initial_capital=initial_capital,
        use_parallel=use_parallel,
        rule_set_version=cfg.canslim_rule_set_version,
        require_pattern=cfg.canslim_require_pattern,
        progress_callback=progress_callback,
    )
    if res is None:
        return None
    return canslim_to_backtest_result(spec, res)
