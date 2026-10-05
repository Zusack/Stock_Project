"""Unified backtest runners for Strategy Backtests tab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.analysis.backtest_engine import BacktestCancelled, BacktestResult, run_backtest
from src.analysis.backtest_schema import save_backtest_run
from src.analysis.canslim_adapter import run_canslim_backtest_unified
from src.analysis.db import load_market_data, resolve_market_ticker
from src.analysis.strategy_spec import StrategySpec, validate_spec
from src.services.stock_config import stock_config

ProgressFn = Callable[[str, float | None, bool], None]


@dataclass
class CompareRunResult:
    results: list[BacktestResult] = field(default_factory=list)
    summary: str = ""
    cancelled: bool = False


def run_strategy_backtest(
    spec: StrategySpec,
    *,
    db_path: str,
    tickers: list[str] | None,
    lookback_days: int,
    initial_capital: float,
    benchmark_ticker: str | None = None,
    use_parallel: bool = True,
    progress: ProgressFn | None = None,
    cancel_event=None,
    save_run: bool = True,
) -> BacktestResult | None:
    """Dispatch to unified or CANSLIM engine based on spec.engine."""
    errors = validate_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))

    cfg = stock_config()
    bench = benchmark_ticker or spec.market_ticker or cfg.market_ticker

    def _prog(p: float) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise BacktestCancelled("Cancelled by user.")
        if progress:
            progress(f"Running backtest… {int(p * 100)}%", p, True)

    if spec.engine == "canslim":
        result = run_canslim_backtest_unified(
            spec,
            db_path=db_path,
            tickers=tickers,
            lookback_days=lookback_days,
            initial_capital=initial_capital,
            use_parallel=use_parallel,
            progress_callback=_prog,
            cancel_event=cancel_event,
        )
    else:
        result = run_backtest(
            spec,
            db_path=db_path,
            tickers=tickers,
            lookback_days=lookback_days,
            initial_capital=initial_capital,
            benchmark_ticker=bench,
            progress_callback=_prog,
            cancel_event=cancel_event,
        )

    if cancel_event is not None and cancel_event.is_set():
        raise BacktestCancelled("Cancelled by user.")

    if result is None:
        return None

    if save_run:
        try:
            save_backtest_run(
                db_path,
                spec=spec,
                metrics=result.metrics.to_dict(),
                tickers=result.tickers_run,
                lookback_days=lookback_days,
                initial_capital=initial_capital,
            )
        except Exception:
            pass

    stock_config().set_last_analysis_label(f"Backtest: {spec.name}")
    return result


def run_compare_backtests(
    specs: list[StrategySpec],
    *,
    db_path: str,
    tickers: list[str] | None,
    lookback_days: int,
    initial_capital: float,
    benchmark_ticker: str | None = None,
    use_parallel: bool = True,
    progress: ProgressFn | None = None,
    cancel_event=None,
) -> CompareRunResult:
    """Run multiple strategies on the same universe for side-by-side comparison."""
    results: list[BacktestResult] = []
    total = len(specs)
    cancelled = False
    for i, spec in enumerate(specs):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        def _prog(p: float, _i=i, _name=spec.name) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise BacktestCancelled("Cancelled by user.")
            if progress:
                overall = (_i + p) / max(total, 1)
                progress(
                    f"Comparing strategies ({_i + 1}/{total}): {_name}… {int(p * 100)}%",
                    overall,
                    True,
                )

        try:
            if spec.engine == "canslim":
                res = run_canslim_backtest_unified(
                    spec,
                    db_path=db_path,
                    tickers=tickers,
                    lookback_days=lookback_days,
                    initial_capital=initial_capital,
                    use_parallel=use_parallel,
                    progress_callback=_prog,
                    cancel_event=cancel_event,
                )
            else:
                res = run_backtest(
                    spec,
                    db_path=db_path,
                    tickers=tickers,
                    lookback_days=lookback_days,
                    initial_capital=initial_capital,
                    benchmark_ticker=benchmark_ticker,
                    progress_callback=_prog,
                    cancel_event=cancel_event,
                )
        except BacktestCancelled:
            cancelled = True
            break
        if res is not None:
            results.append(res)

    if cancelled and not results:
        raise BacktestCancelled("Cancelled by user.")

    summary_parts = [
        f"{r.spec.name}: {r.metrics.total_return_pct:.1f}% ({r.metrics.sharpe:.2f} Sharpe)"
        for r in results
    ]
    return CompareRunResult(
        results=results,
        summary=" | ".join(summary_parts) if summary_parts else "No results.",
        cancelled=cancelled,
    )


def load_benchmark_curve(
    db_path: str,
    benchmark_ticker: str,
    lookback_days: int,
    initial_capital: float,
) -> tuple[list[str], list[float]]:
    """Load benchmark buy-and-hold curve for chart overlay."""
    market = resolve_market_ticker(db_path, benchmark_ticker)
    mdf = load_market_data(db_path, market)
    if mdf is None or mdf.empty:
        return [], []
    from src.analysis.backtest_common import period_bounds

    p_start, p_end = period_bounds(mdf, lookback_days)
    sl = mdf.loc[p_start:p_end]
    if sl.empty:
        return [], []
    closes = sl["Adj Close"].astype(float)
    if closes.iloc[0] == 0:
        return [], []
    eq = (closes / closes.iloc[0]) * initial_capital
    return [d.strftime("%Y-%m-%d") for d in sl.index], eq.tolist()
