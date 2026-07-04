"""Momentum parameter optimization."""

from __future__ import annotations

import inspect
import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.db import list_tickers, load_entire_database
from src.analysis.parallel_exec import default_worker_count, pool_chunksize, run_parallel_map

ProgressCallback = Callable[[float], None]
OptimizationProgressCallback = Callable[[str, float, int, int, str], None]
ResultCallback = Callable[[dict], None]

LOADING_WEIGHT = 0.10
OPTIMIZING_START = 0.10
OPTIMIZING_END = 0.95
FULL_UNIVERSE_BATCH_SIZE = 300

DEFAULT_WINDOWS = [10, 20, 30, 40, 50, 60]
DEFAULT_BUY = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
DEFAULT_SELL = [0.00, -0.02, -0.05]


@dataclass
class FocusedResult:
    results_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_count: int = 0
    cancelled: bool = False


def _momentum_for_window(prices: np.ndarray, window_days: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= window_days:
        return out
    base = prices[:-window_days]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window_days:] = (prices[window_days:] - base) / base
    return out


def _backtest_from_arrays(
    prices: np.ndarray,
    momenta: np.ndarray,
    window_days: int,
    buy_threshold: float,
    sell_threshold: float,
) -> float:
    n = len(prices)
    if n < window_days:
        return 1.0
    position = 0
    cash = 1.0
    shares = 0.0
    for i in range(window_days, n):
        price = prices[i]
        mom = momenta[i]
        if np.isnan(mom):
            continue
        if position == 0 and mom > buy_threshold:
            shares = cash / price
            cash = 0.0
            position = 1
        elif position == 1 and mom < sell_threshold:
            cash = shares * price
            shares = 0.0
            position = 0
    return float(cash if position == 0 else shares * prices[-1])


def backtest_momentum_strategy(
    df: pd.DataFrame,
    window_days: int,
    buy_threshold: float,
    sell_threshold: float,
) -> float:
    prices = df["Adj Close"].to_numpy(dtype=np.float64)
    momenta = _momentum_for_window(prices, window_days)
    return _backtest_from_arrays(prices, momenta, window_days, buy_threshold, sell_threshold)


def optimize_single_ticker(args):
    ticker, prices, windows, buy_thresholds, sell_thresholds = args
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) == 0:
        return {
            "Ticker": ticker,
            "Baseline": 0.0,
            "Best_Return": 1.0,
            "Params": None,
        }
    start_price = prices[0]
    end_price = prices[-1]
    baseline = end_price / start_price if start_price != 0 else 0.0
    momenta = {w: _momentum_for_window(prices, w) for w in windows}
    best_score = -np.inf
    best_params = None
    for w, b, s in itertools.product(windows, buy_thresholds, sell_thresholds):
        ret = _backtest_from_arrays(prices, momenta[w], w, b, s)
        if ret > best_score:
            best_score = ret
            best_params = (w, b, s)
    return {
        "Ticker": ticker,
        "Baseline": baseline,
        "Best_Return": best_score,
        "Params": best_params,
    }


def run_momentum_backtest(
    df: pd.DataFrame,
    window_days: int,
    buy_threshold: float,
    sell_threshold: float,
) -> float:
    return backtest_momentum_strategy(df, window_days, buy_threshold, sell_threshold)


def _normalize_progress_callback(
    callback: OptimizationProgressCallback | ProgressCallback | None,
) -> OptimizationProgressCallback | None:
    if callback is None:
        return None
    try:
        params = list(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError):
        return callback  # type: ignore[return-value]
    if len(params) == 1:

        def _legacy(phase: str, pct: float, done: int, total: int, detail: str) -> None:
            callback(pct)  # type: ignore[misc]

        return _legacy
    return callback  # type: ignore[return-value]


def _emit_progress(
    callback: OptimizationProgressCallback | None,
    phase: str,
    pct: float,
    done: int,
    total: int,
    detail: str,
) -> None:
    if callback is None:
        return
    callback(phase, pct, done, total, detail)


def _optimizing_pct(done: int, total: int) -> float:
    if total <= 0:
        return OPTIMIZING_END
    return OPTIMIZING_START + (OPTIMIZING_END - OPTIMIZING_START) * (done / total)


def _build_tasks(
    all_data: pd.DataFrame,
    windows: list[int],
    buy_thresholds: list[float],
    sell_thresholds: list[float],
    target_tickers: set[str] | None,
) -> list[tuple]:
    tasks = []
    for ticker, group_df in all_data.groupby("Ticker"):
        if target_tickers is not None and ticker not in target_tickers:
            continue
        prices = group_df["Adj Close"].to_numpy(dtype=np.float64)
        tasks.append((ticker, prices, windows, buy_thresholds, sell_thresholds))
    return tasks


def _run_tasks(
    tasks: list[tuple],
    *,
    use_parallel: bool,
    workers: int | None,
    progress_callback: OptimizationProgressCallback | None,
    result_callback: ResultCallback | None,
    cancel_event: threading.Event | None,
    done_offset: int,
    total: int,
) -> tuple[list[dict], bool]:
    if not tasks:
        return [], False

    results: list[dict] = []
    cancelled = False
    n = len(tasks)
    w = workers or default_worker_count()

    if use_parallel and n > 1:
        iterator = run_parallel_map(
            optimize_single_ticker,
            tasks,
            use_parallel=True,
            workers=w,
            chunksize=pool_chunksize(n, w),
            cancel_event=cancel_event,
        )
        for i, res in enumerate(iterator):
            results.append(res)
            row = {**res, "Alpha": res["Best_Return"] - res["Baseline"]}
            if result_callback:
                result_callback(row)
            ticker = res.get("Ticker", "")
            _emit_progress(
                progress_callback,
                "optimizing",
                _optimizing_pct(done_offset + i + 1, total),
                done_offset + i + 1,
                total,
                f"Optimizing {ticker}",
            )
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
    else:
        for i, task in enumerate(tasks):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            res = optimize_single_ticker(task)
            results.append(res)
            row = {**res, "Alpha": res["Best_Return"] - res["Baseline"]}
            if result_callback:
                result_callback(row)
            ticker = res.get("Ticker", "")
            _emit_progress(
                progress_callback,
                "optimizing",
                _optimizing_pct(done_offset + i + 1, total),
                done_offset + i + 1,
                total,
                f"Optimizing {ticker}",
            )

    return results, cancelled


def _finalize_results(results: list[dict]) -> pd.DataFrame:
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df["Alpha"] = res_df["Best_Return"] - res_df["Baseline"]
        res_df = res_df.sort_values(by="Alpha", ascending=False)
    return res_df


def run_focused_optimization(
    db_path: str,
    target_tickers: list[str] | None = None,
    windows: list[int] | None = None,
    buy_thresholds: list[float] | None = None,
    sell_thresholds: list[float] | None = None,
    progress_callback: OptimizationProgressCallback | ProgressCallback | None = None,
    result_callback: ResultCallback | None = None,
    cancel_event: threading.Event | None = None,
    use_parallel: bool = True,
    workers: int | None = None,
) -> FocusedResult | None:
    windows = windows or DEFAULT_WINDOWS
    buy_thresholds = buy_thresholds or DEFAULT_BUY
    sell_thresholds = sell_thresholds or DEFAULT_SELL
    progress = _normalize_progress_callback(progress_callback)

    if target_tickers:
        ticker_universe = [str(t).upper() for t in target_tickers]
    else:
        ticker_universe = list_tickers(db_path)

    total_tickers = len(ticker_universe)
    if total_tickers == 0:
        return FocusedResult()

    _emit_progress(progress, "loading", 0.0, 0, total_tickers, "Loading price data…")
    if cancel_event is not None and cancel_event.is_set():
        return FocusedResult(cancelled=True)

    results: list[dict] = []
    cancelled = False
    target_set = {str(t).upper() for t in target_tickers} if target_tickers else None

    if target_tickers:
        all_data = load_entire_database(db_path, tickers=target_tickers)
        if all_data is None or all_data.empty:
            return None
        _emit_progress(
            progress,
            "loading",
            LOADING_WEIGHT,
            0,
            total_tickers,
            f"Loaded {total_tickers:,} tickers",
        )
        tasks = _build_tasks(all_data, windows, buy_thresholds, sell_thresholds, target_set)
        del all_data
        batch_results, batch_cancelled = _run_tasks(
            tasks,
            use_parallel=use_parallel,
            workers=workers,
            progress_callback=progress,
            result_callback=result_callback,
            cancel_event=cancel_event,
            done_offset=0,
            total=total_tickers,
        )
        results.extend(batch_results)
        cancelled = batch_cancelled
    else:
        batches = [
            ticker_universe[i : i + FULL_UNIVERSE_BATCH_SIZE]
            for i in range(0, len(ticker_universe), FULL_UNIVERSE_BATCH_SIZE)
        ]
        num_batches = len(batches)
        done = 0
        for batch_idx, batch in enumerate(batches):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            batch_data = load_entire_database(db_path, tickers=batch)
            if batch_data is None or batch_data.empty:
                done += len(batch)
                continue
            load_pct = LOADING_WEIGHT * ((batch_idx + 1) / num_batches)
            _emit_progress(
                progress,
                "loading",
                load_pct,
                done,
                total_tickers,
                f"Loaded batch {batch_idx + 1} / {num_batches}",
            )
            tasks = _build_tasks(
                batch_data, windows, buy_thresholds, sell_thresholds, None
            )
            del batch_data
            batch_results, batch_cancelled = _run_tasks(
                tasks,
                use_parallel=use_parallel,
                workers=workers,
                progress_callback=progress,
                result_callback=result_callback,
                cancel_event=cancel_event,
                done_offset=done,
                total=total_tickers,
            )
            results.extend(batch_results)
            done += len(tasks)
            if batch_cancelled:
                cancelled = True
                break

    _emit_progress(
        progress,
        "finalizing",
        0.98,
        len(results),
        total_tickers,
        "Sorting results…",
    )

    res_df = _finalize_results(results)
    _emit_progress(
        progress,
        "finalizing",
        1.0,
        len(results),
        total_tickers,
        "Complete",
    )

    return FocusedResult(
        results_df=res_df,
        stock_count=len(res_df),
        cancelled=cancelled,
    )
