"""Volatility breakout and trailing-stop analysis."""

from __future__ import annotations

import inspect
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.db import db_connection, list_tickers, load_entire_database
from src.analysis.parallel_exec import default_worker_count, pool_chunksize, run_parallel_map

ProgressCallback = Callable[[float], None]
OptimizationProgressCallback = Callable[[str, float, int, int, str], None]
ResultCallback = Callable[[dict], None]

LOADING_WEIGHT = 0.10
OPTIMIZING_START = 0.10
OPTIMIZING_END = 0.95
FULL_UNIVERSE_BATCH_SIZE = 300


@dataclass
class VolatilityResult:
    results_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_count: int = 0
    cancelled: bool = False


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def strat_volatility_breakout(
    df: pd.DataFrame,
    atr_period: int = 14,
    entry_mult: float = 2.0,
    exit_mult: float = 3.0,
) -> tuple[float, int]:
    data = df.copy()
    if "High" not in data.columns:
        data["High"] = data["Adj Close"]
        data["Low"] = data["Adj Close"]
        data["Close"] = data["Adj Close"]

    data["ATR"] = calculate_atr(data, period=atr_period)
    data["SMA"] = data["Close"].rolling(window=20).mean()
    data.dropna(inplace=True)
    if data.empty:
        return 0.0, 0

    prices = data["Close"].values
    atrs = data["ATR"].values
    smas = data["SMA"].values
    position = 0
    cash = 1.0
    shares = 0
    highest_high = 0
    stop_price = 0
    total_trades = 0

    for i in range(len(prices)):
        price = prices[i]
        atr = atrs[i]
        sma = smas[i]
        if position == 0:
            breakout_level = sma + (entry_mult * atr)
            if price > breakout_level:
                shares = cash / price
                cash = 0
                position = 1
                highest_high = price
                stop_price = price - (exit_mult * atr)
                total_trades += 1
        elif position == 1:
            if price > highest_high:
                highest_high = price
                new_stop = highest_high - (exit_mult * atr)
                if new_stop > stop_price:
                    stop_price = new_stop
            if price < stop_price:
                cash = shares * price
                shares = 0
                position = 0

    final_value = cash if position == 0 else shares * prices[-1]
    return float(final_value), total_trades


def worker_volatility(args):
    ticker, df, atr_period, entry_mult, exit_mult = args
    if len(df) < 100:
        return None
    try:
        if "Close" not in df.columns:
            df = df.copy()
            df["Close"] = df["Adj Close"]
            df["High"] = df["Adj Close"]
            df["Low"] = df["Adj Close"]

        start = df["Close"].iloc[0]
        end = df["Close"].iloc[-1]
        baseline = end / start if start != 0 else 0
        strat_ret, trades = strat_volatility_breakout(
            df, atr_period=atr_period, entry_mult=entry_mult, exit_mult=exit_mult
        )
        current_atr = calculate_atr(df).iloc[-1]
        current_price = df["Close"].iloc[-1]
        suggested_stop = current_price - (exit_mult * current_atr)
        current_sma = df["Close"].rolling(window=20).mean().iloc[-1]
        suggested_entry = current_sma + (entry_mult * current_atr)
        return {
            "Ticker": ticker,
            "Baseline": baseline,
            "Vol_Strategy_Return": strat_ret,
            "Alpha": strat_ret - baseline,
            "Trades": trades,
            "Current_Price": current_price,
            "Current_ATR": current_atr,
            "Suggested_Stop_Loss": suggested_stop,
            "Suggested_Entry_Point": suggested_entry,
        }
    except Exception:
        return None


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
    tickers: set[str] | None,
    atr_period: int,
    entry_mult: float,
    exit_mult: float,
) -> list[tuple]:
    tasks = []
    for ticker, df in all_data.groupby("Ticker"):
        if tickers is not None and ticker not in tickers:
            continue
        tasks.append((ticker, df, atr_period, entry_mult, exit_mult))
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
            worker_volatility,
            tasks,
            use_parallel=True,
            workers=w,
            chunksize=pool_chunksize(n, w),
            cancel_event=cancel_event,
        )
        for i, res in enumerate(iterator):
            if res:
                results.append(res)
                if result_callback:
                    result_callback(res)
            ticker = (res or {}).get("Ticker", "")
            _emit_progress(
                progress_callback,
                "optimizing",
                _optimizing_pct(done_offset + i + 1, total),
                done_offset + i + 1,
                total,
                f"Analyzing {ticker}",
            )
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
    else:
        for i, task in enumerate(tasks):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            res = worker_volatility(task)
            if res:
                results.append(res)
                if result_callback:
                    result_callback(res)
            ticker = task[0]
            _emit_progress(
                progress_callback,
                "optimizing",
                _optimizing_pct(done_offset + i + 1, total),
                done_offset + i + 1,
                total,
                f"Analyzing {ticker}",
            )

    return results, cancelled


def run_volatility_analysis(
    db_path: str,
    tickers: list[str] | None = None,
    atr_period: int = 14,
    entry_mult: float = 2.0,
    exit_mult: float = 3.0,
    progress_callback: OptimizationProgressCallback | ProgressCallback | None = None,
    result_callback: ResultCallback | None = None,
    cancel_event: threading.Event | None = None,
    use_parallel: bool = True,
    workers: int | None = None,
    save_to_db: bool = True,
) -> VolatilityResult | None:
    progress = _normalize_progress_callback(progress_callback)

    if tickers:
        ticker_universe = [str(t).upper() for t in tickers]
    else:
        ticker_universe = list_tickers(db_path)

    total_tickers = len(ticker_universe)
    if total_tickers == 0:
        return VolatilityResult()

    _emit_progress(progress, "loading", 0.0, 0, total_tickers, "Loading price data…")
    if cancel_event is not None and cancel_event.is_set():
        return VolatilityResult(cancelled=True)

    results: list[dict] = []
    cancelled = False
    ticker_set = {str(t).upper() for t in tickers} if tickers else None

    if tickers:
        all_data = load_entire_database(db_path, full_ohlcv=True, tickers=tickers)
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
        tasks = _build_tasks(all_data, ticker_set, atr_period, entry_mult, exit_mult)
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
            batch_data = load_entire_database(db_path, full_ohlcv=True, tickers=batch)
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
                batch_data, None, atr_period, entry_mult, exit_mult
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

    if not results:
        return VolatilityResult(cancelled=cancelled)

    res_df = pd.DataFrame(results)
    if save_to_db and not cancelled:
        try:
            with db_connection(db_path, readonly=False) as conn:
                res_df.to_sql("volatility_metrics", conn, if_exists="replace", index=False)
                conn.commit()
        except sqlite3.Error:
            pass

    _emit_progress(
        progress,
        "finalizing",
        1.0,
        len(results),
        total_tickers,
        "Complete",
    )

    return VolatilityResult(
        results_df=res_df,
        stock_count=len(res_df),
        cancelled=cancelled,
    )
