"""Unified vectorized backtest engine for StrategySpec rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.backtest_common import (
    MIN_WARMUP_DAYS,
    TickerChartSeries,
    TradeMarker,
    build_marker_tooltip,
    combine_portfolio_series,
    format_period,
    pct_returns_from_start,
    period_bounds,
    simulate_portfolio_values,
)
from src.analysis.backtest_metrics import MetricsReport, compute_metrics, monthly_returns_grid
from src.analysis.db import load_entire_database, load_market_data, list_tickers, resolve_market_ticker
from src.analysis.general import calculate_rsi
from src.analysis.signal_quality import BacktestCostModel, adjust_trade_return, cost_model_from_config
from src.analysis.strategy_spec import INDICATOR_DEFAULTS, Rule, RuleGroup, StrategySpec

ProgressCallback = Callable[[float], None]

_LOAD_BATCH_SIZE = 40


class BacktestCancelled(Exception):
    """Raised when a backtest is aborted via cancel_event."""


def _is_cancelled(cancel_event) -> bool:
    return cancel_event is not None and bool(getattr(cancel_event, "is_set", lambda: False)())


def _ensure_not_cancelled(cancel_event, *, where: str = "") -> None:
    if _is_cancelled(cancel_event):
        raise BacktestCancelled(where or "Cancelled by user.")


def _load_backtest_price_data(
    db_path: str,
    *,
    tickers: list[str] | None,
    cancel_event=None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Load OHLCV for the run, checking cancel between batches when scanning the universe."""
    _ensure_not_cancelled(cancel_event, where="Cancelled before data load.")
    if tickers:
        symbols = [str(t).strip().upper() for t in tickers if t]
        if progress_callback:
            progress_callback(0.02)
        data = load_entire_database(db_path, full_ohlcv=True, tickers=symbols)
        _ensure_not_cancelled(cancel_event, where="Cancelled during data load.")
        return data, symbols

    symbols = list_tickers(db_path)
    if not symbols:
        return None, []

    frames: list[pd.DataFrame] = []
    total = len(symbols)
    for start in range(0, total, _LOAD_BATCH_SIZE):
        _ensure_not_cancelled(cancel_event, where="Cancelled during data load.")
        batch = symbols[start : start + _LOAD_BATCH_SIZE]
        if progress_callback:
            progress_callback(min(0.25, (start + len(batch)) / max(total, 1) * 0.25))
        chunk = load_entire_database(db_path, full_ohlcv=True, tickers=batch)
        if chunk is not None and not chunk.empty:
            frames.append(chunk)

    if not frames:
        return None, symbols
    return pd.concat(frames, axis=0), symbols


def _index_loc_to_int(loc) -> int:
    """Normalize Index.get_loc() results (int | slice | ndarray) to a single int."""
    if isinstance(loc, slice):
        start = loc.start
        if start is None:
            raise IndexError("slice loc has no start")
        return int(start)
    if isinstance(loc, (np.ndarray, list, tuple, pd.Index)):
        if len(loc) == 0:
            raise IndexError("empty loc")
        return int(loc[0])
    return int(loc)


@dataclass
class BacktestTrade:
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    exit_reason: str

    def to_dict(self) -> dict:
        return {
            "Ticker": self.ticker,
            "Entry_Date": self.entry_date,
            "Exit_Date": self.exit_date,
            "Entry_Price": self.entry_price,
            "Exit_Price": self.exit_price,
            "Return_Pct": self.return_pct,
            "Holding_Days": self.holding_days,
            "Exit_Reason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    spec: StrategySpec
    metrics: MetricsReport = field(default_factory=MetricsReport)
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_dates: list[str] = field(default_factory=list)
    equity_values: list[float] = field(default_factory=list)
    benchmark_dates: list[str] = field(default_factory=list)
    benchmark_values: list[float] = field(default_factory=list)
    buy_hold_dates: list[str] = field(default_factory=list)
    buy_hold_values: list[float] = field(default_factory=list)
    drawdown_pct: list[float] = field(default_factory=list)
    chart_series: dict[str, TickerChartSeries] = field(default_factory=dict)
    monthly_returns: pd.DataFrame | None = None
    period_label: str = ""
    tickers_run: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])


def _macd_line(series: pd.Series, fast: int, slow: int) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def compute_indicator(
    name: str,
    df: pd.DataFrame,
    params: dict,
    *,
    market_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute indicator series aligned to df index."""
    price = df["Adj Close"].astype(float)
    params = {**INDICATOR_DEFAULTS.get(name, {}), **params}
    if name == "PRICE":
        return price
    if name == "SMA":
        return price.rolling(int(params.get("period", 50))).mean()
    if name == "EMA":
        return price.ewm(span=int(params.get("period", 20)), adjust=False).mean()
    if name == "RSI":
        return calculate_rsi(price, period=int(params.get("period", 14)))
    if name == "MACD":
        return _macd_line(price, int(params.get("fast", 12)), int(params.get("slow", 26)))
    if name == "MACD_SIGNAL":
        macd = _macd_line(price, int(params.get("fast", 12)), int(params.get("slow", 26)))
        return macd.ewm(span=int(params.get("signal", 9)), adjust=False).mean()
    if name in ("BOLLINGER_UPPER", "BOLLINGER_LOWER", "BOLLINGER_MID"):
        period = int(params.get("period", 20))
        num_std = float(params.get("num_std", 2))
        mid = price.rolling(period).mean()
        std = price.rolling(period).std()
        if name == "BOLLINGER_UPPER":
            return mid + num_std * std
        if name == "BOLLINGER_LOWER":
            return mid - num_std * std
        return mid
    if name == "ATR":
        period = int(params.get("period", 14))
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        prev_close = price.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()
    if name == "ROC":
        period = int(params.get("period", 10))
        return price.pct_change(periods=period, fill_method=None) * 100.0
    if name == "VOLUME_AVG":
        period = int(params.get("period", 50))
        return df["Volume"].astype(float).rolling(period).mean()
    if name == "HIGH_52W":
        period = int(params.get("period", 252))
        return price.rolling(period).max()
    if name == "LOW_52W":
        period = int(params.get("period", 252))
        return price.rolling(period).min()
    if name == "MARKET_PRICE" and market_df is not None:
        return market_df["Adj Close"].reindex(df.index, method="ffill")
    if name == "MARKET_SMA" and market_df is not None:
        m_price = market_df["Adj Close"].reindex(df.index, method="ffill")
        return m_price.rolling(int(params.get("period", 50))).mean()
    return price * np.nan


def _resolve_target_series(
    rule: Rule,
    df: pd.DataFrame,
    *,
    market_df: pd.DataFrame | None,
) -> pd.Series:
    target = rule.target
    if target.kind == "indicator" and target.indicator:
        return compute_indicator(target.indicator, df, target.params, market_df=market_df)
    val = float(target.value if target.value is not None else 0.0)
    return pd.Series(val, index=df.index, dtype=float)


def evaluate_comparator(
    left: pd.Series,
    right: pd.Series,
    comparator: str,
) -> pd.Series:
    if comparator == "above":
        return left > right
    if comparator == "below":
        return left < right
    if comparator == "crosses_above":
        prev = left.shift(1) <= right.shift(1)
        return prev & (left > right)
    if comparator == "crosses_below":
        prev = left.shift(1) >= right.shift(1)
        return prev & (left < right)
    if comparator == "within_pct":
        pct = right.iloc[0] if len(right) else 0.0
        band = left.abs() * (pct / 100.0)
        return (left - right).abs() <= band
    return pd.Series(False, index=left.index)


def evaluate_rule_group(
    group: RuleGroup,
    df: pd.DataFrame,
    *,
    market_df: pd.DataFrame | None = None,
) -> pd.Series:
    if not group.rules:
        return pd.Series(False, index=df.index)
    results: list[pd.Series] = []
    for rule in group.rules:
        left = compute_indicator(rule.indicator, df, rule.params, market_df=market_df)
        right = _resolve_target_series(rule, df, market_df=market_df)
        results.append(evaluate_comparator(left, right, rule.comparator))
    if group.logic == "or":
        out = results[0]
        for r in results[1:]:
            out = out | r
        return out.fillna(False)
    out = results[0]
    for r in results[1:]:
        out = out & r
    return out.fillna(False)


def _collect_param_periods(params: dict | None) -> list[int]:
    if not params:
        return []
    out: list[int] = []
    for key in ("period", "slow", "fast", "signal"):
        if key in params:
            try:
                out.append(int(params[key]))
            except (TypeError, ValueError):
                pass
    return out


def _spec_warmup_bars(spec: StrategySpec) -> int:
    """Trading bars of prior history needed before the lookback window."""
    periods = [MIN_WARMUP_DAYS]
    for group in (spec.entry, spec.exit_rules):
        if group is None:
            continue
        for rule in group.rules:
            periods.extend(_collect_param_periods(rule.params))
            periods.extend(_collect_param_periods(rule.target.params if rule.target else None))
            defaults = INDICATOR_DEFAULTS.get(rule.indicator, {})
            periods.extend(_collect_param_periods(defaults))
            if rule.target and rule.target.indicator:
                periods.extend(_collect_param_periods(INDICATOR_DEFAULTS.get(rule.target.indicator, {})))
    return max(periods) + 5


def _slice_with_warmup(
    raw: pd.DataFrame,
    p_start: pd.Timestamp,
    p_end: pd.Timestamp,
    warmup_bars: int,
) -> pd.DataFrame:
    """Price history covering indicator warmup plus the trade window."""
    loc = int(raw.index.searchsorted(p_start, side="left"))
    warm_loc = max(0, loc - max(warmup_bars, 0))
    return raw.iloc[warm_loc:].loc[:p_end].copy()


def _rebase_equity_window(
    in_market: pd.Series,
    closes: pd.Series,
    p_start: pd.Timestamp,
    p_end: pd.Timestamp,
    initial_capital: float,
) -> tuple[pd.Series, pd.Series]:
    """Restrict simulation output to the lookback window and rebase capital."""
    window_pos = in_market.loc[p_start:p_end]
    window_closes = closes.loc[p_start:p_end].astype(float)
    if window_pos.empty:
        return window_pos.astype(float), window_pos
    prior = in_market.shift(1)
    pos_for_return = prior.loc[p_start:p_end].astype(float).copy()
    if bool(pos_for_return.isna().iloc[0]):
        # Carry position from the last warmup bar into day-one returns.
        before = in_market.loc[:p_start]
        carry = float(before.iloc[-2]) if len(before) >= 2 else float(window_pos.iloc[0])
        pos_for_return.iloc[0] = carry
    pos_for_return = pos_for_return.fillna(0.0)
    rets = window_closes.pct_change(fill_method=None).fillna(0.0)
    window_equity = (pos_for_return * rets + 1.0).cumprod() * initial_capital
    return window_equity, window_pos.astype(int)


def _filter_trades_to_window(
    trades: list[BacktestTrade],
    p_start: pd.Timestamp,
    p_end: pd.Timestamp,
) -> list[BacktestTrade]:
    start_s = p_start.strftime("%Y-%m-%d")
    end_s = p_end.strftime("%Y-%m-%d")
    kept: list[BacktestTrade] = []
    for t in trades:
        if t.exit_date < start_s or t.entry_date > end_s:
            continue
        if t.entry_date < start_s:
            # Position carried into the window — report from window start.
            kept.append(
                BacktestTrade(
                    ticker=t.ticker,
                    entry_date=start_s,
                    exit_date=t.exit_date,
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    return_pct=t.return_pct,
                    holding_days=t.holding_days,
                    exit_reason=t.exit_reason,
                )
            )
        else:
            kept.append(t)
    return kept


def _simulate_ticker(
    ticker: str,
    df: pd.DataFrame,
    spec: StrategySpec,
    *,
    market_df: pd.DataFrame | None,
    cost: BacktestCostModel,
    initial_capital: float,
) -> tuple[list[BacktestTrade], pd.Series, pd.Series]:
    """One position at a time; entries on next bar open after signal."""
    entry_signal = evaluate_rule_group(spec.entry, df, market_df=market_df)
    exit_signal = (
        evaluate_rule_group(spec.exit_rules, df, market_df=market_df)
        if spec.exit_rules
        else pd.Series(False, index=df.index)
    )
    opens = df["Open"].astype(float)
    closes = df["Adj Close"].astype(float)
    in_position = pd.Series(0, index=df.index, dtype=int)
    trades: list[BacktestTrade] = []
    position = 0
    entry_price = 0.0
    entry_date: pd.Timestamp | None = None
    peak_price = 0.0
    holding_days = 0

    for i, date in enumerate(df.index):
        exited_this_bar = False
        if position:
            holding_days += 1
            price = float(closes.iloc[i])
            peak_price = max(peak_price, price)
            exit_reason = ""
            if spec.risk.stop_loss_pct and entry_price > 0:
                loss = (price / entry_price) - 1.0
                if loss <= -spec.risk.stop_loss_pct:
                    exit_reason = "stop_loss"
            if not exit_reason and spec.risk.take_profit_pct and entry_price > 0:
                gain = (price / entry_price) - 1.0
                if gain >= spec.risk.take_profit_pct:
                    exit_reason = "take_profit"
            if not exit_reason and spec.risk.trailing_stop_pct and peak_price > 0:
                trail = (price / peak_price) - 1.0
                if trail <= -spec.risk.trailing_stop_pct:
                    exit_reason = "trailing_stop"
            if not exit_reason and spec.risk.max_holding_days and holding_days >= spec.risk.max_holding_days:
                exit_reason = "max_hold"
            if not exit_reason and bool(exit_signal.iloc[i]):
                exit_reason = "exit_rule"
            if exit_reason:
                exit_px = float(opens.iloc[i]) if i < len(opens) else price
                ret = adjust_trade_return(entry_price, exit_px, cost=cost) if spec.apply_costs else (
                    (exit_px / entry_price) - 1.0
                )
                trades.append(
                    BacktestTrade(
                        ticker=ticker,
                        entry_date=entry_date.strftime("%Y-%m-%d") if entry_date else "",
                        exit_date=date.strftime("%Y-%m-%d"),
                        entry_price=entry_price,
                        exit_price=exit_px,
                        return_pct=ret * 100.0,
                        holding_days=holding_days,
                        exit_reason=exit_reason,
                    )
                )
                position = 0
                entry_price = 0.0
                entry_date = None
                peak_price = 0.0
                holding_days = 0
                exited_this_bar = True
        # Level-based entries stay True while the regime holds; skip same-bar
        # re-entry after an exit so above/below does not flip-flop in one bar.
        if position == 0 and not exited_this_bar and i > 0 and bool(entry_signal.iloc[i - 1]):
            entry_px = float(opens.iloc[i])
            if entry_px > 0:
                position = 1
                entry_price = entry_px
                entry_date = date
                peak_price = entry_px
                holding_days = 0
        in_position.iloc[i] = position

    if position and entry_date is not None:
        last_date = df.index[-1]
        exit_px = float(closes.iloc[-1])
        ret = adjust_trade_return(entry_price, exit_px, cost=cost) if spec.apply_costs else (
            (exit_px / entry_price) - 1.0
        )
        trades.append(
            BacktestTrade(
                ticker=ticker,
                entry_date=entry_date.strftime("%Y-%m-%d"),
                exit_date=last_date.strftime("%Y-%m-%d"),
                entry_price=entry_price,
                exit_price=exit_px,
                return_pct=ret * 100.0,
                holding_days=holding_days,
                exit_reason="end_of_period",
            )
        )

    equity = (in_position.shift(1).fillna(0) * closes.pct_change(fill_method=None).fillna(0) + 1.0).cumprod()
    equity = equity * initial_capital
    return trades, equity, in_position


def _buy_hold_equity(df: pd.DataFrame, initial: float) -> pd.Series:
    closes = df["Adj Close"].astype(float)
    if closes.empty:
        return closes
    return (closes / closes.iloc[0]) * initial


def _rule_groups(spec: StrategySpec) -> list[RuleGroup]:
    groups: list[RuleGroup] = [spec.entry]
    if spec.exit_rules is not None:
        groups.append(spec.exit_rules)
    return groups


def _spec_needs_market(spec: StrategySpec) -> bool:
    market_indicators = {"MARKET_PRICE", "MARKET_SMA"}
    for group in _rule_groups(spec):
        for rule in group.rules:
            if rule.indicator in market_indicators:
                return True
            if rule.target and rule.target.indicator in market_indicators:
                return True
    return False


def run_backtest(
    spec: StrategySpec,
    *,
    db_path: str,
    tickers: list[str] | None = None,
    lookback_days: int = 365,
    initial_capital: float = 10000.0,
    benchmark_ticker: str | None = None,
    cost: BacktestCostModel | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event=None,
) -> BacktestResult | None:
    """Run unified backtest for a StrategySpec."""
    _ensure_not_cancelled(cancel_event, where="Cancelled before start.")
    if spec.engine == "buy_hold":
        return _run_buy_hold(
            spec,
            db_path=db_path,
            tickers=tickers,
            lookback_days=lookback_days,
            initial_capital=initial_capital,
            benchmark_ticker=benchmark_ticker,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
    if spec.engine == "canslim":
        # Lazy import avoids a circular dependency with canslim_adapter.
        from src.analysis.canslim_adapter import run_canslim_backtest_unified

        return run_canslim_backtest_unified(
            spec,
            db_path=db_path,
            tickers=tickers,
            lookback_days=lookback_days,
            initial_capital=initial_capital,
            use_parallel=False,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    cost = cost or cost_model_from_config()
    all_data, symbols = _load_backtest_price_data(
        db_path,
        tickers=tickers,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    if all_data is None or all_data.empty:
        return None
    _ensure_not_cancelled(cancel_event, where="Cancelled after data load.")

    market_sym = spec.market_ticker or benchmark_ticker
    if not market_sym and _spec_needs_market(spec):
        market_sym = resolve_market_ticker(db_path, None)
    market_df: pd.DataFrame | None = None
    if market_sym:
        _ensure_not_cancelled(cancel_event, where="Cancelled before market load.")
        market_df = load_market_data(db_path, resolve_market_ticker(db_path, market_sym))

    if tickers:
        symbols = [t.upper() for t in tickers]
    elif not symbols:
        symbols = sorted(all_data["Ticker"].unique().tolist())

    all_trades: list[BacktestTrade] = []
    per_ticker_equity: dict[str, pd.Series] = {}
    per_ticker_in_market: dict[str, pd.Series] = {}
    chart_series: dict[str, TickerChartSeries] = {}
    cap_per = initial_capital / max(len(symbols), 1)
    period_start: pd.Timestamp | None = None
    period_end: pd.Timestamp | None = None

    for idx, sym in enumerate(symbols):
        _ensure_not_cancelled(cancel_event, where=f"Cancelled at {sym}.")
        if progress_callback:
            # Reserve 0–25% for load; 25–95% for per-ticker work.
            progress_callback(0.25 + 0.70 * ((idx + 1) / max(len(symbols), 1)))

        raw = all_data[all_data["Ticker"] == sym].copy()
        if raw.empty:
            continue
        if "Date" in raw.columns:
            raw = raw.set_index("Date")
        raw = raw.sort_index()
        p_start, p_end = period_bounds(raw, lookback_days)
        window = raw.loc[p_start:p_end]
        if len(window) < 30:
            continue
        if period_start is None:
            period_start, period_end = p_start, p_end

        warm_df = _slice_with_warmup(raw, p_start, p_end, _spec_warmup_bars(spec))
        trades_all, _equity_all, in_market_all = _simulate_ticker(
            sym, warm_df, spec, market_df=market_df, cost=cost, initial_capital=cap_per,
        )
        equity, in_market = _rebase_equity_window(
            in_market_all,
            warm_df["Adj Close"],
            p_start,
            p_end,
            cap_per,
        )
        trades = _filter_trades_to_window(trades_all, p_start, p_end)
        all_trades.extend(trades)
        per_ticker_equity[sym] = equity
        per_ticker_in_market[sym] = in_market

        closes_w = window["Adj Close"].astype(float)
        pct = pct_returns_from_start(closes_w)
        markers: list[TradeMarker] = []
        for t in trades:
            try:
                ei = _index_loc_to_int(window.index.get_loc(pd.Timestamp(t.entry_date)))
                xi = _index_loc_to_int(window.index.get_loc(pd.Timestamp(t.exit_date)))
                markers.append(
                    TradeMarker(
                        x=ei,
                        y=float(pct.iloc[ei]),
                        side="buy",
                        tooltip=build_marker_tooltip(
                            side="buy", date=pd.Timestamp(t.entry_date), price=t.entry_price,
                        ),
                    )
                )
                markers.append(
                    TradeMarker(
                        x=xi,
                        y=float(pct.iloc[xi]),
                        side="sell",
                        tooltip=build_marker_tooltip(
                            side="sell",
                            date=pd.Timestamp(t.exit_date),
                            price=t.exit_price,
                            reason=t.exit_reason,
                            return_pct=t.return_pct / 100.0,
                        ),
                    )
                )
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        chart_series[sym] = TickerChartSeries(
            ticker=sym,
            dates=[d.strftime("%Y-%m-%d") for d in window.index],
            pct_returns=pct.tolist(),
            portfolio_values=equity.tolist(),
            markers=markers,
        )

    _ensure_not_cancelled(cancel_event, where="Cancelled before results assembly.")
    if not per_ticker_equity:
        return None

    combined_dates, combined_values = combine_portfolio_series(chart_series)
    equity_dates = combined_dates
    equity_values = combined_values

    bench_dates: list[str] = []
    bench_values: list[float] = []
    if market_df is not None and period_start and period_end:
        m_slice = market_df.loc[period_start:period_end]
        if not m_slice.empty:
            bh = _buy_hold_equity(m_slice, initial_capital)
            bench_dates = [d.strftime("%Y-%m-%d") for d in m_slice.index]
            bench_values = bh.tolist()

    bh_dates: list[str] = []
    bh_values: list[float] = []
    if per_ticker_equity:
        frames = []
        for sym, eq in per_ticker_equity.items():
            frames.append(pd.DataFrame({sym: eq}))
        combined_eq = frames[0]
        for f in frames[1:]:
            combined_eq = combined_eq.join(f, how="outer")
        combined_eq = combined_eq.sort_index().ffill().fillna(0)
        first_prices = []
        for sym in per_ticker_equity:
            raw = all_data[all_data["Ticker"] == sym].copy()
            if "Date" in raw.columns:
                raw = raw.set_index("Date")
            raw = raw.sort_index()
            if period_start and period_end:
                sl = raw.loc[period_start:period_end]
                if not sl.empty:
                    first_prices.append(sl["Adj Close"].astype(float))
        if first_prices:
            bh_equity = sum(
                (s / s.iloc[0]) * cap_per for s in first_prices if not s.empty and s.iloc[0] > 0
            )
            if isinstance(bh_equity, pd.Series):
                bh_dates = [d.strftime("%Y-%m-%d") for d in bh_equity.index]
                bh_values = bh_equity.tolist()

    in_market_combined = None
    if per_ticker_in_market:
        frames = [pd.DataFrame({k: v}) for k, v in per_ticker_in_market.items()]
        im = frames[0]
        for f in frames[1:]:
            im = im.join(f, how="outer")
        in_market_combined = im.mean(axis=1).fillna(0)

    _ensure_not_cancelled(cancel_event, where="Cancelled before metrics.")
    metrics = compute_metrics(
        equity_dates,
        equity_values,
        trades=[t.to_dict() for t in all_trades],
        benchmark_dates=bench_dates or None,
        benchmark_values=bench_values or None,
        in_market=in_market_combined,
        cost=cost if spec.apply_costs else None,
    )

    dd_pct: list[float] = []
    if equity_values:
        eq = pd.Series(equity_values)
        peak = eq.cummax()
        dd_pct = ((eq - peak) / peak.replace(0, np.nan) * 100.0).fillna(0).tolist()

    result = BacktestResult(
        spec=spec,
        metrics=metrics,
        trades=all_trades,
        equity_dates=equity_dates,
        equity_values=equity_values,
        benchmark_dates=bench_dates,
        benchmark_values=bench_values,
        buy_hold_dates=bh_dates,
        buy_hold_values=bh_values,
        drawdown_pct=dd_pct,
        chart_series=chart_series,
        monthly_returns=monthly_returns_grid(equity_dates, equity_values),
        period_label=format_period(period_start, period_end),
        tickers_run=symbols,
        summary=(
            f"{spec.name}: {metrics.total_return_pct:.1f}% return, "
            f"Sharpe {metrics.sharpe:.2f}, max DD {metrics.max_drawdown_pct:.1f}%, "
            f"{metrics.trade_count} trades."
        ),
    )
    return result


def _run_buy_hold(
    spec: StrategySpec,
    *,
    db_path: str,
    tickers: list[str] | None,
    lookback_days: int,
    initial_capital: float,
    benchmark_ticker: str | None,
    progress_callback: ProgressCallback | None,
    cancel_event=None,
) -> BacktestResult | None:
    all_data, symbols = _load_backtest_price_data(
        db_path,
        tickers=tickers,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    if all_data is None or all_data.empty:
        return None
    if tickers:
        symbols = [t.upper() for t in tickers]
    elif not symbols:
        symbols = sorted(all_data["Ticker"].unique().tolist())
    cap_per = initial_capital / max(len(symbols), 1)
    chart_series: dict[str, TickerChartSeries] = {}
    period_start = period_end = None

    for idx, sym in enumerate(symbols):
        _ensure_not_cancelled(cancel_event, where=f"Cancelled at {sym}.")
        if progress_callback:
            progress_callback(0.25 + 0.70 * ((idx + 1) / max(len(symbols), 1)))
        raw = all_data[all_data["Ticker"] == sym].copy()
        if "Date" in raw.columns:
            raw = raw.set_index("Date")
        raw = raw.sort_index()
        p_start, p_end = period_bounds(raw, lookback_days)
        df = raw.loc[p_start:p_end]
        if df.empty:
            continue
        period_start, period_end = p_start, p_end
        eq = _buy_hold_equity(df, cap_per)
        pct = pct_returns_from_start(df["Adj Close"].astype(float))
        chart_series[sym] = TickerChartSeries(
            ticker=sym,
            dates=[d.strftime("%Y-%m-%d") for d in df.index],
            pct_returns=pct.tolist(),
            portfolio_values=eq.tolist(),
        )

    _ensure_not_cancelled(cancel_event, where="Cancelled before buy-hold results.")
    if not chart_series:
        return None
    equity_dates, equity_values = combine_portfolio_series(chart_series)
    metrics = compute_metrics(equity_dates, equity_values)
    return BacktestResult(
        spec=spec,
        metrics=metrics,
        equity_dates=equity_dates,
        equity_values=equity_values,
        chart_series=chart_series,
        monthly_returns=monthly_returns_grid(equity_dates, equity_values),
        period_label=format_period(period_start, period_end),
        tickers_run=symbols,
        summary=f"Buy & Hold: {metrics.total_return_pct:.1f}% return.",
    )
