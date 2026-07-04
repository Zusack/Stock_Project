"""CANSLIM rules backtest simulation."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from src.analysis.backtest_common import (
    DEFAULT_LOOKBACK_DAYS,
    MIN_WARMUP_DAYS,
    TickerChartSeries,
    TradeMarker,
    build_marker_tooltip,
    combine_portfolio_series,
    flags_to_trigger_string,
    format_period,
    period_bounds,
    pct_returns_from_start,
    simulate_portfolio_values,
)
from src.analysis.canslim_rulebook import CanslimRuleSet, rule_set_from_config
from src.analysis.canslim_signals import build_canslim_frame
from src.analysis.bulk_loaders import load_fundamentals_eps_by_ticker, load_history_grouped
from src.analysis.db import db_connection, list_tickers, load_market_data
from src.analysis.symbol_universe import resolve_universe
from src.analysis.parallel_exec import default_worker_count, run_parallel_map

ProgressCallback = Callable[[float], None]

CANSLIM_FLAG_COLS = ("Pass_C", "Pass_A", "Pass_N", "Pass_S", "Pass_L", "Pass_M", "Pass_Pattern")


@dataclass
class CanslimBacktestResult:
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_trades: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    ticker_stats: pd.DataFrame | None = None
    period_label: str = ""
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    initial_capital: float = 1.0
    fundamentals_note: str = ""
    rule_set_version: str = ""
    chart_series: dict[str, TickerChartSeries] = field(default_factory=dict)
    combined_portfolio_dates: list[str] = field(default_factory=list)
    combined_portfolio_values: list[float] = field(default_factory=list)


def get_earnings_data(ticker: str, db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        with db_connection(db_path, readonly=True) as conn:
            q_df = pd.read_sql(
                """
                SELECT Report_Date, Value FROM fundamentals
                WHERE Ticker = ? AND Metric = 'Basic EPS' AND Period_Type = 'Quarterly'
                ORDER BY Report_Date ASC
            """,
                conn,
                params=(ticker,),
            )
            a_df = pd.read_sql(
                """
                SELECT Report_Date, Value FROM fundamentals
                WHERE Ticker = ? AND Metric = 'Basic EPS' AND Period_Type = 'Annual'
                ORDER BY Report_Date ASC
            """,
                conn,
                params=(ticker,),
            )
    except sqlite3.Error:
        return pd.DataFrame(), pd.DataFrame()

    if not q_df.empty:
        q_df["Report_Date"] = pd.to_datetime(q_df["Report_Date"])
        q_df["Growth_Q"] = q_df["Value"].pct_change(periods=4, fill_method=None)
    if not a_df.empty:
        a_df["Report_Date"] = pd.to_datetime(a_df["Report_Date"])
        a_df["Growth_A"] = a_df["Value"].pct_change(periods=1, fill_method=None)
    return q_df, a_df


def _entry_flags_row(df: pd.DataFrame, idx: int) -> dict[str, bool]:
    flags = {
        "C": bool(df["Pass_C"].iloc[idx]),
        "A": bool(df["Pass_A"].iloc[idx]),
        "N": bool(df["Pass_N"].iloc[idx]),
        "S": bool(df["Pass_S"].iloc[idx]),
        "L": bool(df["Pass_L"].iloc[idx]),
        "M": bool(df["Pass_M"].iloc[idx]),
    }
    if "Pass_Pattern" in df.columns:
        flags["P"] = bool(df["Pass_Pattern"].iloc[idx])
    return flags


def evaluate_exit(
    pct_change: float,
    price: float,
    entry_price: float,
    peak_gain_pct: float,
    rules: CanslimRuleSet,
    *,
    below_sma: bool = False,
    market_off: bool = False,
) -> tuple[bool, str]:
    """Return (should_exit, reason) using O'Neil-style precedence."""
    ex = rules.exit
    if pct_change <= ex.stop_loss_pct:
        return True, "Stop Loss"
    if pct_change >= ex.take_profit_pct:
        return True, "Take Profit"
    if ex.profit_zone_min_pct <= pct_change <= ex.profit_zone_max_pct:
        return True, "Profit Zone"
    if (
        ex.round_trip_exit_if_below_entry
        and peak_gain_pct >= ex.round_trip_min_gain_pct
        and price < entry_price
    ):
        return True, "Round-Trip Sell"
    if ex.use_below_sma50_exit and below_sma:
        return True, "Below 50-day simple moving average"
    if ex.use_market_downtrend_exit and market_off:
        return True, "Market Downtrend"
    return False, ""


def _build_chart_series(
    ticker: str,
    window_df: pd.DataFrame,
    trades: list[dict],
    initial_capital: float,
) -> TickerChartSeries:
    pct = pct_returns_from_start(window_df["Adj Close"])
    portfolio = simulate_portfolio_values(
        window_df.index, window_df["Adj Close"], trades, initial_capital
    )
    date_to_x = {d: i for i, d in enumerate(window_df.index)}
    markers: list[TradeMarker] = []
    for trade in trades:
        entry_date = pd.Timestamp(trade["Entry_Date"])
        exit_date = pd.Timestamp(trade["Exit_Date"])
        triggers = trade.get("Entry_Triggers", "")
        if entry_date in date_to_x:
            x = date_to_x[entry_date]
            markers.append(
                TradeMarker(
                    x=x,
                    y=float(pct.iloc[x]),
                    side="buy",
                    tooltip=build_marker_tooltip(
                        side="Buy",
                        date=entry_date,
                        price=float(trade["Entry_Price"]),
                        triggers=triggers,
                    ),
                )
            )
        if exit_date in date_to_x:
            x = date_to_x[exit_date]
            markers.append(
                TradeMarker(
                    x=x,
                    y=float(pct.iloc[x]),
                    side="sell",
                    tooltip=build_marker_tooltip(
                        side="Sell",
                        date=exit_date,
                        price=float(trade["Exit_Price"]),
                        reason=str(trade.get("Reason", "")),
                        return_pct=float(trade.get("Return", 0)),
                    ),
                )
            )
    return TickerChartSeries(
        ticker=ticker,
        dates=[d.strftime("%Y-%m-%d") for d in window_df.index],
        pct_returns=pct.round(4).tolist(),
        portfolio_values=[round(v, 4) for v in portfolio],
        markers=markers,
    )


def process_ticker_backtest(args):
    (
        ticker,
        market_data,
        db_path,
        rules,
        lookback_days,
        capital_per_ticker,
        include_charts,
        hist_df,
        q_df,
        a_df,
    ) = args
    if hist_df is not None and not hist_df.empty:
        df = hist_df.copy()
    else:
        try:
            with db_connection(db_path, readonly=True) as conn:
                df = pd.read_sql(
                    'SELECT Date, Open, High, Low, Close, "Adj Close", Volume '
                    "FROM stock_history WHERE Ticker = ?",
                    conn,
                    params=(ticker,),
                )
        except sqlite3.Error:
            return None
        if df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]

    if len(df) < MIN_WARMUP_DAYS:
        return None
    market_data = market_data[~market_data.index.duplicated(keep="last")]
    ohlcv = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df = df.join(market_data, rsuffix="_Mkt", how="inner")
    if df.empty:
        return None

    period_start, period_end = period_bounds(df, lookback_days)

    if q_df is None or a_df is None:
        q_df, a_df = get_earnings_data(ticker, db_path)
    ohlcv_aligned = ohlcv.reindex(df.index)
    df, fund_meta = build_canslim_frame(df, q_df, a_df, rules=rules, ohlcv=ohlcv_aligned)
    fund_note = (
        f"C: {fund_meta.get('C', '?')}, A: {fund_meta.get('A', '?')}; "
        f"rules={rules.version}; "
        f"{len(q_df)} quarterly / {len(a_df)} annual EPS rows in DB."
    )

    trades: list[dict] = []
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_triggers = ""
    entry_flags: dict[str, bool] = {}
    peak_gain = 0.0
    entry_base_type = ""
    entry_pattern_quality = 0.0
    prices = df["Adj Close"].values
    buy_edges = df["Buy_Edge"].values
    sma50 = df["SMA50"].values
    pass_m = df["Pass_M"].values
    dates = df.index

    for i in range(len(prices)):
        if i < MIN_WARMUP_DAYS:
            continue
        price = float(prices[i])
        date = dates[i]
        if position == 0:
            if buy_edges[i]:
                position = 1
                entry_price = price
                entry_date = date
                flags = _entry_flags_row(df, i)
                entry_flags = flags
                extra = []
                if flags.get("P"):
                    extra.append("P")
                entry_triggers = flags_to_trigger_string(flags) + (
                    f" [{','.join(extra)}]" if extra else ""
                )
                peak_gain = 0.0
                entry_base_type = str(df["Base_Type"].iloc[i]) if "Base_Type" in df.columns else ""
                entry_pattern_quality = float(df["Pattern_Quality"].iloc[i]) if "Pattern_Quality" in df.columns else 0.0
        elif position == 1:
            pct_change = (price - entry_price) / entry_price
            peak_gain = max(peak_gain, pct_change)
            sma_val = sma50[i]
            below_sma = not pd.isna(sma_val) and price < float(sma_val)
            market_off = not bool(pass_m[i])
            should_exit, reason = evaluate_exit(
                pct_change,
                price,
                entry_price,
                peak_gain,
                rules,
                below_sma=below_sma,
                market_off=market_off,
            )
            if should_exit:
                position = 0
                trades.append(
                    {
                        "Ticker": ticker,
                        "Entry_Date": entry_date,
                        "Exit_Date": date,
                        "Entry_Price": entry_price,
                        "Exit_Price": price,
                        "Return": pct_change,
                        "Return_Pct": pct_change * 100,
                        "Reason": reason,
                        "Sell_Rule": reason,
                        "Entry_Triggers": entry_triggers,
                        "Base_Type": entry_base_type,
                        "Pattern_Quality": entry_pattern_quality,
                        "Rule_Version": rules.version,
                        **{f"Entry_{k}": entry_flags.get(k, False) for k in "CANSLM"},
                        "Entry_P": entry_flags.get("P", False),
                    }
                )
                entry_triggers = ""
                entry_flags = {}

    if position == 1:
        final_price = float(prices[-1])
        pct_change = (final_price - entry_price) / entry_price
        trades.append(
            {
                "Ticker": ticker,
                "Entry_Date": entry_date,
                "Exit_Date": dates[-1],
                "Entry_Price": entry_price,
                "Exit_Price": final_price,
                "Return": pct_change,
                "Return_Pct": pct_change * 100,
                "Reason": "End of Data",
                "Sell_Rule": "End of Data",
                "Entry_Triggers": entry_triggers,
                "Base_Type": entry_base_type,
                "Pattern_Quality": entry_pattern_quality,
                "Rule_Version": rules.version,
                **{f"Entry_{k}": entry_flags.get(k, False) for k in "CANSLM"},
            }
        )

    trades_in_period = [
        t for t in trades if pd.Timestamp(t["Entry_Date"]) >= period_start
    ]

    chart_series = None
    if include_charts:
        window_df = df[df.index >= period_start]
        if not window_df.empty:
            chart_series = _build_chart_series(
                ticker, window_df, trades_in_period, capital_per_ticker
            )

    return {
        "ticker": ticker,
        "trades": trades_in_period,
        "chart": chart_series,
        "period_start": period_start,
        "period_end": period_end,
        "fundamentals_note": fund_note,
    }


def run_canslim_backtest(
    db_path: str,
    market_ticker: str = "^DJI",
    stop_loss: float = -0.08,
    take_profit: float = 0.25,
    tickers: list[str] | None = None,
    universe_scope: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    initial_capital: float = 1.0,
    progress_callback: ProgressCallback | None = None,
    use_parallel: bool = True,
    workers: int | None = None,
    bulk_preload: bool = True,
    *,
    rule_set_version: str | None = None,
    require_pattern: bool = False,
) -> CanslimBacktestResult | None:
    rules = rule_set_from_config(
        stop_loss=stop_loss,
        take_profit=take_profit,
        require_pattern=require_pattern,
        version=rule_set_version,
    )
    market_data = load_market_data(db_path, market_ticker)
    if market_data is None:
        return None

    if tickers:
        all_tickers = [t for t in tickers if t != market_ticker]
    else:
        all_tickers = resolve_universe(db_path, universe_scope)
        if market_ticker in all_tickers:
            all_tickers.remove(market_ticker)

    include_charts = bool(tickers)
    n_chart = len(all_tickers) if include_charts else 0
    capital_each = initial_capital / n_chart if n_chart else initial_capital

    fundamentals: dict = {}
    history: dict[str, pd.DataFrame] = {}
    if bulk_preload and len(all_tickers) > 1:
        fundamentals = load_fundamentals_eps_by_ticker(db_path)
        history = load_history_grouped(db_path, tickers=all_tickers, full_ohlcv=True)

    tasks = []
    for t in all_tickers:
        q_pre, a_pre = fundamentals.get(t, (None, None)) if fundamentals else (None, None)
        tasks.append(
            (
                t,
                market_data,
                db_path,
                rules,
                lookback_days,
                capital_each,
                include_charts,
                history.get(t),
                q_pre,
                a_pre,
            )
        )
    all_trades: list[dict] = []
    chart_series: dict[str, TickerChartSeries] = {}
    period_start = None
    period_end = None
    fundamentals_notes: list[str] = []

    if progress_callback:
        progress_callback(0.0 if tasks else 1.0)
    w_count = workers or default_worker_count()
    if use_parallel and len(tasks) > 1:
        for i, result in enumerate(
            run_parallel_map(
                process_ticker_backtest,
                tasks,
                use_parallel=True,
                workers=w_count,
            )
        ):
            if result:
                all_trades.extend(result["trades"])
                if result.get("chart"):
                    chart_series[result["chart"].ticker] = result["chart"]
                period_start = result.get("period_start") or period_start
                period_end = result.get("period_end") or period_end
                if result.get("fundamentals_note"):
                    fundamentals_notes.append(
                        f"{result.get('ticker', '?')}: {result['fundamentals_note']}"
                    )
            if progress_callback:
                progress_callback((i + 1) / len(tasks))
    else:
        for i, task in enumerate(tasks):
            result = process_ticker_backtest(task)
            if result:
                all_trades.extend(result["trades"])
                if result.get("chart"):
                    chart_series[result["chart"].ticker] = result["chart"]
                    if result.get("fundamentals_note"):
                        fundamentals_notes.append(
                            f"{result['chart'].ticker}: {result['fundamentals_note']}"
                        )
                elif result.get("fundamentals_note"):
                    fundamentals_notes.append(result["fundamentals_note"])
                period_start = result.get("period_start") or period_start
                period_end = result.get("period_end") or period_end
            if progress_callback:
                progress_callback((i + 1) / len(tasks))

    period_label = format_period(period_start, period_end)
    fund_summary = fundamentals_notes[0] if len(fundamentals_notes) == 1 else (
        "; ".join(fundamentals_notes[:3]) if fundamentals_notes else ""
    )
    combined_dates, combined_values = combine_portfolio_series(chart_series)

    if not all_trades:
        return CanslimBacktestResult(
            period_label=period_label,
            lookback_days=lookback_days,
            initial_capital=initial_capital,
            fundamentals_note=fund_summary,
            rule_set_version=rules.version,
            chart_series=chart_series,
            combined_portfolio_dates=combined_dates,
            combined_portfolio_values=combined_values,
        )

    df_res = pd.DataFrame(all_trades)
    if not df_res.empty and "Entry_Price" in df_res.columns and "Exit_Price" in df_res.columns:
        from src.analysis.signal_quality import adjust_trade_return, cost_model_from_config, record_calibration, summarize_trades

        cost = cost_model_from_config()
        df_res["Return"] = df_res.apply(
            lambda row: adjust_trade_return(
                float(row["Entry_Price"]),
                float(row["Exit_Price"]),
                cost=cost,
            ),
            axis=1,
        )
        df_res["Return_Pct"] = df_res["Return"] * 100
        try:
            record_calibration(
                db_path,
                "canslim",
                summarize_trades(df_res.to_dict("records"), cost=cost),
                lookback_days=lookback_days,
            )
        except Exception:
            pass
    display_cols = [
        "Ticker",
        "Entry_Date",
        "Exit_Date",
        "Entry_Triggers",
        "Reason",
        "Sell_Rule",
        "Base_Type",
        "Entry_Price",
        "Exit_Price",
        "Return_Pct",
    ]
    for col in display_cols:
        if col not in df_res.columns:
            df_res[col] = ""
    trades_display = df_res[display_cols].copy()
    trades_display["Entry_Date"] = pd.to_datetime(trades_display["Entry_Date"]).dt.strftime(
        "%Y-%m-%d"
    )
    trades_display["Exit_Date"] = pd.to_datetime(trades_display["Exit_Date"]).dt.strftime(
        "%Y-%m-%d"
    )
    trades_display["Return_Pct"] = trades_display["Return_Pct"].map(lambda v: f"{v:.2f}%")

    total = len(df_res)
    win_rate = len(df_res[df_res["Return"] > 0]) / total * 100 if total else 0.0
    avg_return = float(df_res["Return"].mean() * 100) if total else 0.0
    ticker_stats = df_res.groupby("Ticker")["Return"].agg(["count", "mean", "sum"])
    ticker_stats.columns = ["Trade_Count", "Avg_Return", "Total_Return"]
    ticker_stats["Avg_Return"] = (ticker_stats["Avg_Return"] * 100).round(2)
    ticker_stats["Total_Return"] = (ticker_stats["Total_Return"] * 100).round(2)

    return CanslimBacktestResult(
        trades_df=trades_display,
        total_trades=total,
        win_rate=win_rate,
        avg_return_pct=avg_return,
        ticker_stats=ticker_stats.reset_index(),
        period_label=period_label,
        lookback_days=lookback_days,
        initial_capital=initial_capital,
        fundamentals_note=fund_summary,
        rule_set_version=rules.version,
        chart_series=chart_series,
        combined_portfolio_dates=combined_dates,
        combined_portfolio_values=combined_values,
    )
