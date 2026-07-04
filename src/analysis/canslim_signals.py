"""CANSLIM signal construction for backtests (point-in-time, no lookahead)."""

from __future__ import annotations

import pandas as pd

from src.analysis.canslim_rulebook import CanslimRuleSet, get_rule_set
from src.analysis.patterns.cup_with_handle import detect_cup_with_handle_series

# Legacy module-level constants (defaults match oneil_v1 rulebook).
_DEFAULT = get_rule_set()
EARNINGS_C_VALID_DAYS = _DEFAULT.earnings.c_valid_days
EARNINGS_A_VALID_DAYS = _DEFAULT.earnings.a_valid_days
MIN_QUARTERS_FOR_EARNINGS = _DEFAULT.earnings.min_quarters
C_YOY_THRESHOLD = _DEFAULT.earnings.c_yoy_threshold
C_PROXY_6M_RETURN = _DEFAULT.earnings.c_proxy_6m_return
A_PROXY_1Y_RETURN = _DEFAULT.earnings.a_proxy_1y_return


def _quarterly_yoy_growth(q_df: pd.DataFrame) -> pd.DataFrame:
    """Add Growth_YoY and Growth_QoQ per report row."""
    q = q_df.copy()
    q["Report_Date"] = pd.to_datetime(q["Report_Date"])
    q = q.sort_values("Report_Date")
    q["Period"] = q["Report_Date"].dt.to_period("Q")
    by_period = q.set_index("Period")["Value"]
    if by_period.index.has_duplicates:
        by_period = by_period[~by_period.index.duplicated(keep="last")]
    yoy_vals = []
    qoq_vals = []
    for _, row in q.iterrows():
        p = row["Period"]
        p_yoy = p - 4
        p_qoq = p - 1
        if p_yoy in by_period.index and by_period[p_yoy] != 0:
            yoy_vals.append(row["Value"] / by_period[p_yoy] - 1)
        else:
            yoy_vals.append(float("nan"))
        if p_qoq in by_period.index and by_period[p_qoq] != 0:
            qoq_vals.append(row["Value"] / by_period[p_qoq] - 1)
        else:
            qoq_vals.append(float("nan"))
    q["Growth_YoY"] = yoy_vals
    q["Growth_QoQ"] = qoq_vals
    return q


def _apply_earnings_window(
    index: pd.DatetimeIndex,
    report_dates: pd.Series,
    growth: pd.Series,
    *,
    threshold: float,
    valid_days: int,
    use_qoq_if_no_yoy: bool = True,
) -> pd.Series:
    """Mark True from day after report through valid_days (point-in-time, no ffill)."""
    out = pd.Series(False, index=index)
    for report_date, yoy, qoq in zip(
        report_dates, growth["Growth_YoY"], growth["Growth_QoQ"]
    ):
        g = yoy
        if pd.isna(g) and use_qoq_if_no_yoy:
            g = qoq
        if pd.isna(g) or g <= threshold:
            continue
        start = pd.Timestamp(report_date) + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=valid_days)
        mask = (index >= start) & (index <= end)
        out.loc[mask] = True
    return out


def build_pass_c(
    index: pd.DatetimeIndex,
    q_df: pd.DataFrame,
    stock_6m_return: pd.Series,
    rules: CanslimRuleSet | None = None,
) -> tuple[pd.Series, str]:
    """[C] Current quarterly earnings growth."""
    rules = rules or get_rule_set()
    er = rules.earnings
    if q_df is None or q_df.empty or len(q_df) < er.min_quarters:
        proxy = stock_6m_return > er.c_proxy_6m_return
        return proxy.reindex(index, fill_value=False), "proxy_6m_return"

    q = _quarterly_yoy_growth(q_df)
    return (
        _apply_earnings_window(
            index,
            q["Report_Date"],
            q,
            threshold=er.c_yoy_threshold,
            valid_days=er.c_valid_days,
        ),
        "quarterly_eps",
    )


def build_pass_a(
    index: pd.DatetimeIndex,
    a_df: pd.DataFrame,
    stock_1y_return: pd.Series,
    rules: CanslimRuleSet | None = None,
) -> tuple[pd.Series, str]:
    """[A] Annual earnings growth — report windows or 1y price proxy."""
    rules = rules or get_rule_set()
    er = rules.earnings
    if a_df is None or a_df.empty:
        proxy = stock_1y_return > er.a_proxy_1y_return
        return proxy.reindex(index, fill_value=False), "proxy_1y_return"

    a = a_df.copy()
    a["Report_Date"] = pd.to_datetime(a["Report_Date"])
    a = a.sort_values("Report_Date")
    a["Growth_A"] = a["Value"].pct_change(periods=1, fill_method=None)
    out = pd.Series(False, index=index)
    for _, row in a.iterrows():
        g = row["Growth_A"]
        if pd.isna(g) or g <= 0:
            continue
        start = pd.Timestamp(row["Report_Date"]) + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=er.a_valid_days)
        mask = (index >= start) & (index <= end)
        out.loc[mask] = True
    if out.any():
        return out, "annual_eps"
    proxy = stock_1y_return > er.a_proxy_1y_return
    return proxy.reindex(index, fill_value=False), "proxy_1y_return"


def build_canslim_frame(
    df: pd.DataFrame,
    q_df: pd.DataFrame,
    a_df: pd.DataFrame,
    *,
    rules: CanslimRuleSet | None = None,
    ohlcv: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Add CANSLIM columns to price/market joined frame.
    Returns frame and metadata describing C/A sources.
    """
    rules = rules or get_rule_set()
    er = rules.earnings
    tr = rules.technical
    out = df.copy()
    meta: dict[str, str] = {"rule_version": rules.version}

    out["High_52"] = out["Adj Close"].rolling(window=252).max()
    out["Pass_N"] = out["Adj Close"] >= (out["High_52"] * tr.near_high_pct)
    out["Stock_Ret_6m"] = out["Adj Close"].pct_change(tr.rs_lookback_days, fill_method=None)
    out["Mkt_Ret_6m"] = out["Adj Close_Mkt"].pct_change(tr.rs_lookback_days, fill_method=None)
    out["Pass_L"] = out["Stock_Ret_6m"] > out["Mkt_Ret_6m"]
    out["Pass_M"] = out["Uptrend"]
    out["Vol_SMA50"] = out["Volume"].rolling(window=tr.sma50_days).mean()
    out["Pass_S"] = out["Volume"] > (out["Vol_SMA50"] * tr.volume_surge_mult)
    out["SMA50"] = out["Adj Close"].rolling(window=tr.sma50_days).mean()
    out["Stock_Ret_1y"] = out["Adj Close"].pct_change(252, fill_method=None)

    out["Pass_C"], meta["C"] = build_pass_c(out.index, q_df, out["Stock_Ret_6m"], rules)
    out["Pass_A"], meta["A"] = build_pass_a(out.index, a_df, out["Stock_Ret_1y"], rules)

    c_a_overlap = (out["Pass_C"] & out["Pass_A"]).any()
    sparse_earnings = q_df is None or len(q_df) < er.min_quarters
    if sparse_earnings or not c_a_overlap:
        out["Pass_C"] = out["Pass_C"] | (out["Stock_Ret_6m"] > er.c_proxy_6m_return)
        out["Pass_A"] = out["Pass_A"] | (out["Stock_Ret_1y"] > er.a_proxy_1y_return)
        meta["C"] = f"{meta['C']}+proxy" if not sparse_earnings else meta["C"]
        meta["A"] = f"{meta['A']}+proxy" if not sparse_earnings else meta["A"]

    pattern_src = ohlcv if ohlcv is not None and len(ohlcv) == len(out) else out
    if "High" not in pattern_src.columns:
        pattern_src = pattern_src.copy()
        pattern_src["High"] = out["Adj Close"]
        pattern_src["Low"] = out["Adj Close"]
    pat_df = detect_cup_with_handle_series(pattern_src, rules=rules.cup_with_handle)
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
        pattern_src = pattern_src[~pattern_src.index.duplicated(keep="last")]
        pat_df = detect_cup_with_handle_series(pattern_src, rules=rules.cup_with_handle)
    out["Pass_Pattern"] = pat_df["Pass_Pattern"].reindex(out.index, fill_value=False)
    out["Pivot"] = pat_df["Pivot"].reindex(out.index)
    out["Pattern_Quality"] = pat_df["Pattern_Quality"].reindex(out.index, fill_value=0.0)
    out["Base_Type"] = pat_df["Base_Type"].reindex(out.index, fill_value="")
    meta["pattern"] = "cup_with_handle"

    out["Setup_Good"] = (
        out["Pass_C"] & out["Pass_A"] & out["Pass_N"] & out["Pass_L"] & out["Pass_M"]
    )
    if rules.entry.require_pattern:
        out["Setup_Good"] = out["Setup_Good"] & out["Pass_Pattern"]

    breakout_vol = out["Volume"] > (out["Vol_SMA50"] * tr.breakout_volume_mult)
    pivot_break = out["Adj Close"] >= out["Pivot"].fillna(float("inf"))
    if rules.entry.require_volume_on_breakout:
        out["Buy_Signal"] = out["Setup_Good"] & (out["Pass_S"] | (out["Pass_Pattern"] & breakout_vol & pivot_break))
    else:
        out["Buy_Signal"] = out["Setup_Good"] & out["Pass_S"]

    prev_buy = out["Buy_Signal"].shift(1)
    prev_buy = prev_buy.where(prev_buy.notna(), False).astype(bool)
    out["Buy_Edge"] = out["Buy_Signal"] & ~prev_buy

    return out, meta
