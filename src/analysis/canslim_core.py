"""Shared CANSLM constants and evaluation used by Leaderboard and Stock Detail."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analysis.canslim_rulebook import get_rule_set
from src.analysis.canslim_signals import build_canslim_frame
from src.analysis.db import db_connection, load_ohlcv, load_price_data

CANSLM_RULE_COUNT = 6
CANSLM_PASS_COLUMNS = (
    "Pass_C",
    "Pass_A",
    "Pass_N",
    "Pass_S",
    "Pass_L",
    "Pass_M",
)
CANSLM_LETTERS = ("C", "A", "N", "S", "L", "M")

# Back-compat aliases (internal module names still use canslim_*).
CANSLIM_RULE_COUNT = CANSLM_RULE_COUNT
CANSLIM_PASS_COLUMNS = CANSLM_PASS_COLUMNS

MARKET_INDEX_LABELS: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "Nasdaq",
}


def canslim_score_from_row(row: pd.Series) -> int:
    return sum(
        1 for c in CANSLM_PASS_COLUMNS if c in row.index and bool(row[c])
    )


def canslm_passes_from_row(row: pd.Series) -> dict[str, bool]:
    """Map each CANSLM letter to pass/fail on the latest bar."""
    return {
        letter: bool(row.get(f"Pass_{letter}", False))
        for letter in CANSLM_LETTERS
    }


def format_canslm_tooltip(lines: list[str]) -> str:
    """Hover text for CANSLM score cells (one line per letter rule)."""
    if not lines:
        return "CANSLM letter ratings unavailable."
    return "\n".join(lines)


def price_vs_sma50(df: pd.DataFrame | None, *, window: int = 50) -> tuple[float, float] | None:
    """Latest adjusted close and SMA using the same ffill logic as dashboard charts."""
    if df is None or df.empty or "Adj Close" not in df.columns:
        return None
    adj = df["Adj Close"].astype(float).ffill()
    if len(adj) < window:
        return None
    sma = adj.rolling(window, min_periods=window).mean().iloc[-1]
    price = adj.iloc[-1]
    if pd.isna(sma) or pd.isna(price):
        return None
    return float(price), float(sma)


def enrich_market_frame(market_df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA50 and Uptrend columns using ffill-aware logic."""
    mkt = market_df.copy()
    adj = mkt["Adj Close"].astype(float).ffill()
    window = get_rule_set().technical.sma50_days
    mkt["SMA50"] = adj.rolling(window, min_periods=window).mean()
    mkt["Uptrend"] = adj > mkt["SMA50"]
    return mkt


def join_market_frame(
    df: pd.DataFrame,
    market_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Align stock prices with market benchmark and uptrend flag."""
    if market_df is not None and not market_df.empty:
        mkt = enrich_market_frame(market_df)
        joined = df.join(
            mkt[["Adj Close", "Uptrend"]].rename(columns={"Adj Close": "Adj Close_Mkt"}),
            how="inner",
        )
    else:
        joined = df.copy()
        joined["Adj Close_Mkt"] = joined["Adj Close"]
        joined["Uptrend"] = True
    if joined.index.has_duplicates:
        joined = joined[~joined.index.duplicated(keep="last")]
    return joined


def evaluate_risk_flags(
    *,
    price: float,
    stock_sma50: float,
    pass_m: bool,
    near_high: float,
    near_high_threshold: float = 0.75,
) -> str:
    """Risk labels shared by Leaderboard and Stock Detail."""
    if stock_sma50 > 0 and price < stock_sma50:
        return "Below 50-day simple moving average"
    if not pass_m:
        return "Market weak"
    if near_high < near_high_threshold:
        return "Extended below highs"
    return ""


def market_direction_line(market_ticker: str, pass_m: bool) -> str:
    label = MARKET_INDEX_LABELS.get(market_ticker, market_ticker)
    if pass_m:
        return f"[M] Market Direction: PASS ({label} above 50-day simple moving average)"
    return f"[M] Market Direction: FAIL ({label} below 50-day simple moving average)"


def _format_canslim_lines(
    last: pd.Series,
    *,
    market_ticker: str,
    meta: dict[str, str],
    price: float,
    near_high: float,
    stock_ret: float,
    mkt_ret: float,
    vol_ratio: float,
) -> list[str]:
    rules = get_rule_set()
    near_high_pct = rules.technical.near_high_pct
    high_52 = float(last.get("High_52", 0) or 0)

    lines: list[str] = []
    c_src = meta.get("C", "quarterly_eps")
    if bool(last.get("Pass_C", False)):
        lines.append(f"[C] Current Earnings: PASS ({c_src})")
    else:
        lines.append(f"[C] Current Earnings: FAIL ({c_src})")

    a_src = meta.get("A", "annual_eps")
    if bool(last.get("Pass_A", False)):
        lines.append(f"[A] Annual Growth: PASS ({a_src})")
    else:
        lines.append(f"[A] Annual Growth: FAIL ({a_src})")

    if bool(last.get("Pass_N", False)):
        lines.append(
            f"[N] Near Highs: PASS (Price {price:.2f}, within {near_high_pct:.0%} of 52w high {high_52:.2f})"
        )
    else:
        lines.append(
            f"[N] Near Highs: FAIL (Price {price:.2f}, >{1 - near_high_pct:.0%} below 52w high {high_52:.2f})"
        )

    if bool(last.get("Pass_S", False)):
        lines.append(f"[S] Supply/Demand: PASS (Volume {vol_ratio:.2f}x 50-day average)")
    else:
        lines.append(f"[S] Supply/Demand: FAIL (Volume below surge threshold)")

    if bool(last.get("Pass_L", False)):
        lines.append(
            f"[L] Leader: PASS (Stock +{stock_ret * 100:.1f}% vs Market +{mkt_ret * 100:.1f}% over 6 months)"
        )
    else:
        lines.append("[L] Leader: FAIL (Lagging the market over 6 months)")

    lines.append(market_direction_line(market_ticker, bool(last.get("Pass_M", False))))
    return lines


def _load_eps_frames(
    ticker: str,
    db_path: str,
    *,
    q_df: pd.DataFrame | None,
    a_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    if q_df is not None and a_df is not None:
        return q_df, a_df, not q_df.empty or not a_df.empty

    q_df = q_df if q_df is not None else pd.DataFrame()
    a_df = a_df if a_df is not None else pd.DataFrame()
    has_eps = not q_df.empty or not a_df.empty
    if has_eps:
        return q_df, a_df, True

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
            has_eps = not q_df.empty or not a_df.empty
    except Exception:
        pass
    return q_df, a_df, has_eps


@dataclass
class CanslimBarEvaluation:
    ticker: str
    score: int
    max_score: int
    lines: list[str] = field(default_factory=list)
    risk_flag: str = ""
    price: float = 0.0
    near_high: float = 0.0
    rs_pct: float = 0.0
    vol_ratio: float = 0.0
    pass_setup: bool = False
    pass_pattern: bool = False
    pattern_quality: float = 0.0
    last_row: pd.Series | None = None
    meta: dict[str, str] = field(default_factory=dict)
    has_eps_data: bool = False
    history_bars: int = 0
    error: str | None = None

    @property
    def verdict(self) -> str:
        if self.score >= 5:
            return "STRONG BUY CANDIDATE"
        if self.score >= 4:
            return "WATCHLIST"
        return "PASS"


def evaluate_canslim_bar(
    ticker: str,
    db_path: str,
    market_ticker: str,
    *,
    market_df: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
    q_df: pd.DataFrame | None = None,
    a_df: pd.DataFrame | None = None,
    price_df: pd.DataFrame | None = None,
    min_history_bars: int = 60,
) -> CanslimBarEvaluation | None:
    """Evaluate the latest CANSLM bar using the same pipeline as the Leaderboard."""
    sym = ticker.upper().strip()
    if price_df is not None:
        df = price_df
    else:
        df = load_price_data(sym, db_path, 'Date, "Adj Close", Volume')
    if market_df is None:
        market_df = load_price_data(market_ticker, db_path)
    if ohlcv is None:
        ohlcv = load_ohlcv(sym, db_path)

    if df is None or len(df) < min_history_bars:
        return None

    joined = join_market_frame(df, market_df)
    if joined.empty:
        return None

    q_df, a_df, has_eps = _load_eps_frames(sym, db_path, q_df=q_df, a_df=a_df)
    ohlcv_use = ohlcv.reindex(joined.index) if ohlcv is not None else joined
    frame, meta = build_canslim_frame(joined, q_df, a_df, ohlcv=ohlcv_use)
    last = frame.iloc[-1]

    score = canslim_score_from_row(last)
    pattern_q = float(last.get("Pattern_Quality", 0) or 0)
    high_52 = float(last.get("High_52", 0) or 0)
    price = float(last["Adj Close"])
    near_high = (price / high_52) if high_52 > 0 else 0.0
    vol_sma = float(last.get("Vol_SMA50", 0) or 1)
    vol_ratio = float(last["Volume"]) / vol_sma if vol_sma > 0 else 0.0
    stock_ret = float(last.get("Stock_Ret_6m", 0) or 0)
    mkt_ret = float(last.get("Mkt_Ret_6m", 0) or 0)
    rs_pct = (stock_ret - mkt_ret) * 100 if not pd.isna(stock_ret) else 0.0
    sma50 = float(last.get("SMA50", 0) or 0)
    pass_m = bool(last.get("Pass_M", False))

    risk = evaluate_risk_flags(
        price=price,
        stock_sma50=sma50,
        pass_m=pass_m,
        near_high=near_high,
    )

    lines = _format_canslim_lines(
        last,
        market_ticker=market_ticker,
        meta=meta,
        price=price,
        near_high=near_high,
        stock_ret=stock_ret,
        mkt_ret=mkt_ret,
        vol_ratio=vol_ratio,
    )

    return CanslimBarEvaluation(
        ticker=sym,
        score=score,
        max_score=CANSLM_RULE_COUNT,
        lines=lines,
        risk_flag=risk,
        price=price,
        near_high=near_high,
        rs_pct=rs_pct,
        vol_ratio=vol_ratio,
        pass_setup=bool(last.get("Setup_Good", False)),
        pass_pattern=bool(last.get("Pass_Pattern", False)),
        pattern_quality=pattern_q,
        last_row=last,
        meta=meta,
        has_eps_data=has_eps,
        history_bars=len(df),
    )
