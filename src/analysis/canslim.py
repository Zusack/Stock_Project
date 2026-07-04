"""Interactive CANSLIM scoring for a single ticker (7 letters incl. Institutions).

Note: leaderboard/guidance scoring uses 6 pass columns (C,A,N,S,L,M) via
CANSLIM_PASS_COLUMNS in leaderboard_scoring.py. The extra I letter here
reflects live yfinance institutional data not used in batch rankings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yfinance as yf

from src.analysis.db import load_price_data


@dataclass
class CanslimResult:
    ticker: str
    score: int
    max_score: int = 7
    lines: list[str] = field(default_factory=list)
    verdict: str = "PASS"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "score": self.score,
            "max_score": self.max_score,
            "lines": self.lines,
            "verdict": self.verdict,
            "error": self.error,
        }


def get_fundamentals(ticker: str) -> tuple:
    try:
        stock = yf.Ticker(ticker)
        return stock.quarterly_financials, stock.financials, stock.info.get(
            "heldPercentInstitutions", 0
        )
    except Exception:
        return None, None, 0


def _verdict(score: int) -> str:
    if score >= 6:
        return "STRONG BUY CANDIDATE"
    if score >= 4:
        return "WATCHLIST"
    return "PASS"


def analyze_canslim(
    ticker: str,
    db_path: str,
    market_ticker: str = "^DJI",
    use_live_fundamentals: bool = True,
) -> CanslimResult:
    ticker = ticker.upper().strip()
    df = load_price_data(ticker, db_path)
    market_df = load_price_data(market_ticker, db_path)

    if df is None or len(df) < 250:
        return CanslimResult(
            ticker=ticker,
            score=0,
            lines=[],
            verdict="PASS",
            error="Insufficient price data in database (need 250+ days).",
        )

    q_fin, a_fin, inst_own = (None, None, 0)
    if use_live_fundamentals:
        q_fin, a_fin, inst_own = get_fundamentals(ticker)

    score = 0
    report: list[str] = []

    try:
        if q_fin is not None and not q_fin.empty:
            net_income_row = q_fin.loc[q_fin.index.str.contains("Net Income", case=False)]
            if not net_income_row.empty and net_income_row.iloc[0, 0] > 0:
                score += 1
                report.append("[C] Current Earnings: PASS (Positive recent income)")
            else:
                report.append("[C] Current Earnings: FAIL (Negative or missing)")
        else:
            report.append("[C] Current Earnings: N/A (No data)")
    except Exception:
        report.append("[C] Current Earnings: ERROR")

    try:
        if a_fin is not None and not a_fin.empty:
            income_row = a_fin.loc[a_fin.index.str.contains("Net Income", case=False)]
            if not income_row.empty:
                vals = income_row.iloc[0].values
                if len(vals) >= 3 and vals[0] > vals[-1]:
                    score += 1
                    report.append("[A] Annual Growth: PASS (Growth over 3 years)")
                else:
                    report.append("[A] Annual Growth: FAIL (No long term growth)")
        else:
            report.append("[A] Annual Growth: N/A")
    except Exception:
        report.append("[A] Annual Growth: ERROR")

    current_price = df["Adj Close"].iloc[-1]
    high_52 = df["Adj Close"].tail(252).max()
    if current_price >= (high_52 * 0.85):
        score += 1
        report.append(f"[N] Near Highs: PASS (Price {current_price:.2f}, High {high_52:.2f})")
    else:
        report.append(
            f"[N] Near Highs: FAIL (Price {current_price:.2f} >15% below High {high_52:.2f})"
        )

    recent = df.tail(20).copy()
    recent["Change"] = recent["Adj Close"].pct_change(fill_method=None)
    vol_up = recent[recent["Change"] > 0]["Volume"].mean()
    vol_down = recent[recent["Change"] < 0]["Volume"].mean()
    vol_up = 0 if np.isnan(vol_up) else vol_up
    vol_down = 0 if np.isnan(vol_down) else vol_down
    if vol_up > vol_down:
        score += 1
        report.append("[S] Supply/Demand: PASS (Volume higher on up days)")
    else:
        report.append("[S] Supply/Demand: FAIL (Selling pressure higher)")

    if market_df is not None:
        combined = df[["Adj Close"]].join(market_df[["Adj Close"]], rsuffix="_Mkt", how="inner")
        if not combined.empty and len(combined) > 200:
            stock_ret = (combined["Adj Close"].iloc[-1] / combined["Adj Close"].iloc[-200]) - 1
            mkt_ret = (combined["Adj Close_Mkt"].iloc[-1] / combined["Adj Close_Mkt"].iloc[-200]) - 1
            if stock_ret > mkt_ret:
                score += 1
                report.append(
                    f"[L] Leader: PASS (Stock +{stock_ret*100:.1f}% vs Market +{mkt_ret*100:.1f}%)"
                )
            else:
                report.append("[L] Leader: FAIL (Lagging the market)")
        else:
            report.append("[L] Leader: N/A (Insufficient aligned data)")
    else:
        report.append("[L] Leader: N/A (No market data)")

    if inst_own is not None and inst_own > 0.3:
        score += 1
        report.append(f"[I] Institutions: PASS ({inst_own*100:.1f}% ownership)")
    else:
        val = 0 if inst_own is None else inst_own
        report.append(f"[I] Institutions: WEAK ({val*100:.1f}% ownership)")

    if market_df is not None:
        mkt_price = market_df["Adj Close"].iloc[-1]
        mkt_sma50 = market_df["Adj Close"].rolling(window=50).mean().iloc[-1]
        if mkt_price > mkt_sma50:
            score += 1
            report.append("[M] Market Direction: PASS (Market in uptrend)")
        else:
            report.append(
                "[M] Market Direction: FAIL (Market below 50-day simple moving average)"
            )
    else:
        report.append("[M] Market Direction: N/A")

    return CanslimResult(
        ticker=ticker,
        score=score,
        lines=report,
        verdict=_verdict(score),
    )
