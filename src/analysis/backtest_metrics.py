"""Professional backtest metrics from equity curves and trade lists."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.signal_quality import BacktestCostModel, adjust_trade_return, cost_model_from_config


@dataclass
class MetricsReport:
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    volatility_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_days: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    expectancy_pct: float = 0.0
    exposure_pct: float = 0.0
    trade_count: int = 0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0
    beta: float = 0.0
    period_days: int = 0
    interpretation_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "volatility_pct": self.volatility_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_days": self.max_drawdown_days,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "expectancy_pct": self.expectancy_pct,
            "exposure_pct": self.exposure_pct,
            "trade_count": self.trade_count,
            "benchmark_return_pct": self.benchmark_return_pct,
            "alpha_pct": self.alpha_pct,
            "beta": self.beta,
            "period_days": self.period_days,
            "interpretation_hints": list(self.interpretation_hints),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricsReport:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def _max_drawdown_stats(equity: pd.Series) -> tuple[float, int]:
    if equity.empty:
        return 0.0, 0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    max_dd = float(dd.min()) if not dd.empty else 0.0
    # Duration: longest stretch below previous peak
    underwater = dd < 0
    max_days = 0
    current = 0
    for u in underwater:
        if u:
            current += 1
            max_days = max(max_days, current)
        else:
            current = 0
    return abs(max_dd) * 100.0, max_days


def compute_metrics(
    equity_dates: list[str],
    equity_values: list[float],
    *,
    trades: list[dict] | None = None,
    benchmark_dates: list[str] | None = None,
    benchmark_values: list[float] | None = None,
    in_market: pd.Series | None = None,
    cost: BacktestCostModel | None = None,
    risk_free_rate: float = 0.0,
) -> MetricsReport:
    """Compute full metrics report from equity curve and optional trades."""
    cost = cost or cost_model_from_config()
    report = MetricsReport()

    if not equity_dates or not equity_values or len(equity_dates) < 2:
        report.interpretation_hints.append("Not enough data to compute metrics.")
        return report

    idx = pd.to_datetime(equity_dates)
    equity = pd.Series(equity_values, index=idx).astype(float)
    equity = equity.sort_index()
    report.period_days = max(1, (equity.index[-1] - equity.index[0]).days)

    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val > 0:
        report.total_return_pct = (end_val / start_val - 1.0) * 100.0
        years = report.period_days / 365.25
        if years > 0 and end_val > 0:
            report.cagr_pct = ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0

    daily_returns = equity.pct_change(fill_method=None).dropna()
    if not daily_returns.empty:
        ann_factor = np.sqrt(252)
        vol = float(daily_returns.std() * ann_factor * 100.0)
        report.volatility_pct = vol
        excess = daily_returns - risk_free_rate / 252.0
        std = float(excess.std())
        if std > 0:
            report.sharpe = float(excess.mean() / std * ann_factor)
        downside = excess[excess < 0]
        down_std = float(downside.std()) if len(downside) else 0.0
        if down_std > 0:
            report.sortino = float(excess.mean() / down_std * ann_factor)

    report.max_drawdown_pct, report.max_drawdown_days = _max_drawdown_stats(equity)

    if in_market is not None and len(in_market) > 0:
        report.exposure_pct = float(in_market.astype(float).mean() * 100.0)

    trade_returns: list[float] = []
    if trades:
        for t in trades:
            ep = float(t.get("Entry_Price") or t.get("entry_price") or 0)
            xp = float(t.get("Exit_Price") or t.get("exit_price") or 0)
            if ep > 0 and xp > 0:
                trade_returns.append(adjust_trade_return(ep, xp, cost=cost) * 100.0)
            elif "Return_Pct" in t:
                trade_returns.append(float(t["Return_Pct"]))
            elif "return_pct" in t:
                trade_returns.append(float(t["return_pct"]))

    if trade_returns:
        report.trade_count = len(trade_returns)
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        report.win_rate_pct = len(wins) / len(trade_returns) * 100.0
        report.avg_win_pct = float(np.mean(wins)) if wins else 0.0
        report.avg_loss_pct = float(np.mean(losses)) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        report.expectancy_pct = float(np.mean(trade_returns))

    if benchmark_dates and benchmark_values and len(benchmark_dates) >= 2:
        b_idx = pd.to_datetime(benchmark_dates)
        bench = pd.Series(benchmark_values, index=b_idx).astype(float).sort_index()
        b_start = float(bench.iloc[0])
        b_end = float(bench.iloc[-1])
        if b_start > 0:
            report.benchmark_return_pct = (b_end / b_start - 1.0) * 100.0
        report.alpha_pct = report.total_return_pct - report.benchmark_return_pct
        aligned = pd.concat(
            [
                daily_returns.rename("strat"),
                bench.pct_change(fill_method=None).dropna().rename("bench"),
            ],
            axis=1,
        ).dropna()
        if len(aligned) > 1:
            cov = float(aligned["strat"].cov(aligned["bench"]))
            var = float(aligned["bench"].var())
            if var > 0:
                report.beta = cov / var

    report.interpretation_hints = _build_hints(report)
    return report


def _build_hints(report: MetricsReport) -> list[str]:
    hints: list[str] = []
    if report.sharpe < 1.0 and report.period_days > 30:
        hints.append(
            "Sharpe below 1.0 means returns may not have compensated for the volatility taken."
        )
    if report.max_drawdown_pct > 25:
        hints.append(
            f"Max drawdown of {report.max_drawdown_pct:.1f}% is substantial — consider tighter risk exits."
        )
    if report.exposure_pct < 30 and report.trade_count > 0:
        hints.append("Low market exposure — the strategy spends most of the time in cash.")
    if report.trade_count < 5 and report.period_days > 180:
        hints.append("Very few trades — results may not be statistically meaningful.")
    if report.alpha_pct > 0:
        hints.append(
            f"Strategy outperformed the benchmark by {report.alpha_pct:.1f} percentage points."
        )
    elif report.benchmark_return_pct != 0:
        hints.append(
            f"Strategy underperformed the benchmark by {abs(report.alpha_pct):.1f} percentage points."
        )
    return hints


def monthly_returns_grid(equity_dates: list[str], equity_values: list[float]) -> pd.DataFrame:
    """Pivot table of monthly returns (rows=years, cols=months)."""
    if not equity_dates or not equity_values:
        return pd.DataFrame()
    idx = pd.to_datetime(equity_dates)
    equity = pd.Series(equity_values, index=idx).sort_index()
    monthly = equity.resample("ME").last().pct_change(fill_method=None).dropna() * 100.0
    if monthly.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }
    )
    pivot = frame.pivot(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][: len(pivot.columns)]
    pivot["Year"] = pivot.sum(axis=1)
    return pivot.round(2)
