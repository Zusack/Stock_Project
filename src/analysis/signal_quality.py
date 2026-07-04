"""Backtest realism helpers and signal calibration metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.services.stock_config import stock_config


@dataclass
class BacktestCostModel:
    slippage_bps: float = 5.0
    fee_per_trade: float = 0.0
    spread_bps: float = 2.0

    def apply_entry(self, price: float) -> float:
        """Worsen buy fill."""
        mult = 1.0 + (self.slippage_bps + self.spread_bps) / 10000.0
        return price * mult + self.fee_per_trade

    def apply_exit(self, price: float) -> float:
        """Worsen sell fill."""
        mult = 1.0 - (self.slippage_bps + self.spread_bps) / 10000.0
        return price * mult - self.fee_per_trade


def cost_model_from_config() -> BacktestCostModel:
    cfg = stock_config()
    return BacktestCostModel(
        slippage_bps=cfg.backtest_slippage_bps,
        fee_per_trade=cfg.backtest_fee_per_trade,
        spread_bps=cfg.backtest_spread_bps,
    )


def adjust_trade_return(
    entry_price: float,
    exit_price: float,
    *,
    cost: BacktestCostModel | None = None,
) -> float:
    """Return fractional P&L after slippage/fees."""
    cost = cost or cost_model_from_config()
    if entry_price <= 0:
        return 0.0
    adj_entry = cost.apply_entry(entry_price)
    adj_exit = cost.apply_exit(exit_price)
    return (adj_exit / adj_entry) - 1.0


def summarize_trades(
    trades: list[dict],
    *,
    cost: BacktestCostModel | None = None,
) -> dict[str, float]:
    """Aggregate win rate, avg return, max drawdown on trade list."""
    cost = cost or cost_model_from_config()
    if not trades:
        return {"win_rate": 0.0, "avg_return": 0.0, "max_drawdown": 0.0, "sample_count": 0}

    returns: list[float] = []
    for t in trades:
        ep = float(t.get("Entry_Price") or t.get("entry_price") or 0)
        xp = float(t.get("Exit_Price") or t.get("exit_price") or 0)
        if ep > 0 and xp > 0:
            returns.append(adjust_trade_return(ep, xp, cost=cost))
        elif "Return" in t:
            returns.append(float(t["Return"]))

    if not returns:
        return {"win_rate": 0.0, "avg_return": 0.0, "max_drawdown": 0.0, "sample_count": 0}

    wins = sum(1 for r in returns if r > 0)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak else 0.0
        max_dd = max(max_dd, dd)

    return {
        "win_rate": round(wins / len(returns), 4),
        "avg_return": round(sum(returns) / len(returns), 4),
        "max_drawdown": round(max_dd, 4),
        "sample_count": len(returns),
    }


def record_calibration(
    db_path: str,
    strategy_name: str,
    metrics: dict[str, float],
    *,
    lookback_days: int = 365,
    payload: dict | None = None,
) -> None:
    ensure_intelligence_schema(db_path)
    cfg = stock_config()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                INSERT INTO signal_calibration
                (recorded_at, strategy_name, lookback_days, slippage_bps, fee_per_trade,
                 win_rate, avg_return, max_drawdown, sample_count, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    strategy_name,
                    lookback_days,
                    cfg.backtest_slippage_bps,
                    cfg.backtest_fee_per_trade,
                    metrics.get("win_rate", 0),
                    metrics.get("avg_return", 0),
                    metrics.get("max_drawdown", 0),
                    int(metrics.get("sample_count", 0)),
                    json.dumps(payload or {}),
                ),
            )
            conn.commit()


def load_calibration_history(
    db_path: str,
    strategy_name: str,
    *,
    limit: int = 20,
) -> pd.DataFrame:
    ensure_intelligence_schema(db_path)
    with db_connection(db_path, readonly=True) as conn:
        return pd.read_sql(
            """
            SELECT * FROM signal_calibration
            WHERE strategy_name = ?
            ORDER BY recorded_at DESC LIMIT ?
            """,
            conn,
            params=(strategy_name, limit),
        )


def calibration_drift(
    db_path: str,
    strategy_name: str,
    *,
    window: int = 5,
) -> dict[str, Any]:
    """Compare recent vs prior calibration windows for drift detection."""
    df = load_calibration_history(db_path, strategy_name, limit=window * 2)
    if len(df) < 2:
        return {"drift_detected": False, "message": "Insufficient calibration history"}
    recent = df.head(window)
    prior = df.iloc[window : window * 2] if len(df) >= window * 2 else df.iloc[window:]
    if prior.empty:
        return {"drift_detected": False, "message": "Need more history"}

    r_wr = float(recent["win_rate"].mean())
    p_wr = float(prior["win_rate"].mean())
    r_ret = float(recent["avg_return"].mean())
    p_ret = float(prior["avg_return"].mean())
    drift = abs(r_wr - p_wr) > 0.12 or abs(r_ret - p_ret) > 0.05
    return {
        "drift_detected": drift,
        "recent_win_rate": round(r_wr, 4),
        "prior_win_rate": round(p_wr, 4),
        "recent_avg_return": round(r_ret, 4),
        "prior_avg_return": round(p_ret, 4),
        "message": "Calibration drift detected — review strategy parameters" if drift else "Stable",
    }
