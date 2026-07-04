"""Backtest and compare leaderboard score versions (v1 vs v2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.analysis.leaderboard import build_leaderboard
from src.analysis.leaderboard_scoring import SCORE_VERSION_V1, SCORE_VERSION_V2


def compare_score_versions(
    db_path: str,
    *,
    tickers: list[str] | None = None,
    market_ticker: str = "^DJI",
    limit: int = 500,
) -> dict[str, Any]:
    """
    Score the same universe with v1 and v2; return correlation and decile stats.

    Used to gate rollout: v2 should not invert rankings without justification.
    """
    v1_df = build_leaderboard(
        db_path,
        market_ticker,
        tickers=tickers,
        segment=None,
        limit=limit,
        score_version=SCORE_VERSION_V1,
        use_parallel=True,
    )
    v2_df = build_leaderboard(
        db_path,
        market_ticker,
        tickers=tickers,
        segment=None,
        limit=limit,
        score_version=SCORE_VERSION_V2,
        use_parallel=True,
    )
    if v1_df.empty or v2_df.empty:
        return {"ok": False, "reason": "empty_universe", "count": 0}

    merged = v1_df.merge(
        v2_df,
        on="ticker",
        suffixes=("_v1", "_v2"),
        how="inner",
    )
    if merged.empty:
        return {"ok": False, "reason": "no_overlap", "count": 0}

    corr = merged["composite_score_v1"].corr(merged["composite_score_v2"])
    buy_top = merged.nlargest(max(1, len(merged) // 10), "buy_readiness")["ticker"].tolist()
    sell_top = merged.nlargest(max(1, len(merged) // 10), "sell_pressure")["ticker"].tolist()
    mean_delta = (merged["composite_score_v2"] - merged["composite_score_v1"]).mean()

    return {
        "ok": True,
        "count": len(merged),
        "correlation_v1_v2": round(float(corr), 4) if corr == corr else None,
        "mean_composite_delta_v2_minus_v1": round(float(mean_delta), 2),
        "top_buy_readiness": buy_top[:10],
        "top_sell_pressure": sell_top[:10],
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def persist_calibration_report(db_path: str, report: dict[str, Any]) -> None:
    """Store calibration summary in signal_calibration table."""
    ensure_intelligence_schema(db_path)
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
                    "leaderboard_v1_v2",
                    0,
                    0.0,
                    0.0,
                    None,
                    report.get("mean_composite_delta_v2_minus_v1"),
                    None,
                    report.get("count", 0),
                    json.dumps(report),
                ),
            )
            conn.commit()


def rollout_gate_passes(report: dict[str, Any], *, min_correlation: float = 0.5) -> bool:
    """True when v1/v2 correlation is stable enough to default v2."""
    if not report.get("ok"):
        return False
    corr = report.get("correlation_v1_v2")
    if corr is None:
        return False
    return float(corr) >= min_correlation and int(report.get("count") or 0) >= 5
