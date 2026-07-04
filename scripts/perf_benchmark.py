#!/usr/bin/env python3
"""Repeatable performance benchmarks for DB and analytics workflows.

Usage:
  python scripts/perf_benchmark.py --db market_data.db --tickers AAPL,MSFT,NVDA
  python scripts/perf_benchmark.py --db market_data.db --benchmark-registry
  python scripts/perf_benchmark.py --db market_data.db --skip-backtest --skip-leaderboard
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis import ticker_registry as registry
from src.analysis.canslim_backtest import run_canslim_backtest
from src.analysis.db import get_db_summary, load_prices_bulk
from src.analysis.db_perf import OPERATION_SLAS, check_sla, get_recent_db_ops
from src.analysis.leaderboard import LeaderboardSegment, build_leaderboard
from src.services.stock_config import stock_config


def _timed(label: str, fn) -> dict:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    meta: dict = {
        "label": label,
        "seconds": round(elapsed, 3),
        "sla_sec": OPERATION_SLAS.get(label),
        "within_sla": check_sla(label, elapsed),
    }
    if hasattr(result, "__len__"):
        try:
            meta["rows"] = len(result)
        except TypeError:
            pass
    if isinstance(result, dict):
        for key in ("removed_db", "skipped", "ticker_count"):
            if key in result:
                meta[key] = result[key]
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Stock project performance benchmarks")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers (subset)")
    parser.add_argument("--market", default=None, help="Market benchmark ticker")
    parser.add_argument("--leaderboard-limit", type=int, default=100)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-leaderboard", action="store_true")
    parser.add_argument("--benchmark-registry", action="store_true", help="Run registry/DB benchmarks")
    parser.add_argument("--index-tickers", default="^GSPC,^DJI,^IXIC", help="For bulk price load test")
    args = parser.parse_args()

    cfg = stock_config()
    db_path = args.db or cfg.db_path
    market = args.market or cfg.market_ticker
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None

    results: list[dict] = []

    results.append(_timed("get_db_summary", lambda: get_db_summary(db_path, light=True)))
    results.append(_timed("count_summary", lambda: registry.count_summary(db_path)))

    index_list = [t.strip() for t in args.index_tickers.split(",") if t.strip()]
    if index_list:
        results.append(
            _timed(
                "load_prices_bulk",
                lambda: load_prices_bulk(index_list, db_path, max_days=400),
            )
        )

    if args.benchmark_registry:
        removable = registry.count_summary(db_path).get("removable", 0)
        results.append(
            {
                "label": "removable_dead_count",
                "count": removable,
                "note": "bulk_remove not executed automatically (destructive)",
            }
        )

    if not args.skip_backtest:
        results.append(
            _timed(
                "canslim_backtest",
                lambda: run_canslim_backtest(
                    db_path,
                    market_ticker=market,
                    tickers=tickers,
                    use_parallel=True,
                    progress_callback=None,
                ),
            )
        )

    if not args.skip_leaderboard:
        results.append(
            _timed(
                "leaderboard_build",
                lambda: build_leaderboard(
                    db_path,
                    market_ticker=market,
                    tickers=tickers,
                    segment=LeaderboardSegment.CANDIDATES,
                    limit=args.leaderboard_limit,
                    use_parallel=True,
                ),
            )
        )

    failed_sla = [r for r in results if r.get("within_sla") is False]
    payload = {
        "db_path": db_path,
        "tickers": tickers,
        "results": results,
        "sla_failures": failed_sla,
        "recent_db_ops": get_recent_db_ops(15),
    }
    print(json.dumps(payload, indent=2))
    return 1 if failed_sla else 0


if __name__ == "__main__":
    raise SystemExit(main())
