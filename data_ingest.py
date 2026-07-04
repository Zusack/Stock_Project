"""CLI wrapper — research universe ingest (weekly bulk update)."""

import argparse
import multiprocessing
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.analysis import ticker_registry as registry
from src.analysis.ingest import ingest_stock_data
from src.services.stock_config import stock_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest market data into SQLite")
    parser.add_argument(
        "--scope",
        choices=("focus", "universe", "all"),
        default="universe",
        help="focus = watchlist (<=20), universe = full research database (default)",
    )
    args = parser.parse_args()

    multiprocessing.freeze_support()
    cfg = stock_config()
    db_path = cfg.db_path

    if registry.watchlist_is_empty(db_path):
        csv_file = cfg.ticker_csv_path
        if os.path.isfile(csv_file):
            print(f"Registry empty — importing research universe from {csv_file}...")
            result = registry.migrate_csv_to_db(db_path, csv_file)
            print(
                f"Imported {result.get('newly_added', 0)} symbols "
                f"({result.get('total_in_watchlist', 0)} in universe)."
            )

    summary = ingest_stock_data(db_path=db_path, scope=args.scope, use_parallel=True)
    print(
        f"Done ({summary.get('ingest_scope', args.scope)}): "
        f"{summary.get('success', 0)}/{summary.get('total', 0)} succeeded. "
        f"DB: {summary.get('db_path', '')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
