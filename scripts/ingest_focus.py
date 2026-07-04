#!/usr/bin/env python3
"""Ingest focus watchlist only (fast daily update)."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.analysis.ingest import ingest_stock_data
from src.services.stock_config import stock_config


def main() -> int:
    cfg = stock_config()
    summary = ingest_stock_data(db_path=cfg.db_path, scope="focus", use_parallel=True)
    print(
        f"Focus ingest: {summary.get('success', 0)}/{summary.get('total', 0)} succeeded "
        f"(scope={summary.get('ingest_scope', 'focus')})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
