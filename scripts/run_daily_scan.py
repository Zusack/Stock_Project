#!/usr/bin/env python3
"""Run end-of-day guidance scan, portfolio templates, and alerts."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.services.daily_monitor import format_digest_text, run_daily_scan
from src.services.stock_config import stock_config


def main() -> int:
    cfg = stock_config()
    digest = run_daily_scan(
        cfg.db_path,
        ticker_limit=cfg.daily_scan_ticker_limit,
        progress=lambda msg: print(msg),
    )
    print()
    print(format_digest_text(digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
