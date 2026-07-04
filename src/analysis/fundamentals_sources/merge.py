"""Merge fundamentals rows from multiple providers."""

from __future__ import annotations

from src.analysis.fundamentals_sources.types import FundamentalRow

# Lower = higher priority when dates/metrics collide.
_SOURCE_PRIORITY = {"sec": 0, "fmp": 1, "yahoo": 2}


def merge_fundamental_rows(
    *batches: tuple[list[FundamentalRow], str],
) -> list[FundamentalRow]:
    """
    Merge row lists; keep the highest-priority source per
    (ticker, report_date, metric, period_type).
    """
    best: dict[tuple[str, str, str, str], tuple[FundamentalRow, int]] = {}
    for rows, source_tag in batches:
        prio = _SOURCE_PRIORITY.get(source_tag, 99)
        for row in rows:
            key = (row[0], row[1], row[2], row[4])
            prev = best.get(key)
            if prev is None or prio < prev[1]:
                best[key] = (row, prio)
    return [v[0] for v in best.values()]
