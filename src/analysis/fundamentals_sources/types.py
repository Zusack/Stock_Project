"""Shared types for fundamentals ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FundamentalRow = tuple[str, str, str, float, str]
# (ticker, report_date, metric, value, period_type)

FundamentalsSourceMode = Literal["sec+yahoo", "sec", "yahoo", "fmp", "fmp+yahoo"]


@dataclass
class FundamentalsMeta:
    """Per-ticker fetch metadata returned to ingest logs."""

    source: str = ""
    row_count: int = 0
    warning: str = ""
    sec_used: bool = False
    yahoo_used: bool = False
    fmp_used: bool = False

    def summary(self) -> str:
        parts = [self.source or "none"]
        if self.row_count:
            parts.append(f"{self.row_count} rows")
        if self.warning:
            parts.append(self.warning)
        return ", ".join(parts)
