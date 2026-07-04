"""Chart pattern detection for growth-stock entries."""

from src.analysis.patterns.cup_with_handle import (
    CupWithHandleMatch,
    detect_cup_with_handle_series,
    scan_cup_with_handle,
)

__all__ = [
    "CupWithHandleMatch",
    "detect_cup_with_handle_series",
    "scan_cup_with_handle",
]
