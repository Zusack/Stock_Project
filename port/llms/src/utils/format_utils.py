# src/utils/format_utils.py
"""
Shared formatting and sort-key helpers for UI and reports.
"""
import re
from typing import Any, Optional


def format_timestamp(ts_val: Any) -> str:
    """Format a timestamp for display. Handles None, datetime, or string."""
    if not ts_val:
        return "N/A"
    if hasattr(ts_val, "strftime"):
        return ts_val.strftime("%Y-%m-%d %H:%M")
    return str(ts_val)


def model_sort_key(m: dict) -> tuple:
    """
    Return a sort key tuple for a model row (architecture, params, name).
    Used by inspector views and model lists for consistent ordering.
    """
    arch = (m.get("architecture") or "").lower()
    params_str = m.get("params_string") or ""

    def _parse(s: str) -> float:
        try:
            match = re.findall(r"(\d+\.?\d*)", s)
            return float(match[0]) if match else 0.0
        except (ValueError, IndexError, TypeError):
            return 0.0

    params = _parse(params_str)
    name = (m.get("display_name") or m.get("llm_identifier") or "").lower()
    return (arch, params, name)
