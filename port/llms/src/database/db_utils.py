# src/database/db_utils.py
"""
Shared helpers for database row handling.
Use these when converting sqlite3.Row (or sequences of rows) to dicts for consistent behavior.
"""
from typing import Any, List, Optional, Sequence


def row_to_dict(row: Any) -> Optional[dict]:
    """Convert a single sqlite3.Row (or row-like object) to a dict. Returns None if row is None."""
    if row is None:
        return None
    return dict(row)


def rows_to_dicts(rows: Optional[Sequence[Any]]) -> List[dict]:
    """Convert a sequence of sqlite3.Row (or row-like objects) to a list of dicts. Returns [] if rows is None or empty."""
    if not rows:
        return []
    return [dict(r) for r in rows]
