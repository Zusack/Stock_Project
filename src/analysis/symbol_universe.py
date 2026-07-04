"""Resolve ticker lists for backtests and scans by scope."""

from __future__ import annotations

from src.analysis.db import list_tickers
from src.analysis.ticker_registry import (
    list_focus_symbols,
    list_symbols_for_ingest,
)


def resolve_universe(
    db_path: str,
    scope: str | None = None,
    *,
    explicit_tickers: list[str] | None = None,
) -> list[str]:
    """
    scope: focus | universe | with_history | all (default from config).
    explicit_tickers overrides scope when provided.
    """
    if explicit_tickers:
        return [str(t).strip().upper() for t in explicit_tickers if str(t).strip()]

    from src.services.stock_config import stock_config

    scope = (scope or stock_config().default_backtest_universe or "with_history").strip().lower()
    if scope == "focus":
        return [r.symbol for r in list_focus_symbols(db_path)]
    if scope == "universe":
        return list_symbols_for_ingest(db_path, "universe")
    if scope == "with_history":
        return list_tickers(db_path) or []
    return list_symbols_for_ingest(db_path, "all")
