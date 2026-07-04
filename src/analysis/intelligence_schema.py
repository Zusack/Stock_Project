"""SQLite schema for market intelligence (guidance, alerts, quality, context)."""

from __future__ import annotations

import sqlite3

from src.analysis.db import db_connection, ingest_write_lock

INTELLIGENCE_TABLES = [
    "data_quality_events",
    "market_context_daily",
    "guidance_snapshots",
    "guidance_changes",
    "portfolio_suggestions",
    "alert_events",
    "signal_calibration",
    "leaderboard_runs",
    "leaderboard_snapshots",
    "ai_insights",
]


def ensure_intelligence_schema(db_path: str) -> None:
    """Create intelligence tables if missing."""
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS data_quality_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_at TEXT NOT NULL,
                    phase TEXT,
                    status TEXT,
                    error_category TEXT,
                    price_rows INTEGER DEFAULT 0,
                    fund_rows INTEGER DEFAULT 0,
                    notes TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_dqe_symbol ON data_quality_events(symbol, event_at);

                CREATE TABLE IF NOT EXISTS market_context_daily (
                    as_of_date TEXT PRIMARY KEY,
                    market_ticker TEXT,
                    regime TEXT,
                    trend_score REAL,
                    breadth_pct REAL,
                    vix_proxy_ticker TEXT,
                    vix_level REAL,
                    sector_leaders TEXT,
                    sector_laggards TEXT,
                    payload_json TEXT
                );

                CREATE TABLE IF NOT EXISTS guidance_snapshots (
                    scan_id TEXT NOT NULL,
                    scanned_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    composite_score REAL,
                    confidence REAL,
                    recommendation_band TEXT,
                    canslim_score INTEGER,
                    news_sentiment REAL,
                    regime_alignment REAL,
                    drivers_json TEXT,
                    risk_notes TEXT,
                    PRIMARY KEY (scan_id, ticker)
                );
                CREATE INDEX IF NOT EXISTS idx_guidance_scan ON guidance_snapshots(scan_id);

                CREATE TABLE IF NOT EXISTS guidance_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT NOT NULL,
                    prior_scan_id TEXT,
                    ticker TEXT NOT NULL,
                    prior_band TEXT,
                    new_band TEXT,
                    change_type TEXT,
                    detected_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_suggestions (
                    scan_id TEXT NOT NULL,
                    template_name TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    total_positions INTEGER,
                    cash_pct REAL,
                    payload_json TEXT,
                    PRIMARY KEY (scan_id, template_name)
                );

                CREATE TABLE IF NOT EXISTS alert_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    ticker TEXT,
                    alert_type TEXT NOT NULL,
                    severity TEXT,
                    title TEXT,
                    detail TEXT,
                    scan_id TEXT,
                    acknowledged INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_created ON alert_events(created_at);

                CREATE TABLE IF NOT EXISTS signal_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    lookback_days INTEGER,
                    slippage_bps REAL,
                    fee_per_trade REAL,
                    win_rate REAL,
                    avg_return REAL,
                    max_drawdown REAL,
                    sample_count INTEGER,
                    payload_json TEXT
                );

                -- Single active leaderboard snapshot (run_id='current'); future runs use new UUIDs.
                CREATE TABLE IF NOT EXISTS leaderboard_runs (
                    run_id TEXT PRIMARY KEY,
                    scored_at TEXT NOT NULL,
                    universe_key TEXT NOT NULL,
                    universe_label TEXT,
                    market_ticker TEXT NOT NULL,
                    full_universe INTEGER NOT NULL DEFAULT 0,
                    symbol_count INTEGER NOT NULL DEFAULT 0,
                    elapsed_sec REAL,
                    data_fingerprint TEXT NOT NULL,
                    data_as_of TEXT,
                    tickers_json TEXT
                );

                CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
                    run_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    composite_score REAL,
                    canslim_score INTEGER,
                    pattern_quality REAL,
                    rs_pct REAL,
                    volume_ratio REAL,
                    near_high_pct REAL,
                    pass_setup INTEGER,
                    pass_pattern INTEGER,
                    risk_flag TEXT,
                    latest_price REAL,
                    sector TEXT,
                    industry TEXT,
                    news_sentiment REAL,
                    notes TEXT,
                    extra_json TEXT,
                    PRIMARY KEY (run_id, ticker)
                );
                CREATE INDEX IF NOT EXISTS idx_lb_snap_ticker
                    ON leaderboard_snapshots(ticker);

                CREATE TABLE IF NOT EXISTS ai_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT,
                    kind TEXT NOT NULL,
                    payload_json TEXT,
                    source_context_hash TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_ai_insights_ticker
                    ON ai_insights(ticker, created_at);
                """
            )
            conn.commit()


def save_ai_insight(
    db_path: str,
    *,
    ticker: str,
    provider: str,
    model: str,
    kind: str,
    payload: dict,
    source_context_hash: str = "",
) -> None:
    """Persist an LLM-generated insight for audit/cache."""
    import json
    from datetime import datetime, timezone

    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                INSERT INTO ai_insights
                (ticker, created_at, provider, model, kind, payload_json, source_context_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(ticker).strip().upper(),
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    provider,
                    model,
                    kind,
                    json.dumps(payload, default=str),
                    source_context_hash,
                ),
            )
            conn.commit()


def load_latest_ai_insight(db_path: str, ticker: str, *, kind: str | None = None) -> dict | None:
    """Load the most recent AI insight for a ticker."""
    import json

    sym = str(ticker).strip().upper()
    sql = """
        SELECT ticker, created_at, provider, model, kind, payload_json, source_context_hash
        FROM ai_insights WHERE ticker = ?
    """
    params: list = [sym]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY created_at DESC LIMIT 1"
    with db_connection(db_path, readonly=True) as conn:
        row = conn.execute(sql, params).fetchone()
    if not row:
        return None
    return {
        "ticker": row[0],
        "created_at": row[1],
        "provider": row[2],
        "model": row[3],
        "kind": row[4],
        "payload": json.loads(row[5] or "{}"),
        "source_context_hash": row[6],
    }


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None
