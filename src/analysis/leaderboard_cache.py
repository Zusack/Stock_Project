"""Persist and restore leaderboard scoring runs (single active snapshot).

Designed for a future history of runs (``run_id`` UUID per save); today only one
row with ``run_id = 'current'`` is kept and replaced on each save.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.history_coverage import COVERAGE_STALE, assess_coverage
from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.analysis.market_calendar import last_completed_trading_day
from src.services.stock_config import stock_config

CURRENT_RUN_ID = "current"

# Freshness states returned by assess_leaderboard_cache_freshness
FRESH = "fresh"
STALE_DATA = "stale_data"
INGEST_NEEDED = "ingest_needed"
PARTIALLY_STALE = "partially_stale"
UNIVERSE_MISMATCH = "universe_mismatch"
MISSING = "missing"


@dataclass(frozen=True)
class LeaderboardRunMeta:
    scored_at: datetime
    universe_key: str
    universe_label: str
    market_ticker: str
    full_universe: bool
    symbol_count: int
    elapsed_sec: float
    data_fingerprint: str
    data_as_of: str | None
    tickers_json: str | None = None

    @property
    def run_id(self) -> str:
        return CURRENT_RUN_ID


@dataclass(frozen=True)
class LeaderboardCacheFreshness:
    state: str
    detail: str
    stale_symbol_count: int = 0
    ingest_overdue: bool = False
    calendar_behind: bool = False


@dataclass(frozen=True)
class LoadedLeaderboardCache:
    meta: LeaderboardRunMeta
    df: pd.DataFrame


def _parse_utc(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _parse_date(val: str | None) -> date | None:
    if not val:
        return None
    try:
        return date.fromisoformat(str(val)[:10])
    except ValueError:
        return None


def _max_price_date_for_universe(db_path: str, tickers: list[str] | None) -> date | None:
    """Latest settled bar date in stock_history for the scoring universe.

    Uses actual OHLCV rows, not watchlist ``price_max_date`` metadata, which can lag
    after ingest when coverage refresh was skipped or smart ingest did not re-touch a row.
    """
    with db_connection(db_path, readonly=True) as conn:
        if tickers:
            placeholders = ",".join("?" * len(tickers))
            row = conn.execute(
                f"""
                SELECT MAX(Date) FROM stock_history
                WHERE Ticker IN ({placeholders})
                """,
                [t.upper() for t in tickers],
            ).fetchone()
        else:
            row = conn.execute("SELECT MAX(Date) FROM stock_history").fetchone()
    if row and row[0]:
        return _parse_date(str(row[0]))
    return None


def compute_leaderboard_data_fingerprint(
    db_path: str,
    *,
    universe_key: str,
    tickers: list[str] | None,
    market_ticker: str,
) -> tuple[str, str | None]:
    """Fingerprint of ingest + price coverage used to detect DB changes since last score."""
    cfg = stock_config()
    as_of = _max_price_date_for_universe(db_path, tickers)
    parts = [
        universe_key,
        market_ticker,
        cfg.last_ingest_at or "",
        cfg.last_universe_ingest_at or "",
        as_of.isoformat() if as_of else "",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]
    return digest, (as_of.isoformat() if as_of else None)


def _count_stale_symbols(db_path: str, tickers: list[str]) -> int:
    if not tickers:
        return 0
    n = 0
    for sym in tickers:
        try:
            status, _, _ = assess_coverage(db_path, sym)
            if status == COVERAGE_STALE:
                n += 1
        except Exception:
            continue
    return n


def _universe_ingest_overdue(cfg) -> bool:
    last = cfg.last_universe_ingest_at
    if not last:
        return True
    try:
        dt = datetime.strptime(last[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    age = (datetime.now(timezone.utc) - dt).days
    return age >= int(cfg.universe_ingest_interval_days or 7)


def assess_leaderboard_cache_freshness(
    db_path: str,
    meta: LeaderboardRunMeta,
    *,
    current_universe_key: str,
    tickers: list[str] | None,
) -> LeaderboardCacheFreshness:
    """Compare a stored run to the current DB and calendar."""
    if meta.universe_key != current_universe_key:
        return LeaderboardCacheFreshness(
            state=UNIVERSE_MISMATCH,
            detail="Scoring scope changed (watchlist, filter, or full-universe flag).",
        )

    current_fp, current_as_of = compute_leaderboard_data_fingerprint(
        db_path,
        universe_key=current_universe_key,
        tickers=tickers,
        market_ticker=meta.market_ticker,
    )
    if meta.data_fingerprint != current_fp:
        return LeaderboardCacheFreshness(
            state=STALE_DATA,
            detail=(
                "Price or ingest data changed since these rankings were computed. "
                "Refresh rankings to rescore."
            ),
        )

    scored_symbols = []
    if meta.tickers_json:
        try:
            scored_symbols = json.loads(meta.tickers_json)
        except json.JSONDecodeError:
            scored_symbols = []

    stale_n = _count_stale_symbols(db_path, scored_symbols) if scored_symbols else 0
    ingest_overdue = bool(meta.full_universe and _universe_ingest_overdue(stock_config()))

    required_day = last_completed_trading_day()
    stored_as_of = _parse_date(meta.data_as_of)
    current_as_of_date = _parse_date(current_as_of)
    calendar_behind = bool(
        current_as_of_date
        and current_as_of_date < required_day
    )

    if ingest_overdue or calendar_behind or stale_n > 0:
        parts: list[str] = []
        if calendar_behind:
            parts.append(
                f"Market data ends {current_as_of_date}; latest trading day is {required_day}."
            )
        if ingest_overdue:
            parts.append("Scheduled universe ingest may be overdue.")
        if stale_n > 0:
            parts.append(f"{stale_n} symbol(s) have stale price coverage in the database.")
        parts.append("Run ingest on Research Universe before trusting these rankings.")
        state = PARTIALLY_STALE if stale_n > 0 else INGEST_NEEDED
        return LeaderboardCacheFreshness(
            state=state,
            detail=" ".join(parts),
            stale_symbol_count=stale_n,
            ingest_overdue=ingest_overdue,
            calendar_behind=calendar_behind,
        )

    return LeaderboardCacheFreshness(
        state=FRESH,
        detail="Rankings match the latest data in your database.",
    )


def save_leaderboard_snapshot(
    db_path: str,
    df: pd.DataFrame,
    *,
    meta: LeaderboardRunMeta,
) -> None:
    """Replace the single stored leaderboard run and per-ticker rows."""
    if df is None or df.empty:
        return
    ensure_intelligence_schema(db_path)
    scored_at = meta.scored_at.astimezone(timezone.utc).isoformat()
    rows: list[tuple[Any, ...]] = []
    for _, r in df.iterrows():
        tags = r.get("news_tags")
        extra = {
            "news_tags": tags,
            "score_version": r.get("score_version"),
            "composite_score_v1": r.get("composite_score_v1"),
            "momentum_score": r.get("momentum_score"),
            "quality_score": r.get("quality_score"),
            "value_score": r.get("value_score"),
            "risk_score": r.get("risk_score"),
            "sentiment_score": r.get("sentiment_score"),
            "buy_readiness": r.get("buy_readiness"),
            "sell_pressure": r.get("sell_pressure"),
            "rating_value": r.get("rating_value"),
            "rating_quality": r.get("rating_quality"),
            "rating_momentum": r.get("rating_momentum"),
            "rating_risk": r.get("rating_risk"),
            "data_confidence": r.get("data_confidence"),
            "score_components_json": r.get("score_components_json"),
            "canslm_lines": r.get("canslm_lines"),
        }
        extra = {k: v for k, v in extra.items() if v is not None}
        rows.append(
            (
                CURRENT_RUN_ID,
                str(r["ticker"]).upper(),
                float(r.get("composite_score", 0) or 0),
                int(r.get("canslim_score", 0) or 0),
                float(r.get("pattern_quality", 0) or 0),
                float(r.get("rs_pct", 0) or 0),
                float(r.get("volume_ratio", 0) or 0),
                float(r.get("near_high_pct", 0) or 0),
                1 if bool(r.get("pass_setup")) else 0,
                1 if bool(r.get("pass_pattern")) else 0,
                str(r.get("risk_flag") or ""),
                float(r.get("latest_price", 0) or 0),
                str(r.get("sector") or ""),
                str(r.get("industry") or ""),
                float(r.get("news_sentiment", 0.5) or 0.5),
                str(r.get("notes") or ""),
                json.dumps(extra) if extra else None,
            )
        )

    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute("DELETE FROM leaderboard_snapshots WHERE run_id = ?", (CURRENT_RUN_ID,))
            conn.execute("DELETE FROM leaderboard_runs WHERE run_id = ?", (CURRENT_RUN_ID,))
            conn.execute(
                """
                INSERT INTO leaderboard_runs (
                    run_id, scored_at, universe_key, universe_label, market_ticker,
                    full_universe, symbol_count, elapsed_sec, data_fingerprint,
                    data_as_of, tickers_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    CURRENT_RUN_ID,
                    scored_at,
                    meta.universe_key,
                    meta.universe_label,
                    meta.market_ticker,
                    1 if meta.full_universe else 0,
                    meta.symbol_count,
                    meta.elapsed_sec,
                    meta.data_fingerprint,
                    meta.data_as_of,
                    meta.tickers_json,
                ),
            )
            conn.executemany(
                """
                INSERT INTO leaderboard_snapshots (
                    run_id, ticker, composite_score, canslim_score, pattern_quality,
                    rs_pct, volume_ratio, near_high_pct, pass_setup, pass_pattern,
                    risk_flag, latest_price, sector, industry, news_sentiment, notes,
                    extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()


def load_latest_leaderboard(db_path: str) -> LoadedLeaderboardCache | None:
    """Load the stored leaderboard if present."""
    ensure_intelligence_schema(db_path)
    with db_connection(db_path, readonly=True) as conn:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, "leaderboard_runs"):
            return None
        run = conn.execute(
            "SELECT * FROM leaderboard_runs WHERE run_id = ?",
            (CURRENT_RUN_ID,),
        ).fetchone()
        if not run:
            return None
        run_d = dict(run)
        snap_rows = conn.execute(
            """
            SELECT ticker, composite_score, canslim_score, pattern_quality,
                   rs_pct, volume_ratio, near_high_pct, pass_setup, pass_pattern,
                   risk_flag, latest_price, sector, industry, news_sentiment, notes,
                   extra_json
            FROM leaderboard_snapshots WHERE run_id = ?
            """,
            (CURRENT_RUN_ID,),
        ).fetchall()
        if not snap_rows:
            return None

    records: list[dict[str, Any]] = []
    for row in snap_rows:
        d = dict(row)
        extra = {}
        if d.get("extra_json"):
            try:
                extra = json.loads(d["extra_json"])
            except json.JSONDecodeError:
                extra = {}
        records.append(
            {
                "ticker": d["ticker"],
                "composite_score": d["composite_score"],
                "canslim_score": d["canslim_score"],
                "pattern_quality": d["pattern_quality"],
                "rs_pct": d["rs_pct"],
                "volume_ratio": d["volume_ratio"],
                "near_high_pct": d["near_high_pct"],
                "pass_setup": bool(d["pass_setup"]),
                "pass_pattern": bool(d["pass_pattern"]),
                "risk_flag": d.get("risk_flag") or "",
                "latest_price": d.get("latest_price") or 0.0,
                "sector": d.get("sector") or "",
                "industry": d.get("industry") or "",
                "notes": d.get("notes") or "",
                "news_sentiment": d.get("news_sentiment") or 0.5,
                "news_tags": extra.get("news_tags"),
                "score_version": extra.get("score_version", "v2"),
                "composite_score_v1": extra.get("composite_score_v1", d["composite_score"]),
                "momentum_score": extra.get("momentum_score", 0.0),
                "quality_score": extra.get("quality_score", 0.0),
                "value_score": extra.get("value_score", 0.0),
                "risk_score": extra.get("risk_score", 0.0),
                "sentiment_score": extra.get("sentiment_score", 0.0),
                "buy_readiness": extra.get("buy_readiness", 0.0),
                "sell_pressure": extra.get("sell_pressure", 0.0),
                "rating_value": extra.get("rating_value", "C"),
                "rating_quality": extra.get("rating_quality", "C"),
                "rating_momentum": extra.get("rating_momentum", "C"),
                "rating_risk": extra.get("rating_risk", "C"),
                "data_confidence": extra.get("data_confidence", 1.0),
                "score_components_json": extra.get("score_components_json", ""),
                "canslm_lines": extra.get("canslm_lines") or [],
            }
        )

    scored_at = _parse_utc(run_d.get("scored_at"))
    if scored_at is None:
        return None

    meta = LeaderboardRunMeta(
        scored_at=scored_at,
        universe_key=str(run_d["universe_key"]),
        universe_label=str(run_d.get("universe_label") or ""),
        market_ticker=str(run_d.get("market_ticker") or ""),
        full_universe=bool(run_d.get("full_universe")),
        symbol_count=int(run_d.get("symbol_count") or len(records)),
        elapsed_sec=float(run_d.get("elapsed_sec") or 0.0),
        data_fingerprint=str(run_d.get("data_fingerprint") or ""),
        data_as_of=run_d.get("data_as_of"),
        tickers_json=run_d.get("tickers_json"),
    )
    df = pd.DataFrame(records)
    return LoadedLeaderboardCache(meta=meta, df=df)


def can_reuse_cached_leaderboard(
    db_path: str,
    *,
    universe_key: str,
    tickers: list[str] | None,
) -> LoadedLeaderboardCache | None:
    """Return cached scores when DB fingerprint matches (skip expensive rescore)."""
    loaded = load_latest_leaderboard(db_path)
    if loaded is None:
        return None
    freshness = assess_leaderboard_cache_freshness(
        db_path,
        loaded.meta,
        current_universe_key=universe_key,
        tickers=tickers,
    )
    if freshness.state == FRESH:
        return loaded
    return None


def leaderboard_row_from_record(rec: dict[str, Any]):
    """Build a LeaderboardRow from a cache snapshot record dict."""
    from src.analysis.leaderboard import LeaderboardRow

    return LeaderboardRow(
        ticker=str(rec.get("ticker", "")),
        composite_score=float(rec.get("composite_score", 0) or 0),
        canslim_score=int(rec.get("canslim_score", 0) or 0),
        pattern_quality=float(rec.get("pattern_quality", 0) or 0),
        rs_pct=float(rec.get("rs_pct", 0) or 0),
        volume_ratio=float(rec.get("volume_ratio", 0) or 0),
        near_high_pct=float(rec.get("near_high_pct", 0) or 0),
        pass_setup=bool(rec.get("pass_setup")),
        pass_pattern=bool(rec.get("pass_pattern")),
        risk_flag=str(rec.get("risk_flag") or ""),
        latest_price=float(rec.get("latest_price", 0) or 0),
        sector=str(rec.get("sector") or ""),
        industry=str(rec.get("industry") or ""),
        notes=str(rec.get("notes") or ""),
        news_sentiment=float(rec.get("news_sentiment", 0.5) or 0.5),
        news_tags=list(rec.get("news_tags") or []),
        score_version=str(rec.get("score_version") or "v2"),
        composite_score_v1=float(rec.get("composite_score_v1", 0) or 0),
        momentum_score=float(rec.get("momentum_score", 0) or 0),
        quality_score=float(rec.get("quality_score", 0) or 0),
        value_score=float(rec.get("value_score", 0) or 0),
        risk_score=float(rec.get("risk_score", 0) or 0),
        sentiment_score=float(rec.get("sentiment_score", 0) or 0),
        buy_readiness=float(rec.get("buy_readiness", 0) or 0),
        sell_pressure=float(rec.get("sell_pressure", 0) or 0),
        rating_value=str(rec.get("rating_value") or "C"),
        rating_quality=str(rec.get("rating_quality") or "C"),
        rating_momentum=str(rec.get("rating_momentum") or "C"),
        rating_risk=str(rec.get("rating_risk") or "C"),
        data_confidence=float(rec.get("data_confidence", 1) or 1),
        score_components_json=str(rec.get("score_components_json") or ""),
        canslm_lines=list(rec.get("canslm_lines") or []),
    )


def lookup_leaderboard_row(db_path: str, ticker: str):
    """Return a cached LeaderboardRow for one symbol when a snapshot exists."""
    loaded = load_latest_leaderboard(db_path)
    if loaded is None or loaded.df.empty:
        return None
    sym = str(ticker).strip().upper()
    rows = loaded.df[loaded.df["ticker"].astype(str).str.upper() == sym]
    if rows.empty:
        return None
    return leaderboard_row_from_record(rows.iloc[0].to_dict())


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None
