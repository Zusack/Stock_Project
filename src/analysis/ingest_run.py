"""Persisted ingest run state for pause/resume across sessions."""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.analysis.db import db_connection
from src.analysis.history_coverage import TickerIngestPlan, init_coverage_schema

RUN_RUNNING = "running"
RUN_PAUSED = "paused"
RUN_COMPLETED = "completed"
RUN_CANCELLED = "cancelled"

ITEM_PENDING = "pending"
ITEM_IN_PROGRESS = "in_progress"
ITEM_PRICE_DONE = "price_done"
ITEM_FUND_DONE = "fund_done"
ITEM_DONE = "done"
ITEM_SKIPPED = "skipped"
ITEM_FAILED = "failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass
class IngestRun:
    run_id: str
    status: str
    mode: str
    started_at: str
    paused_at: str | None
    completed_at: str | None
    fundamentals_source: str
    total_count: int
    done_count: int
    skipped_count: int
    failed_count: int
    last_ticker: str | None
    error_message: str | None
    ingest_scope: str = "all"


def init_ingest_run_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            mode TEXT NOT NULL,
            started_at TEXT NOT NULL,
            paused_at TEXT,
            completed_at TEXT,
            fundamentals_source TEXT,
            total_count INTEGER NOT NULL DEFAULT 0,
            done_count INTEGER NOT NULL DEFAULT 0,
            skipped_count INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0,
            last_ticker TEXT,
            error_message TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_run_items (
            run_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            price_action TEXT,
            fund_action TEXT,
            planned_start_date TEXT,
            planned_end_date TEXT,
            rows_price_added INTEGER DEFAULT 0,
            rows_fund_added INTEGER DEFAULT 0,
            message TEXT,
            finished_at TEXT,
            PRIMARY KEY (run_id, symbol),
            FOREIGN KEY (run_id) REFERENCES ingest_runs(run_id)
        )
    """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingest_run_items_state ON ingest_run_items(run_id, state)"
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ingest_runs)")}
    if "ingest_scope" not in cols:
        try:
            conn.execute(
                "ALTER TABLE ingest_runs ADD COLUMN ingest_scope TEXT DEFAULT 'all'"
            )
        except sqlite3.OperationalError:
            pass
    item_cols = {row[1] for row in conn.execute("PRAGMA table_info(ingest_run_items)")}
    if "focus_priority" not in item_cols:
        try:
            conn.execute(
                "ALTER TABLE ingest_run_items ADD COLUMN focus_priority INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass


def _row_to_run(row: sqlite3.Row | tuple) -> IngestRun:
    if hasattr(row, "keys"):
        d = dict(row)
        return IngestRun(
            run_id=d["run_id"],
            status=d["status"],
            mode=d["mode"],
            started_at=d["started_at"],
            paused_at=d.get("paused_at"),
            completed_at=d.get("completed_at"),
            fundamentals_source=d.get("fundamentals_source") or "",
            total_count=int(d.get("total_count") or 0),
            done_count=int(d.get("done_count") or 0),
            skipped_count=int(d.get("skipped_count") or 0),
            failed_count=int(d.get("failed_count") or 0),
            last_ticker=d.get("last_ticker"),
            error_message=d.get("error_message"),
            ingest_scope=d.get("ingest_scope") or "all",
        )
    return IngestRun(
        run_id=row[0],
        status=row[1],
        mode=row[2],
        started_at=row[3],
        paused_at=row[4],
        completed_at=row[5],
        fundamentals_source=row[6] or "",
        total_count=int(row[7] or 0),
        done_count=int(row[8] or 0),
        skipped_count=int(row[9] or 0),
        failed_count=int(row[10] or 0),
        last_ticker=row[11],
        error_message=row[12],
    )


def get_run(db_path: str | os.PathLike, run_id: str) -> IngestRun | None:
    with db_connection(db_path, readonly=True) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM ingest_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return _row_to_run(row) if row else None


def get_resumable_run(db_path: str | os.PathLike) -> IngestRun | None:
    with db_connection(db_path, readonly=True) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT * FROM ingest_runs
            WHERE status IN (?, ?)
            ORDER BY started_at DESC LIMIT 1
        """,
            (RUN_PAUSED, RUN_RUNNING),
        ).fetchone()
    return _row_to_run(row) if row else None


def create_run(
    db_path: str | os.PathLike,
    symbols: list[str],
    mode: str,
    fundamentals_source: str,
    plans: dict[str, TickerIngestPlan] | None = None,
    *,
    ingest_scope: str = "all",
    focus_symbols: set[str] | None = None,
) -> str:
    from src.analysis.ticker_registry import is_focus_symbol

    init_coverage_schema(db_path)
    run_id = f"ingest_{uuid.uuid4().hex[:12]}"
    syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
    focus_set = focus_symbols or set()
    if not focus_set and ingest_scope in ("universe", "all"):
        focus_set = {s for s in syms if is_focus_symbol(db_path, s)}
    now = _utc_now()
    with db_connection(db_path, readonly=False) as conn:
        init_ingest_run_schema(conn)
        conn.execute(
            """
            INSERT INTO ingest_runs (
                run_id, status, mode, started_at, fundamentals_source, total_count, ingest_scope
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (run_id, RUN_RUNNING, mode, now, fundamentals_source, len(syms), ingest_scope),
        )
        for sym in syms:
            plan = (plans or {}).get(sym)
            price_action = plan.price_action if plan else None
            fund_action = plan.fund_action if plan else None
            p_start = plan.price_ranges[0].start_str() if plan and plan.price_ranges else None
            p_end = plan.price_ranges[-1].end_str() if plan and plan.price_ranges else None
            prio = 1 if sym in focus_set else 0
            conn.execute(
                """
                INSERT INTO ingest_run_items (
                    run_id, symbol, state, price_action, fund_action,
                    planned_start_date, planned_end_date, focus_priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    sym,
                    ITEM_PENDING,
                    price_action,
                    fund_action,
                    p_start,
                    p_end,
                    prio,
                ),
            )
        conn.commit()
    return run_id


def resume_run(db_path: str | os.PathLike, run_id: str) -> None:
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE ingest_runs SET status = ? WHERE run_id = ?
        """,
            (RUN_RUNNING, run_id),
        )
        conn.execute(
            """
            UPDATE ingest_run_items SET state = ?
            WHERE run_id = ? AND state = ?
        """,
            (ITEM_PENDING, run_id, ITEM_IN_PROGRESS),
        )
        conn.commit()


def pause_run(db_path: str | os.PathLike, run_id: str, message: str = "") -> None:
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE ingest_runs
            SET status = ?, paused_at = ?, error_message = ?
            WHERE run_id = ?
        """,
            (RUN_PAUSED, _utc_now(), message[:500] if message else None, run_id),
        )
        conn.execute(
            """
            UPDATE ingest_run_items SET state = ?
            WHERE run_id = ? AND state = ?
        """,
            (ITEM_PENDING, run_id, ITEM_IN_PROGRESS),
        )
        conn.commit()


def cancel_run(db_path: str | os.PathLike, run_id: str) -> None:
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE ingest_runs SET status = ?, completed_at = ? WHERE run_id = ?
        """,
            (RUN_CANCELLED, _utc_now(), run_id),
        )
        conn.commit()


def complete_run(db_path: str | os.PathLike, run_id: str) -> None:
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE ingest_runs SET status = ?, completed_at = ? WHERE run_id = ?
        """,
            (RUN_COMPLETED, _utc_now(), run_id),
        )
        conn.commit()


def claim_next_pending(db_path: str | os.PathLike, run_id: str) -> str | None:
    return claim_next_for_phase(db_path, run_id, "full")


def claim_next_for_phase(
    db_path: str | os.PathLike,
    run_id: str,
    phase: str,
) -> str | None:
    """Claim next symbol for sequential (full) or phased parallel ingest."""
    if phase == "full":
        claim_states = (ITEM_PENDING,)
    elif phase == "price":
        claim_states = (ITEM_PENDING,)
    elif phase == "fundamentals":
        claim_states = (ITEM_PRICE_DONE,)
    elif phase == "enrich":
        claim_states = (ITEM_FUND_DONE,)
    else:
        claim_states = (ITEM_PENDING,)

    placeholders = ",".join("?" * len(claim_states))
    with db_connection(db_path, readonly=False) as conn:
        row = conn.execute(
            f"""
            UPDATE ingest_run_items
            SET state = ?
            WHERE rowid = (
                SELECT rowid FROM ingest_run_items
                WHERE run_id = ? AND state IN ({placeholders})
                ORDER BY COALESCE(focus_priority, 0) DESC, symbol
                LIMIT 1
            )
            RETURNING symbol
            """,
            (ITEM_IN_PROGRESS, run_id, *claim_states),
        ).fetchone()
        if not row:
            return None
        sym = row[0]
        conn.execute(
            "UPDATE ingest_runs SET last_ticker = ? WHERE run_id = ?",
            (sym, run_id),
        )
        conn.commit()
        return sym


def set_item_phase_state(
    db_path: str | os.PathLike,
    run_id: str,
    symbol: str,
    state: str,
) -> None:
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE ingest_run_items SET state = ? WHERE run_id = ? AND symbol = ?
        """,
            (state, run_id, symbol),
        )
        conn.commit()


def update_item_plan(
    db_path: str | os.PathLike,
    run_id: str,
    symbol: str,
    plan: TickerIngestPlan,
) -> None:
    p_start = plan.price_ranges[0].start_str() if plan.price_ranges else None
    p_end = plan.price_ranges[-1].end_str() if plan.price_ranges else None
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE ingest_run_items SET
                price_action = ?, fund_action = ?,
                planned_start_date = ?, planned_end_date = ?
            WHERE run_id = ? AND symbol = ?
        """,
            (
                plan.price_action,
                plan.fund_action,
                p_start,
                p_end,
                run_id,
                symbol,
            ),
        )
        conn.commit()


def finish_item(
    db_path: str | os.PathLike,
    run_id: str,
    symbol: str,
    result: dict[str, Any],
    *,
    skipped: bool = False,
) -> None:
    status = result.get("Status", "")
    state = ITEM_SKIPPED if skipped else (ITEM_DONE if status == "Success" else ITEM_FAILED)
    msg = result.get("Fund_Note") or status
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE ingest_run_items SET
                state = ?, rows_price_added = ?, rows_fund_added = ?,
                message = ?, finished_at = ?
            WHERE run_id = ? AND symbol = ?
        """,
            (
                state,
                int(result.get("Price_Rows") or 0),
                int(result.get("Fund_Rows") or 0),
                str(msg)[:300],
                _utc_now(),
                run_id,
                symbol,
            ),
        )
        col = "skipped_count" if skipped else ("failed_count" if state == ITEM_FAILED else "done_count")
        conn.execute(
            f"UPDATE ingest_runs SET {col} = {col} + 1 WHERE run_id = ?",
            (run_id,),
        )
        conn.commit()


def count_items_by_state(db_path: str | os.PathLike, run_id: str) -> dict[str, int]:
    """Per-state counts for ingest run diagnostics."""
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(
            """
            SELECT state, COUNT(*) FROM ingest_run_items
            WHERE run_id = ? GROUP BY state
        """,
            (run_id,),
        ).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}


def release_in_progress(db_path: str | os.PathLike, run_id: str, symbol: str, *, to_state: str = ITEM_PENDING) -> None:
    """Reset a stuck in-progress item (e.g. after worker error)."""
    with db_connection(db_path, readonly=False) as conn:
        conn.execute(
            """
            UPDATE ingest_run_items SET state = ?
            WHERE run_id = ? AND symbol = ? AND state = ?
        """,
            (to_state, run_id, symbol, ITEM_IN_PROGRESS),
        )
        conn.commit()


def count_pending(db_path: str | os.PathLike, run_id: str) -> int:
    """Symbols not yet finished (queued, in flight, or mid-phase)."""
    with db_connection(db_path, readonly=True) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM ingest_run_items
            WHERE run_id = ? AND state IN (?, ?, ?, ?)
        """,
            (run_id, ITEM_PENDING, ITEM_IN_PROGRESS, ITEM_PRICE_DONE, ITEM_FUND_DONE),
        ).fetchone()
    return int(row[0]) if row else 0


def run_progress_summary(db_path: str | os.PathLike, run_id: str) -> dict[str, Any]:
    run = get_run(db_path, run_id)
    if not run:
        return {}
    remaining = count_pending(db_path, run_id)
    finished = run.done_count + run.skipped_count + run.failed_count
    return {
        "run_id": run.run_id,
        "status": run.status,
        "mode": run.mode,
        "total": run.total_count,
        "done": run.done_count,
        "skipped": run.skipped_count,
        "failed": run.failed_count,
        "finished": finished,
        "remaining": remaining,
        "pending": remaining,
        "last_ticker": run.last_ticker,
        "paused_at": run.paused_at,
    }


def list_pending_symbols(db_path: str | os.PathLike, run_id: str) -> list[str]:
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(
            """
            SELECT symbol FROM ingest_run_items
            WHERE run_id = ? AND state IN (?, ?)
            ORDER BY symbol
        """,
            (run_id, ITEM_PENDING, ITEM_FAILED),
        ).fetchall()
    return [r[0] for r in rows]
