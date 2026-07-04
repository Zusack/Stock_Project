"""Staged and parallel ingest worker pools (price → fundamentals → enrich)."""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.analysis import ticker_registry as registry
from src.analysis.history_coverage import TickerIngestPlan, build_ticker_plan
from src.analysis.ingest import (
    _format_ticker_log_line,
    _ingest_item_skipped,
    _status_counts_line,
    process_ticker_enrich_phase,
    process_ticker_fundamentals_phase,
    process_ticker_from_plan,
    process_ticker_price_phase,
)
from src.analysis.ingest_run import (
    ITEM_FUND_DONE,
    ITEM_IN_PROGRESS,
    ITEM_PRICE_DONE,
    claim_next_for_phase,
    count_items_by_state,
    finish_item,
    get_run,
    release_in_progress,
    run_progress_summary,
    set_item_phase_state,
    update_item_plan,
)
from src.services.stock_config import stock_config
from src.utils.logger_utils import append_ingest_detail_log

ProgressReport = Callable[[float, str, str | None], None]

_HEARTBEAT_INTERVAL_SEC = 30.0


def _log_worker(message: str, level: str = "INFO", **kwargs: Any) -> None:
    try:
        from src.analysis.ingest import _log_ingest

        _log_ingest(message, level=level, **kwargs)
    except Exception:
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        detail = message if not parts else f"{message} | " + ", ".join(parts)
        append_ingest_detail_log(f"[{level}] {detail}")


@dataclass
class IngestWorkerContext:
    db_path: str
    run_id: str
    ingest_mode: str
    force_full: bool
    total_tasks: int
    cancel_event: threading.Event | None
    report: ProgressReport
    results: list[dict[str, Any]] = field(default_factory=list)
    results_lock: threading.Lock = field(default_factory=threading.Lock)
    processed: int = 0
    processed_lock: threading.Lock = field(default_factory=threading.Lock)
    active_ticker: str | None = None
    active_phase: str | None = None
    active_lock: threading.Lock = field(default_factory=threading.Lock)
    active_since: float = 0.0

    def cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def set_active(self, ticker: str | None, phase: str | None) -> None:
        with self.active_lock:
            self.active_ticker = ticker
            self.active_phase = phase
            self.active_since = time.monotonic() if ticker else 0.0

    def append_result(self, res: dict[str, Any]) -> None:
        with self.results_lock:
            self.results.append(res)

    def bump_processed(self) -> int:
        with self.processed_lock:
            self.processed += 1
            return self.processed

    def emit_progress(self, log_line: str | None = None, *, status_msg: str | None = None) -> None:
        prog = run_progress_summary(self.db_path, self.run_id)
        finished = int(prog.get("finished") or 0)
        pct = finished / max(self.total_tasks, 1)
        msg = status_msg or _status_counts_line(prog)
        self.report(pct, msg, log_line)


def _plan_for_symbol(ctx: IngestWorkerContext, sym: str) -> TickerIngestPlan:
    run = get_run(ctx.db_path, ctx.run_id)
    paused_at = run.paused_at if run else None
    mode = ctx.ingest_mode if ctx.ingest_mode != "resume" else "smart"
    plan = build_ticker_plan(
        ctx.db_path,
        sym,
        mode=mode,
        paused_at=paused_at,
        force_full=ctx.force_full,
    )
    plan.db_path = ctx.db_path
    update_item_plan(ctx.db_path, ctx.run_id, sym, plan)
    return plan


def _base_partial(sym: str) -> dict[str, Any]:
    return {
        "Ticker": sym,
        "Status": "Success",
        "Price_Rows": 0,
        "Fund_Rows": 0,
        "Fund_Source": "",
        "Fund_Note": "",
        "Insider": 0,
        "News": 0,
    }


def _finish_ticker(
    ctx: IngestWorkerContext,
    sym: str,
    res: dict[str, Any],
    *,
    skipped: bool = False,
) -> None:
    finish_item(ctx.db_path, ctx.run_id, sym, res, skipped=skipped)
    registry.record_ingest_result(ctx.db_path, sym, res.get("Status", ""))
    ctx.append_result(res)
    ctx.bump_processed()
    ctx.set_active(None, None)
    log_line = _format_ticker_log_line(res)
    ctx.emit_progress(log_line=log_line)


def _advance_after_price(ctx: IngestWorkerContext, sym: str, plan: TickerIngestPlan, res: dict[str, Any]) -> None:
    status = res.get("Status", "")
    if plan.all_skip or _ingest_item_skipped(status):
        _finish_ticker(ctx, sym, res, skipped=True)
        return
    if status != "Success":
        _finish_ticker(ctx, sym, res, skipped=_ingest_item_skipped(status))
        return
    if plan.fund_action == "skip" and plan.enrich_action == "skip":
        _finish_ticker(ctx, sym, res)
        return
    set_item_phase_state(ctx.db_path, ctx.run_id, sym, ITEM_PRICE_DONE)


def _advance_after_fundamentals(ctx: IngestWorkerContext, sym: str, plan: TickerIngestPlan, res: dict[str, Any]) -> None:
    if plan.enrich_action == "skip":
        _finish_ticker(ctx, sym, res)
        return
    set_item_phase_state(ctx.db_path, ctx.run_id, sym, ITEM_FUND_DONE)


def _handle_worker_error(ctx: IngestWorkerContext, sym: str, phase: str, exc: BaseException) -> None:
    release_in_progress(ctx.db_path, ctx.run_id, sym)
    ctx.set_active(None, None)
    tb = traceback.format_exc(limit=3)
    _log_worker(
        f"Worker error ({phase})",
        level="ERROR",
        ticker=sym,
        error=str(exc)[:300],
        traceback=tb[:500],
    )
    res = _base_partial(sym)
    res["Status"] = f"Error ({phase}): {exc}"[:200]
    _finish_ticker(ctx, sym, res)


def _process_price_claim(ctx: IngestWorkerContext, sym: str) -> None:
    ctx.set_active(sym, "price")
    plan = _plan_for_symbol(ctx, sym)
    if plan.all_skip:
        res = _base_partial(sym)
        res["Status"] = f"Skipped ({plan.skip_reason})" if plan.skip_reason else "Skipped (up to date)"
        res["Skipped"] = True
        _finish_ticker(ctx, sym, res, skipped=True)
        return

    res = process_ticker_price_phase(plan)
    if not res:
        res = _base_partial(sym)
        res["Status"] = "General Error"
    _advance_after_price(ctx, sym, plan, res)


def _process_fundamentals_claim(ctx: IngestWorkerContext, sym: str) -> None:
    ctx.set_active(sym, "fundamentals")
    plan = _plan_for_symbol(ctx, sym)
    partial = _base_partial(sym)
    if plan.fund_action != "refresh":
        _advance_after_fundamentals(ctx, sym, plan, partial)
        return
    res = process_ticker_fundamentals_phase(plan, partial)
    _advance_after_fundamentals(ctx, sym, plan, res)


def _process_enrich_claim(ctx: IngestWorkerContext, sym: str) -> None:
    ctx.set_active(sym, "enrich")
    plan = _plan_for_symbol(ctx, sym)
    partial = _base_partial(sym)
    res = process_ticker_enrich_phase(plan, partial)
    _finish_ticker(ctx, sym, res)


def _run_sequential_per_ticker(ctx: IngestWorkerContext) -> None:
    """One ticker at a time: price → fundamentals → enrich (same as pre-phased ingest)."""
    _log_worker("Sequential ingest loop started", run_id=ctx.run_id)
    while not ctx.cancelled():
        sym = claim_next_for_phase(ctx.db_path, ctx.run_id, "price")
        if not sym:
            break
        ctx.emit_progress(
            log_line=None,
            status_msg=f"Processing {sym}…",
        )
        _log_worker("Claimed ticker", ticker=sym, mode="sequential")
        try:
            plan = _plan_for_symbol(ctx, sym)
            ctx.set_active(sym, "full")
            if plan.all_skip:
                res = _base_partial(sym)
                res["Status"] = (
                    f"Skipped ({plan.skip_reason})" if plan.skip_reason else "Skipped (up to date)"
                )
                _finish_ticker(ctx, sym, res, skipped=True)
                continue
            res = process_ticker_from_plan(plan)
            if res:
                skipped = _ingest_item_skipped(res.get("Status", ""))
                _finish_ticker(ctx, sym, res, skipped=skipped)
            else:
                release_in_progress(ctx.db_path, ctx.run_id, sym)
        except Exception as exc:
            _handle_worker_error(ctx, sym, "sequential", exc)


def _price_worker(ctx: IngestWorkerContext) -> None:
    while not ctx.cancelled():
        sym = claim_next_for_phase(ctx.db_path, ctx.run_id, "price")
        if not sym:
            break
        _log_worker("Claimed ticker", ticker=sym, phase="price")
        try:
            _process_price_claim(ctx, sym)
        except Exception as exc:
            _handle_worker_error(ctx, sym, "price", exc)


def _fundamentals_worker(ctx: IngestWorkerContext) -> None:
    while not ctx.cancelled():
        sym = claim_next_for_phase(ctx.db_path, ctx.run_id, "fundamentals")
        if not sym:
            break
        _log_worker("Claimed ticker", ticker=sym, phase="fundamentals")
        try:
            _process_fundamentals_claim(ctx, sym)
        except Exception as exc:
            _handle_worker_error(ctx, sym, "fundamentals", exc)


def _enrich_worker(ctx: IngestWorkerContext) -> None:
    while not ctx.cancelled():
        sym = claim_next_for_phase(ctx.db_path, ctx.run_id, "enrich")
        if not sym:
            break
        _log_worker("Claimed ticker", ticker=sym, phase="enrich")
        try:
            _process_enrich_claim(ctx, sym)
        except Exception as exc:
            _handle_worker_error(ctx, sym, "enrich", exc)


def _heartbeat_loop(ctx: IngestWorkerContext, stop: threading.Event) -> None:
    while not stop.wait(_HEARTBEAT_INTERVAL_SEC):
        if ctx.cancelled():
            continue
        prog = run_progress_summary(ctx.db_path, ctx.run_id)
        states = count_items_by_state(ctx.db_path, ctx.run_id)
        in_prog = states.get(ITEM_IN_PROGRESS, 0)
        with ctx.active_lock:
            active = ctx.active_ticker
            phase = ctx.active_phase
            elapsed = time.monotonic() - ctx.active_since if active else 0.0
        _log_worker(
            "Ingest heartbeat",
            finished=prog.get("finished"),
            pending=prog.get("pending"),
            in_progress=in_prog,
            states=states,
            active_ticker=active,
            active_phase=phase,
            active_sec=round(elapsed, 1) if active else 0,
        )
        if active and elapsed > 120:
            _log_worker(
                "Ticker running longer than 120s",
                level="WARN",
                ticker=active,
                phase=phase,
                elapsed_sec=round(elapsed, 1),
            )


def run_phased_ingest(ctx: IngestWorkerContext, *, use_parallel: bool) -> None:
    """Run ingest with sequential per-ticker or parallel phased worker pools."""
    cfg = stock_config()
    enrich_on = cfg.ingest_fetch_profile or cfg.ingest_fetch_insider or cfg.ingest_fetch_news
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(ctx, stop),
        name="ingest-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    _log_worker(
        "Phased ingest starting",
        parallel=use_parallel,
        workers=cfg.ingest_worker_count if use_parallel else 1,
        enrich=enrich_on,
    )

    try:
        if not use_parallel:
            _run_sequential_per_ticker(ctx)
            return

        workers = cfg.ingest_worker_count
        threads: list[threading.Thread] = [
            threading.Thread(target=_price_worker, args=(ctx,), name=f"ingest-price-{i}", daemon=True)
            for i in range(workers)
        ]
        fund_workers = max(1, cfg.ingest_fund_worker_count)
        threads.extend(
            threading.Thread(
                target=_fundamentals_worker,
                args=(ctx,),
                name=f"ingest-fund-{i}",
                daemon=True,
            )
            for i in range(fund_workers)
        )
        if enrich_on:
            threads.append(
                threading.Thread(target=_enrich_worker, args=(ctx,), name="ingest-enrich", daemon=True)
            )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)
        states = count_items_by_state(ctx.db_path, ctx.run_id)
        _log_worker("Phased ingest workers finished", states=states)
