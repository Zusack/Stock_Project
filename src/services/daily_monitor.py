"""Daily EOD monitoring: guidance scan, alerts, and digest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.analysis.data_quality import freshness_summary
from src.analysis.db import db_connection, ingest_write_lock
from src.analysis.guidance import (
    RecommendationBand,
    guidance_summary,
    run_guidance_scan,
)
from src.analysis.intelligence_schema import ensure_intelligence_schema
from src.analysis.market_context import build_market_context, persist_market_context
from src.analysis.portfolio_intelligence import build_all_template_plans
from src.services.stock_config import stock_config

ProgressCallback = Callable[[str], None]


@dataclass
class DailyDigest:
    scan_id: str
    generated_at: str
    market_regime: str
    guidance_by_band: dict[str, int] = field(default_factory=dict)
    new_high_priority: list[str] = field(default_factory=list)
    upgrades: list[str] = field(default_factory=list)
    downgrades: list[str] = field(default_factory=list)
    exit_watch: list[str] = field(default_factory=list)
    data_quality: dict[str, Any] = field(default_factory=dict)
    sector_leaders: list[str] = field(default_factory=list)
    sector_laggards: list[str] = field(default_factory=list)
    alert_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "generated_at": self.generated_at,
            "market_regime": self.market_regime,
            "guidance_by_band": self.guidance_by_band,
            "new_high_priority": self.new_high_priority,
            "upgrades": self.upgrades,
            "downgrades": self.downgrades,
            "exit_watch": self.exit_watch,
            "data_quality": self.data_quality,
            "sector_leaders": self.sector_leaders,
            "sector_laggards": self.sector_laggards,
            "alert_count": self.alert_count,
        }


def _emit_alert(
    db_path: str,
    *,
    ticker: str | None,
    alert_type: str,
    severity: str,
    title: str,
    detail: str,
    scan_id: str,
) -> None:
    ensure_intelligence_schema(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                INSERT INTO alert_events
                (created_at, ticker, alert_type, severity, title, detail, scan_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now, ticker, alert_type, severity, title, detail[:1000], scan_id),
            )
            conn.commit()


def _alerts_from_scan(db_path: str, scan_id: str) -> int:
    ensure_intelligence_schema(db_path)
    count = 0
    with db_connection(db_path, readonly=True) as conn:
        changes = conn.execute(
            """
            SELECT ticker, prior_band, new_band, change_type
            FROM guidance_changes WHERE scan_id = ?
            """,
            (scan_id,),
        ).fetchall()
        hp = conn.execute(
            """
            SELECT ticker FROM guidance_snapshots
            WHERE scan_id = ? AND recommendation_band = ?
            """,
            (scan_id, RecommendationBand.HIGH_PRIORITY.value),
        ).fetchall()
        exit_rows = conn.execute(
            """
            SELECT ticker, risk_notes FROM guidance_snapshots
            WHERE scan_id = ? AND recommendation_band = ?
            """,
            (scan_id, RecommendationBand.EXIT_WATCH.value),
        ).fetchall()

    for ticker, prior, new, ctype in changes:
        sev = "info" if ctype == "upgrade" else "warning"
        _emit_alert(
            db_path,
            ticker=ticker,
            alert_type="band_change",
            severity=sev,
            title=f"{ticker}: {prior} → {new}",
            detail=f"Recommendation band {ctype}",
            scan_id=scan_id,
        )
        count += 1

    for (ticker,) in hp:
        _emit_alert(
            db_path,
            ticker=ticker,
            alert_type="high_priority",
            severity="info",
            title=f"{ticker}: High priority candidate",
            detail="Passes setup with strong composite and regime alignment",
            scan_id=scan_id,
        )
        count += 1

    for ticker, notes in exit_rows[:20]:
        _emit_alert(
            db_path,
            ticker=ticker,
            alert_type="exit_watch",
            severity="warning",
            title=f"{ticker}: Exit watch",
            detail=(notes or "Risk flag raised")[:500],
            scan_id=scan_id,
        )
        count += 1

    return count


def run_daily_scan(
    db_path: str | None = None,
    *,
    ticker_limit: int = 200,
    progress: ProgressCallback | None = None,
) -> DailyDigest:
    """End-of-day pipeline: market context → guidance → portfolios → alerts."""
    cfg = stock_config()
    db_path = db_path or cfg.db_path
    ensure_intelligence_schema(db_path)

    def status(msg: str) -> None:
        if progress:
            progress(msg)

    status("Building market context…")
    ctx = build_market_context(db_path)
    persist_market_context(db_path, ctx)

    status("Running guidance scan…")
    from src.analysis.ticker_registry import list_focus_symbols

    scan_tickers = None
    if cfg.daily_scan_scope == "focus":
        scan_tickers = [r.symbol for r in list_focus_symbols(db_path)]
    scan_id, _rows = run_guidance_scan(
        db_path,
        tickers=scan_tickers,
        limit=ticker_limit,
        use_parallel=cfg.use_parallel,
    )

    status("Building portfolio templates…")
    build_all_template_plans(db_path, scan_id)

    status("Generating alerts…")
    alert_count = _alerts_from_scan(db_path, scan_id)

    summary = guidance_summary(db_path, scan_id)
    dq = freshness_summary(db_path)

    upgrades: list[str] = []
    downgrades: list[str] = []
    with db_connection(db_path, readonly=True) as conn:
        for ticker, _, _, ctype in conn.execute(
            "SELECT ticker, prior_band, new_band, change_type FROM guidance_changes WHERE scan_id = ?",
            (scan_id,),
        ).fetchall():
            if ctype == "upgrade":
                upgrades.append(ticker)
            elif ctype == "downgrade":
                downgrades.append(ticker)

    digest = DailyDigest(
        scan_id=scan_id,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        market_regime=summary.get("market_regime", ctx.regime),
        guidance_by_band=summary.get("by_band", {}),
        new_high_priority=summary.get("top_candidates", []),
        upgrades=upgrades,
        downgrades=downgrades,
        exit_watch=[],
        data_quality=dq,
        sector_leaders=ctx.sector_leaders,
        sector_laggards=ctx.sector_laggards,
        alert_count=alert_count,
    )

    with db_connection(db_path, readonly=True) as conn:
        digest.exit_watch = [
            r[0]
            for r in conn.execute(
                """
                SELECT ticker FROM guidance_snapshots
                WHERE scan_id = ? AND recommendation_band = ?
                LIMIT 30
                """,
                (scan_id, RecommendationBand.EXIT_WATCH.value),
            ).fetchall()
        ]

    cfg.set_last_analysis_label(f"Daily scan {scan_id}")
    status("Daily scan complete.")
    return digest


def load_recent_alerts(db_path: str, *, limit: int = 50, unacked_only: bool = False) -> list[dict]:
    ensure_intelligence_schema(db_path)
    sql = "SELECT id, created_at, ticker, alert_type, severity, title, detail, scan_id FROM alert_events"
    if unacked_only:
        sql += " WHERE acknowledged = 0"
    sql += " ORDER BY created_at DESC LIMIT ?"
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [
        {
            "id": r[0],
            "created_at": r[1],
            "ticker": r[2],
            "alert_type": r[3],
            "severity": r[4],
            "title": r[5],
            "detail": r[6],
            "scan_id": r[7],
        }
        for r in rows
    ]


def format_digest_text(digest: DailyDigest) -> str:
    lines = [
        f"Daily Market Digest — {digest.generated_at}",
        f"Scan: {digest.scan_id} | Regime: {digest.market_regime}",
        "",
        "Guidance bands:",
    ]
    for band, n in sorted(digest.guidance_by_band.items()):
        lines.append(f"  {band}: {n}")
    if digest.new_high_priority:
        lines.append(f"\nHigh priority: {', '.join(digest.new_high_priority[:15])}")
    if digest.upgrades:
        lines.append(f"Upgrades: {', '.join(digest.upgrades[:15])}")
    if digest.downgrades:
        lines.append(f"Downgrades: {', '.join(digest.downgrades[:15])}")
    if digest.exit_watch:
        lines.append(f"Exit watch: {', '.join(digest.exit_watch[:15])}")
    if digest.sector_leaders:
        lines.append(f"\nSector leaders (63d): {', '.join(digest.sector_leaders)}")
    if digest.sector_laggards:
        lines.append(f"Sector laggards: {', '.join(digest.sector_laggards)}")
    lines.append(f"\nAlerts generated: {digest.alert_count}")
    return "\n".join(lines)
