"""IBD-style proxy leaderboard rankings from local DB data."""

from __future__ import annotations

import gc
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

LeaderboardProgressCallback = Callable[[int, int, str], None]

from src.analysis.bulk_loaders import (
    load_fundamentals_eps_by_ticker,
    load_fundamentals_profile_map,
    load_history_grouped,
    load_profile_map,
    prepare_market_frame,
)
from src.analysis.db import db_connection, list_tickers

MIN_PARALLEL_UNIVERSE = 15
from src.analysis.canslim_core import CANSLM_RULE_COUNT, CANSLIM_RULE_COUNT
from src.analysis.leaderboard_scoring import (
    DEFAULT_SCORE_VERSION,
    TickerFeatures,
    apply_universe_scoring,
)
from src.analysis.market_context import load_latest_market_context
from src.analysis.ticker_evaluation import evaluate_ticker_snapshot, score_ticker_with_cache
from src.analysis.parallel_exec import default_worker_count, run_parallel_map
from src.services.stock_config import stock_config

_LB_WORKER_CTX: dict = {}


class LeaderboardSegment(str, Enum):
    ALL_SCORED = "all_scored"
    CANDIDATES = "candidates"
    WATCHLIST = "watchlist"
    BREAKOUTS = "breakouts"
    RISK_FLAGS = "risk_flags"
    BUY_READY = "buy_ready"
    SELL_PRESSURE = "sell_pressure"


@dataclass
class LeaderboardRow:
    ticker: str
    composite_score: float
    canslim_score: int
    pattern_quality: float
    rs_pct: float
    volume_ratio: float
    near_high_pct: float
    pass_setup: bool
    pass_pattern: bool
    risk_flag: str
    latest_price: float
    sector: str
    industry: str = ""
    notes: str = ""
    news_sentiment: float = 0.5
    news_tags: list[str] | None = None
    score_version: str = DEFAULT_SCORE_VERSION
    composite_score_v1: float = 0.0
    momentum_score: float = 0.0
    quality_score: float = 0.0
    value_score: float = 0.0
    risk_score: float = 0.0
    sentiment_score: float = 0.0
    buy_readiness: float = 0.0
    sell_pressure: float = 0.0
    rating_value: str = "C"
    rating_quality: str = "C"
    rating_momentum: str = "C"
    rating_risk: str = "C"
    data_confidence: float = 1.0
    score_components_json: str = ""
    canslm_lines: list[str] = field(default_factory=list)


def extract_ticker_features(
    ticker: str,
    db_path: str,
    market_ticker: str,
    *,
    market_df: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
    q_df: pd.DataFrame | None = None,
    a_df: pd.DataFrame | None = None,
    sector: str | None = None,
    industry: str | None = None,
    price_df: pd.DataFrame | None = None,
    headlines: list[dict] | None = None,
    fund_profile: dict[str, float | None] | None = None,
    market_regime: str = "neutral",
) -> TickerFeatures | None:
    """Compute raw per-ticker features before universe normalization."""
    snap = evaluate_ticker_snapshot(
        ticker,
        db_path,
        market_ticker,
        market_df=market_df,
        ohlcv=ohlcv,
        q_df=q_df,
        a_df=a_df,
        sector=sector,
        industry=industry,
        price_df=price_df,
        headlines=headlines,
        fund_profile=fund_profile,
        market_regime=market_regime,
        apply_scores=False,
    )
    return snap.features if snap is not None else None


def _scored_to_row(scored) -> LeaderboardRow:
    f = scored.features
    low_conf_note = ""
    if scored.data_confidence < 0.45:
        low_conf_note = "Low data confidence"
    return LeaderboardRow(
        ticker=f.ticker,
        composite_score=scored.composite_score,
        canslim_score=f.canslim_n,
        pattern_quality=round(f.pattern_q, 3),
        rs_pct=round(f.rs_pct, 2),
        volume_ratio=round(f.vol_ratio, 2),
        near_high_pct=round(f.near_high * 100, 1),
        pass_setup=f.pass_setup,
        pass_pattern=f.pass_pattern,
        risk_flag=f.risk_flag,
        latest_price=round(f.price, 2),
        sector=f.sector,
        industry=f.industry,
        notes=low_conf_note,
        news_sentiment=f.news_norm,
        news_tags=f.news_tags,
        score_version=scored.score_version,
        composite_score_v1=scored.composite_score_v1,
        momentum_score=scored.momentum_score,
        quality_score=scored.quality_score,
        value_score=scored.value_score,
        risk_score=scored.risk_score,
        sentiment_score=scored.sentiment_score,
        buy_readiness=scored.buy_readiness,
        sell_pressure=scored.sell_pressure,
        rating_value=scored.rating_value,
        rating_quality=scored.rating_quality,
        rating_momentum=scored.rating_momentum,
        rating_risk=scored.rating_risk,
        data_confidence=scored.data_confidence,
        score_components_json=scored.components_json(),
        canslm_lines=list(f.canslm_lines),
    )


def score_ticker(
    ticker: str,
    db_path: str,
    market_ticker: str,
    *,
    market_df: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
    q_df: pd.DataFrame | None = None,
    a_df: pd.DataFrame | None = None,
    sector: str | None = None,
    industry: str | None = None,
    price_df: pd.DataFrame | None = None,
    headlines: list[dict] | None = None,
    fund_profile: dict[str, float | None] | None = None,
    market_regime: str = "neutral",
    score_version: str | None = None,
) -> LeaderboardRow | None:
    """Compute leaderboard metrics for one symbol (single-ticker path)."""
    return score_ticker_with_cache(
        ticker,
        db_path,
        market_ticker,
        market_df=market_df,
        ohlcv=ohlcv,
        q_df=q_df,
        a_df=a_df,
        sector=sector,
        industry=industry,
        price_df=price_df,
        headlines=headlines,
        fund_profile=fund_profile,
        market_regime=market_regime,
        score_version=score_version,
    )


def _init_leaderboard_worker(ctx: dict) -> None:
    global _LB_WORKER_CTX
    _LB_WORKER_CTX = ctx


def _extract_features_worker(ticker: str) -> TickerFeatures | None:
    ctx = _LB_WORKER_CTX
    hist = ctx.get("history", {})
    fund = ctx.get("fundamentals", {})
    profiles = ctx.get("profiles", {})
    fund_profiles = ctx.get("fund_profiles", {})
    prof = profiles.get(ticker.upper(), profiles.get(ticker, {}))
    ohlcv = hist.get(ticker)
    price_df = None
    if ohlcv is not None and not ohlcv.empty:
        if "Adj Close" in ohlcv.columns:
            price_df = (
                ohlcv[["Adj Close", "Volume"]].copy()
                if "Volume" in ohlcv.columns
                else ohlcv[["Adj Close"]].copy()
            )
    q_df, a_df = fund.get(ticker, (pd.DataFrame(), pd.DataFrame()))
    headlines_map = ctx.get("headlines") or {}
    sym_headlines = headlines_map.get(ticker.upper()) or headlines_map.get(ticker)
    fp = fund_profiles.get(ticker.upper()) or fund_profiles.get(ticker)
    return extract_ticker_features(
        ticker,
        ctx["db_path"],
        ctx["market_ticker"],
        market_df=ctx.get("market_df"),
        ohlcv=ohlcv,
        q_df=q_df,
        a_df=a_df,
        sector=prof.get("sector", "") if isinstance(prof, dict) else "",
        industry=prof.get("industry", "") if isinstance(prof, dict) else "",
        price_df=price_df,
        headlines=sym_headlines,
        fund_profile=fp,
        market_regime=ctx.get("market_regime", "neutral"),
    )


def fill_unscored_watchlist_gaps(
    df: pd.DataFrame | None,
    tickers: list[str] | None,
) -> pd.DataFrame:
    """Append placeholder rows for watchlist symbols that could not be scored."""
    if not tickers:
        return df.copy() if df is not None and not df.empty else pd.DataFrame()
    base = df.copy() if df is not None and not df.empty else pd.DataFrame()
    scored = (
        set(base["ticker"].astype(str).str.upper())
        if not base.empty and "ticker" in base.columns
        else set()
    )
    missing = [str(t).strip().upper() for t in tickers if str(t).strip().upper() not in scored]
    if not missing:
        return base
    placeholders = [
        {
            "ticker": sym,
            "composite_score": 0.0,
            "canslim_score": 0,
            "pattern_quality": 0.0,
            "rs_pct": 0.0,
            "volume_ratio": 0.0,
            "near_high_pct": 0.0,
            "pass_setup": False,
            "pass_pattern": False,
            "risk_flag": "No price data — ingest on Research Universe tab",
            "latest_price": 0.0,
            "sector": "",
            "industry": "",
            "notes": "Not scored",
            "data_confidence": 0.0,
        }
        for sym in missing
    ]
    extra = pd.DataFrame(placeholders)
    if base.empty:
        return extra
    return pd.concat([base, extra], ignore_index=True)


def filter_leaderboard_segment(
    df: pd.DataFrame,
    segment: LeaderboardSegment,
    db_path: str,
    *,
    limit: int | None = 100,
) -> pd.DataFrame:
    """Apply segment filter/sort to a pre-scored universe DataFrame."""
    if df.empty:
        return df

    out = df.copy()
    if segment == LeaderboardSegment.ALL_SCORED:
        out = out.sort_values("composite_score", ascending=False)
    elif segment == LeaderboardSegment.CANDIDATES:
        out = out[out["pass_setup"]].sort_values("composite_score", ascending=False)
    elif segment == LeaderboardSegment.BREAKOUTS:
        out = out[(out["pass_pattern"]) & (out["pass_setup"])].sort_values(
            "pattern_quality", ascending=False
        )
    elif segment == LeaderboardSegment.RISK_FLAGS:
        out = out[out["risk_flag"].astype(str).str.len() > 0].sort_values(
            "composite_score", ascending=True
        )
    elif segment == LeaderboardSegment.BUY_READY:
        sort_col = "buy_readiness" if "buy_readiness" in out.columns else "composite_score"
        out = out.sort_values(sort_col, ascending=False)
    elif segment == LeaderboardSegment.SELL_PRESSURE:
        sort_col = "sell_pressure" if "sell_pressure" in out.columns else "composite_score"
        out = out[out["sell_pressure"].fillna(0) > 0].sort_values(sort_col, ascending=False)
    elif segment == LeaderboardSegment.WATCHLIST:
        try:
            from src.analysis.watchlist_schema import resolve_watchlist_symbols

            wl = {s.upper() for s in resolve_watchlist_symbols(db_path)}
            out = out[out["ticker"].isin(wl)].sort_values("composite_score", ascending=False)
        except Exception:
            out = out.sort_values("composite_score", ascending=False)
    else:
        out = out.sort_values("composite_score", ascending=False)

    if limit is not None:
        out = out.head(limit)
    return out.reset_index(drop=True)


def build_leaderboard(
    db_path: str,
    market_ticker: str = "^DJI",
    *,
    tickers: list[str] | None = None,
    segment: LeaderboardSegment | None = LeaderboardSegment.CANDIDATES,
    limit: int = 100,
    progress_callback: LeaderboardProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    use_parallel: bool = True,
    workers: int | None = None,
    bulk_preload: bool = True,
    score_version: str | None = None,
) -> pd.DataFrame:
    """Rank universe and filter by segment."""
    from src.utils.logger_utils import app_logger

    t0 = time.perf_counter()
    version = score_version or stock_config().leaderboard_score_version
    universe = tickers or list_tickers(db_path)
    if market_ticker in universe:
        universe = [t for t in universe if t != market_ticker]

    total = len(universe)
    if progress_callback:
        progress_callback(0, total, "")

    if progress_callback:
        progress_callback(0, total, "Loading market data…")

    app_logger.log(
        "LEADERBOARD",
        "Scoring started.",
        level="INFO",
        universe_size=total,
        score_version=version,
        parallel=use_parallel,
    )
    market_df = prepare_market_frame(db_path, market_ticker)
    ctx_snap = load_latest_market_context(db_path)
    market_regime = ctx_snap.regime if ctx_snap else "neutral"

    if progress_callback:
        progress_callback(0, total, "Loading fundamentals and profiles…")

    fundamentals = load_fundamentals_eps_by_ticker(db_path) if bulk_preload else {}
    profiles = load_profile_map(db_path) if bulk_preload else {}
    fund_profiles = load_fundamentals_profile_map(db_path) if bulk_preload else {}

    history: dict[str, pd.DataFrame] = {}
    if bulk_preload and universe:
        if progress_callback:
            progress_callback(0, total, "Loading price history…")
        history = load_history_grouped(db_path, tickers=universe, full_ohlcv=True)

    headlines: dict[str, list] = {}
    if bulk_preload and universe:
        if progress_callback:
            progress_callback(0, total, "Loading headlines…")
        try:
            from src.analysis.news_signals import load_headlines_bulk

            headlines = load_headlines_bulk(db_path, universe, limit_per_ticker=10)
        except Exception:
            headlines = {}

    ctx = {
        "db_path": db_path,
        "market_ticker": market_ticker,
        "market_df": market_df,
        "fundamentals": fundamentals,
        "profiles": profiles,
        "fund_profiles": fund_profiles,
        "history": history,
        "headlines": headlines,
        "market_regime": market_regime,
    }

    feature_rows: list[TickerFeatures] = []
    done = 0
    use_parallel_effective = use_parallel and len(universe) >= MIN_PARALLEL_UNIVERSE

    if use_parallel_effective and len(universe) > 1:
        pending = list(universe)
        for result in run_parallel_map(
            _extract_features_worker,
            pending,
            use_parallel=True,
            workers=workers,
            initializer=_init_leaderboard_worker,
            initargs=(ctx,),
        ):
            if cancel_event is not None and cancel_event.is_set():
                break
            done += 1
            if progress_callback:
                sym = result.ticker if result else ""
                progress_callback(done, total, sym)
            if result is not None:
                feature_rows.append(result)
    else:
        for i, sym in enumerate(universe):
            if cancel_event is not None and cancel_event.is_set():
                break
            if progress_callback:
                progress_callback(i + 1, total, sym)
            q_df, a_df = fundamentals.get(sym, (pd.DataFrame(), pd.DataFrame()))
            prof = profiles.get(sym.upper(), profiles.get(sym, {}))
            sym_headlines = (headlines.get(sym.upper()) or headlines.get(sym)) if headlines else None
            fp = fund_profiles.get(sym.upper()) or fund_profiles.get(sym)
            feat = extract_ticker_features(
                sym,
                db_path,
                market_ticker,
                market_df=market_df,
                ohlcv=history.get(sym),
                q_df=q_df,
                a_df=a_df,
                sector=prof.get("sector", "") if isinstance(prof, dict) else "",
                industry=prof.get("industry", "") if isinstance(prof, dict) else "",
                price_df=(
                    history[sym][["Adj Close", "Volume"]].copy()
                    if sym in history
                    and "Adj Close" in history[sym].columns
                    and "Volume" in history[sym].columns
                    else None
                ),
                headlines=sym_headlines,
                fund_profile=fp,
                market_regime=market_regime,
            )
            if feat is not None:
                feature_rows.append(feat)

    if not feature_rows:
        app_logger.log("LEADERBOARD", "No scorable symbols in universe.", level="WARN")
        return pd.DataFrame()

    extract_sec = time.perf_counter() - t0
    if progress_callback:
        progress_callback(total, total, "Applying factor scores…")

    t_factors = time.perf_counter()
    scored = apply_universe_scoring(
        feature_rows, score_version=version, market_regime=market_regime
    )
    factor_sec = time.perf_counter() - t_factors
    rows = [_scored_to_row(s) for s in scored]

    del feature_rows, history, fundamentals, profiles, fund_profiles, headlines, ctx
    gc.collect()

    df = pd.DataFrame([r.__dict__ for r in rows])
    total_sec = time.perf_counter() - t0
    app_logger.log(
        "LEADERBOARD",
        "Scoring finished.",
        level="INFO",
        scored=len(df),
        extract_sec=round(extract_sec, 2),
        factor_sec=round(factor_sec, 2),
        total_sec=round(total_sec, 2),
        score_version=version,
    )
    if segment is None:
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return filter_leaderboard_segment(df, segment, db_path, limit=limit)


def rows_to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format for UI table."""
    if df.empty:
        return df
    out = df.copy()
    rename = {
        "ticker": "Ticker",
        "composite_score": "Composite",
        "canslim_score": "CANSLM",
        "industry": "Industry",
        "pattern_quality": "Pattern",
        "rs_pct": "RS%",
        "volume_ratio": "Vol x",
        "near_high_pct": "% of High",
        "pass_setup": "Setup",
        "pass_pattern": "Pattern OK",
        "risk_flag": "Risk",
        "latest_price": "Price",
        "sector": "Sector",
        "buy_readiness": "Buy Ready",
        "sell_pressure": "Sell Press",
        "rating_value": "Value",
        "rating_quality": "Quality",
        "rating_momentum": "Momentum",
        "rating_risk": "Risk Gr",
        "data_confidence": "Conf",
    }
    cols = [c for c in rename if c in out.columns]
    out = out[cols].rename(columns={k: rename[k] for k in cols})
    if "CANSLM" in out.columns:
        out["CANSLM"] = out["CANSLM"].map(
            lambda v: f"{int(v)}/{CANSLM_RULE_COUNT}" if pd.notna(v) else "—"
        )
    if "Setup" in out.columns:
        out["Setup"] = out["Setup"].map(lambda v: "Yes" if v else "No")
    if "Pattern OK" in out.columns:
        out["Pattern OK"] = out["Pattern OK"].map(lambda v: "Yes" if v else "No")
    return out
