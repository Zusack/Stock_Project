"""Unified single-ticker evaluation for Leaderboard, Stock Detail, and AI context."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from src.analysis.canslim_core import (
    CanslimBarEvaluation,
    evaluate_canslim_bar,
    format_canslm_tooltip,
)
from src.analysis.db import db_connection, load_ohlcv, load_price_data
from src.analysis.leaderboard_scoring import (
    DEFAULT_SCORE_VERSION,
    TickerFeatures,
    annualized_volatility_pct,
    apply_universe_scoring,
)
from src.analysis.market_context import load_latest_market_context, regime_alignment_score
from src.services.stock_config import stock_config

if False:  # TYPE_CHECKING
    from src.analysis.leaderboard import LeaderboardRow

_SESSION_ROWS: dict[tuple[str, str, str, str, str], "LeaderboardRow"] = {}


@dataclass
class TickerEvaluationSnapshot:
    """One-pass evaluation: bar metrics, feature row, and optional scored leaderboard row."""

    bar: CanslimBarEvaluation
    features: TickerFeatures
    leaderboard_row: "LeaderboardRow | None" = None


def resolve_market_regime(db_path: str) -> str:
    ctx = load_latest_market_context(db_path)
    return ctx.regime if ctx else "neutral"


def _profile_metrics(
    fund_profile: dict[str, float | None] | None,
) -> dict[str, float | None]:
    fund_profile = fund_profile or {}
    return {
        "roe": fund_profile.get("ROE"),
        "profit_margins": fund_profile.get("Profit_Margins"),
        "debt_to_equity": fund_profile.get("Debt_to_Equity"),
        "peg_ratio": fund_profile.get("PEG_Ratio"),
        "trailing_pe": fund_profile.get("Trailing_PE"),
    }


def _resolve_sector_industry(
    ticker: str,
    db_path: str,
    *,
    sector: str | None,
    industry: str | None,
) -> tuple[str, str]:
    sec = sector or ""
    ind = industry or ""
    if sec and ind:
        return sec, ind
    try:
        with db_connection(db_path, readonly=True) as conn:
            prof = conn.execute(
                "SELECT Sector, Industry FROM stock_profiles WHERE Ticker = ?",
                (ticker.upper(),),
            ).fetchone()
            if prof:
                if not sec and prof[0]:
                    sec = prof[0]
                if not ind and len(prof) > 1 and prof[1]:
                    ind = prof[1]
    except Exception:
        pass
    return sec, ind


def score_news_for_ticker(
    db_path: str,
    ticker: str,
    *,
    headlines: list[dict] | None = None,
    max_headlines: int = 3,
) -> tuple[float, list[str], float, list[str], bool]:
    """Shared news sentiment bundle for leaderboard and stock detail."""
    try:
        from src.analysis.news_signals import score_headlines_list, score_ticker_news

        if headlines is not None:
            news_norm, news_tags, news_confidence, catalyst_tags = score_headlines_list(
                headlines, ticker, max_headlines=max_headlines
            )
            return news_norm, news_tags, news_confidence, catalyst_tags, bool(headlines)
        news_norm, news_tags, news_confidence, catalyst_tags = score_ticker_news(
            db_path, ticker, max_headlines=max_headlines
        )
        has_news = news_norm != 0.5 or bool(news_tags)
        return news_norm, news_tags, news_confidence, catalyst_tags, has_news
    except Exception:
        return 0.5, [], 0.5, [], False


def features_from_bar(
    bar: CanslimBarEvaluation,
    *,
    ticker: str,
    sector: str,
    industry: str,
    fund_profile: dict[str, float | None] | None,
    news_norm: float,
    news_tags: list[str],
    news_confidence: float,
    catalyst_tags: list[str],
    has_news: bool,
    market_regime: str,
    ohlcv: pd.DataFrame | None,
    price_df: pd.DataFrame | None,
) -> TickerFeatures:
    """Map shared bar evaluation into leaderboard feature inputs."""
    metrics = _profile_metrics(fund_profile)
    vol_df = ohlcv if ohlcv is not None and not ohlcv.empty else price_df
    vol_ann = annualized_volatility_pct(vol_df)
    regime_align = regime_alignment_score(
        market_regime,
        pass_setup=bar.pass_setup,
        risk_flag=bar.risk_flag,
    )
    return TickerFeatures(
        ticker=ticker,
        canslim_n=bar.score,
        pattern_q=bar.pattern_quality,
        rs_pct=bar.rs_pct,
        vol_ratio=bar.vol_ratio,
        near_high=bar.near_high,
        news_norm=news_norm,
        news_confidence=news_confidence,
        news_tags=news_tags,
        catalyst_tags=catalyst_tags,
        pass_setup=bar.pass_setup,
        pass_pattern=bar.pass_pattern,
        risk_flag=bar.risk_flag,
        price=bar.price,
        sector=sector,
        industry=industry,
        roe=metrics["roe"],
        profit_margins=metrics["profit_margins"],
        debt_to_equity=metrics["debt_to_equity"],
        peg_ratio=metrics["peg_ratio"],
        trailing_pe=metrics["trailing_pe"],
        vol_annual_pct=vol_ann,
        regime_align=regime_align,
        has_eps_data=bar.has_eps_data,
        has_news=has_news,
        history_bars=bar.history_bars,
        canslm_lines=list(bar.lines),
    )


def evaluate_ticker_snapshot(
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
    market_regime: str | None = None,
    score_version: str | None = None,
    apply_scores: bool = True,
) -> TickerEvaluationSnapshot | None:
    """Evaluate one ticker once and optionally produce a scored LeaderboardRow."""
    sym = ticker.upper().strip()
    regime = market_regime or resolve_market_regime(db_path)
    sec, ind = _resolve_sector_industry(sym, db_path, sector=sector, industry=industry)

    bar = evaluate_canslim_bar(
        sym,
        db_path,
        market_ticker,
        market_df=market_df,
        ohlcv=ohlcv,
        q_df=q_df,
        a_df=a_df,
        price_df=price_df,
    )
    if bar is None:
        return None

    if price_df is None:
        price_df = load_price_data(sym, db_path, 'Date, "Adj Close", Volume')
    if ohlcv is None:
        ohlcv = load_ohlcv(sym, db_path)

    news_norm, news_tags, news_confidence, catalyst_tags, has_news = score_news_for_ticker(
        db_path, sym, headlines=headlines
    )
    features = features_from_bar(
        bar,
        ticker=sym,
        sector=sec,
        industry=ind,
        fund_profile=fund_profile,
        news_norm=news_norm,
        news_tags=news_tags,
        news_confidence=news_confidence,
        catalyst_tags=catalyst_tags,
        has_news=has_news,
        market_regime=regime,
        ohlcv=ohlcv,
        price_df=price_df,
    )

    row = None
    version = score_version or stock_config().leaderboard_score_version
    if apply_scores:
        from src.analysis.leaderboard import _scored_to_row

        scored = apply_universe_scoring([features], score_version=version, market_regime=regime)
        if scored:
            row = _scored_to_row(scored[0])

    snap = TickerEvaluationSnapshot(bar=bar, features=features, leaderboard_row=row)
    if row is not None:
        session_key = (sym, db_path, market_ticker, version, regime)
        _SESSION_ROWS[session_key] = row
    return snap


def apply_canslm_bar_to_leaderboard_row(
    row: "LeaderboardRow",
    bar: CanslimBarEvaluation,
) -> "LeaderboardRow":
    """Overlay live CANSLM metrics onto a cached leaderboard row."""
    return replace(
        row,
        canslim_score=bar.score,
        pattern_quality=round(bar.pattern_quality, 3),
        rs_pct=round(bar.rs_pct, 2),
        volume_ratio=round(bar.vol_ratio, 2),
        near_high_pct=round(bar.near_high * 100, 1),
        pass_setup=bar.pass_setup,
        pass_pattern=bar.pass_pattern,
        risk_flag=bar.risk_flag,
        latest_price=round(bar.price, 2),
        canslm_lines=list(bar.lines),
    )


def refresh_canslm_metrics_df(
    df: pd.DataFrame,
    db_path: str,
    market_ticker: str,
) -> pd.DataFrame:
    """Recompute live CANSLM letter scores and related technical columns for display."""
    if df is None or df.empty or "ticker" not in df.columns:
        return df

    from src.analysis.bulk_loaders import load_history_grouped, prepare_market_frame

    out = df.copy()
    tickers = out["ticker"].astype(str).str.upper().unique().tolist()
    market_df = prepare_market_frame(db_path, market_ticker)
    history = load_history_grouped(db_path, tickers=tickers, full_ohlcv=True)

    score_by_ticker: dict[str, int] = {}
    rs_by_ticker: dict[str, float] = {}
    vol_by_ticker: dict[str, float] = {}
    near_by_ticker: dict[str, float] = {}
    setup_by_ticker: dict[str, bool] = {}
    pattern_by_ticker: dict[str, bool] = {}
    risk_by_ticker: dict[str, str] = {}
    price_by_ticker: dict[str, float] = {}
    pattern_q_by_ticker: dict[str, float] = {}
    lines_by_ticker: dict[str, list[str]] = {}
    tooltip_by_ticker: dict[str, str] = {}

    for sym in tickers:
        ohlcv = history.get(sym)
        price_df = None
        if ohlcv is not None and not ohlcv.empty and "Adj Close" in ohlcv.columns:
            price_df = (
                ohlcv[["Adj Close", "Volume"]].copy()
                if "Volume" in ohlcv.columns
                else ohlcv[["Adj Close"]].copy()
            )
        bar = evaluate_canslim_bar(
            sym,
            db_path,
            market_ticker,
            market_df=market_df,
            ohlcv=ohlcv,
            price_df=price_df,
        )
        if bar is None:
            continue
        score_by_ticker[sym] = bar.score
        rs_by_ticker[sym] = round(bar.rs_pct, 2)
        vol_by_ticker[sym] = round(bar.vol_ratio, 2)
        near_by_ticker[sym] = round(bar.near_high * 100, 1)
        setup_by_ticker[sym] = bar.pass_setup
        pattern_by_ticker[sym] = bar.pass_pattern
        risk_by_ticker[sym] = bar.risk_flag
        price_by_ticker[sym] = round(bar.price, 2)
        pattern_q_by_ticker[sym] = round(bar.pattern_quality, 3)
        lines_by_ticker[sym] = list(bar.lines)
        tooltip_by_ticker[sym] = format_canslm_tooltip(bar.lines)

    sym_series = out["ticker"].astype(str).str.upper()

    def _set_col(col: str, mapping: dict) -> None:
        if not mapping:
            return
        out[col] = sym_series.map(mapping)

    _set_col("canslim_score", score_by_ticker)
    _set_col("rs_pct", rs_by_ticker)
    _set_col("volume_ratio", vol_by_ticker)
    _set_col("near_high_pct", near_by_ticker)
    _set_col("pass_setup", setup_by_ticker)
    _set_col("pass_pattern", pattern_by_ticker)
    _set_col("risk_flag", risk_by_ticker)
    _set_col("latest_price", price_by_ticker)
    _set_col("pattern_quality", pattern_q_by_ticker)
    out["canslm_lines"] = sym_series.map(lambda s: lines_by_ticker.get(s, []))
    out["canslm_tooltip"] = sym_series.map(lambda s: tooltip_by_ticker.get(s, ""))
    return out


def clear_ticker_evaluation_session_cache() -> None:
    _SESSION_ROWS.clear()


def score_ticker_with_cache(
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
    market_regime: str | None = None,
    score_version: str | None = None,
    prefer_leaderboard_cache: bool = True,
) -> "LeaderboardRow | None":
    """Score one ticker, reusing session or persisted leaderboard cache when possible."""
    from src.analysis.leaderboard import LeaderboardRow
    from src.analysis.leaderboard_cache import lookup_leaderboard_row

    sym = ticker.upper().strip()
    version = score_version or stock_config().leaderboard_score_version
    regime = market_regime or resolve_market_regime(db_path)
    session_key = (sym, db_path, market_ticker, version, regime)

    cached_session = _SESSION_ROWS.get(session_key)
    if cached_session is not None:
        return cached_session

    if prefer_leaderboard_cache:
        cached_row = lookup_leaderboard_row(db_path, sym)
        if cached_row is not None:
            bar = evaluate_canslim_bar(sym, db_path, market_ticker, market_df=market_df, ohlcv=ohlcv, price_df=price_df)
            if bar is not None:
                cached_row = apply_canslm_bar_to_leaderboard_row(cached_row, bar)
            _SESSION_ROWS[session_key] = cached_row
            return cached_row

    snap = evaluate_ticker_snapshot(
        sym,
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
        market_regime=regime,
        score_version=version,
        apply_scores=True,
    )
    if snap is None or snap.leaderboard_row is None:
        return None
    _SESSION_ROWS[session_key] = snap.leaderboard_row
    return snap.leaderboard_row
