"""Leaderboard — IBD-style proxy rankings and watchlist workflow."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone

import flet as ft
import pandas as pd

from src.analysis import ticker_registry as registry
from src.analysis.watchlist_schema import resolve_watchlist_symbols, sync_watchlist_members_to_registry
from src.analysis.ticker_evaluation import (
    clear_ticker_evaluation_session_cache,
    refresh_canslm_metrics_df,
)
from src.analysis.db import count_tickers
from src.analysis.leaderboard import (
    LeaderboardSegment,
    fill_unscored_watchlist_gaps,
    filter_leaderboard_segment,
)
from src.analysis.leaderboard_runner import run_leaderboard_build
from src.utils.format_utils import format_timestamp, natural_sort_key
from src.analysis.leaderboard_cache import (
    FRESH,
    INGEST_NEEDED,
    PARTIALLY_STALE,
    STALE_DATA,
    LeaderboardCacheFreshness,
    LeaderboardRunMeta,
    assess_leaderboard_cache_freshness,
    can_reuse_cached_leaderboard,
    compute_leaderboard_data_fingerprint,
    load_latest_leaderboard,
    save_leaderboard_snapshot,
)
from src.services.event_bus import event_bus
from src.services.stock_config import stock_config
from src.services.tab_indices import TAB_DATA_MANAGEMENT, TAB_STOCK_DETAIL
from src.views.base_view import BaseView
from src.views.components.feedback import AsyncStatusRow, show_snackbar
from src.views.components.layouts import ViewTitleBar
from src.views.components.leaderboard_table import (
    LEADERBOARD_COLUMN_DEFS,
    METHODOLOGY_DIALOG_TEXT,
    SEGMENT_EMPTY_MESSAGES,
    build_info_row,
    build_leaderboard_columns,
    build_leaderboard_rows,
)
from src.views.components.modals import show_expandable_text
from src.views.components.tables import (
    TableViewState,
    bordered_table_wrap,
    filter_rows,
    sort_rows,
    themed_data_table,
)
from src.views.theme import ButtonStyles, InputStyles, Palette, ThemeHelper
from src.views.ui_helpers import on_ticker_field_blur
from src.utils.progress_format import format_duration, format_eta_remaining
from src.views.base_view import schedule_ui_update

# Cap rows rendered in Flet — building thousands of DataRows blocks the event loop.
DEFAULT_DISPLAY_ROWS = 200

_SEGMENT_DISPLAY: dict[LeaderboardSegment, str] = {
    LeaderboardSegment.ALL_SCORED: "All scored",
    LeaderboardSegment.CANDIDATES: "Candidates",
    LeaderboardSegment.WATCHLIST: "Watchlist",
    LeaderboardSegment.BREAKOUTS: "Breakouts",
    LeaderboardSegment.BUY_READY: "Buy Ready",
    LeaderboardSegment.SELL_PRESSURE: "Sell Press",
    LeaderboardSegment.RISK_FLAGS: "Risk Flags",
}


def _universe_key(tickers: list[str] | None, *, full_universe: bool) -> str:
    if full_universe:
        return "full"
    if tickers is None:
        return "all_db"
    normalized = ",".join(sorted(t.upper() for t in tickers))
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class LeaderboardView(BaseView):
    _tab_index = 6

    def __init__(self, page: ft.Page):
        super().__init__(page)
        # Tab scrolls when headers/banners exceed the window; table region scrolls for long row lists.
        self.scroll = ft.ScrollMode.AUTO
        self.expand = True

        self._running = False
        self._cancel_event: threading.Event | None = None
        self._scored_cache: pd.DataFrame | None = None
        self._scored_at: datetime | None = None
        self._scored_universe_key: str | None = None
        self._scored_universe_label: str = ""
        self._last_elapsed_sec: float = 0.0
        self._progress_last_at: float = 0.0
        self._score_full_universe = False
        self._cache_stale = False
        self._cache_freshness: LeaderboardCacheFreshness | None = None
        self._loaded_from_db = False
        self._segment = LeaderboardSegment.ALL_SCORED
        self._table_build_generation = 0
        self._display_truncated_note = ""

        self.ticker_filter = InputStyles.text_field(
            page,
            label="Symbols to score (optional)",
            hint_text="AAPL MSFT — empty uses watchlist",
            expand=True,
            on_blur=on_ticker_field_blur(multi=True),
            tooltip=(
                "Applied when you click Refresh rankings: only these symbols are "
                "scored and cached. Leave empty to use the watchlist or full "
                "universe (Score full universe). Does not filter the table below."
            ),
        )
        self.limit_field = InputStyles.text_field(
            page,
            label="Max rows",
            value="",
            width=100,
            hint_text="All",
            tooltip=(
                f"Max rows shown in the table (default {DEFAULT_DISPLAY_ROWS}). "
                "Scoring still includes the full universe; this only limits UI rendering."
            ),
        )
        self.table_search = InputStyles.text_field(
            page,
            label="Filter table",
            hint_text="Symbol, industry, score…",
            expand=True,
            on_change=self._on_table_search,
            tooltip=(
                "Narrows the rows already shown in the table — does not change which "
                "symbols were scored. Use Symbols to score (optional) before refresh."
            ),
        )
        self._table_state = TableViewState(all_rows=[])

        self.refresh_btn = ft.ElevatedButton(
            "Refresh rankings",
            icon=ft.Icons.REFRESH,
            style=ButtonStyles.primary(),
            on_click=self._on_refresh,
            tooltip="Score the selected universe from local database data.",
        )
        self.stop_btn = ft.ElevatedButton(
            "Stop",
            icon=ft.Icons.STOP,
            style=ButtonStyles.secondary(),
            disabled=True,
            on_click=self._on_stop,
            tooltip="Stop after the current symbol.",
        )
        self.full_universe_btn = ft.OutlinedButton(
            "Score full universe",
            icon=ft.Icons.DATASET,
            on_click=self._on_score_full_universe,
            tooltip="Score all symbols with price history (slow).",
        )

        self.async_status = AsyncStatusRow(page)
        self.scoring_progress = ft.ProgressBar(value=0, visible=False)
        self.scoring_eta = ft.Text(
            "",
            size=12,
            color=ThemeHelper.text_muted(page),
            visible=False,
        )
        self.scoring_status = ft.Text(
            "",
            size=12,
            color=ThemeHelper.text_muted(page),
        )
        self.meta_text = ft.Text(
            "Not scored yet — click Refresh rankings.",
            size=12,
            color=ThemeHelper.text_muted(page),
        )
        self.stale_banner = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.INFO_OUTLINE, size=18, color=ThemeHelper.text_error(page)),
                    ft.Text(
                        "Watchlist or filter changed — refresh recommended.",
                        size=12,
                        color=ThemeHelper.text_error(page),
                        expand=True,
                    ),
                ],
                spacing=8,
            ),
            visible=False,
            padding=ft.Padding(10, 6, 10, 6),
            border_radius=6,
            bgcolor=(
                Palette.amber.s900
                if ThemeHelper.is_dark(page)
                else Palette.amber.s100
            ),
        )
        self._data_stale_message = ft.Text(
            "",
            size=12,
            color=ThemeHelper.text_error(page),
            expand=True,
        )
        self.data_stale_banner = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.WARNING_AMBER, size=18, color=ThemeHelper.text_error(page)),
                    ft.Column(
                        [
                            self._data_stale_message,
                            ft.TextButton(
                                "Open Research Universe →",
                                icon=ft.Icons.LIST_ALT,
                                on_click=self._go_research_universe,
                                style=ft.ButtonStyle(
                                    color=ThemeHelper.text_error(page),
                                    padding=ft.Padding(0, 0, 0, 0),
                                ),
                            ),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            visible=False,
            padding=ft.Padding(10, 6, 10, 6),
            border_radius=6,
            bgcolor=(
                Palette.rose.s900
                if ThemeHelper.is_dark(page)
                else Palette.rose.s50
            ),
        )

        self._segment_btn = ft.SegmentedButton(
            selected=[LeaderboardSegment.ALL_SCORED.value],
            allow_empty_selection=False,
            allow_multiple_selection=False,
            segments=[
                ft.Segment(
                    value=LeaderboardSegment.ALL_SCORED.value,
                    label=ft.Text("All scored"),
                    icon=ft.Icon(ft.Icons.TABLE_ROWS),
                    tooltip="Every symbol from the last refresh, ranked by composite.",
                ),
                ft.Segment(
                    value=LeaderboardSegment.CANDIDATES.value,
                    label=ft.Text("Candidates"),
                    icon=ft.Icon(ft.Icons.STAR),
                    tooltip="Setup pass only, ranked by composite score.",
                ),
                ft.Segment(
                    value=LeaderboardSegment.WATCHLIST.value,
                    label=ft.Text("Watchlist"),
                    icon=ft.Icon(ft.Icons.BOOKMARK),
                    tooltip="Saved watchlist symbols, ranked by composite.",
                ),
                ft.Segment(
                    value=LeaderboardSegment.BREAKOUTS.value,
                    label=ft.Text("Breakouts"),
                    icon=ft.Icon(ft.Icons.TRENDING_UP),
                    tooltip="Setup + cup-with-handle pattern pass.",
                ),
                ft.Segment(
                    value=LeaderboardSegment.BUY_READY.value,
                    label=ft.Text("Buy Ready"),
                    icon=ft.Icon(ft.Icons.SHOPPING_CART),
                    tooltip="Ranked by buy readiness — best when deploying cash.",
                ),
                ft.Segment(
                    value=LeaderboardSegment.SELL_PRESSURE.value,
                    label=ft.Text("Sell Press"),
                    icon=ft.Icon(ft.Icons.SELL),
                    tooltip="Ranked by sell pressure — trim or exit candidates.",
                ),
                ft.Segment(
                    value=LeaderboardSegment.RISK_FLAGS.value,
                    label=ft.Text("Risk Flags"),
                    icon=ft.Icon(ft.Icons.WARNING_AMBER),
                    tooltip=(
                        "Below 50-day simple moving average, weak market, or far from highs."
                    ),
                ),
            ],
            on_change=self._on_segment_change,
        )

        self.leaderboard_table = themed_data_table(
            page,
            columns=build_leaderboard_columns(page, on_sort=self._on_table_sort),
            rows=[],
            column_spacing=12,
        )
        # expand=False so the table grows with row count and the tab scroll can reach every row.
        self._table_wrap = bordered_table_wrap(page, self.leaderboard_table, expand=False)

        cfg = stock_config()
        ver = cfg.leaderboard_score_version
        if ver == "v1":
            legend = (
                "Composite v1 (0–100): CANSLM 35% · Pattern 20% · RS 20% · "
                "Volume 15% · News 10%"
            )
        else:
            legend = (
                "Composite v2 (0–100): Momentum 35% · Quality 20% · Value 15% · "
                "Risk 20% · Sentiment 10% · Buy Ready / Sell Press action scores"
            )
        self.legend_text = ft.Text(
            legend,
            size=11,
            color=ThemeHelper.text_muted(page),
        )
        self.methodology_btn = ft.TextButton(
            "See methodology",
            icon=ft.Icons.HELP_OUTLINE,
            on_click=self._on_methodology,
            tooltip="How composite score and segments are calculated.",
        )

        advanced_controls = ft.Column(
            [
                ft.Row(
                    [
                        self.ticker_filter,
                        self.limit_field,
                    ],
                    spacing=8,
                ),
            ],
            spacing=8,
        )
        self._advanced_tile = ft.ExpansionTile(
            title=ft.Text("Advanced scoring options", size=13),
            subtitle=ft.Text("Limit which symbols to score and max rows", size=11),
            expanded=False,
            controls=[advanced_controls],
        )

        self._rankings_header = ft.Row(
            [
                ft.Icon(ft.Icons.LEADERBOARD, size=18),
                ft.Text("Rankings", weight=ft.FontWeight.BOLD),
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self._rankings_panel = ft.Column(
            [
                ft.Row(
                    [self._rankings_header, self.table_search],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self._table_wrap,
            ],
            spacing=4,
        )

        self.controls = [
            ViewTitleBar("Leaderboard"),
            ft.Text(
                "Proxy IBD-style rankings from local data — not proprietary IBD ratings or list membership.",
                size=11,
                color=ThemeHelper.text_muted(page),
            ),
            ft.Divider(height=1),
            ft.Row(
                [self._segment_btn],
                wrap=True,
                spacing=8,
            ),
            ft.Row(
                [
                    self.refresh_btn,
                    self.full_universe_btn,
                    self.stop_btn,
                ],
                wrap=True,
                spacing=8,
            ),
            self.async_status,
            self.scoring_progress,
            self.scoring_eta,
            self.scoring_status,
            self.meta_text,
            self.stale_banner,
            self.data_stale_banner,
            ft.Row(
                [self.legend_text, self.methodology_btn],
                wrap=True,
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            self._advanced_tile,
            self._rankings_panel,
        ]

        self._setup_pubsub({"ingest_finished": self._on_ingest_finished})

    def _go_research_universe(self, _e) -> None:
        event_bus.emit("navigate_tab", tab_index=TAB_DATA_MANAGEMENT)

    def _on_ingest_finished(self, **_kwargs) -> None:
        """Re-evaluate data freshness after ingest without clearing stored scores."""
        if self._scored_cache is None:
            cfg = stock_config()
            self._try_load_persisted_cache(cfg.db_path)
        self._refresh_cache_freshness_async()

    def _on_methodology(self, _e) -> None:
        show_expandable_text(
            self.page_ref,
            title="Leaderboard methodology",
            text=METHODOLOGY_DIALOG_TEXT,
            max_width=720,
            max_height=520,
        )

    def _parse_display_limit(self) -> int:
        """Max rows shown in the table (always capped to keep UI responsive)."""
        raw = (self.limit_field.value or "").strip()
        if not raw:
            return DEFAULT_DISPLAY_ROWS
        try:
            return max(5, min(10_000, int(raw)))
        except ValueError:
            return DEFAULT_DISPLAY_ROWS

    def _df_for_display(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            self._display_truncated_note = ""
            return df
        limit = self._parse_display_limit()
        if len(df) <= limit:
            self._display_truncated_note = ""
            display = df.copy()
        else:
            self._display_truncated_note = (
                f"Showing top {limit} of {len(df)} in this segment "
                f"(raise Max rows to see more)."
            )
            display = df.head(limit).copy()
        cfg = stock_config()
        return refresh_canslm_metrics_df(display, cfg.db_path, cfg.market_ticker)

    def _parse_tickers(self) -> list[str] | None:
        raw = (self.ticker_filter.value or "").strip()
        if not raw:
            return None
        parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
        return parts or None

    def _effective_tickers(self) -> list[str] | None:
        explicit = self._parse_tickers()
        if explicit:
            return explicit
        if self._score_full_universe:
            return None
        return resolve_watchlist_symbols(stock_config().db_path)

    def _universe_count(self) -> int:
        tickers = self._effective_tickers()
        if tickers is not None:
            return len(tickers)
        cfg = stock_config()
        n = count_tickers(cfg.db_path)
        market = cfg.market_ticker
        return max(0, n - 1) if n else 0

    def _universe_scope_label(self) -> str:
        if self._parse_tickers():
            n = len(self._parse_tickers() or [])
            return f"{n} filtered symbol(s)"
        if self._score_full_universe:
            return "full universe"
        n = self._universe_count()
        return f"{n} watchlist symbol(s)" if n else "empty watchlist"

    def _current_universe_key(self) -> str:
        return _universe_key(
            self._effective_tickers(),
            full_universe=self._score_full_universe,
        )

    def _check_cache_stale(self) -> bool:
        if self._scored_cache is None or self._scored_universe_key is None:
            return False
        return self._scored_universe_key != self._current_universe_key()

    def _effective_tickers_for_fingerprint(self) -> list[str] | None:
        explicit = self._parse_tickers()
        if explicit:
            return explicit
        if self._score_full_universe:
            return None
        effective = self._effective_tickers()
        return effective

    def _try_load_persisted_cache(self, db_path: str) -> bool:
        if self._scored_cache is not None:
            return False
        loaded = load_latest_leaderboard(db_path)
        if loaded is None:
            return False
        if loaded.meta.universe_key != self._current_universe_key():
            return False
        self._scored_cache = loaded.df
        self._scored_at = loaded.meta.scored_at
        self._scored_universe_key = loaded.meta.universe_key
        self._scored_universe_label = loaded.meta.universe_label
        self._last_elapsed_sec = loaded.meta.elapsed_sec
        self._score_full_universe = loaded.meta.full_universe
        self._loaded_from_db = True
        return True

    def _refresh_cache_freshness(self) -> None:
        if self._scored_cache is None or self._scored_at is None:
            self._cache_freshness = None
            self.data_stale_banner.visible = False
            return
        cfg = stock_config()
        loaded = load_latest_leaderboard(cfg.db_path)
        if loaded is not None and loaded.meta.universe_key == (self._scored_universe_key or ""):
            meta = loaded.meta
        else:
            fingerprint, data_as_of = compute_leaderboard_data_fingerprint(
                cfg.db_path,
                universe_key=self._scored_universe_key or "",
                tickers=self._effective_tickers_for_fingerprint(),
                market_ticker=cfg.market_ticker,
            )
            scored_syms = sorted(
                self._scored_cache["ticker"].astype(str).str.upper().unique().tolist()
            )
            meta = LeaderboardRunMeta(
                scored_at=self._scored_at,
                universe_key=self._scored_universe_key or "",
                universe_label=self._scored_universe_label,
                market_ticker=cfg.market_ticker,
                full_universe=self._score_full_universe,
                symbol_count=len(self._scored_cache),
                elapsed_sec=self._last_elapsed_sec,
                data_fingerprint=fingerprint,
                data_as_of=data_as_of,
                tickers_json=json.dumps(scored_syms) if scored_syms else None,
            )
        self._cache_freshness = assess_leaderboard_cache_freshness(
            cfg.db_path,
            meta,
            current_universe_key=self._current_universe_key(),
            tickers=self._effective_tickers_for_fingerprint(),
        )
        show_data_banner = self._cache_freshness.state in (
            STALE_DATA,
            INGEST_NEEDED,
            PARTIALLY_STALE,
        )
        self.data_stale_banner.visible = show_data_banner
        if show_data_banner:
            self._data_stale_message.value = self._cache_freshness.detail

    def _persist_scored_cache(
        self,
        df: pd.DataFrame,
        *,
        universe_key: str,
        universe_label: str,
        elapsed_sec: float,
        tickers: list[str] | None,
    ) -> None:
        cfg = stock_config()
        fingerprint, data_as_of = compute_leaderboard_data_fingerprint(
            cfg.db_path,
            universe_key=universe_key,
            tickers=tickers,
            market_ticker=cfg.market_ticker,
        )
        scored_syms = sorted(
            df["ticker"].astype(str).str.upper().unique().tolist(),
            key=natural_sort_key,
        )
        tickers_json = json.dumps(scored_syms) if scored_syms else None
        meta = LeaderboardRunMeta(
            scored_at=self._scored_at or datetime.now(timezone.utc),
            universe_key=universe_key,
            universe_label=universe_label,
            market_ticker=cfg.market_ticker,
            full_universe=self._score_full_universe,
            symbol_count=len(df),
            elapsed_sec=elapsed_sec,
            data_fingerprint=fingerprint,
            data_as_of=data_as_of,
            tickers_json=tickers_json,
        )
        try:
            save_leaderboard_snapshot(cfg.db_path, df, meta=meta)
        except Exception as ex:
            show_snackbar(
                self.page_ref,
                f"Could not save leaderboard cache: {ex}"[:120],
                severity="warning",
            )
        self._loaded_from_db = False
        self._refresh_cache_freshness()

    def _segment_display_name(self) -> str:
        return _SEGMENT_DISPLAY.get(self._segment, self._segment.value.replace("_", " ").title())

    def _show_segment_loading(self, *, message: str | None = None) -> None:
        status = getattr(self, "async_status", None)
        if status is None:
            return
        label = self._segment_display_name()
        status.set_running(message or f"Loading {label}…")
        try:
            status.update()
        except RuntimeError:
            pass

    def _finish_table_status(self, row_count: int, *, total_in_segment: int | None = None) -> None:
        status = getattr(self, "async_status", None)
        if status is None:
            return
        name = self._segment_display_name()
        total = total_in_segment if total_in_segment is not None else row_count
        if row_count <= 0:
            msg = f"{name}: no rows in this segment"
        elif self._display_truncated_note and total > row_count:
            msg = f"{name}: showing top {row_count} of {total} rows"
        else:
            word = "row" if row_count == 1 else "rows"
            msg = f"{name}: {row_count} {word}"
        status.set_idle(msg)
        try:
            status.update()
        except RuntimeError:
            pass

    def _refresh_cache_freshness_async(self) -> None:
        """Assess coverage for thousands of symbols off the UI thread."""

        def _work() -> None:
            try:
                self._refresh_cache_freshness()
            except Exception as ex:
                from src.utils.logger_utils import app_logger

                app_logger.log(
                    "LEADERBOARD",
                    "Cache freshness refresh failed.",
                    level="DEBUG",
                    error=str(ex),
                )

            def _ui() -> None:
                self._update_meta_text(refresh_freshness=False)

            schedule_ui_update(self.page_ref, _ui, label="leaderboard_freshness")

        threading.Thread(
            target=_work,
            daemon=True,
            name="LeaderboardFreshness",
        ).start()

    def _update_meta_text(self, *, refresh_freshness: bool = False) -> None:
        if refresh_freshness:
            self._refresh_cache_freshness_async()
            return
        stale = self._check_cache_stale()
        self._cache_stale = stale
        self.stale_banner.visible = stale
        if self._scored_at is None or self._scored_cache is None:
            self.meta_text.value = "Not scored yet — click Refresh rankings."
        else:
            ts = format_timestamp(self._scored_at) + " UTC"
            n = len(self._scored_cache)
            elapsed = self._last_elapsed_sec
            source = " · restored from database" if self._loaded_from_db else ""
            fresh_note = ""
            if self._cache_freshness and self._cache_freshness.state == FRESH:
                fresh_note = " · data unchanged"
            trunc = f" · {self._display_truncated_note}" if self._display_truncated_note else ""
            self.meta_text.value = (
                f"Last scored: {n} symbols ({self._scored_universe_label}) · "
                f"{elapsed:.1f}s · {ts}{source}{fresh_note}{trunc}"
            )
        try:
            self.meta_text.update()
            self.stale_banner.update()
            self.data_stale_banner.update()
        except RuntimeError:
            pass

    def _filtered_from_cache(self) -> pd.DataFrame | None:
        if self._scored_cache is None or self._scored_cache.empty:
            return None
        cfg = stock_config()
        return filter_leaderboard_segment(
            self._scored_cache,
            self._segment,
            cfg.db_path,
            limit=None,
        )

    def _fetch_data(self) -> dict:
        """Lightweight tab refresh — never runs build_leaderboard."""
        cfg = stock_config()
        if self._scored_cache is None:
            self._try_load_persisted_cache(cfg.db_path)
        df = self._filtered_from_cache()
        return {
            "df": df,
            "segment": self._segment.value,
            "needs_refresh": self._scored_cache is None,
            "stale": self._check_cache_stale(),
            "empty_message": None,
        }

    def _begin_scoring_ui(self, message: str) -> None:
        """Show progress bar and status (mirrors Research Universe ingest start)."""
        self.async_status.set_running(message)
        self.scoring_progress.visible = True
        self.scoring_progress.value = None
        self.scoring_eta.visible = True
        self.scoring_eta.value = "Estimating time remaining…"
        self.scoring_status.value = message
        self.scoring_status.color = ThemeHelper.text_primary(self.page_ref)
        try:
            self.async_status.update()
            self.scoring_progress.update()
            self.scoring_eta.update()
            self.scoring_status.update()
            self.page_ref.update()
        except RuntimeError:
            pass

    def _update_scoring_ui(
        self,
        message: str,
        *,
        pct: float | None = None,
        eta: str | None = None,
    ) -> None:
        def _ui() -> None:
            self.scoring_progress.visible = True
            if pct is None:
                self.scoring_progress.value = None
            else:
                self.scoring_progress.value = max(0.0, min(1.0, pct))
            self.scoring_status.value = message
            self.scoring_status.color = ThemeHelper.text_primary(self.page_ref)
            self.async_status.set_running(message, progress=pct)
            if eta is not None:
                self.scoring_eta.value = eta
                self.scoring_eta.visible = bool(eta)
            try:
                self.scoring_progress.update()
                self.scoring_status.update()
                self.scoring_eta.update()
                self.async_status.update()
            except RuntimeError:
                pass

        self._safe_update_throttled("leaderboard_progress", 0.35, _ui)

    def _finish_scoring_ui(self, message: str, *, error: bool = False) -> None:
        def _ui() -> None:
            self.scoring_progress.visible = False
            self.scoring_progress.value = 0
            self.scoring_eta.visible = False
            self.scoring_eta.value = ""
            self.scoring_status.value = message
            self.scoring_status.color = (
                ThemeHelper.text_error(self.page_ref)
                if error
                else ThemeHelper.text_muted(self.page_ref)
            )
            if error:
                self.async_status.set_error(message)
            else:
                self.async_status.set_idle(message)
            try:
                self.scoring_progress.update()
                self.scoring_eta.update()
                self.scoring_status.update()
                self.async_status.update()
            except RuntimeError:
                pass

        self._safe_update_critical(_ui)

    def _set_controls_busy(self, busy: bool) -> None:
        self.refresh_btn.disabled = busy
        self.stop_btn.disabled = not busy
        self._segment_btn.disabled = busy
        self.ticker_filter.disabled = busy
        self.limit_field.disabled = busy
        self.full_universe_btn.disabled = busy

        def _ui():
            for c in (
                self.refresh_btn,
                self.stop_btn,
                self._segment_btn,
                self.ticker_filter,
                self.limit_field,
                self.full_universe_btn,
            ):
                try:
                    c.update()
                except RuntimeError:
                    pass

        self._safe_update(_ui, label="leaderboard_controls")

    def _on_stop(self, e) -> None:
        if self._cancel_event is not None and not self._cancel_event.is_set():
            self._cancel_event.set()
            self._update_scoring_ui(
                "Stop requested — finishing current symbol…",
                pct=None,
            )

    def _on_segment_change(self, e) -> None:
        selected = list(getattr(e.control, "selected", None) or [])
        if not selected:
            return
        try:
            self._segment = LeaderboardSegment(selected[0])
        except ValueError:
            self._segment = LeaderboardSegment.ALL_SCORED

        self._show_segment_loading()

        if self._scored_cache is not None:
            df = self._filtered_from_cache()
            self._apply_data(
                {
                    "df": df,
                    "segment": self._segment.value,
                    "needs_refresh": False,
                    "stale": self._check_cache_stale(),
                    "empty_message": None,
                }
            )
            return

        self._apply_data(
            {
                "df": None,
                "segment": self._segment.value,
                "needs_refresh": True,
                "stale": False,
                "empty_message": (
                    "Click Refresh rankings to score your watchlist. "
                    "Segment filters apply instantly after the first scan."
                ),
            }
        )

    def _on_refresh(self, e) -> None:
        if self._running:
            show_snackbar(
                self.page_ref,
                "Leaderboard scan already in progress.",
                severity="warning",
            )
            return
        self._score_full_universe = False
        self._start_refresh()

    def _on_score_full_universe(self, e) -> None:
        if self._running:
            return
        self._score_full_universe = True
        self._start_refresh()

    def refresh_data(self) -> None:
        self.refresh_data_async(label="leaderboard")

    def refresh_data_async(self, label: str | None = None) -> None:
        if self._running:
            return
        if stock_config().leaderboard_auto_refresh:
            self._start_refresh()
            return
        super().refresh_data_async(label=label)

    def _start_refresh(self) -> None:
        effective = self._effective_tickers()
        if effective is not None and len(effective) == 0:
            show_snackbar(
                self.page_ref,
                "Add symbols to the Watchlist tab first, or use Score full universe.",
                severity="warning",
            )
            return

        total = self._universe_count()
        scope = self._universe_scope_label()
        hint = (
            f"Scoring {total} symbols ({scope}). This may take several minutes."
            if total > 100
            else f"Scoring {total} symbols ({scope})…"
        )
        self._running = True
        self._cancel_event = threading.Event()
        self._set_controls_busy(True)
        self._begin_scoring_ui(hint)
        universe_key = self._current_universe_key()
        universe_label = scope

        def _work():
            from src.utils.logger_utils import app_logger

            self.global_control.push_db_activity("Leaderboard scoring")
            started = time.perf_counter()
            cfg = stock_config()
            cancel_ev = self._cancel_event
            app_logger.log(
                "LEADERBOARD",
                "UI refresh worker started.",
                level="INFO",
                universe_key=universe_key,
                universe_label=universe_label,
                full_universe=self._score_full_universe,
                score_version=cfg.leaderboard_score_version,
            )

            def on_progress(done: int, total_n: int, sym: str) -> None:
                now = time.perf_counter()
                if done > 0 and sym and not sym.startswith("Loading") and (now - self._progress_last_at) < 0.25:
                    return
                self._progress_last_at = now
                elapsed = now - started
                pct = (done / total_n) if total_n else 0.0
                eta_label = ""
                if done > 0 and total_n > done and sym and not sym.startswith("Loading"):
                    avg = elapsed / done
                    eta_label = format_eta_remaining(avg * (total_n - done))
                elif done == 0 and total_n > 0:
                    eta_label = "Estimating time remaining…"
                if sym and sym.startswith("Loading"):
                    msg = f"{sym} — {total_n} symbols"
                    bar_pct = None
                elif sym:
                    msg = f"Scoring {sym} — {done}/{total_n} ({int(pct * 100)}%)"
                    bar_pct = pct
                else:
                    msg = f"Preparing scan — {total_n} symbols…"
                    bar_pct = None
                self._update_scoring_ui(msg, pct=bar_pct, eta=eta_label)

            cancelled = False
            from_cache = False
            effective_tickers = self._effective_tickers()
            try:
                if effective_tickers is not None:
                    sync_watchlist_members_to_registry(cfg.db_path)
                self._update_scoring_ui(
                    "Checking for cached rankings…",
                    pct=None,
                    eta="",
                )
                reused = can_reuse_cached_leaderboard(
                    cfg.db_path,
                    universe_key=universe_key,
                    tickers=self._effective_tickers_for_fingerprint(),
                )
                if reused is not None and not (cancel_ev is not None and cancel_ev.is_set()):
                    self._update_scoring_ui(
                        f"Loading cached rankings ({len(reused.df)} symbols)…",
                        pct=1.0,
                        eta="",
                    )
                    df_all = reused.df
                    self._scored_at = reused.meta.scored_at
                    self._last_elapsed_sec = reused.meta.elapsed_sec
                    from_cache = True
                    self._loaded_from_db = True
                else:
                    df_all = run_leaderboard_build(
                        cfg.db_path,
                        cfg.market_ticker,
                        tickers=effective_tickers,
                        segment=None,
                        limit=999_999,
                        progress_callback=on_progress,
                        cancel_event=cancel_ev,
                        use_parallel=cfg.use_parallel,
                        workers=cfg.worker_count,
                    )
                    self._scored_at = datetime.now(timezone.utc)
                    self._last_elapsed_sec = time.perf_counter() - started

                if effective_tickers is not None:
                    df_all = fill_unscored_watchlist_gaps(df_all, effective_tickers)

                self._scored_cache = df_all if df_all is not None else pd.DataFrame()
                self._scored_universe_key = universe_key
                self._scored_universe_label = universe_label
                if not from_cache:
                    clear_ticker_evaluation_session_cache()
                cancelled = bool(cancel_ev is not None and cancel_ev.is_set())
                if (
                    not from_cache
                    and not cancelled
                    and self._scored_cache is not None
                    and not self._scored_cache.empty
                ):
                    self._persist_scored_cache(
                        self._scored_cache,
                        universe_key=universe_key,
                        universe_label=universe_label,
                        elapsed_sec=self._last_elapsed_sec,
                        tickers=effective_tickers,
                    )
                df = filter_leaderboard_segment(
                    self._scored_cache,
                    self._segment,
                    cfg.db_path,
                    limit=None,
                )
                app_logger.log(
                    "LEADERBOARD",
                    "Scoring worker preparing UI.",
                    level="INFO",
                    total_scored=len(self._scored_cache) if self._scored_cache is not None else 0,
                    segment_rows=len(df) if df is not None else 0,
                    from_cache=from_cache,
                    elapsed_sec=round(time.perf_counter() - started, 2),
                )
                data = {
                    "df": df,
                    "segment": self._segment.value,
                    "elapsed_sec": self._last_elapsed_sec,
                    "scored": total,
                    "cancelled": cancelled,
                    "from_cache": from_cache,
                    "needs_refresh": False,
                    "stale": False,
                    "empty_message": None,
                }
            except Exception as ex:
                data = {
                    "df": None,
                    "segment": self._segment.value,
                    "error": str(ex),
                    "elapsed_sec": time.perf_counter() - started,
                    "cancelled": False,
                    "needs_refresh": self._scored_cache is None,
                    "stale": False,
                    "empty_message": None,
                }
            finally:
                self.global_control.pop_db_activity()

            def _ui():
                self._running = False
                self._cancel_event = None
                self._set_controls_busy(False)
                elapsed = data.get("elapsed_sec", 0)
                n = len(data["df"]) if data.get("df") is not None and not data["df"].empty else 0
                if data.get("error"):
                    err_msg = f"Scoring failed: {data['error']}"
                    show_snackbar(self.page_ref, data["error"][:160], severity="error")
                    self._finish_scoring_ui(err_msg[:200], error=True)
                else:
                    self._apply_data(data)
                    if data.get("cancelled"):
                        summary = f"Stopped early — {n} rows in {elapsed:.1f}s."
                        sev = "warning"
                    elif data.get("from_cache"):
                        summary = (
                            f"Loaded cached rankings ({n} rows) — "
                            "database unchanged since last score."
                        )
                        sev = "info"
                    else:
                        dur = format_duration(elapsed)
                        summary = f"Scoring complete — {n} rows in {dur}."
                        sev = "success"
                    show_snackbar(self.page_ref, summary, severity=sev)
                    self._finish_scoring_ui(summary)
                self._update_meta_text(refresh_freshness=True)

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _show_symbol_detail(self, ticker: str) -> None:
        """Drill-down drawer with metrics, news, and guidance drivers."""
        import threading

        from src.views.components.symbol_detail_dialog import build_symbol_detail_dialog

        sym = str(ticker).strip().upper()

        def _work():
            dlg = build_symbol_detail_dialog(self.page_ref, sym)

            def _ui():
                self.page_ref.dialog = dlg
                dlg.open = True
                try:
                    self.page_ref.update()
                except RuntimeError:
                    pass

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _close_dialog(self, dlg: ft.AlertDialog) -> None:
        dlg.open = False
        try:
            self.page_ref.update()
        except RuntimeError:
            pass

    def _build_action_row(self, ticker: str) -> ft.Row:
        sym = ticker

        def _nav(_e):
            event_bus.emit(
                "navigate_tab",
                tab_index=TAB_STOCK_DETAIL,
                ticker=sym,
                run_analysis=True,
            )

        def _add_wl(_e):
            result = registry.add_to_focus(stock_config().db_path, sym)
            if result.get("ok"):
                show_snackbar(self.page_ref, f"{sym} added to focus watchlist.", severity="success")
                self._update_meta_text()
            else:
                show_snackbar(self.page_ref, result.get("error", "Could not add"), severity="error")

        return ft.Row(
            [
                ft.IconButton(
                    icon=ft.Icons.QUERY_STATS,
                    tooltip="Analyze in Single-Ticker",
                    on_click=_nav,
                ),
                ft.IconButton(
                    icon=ft.Icons.BOOKMARK_ADD,
                    tooltip="Add to focus watchlist",
                    on_click=_add_wl,
                ),
            ],
            spacing=0,
        )

    def _rebuild_table_columns(self) -> None:
        self.leaderboard_table.columns = build_leaderboard_columns(
            self.page_ref, on_sort=self._on_table_sort
        )

    def _apply_table_filter_sort(self) -> None:
        self.leaderboard_table.rows = self._table_state.visible_rows()
        if self._table_state.sort_column_index is not None:
            self.leaderboard_table.sort_column_index = self._table_state.sort_column_index
            self.leaderboard_table.sort_ascending = self._table_state.sort_ascending

    def _set_table_rows(self, rows: list[ft.DataRow]) -> None:
        self._table_state.all_rows = list(rows)
        self._apply_table_filter_sort()

    def _on_table_search(self, e: ft.ControlEvent) -> None:
        self._table_state.search_query = (e.control.value or "").strip()
        self._apply_table_filter_sort()
        try:
            self.leaderboard_table.update()
        except RuntimeError:
            pass

    def _on_table_sort(self, e: ft.DataColumnSortEvent) -> None:
        ci = e.column_index
        if self._table_state.sort_column_index == ci:
            self._table_state.sort_ascending = not self._table_state.sort_ascending
        else:
            self._table_state.sort_column_index = ci
            self._table_state.sort_ascending = e.ascending
        self._table_state.sort_numeric = (
            ci < len(LEADERBOARD_COLUMN_DEFS) and LEADERBOARD_COLUMN_DEFS[ci].numeric
        )
        self._apply_table_filter_sort()
        try:
            self.leaderboard_table.update()
        except RuntimeError:
            pass

    def _apply_data(self, data: dict) -> None:
        from src.utils.logger_utils import app_logger

        self._update_meta_text(refresh_freshness=False)
        df = data.get("df")
        needs_refresh = data.get("needs_refresh", False)
        empty_message = data.get("empty_message")

        self._rebuild_table_columns()
        if needs_refresh:
            msg = empty_message or (
                "Click Refresh rankings to score your watchlist from local data."
            )
            self._set_table_rows(build_info_row(self.page_ref, msg))
            self._flush_table_updates()
            self.async_status.set_idle(msg)
            try:
                self.async_status.update()
            except RuntimeError:
                pass
            return
        if df is None or df.empty:
            seg_msg = SEGMENT_EMPTY_MESSAGES.get(
                self._segment,
                "No rows for this segment.",
            )
            self._set_table_rows(build_info_row(self.page_ref, seg_msg))
            self._flush_table_updates()
            self._finish_table_status(0)
            return

        total_in_segment = len(df)
        display_df = self._df_for_display(df)
        self._update_meta_text(refresh_freshness=False)
        row_count = len(display_df) if display_df is not None else 0
        self._show_segment_loading(
            message=f"Building {self._segment_display_name()} table ({row_count} rows)…",
        )
        self._table_build_generation += 1
        gen = self._table_build_generation
        page = self.page_ref
        action_builder = self._build_action_row
        symbol_click = self._show_symbol_detail

        def _build_rows() -> list:
            t0 = time.perf_counter()
            rows = build_leaderboard_rows(
                display_df,
                page=page,
                action_builder=action_builder,
                symbol_click=symbol_click,
            )
            elapsed = time.perf_counter() - t0
            app_logger.log(
                "LEADERBOARD",
                "Table rows built off UI thread.",
                level="INFO",
                rows=row_count,
                build_sec=round(elapsed, 3),
                generation=gen,
            )
            return rows

        def _worker() -> None:
            try:
                rows = _build_rows()
            except Exception as ex:
                app_logger.log(
                    "LEADERBOARD",
                    f"Table build failed: {ex}",
                    level="ERROR",
                    generation=gen,
                )

                def _err_ui() -> None:
                    if gen != self._table_build_generation:
                        return
                    self._set_table_rows(
                        build_info_row(self.page_ref, f"Could not build table: {ex}"[:120])
                    )
                    self._flush_table_updates()
                    self.async_status.set_error(f"Could not build table: {ex}"[:120])

                schedule_ui_update(page, _err_ui, critical=True, label="leaderboard_table_err")
                return

            def _ui() -> None:
                if gen != self._table_build_generation:
                    return
                t0 = time.perf_counter()
                self._set_table_rows(rows)
                self._flush_table_updates()
                self._finish_table_status(len(rows), total_in_segment=total_in_segment)
                elapsed = time.perf_counter() - t0
                if elapsed > 1.0:
                    app_logger.log(
                        "LEADERBOARD",
                        "Slow table attach on UI thread.",
                        level="WARN",
                        rows=row_count,
                        attach_sec=round(elapsed, 3),
                        generation=gen,
                    )

            schedule_ui_update(page, _ui, critical=True, label="leaderboard_table")

        threading.Thread(target=_worker, daemon=True, name="LeaderboardTableBuild").start()

    def _flush_table_updates(self) -> None:
        try:
            self.leaderboard_table.update()
            self._table_wrap.update()
            self.update()
        except RuntimeError:
            pass

    def refresh_theme(self) -> None:
        page = self.page_ref
        self.async_status.refresh_theme(page)
        self.scoring_eta.color = ThemeHelper.text_muted(page)
        self.scoring_status.color = ThemeHelper.text_muted(page)
        self.meta_text.color = ThemeHelper.text_muted(page)
        self.legend_text.color = ThemeHelper.text_muted(page)
        self.stale_banner.bgcolor = ThemeHelper.status_bg(page, "warning")
        self.data_stale_banner.bgcolor = (
            Palette.rose.s900 if ThemeHelper.is_dark(page) else Palette.rose.s50
        )
        self._data_stale_message.color = ThemeHelper.text_error(page)
        InputStyles.refresh_fields(
            page,
            [self.ticker_filter, self.limit_field, self.table_search],
        )
        kwargs = ThemeHelper.results_summary_data_table_kwargs(page)
        for key, val in kwargs.items():
            setattr(self.leaderboard_table, key, val)
        self._table_wrap.border = ft.border.all(1, ThemeHelper.border_default(page))
        try:
            self.update()
        except RuntimeError:
            pass
