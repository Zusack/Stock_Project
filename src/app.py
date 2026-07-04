"""
Top-level Flet app — page setup, tab routing, theme broadcast, lifecycle.

Mirrors the architecture of the parent project's src/app.py but trimmed to
the bits a generic Flet UI needs:

- Apply flet_v1_compat shims at import time.
- Build a Tabs control with an AnimatedSwitcher body host.
- Lazy-refresh each view's data when its tab becomes active.
- Broadcast theme_changed via pubsub so every view re-paints in unison.
- Tear down the health monitor and event bus on close.
"""
from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timezone

import flet as ft

# IMPORTANT: import flet_v1_compat before any other view/component module so
# the chart re-exports and module helpers are installed before they're read.
from src.flet_v1_compat import apply_flet_v1_compat  # noqa: F401  (side-effect import)

from src.analysis.db import checkpoint_db_on_shutdown, db_connection
from src.analysis.ingest import ingest_stock_data
from src.analysis.market_calendar import is_weekday, last_completed_trading_day
from src.analysis.ticker_registry import ensure_registry
from src.services.event_bus import event_bus
from src.services.stock_config import stock_config
from src.services.global_control_service import GlobalControlService
from src.services.live_stream_service import live_stream_service
from src.services.theme_settings import app_theme
from src.services.ui_health_monitor import UIHealthMonitor
from src.utils.logger_utils import app_logger
from src.views.base_view import schedule_ui_update
from src.views.dashboard_view import DashboardView
from src.views.data_ingest_view import DataManagementView
from src.views.watchlist_view import WatchlistView
from src.views.portfolio_view import PortfolioView
from src.views.leaderboard_view import LeaderboardView
from src.views.live_view import LiveView
from src.views.optimization_view import OptimizationView
from src.views.llm_view import AssistantView
from src.views.settings_view import SettingsView
from src.views.single_ticker_view import SingleTickerView
from src.views.compare_view import CompareView
from src.views.strategy_backtest_view import StrategyBacktestView
from src.views.components.feedback import PersistentBanner
from src.views.theme import MotionSpec, Palette, ThemeHelper

_STARTUP_INDEX_TICKERS: tuple[str, ...] = ("^GSPC", "^DJI", "^IXIC")


def _latest_index_dates(db_path: str, tickers: list[str]) -> dict[str, date | None]:
    latest: dict[str, date | None] = {t: None for t in tickers}
    if not tickers:
        return latest
    placeholders = ", ".join(["?"] * len(tickers))
    sql = (
        f"SELECT Ticker, MAX(Date) as max_date FROM stock_history "
        f"WHERE Ticker IN ({placeholders}) GROUP BY Ticker"
    )
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(sql, tickers).fetchall()
    for ticker, max_date in rows:
        if not max_date:
            continue
        try:
            latest[str(ticker).upper()] = date.fromisoformat(str(max_date)[:10])
        except ValueError:
            latest[str(ticker).upper()] = None
    return latest


def _plan_startup_index_refresh(db_path: str, tickers: list[str]) -> dict:
    now_utc = datetime.now(timezone.utc)
    required_day = last_completed_trading_day(now_utc)
    latest_dates = _latest_index_dates(db_path, tickers)
    missing = [t for t in tickers if latest_dates.get(t) is None]
    stale = [t for t in tickers if latest_dates.get(t) is not None and latest_dates[t] < required_day]
    refresh = missing + stale
    return {
        "now_utc": now_utc,
        "is_business_day_utc": is_weekday(now_utc.date()),
        "settle_cutoff_passed_utc": bool(now_utc.hour >= 21),
        "required_day": required_day,
        "latest_dates": latest_dates,
        "missing": missing,
        "stale": stale,
        "refresh_tickers": refresh,
        "needs_refresh": bool(refresh),
    }


def _startup_refresh_market_indices(status_callback) -> dict:
    """
    Ensure DB schema is ready and refresh key market indices before first render.

    Runs on a worker thread; use status_callback for lightweight UI progress text.
    """
    cfg = stock_config()
    ensure_registry(cfg.db_path)
    status_callback("Checking latest market index data in database…")

    plan = _plan_startup_index_refresh(cfg.db_path, list(_STARTUP_INDEX_TICKERS))
    required_day = plan["required_day"]
    latest_dates = {
        ticker: (d.isoformat() if d else None) for ticker, d in plan["latest_dates"].items()
    }
    app_logger.log(
        "APP",
        "Startup index freshness check complete.",
        level="INFO",
        now_utc=plan["now_utc"].strftime("%Y-%m-%d %H:%M:%S UTC"),
        business_day_utc=plan["is_business_day_utc"],
        settle_cutoff_passed_utc=plan["settle_cutoff_passed_utc"],
        required_day=required_day.isoformat(),
        latest_dates=latest_dates,
        missing=plan["missing"],
        stale=plan["stale"],
    )
    if not plan["needs_refresh"]:
        status_callback(
            f"Index data already current through {required_day.isoformat()} — skipping refresh."
        )
        return {
            "total": len(_STARTUP_INDEX_TICKERS),
            "processed": len(_STARTUP_INDEX_TICKERS),
            "success": len(_STARTUP_INDEX_TICKERS),
            "skipped": len(_STARTUP_INDEX_TICKERS),
            "failed": 0,
            "paused": False,
            "startup_skipped": True,
            "required_day": required_day.isoformat(),
            "latest_dates": latest_dates,
        }

    tickers_to_refresh = plan["refresh_tickers"]
    status_callback(
        "Refreshing market indices… " + ", ".join(tickers_to_refresh)
    )

    def _progress(_pct: float, status_msg: str, _prog=None, _line=None) -> None:
        status = (status_msg or "").strip()
        if status:
            status_callback(f"Refreshing market indices… {status}")

    summary = ingest_stock_data(
        db_path=cfg.db_path,
        tickers=tickers_to_refresh,
        mode="smart",
        use_parallel=False,
        progress_callback=_progress,
    )
    if not summary.get("paused"):
        cfg.set_last_ingest_at(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    return summary


def main(page: ft.Page) -> None:
    """Flet entry point. Pass to ft.run(main=main)."""
    page.title = "Stock Analyzer"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.START
    page.padding = 10
    app_logger.log(
        "APP",
        "Page initialized.",
        level="DEBUG",
        width=page.width,
        height=page.height,
    )

    # Apply persisted theme + base Material theming.
    app_theme().apply_to_page(page)
    page.theme = ft.Theme(color_scheme_seed=Palette.indigo.s600, use_material3=True)
    page.dark_theme = ft.Theme(color_scheme_seed=Palette.indigo.s400, use_material3=True)
    try:
        page.theme_animation_style = ft.AnimationStyle(
            duration=ft.Duration(milliseconds=MotionSpec.STATUS_MS),
            curve=ft.AnimationCurve.EASE_IN_OUT,
        )
    except Exception:
        pass

    # Wire the singleton page reference (used by GlobalControlService loading spinner).
    global_control = GlobalControlService()
    global_control.register_page(page)

    # Start the UI health monitor (logs event-loop latency spikes).
    ui_monitor = UIHealthMonitor(page)
    ui_monitor.start()

    # ------------------------------------------------------------------
    # Lifecycle / cleanup
    # ------------------------------------------------------------------
    _cleanup_done = False

    def _shutdown_once(_event=None):
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        print("[APP] Closing app. Shutting down threads...")
        app_logger.log("APP", "Shutdown initiated.", level="INFO")
        try:
            checkpoint_db_on_shutdown(stock_config().db_path, truncate=False)
        except Exception as ex:
            app_logger.log("APP", "DB checkpoint on shutdown failed.", level="WARN", error=str(ex))
        try:
            live_stream_service().stop()
        except Exception as ex:
            app_logger.log("APP", "Live stream shutdown failed.", level="WARN", error=str(ex))
        ui_monitor.stop()
        event_bus.shutdown()

    page.on_close = _shutdown_once
    page.on_disconnect = _shutdown_once

    splash_status = ft.Text("Loading startup services…", size=13, color=ThemeHelper.text_muted(page))
    splash_shell = ft.Container(
        expand=True,
        alignment=ft.alignment.center,
        content=ft.Column(
            controls=[
                ft.Container(
                    width=108,
                    height=108,
                    border_radius=54,
                    alignment=ft.alignment.center,
                    bgcolor=ThemeHelper.card_feature_bg(page),
                    shadow=[ThemeHelper.shadow_pop(page)],
                    content=ft.Icon(ft.Icons.SHOW_CHART, size=54, color=ThemeHelper.text_on_dark_box(page)),
                ),
                ft.Text("Stock Analyzer", size=24, weight=ft.FontWeight.W_700),
                splash_status,
                ft.ProgressRing(width=34, height=34, stroke_width=3),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=14,
            tight=True,
        ),
    )
    root_host = ft.Container(expand=True, content=splash_shell)
    page.add(ft.SafeArea(expand=True, content=root_host))

    def _set_splash_status(message: str) -> None:
        msg = (message or "").strip() or "Loading…"

        def _ui():
            splash_status.value = msg
            try:
                splash_status.update()
            except Exception:
                try:
                    page.update()
                except Exception:
                    pass

        schedule_ui_update(page, _ui, critical=True, label="startup_status")

    _startup_banner_ref: list[PersistentBanner | None] = [None]
    _startup_banner_sticky: list[tuple[str, str] | None] = [None]

    def _sync_app_banner() -> None:
        banner = _startup_banner_ref[0]
        if banner is None:
            return
        if global_control.is_db_busy:
            banner.show(global_control.db_busy_message, severity="info")
        elif _startup_banner_sticky[0]:
            msg, sev = _startup_banner_sticky[0]
            banner.show(msg, severity=sev)
        else:
            banner.hide()
        try:
            page.update()
        except Exception:
            pass

    def _mount_main_tabs() -> None:
        dashboard = DashboardView(page)
        portfolio = PortfolioView(page)
        watchlist = WatchlistView(page)
        single_ticker = SingleTickerView(page)
        compare = CompareView(page)
        strategy_backtests = StrategyBacktestView(page)
        leaderboard = LeaderboardView(page)
        optimization = OptimizationView(page)
        live = LiveView(page)
        data_management = DataManagementView(page)
        assistant = AssistantView(page)
        settings = SettingsView(page)

        tab_views: list[ft.Control] = [
            dashboard,
            portfolio,
            watchlist,
            single_ticker,
            compare,
            strategy_backtests,
            leaderboard,
            optimization,
            live,
            data_management,
            assistant,
            settings,
        ]
        tab_names: list[str] = [
            "Dashboard",
            "Portfolio",
            "Watchlists",
            "Stock Detail",
            "Compare",
            "Strategy Backtests",
            "Leaderboard",
            "Optimization",
            "Live",
            "Data Management",
            "Assistant",
            "Settings",
        ]
        TAB_ICON_BY_NAME = {
            "Dashboard": ft.Icons.DASHBOARD,
            "Portfolio": ft.Icons.ACCOUNT_BALANCE_WALLET,
            "Watchlists": ft.Icons.BOOKMARK,
            "Stock Detail": ft.Icons.QUERY_STATS,
            "Compare": ft.Icons.COMPARE_ARROWS,
            "Strategy Backtests": ft.Icons.ANALYTICS,
            "Leaderboard": ft.Icons.LEADERBOARD,
            "Optimization": ft.Icons.TUNE,
            "Live": ft.Icons.CANDLESTICK_CHART,
            "Data Management": ft.Icons.STORAGE,
            "Assistant": ft.Icons.SMART_TOY,
            "Settings": ft.Icons.SETTINGS,
        }

        for i, v in enumerate(tab_views):
            v.key = f"main_tab_{i}"

        mounted_views_host = ft.AnimatedSwitcher(
            content=tab_views[0] if tab_views else ft.Container(),
            transition=ft.AnimatedSwitcherTransition.FADE,
            duration=MotionSpec.CONTENT_MS,
            reverse_duration=MotionSpec.CONTENT_MS,
            switch_in_curve=ft.AnimationCurve.EASE_OUT,
            switch_out_curve=ft.AnimationCurve.EASE_IN,
            expand=True,
        )

        _tab_switch_seq = 0

        def _set_active_view(idx: int) -> None:
            if 0 <= idx < len(tab_views):
                mounted_views_host.content = tab_views[idx]
                app_logger.log(
                    "APP",
                    "Active view set.",
                    level="DEBUG",
                    tab_index=idx,
                    tab_name=tab_names[idx],
                    view_type=type(tab_views[idx]).__name__,
                )

        def _safe_refresh(view_name: str, refresh_fn):
            try:
                refresh_fn()
            except RuntimeError as ex:
                if "Control must be added to the page first" in str(ex):
                    app_logger.log(
                        "APP",
                        "Refresh deferred until control mount.",
                        level="WARN",
                        view=view_name,
                        reason=str(ex),
                    )

                    def _retry():
                        try:
                            refresh_fn()
                        except Exception:
                            pass

                    schedule_ui_update(page, _retry, label=f"tab_refresh:{view_name}")
                    return
                app_logger.log("APP", "Refresh runtime error.", level="ERROR", view=view_name, error=str(ex))
            except Exception as ex:
                app_logger.log("APP", "Refresh failed.", level="ERROR", view=view_name, error=str(ex))

        def _run_tab_refresh(full_idx: int) -> None:
            view = tab_views[full_idx]
            name = tab_names[full_idx]
            if hasattr(view, "refresh_data_async") and getattr(view, "_fetch_data", None):
                view.refresh_data_async(label=f"tab:{name}")
            elif hasattr(view, "refresh_data"):
                _safe_refresh(name, view.refresh_data)

        def on_tab_change(e):
            nonlocal _tab_switch_seq
            idx = e.control.selected_index
            _tab_switch_seq += 1
            switch_seq = _tab_switch_seq
            global_control.active_tab_index = idx
            app_logger.log(
                "APP",
                "Tab changed.",
                level="DEBUG",
                selected_index=idx,
                selected_name=tab_names[idx] if 0 <= idx < len(tab_names) else "unknown",
            )

            _set_active_view(idx)
            try:
                page.update()
            except Exception as ex:
                app_logger.log("APP", "Page update after tab switch failed.", level="ERROR", error=str(ex))

            tab_label = tab_names[idx] if 0 <= idx < len(tab_names) else "Tab"
            global_control.push_tab_strip_loading(f"Loading {tab_label}…", flush_page=False)
            try:
                page.update()
            except Exception:
                pass

            async def _async_tab_load():
                await asyncio.sleep(0)
                try:
                    if switch_seq != _tab_switch_seq:
                        return
                    _run_tab_refresh(idx)
                finally:
                    global_control.pop_tab_strip_loading(flush_page=False)
                    try:
                        page.update()
                    except Exception:
                        pass

            try:
                page.run_task(_async_tab_load)
            except Exception:

                def _fallback():
                    try:
                        if switch_seq == _tab_switch_seq:
                            _run_tab_refresh(idx)
                    finally:
                        global_control.pop_tab_strip_loading(flush_page=False)
                        try:
                            page.update()
                        except Exception:
                            pass

                schedule_ui_update(page, _fallback, label=f"tab_load_fallback:{tab_label}")

        def on_pubsub_message(msg):
            if msg != "theme_changed":
                return
            app_logger.log("APP", "Theme change pubsub received.", level="DEBUG")

            def _refresh():
                try:
                    for view in tab_views:
                        if hasattr(view, "refresh_theme"):
                            view.refresh_theme()
                    page.update()
                except Exception as ex:
                    print(f"[APP] Theme refresh error: {ex}")
                    app_logger.log("APP", "Theme refresh error.", level="ERROR", error=str(ex))

            schedule_ui_update(page, _refresh)

        page.pubsub.subscribe(on_pubsub_message)

        def _on_navigate_tab(**kwargs):
            tab_index = kwargs.get("tab_index")
            ticker = kwargs.get("ticker")
            run_analysis = bool(kwargs.get("run_analysis"))
            if tab_index is None or not (0 <= tab_index < len(tab_views)):
                return
            tabs.selected_index = tab_index
            global_control.active_tab_index = tab_index
            _set_active_view(tab_index)
            if ticker and hasattr(single_ticker, "set_ticker"):
                single_ticker.set_ticker(str(ticker), run_analysis=run_analysis)
            try:
                page.update()
            except Exception:
                pass
            _run_tab_refresh(tab_index)

        event_bus.subscribe("navigate_tab", _on_navigate_tab)

        tab_strip_loading_ring = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        tab_strip_loading_label = ft.Text(
            "",
            size=12,
            visible=False,
            overflow=ft.TextOverflow.ELLIPSIS,
            max_lines=1,
            expand=False,
        )
        global_control.register_tab_strip_loading_controls(tab_strip_loading_ring, tab_strip_loading_label)

        _startup_banner_ref[0] = PersistentBanner(page, visible=False)
        global_control.register_db_banner_callback(_sync_app_banner)

        main_tab_bar = ft.TabBar(
            scrollable=True,
            tabs=[
                ft.Tab(
                    label=name,
                    icon=TAB_ICON_BY_NAME.get(name, ft.Icons.TAB),
                    tooltip=name,
                )
                for name in tab_names
            ],
        )
        tab_strip_header = ft.Row(
            [
                ft.Container(content=main_tab_bar, expand=True),
                ft.Row(
                    [tab_strip_loading_ring, tab_strip_loading_label],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            on_change=on_tab_change,
            length=len(tab_views),
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    tab_strip_header,
                    _startup_banner_ref[0],
                    mounted_views_host,
                ],
            ),
        )

        root_host.content = tabs
        page.update()
        app_logger.log(
            "APP",
            "Top-level tabs added to page.",
            level="INFO",
            tab_count=len(tab_views),
            selected_index=tabs.selected_index,
        )

        _run_tab_refresh(0)

    async def _startup_then_mount() -> None:
        _set_splash_status("Loading database…")
        await asyncio.sleep(0)
        pending_banner: tuple[str, str] | None = None
        try:
            summary = await asyncio.to_thread(_startup_refresh_market_indices, _set_splash_status)
            if summary.get("failed"):
                app_logger.log(
                    "APP",
                    "Startup market index ingest had failures.",
                    level="WARN",
                    failed=summary.get("failed"),
                    total=summary.get("total"),
                )
                pending_banner = (
                    "Startup index refresh had failures — check logs or run Data Ingest.",
                    "warning",
                )
        except Exception as ex:
            app_logger.log("APP", "Startup market index ingest failed.", level="WARN", error=str(ex))
            _set_splash_status("Startup market refresh failed — loading app…")
            pending_banner = (f"Startup market refresh failed: {ex}", "error")
            await asyncio.sleep(0.2)
        _set_splash_status("Loading interface…")
        await asyncio.sleep(0)
        _mount_main_tabs()
        if pending_banner:
            _startup_banner_sticky[0] = pending_banner
        _sync_app_banner()

    page.run_task(_startup_then_mount)


if __name__ == "__main__":
    print("Run main.py from the project root instead of this file.")
