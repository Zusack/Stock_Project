"""Dashboard — market overview, portfolio summary, movers, guidance."""

from __future__ import annotations

import time
from datetime import timedelta

import flet as ft
import pandas as pd

from src.analysis.db import load_prices_bulk
from src.analysis.guidance import guidance_summary
from src.analysis.market_context import load_latest_market_context
from src.analysis.market_news import FeedHeadline, build_dashboard_news_feed, clear_news_cache
from src.analysis.portfolio_schema import list_accounts, list_holdings
from src.analysis.quote_snapshot import get_quotes_bulk
from src.analysis.ticker_registry import list_focus_symbols
from src.analysis.watchlist_schema import list_members, list_watchlists
from src.services.event_bus import event_bus
from src.services.stock_config import stock_config
from src.services.tab_indices import TAB_PORTFOLIO
from src.utils.format_utils import format_change, format_currency, format_percent, slice_price_df
from src.views.base_view import BaseView
from src.views.ui_helpers import navigate_to_stock_detail
from src.views.components.chart_factory import chart_empty_state, chart_panel_container
from src.views.components.cards import SelectableMetricCard
from src.views.components.chart_range import (
    DEFAULT_CHART_RANGE,
    chart_range_label,
    chart_range_selector,
    selected_chart_range,
)
from src.views.components.model_setup_bar import ModelSetupBar
from src.views.components.display import StatusBanner
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.components.news_feed import build_feed_headline_card
from src.views.components.price_charts import IndexSeries, build_index_comparison_chart
from src.views.theme import ThemeHelper

_MARKET_INDICES: tuple[tuple[str, str], ...] = (
    ("^GSPC", "S&P 500"),
    ("^DJI", "Dow"),
    ("^IXIC", "Nasdaq"),
)
_INDEX_COLOR_KEYS = {"^GSPC": "price", "^DJI": "ma_short", "^IXIC": "ma_long"}



class DashboardView(BaseView):
    _tab_index = 0

    def __init__(self, page: ft.Page):
        super().__init__(page)
        self._fetch_cache: dict | None = None
        self._fetch_cache_at: float = 0.0

        self.status_banner = StatusBanner()
        self.model_setup_bar = ModelSetupBar(page)
        self.metric_regime = SelectableMetricCard(
            page, title="Market regime", value="—", subtitle="trend & breadth", accent="primary"
        )
        self.metric_guidance = SelectableMetricCard(
            page, title="High priority", value="—", subtitle="guidance scan", accent="teal"
        )
        self.metric_portfolio = SelectableMetricCard(
            page, title="Portfolio", value="—", subtitle="total positions", accent="amber"
        )
        self.metric_sp500 = SelectableMetricCard(
            page, title="S&P 500", value="—", subtitle="last close", accent="amber"
        )
        self.metric_dow = SelectableMetricCard(
            page, title="Dow", value="—", subtitle="last close", accent="amber"
        )
        self.metric_nasdaq = SelectableMetricCard(
            page, title="Nasdaq", value="—", subtitle="last close", accent="amber"
        )
        self._interval_key = DEFAULT_CHART_RANGE
        self._latest_market_data: dict[str, pd.DataFrame | None] = {}
        self.movers_list = ft.Column(spacing=4)
        self.news_feed = ft.Column(spacing=6)
        self.news_status = ft.Text("", size=12, color=ThemeHelper.text_muted(page))
        self.news_refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Refresh news headlines",
            on_click=self._on_refresh_news,
        )
        self.interval_selector = chart_range_selector(
            selected=self._interval_key,
            on_change=self._on_interval_changed,
        )
        self.index_chart_slot = ft.Container(content=chart_empty_state(page, "Loading…"))
        self.hint_text = ft.Text("", size=12, color=ThemeHelper.text_muted(page))

        self._setup_pubsub({"ingest_finished": self._on_ingest_finished})

        self.controls = [
            ViewTitleBar("Dashboard"),
            ft.Divider(),
            self.model_setup_bar,
            self.status_banner,
            SectionHeader("Overview", icon=ft.Icons.HOME, page_ref=page),
            ft.ResponsiveRow(
                [
                    ft.Container(self.metric_regime, col={"xs": 12, "sm": 6, "md": 4}),
                    ft.Container(self.metric_guidance, col={"xs": 12, "sm": 6, "md": 4}),
                    ft.Container(
                        ft.Container(
                            content=self.metric_portfolio,
                            on_click=lambda e: event_bus.emit("navigate_tab", tab_index=TAB_PORTFOLIO),
                        ),
                        col={"xs": 12, "sm": 6, "md": 4},
                    ),
                    ft.Container(self.metric_sp500, col={"xs": 12, "sm": 6, "md": 4}),
                    ft.Container(self.metric_dow, col={"xs": 12, "sm": 6, "md": 4}),
                    ft.Container(self.metric_nasdaq, col={"xs": 12, "sm": 6, "md": 4}),
                ],
                spacing=12,
            ),
            SectionHeader("Market indices", icon=ft.Icons.SHOW_CHART, page_ref=page),
            ft.Row([self.interval_selector]),
            chart_panel_container(page, self.index_chart_slot, panel_height=400),
            SectionHeader("Watchlist movers", icon=ft.Icons.TRENDING_UP, page_ref=page),
            self.movers_list,
            SectionHeader("Market news", icon=ft.Icons.NEWSPAPER, page_ref=page),
            ft.Row([self.news_status, self.news_refresh_btn]),
            self.news_feed,
            ft.Container(content=self.hint_text, padding=ft.padding.only(top=8)),
        ]

    def _fetch_data(self):
        cfg = stock_config()
        now = time.time()
        if self._fetch_cache and cfg.dashboard_cache_sec > 0 and (now - self._fetch_cache_at) < cfg.dashboard_cache_sec:
            return self._fetch_cache

        index_tickers = [t for t, _ in _MARKET_INDICES]
        market_data = load_prices_bulk(index_tickers, cfg.db_path, max_days=400)
        market_ctx = load_latest_market_context(cfg.db_path)
        guidance = guidance_summary(cfg.db_path)

        # Portfolio summary
        accounts = list_accounts(cfg.db_path)
        all_holdings = list_holdings(cfg.db_path)
        pos_count = len(all_holdings)

        # Watchlist movers — focus symbols or first named watchlist
        mover_syms: list[str] = []
        focus = list_focus_symbols(cfg.db_path)
        mover_syms = [r.symbol for r in focus[:10]]
        if not mover_syms:
            wls = list_watchlists(cfg.db_path)
            if wls:
                mover_syms = [m.ticker for m in list_members(cfg.db_path, wls[0].id)[:10]]
        quotes = get_quotes_bulk(cfg.db_path, mover_syms) if mover_syms else {}

        news_feed = build_dashboard_news_feed(
            cfg.db_path,
            watchlist_symbols=mover_syms,
        )

        payload = {
            "market_data": market_data,
            "market_context": market_ctx,
            "guidance_summary": guidance,
            "pos_count": pos_count,
            "accounts": accounts,
            "mover_syms": mover_syms,
            "quotes": quotes,
            "news_feed": news_feed,
            "last_ingest": cfg.last_ingest_at,
        }
        self._fetch_cache = payload
        self._fetch_cache_at = now
        return payload

    def _apply_data(self, data: dict) -> None:
        self.model_setup_bar.refresh()
        self.model_setup_bar.touch()
        self.status_banner.set_status("Market snapshot ready.", "SUCCESS")

        ctx = data.get("market_context")
        if ctx:
            self.metric_regime.set_value(
                ctx.regime.replace("_", " ").title(),
                f"Trend {ctx.trend_score:.0%} · Breadth {ctx.breadth_pct:.0%}",
            )
        else:
            self.metric_regime.set_value("—", "run daily scan")

        gsum = data.get("guidance_summary") or {}
        hp = int(gsum.get("by_band", {}).get("high_priority", 0))
        self.metric_guidance.set_value(str(hp) if gsum.get("scan_id") else "—", "high-priority signals")

        pos = data.get("pos_count", 0)
        accts = len(data.get("accounts") or [])
        self.metric_portfolio.set_value(str(pos), f"across {accts} account(s)")

        market_cards = {"^GSPC": self.metric_sp500, "^DJI": self.metric_dow, "^IXIC": self.metric_nasdaq}
        market_data = data.get("market_data") or {}
        self._latest_market_data = market_data
        for ticker, card in market_cards.items():
            mdf = market_data.get(ticker)
            if mdf is not None and len(mdf) >= 2:
                last = float(mdf["Adj Close"].iloc[-1])
                prev = float(mdf["Adj Close"].iloc[-2])
                chg = (last / prev - 1) if prev else 0
                card.set_value(format_currency(last), format_percent(chg * 100, signed=True))
            else:
                card.set_value("—", "no data")

        self._render_indices_chart()
        self._render_movers(data.get("mover_syms") or [], data.get("quotes") or {})
        self._render_news(data.get("news_feed") or [])

        last_ingest = data.get("last_ingest")
        self.hint_text.value = f"Last ingest: {last_ingest}" if last_ingest else ""
        try:
            self.update()
        except RuntimeError:
            pass

    def _render_movers(self, syms: list[str], quotes: dict) -> None:
        self.movers_list.controls = []
        if not syms:
            self.movers_list.controls.append(
                ft.Text("Add symbols to a watchlist to see movers.", size=12, color=ThemeHelper.text_muted(self.page_ref))
            )
            return
        ranked = sorted(
            syms,
            key=lambda s: abs(quotes[s].change_pct or 0) if s in quotes else 0,
            reverse=True,
        )
        for sym in ranked[:8]:
            q = quotes.get(sym)
            price_text = format_currency(q.last_price) if q else "—"
            chg_text = format_change(q.change, q.change_pct) if q else "quote pending"
            self.movers_list.controls.append(
                ft.Row(
                    [
                        ft.TextButton(
                            sym,
                            on_click=lambda e, s=sym: navigate_to_stock_detail(s),
                        ),
                        ft.Text(price_text, size=13),
                        ft.Text(chg_text, size=12),
                    ],
                    spacing=12,
                )
            )

    def _on_refresh_news(self, e) -> None:
        clear_news_cache(stock_config().db_path)
        self._fetch_cache = None
        self.refresh_data_async(label="dashboard_news_refresh")

    def _render_news(self, items: list[FeedHeadline]) -> None:
        self.news_feed.controls = []
        category_labels = {
            "watchlist": "Watchlist",
            "market": "Market",
            "movers": "Top movers",
            "international": "International",
            "regulatory": "Regulatory",
        }
        if not items:
            self.news_status.value = "No headlines loaded — check network and try Refresh."
            self.news_feed.controls.append(
                ft.Text(
                    "Headlines are fetched live from Yahoo Finance (and SEC RSS for regulatory news).",
                    size=12,
                    color=ThemeHelper.text_muted(self.page_ref),
                )
            )
            return

        self.news_status.value = f"{len(items)} headlines · watchlist, market, movers, international"

        for item in items:
            self.news_feed.controls.append(
                build_feed_headline_card(
                    self.page_ref,
                    item,
                    category_labels=category_labels,
                )
            )

    def _on_ingest_finished(self, **_kwargs) -> None:
        clear_news_cache(stock_config().db_path)
        self._fetch_cache = None
        if self._is_active_tab():
            self.refresh_data_async(label="dashboard_after_ingest")

    def refresh_data(self) -> None:
        self.refresh_data_async(label="dashboard")

    def _on_interval_changed(self, e) -> None:
        self._interval_key = selected_chart_range(
            e.control if e is not None else self.interval_selector
        )
        self._render_indices_chart()

    def _render_indices_chart(self) -> None:
        if not self._latest_market_data:
            return
        series_specs: list[IndexSeries] = []
        interval = self._interval_key
        for ticker, label in _MARKET_INDICES:
            df = slice_price_df(self._latest_market_data.get(ticker), interval)
            if df is None or df.empty:
                continue
            close = df["Adj Close"].astype(float).dropna()
            if len(close) < 2:
                continue
            base = float(close.iloc[0])
            if base == 0:
                continue
            pct = [((float(p) / base) - 1.0) * 100.0 for p in close.tolist()]
            dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in close.index]
            color = ThemeHelper.chart_named(self.page_ref, _INDEX_COLOR_KEYS.get(ticker, "price"))
            series_specs.append(
                IndexSeries(label=label, values=pct, color=color, timestamps=dates)
            )
        if series_specs:
            self.index_chart_slot.content = build_index_comparison_chart(
                self.page_ref,
                series_specs,
                interval_key=interval,
                height=300,
                subtitle=f"Interval: {chart_range_label(interval)}",
            )
        try:
            self.index_chart_slot.update()
        except RuntimeError:
            pass

    def refresh_theme(self) -> None:
        self.hint_text.color = ThemeHelper.text_muted(self.page_ref)
        self._render_indices_chart()
        try:
            self.update()
        except RuntimeError:
            pass
