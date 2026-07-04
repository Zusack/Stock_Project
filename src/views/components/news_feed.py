"""Clickable news headline cards for dashboard feeds."""
from __future__ import annotations

import webbrowser

import flet as ft

from src.analysis.market_news import FeedHeadline
from src.views.theme import ThemeHelper

DEFAULT_CATEGORY_LABELS: dict[str, str] = {
    "watchlist": "Watchlist",
    "market": "Market",
    "movers": "Top movers",
    "international": "International",
    "regulatory": "Regulatory",
}


def open_external_url(page: ft.Page | None, url: str) -> None:
    """Open a headline URL in the system browser (desktop-safe fallback)."""
    if not (url or "").strip():
        return
    if page is not None:
        try:
            async def _go() -> None:
                from flet.controls.services.url_launcher import LaunchMode, UrlLauncher

                await UrlLauncher().launch_url(url, mode=LaunchMode.EXTERNAL_APPLICATION)

            page.run_task(_go)
            return
        except Exception:
            pass
    webbrowser.open(url)


def build_feed_headline_card(
    page: ft.Page | None,
    item: FeedHeadline,
    *,
    category_labels: dict[str, str] | None = None,
) -> ft.Container:
    """Single headline row: category chip, title (clickable), publisher."""
    labels = category_labels or DEFAULT_CATEGORY_LABELS
    cat_label = labels.get(item.category, item.category.replace("_", " ").title())
    ticker_label = item.ticker if item.ticker and item.category != "regulatory" else ""
    link = (item.link or "").strip()
    title_color = ThemeHelper.accent_blue(page) if link else ThemeHelper.text_primary(page)

    title_control: ft.Control
    if link:
        title_control = ft.Container(
            content=ft.Text(
                item.title,
                size=14,
                weight=ft.FontWeight.W_500,
                color=title_color,
                max_lines=4,
            ),
            on_click=lambda e, u=link: open_external_url(page, u),
            ink=True,
            tooltip="Open article in browser",
        )
    else:
        title_control = ft.Text(
            item.title,
            size=14,
            weight=ft.FontWeight.W_500,
            color=title_color,
            max_lines=4,
        )

    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Text(cat_label, size=10, weight=ft.FontWeight.W_600),
                            bgcolor=ThemeHelper.card_feature_bg(page),
                            padding=ft.padding.symmetric(horizontal=8, vertical=2),
                            border_radius=4,
                        ),
                        ft.Text(
                            ticker_label,
                            size=11,
                            color=ThemeHelper.text_muted(page),
                            visible=bool(ticker_label),
                        ),
                        ft.Text(
                            item.date or "",
                            size=11,
                            color=ThemeHelper.text_muted(page),
                        ),
                    ],
                    spacing=8,
                    wrap=True,
                ),
                title_control,
                ft.Text(
                    item.publisher or item.source,
                    size=11,
                    color=ThemeHelper.text_muted(page),
                ),
            ],
            spacing=4,
            tight=True,
        ),
        padding=10,
        border=ft.border.all(1, ThemeHelper.border_subtle(page)),
        border_radius=6,
    )
