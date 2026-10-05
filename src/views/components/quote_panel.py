"""Reusable quote metrics header (Yahoo Finance-style)."""

from __future__ import annotations

import flet as ft

from src.analysis.quote_snapshot import QuoteSnapshot
from src.utils.format_utils import (
    format_change,
    format_currency,
    format_large_number,
    format_range,
    format_volume,
)
from src.views.components.cards import SelectableMetricCard
from src.views.theme import ThemeHelper


def build_quote_header(page: ft.Page, quote: QuoteSnapshot | None, ticker: str) -> ft.Control:
    """Build a quote summary row for Stock Detail and drill-down panels."""
    if quote is None:
        return ft.Text(
            f"No quote data for {ticker}. Check network or refresh.",
            color=ThemeHelper.text_muted(page),
            size=13,
        )

    price_color = ThemeHelper.text_primary(page)
    if quote.change_pct is not None:
        if quote.change_pct > 0:
            price_color = ThemeHelper.accent_green(page)
        elif quote.change_pct < 0:
            price_color = ThemeHelper.text_error(page)

    from_open_color = ThemeHelper.text_muted(page)
    if quote.change_from_open_pct is not None:
        if quote.change_from_open_pct > 0:
            from_open_color = ThemeHelper.accent_green(page)
        elif quote.change_from_open_pct < 0:
            from_open_color = ThemeHelper.text_error(page)

    change_parts: list[ft.Control] = [
        ft.Text(
            format_change(quote.change, quote.change_pct),
            size=16,
            color=price_color,
        ),
        ft.Text(
            "vs prev close",
            size=12,
            color=ThemeHelper.text_muted(page),
        ),
    ]
    if quote.change_from_open is not None or quote.change_from_open_pct is not None:
        change_parts.extend(
            [
                ft.Text("·", size=12, color=ThemeHelper.text_muted(page)),
                ft.Text(
                    format_change(quote.change_from_open, quote.change_from_open_pct),
                    size=14,
                    color=from_open_color,
                ),
                ft.Text(
                    "from open",
                    size=12,
                    color=ThemeHelper.text_muted(page),
                ),
            ]
        )

    header = ft.Column(
        [
            ft.Row(
                [
                    ft.Text(ticker, size=28, weight=ft.FontWeight.W_700),
                    ft.Text(
                        format_currency(quote.last_price),
                        size=28,
                        weight=ft.FontWeight.W_700,
                        color=price_color,
                    ),
                    ft.Row(
                        change_parts,
                        spacing=6,
                        wrap=True,
                        vertical_alignment=ft.CrossAxisAlignment.BASELINE,
                    ),
                ],
                spacing=16,
                wrap=True,
                vertical_alignment=ft.CrossAxisAlignment.BASELINE,
            ),
            ft.Text(
                f"{quote.sector} · {quote.industry}".strip(" · "),
                size=12,
                color=ThemeHelper.text_muted(page),
            ),
        ],
        spacing=4,
        tight=True,
    )

    cards = [
        SelectableMetricCard(page, title="Open", value=format_currency(quote.open), accent="teal"),
        SelectableMetricCard(
            page,
            title="Prev close",
            value=format_currency(quote.prev_close),
            accent="amber",
        ),
        SelectableMetricCard(
            page,
            title="Day range",
            value=format_range(quote.day_low, quote.day_high),
            accent="teal",
        ),
        SelectableMetricCard(
            page,
            title="52-week range",
            value=format_range(quote.fifty_two_week_low, quote.fifty_two_week_high),
            accent="teal",
        ),
        SelectableMetricCard(
            page,
            title="Volume",
            value=format_volume(quote.volume),
            subtitle=(
                f"vs avg {format_volume(int(quote.avg_volume_10d))}"
                if quote.avg_volume_10d
                else "avg n/a"
            ),
            accent="amber",
        ),
        SelectableMetricCard(
            page,
            title="Market cap",
            value=format_large_number(quote.market_cap),
            accent="primary",
        ),
        SelectableMetricCard(
            page,
            title="P/E (trail / fwd)",
            value=(
                f"{quote.trailing_pe:.1f} / {quote.forward_pe:.1f}"
                if quote.trailing_pe and quote.forward_pe
                else f"{quote.trailing_pe:.1f}" if quote.trailing_pe
                else f"{quote.forward_pe:.1f}" if quote.forward_pe
                else "—"
            ),
            accent="primary",
        ),
        SelectableMetricCard(
            page,
            title="EPS",
            value=format_currency(quote.eps) if quote.eps else "—",
            accent="primary",
        ),
        SelectableMetricCard(
            page,
            title="Beta",
            value=f"{quote.beta:.2f}" if quote.beta is not None else "—",
            accent="teal",
        ),
        SelectableMetricCard(
            page,
            title="Shares out",
            value=format_large_number(quote.shares_outstanding),
            accent="teal",
        ),
    ]

    return ft.Column(
        [
            header,
            ft.ResponsiveRow(
                [ft.Container(c, col={"xs": 12, "sm": 6, "md": 4, "lg": 3}) for c in cards],
                spacing=10,
                run_spacing=10,
            ),
        ],
        spacing=12,
    )
