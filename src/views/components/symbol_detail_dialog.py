"""Symbol drill-down dialog for the Leaderboard tab."""

from __future__ import annotations

import flet as ft

from src.analysis.ai.advisor import get_signal_advisor
from src.analysis.news_signals import load_recent_headlines, score_ticker_news
from src.analysis.quote_snapshot import get_quote
from src.services.stock_config import stock_config
from src.views.components.insights_panel import build_insights_panel
from src.views.components.quote_panel import build_quote_header
from src.views.theme import ThemeHelper
from src.views.ui_helpers import navigate_to_stock_detail


def build_symbol_detail_dialog(page: ft.Page, ticker: str) -> ft.AlertDialog:
    """Build a modal with quote, news sentiment, and advisor insights."""
    sym = str(ticker).strip().upper()
    cfg = stock_config()
    quote = get_quote(cfg.db_path, sym)
    headlines = load_recent_headlines(cfg.db_path, sym, limit=6)
    sent, tags, _, catalysts = score_ticker_news(cfg.db_path, sym, headlines=headlines)
    suggestions = get_signal_advisor().analyze_ticker(sym, db_path=cfg.db_path)

    news_lines = []
    for h in headlines[:4]:
        news_lines.append(ft.Text(f"{h.get('date', '')}: {h.get('title', '')}", size=12))
    if not news_lines:
        news_lines.append(ft.Text("No recent headlines in database.", size=12))

    content = ft.Column(
        [
            build_quote_header(page, quote, sym) if quote else ft.Text(sym, size=16, weight=ft.FontWeight.W_600),
            ft.Text(f"News sentiment: {sent:.2f} · tags: {', '.join(tags[:4]) or '—'}", size=12),
            ft.Text(f"Catalysts: {', '.join(catalysts[:4]) or '—'}", size=12),
            ft.Column(news_lines, spacing=4, tight=True),
            build_insights_panel(page, suggestions),
        ],
        tight=True,
        spacing=10,
        scroll=ft.ScrollMode.AUTO,
        width=520,
        height=420,
    )

    def _open_detail(_e):
        navigate_to_stock_detail(sym, run_analysis=True)

    return ft.AlertDialog(
        modal=True,
        title=ft.Text(f"{sym} detail"),
        content=content,
        actions=[
            ft.TextButton("Open Stock Detail", on_click=_open_detail),
            ft.TextButton("Close"),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
        bgcolor=ThemeHelper.card_bg(page),
    )
