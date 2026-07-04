"""IBD-style leaderboard table columns, tooltips, and row builders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import flet as ft
import pandas as pd

from src.analysis.leaderboard import LeaderboardSegment
from src.analysis.leaderboard_scoring import CANSLIM_RULE_COUNT
from src.utils.format_utils import format_number
from src.views.theme import Palette, ThemeHelper


@dataclass(frozen=True)
class LeaderboardColumnDef:
    key: str
    label: str
    tooltip: str
    numeric: bool = False


LEADERBOARD_COLUMN_DEFS: tuple[LeaderboardColumnDef, ...] = (
    LeaderboardColumnDef(
        "rank",
        "Rank",
        "Position within the current segment list (1 = highest composite in segment).",
        numeric=True,
    ),
    LeaderboardColumnDef("ticker", "Symbol", "Ticker symbol. Use Actions to open analysis."),
    LeaderboardColumnDef(
        "industry",
        "Industry",
        "Industry group from stored profile data (Yahoo/SEC ingest).",
    ),
    LeaderboardColumnDef(
        "latest_price",
        "Price",
        "Last adjusted close in your local database.",
        numeric=True,
    ),
    LeaderboardColumnDef(
        "composite_score",
        "Composite",
        "Weighted 0–100 balanced score (v2): Momentum 35%, Quality 20%, "
        "Value 15%, Risk/Regime 20%, Sentiment 10%. Toggle v1 in settings for legacy weights.",
        numeric=True,
    ),
    LeaderboardColumnDef(
        "buy_readiness",
        "Buy Ready",
        "0–100 action score for new purchases when funds are available. "
        "Rewards setup, momentum, quality, and supportive market regime.",
        numeric=True,
    ),
    LeaderboardColumnDef(
        "sell_pressure",
        "Sell Press",
        "0–100 action score for trimming or selling holdings. "
        "Rises on trend breaks, weak RS, negative news, and risk flags.",
        numeric=True,
    ),
    LeaderboardColumnDef(
        "rating_value",
        "Value",
        "Letter grade (A–F) for valuation relative to sector/universe.",
    ),
    LeaderboardColumnDef(
        "rating_quality",
        "Quality",
        "Letter grade (A–F) for ROE, margins, and balance-sheet quality.",
    ),
    LeaderboardColumnDef(
        "rating_momentum",
        "Momentum",
        "Letter grade (A–F) for relative strength, pattern, and volume.",
    ),
    LeaderboardColumnDef(
        "rating_risk",
        "Risk Gr",
        "Letter grade (A–F) for volatility and regime alignment (higher = lower risk).",
    ),
    LeaderboardColumnDef(
        "data_confidence",
        "Conf",
        "Data confidence 0–1 based on history, fundamentals, and news coverage.",
        numeric=True,
    ),
    LeaderboardColumnDef(
        "canslim_score",
        "CANSLIM",
        f"How many of {CANSLIM_RULE_COUNT} letter rules pass on the latest bar: "
        f"C, A, N, S, L, M (shown as n/{CANSLIM_RULE_COUNT}). S may pass via strength "
        "or breakout volume. Not proprietary IBD EPS Rating. Stock Detail uses a "
        "separate 7-letter interactive score that also includes Institutions (I).",
        numeric=True,
    ),
    LeaderboardColumnDef(
        "rs_pct",
        "RS vs Mkt",
        "Six-month stock return minus market index return (%).",
        numeric=True,
    ),
    LeaderboardColumnDef(
        "volume_ratio",
        "Vol vs 50d",
        "Latest volume divided by 50-day average volume.",
        numeric=True,
    ),
    LeaderboardColumnDef(
        "near_high_pct",
        "% of 52w High",
        "How close the current price is to the 52-week high.",
        numeric=True,
    ),
    LeaderboardColumnDef(
        "pass_setup",
        "Setup",
        "Yes only if C, A, N, L, and M all pass on the latest bar (stricter than "
        f"the CANSLIM count — S is not required for Setup). When pattern is required "
        "in settings, cup-with-handle must pass too. A stock can show "
        f"5/{CANSLIM_RULE_COUNT} CANSLIM and still fail Setup.",
    ),
    LeaderboardColumnDef(
        "pass_pattern",
        "Pattern",
        "Yes if the cup-with-handle detector passes on the latest bar (used in "
        "composite score and Breakouts segment; not counted in the CANSLIM "
        f"n/{CANSLIM_RULE_COUNT} column).",
    ),
    LeaderboardColumnDef(
        "risk_flag",
        "Risk",
        "Risk flag: below 50-day simple moving average, weak market, or far from highs.",
    ),
)

METHODOLOGY_DIALOG_TEXT = """Leaderboard — proxy methodology (local data only)

Composite score v2 (0–100, default)
  • Momentum / Technical — 35%
  • Quality — 20%
  • Value — 15%
  • Risk / Regime — 20%
  • Sentiment / Catalyst — 10%

Composite v1 (legacy): CANSLIM 35%, Pattern 20%, RS 20%, Volume 15%, News 10%

Action scores: Buy Ready (deploy cash) · Sell Press (trim/sell urgency)

Segments include Buy Ready and Sell Pressure workflows.

Toggle Composite v1/v2 in Settings. Not proprietary IBD RS or list membership.
Run Research Universe ingest so prices, fundamentals, and news are current.
"""

SEGMENT_EMPTY_MESSAGES: dict[LeaderboardSegment, str] = {
    LeaderboardSegment.ALL_SCORED: (
        "No scored symbols yet. Click Refresh rankings or Score full universe."
    ),
    LeaderboardSegment.CANDIDATES: (
        "No setup candidates in this universe. Try All scored or refresh after ingest."
    ),
    LeaderboardSegment.WATCHLIST: (
        "No focus watchlist symbols in the scored set. Add symbols on the Watchlist tab."
    ),
    LeaderboardSegment.BREAKOUTS: (
        "No breakout patterns (setup + cup-with-handle pass) in this universe."
    ),
    LeaderboardSegment.RISK_FLAGS: (
        "No risk flags in this universe — nothing below 50-day simple moving average, weak market, or far from highs."
    ),
    LeaderboardSegment.BUY_READY: (
        "No scored symbols yet. Refresh rankings to compute buy readiness scores."
    ),
    LeaderboardSegment.SELL_PRESSURE: (
        "No sell-pressure signals in this universe."
    ),
}


def _dark(page) -> bool:
    return ThemeHelper.is_dark(page)


def composite_text_color(score: float, page) -> str:
    if score >= 80:
        return Palette.emerald.s400 if _dark(page) else Palette.emerald.s800
    if score >= 60:
        return Palette.amber.s400 if _dark(page) else Palette.amber.s800
    return Palette.rose.s400 if _dark(page) else Palette.rose.s800


def _badge(label: str, *, positive: bool, page) -> ft.Control:
    if positive:
        bg = Palette.emerald.s900 if _dark(page) else Palette.emerald.s100
        fg = Palette.emerald.s400 if _dark(page) else Palette.emerald.s800
    else:
        bg = Palette.slate.s800 if _dark(page) else Palette.slate.s200
        fg = ThemeHelper.text_muted(page)
    return ft.Container(
        content=ft.Text(label, size=11, color=fg, weight=ft.FontWeight.W_500),
        bgcolor=bg,
        padding=ft.Padding(8, 2, 8, 2),
        border_radius=4,
    )


def build_leaderboard_columns(
    page,
    *,
    on_sort=None,
) -> list[ft.DataColumn]:
    cols: list[ft.DataColumn] = []
    sort_kw = {"on_sort": on_sort} if on_sort else {}
    for col in LEADERBOARD_COLUMN_DEFS:
        cols.append(
            ft.DataColumn(
                ft.Text(col.label, weight=ft.FontWeight.W_600),
                tooltip=col.tooltip,
                numeric=col.numeric,
                **sort_kw,
            )
        )
    cols.append(
        ft.DataColumn(
            ft.Text("Actions"),
            tooltip="Analyze or add to focus watchlist.",
        )
    )
    return cols


def _fmt_num(val, *, decimals: int = 1) -> str:
    return format_number(val, decimals=decimals)


def _rating_color(letter: str, page) -> str:
    letter = (letter or "C").upper()
    if letter in ("A", "B"):
        return Palette.emerald.s400 if _dark(page) else Palette.emerald.s800
    if letter == "C":
        return ThemeHelper.text_muted(page)
    return Palette.rose.s400 if _dark(page) else Palette.rose.s800


def build_leaderboard_row_cells(
    row: pd.Series,
    *,
    rank: int,
    page,
    action_cell: ft.Control,
    symbol_click: Callable[[str], None] | None = None,
) -> list[ft.DataCell]:
    sym = str(row.get("ticker", ""))
    composite = float(row.get("composite_score", 0) or 0)
    canslim_n = int(row.get("canslim_score", 0) or 0)
    industry = str(row.get("industry", "") or row.get("sector", "") or "—")[:32]
    setup = bool(row.get("pass_setup", False))
    pattern = bool(row.get("pass_pattern", False))
    risk = str(row.get("risk_flag", "") or "—")
    buy_ready = float(row.get("buy_readiness", 0) or 0)
    sell_press = float(row.get("sell_pressure", 0) or 0)
    conf = float(row.get("data_confidence", 1) or 1)

    sym_cell: ft.Control = ft.Text(sym, weight=ft.FontWeight.W_600)
    if symbol_click is not None:
        sym_cell = ft.TextButton(sym, on_click=lambda e, s=sym: symbol_click(s))

    cells = [
        ft.DataCell(ft.Text(str(rank))),
        ft.DataCell(sym_cell),
        ft.DataCell(ft.Text(industry)),
        ft.DataCell(ft.Text(_fmt_num(row.get("latest_price"), decimals=2))),
        ft.DataCell(
            ft.Text(
                _fmt_num(composite, decimals=1),
                color=composite_text_color(composite, page),
                weight=ft.FontWeight.W_600,
            )
        ),
        ft.DataCell(
            ft.Text(
                _fmt_num(buy_ready, decimals=0),
                color=composite_text_color(buy_ready, page),
            )
        ),
        ft.DataCell(
            ft.Text(
                _fmt_num(sell_press, decimals=0),
                color=composite_text_color(100 - sell_press, page),
            )
        ),
        ft.DataCell(
            ft.Text(
                str(row.get("rating_value") or "—"),
                color=_rating_color(str(row.get("rating_value") or "C"), page),
                weight=ft.FontWeight.W_600,
            )
        ),
        ft.DataCell(
            ft.Text(
                str(row.get("rating_quality") or "—"),
                color=_rating_color(str(row.get("rating_quality") or "C"), page),
                weight=ft.FontWeight.W_600,
            )
        ),
        ft.DataCell(
            ft.Text(
                str(row.get("rating_momentum") or "—"),
                color=_rating_color(str(row.get("rating_momentum") or "C"), page),
                weight=ft.FontWeight.W_600,
            )
        ),
        ft.DataCell(
            ft.Text(
                str(row.get("rating_risk") or "—"),
                color=_rating_color(str(row.get("rating_risk") or "C"), page),
                weight=ft.FontWeight.W_600,
            )
        ),
        ft.DataCell(ft.Text(_fmt_num(conf * 100, decimals=0))),
        ft.DataCell(ft.Text(f"{canslim_n}/{CANSLIM_RULE_COUNT}")),
        ft.DataCell(ft.Text(_fmt_num(row.get("rs_pct"), decimals=1))),
        ft.DataCell(ft.Text(_fmt_num(row.get("volume_ratio"), decimals=2))),
        ft.DataCell(ft.Text(_fmt_num(row.get("near_high_pct"), decimals=1))),
        ft.DataCell(_badge("Yes" if setup else "No", positive=setup, page=page)),
        ft.DataCell(_badge("Yes" if pattern else "No", positive=pattern, page=page)),
        ft.DataCell(
            ft.Text(
                risk,
                color=ThemeHelper.text_error(page) if risk and risk != "—" else ThemeHelper.text_muted(page),
                size=11,
            )
        ),
        ft.DataCell(action_cell),
    ]
    return cells


def build_info_row(page, message: str) -> list[ft.DataRow]:
    ncols = len(LEADERBOARD_COLUMN_DEFS) + 1
    return [
        ft.DataRow(
            cells=[
                ft.DataCell(
                    ft.Text(message, color=ThemeHelper.text_muted(page)),
                )
            ]
            + [ft.DataCell(ft.Text(""))] * (ncols - 1)
        )
    ]


def build_leaderboard_rows(
    df: pd.DataFrame,
    *,
    page,
    action_builder: Callable[[str], ft.Control],
    symbol_click: Callable[[str], None] | None = None,
) -> list[ft.DataRow]:
    if df is None or df.empty:
        return []
    rows: list[ft.DataRow] = []
    for i, (_, row) in enumerate(df.iterrows()):
        sym = str(row.get("ticker", ""))
        cells = build_leaderboard_row_cells(
            row,
            rank=i + 1,
            page=page,
            action_cell=action_builder(sym),
            symbol_click=symbol_click,
        )
        rows.append(ft.DataRow(cells=cells))
    return rows
