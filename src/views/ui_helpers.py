"""Shared helpers for binding analysis results to Flet controls."""

from __future__ import annotations

import re

import flet as ft
import pandas as pd

from src.analysis.db import load_ohlcv
from src.services.event_bus import event_bus
from src.services.tab_indices import TAB_ASSISTANT, TAB_STOCK_DETAIL
from src.utils.format_utils import format_number, shorten
from src.views.components.chart_factory import chart_empty_state
from src.views.components.price_charts import build_candlestick_chart

BENCHMARK_OPTIONS: tuple[tuple[str, str], ...] = (
    ("^GSPC", "S&P 500"),
    ("SPY", "SPY"),
    ("QQQ", "QQQ"),
    ("^DJI", "Dow"),
)

_DEFAULT_CANDLE_SESSIONS = 120


def normalize_ticker_text(text: str | None, *, multi: bool = False) -> str:
    """Uppercase ticker symbol(s). Preserves separators when ``multi`` is True."""
    if text is None:
        return ""
    raw = str(text)
    if not raw.strip():
        return raw
    if not multi:
        return raw.strip().upper()
    parts = re.split(r"([\s,;]+)", raw)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[\s,;]+", part):
            out.append(part)
        else:
            out.append(part.strip().upper())
    return "".join(out)


def on_ticker_field_blur(*, multi: bool = False):
    """Return an ``on_blur`` handler that uppercases ticker input in a TextField."""

    def _handler(e: ft.ControlEvent) -> None:
        field = e.control
        current = field.value if field.value is not None else ""
        normalized = normalize_ticker_text(current, multi=multi)
        if normalized != current:
            field.value = normalized
            try:
                field.update()
            except RuntimeError:
                pass

    return _handler


def parse_ticker_filter(text: str | None) -> list[str] | None:
    if not text or not str(text).strip():
        return None
    normalized = normalize_ticker_text(text, multi=True)
    return [t.strip().upper() for t in normalized.replace(",", " ").split() if t.strip()]


def parse_symbols(text: str | None, *, exclude_indices: bool = False) -> list[str]:
    """Parse comma/space/semicolon-separated ticker symbols."""
    normalized = normalize_ticker_text(text, multi=True)
    parts = re.split(r"[\s,;]+", normalized.strip())
    out = [p for p in parts if p and (not exclude_indices or not p.startswith("^"))]
    return list(dict.fromkeys(out))


def navigate_to_assistant(ticker: str = "") -> None:
    """Emit navigation to the Assistant tab, optionally with a ticker context."""
    event_bus.emit(
        "navigate_tab",
        tab_index=TAB_ASSISTANT,
        ticker=str(ticker).strip().upper() if ticker else "",
    )
    if ticker:
        event_bus.emit("navigate_assistant", ticker=str(ticker).strip().upper())


def navigate_to_stock_detail(ticker: str, *, run_analysis: bool = False) -> None:
    """Emit navigation to the Stock Detail tab for a ticker."""
    event_bus.emit(
        "navigate_tab",
        tab_index=TAB_STOCK_DETAIL,
        ticker=str(ticker).strip().upper(),
        run_analysis=run_analysis,
    )


def dataframe_to_rows(df: pd.DataFrame, max_rows: int = 200) -> list[ft.DataRow]:
    if df is None or df.empty:
        return []
    view = df.head(max_rows)
    cols = [str(c) for c in view.columns]
    rows: list[ft.DataRow] = []
    for _, row in view.iterrows():
        cells = [ft.DataCell(ft.Text(_fmt_cell(row[c]))) for c in cols]
        rows.append(ft.DataRow(cells=cells))
    return rows


def dataframe_columns(df: pd.DataFrame) -> list[ft.DataColumn]:
    from src.views.components.tables import data_column

    return [data_column(str(c), tooltip=str(c)) for c in df.columns]


def _fmt_cell(val) -> str:
    return format_number(val, decimals=4 if isinstance(val, float) and val is not None and abs(float(val)) < 100 else 2)


def build_ticker_candlestick_chart(
    page: ft.Page,
    ticker: str,
    db_path: str,
    *,
    height: int = 360,
    sessions: int = _DEFAULT_CANDLE_SESSIONS,
) -> ft.Control:
    """Daily OHLC candlestick for a single ticker (last ``sessions`` trading days)."""
    df = load_ohlcv(ticker, db_path)
    if df is None or df.empty:
        return chart_empty_state(page, f"No OHLC history for {ticker} in database.")

    required = ("Open", "High", "Low", "Close")
    for col in required:
        if col not in df.columns:
            return chart_empty_state(page, f"Missing {col} column for {ticker}.")

    clean = df.dropna(subset=list(required)).tail(max(2, sessions))
    if len(clean) < 2:
        return chart_empty_state(page, f"Not enough OHLC bars to chart for {ticker}.")

    dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in clean.index]
    opens = clean["Open"].astype(float).tolist()
    highs = clean["High"].astype(float).tolist()
    lows = clean["Low"].astype(float).tolist()
    closes = clean["Close"].astype(float).tolist()

    return build_candlestick_chart(
        page,
        ticker,
        dates,
        opens,
        highs,
        lows,
        closes,
        height=height,
    )
