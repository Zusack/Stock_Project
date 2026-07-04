"""Finnhub WebSocket live trade streaming, aggregated into intraday bars.

Finnhub's free tier streams individual trades over a WebSocket (no historical
candles). This service connects, aggregates trades into OHLCV bars sized by the
configured interval, persists completed bars to ``intraday_bars``, and publishes
``intraday_bar`` events for live charts.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.analysis.intraday_common import interval_to_seconds, utc_timestamp_str
from src.analysis.intraday_ingest import upsert_intraday_rows
from src.analysis.ticker_registry import list_focus_symbols
from src.services.event_bus import event_bus
from src.services.stock_config import stock_config
from src.utils.logger_utils import app_logger

_RING_MAXLEN = 500
_FINNHUB_WS_URL = "wss://ws.finnhub.io?token={token}"
_FLUSH_INTERVAL_SEC = 5.0


@dataclass
class LiveBarSnapshot:
    ticker: str
    timestamp: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    trade_count: int | None = None

    def to_row(self) -> tuple[Any, ...]:
        return (
            self.ticker,
            self.timestamp,
            self.interval,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.vwap,
            self.trade_count,
        )


@dataclass
class _WorkingBar:
    window_start: int  # epoch seconds (bar start)
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int


@dataclass
class _StreamState:
    thread: threading.Thread | None = None
    flush_thread: threading.Thread | None = None
    ws: Any = None
    running: bool = False
    symbols: list[str] = field(default_factory=list)
    interval: str = "1Min"
    stop_event: threading.Event = field(default_factory=threading.Event)


class LiveStreamService:
    """Singleton Finnhub trade stream with ring buffer + SQLite persistence."""

    _instance: LiveStreamService | None = None
    _lock = threading.Lock()

    def __new__(cls) -> LiveStreamService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._state = _StreamState()
                cls._instance._buffers: dict[str, deque[LiveBarSnapshot]] = {}
                cls._instance._buffer_lock = threading.Lock()
                cls._instance._working: dict[str, _WorkingBar] = {}
                cls._instance._working_lock = threading.Lock()
            return cls._instance

    @property
    def is_running(self) -> bool:
        return bool(self._state.running)

    @property
    def subscribed_symbols(self) -> list[str]:
        return list(self._state.symbols)

    def resolve_stream_symbols(
        self,
        *,
        chart_symbol: str | None = None,
        extra_symbols: list[str] | None = None,
    ) -> list[str]:
        """Build deduped symbol list for WebSocket subscription (max 30)."""
        cfg = stock_config()
        out: list[str] = []
        seen: set[str] = set()

        def _add(sym: str | None) -> None:
            if not sym:
                return
            s = str(sym).strip().upper()
            if not s or s.startswith("^") or s in seen:
                return
            seen.add(s)
            out.append(s)

        if cfg.live_stream_symbols_source != "custom":
            for row in list_focus_symbols(cfg.db_path):
                _add(row.symbol)

        _add(chart_symbol)
        for sym in extra_symbols or []:
            _add(sym)

        return out[:30]

    def get_buffer(self, ticker: str) -> list[LiveBarSnapshot]:
        sym = ticker.strip().upper()
        with self._buffer_lock:
            buf = self._buffers.get(sym)
            return list(buf) if buf else []

    def _append_buffer(self, snapshot: LiveBarSnapshot) -> None:
        sym = snapshot.ticker.upper()
        with self._buffer_lock:
            if sym not in self._buffers:
                self._buffers[sym] = deque(maxlen=_RING_MAXLEN)
            buf = self._buffers[sym]
            if buf and buf[-1].timestamp == snapshot.timestamp:
                buf[-1] = snapshot
            else:
                buf.append(snapshot)

    def _snapshot_from_working(self, sym: str, bar: _WorkingBar) -> LiveBarSnapshot:
        ts = datetime.fromtimestamp(bar.window_start, tz=timezone.utc)
        return LiveBarSnapshot(
            ticker=sym,
            timestamp=utc_timestamp_str(ts),
            interval=self._state.interval,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            vwap=None,
            trade_count=bar.trade_count,
        )

    def _emit(self, snapshot: LiveBarSnapshot, *, persist: bool) -> None:
        self._append_buffer(snapshot)
        if persist:
            try:
                upsert_intraday_rows(stock_config().db_path, [snapshot.to_row()])
            except Exception as exc:
                app_logger.log(
                    "LIVE_STREAM",
                    "Failed to persist intraday bar.",
                    level="WARN",
                    ticker=snapshot.ticker,
                    error=str(exc)[:200],
                )
        event_bus.emit(
            "intraday_bar",
            ticker=snapshot.ticker,
            bar=snapshot,
            interval=snapshot.interval,
        )

    def _handle_trade(self, sym: str, price: float, volume: int, epoch_ms: int) -> None:
        window_sec = interval_to_seconds(self._state.interval)
        window_start = int(math.floor((epoch_ms / 1000.0) / window_sec) * window_sec)

        finalized: LiveBarSnapshot | None = None
        in_progress: LiveBarSnapshot | None = None
        with self._working_lock:
            current = self._working.get(sym)
            if current is not None and current.window_start != window_start:
                finalized = self._snapshot_from_working(sym, current)
                current = None
            if current is None:
                current = _WorkingBar(
                    window_start=window_start,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume,
                    trade_count=1,
                )
            else:
                current.high = max(current.high, price)
                current.low = min(current.low, price)
                current.close = price
                current.volume += volume
                current.trade_count += 1
            self._working[sym] = current
            in_progress = self._snapshot_from_working(sym, current)

        if finalized is not None:
            self._emit(finalized, persist=True)
        if in_progress is not None:
            self._emit(in_progress, persist=False)

    def _flush_stale_bars(self) -> None:
        """Finalize bars whose window has elapsed (handles sparse/no-trade gaps)."""
        window_sec = interval_to_seconds(self._state.interval)
        now = time.time()
        to_finalize: list[LiveBarSnapshot] = []
        with self._working_lock:
            for sym, bar in list(self._working.items()):
                if now >= bar.window_start + window_sec:
                    to_finalize.append(self._snapshot_from_working(sym, bar))
                    del self._working[sym]
        for snapshot in to_finalize:
            self._emit(snapshot, persist=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(
        self,
        *,
        chart_symbol: str | None = None,
        extra_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._state.running:
            return {"ok": True, "message": "Stream already running.", "symbols": self._state.symbols}

        cfg = stock_config()
        if not cfg.finnhub_api_key:
            return {
                "ok": False,
                "error": "Finnhub API key required (Settings -> Intraday / Live Data).",
            }

        symbols = self.resolve_stream_symbols(
            chart_symbol=chart_symbol,
            extra_symbols=extra_symbols,
        )
        if not symbols:
            return {
                "ok": False,
                "error": (
                    "No symbols to stream. Enter a chart symbol, add symbols in "
                    "'Also stream', or add tickers to the focus watchlist."
                ),
            }

        self._state.stop_event.clear()
        self._state.symbols = symbols
        self._state.interval = cfg.intraday_interval
        self._state.running = True
        with self._working_lock:
            self._working.clear()

        self._state.thread = threading.Thread(
            target=self._run_stream_loop,
            name="finnhub-live-stream",
            daemon=True,
        )
        self._state.thread.start()
        self._state.flush_thread = threading.Thread(
            target=self._run_flush_loop,
            name="finnhub-bar-flush",
            daemon=True,
        )
        self._state.flush_thread.start()
        app_logger.log(
            "LIVE_STREAM",
            "Started Finnhub live stream.",
            level="INFO",
            symbols=symbols,
            interval=self._state.interval,
        )
        return {"ok": True, "symbols": symbols}

    def stop(self) -> None:
        if not self._state.running:
            return
        self._state.stop_event.set()
        ws = self._state.ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        for thread in (self._state.thread, self._state.flush_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=6.0)
        self._state.running = False
        self._state.ws = None
        self._state.thread = None
        self._state.flush_thread = None
        app_logger.log("LIVE_STREAM", "Stopped live stream.", level="INFO")

    def _run_flush_loop(self) -> None:
        while not self._state.stop_event.is_set():
            self._state.stop_event.wait(_FLUSH_INTERVAL_SEC)
            if self._state.stop_event.is_set():
                break
            try:
                self._flush_stale_bars()
            except Exception as exc:
                app_logger.log(
                    "LIVE_STREAM",
                    "Bar flush error.",
                    level="WARN",
                    error=str(exc)[:200],
                )

    def _run_stream_loop(self) -> None:
        import websocket  # websocket-client

        cfg = stock_config()
        symbols = list(self._state.symbols)
        token = cfg.finnhub_api_key
        backoff = 1.0

        def on_open(ws):
            for sym in symbols:
                try:
                    ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
                except Exception:
                    pass

        def on_message(_ws, message):
            if self._state.stop_event.is_set():
                return
            try:
                payload = json.loads(message)
            except (ValueError, TypeError):
                return
            if payload.get("type") != "trade":
                return
            for trade in payload.get("data", []) or []:
                sym = str(trade.get("s", "")).upper()
                if not sym:
                    continue
                try:
                    price = float(trade.get("p"))
                    volume = int(trade.get("v") or 0)
                    epoch_ms = int(trade.get("t"))
                except (TypeError, ValueError):
                    continue
                self._handle_trade(sym, price, volume, epoch_ms)

        def on_error(_ws, error):
            if not self._state.stop_event.is_set():
                app_logger.log(
                    "LIVE_STREAM",
                    "WebSocket error.",
                    level="WARN",
                    error=str(error)[:200],
                )

        while not self._state.stop_event.is_set():
            try:
                ws_app = websocket.WebSocketApp(
                    _FINNHUB_WS_URL.format(token=token),
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                )
                self._state.ws = ws_app
                backoff = 1.0
                ws_app.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                if self._state.stop_event.is_set():
                    break
                app_logger.log(
                    "LIVE_STREAM",
                    "Stream loop error; reconnecting.",
                    level="WARN",
                    error=str(exc)[:200],
                    backoff_sec=backoff,
                )
            finally:
                self._state.ws = None
            if self._state.stop_event.is_set():
                break
            self._state.stop_event.wait(backoff)
            backoff = min(backoff * 2.0, 30.0)

        self._state.running = False


def live_stream_service() -> LiveStreamService:
    return LiveStreamService()
