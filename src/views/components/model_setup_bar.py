"""Shared local-AI model status bar (Dashboard + Assistant)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

import flet as ft

from src.llm.manager import llm_manager
from src.llm.utils.model_display_label import backend_display_name
from src.services.stock_config import stock_config
from src.services.tab_indices import TAB_ASSISTANT
from src.services.event_bus import event_bus
from src.views.base_view import schedule_ui_update
from src.views.theme import ThemeHelper

_PROBE_TTL_SEC = 30.0
_PROBE_TIMEOUT_SEC = 5.0
_conn_lock = threading.Lock()
_conn_ok: bool | None = None
_conn_checked_at: float = 0.0
_conn_probe_in_flight = False

StatusTone = Literal["success", "warning", "error", "info", "setup"]


@dataclass(frozen=True)
class ModelStatus:
    message: str
    tone: StatusTone
    setup_btn_text: str


def active_model_id() -> str:
    loaded = (llm_manager().loaded_model or "").strip()
    if loaded:
        return loaded
    return (stock_config().llm_chat_model or "").strip()


def resolve_model_status(
    *,
    lm_studio_enabled: bool,
    setup_mode: bool,
    connection_ok: bool | None,
    loaded_model: str,
    configured_model: str,
    backend_label: str,
) -> ModelStatus | None:
    """Pure status resolver for ModelSetupBar. None means the bar should be hidden."""
    if not lm_studio_enabled:
        return None
    if setup_mode:
        return ModelStatus(
            message="Choose a model and click Load selected.",
            tone="setup",
            setup_btn_text="Setup model",
        )

    active_loaded = (loaded_model or "").strip()
    configured = (configured_model or "").strip()

    if connection_ok is None:
        if active_loaded:
            return ModelStatus(
                message=f"Active model: {active_loaded}",
                tone="success",
                setup_btn_text="Change model",
            )
        return ModelStatus(
            message=f"Checking {backend_label} connection…",
            tone="info",
            setup_btn_text="Setup model",
        )

    if not connection_ok:
        if active_loaded:
            return ModelStatus(
                message=(
                    f"{backend_label} disconnected. Start it in the background "
                    "to continue using the loaded model."
                ),
                tone="error",
                setup_btn_text="Setup model",
            )
        return ModelStatus(
            message=(
                f"{backend_label} is not running. Start it in the background, "
                "then open Setup model to connect."
            ),
            tone="error",
            setup_btn_text="Setup model",
        )

    active = active_loaded or configured
    if active:
        return ModelStatus(
            message=f"Active model: {active}",
            tone="success",
            setup_btn_text="Change model",
        )
    return ModelStatus(
        message="Load a local AI model to enable insights, chat, and agents.",
        tone="warning",
        setup_btn_text="Setup model",
    )


def invalidate_backend_connection_cache() -> None:
    """Drop cached reachability so the next refresh re-probes."""
    global _conn_ok, _conn_checked_at
    with _conn_lock:
        _conn_ok = None
        _conn_checked_at = 0.0


def note_backend_connection(ok: bool) -> None:
    """Record a fresh reachability result (e.g. after load or Settings test)."""
    global _conn_ok, _conn_checked_at
    with _conn_lock:
        _conn_ok = ok
        _conn_checked_at = time.time()


def backend_connection_ok(*, max_age_sec: float = _PROBE_TTL_SEC) -> bool | None:
    """Cached reachability for UI. None means unknown or stale."""
    with _conn_lock:
        if _conn_ok is None:
            return None
        if (time.time() - _conn_checked_at) > max_age_sec:
            return None
        return _conn_ok


def navigate_to_model_setup() -> None:
    """Open Assistant → model setup from anywhere in the app."""
    event_bus.emit("navigate_tab", tab_index=TAB_ASSISTANT)
    event_bus.emit("navigate_assistant", open_models=True)


class ModelSetupBar(ft.Container):
    """Shows active model status and entry point for model setup."""

    def __init__(
        self,
        page: ft.Page,
        *,
        on_open_setup=None,
        on_close_setup=None,
    ):
        self._page = page
        self._on_open_setup = on_open_setup
        self._on_close_setup = on_close_setup
        self._setup_mode = False
        self._status_icon = ft.Icon(ft.Icons.MEMORY, size=22)
        self._status_text = ft.Text("", size=13, expand=True)
        self._setup_btn = ft.OutlinedButton(
            "Setup model",
            icon=ft.Icons.SETTINGS,
            on_click=self._handle_setup_click,
        )
        self._back_btn = ft.OutlinedButton(
            "Back to Assistant",
            icon=ft.Icons.ARROW_BACK,
            visible=False,
            on_click=self._handle_back_click,
        )
        super().__init__(
            content=ft.Row(
                [
                    self._status_icon,
                    self._status_text,
                    self._setup_btn,
                    self._back_btn,
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding(12, 10, 12, 10),
            border_radius=8,
            border=ft.border.all(1, ThemeHelper.border_default(page)),
        )
        self.refresh()

    @property
    def setup_mode(self) -> bool:
        return self._setup_mode

    @setup_mode.setter
    def setup_mode(self, value: bool) -> None:
        self._setup_mode = bool(value)
        self.refresh()

    def _handle_setup_click(self, _e) -> None:
        if self._on_open_setup:
            self._on_open_setup()
        else:
            navigate_to_model_setup()

    def _handle_back_click(self, _e) -> None:
        if self._on_close_setup:
            self._on_close_setup()

    def _maybe_probe_connection(self, *, force: bool = False) -> None:
        global _conn_probe_in_flight, _conn_ok, _conn_checked_at

        cfg = stock_config()
        if not cfg.lm_studio_enabled or self._setup_mode:
            return

        now = time.time()
        with _conn_lock:
            cache_fresh = (
                _conn_ok is not None
                and (now - _conn_checked_at) <= _PROBE_TTL_SEC
            )
            if _conn_probe_in_flight or (cache_fresh and not force):
                return
            _conn_probe_in_flight = True

        page = self._page

        def _work() -> None:
            ok = False
            try:
                ok, _msg = llm_manager().test_connection(timeout_sec=_PROBE_TIMEOUT_SEC)
            except Exception:
                ok = False

            def _ui() -> None:
                global _conn_probe_in_flight, _conn_ok, _conn_checked_at
                with _conn_lock:
                    _conn_ok = ok
                    _conn_checked_at = time.time()
                    _conn_probe_in_flight = False
                self.refresh()
                self.touch()

            schedule_ui_update(page, _ui, critical=True)

        threading.Thread(target=_work, daemon=True, name="model-setup-probe").start()

    def refresh(self, *, force_probe: bool = False) -> None:
        page = self._page
        cfg = stock_config()
        self._maybe_probe_connection(force=force_probe)

        status = resolve_model_status(
            lm_studio_enabled=cfg.lm_studio_enabled,
            setup_mode=self._setup_mode,
            connection_ok=backend_connection_ok() if cfg.lm_studio_enabled and not self._setup_mode else None,
            loaded_model=llm_manager().loaded_model,
            configured_model=cfg.llm_chat_model,
            backend_label=backend_display_name(cfg.llm_backend_type),
        )
        if status is None:
            self.visible = False
            return

        self.visible = True
        self._status_text.value = status.message
        self._status_text.color = ThemeHelper.text_primary(page)
        self._setup_btn.text = status.setup_btn_text

        if status.tone == "setup":
            self._status_icon.name = ft.Icons.MEMORY
            self._status_icon.color = ThemeHelper.accent_blue(page)
            self.bgcolor = ThemeHelper.surface_dim(page)
            self._setup_btn.visible = False
            self._back_btn.visible = bool(self._on_close_setup)
        elif status.tone == "info":
            self._status_icon.name = ft.Icons.SYNC
            self._status_icon.color = ThemeHelper.accent_blue(page)
            self.bgcolor = ThemeHelper.surface_dim(page)
            self._setup_btn.visible = True
            self._back_btn.visible = False
        elif status.tone == "error":
            self._status_icon.name = ft.Icons.CLOUD_OFF
            self._status_icon.color = ThemeHelper.accent_red(page)
            self.bgcolor = ThemeHelper.status_bg(page, "error")
            self._setup_btn.visible = True
            self._back_btn.visible = False
        elif status.tone == "success":
            self._status_icon.name = ft.Icons.CHECK_CIRCLE
            self._status_icon.color = ThemeHelper.accent_green(page)
            self.bgcolor = ThemeHelper.status_bg(page, "success")
            self._setup_btn.visible = True
            self._back_btn.visible = False
        else:
            self._status_icon.name = ft.Icons.WARNING_AMBER
            self._status_icon.color = ThemeHelper.accent_yellow(page)
            self.bgcolor = ThemeHelper.status_bg(page, "warning")
            self._setup_btn.visible = True
            self._back_btn.visible = False

    def touch(self) -> None:
        try:
            self.update()
        except RuntimeError:
            pass
