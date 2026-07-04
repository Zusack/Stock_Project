"""LLM manager: backend connection, model load, streaming chat."""

from __future__ import annotations

import threading
from typing import Callable, Optional

from src.llm.backends import BackendFactory, LLMHandle, ModelInfo
from src.llm.backends.base import LLMBackend
from src.llm.connection_probe import probe_backend
from src.llm.conversation import trim_messages
from src.llm.utils.stream_utils import iterate_stream
from src.services.event_bus import event_bus
from src.services.stock_config import stock_config


class LLMManager:
    """Singleton managing LLM backend lifecycle and streaming."""

    _instance: Optional["LLMManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "LLMManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._backend: LLMBackend | None = None
        self._handle: LLMHandle | None = None
        self._backend_type: str = ""
        self._model_id: str = ""
        self._connect_lock = threading.Lock()

    def _cfg(self):
        return stock_config()

    def is_enabled(self) -> bool:
        return self._cfg().lm_studio_enabled

    def backend_type(self) -> str:
        return self._cfg().llm_backend_type

    def inference_config(self) -> dict:
        cfg = self._cfg()
        return {
            "temperature": cfg.llm_temperature,
            "max_tokens": cfg.llm_max_tokens,
        }

    def _release_backend(self) -> None:
        """Tear down active backend resources. Caller must hold ``_connect_lock``."""
        if self._handle and self._backend:
            try:
                self._backend.unload_model(self._handle)
            except Exception:
                pass
        self._handle = None
        if self._backend:
            try:
                self._backend.disconnect()
            except Exception:
                pass
        self._backend = None
        self._model_id = ""

    def reset(self) -> None:
        with self._connect_lock:
            self._release_backend()
            self._backend_type = ""

    def _log(self, message: str, *, level: str = "INFO", **kwargs) -> None:
        try:
            from src.utils.logger_utils import app_logger

            app_logger.log("LLM", message, level=level, **kwargs)
        except Exception:
            pass

    def _apply_sync_timeout(self) -> None:
        """Apply per-request timeout to LM Studio SDK sync calls."""
        try:
            import lmstudio as lms

            lms.set_sync_api_timeout(self._cfg().lm_studio_timeout_sec)
        except Exception:
            pass

    def _ensure_backend(self) -> LLMBackend:
        """Return a connected backend. Caller must hold ``_connect_lock``."""
        cfg = self._cfg()
        bt = cfg.llm_backend_type
        if self._backend is not None and self._backend_type == bt:
            return self._backend
        self._release_backend()
        self._apply_sync_timeout()
        self._log(
            f"Connecting backend {bt}.",
            backend=bt,
            base_url=cfg.backend_settings().get("lmstudio_base_url", ""),
        )
        backend = BackendFactory.create_from_string(bt, cfg.backend_settings())
        backend.connect()
        self._backend = backend
        self._backend_type = bt
        self._log(f"Backend {bt} connected.", backend=bt, level="DEBUG")
        return backend

    def connect(self) -> LLMBackend:
        with self._connect_lock:
            return self._ensure_backend()

    def list_models(self) -> tuple[list[ModelInfo], str | None]:
        try:
            backend = self.connect()
            return backend.list_available_models(), None
        except Exception as ex:
            return [], str(ex)

    def test_connection(self, *, timeout_sec: float | None = None) -> tuple[bool, str]:
        """HTTP probe for backend reachability and model listing (no SDK connect)."""
        cfg = self._cfg()
        timeout = max(3.0, float(timeout_sec if timeout_sec is not None else cfg.lm_studio_timeout_sec))
        return probe_backend(
            cfg.llm_backend_type,
            cfg.backend_settings(),
            timeout_sec=timeout,
        )

    def load_model(self, identifier: str | None = None) -> LLMHandle:
        cfg = self._cfg()
        model_id = (identifier or cfg.llm_chat_model or "").strip()
        if not model_id:
            raise ValueError("No model specified. Set llm_chat_model in Settings.")
        with self._connect_lock:
            backend = self._ensure_backend()
            if self._handle and self._model_id == model_id:
                return self._handle
            if self._handle:
                try:
                    backend.unload_model(self._handle)
                except Exception:
                    pass
            load_config = {
                "gpu_layers": -1,
                "context_length": cfg.llm_context_length,
            }
            self._log(f"Loading model {model_id}.", model=model_id, backend=cfg.llm_backend_type)
            self._handle = backend.load_model(model_id, load_config)
            self._model_id = model_id
            self._log(f"Model loaded: {model_id}.", model=model_id, level="DEBUG")
            return self._handle

    def unload_model(self) -> None:
        with self._connect_lock:
            if self._handle and self._backend:
                try:
                    self._backend.unload_model(self._handle)
                except Exception:
                    pass
            self._handle = None
            self._model_id = ""

    def chat_stream(
        self,
        messages: list[dict],
        *,
        cancel_event: threading.Event | None = None,
        on_token: Callable[[str], None] | None = None,
        model: str | None = None,
    ) -> str:
        """Stream a chat completion; return full response text."""
        cfg = self._cfg()
        self._log(
            "Chat stream started.",
            model=model or cfg.llm_chat_model,
            backend=cfg.llm_backend_type,
            message_count=len(messages),
        )
        try:
            backend = self.connect()
            handle = self.load_model(model)
            trimmed = trim_messages(messages, max_chars=cfg.llm_context_length * 3)
            stream = backend.chat_stream(handle, trimmed, self.inference_config())
            full = ""
            job_timeout = max(30.0, float(cfg.lm_studio_timeout_sec))
            for chunk in iterate_stream(
                stream,
                cancel_event=cancel_event,
                job_timeout_sec=job_timeout,
            ):
                token = chunk.content or ""
                if token:
                    full += token
                    if on_token:
                        on_token(token)
                    event_bus.emit("llm_token", token=token)
            if not full.strip():
                self._log("Chat stream finished with empty response.", level="WARN")
            else:
                self._log(
                    "Chat stream finished.",
                    level="DEBUG",
                    chars=len(full),
                )
            return full
        except Exception as ex:
            self._log(f"Chat stream failed: {ex}", level="ERROR", error=str(ex))
            raise

    def ensure_model_loaded(self, model: str | None = None) -> tuple[bool, str]:
        """Load configured model if needed. Returns (ok, model_id or error message)."""
        cfg = self._cfg()
        if not cfg.lm_studio_enabled:
            return False, "Local AI is disabled in Settings."
        model_id = (model or cfg.llm_chat_model or "").strip()
        if not model_id:
            return (
                False,
                "No model configured. Open Dashboard → Setup model to load one.",
            )
        try:
            with self._connect_lock:
                if self._handle and self._model_id == model_id:
                    return True, model_id
            self.load_model(model_id)
            return True, model_id
        except Exception as ex:
            self._log(f"ensure_model_loaded failed: {ex}", level="ERROR", model=model_id)
            return False, str(ex)

    def complete(
        self,
        prompt: str,
        *,
        cancel_event: threading.Event | None = None,
        on_token: Callable[[str], None] | None = None,
        model: str | None = None,
    ) -> str:
        """Non-chat completion for simple prompts (headline analysis)."""
        backend = self.connect()
        handle = self.load_model(model)
        stream = backend.complete_stream(handle, prompt, self.inference_config())
        full = ""
        job_timeout = max(30.0, float(self._cfg().lm_studio_timeout_sec))
        for chunk in iterate_stream(
            stream,
            cancel_event=cancel_event,
            job_timeout_sec=job_timeout,
        ):
            token = chunk.content or ""
            if token:
                full += token
                if on_token:
                    on_token(token)
        return full

    @property
    def loaded_model(self) -> str:
        return self._model_id

    @property
    def handle(self) -> LLMHandle | None:
        return self._handle

    @property
    def backend(self) -> LLMBackend | None:
        return self._backend


def llm_manager() -> LLMManager:
    return LLMManager()
