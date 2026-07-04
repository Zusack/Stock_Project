# src/backends/lmstudio_backend.py
"""
LM Studio backend implementation.
Wraps the lmstudio Python SDK (v1.5.0+) behind the LLMBackend interface.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterator, Optional

import lmstudio as lms

from src.backends.base import LLMBackend
from src.backends.types import (
    BackendType,
    CancellableStream,
    LLMHandle,
    ModelInfo,
    StreamChunk,
    ToolResult,
)


class LMStudioBackend(LLMBackend):
    """LLM backend using the LM Studio desktop application and its Python SDK."""

    def __init__(self, base_url: str = ""):
        self._client: Optional[Any] = None
        self._base_url = (base_url or "").strip() or None

    # -- Identity --

    @property
    def backend_type(self) -> BackendType:
        return BackendType.LMSTUDIO

    def _client_kwargs(self) -> dict:
        """Kwargs for lms.Client(); use base_url if set (custom LM Studio server). Some SDK versions do not support base_url."""
        if self._base_url:
            return {"base_url": self._base_url}
        return {}

    def _create_client(self):
        """Create LM Studio client; fall back to no base_url if SDK does not support it."""
        kwargs = self._client_kwargs()
        if not kwargs:
            return lms.Client()
        try:
            return lms.Client(**kwargs)
        except TypeError:
            return lms.Client()

    # -- Connection lifecycle --

    def connect(self) -> None:
        lms.set_sync_api_timeout(None)
        try:
            self._client = self._create_client()
            self._client.__enter__()
        except Exception as e:
            self._client = None
            if self.is_unreachable_error(e):
                raise RuntimeError(self.unreachable_message) from e
            raise

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.__exit__(None, None, None)
            except Exception:
                pass
            self._client = None

    def is_available(self) -> bool:
        try:
            with self._create_client():
                return True
        except Exception:
            return False

    # -- Model management --

    def list_available_models(self) -> list[ModelInfo]:
        try:
            client = self._client or self._create_client().__enter__()
            own_client = self._client is None
            try:
                downloaded = client.llm.list_downloaded()
                if not downloaded:
                    return []
                models = []
                for model in downloaded:
                    info = getattr(model, "info", None)
                    if info:
                        models.append(ModelInfo(
                            identifier=model.model_key,
                            backend_type=BackendType.LMSTUDIO,
                            model_key=getattr(info, "model_key", model.model_key),
                            display_name=getattr(info, "display_name", "N/A"),
                            format=getattr(info, "format", "N/A"),
                            size_bytes=getattr(info, "size_bytes", 0),
                            vision=getattr(info, "vision", False),
                            trained_for_tool_use=getattr(info, "trained_for_tool_use", False),
                            max_context_length=getattr(info, "max_context_length", 0),
                            params_string=getattr(info, "params_string", "N/A"),
                            architecture=getattr(info, "architecture", "N/A"),
                            model_path=getattr(info, "path", "N/A"),
                        ))
                    else:
                        models.append(ModelInfo(
                            identifier=model.model_key,
                            backend_type=BackendType.LMSTUDIO,
                            params_string="N/A",
                        ))
                return models
            finally:
                if own_client:
                    client.__exit__(None, None, None)
        except Exception as e:
            if self.is_unreachable_error(e):
                raise RuntimeError(self.unreachable_message) from e
            raise

    def list_loaded_models(self) -> list[str]:
        try:
            loaded = lms.list_loaded_models("llm")
        except AttributeError:
            try:
                client = self._client or self._create_client().__enter__()
                own_client = self._client is None
                try:
                    loaded = client.llm.list_loaded()
                finally:
                    if own_client:
                        client.__exit__(None, None, None)
            except Exception:
                return self._fallback_list_loaded()
        except Exception:
            return self._fallback_list_loaded()

        identifiers = []
        for model in (loaded or []):
            key = (
                getattr(model, "identifier", None)
                or getattr(model, "model_key", None)
                or getattr(model, "key", None)
            )
            if key:
                identifiers.append(str(key))
        return identifiers

    def _fallback_list_loaded(self) -> list[str]:
        try:
            model = lms.llm()
            if model is not None:
                key = (
                    getattr(model, "identifier", None)
                    or getattr(model, "model_key", None)
                    or getattr(model, "key", None)
                )
                if key:
                    return [str(key)]
                return ["(model loaded)"]
        except Exception:
            pass
        return []

    def load_model(self, identifier: str, config: dict) -> LLMHandle:
        load_config = {
            "gpu_layers": config.get("gpu_layers", -1),
            "context_length": config.get("context_length", 8192),
        }
        try:
            native = lms.llm(identifier, config=load_config)
        except Exception as e:
            if self.is_unreachable_error(e):
                raise RuntimeError(self.unreachable_message) from e
            raise
        return LLMHandle(
            native=native,
            identifier=identifier,
            backend_type=BackendType.LMSTUDIO,
            config=config,
        )

    def load_model_alongside(self, identifier: str, config: dict) -> LLMHandle:
        load_config = {"context_length": config.get("context_length", 8192)}
        client = self._client
        if client is None:
            raise RuntimeError("LM Studio client not connected. Call connect() first.")
        try:
            native = client.llm.load_new_instance(identifier, None, config=load_config)
        except Exception as e:
            if self.is_unreachable_error(e):
                raise RuntimeError(self.unreachable_message) from e
            raise
        return LLMHandle(
            native=native,
            identifier=identifier,
            backend_type=BackendType.LMSTUDIO,
            config=config,
        )

    def supports_load_alongside(self) -> bool:
        return True

    def unload_model(self, handle: LLMHandle, stream: Optional[CancellableStream] = None) -> bool:
        if handle is None or handle.native is None:
            return True

        if stream is not None:
            try:
                stream.cancel()
                stream.close()
                time.sleep(0.8)
            except Exception:
                pass

        time.sleep(0.2)

        try:
            handle.native.unload()
            time.sleep(0.15)
            handle.native = None
            return True
        except Exception as e:
            error_msg = str(e).lower()
            benign = ["model unloaded", "no model loaded", "model not found",
                       "not loaded", "channel", "already closed"]
            if any(phrase in error_msg for phrase in benign):
                handle.native = None
                return True
            return False

    def unload_all_models(self) -> tuple[bool, Optional[str]]:
        try:
            loaded = lms.list_loaded_models("llm")
        except AttributeError:
            try:
                client = self._client or self._create_client().__enter__()
                own_client = self._client is None
                try:
                    loaded = client.llm.list_loaded()
                finally:
                    if own_client:
                        client.__exit__(None, None, None)
            except Exception as e:
                return self._fallback_unload_all(e)
        except Exception as e:
            return self._fallback_unload_all(e)

        errors = []
        for model in (loaded or []):
            try:
                if hasattr(model, "unload"):
                    model.unload()
                    time.sleep(0.2)
            except Exception as e:
                errors.append(str(e))
        return (len(errors) == 0, "; ".join(errors) if errors else None)

    def _fallback_unload_all(self, prior_error: Exception) -> tuple[bool, Optional[str]]:
        max_attempts = 5
        for _ in range(max_attempts):
            try:
                model = lms.llm()
                if model is None:
                    return (True, None)
                if hasattr(model, "unload"):
                    model.unload()
                    time.sleep(0.3)
            except Exception as e:
                err_str = str(e).lower()
                if "no model" in err_str or "not loaded" in err_str or "model not found" in err_str:
                    return (True, None)
                return (False, str(prior_error))
        return (True, None)

    # -- Generation --

    def complete_stream(
        self, handle: LLMHandle, prompt: str, config: dict
    ) -> CancellableStream:
        native_stream = handle.native.complete_stream(prompt, config=config)

        def _iter():
            for chunk in native_stream:
                yield StreamChunk(
                    content=getattr(chunk, "content", None),
                    stop_reason=getattr(chunk, "stop_reason", None),
                )

        return CancellableStream(
            iterator=_iter(),
            cancel_fn=getattr(native_stream, "cancel", None),
            close_fn=getattr(native_stream, "close", None),
        )

    def chat_stream(
        self,
        handle: LLMHandle,
        messages: list[dict],
        config: dict,
        images: Optional[list[Any]] = None,
    ) -> CancellableStream:
        chat = lms.Chat()
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                if images:
                    chat.add_user_message(content, images=images)
                else:
                    chat.add_user_message(content)
            elif role == "system":
                chat.add_user_message(content)
            elif role == "assistant":
                chat.add_user_message(content)

        native_stream = handle.native.respond_stream(chat, config=config)

        def _iter():
            for chunk in native_stream:
                yield StreamChunk(
                    content=getattr(chunk, "content", None),
                    stop_reason=getattr(chunk, "stop_reason", None),
                )

        return CancellableStream(
            iterator=_iter(),
            cancel_fn=getattr(native_stream, "cancel", None),
            close_fn=getattr(native_stream, "close", None),
        )

    def act_with_tools(
        self,
        handle: LLMHandle,
        messages: list[dict],
        tools: list[Callable],
        config: dict,
        on_message: Optional[Callable] = None,
        on_prediction_fragment: Optional[Callable] = None,
    ) -> ToolResult:
        chat = lms.Chat()
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                chat.add_user_message(content)

        collected_messages = []

        def _on_msg(m):
            collected_messages.append(m)
            if on_message:
                on_message(m)

        act_kwargs = {
            "config": config,
            "on_message": _on_msg,
        }
        if on_prediction_fragment:
            act_kwargs["on_prediction_fragment"] = on_prediction_fragment

        handle.native.act(chat, tools, **act_kwargs)

        full_response = ""
        for m in reversed(collected_messages):
            role = getattr(m, "role", None) or ""
            text = self._extract_text_from_message(m)
            if text:
                if role == "assistant":
                    full_response = text
                    break
                elif not full_response and role == "tool":
                    full_response = text

        if not full_response:
            full_response = "[No final answer from model]"

        tool_log = self._build_tool_log(collected_messages)
        tokens_gen = max(0, int(len(full_response.split()) * 1.3))

        return ToolResult(
            response_text=full_response,
            tool_call_log=tool_log,
            tokens_generated=tokens_gen,
        )

    @staticmethod
    def _extract_text_from_message(m) -> Optional[str]:
        content = getattr(m, "content", None)
        if content is None:
            return getattr(m, "text", None) or None
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, (list, tuple)):
            parts = []
            for item in content:
                if hasattr(item, "text") and getattr(item, "text", None):
                    parts.append(str(item.text).strip())
                elif hasattr(item, "content") and getattr(item, "content", None):
                    parts.append(str(item.content).strip())
                elif isinstance(item, dict):
                    parts.append((item.get("text") or item.get("content") or "").strip())
            return " ".join(p for p in parts if p) or None
        return None

    @staticmethod
    def _build_tool_log(msgs) -> Optional[str]:
        parts = []
        for m in msgs:
            role = getattr(m, "role", None) or getattr(m, "type", None) or ""
            text = LMStudioBackend._extract_text_from_message(m)
            if text:
                prefix = f"[{role}] " if role else ""
                parts.append(f"{prefix}{text}")
        return "\n".join(parts) if parts else None

    # -- Utilities --

    def prepare_image(self, path: str) -> Any:
        return lms.prepare_image(path)

    def tokenize(self, handle: LLMHandle, text: str) -> list[int]:
        return handle.native.tokenize(text)

    # -- Capability queries --

    def supports_vision(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    # -- Error classification --

    def is_unreachable_error(self, error: Exception) -> bool:
        msg = str(error).lower() if error else ""
        triggers = [
            "connection refused", "econnrefused", "failed to connect",
            "cannot connect", "not reachable", "connectionrefusederror",
            "connection error", "lm studio is not reachable", "no connection",
            "connect econnrefused", "actively refused", "connection reset",
            "network is unreachable",
        ]
        return any(t in msg for t in triggers)

    def is_load_blocked_error(self, error: Exception) -> bool:
        msg = str(error).lower() if error else ""
        triggers = [
            "failed to load model", "error loading model",
            "model has unloaded or crashed", "model already loaded",
            "cannot load", "operation canceled",
        ]
        return any(t in msg for t in triggers)

    @property
    def unreachable_message(self) -> str:
        return "LM Studio is not reachable. Please start LM Studio and try again."

    @property
    def model_already_loaded_message(self) -> str:
        return (
            "A model is already loaded in LM Studio. Unload it before loading another, or use "
            "Settings > LLM Backend > 'Unload all models before process' to auto-unload."
        )
