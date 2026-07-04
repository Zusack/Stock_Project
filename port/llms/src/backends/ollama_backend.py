# src/backends/ollama_backend.py
"""
Ollama backend implementation.
Uses the `ollama` Python package to communicate with a running Ollama service.
Install: pip install ollama
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Callable, Iterator, Optional

from src.backends.base import LLMBackend
from src.backends.types import (
    BackendType,
    CancellableStream,
    LLMHandle,
    ModelInfo,
    StreamChunk,
    ToolResult,
)

# Deferred import -- ollama may not be installed
_ollama = None


def _ensure_ollama():
    global _ollama
    if _ollama is None:
        try:
            import ollama as _ol
            _ollama = _ol
        except ImportError:
            raise ImportError(
                "The 'ollama' package is not installed. "
                "Install it with: pip install ollama"
            )
    return _ollama


class OllamaBackend(LLMBackend):
    """LLM backend using a running Ollama service."""

    def __init__(self, host: str = "http://localhost:11434"):
        self._host = host
        self._client = None

    # -- Identity --

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA

    # -- Connection lifecycle --

    def connect(self) -> None:
        ol = _ensure_ollama()
        try:
            self._client = ol.Client(host=self._host)
            self._client.list()
        except Exception as e:
            self._client = None
            if self.is_unreachable_error(e):
                raise RuntimeError(self.unreachable_message) from e
            raise

    def disconnect(self) -> None:
        self._client = None

    def is_available(self) -> bool:
        try:
            ol = _ensure_ollama()
            client = ol.Client(host=self._host)
            client.list()
            return True
        except Exception:
            return False

    # -- Model management --

    def list_available_models(self) -> list[ModelInfo]:
        if self._client is None:
            raise RuntimeError("Ollama not connected. Call connect() first.")
        try:
            result = self._client.list()

            # Extract the models list from the response (dict or Pydantic object).
            # Explicit parentheses avoid operator-precedence surprises.
            if isinstance(result, dict):
                models_data = result.get("models", [])
            else:
                models_data = getattr(result, "models", None)
                if models_data is None:
                    models_data = []

            models = []
            for m in models_data:
                if isinstance(m, dict):
                    name = m.get("name", "") or m.get("model", "")
                    size = m.get("size", 0)
                    details = m.get("details", {}) or {}
                else:
                    name = getattr(m, "model", "") or getattr(m, "name", "")
                    size = getattr(m, "size", 0)
                    details = getattr(m, "details", {}) or {}
                    if hasattr(details, "__dict__"):
                        details = details.__dict__

                # Skip entries with no valid identifier (defensive guard)
                if not name:
                    continue

                params = details.get("parameter_size", "") if isinstance(details, dict) else ""
                family = details.get("family", "") if isinstance(details, dict) else ""
                quant = details.get("quantization_level", "") if isinstance(details, dict) else ""
                fmt = quant if quant else "unknown"

                models.append(ModelInfo(
                    identifier=name,
                    backend_type=BackendType.OLLAMA,
                    display_name=name.split(":")[0] if ":" in name else name,
                    model_key=name,
                    format=fmt,
                    size_bytes=size,
                    vision=self._is_vision_model(name, family),
                    trained_for_tool_use=self._is_tool_model(name, family),
                    max_context_length=0,
                    params_string=params,
                    architecture=family,
                    model_path="",
                ))
            return models
        except Exception as e:
            if self.is_unreachable_error(e):
                raise RuntimeError(self.unreachable_message) from e
            raise

    @staticmethod
    def _is_vision_model(name: str, family: str) -> bool:
        vision_indicators = ["llava", "vision", "bakllava", "moondream", "minicpm-v", "gemma3"]
        combined = f"{name} {family}".lower()
        return any(v in combined for v in vision_indicators)

    @staticmethod
    def _is_tool_model(name: str, family: str) -> bool:
        tool_indicators = ["llama3.1", "llama3.2", "llama3.3", "command-r", "mistral-nemo",
                           "qwen2.5", "qwen3", "firefunction", "hermes"]
        combined = f"{name} {family}".lower()
        return any(t in combined for t in tool_indicators)

    def list_loaded_models(self) -> list[str]:
        if self._client is None:
            return []
        try:
            result = self._client.ps()
            running = getattr(result, "models", None) or (result.get("models", []) if isinstance(result, dict) else [])
            identifiers = []
            for m in running:
                name = m.get("name", "") if isinstance(m, dict) else getattr(m, "name", "")
                if name:
                    identifiers.append(name)
            return identifiers
        except Exception:
            return []

    def load_model(self, identifier: str, config: dict) -> LLMHandle:
        if self._client is None:
            raise RuntimeError("Ollama not connected. Call connect() first.")
        ctx = config.get("context_length", 8192)
        return LLMHandle(
            native={"model": identifier, "num_ctx": ctx},
            identifier=identifier,
            backend_type=BackendType.OLLAMA,
            config=config,
        )

    def load_model_alongside(self, identifier: str, config: dict) -> LLMHandle:
        return self.load_model(identifier, config)

    def unload_model(self, handle: LLMHandle, stream: Optional[CancellableStream] = None) -> bool:
        if stream is not None:
            try:
                stream.cancel()
                stream.close()
            except Exception:
                pass
        if handle is None:
            return True
        try:
            if self._client is not None:
                self._client.generate(
                    model=handle.identifier,
                    prompt="",
                    keep_alive=0,
                )
            handle.native = None
            return True
        except Exception:
            handle.native = None
            return True

    def unload_all_models(self) -> tuple[bool, Optional[str]]:
        loaded = self.list_loaded_models()
        errors = []
        for name in loaded:
            try:
                if self._client:
                    self._client.generate(model=name, prompt="", keep_alive=0)
            except Exception as e:
                errors.append(f"{name}: {e}")
        return (len(errors) == 0, "; ".join(errors) if errors else None)

    # -- Generation --

    def complete_stream(
        self, handle: LLMHandle, prompt: str, config: dict
    ) -> CancellableStream:
        if self._client is None:
            raise RuntimeError("Ollama not connected.")
        model_name = handle.identifier
        num_ctx = handle.native.get("num_ctx", 8192) if isinstance(handle.native, dict) else 8192
        options = {
            "num_ctx": num_ctx,
            "temperature": config.get("temperature", 0.7),
        }
        max_tokens = config.get("max_tokens")
        if max_tokens:
            options["num_predict"] = max_tokens

        gen = self._client.generate(
            model=model_name,
            prompt=prompt,
            stream=True,
            options=options,
        )

        def _iter():
            for chunk in gen:
                resp = chunk if isinstance(chunk, dict) else (chunk.__dict__ if hasattr(chunk, "__dict__") else {"response": str(chunk)})
                content = resp.get("response", "")
                done = resp.get("done", False)
                stop_reason = "stop" if done and content == "" else None
                if content or stop_reason:
                    yield StreamChunk(content=content if content else None, stop_reason=stop_reason)

        return CancellableStream(iterator=_iter())

    def chat_stream(
        self,
        handle: LLMHandle,
        messages: list[dict],
        config: dict,
        images: Optional[list[Any]] = None,
    ) -> CancellableStream:
        if self._client is None:
            raise RuntimeError("Ollama not connected.")
        model_name = handle.identifier
        num_ctx = handle.native.get("num_ctx", 8192) if isinstance(handle.native, dict) else 8192
        options = {
            "num_ctx": num_ctx,
            "temperature": config.get("temperature", 0.7),
        }
        max_tokens = config.get("max_tokens")
        if max_tokens:
            options["num_predict"] = max_tokens

        ollama_messages = []
        for msg in messages:
            entry = {"role": msg["role"], "content": msg["content"]}
            if msg["role"] == "user" and images:
                entry["images"] = images
            ollama_messages.append(entry)

        gen = self._client.chat(
            model=model_name,
            messages=ollama_messages,
            stream=True,
            options=options,
        )

        def _iter():
            for chunk in gen:
                if isinstance(chunk, dict):
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    done = chunk.get("done", False)
                else:
                    msg = getattr(chunk, "message", None) or {}
                    if hasattr(msg, "content"):
                        content = msg.content or ""
                    elif isinstance(msg, dict):
                        content = msg.get("content", "")
                    else:
                        content = ""
                    done = getattr(chunk, "done", False)
                stop_reason = "stop" if done and not content else None
                if content or stop_reason:
                    yield StreamChunk(content=content if content else None, stop_reason=stop_reason)

        return CancellableStream(iterator=_iter())

    def act_with_tools(
        self,
        handle: LLMHandle,
        messages: list[dict],
        tools: list[Callable],
        config: dict,
        on_message: Optional[Callable] = None,
        on_prediction_fragment: Optional[Callable] = None,
    ) -> ToolResult:
        if self._client is None:
            raise RuntimeError("Ollama not connected.")
        ol = _ensure_ollama()

        model_name = handle.identifier
        num_ctx = handle.native.get("num_ctx", 8192) if isinstance(handle.native, dict) else 8192
        options = {
            "num_ctx": num_ctx,
            "temperature": config.get("temperature", 0.7),
        }

        tool_map = {}
        for fn in tools:
            tool_map[fn.__name__] = fn

        ollama_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        tool_log_parts = []
        max_rounds = 10

        for _round in range(max_rounds):
            response = self._client.chat(
                model=model_name,
                messages=ollama_messages,
                tools=tools,
                options=options,
            )

            if isinstance(response, dict):
                msg = response.get("message", {})
            else:
                msg = getattr(response, "message", {})
                if hasattr(msg, "__dict__"):
                    msg = {
                        "role": getattr(msg, "role", "assistant"),
                        "content": getattr(msg, "content", ""),
                        "tool_calls": getattr(msg, "tool_calls", None),
                    }

            assistant_content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", None)

            ollama_messages.append(msg)
            if on_message:
                on_message(msg)

            if not tool_calls:
                tool_log = "\n".join(tool_log_parts) if tool_log_parts else None
                return ToolResult(
                    response_text=assistant_content or "[No final answer from model]",
                    tool_call_log=tool_log,
                    tokens_generated=max(0, int(len((assistant_content or "").split()) * 1.3)),
                )

            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn_data = tc.get("function", {})
                    fn_name = fn_data.get("name", "")
                    fn_args = fn_data.get("arguments", {})
                else:
                    fn_obj = getattr(tc, "function", None)
                    fn_name = getattr(fn_obj, "name", "") if fn_obj else ""
                    fn_args = getattr(fn_obj, "arguments", {}) if fn_obj else {}

                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {}

                fn = tool_map.get(fn_name)
                if fn is None:
                    result_str = f"Error: Unknown tool '{fn_name}'"
                else:
                    try:
                        result_val = fn(**fn_args)
                        result_str = json.dumps(result_val) if not isinstance(result_val, str) else result_val
                    except Exception as e:
                        result_str = f"Error calling {fn_name}: {e}"

                tool_log_parts.append(f"[tool_call] {fn_name}({fn_args}) -> {result_str}")
                if on_message:
                    on_message({"role": "tool", "content": result_str})

                ollama_messages.append({"role": "tool", "content": result_str})

        return ToolResult(
            response_text="[Max tool rounds exceeded]",
            tool_call_log="\n".join(tool_log_parts) if tool_log_parts else None,
            tokens_generated=0,
        )

    # -- Utilities --

    def prepare_image(self, path: str) -> Any:
        if not os.path.exists(path):
            raise ValueError(f"Image file not found: {path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def tokenize(self, handle: LLMHandle, text: str) -> list[int]:
        word_count = len(text.split())
        estimated_tokens = int(word_count * 1.33)
        return list(range(estimated_tokens))

    # -- Capability queries --

    def supports_vision(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    # -- Error classification --

    def is_unreachable_error(self, error: Exception) -> bool:
        msg = str(error).lower() if error else ""
        triggers = [
            "connection refused", "failed to connect", "cannot connect",
            "connect error", "connection error", "unreachable",
            "econnrefused", "no connection", "refused",
        ]
        return any(t in msg for t in triggers)

    def is_load_blocked_error(self, error: Exception) -> bool:
        msg = str(error).lower() if error else ""
        triggers = [
            "out of memory", "oom", "not enough memory",
            "insufficient", "failed to load",
        ]
        return any(t in msg for t in triggers)

    @property
    def unreachable_message(self) -> str:
        return f"Ollama is not reachable at {self._host}. Please start Ollama and try again."

    # -- Ollama-specific management methods --

    def pull_model(self, name: str, stream: bool = True):
        """Pull/download a model. Yields progress dicts if stream=True."""
        if self._client is None:
            raise RuntimeError("Ollama not connected.")
        return self._client.pull(name, stream=stream)

    def delete_model(self, name: str) -> None:
        """Delete a model from Ollama."""
        if self._client is None:
            raise RuntimeError("Ollama not connected.")
        self._client.delete(name)

    def show_model(self, name: str) -> dict:
        """Show model details."""
        if self._client is None:
            raise RuntimeError("Ollama not connected.")
        result = self._client.show(name)
        return result if isinstance(result, dict) else getattr(result, "__dict__", {})
