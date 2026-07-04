# src/backends/vllm_backend.py
"""
vLLM backend implementation.
Communicates with a running vLLM server via its OpenAI-compatible API.
Also works with any OpenAI-compatible server (Aphrodite, TGI, etc.).
Install: pip install openai (for the client)

Backend is selected per-model (Model Manager); no global "active" vLLM setting for runs.
Connection and model listing are defensive: clear errors for unreachable server,
timeouts, auth, and malformed /v1/models responses.
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

_openai = None


def _ensure_openai():
    global _openai
    if _openai is None:
        try:
            import openai as _oa
            _openai = _oa
        except ImportError:
            raise ImportError(
                "The 'openai' package is not installed. "
                "Install it with: pip install openai"
            )
    return _openai


def _safe_model_id(m: Any) -> Optional[str]:
    """Get model id from API response object or dict."""
    if m is None:
        return None
    if isinstance(m, dict):
        return m.get("id") or m.get("model")
    return getattr(m, "id", None) or getattr(m, "model", None)


class VLLMBackend(LLMBackend):
    """LLM backend using a vLLM (or any OpenAI-compatible) server."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = ""):
        self._base_url = (base_url or "http://localhost:8000").strip().rstrip("/")
        # Use None when empty so servers that don't require auth accept the client
        self._api_key = api_key.strip() if api_key else None
        if self._api_key == "":
            self._api_key = None
        self._client = None
        self._available_models: list[str] = []

    # -- Identity --

    @property
    def backend_type(self) -> BackendType:
        return BackendType.VLLM

    # -- Connection lifecycle --

    def connect(self) -> None:
        oa = _ensure_openai()
        # OpenAI client requires api_key to be set; use placeholder when no key so vLLM (no-auth) works
        api_key = self._api_key if self._api_key else "dummy"
        self._client = oa.OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key=api_key,
        )
        try:
            models_resp = self._client.models.list()
            data = getattr(models_resp, "data", None)
            if data is None or not isinstance(data, (list, tuple)):
                self._available_models = []
                return
            self._available_models = []
            for m in data:
                mid = _safe_model_id(m)
                if mid:
                    self._available_models.append(mid)
        except Exception as e:
            self._client = None
            if self.is_unreachable_error(e):
                raise RuntimeError(self.unreachable_message) from e
            raise RuntimeError(
                f"Could not list models from vLLM at {self._base_url}. "
                f"Ensure the server is running and exposes /v1/models. Error: {e}"
            ) from e

    def disconnect(self) -> None:
        self._client = None
        self._available_models = []

    def is_available(self) -> bool:
        try:
            oa = _ensure_openai()
            api_key = self._api_key if self._api_key else "dummy"
            client = oa.OpenAI(
                base_url=f"{self._base_url}/v1",
                api_key=api_key,
            )
            client.models.list()
            return True
        except Exception:
            return False

    # -- Model management --

    def list_available_models(self) -> list[ModelInfo]:
        if self._client is None:
            raise RuntimeError("vLLM not connected. Call connect() first.")
        try:
            models_resp = self._client.models.list()
            data = getattr(models_resp, "data", None)
            if data is None or not isinstance(data, (list, tuple)):
                return []
            result = []
            for m in data:
                mid = _safe_model_id(m)
                if not mid:
                    continue
                display = mid.split("/")[-1] if "/" in mid else mid
                result.append(ModelInfo(
                    identifier=mid,
                    backend_type=BackendType.VLLM,
                    display_name=display,
                    model_key=mid,
                    format="vLLM",
                    size_bytes=0,
                    vision=False,
                    trained_for_tool_use=False,
                    max_context_length=0,
                    params_string="",
                    architecture="",
                    model_path="",
                ))
            return result
        except Exception as e:
            if self.is_unreachable_error(e):
                raise RuntimeError(self.unreachable_message) from e
            raise RuntimeError(
                f"Error listing vLLM models at {self._base_url}: {e}"
            ) from e

    def list_loaded_models(self) -> list[str]:
        if self._client is None:
            return []
        try:
            models_resp = self._client.models.list()
            data = getattr(models_resp, "data", None)
            if data is None or not isinstance(data, (list, tuple)):
                return []
            return [mid for m in data if (mid := _safe_model_id(m))]
        except Exception:
            return []

    def load_model(self, identifier: str, config: dict) -> LLMHandle:
        if self._client is None:
            raise RuntimeError("vLLM not connected. Call connect() first.")
        return LLMHandle(
            native=self._client,
            identifier=identifier,
            backend_type=BackendType.VLLM,
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
        if handle is not None:
            handle.native = None
        return True

    def unload_all_models(self) -> tuple[bool, Optional[str]]:
        return (True, None)

    # -- Generation --

    def complete_stream(
        self, handle: LLMHandle, prompt: str, config: dict
    ) -> CancellableStream:
        client = handle.native
        if client is None:
            raise RuntimeError("vLLM client not available.")

        stream = client.completions.create(
            model=handle.identifier,
            prompt=prompt,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4096),
            stream=True,
        )

        def _iter():
            try:
                for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        text = choice.text or ""
                        finish = choice.finish_reason
                        if text or finish:
                            yield StreamChunk(
                                content=text if text else None,
                                stop_reason=finish,
                            )
            except GeneratorExit:
                pass

        def _cancel():
            try:
                stream.close()
            except Exception:
                pass

        return CancellableStream(iterator=_iter(), cancel_fn=_cancel, close_fn=_cancel)

    def chat_stream(
        self,
        handle: LLMHandle,
        messages: list[dict],
        config: dict,
        images: Optional[list[Any]] = None,
    ) -> CancellableStream:
        client = handle.native
        if client is None:
            raise RuntimeError("vLLM client not available.")

        api_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user" and images:
                content_parts = [{"type": "text", "text": content}]
                for img_data in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    })
                api_messages.append({"role": role, "content": content_parts})
            else:
                api_messages.append({"role": role, "content": content})

        stream = client.chat.completions.create(
            model=handle.identifier,
            messages=api_messages,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4096),
            stream=True,
        )

        def _iter():
            try:
                for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        delta = choice.delta
                        content = getattr(delta, "content", None) or ""
                        finish = choice.finish_reason
                        if content or finish:
                            yield StreamChunk(
                                content=content if content else None,
                                stop_reason=finish,
                            )
            except GeneratorExit:
                pass

        def _cancel():
            try:
                stream.close()
            except Exception:
                pass

        return CancellableStream(iterator=_iter(), cancel_fn=_cancel, close_fn=_cancel)

    def act_with_tools(
        self,
        handle: LLMHandle,
        messages: list[dict],
        tools: list[Callable],
        config: dict,
        on_message: Optional[Callable] = None,
        on_prediction_fragment: Optional[Callable] = None,
    ) -> ToolResult:
        client = handle.native
        if client is None:
            raise RuntimeError("vLLM client not available.")

        import inspect

        tool_schemas = []
        tool_map = {}
        for fn in tools:
            tool_map[fn.__name__] = fn
            sig = inspect.signature(fn)
            params = {}
            for pname, param in sig.parameters.items():
                ptype = "string"
                if param.annotation == int:
                    ptype = "integer"
                elif param.annotation == float:
                    ptype = "number"
                elif param.annotation == bool:
                    ptype = "boolean"
                elif param.annotation == list:
                    ptype = "array"
                params[pname] = {"type": ptype}
            required = [
                pname for pname, param in sig.parameters.items()
                if param.default is inspect.Parameter.empty
            ]
            tool_schemas.append({
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": fn.__doc__ or "",
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required,
                    },
                },
            })

        api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        tool_log_parts = []
        max_rounds = 10

        for _round in range(max_rounds):
            response = client.chat.completions.create(
                model=handle.identifier,
                messages=api_messages,
                tools=tool_schemas,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 4096),
            )

            choice = response.choices[0] if response.choices else None
            if choice is None:
                break

            message = choice.message
            try:
                msg_dict = message.model_dump() if hasattr(message, "model_dump") else dict(message)
            except Exception:
                msg_dict = {"role": getattr(message, "role", "assistant"), "content": getattr(message, "content", "")}
            api_messages.append(msg_dict)
            if on_message:
                on_message(message)

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                text = message.content or "[No final answer from model]"
                return ToolResult(
                    response_text=text,
                    tool_call_log="\n".join(tool_log_parts) if tool_log_parts else None,
                    tokens_generated=max(0, int(len(text.split()) * 1.3)),
                )

            for tc in tool_calls:
                fn_obj = getattr(tc, "function", tc)
                fn_name = getattr(fn_obj, "name", None) or (fn_obj.get("name") if isinstance(fn_obj, dict) else None)
                if not fn_name:
                    continue
                args_raw = getattr(fn_obj, "arguments", None) or (fn_obj.get("arguments") if isinstance(fn_obj, dict) else "{}")
                try:
                    fn_args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}
                tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "")

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
                    on_message({"role": "tool", "content": result_str, "tool_call_id": tc_id})
                api_messages.append({
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tc_id,
                })

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
        return list(range(int(word_count * 1.33)))

    # -- Capability queries --

    def supports_vision(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    # -- Error classification --

    def is_unreachable_error(self, error: Exception) -> bool:
        msg = str(error).lower() if error else ""
        triggers = [
            "connection refused", "failed to connect", "connection error",
            "unreachable", "econnrefused", "cannot connect", "connection reset",
            "timeout", "timed out", "no route to host", "name or service not known",
            "nodename nor servname provided", "ssl:", "certificate", "tls",
            "502 bad gateway", "503 service unavailable", "504 gateway time",
        ]
        return any(t in msg for t in triggers)

    def is_load_blocked_error(self, error: Exception) -> bool:
        msg = str(error).lower() if error else ""
        return "out of memory" in msg or "oom" in msg or "cuda out of memory" in msg

    @property
    def unreachable_message(self) -> str:
        return (
            f"vLLM server is not reachable at {self._base_url}. "
            "Ensure the server is running (e.g. vllm serve <model>), the URL is correct in Settings, and no firewall blocks the connection."
        )

    @property
    def model_already_loaded_message(self) -> str:
        return (
            f"vLLM at {self._base_url} may already be serving a different model or is busy. "
            "Start another server instance, change the model with vllm serve, or retry later."
        )
