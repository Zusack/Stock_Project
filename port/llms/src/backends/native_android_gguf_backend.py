"""
GGUF inference via NDK-built libllama_bench_native (llama.cpp), loaded with ctypes.

Desktop continues to use GGUFBackend + gguf_worker + llama-cpp-python.
"""
from __future__ import annotations

import base64
import os
import queue
import threading
from typing import Any, Callable, Iterator, Optional

from src.backends.base import LLMBackend
from src.backends.gguf_metadata import format_parameter_count, read_gguf_metadata
from src.backends.native_bridge import (
    lb_complete_stream as nb_complete_stream,
    lb_init,
    lb_last_error,
    lb_load_model,
    lb_shutdown,
    lb_unload_model,
    load_native_library,
)
from src.backends.types import (
    BackendType,
    CancellableStream,
    LLMHandle,
    ModelInfo,
    StreamChunk,
    ToolResult,
)
from shared_core.config import get_active_profile
from shared_core.mobile_capabilities import inspect_device_capabilities


class NativeAndroidGgufBackend(LLMBackend):
    """llama.cpp via libllama_bench_native.so (no llama-cpp-python)."""

    def __init__(
        self,
        model_dir: str = "",
        n_gpu_layers: int = 0,
        main_gpu: int = 0,
        include_subfolders: bool = True,
    ):
        self._model_dir = model_dir or ""
        self._n_gpu_layers = int(n_gpu_layers)
        self._main_gpu = int(main_gpu)
        self._include_subfolders = include_subfolders
        self._connected = False
        self._loaded_identifier: Optional[str] = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.GGUF

    def connect(self) -> None:
        cap = inspect_device_capabilities()
        if not cap.meets_minimum_ram:
            raise RuntimeError(
                "Android native GGUF runtime requires at least 4 GB RAM for stable local inference."
            )
        load_native_library()
        rc = lb_init()
        if rc != 0:
            raise RuntimeError(f"lb_init failed: {lb_last_error()}")
        self._connected = True

    def disconnect(self) -> None:
        lb_shutdown()
        self._connected = False
        self._loaded_identifier = None

    def is_available(self) -> bool:
        try:
            load_native_library()
            return True
        except OSError:
            return False

    def list_available_models(self) -> list[ModelInfo]:
        if not self._model_dir or not os.path.isdir(self._model_dir):
            return []
        models: list[ModelInfo] = []
        if self._include_subfolders:
            candidates = (
                os.path.join(root, f) for root, _dirs, files in os.walk(self._model_dir) for f in files
            )
        else:
            candidates = (
                os.path.join(self._model_dir, f)
                for f in os.listdir(self._model_dir)
                if os.path.isfile(os.path.join(self._model_dir, f))
            )
        for full_path in candidates:
            if not full_path.lower().endswith(".gguf"):
                continue
            fname = os.path.basename(full_path)
            if "mmproj" in fname.lower():
                continue
            rel_path = os.path.relpath(full_path, self._model_dir)
            size = os.path.getsize(full_path)
            display = fname.replace(".gguf", "").replace("-", " ").replace("_", " ")
            meta = read_gguf_metadata(full_path)
            max_ctx = 0
            params_str = ""
            arch = ""
            vision = False
            if meta:
                max_ctx = meta.context_length or 0
                params_str = format_parameter_count(meta.parameter_count) if meta.parameter_count else ""
                arch = meta.architecture or ""
                vision = meta.vision
            if not vision:
                base = full_path[:-5]
                for ext in (".mmproj", "-mmproj.gguf"):
                    if os.path.isfile(base + ext):
                        vision = True
                        break
            models.append(
                ModelInfo(
                    identifier=full_path,
                    backend_type=BackendType.GGUF,
                    display_name=display,
                    model_key=rel_path,
                    format="GGUF",
                    size_bytes=size,
                    vision=vision,
                    trained_for_tool_use=False,
                    max_context_length=max_ctx,
                    params_string=params_str,
                    architecture=arch,
                    model_path=full_path,
                )
            )
        models.sort(key=lambda m: m.display_name.lower())
        return models

    def list_loaded_models(self) -> list[str]:
        if self._loaded_identifier:
            return [self._loaded_identifier]
        return []

    def load_model(self, identifier: str, config: dict) -> LLMHandle:
        profile = get_active_profile()
        n_gpu = int(config.get("gpu_layers", self._n_gpu_layers))
        n_ctx = min(
            int(config.get("context_length", profile.default_context_tokens)),
            profile.default_context_tokens,
        )
        rc = lb_load_model(identifier, n_ctx, n_gpu)
        if rc != 0:
            raise RuntimeError(lb_last_error() or f"lb_load_model failed ({rc})")
        self._loaded_identifier = identifier
        return LLMHandle(
            native="native",
            identifier=identifier,
            backend_type=BackendType.GGUF,
            config=dict(config),
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
        lb_unload_model()
        self._loaded_identifier = None
        if handle is not None:
            handle.native = None
        return True

    def unload_all_models(self) -> tuple[bool, Optional[str]]:
        lb_unload_model()
        self._loaded_identifier = None
        return (True, None)

    def complete_stream(self, handle: LLMHandle, prompt: str, config: dict) -> CancellableStream:
        max_tokens = int(config.get("max_tokens", 512))
        temperature = float(config.get("temperature", 0.7))

        def _iter() -> Iterator[StreamChunk]:
            q: queue.Queue[Optional[bytes]] = queue.Queue()
            err: list[BaseException] = []

            def run() -> None:
                try:

                    def on_chunk(b: bytes) -> None:
                        q.put(b)

                    rc = nb_complete_stream(prompt, max_tokens, temperature, on_chunk)
                    if rc != 0:
                        err.append(RuntimeError(lb_last_error() or f"native completion failed ({rc})"))
                except BaseException as e:
                    err.append(e)
                finally:
                    q.put(None)

            threading.Thread(target=run, daemon=True).start()
            while True:
                item = q.get()
                if item is None:
                    break
                text = item.decode("utf-8", errors="replace")
                if text:
                    yield StreamChunk(content=text)
            if err:
                raise err[0]
            yield StreamChunk(stop_reason="stop")

        return CancellableStream(iterator=_iter())

    def chat_stream(
        self,
        handle: LLMHandle,
        messages: list[dict],
        config: dict,
        images: Optional[list[Any]] = None,
    ) -> CancellableStream:
        if images:
            raise NotImplementedError("Vision chat requires mmproj; not implemented for native Android yet.")
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        prompt = "\n".join(parts)
        return self.complete_stream(handle, prompt, config)

    def act_with_tools(
        self,
        handle: LLMHandle,
        messages: list[dict],
        tools: list[Callable],
        config: dict,
        on_message: Optional[Callable] = None,
        on_prediction_fragment: Optional[Callable] = None,
    ) -> ToolResult:
        raise NotImplementedError("Tool calling is not implemented for native Android GGUF.")

    def prepare_image(self, path: str) -> Any:
        if not os.path.exists(path):
            raise ValueError(f"Image file not found: {path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def tokenize(self, handle: LLMHandle, text: str) -> list[int]:
        word_count = len(text.split())
        return list(range(int(word_count * 1.33)))

    def supports_vision(self) -> bool:
        return False

    def supports_tools(self) -> bool:
        return False

    def is_unreachable_error(self, error: Exception) -> bool:
        return False

    def is_load_blocked_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return "out of memory" in msg or "oom" in msg or "failed to load" in msg

    @property
    def unreachable_message(self) -> str:
        return "Native GGUF runtime failed. Check LLAMA_BENCH_NATIVE_LIB_DIR and .so dependencies."
