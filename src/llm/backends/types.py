# src/backends/types.py
"""
Shared types for the backend abstraction layer.
All backends produce and consume these normalized types.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class BackendType(enum.Enum):
    """Supported LLM backend providers."""
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    GGUF = "gguf"
    VLLM = "vllm"

    @classmethod
    def from_string(cls, value: str) -> "BackendType":
        """Convert a string to BackendType (case-insensitive)."""
        val = (value or "").strip().lower()
        for member in cls:
            if member.value == val:
                return member
        raise ValueError(f"Unknown backend type: {value!r}. Valid: {[m.value for m in cls]}")

    def display_name(self) -> str:
        """Human-readable name for UI display."""
        names = {
            BackendType.LMSTUDIO: "LM Studio",
            BackendType.OLLAMA: "Ollama",
            BackendType.GGUF: "Local File",
            BackendType.VLLM: "vLLM",
        }
        return names.get(self, self.value)


@dataclass
class StreamChunk:
    """Normalized streaming token from any backend."""
    content: Optional[str] = None
    stop_reason: Optional[str] = None

    @property
    def is_final(self) -> bool:
        return self.stop_reason is not None


@dataclass
class ModelInfo:
    """Unified model metadata returned by list_available_models()."""
    identifier: str
    backend_type: BackendType
    display_name: str = ""
    model_key: str = ""
    format: str = ""
    size_bytes: int = 0
    vision: bool = False
    trained_for_tool_use: bool = False
    max_context_length: int = 0
    params_string: str = ""
    architecture: str = ""
    model_path: str = ""


@dataclass
class LLMHandle:
    """
    Opaque wrapper holding a backend's native model object.
    Engines interact with this through the backend interface only.
    """
    native: Any = None
    identifier: str = ""
    backend_type: BackendType = BackendType.LMSTUDIO
    config: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from a tool-calling invocation."""
    response_text: str = ""
    tool_call_log: Optional[str] = None
    tokens_generated: int = 0


class CancellableStream:
    """
    Wraps a backend-specific stream iterator to provide a uniform cancel/close interface.
    Engines and stream_utils use this to cancel ongoing generation.
    """

    def __init__(self, iterator, cancel_fn=None, close_fn=None):
        """
        Args:
            iterator: The underlying iterator yielding StreamChunk objects.
            cancel_fn: Optional callable to cancel the stream (e.g. stream.cancel).
            close_fn: Optional callable to close the stream (e.g. stream.close).
        """
        self._iterator = iterator
        self._cancel_fn = cancel_fn
        self._close_fn = close_fn
        self._cancelled = False

    def __iter__(self):
        return self

    def __next__(self) -> StreamChunk:
        if self._cancelled:
            raise StopIteration
        return next(self._iterator)

    def cancel(self):
        """Cancel the stream (best-effort)."""
        self._cancelled = True
        if self._cancel_fn:
            try:
                self._cancel_fn()
            except Exception:
                pass

    def close(self):
        """Close the stream and release resources."""
        self._cancelled = True
        if self._close_fn:
            try:
                self._close_fn()
            except Exception:
                pass
