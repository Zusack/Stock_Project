"""Backend abstraction layer for LLM providers (LM Studio, Ollama, vLLM)."""

from src.llm.backends.types import BackendType, StreamChunk, ModelInfo, LLMHandle, ToolResult
from src.llm.backends.base import LLMBackend
from src.llm.backends.factory import BackendFactory

__all__ = [
    "BackendType",
    "StreamChunk",
    "ModelInfo",
    "LLMHandle",
    "ToolResult",
    "LLMBackend",
    "BackendFactory",
]
