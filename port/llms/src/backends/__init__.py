# src/backends/__init__.py
"""
Backend abstraction layer for LLM providers.
Supports LM Studio, Ollama, GGUF (llama-cpp-python), and vLLM.
"""
from src.backends.types import BackendType, StreamChunk, ModelInfo, LLMHandle, ToolResult
from src.backends.base import LLMBackend
from src.backends.factory import BackendFactory

__all__ = [
    "BackendType",
    "StreamChunk",
    "ModelInfo",
    "LLMHandle",
    "ToolResult",
    "LLMBackend",
    "BackendFactory",
]
