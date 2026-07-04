# src/backends/base.py
"""
Abstract base class defining the interface all LLM backends must implement.
Adding a new backend (e.g. TGI, Aphrodite) requires only implementing this interface,
registering in factory.py, and adding a BackendType enum value.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Optional

from src.llm.backends.types import (
    BackendType,
    CancellableStream,
    LLMHandle,
    ModelInfo,
    StreamChunk,
    ToolResult,
)


class LLMBackend(ABC):
    """
    Abstract interface for LLM backend providers.

    Lifecycle:
        connect() -> [use model operations] -> disconnect()

    All implementations must be thread-safe for concurrent reads (e.g. thermal
    monitoring while streaming), though only one model load/unload should happen
    at a time within the benchmark controller.
    """

    # -- Identity --

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the BackendType enum value for this backend."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        return self.backend_type.display_name()

    # -- Connection lifecycle --

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the backend service.
        For local backends (GGUF), this may be a no-op.
        Raises RuntimeError if the service is unreachable.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Cleanly disconnect from the backend. Safe to call multiple times."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend service is reachable and ready."""
        ...

    # -- Model management --

    @abstractmethod
    def list_available_models(self) -> list[ModelInfo]:
        """
        List all models available (downloaded/installed) in this backend.
        Returns an empty list if no models or the backend is unreachable.
        """
        ...

    @abstractmethod
    def list_loaded_models(self) -> list[str]:
        """
        Return identifiers of models currently loaded in memory.
        Returns an empty list if none or not supported.
        """
        ...

    @abstractmethod
    def load_model(self, identifier: str, config: dict) -> LLMHandle:
        """
        Load a model into memory and return a handle for generation.

        Args:
            identifier: Model identifier (path for GGUF, model name for others).
            config: Dict with keys like 'context_length', 'gpu_layers', etc.

        Returns:
            LLMHandle wrapping the native model object.

        Raises:
            RuntimeError: If model cannot be loaded (OOM, not found, etc.).
        """
        ...

    @abstractmethod
    def load_model_alongside(self, identifier: str, config: dict) -> LLMHandle:
        """
        Load a model without unloading existing models (for backends that support it).
        Falls back to load_model() if not supported.

        Args:
            identifier: Model identifier.
            config: Dict with keys like 'context_length', etc.

        Returns:
            LLMHandle wrapping the native model object.
        """
        ...

    @abstractmethod
    def unload_model(self, handle: LLMHandle, stream: Optional[CancellableStream] = None) -> bool:
        """
        Unload a model from memory. Closes any active stream first.

        Args:
            handle: The LLMHandle returned by load_model().
            stream: Optional active stream to close before unloading.

        Returns:
            True if unloaded successfully, False on error.
        """
        ...

    @abstractmethod
    def unload_all_models(self) -> tuple[bool, Optional[str]]:
        """
        Unload all models currently in memory.
        Returns (success, error_message).
        """
        ...

    # -- Generation --

    @abstractmethod
    def complete_stream(
        self, handle: LLMHandle, prompt: str, config: dict
    ) -> CancellableStream:
        """
        Stream a text completion.

        Args:
            handle: Loaded model handle.
            prompt: Raw text prompt.
            config: Inference config (temperature, max_tokens, etc.).

        Returns:
            CancellableStream yielding StreamChunk objects.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        handle: LLMHandle,
        messages: list[dict],
        config: dict,
        images: Optional[list[Any]] = None,
    ) -> CancellableStream:
        """
        Stream a chat completion.

        Args:
            handle: Loaded model handle.
            messages: List of message dicts with 'role' and 'content'.
            config: Inference config (temperature, max_tokens, etc.).
            images: Optional list of prepared image objects for vision prompts.

        Returns:
            CancellableStream yielding StreamChunk objects.
        """
        ...

    @abstractmethod
    def act_with_tools(
        self,
        handle: LLMHandle,
        messages: list[dict],
        tools: list[Callable],
        config: dict,
        on_message: Optional[Callable] = None,
        on_prediction_fragment: Optional[Callable] = None,
    ) -> ToolResult:
        """
        Execute a tool-calling loop (model calls tools, gets results, responds).

        Args:
            handle: Loaded model handle.
            messages: Initial messages (usually user prompt).
            tools: List of Python callables the model can invoke.
            config: Inference config.
            on_message: Callback for each message in the tool loop.
            on_prediction_fragment: Callback for streaming prediction tokens.

        Returns:
            ToolResult with the final response and tool call log.

        Raises:
            NotImplementedError: If the backend does not support tool calling.
        """
        ...

    # -- Utilities --

    @abstractmethod
    def prepare_image(self, path: str) -> Any:
        """
        Prepare an image file for use in vision prompts.

        Args:
            path: Filesystem path to the image.

        Returns:
            Backend-specific image object to pass to chat_stream(images=...).

        Raises:
            NotImplementedError: If the backend does not support vision.
            ValueError: If the image file is invalid or not found.
        """
        ...

    @abstractmethod
    def tokenize(self, handle: LLMHandle, text: str) -> list[int]:
        """
        Tokenize text using the loaded model's tokenizer.

        Args:
            handle: Loaded model handle.
            text: Text to tokenize.

        Returns:
            List of token IDs. If tokenization is not supported, returns an
            estimated list based on word count (4/3 tokens per word heuristic).
        """
        ...

    # -- Capability queries --

    @abstractmethod
    def supports_vision(self) -> bool:
        """Return True if this backend supports vision/image prompts."""
        ...

    @abstractmethod
    def supports_tools(self) -> bool:
        """Return True if this backend supports tool/function calling."""
        ...

    def supports_load_alongside(self) -> bool:
        """Return True if load_model_alongside() is meaningful (not just load_model)."""
        return False

    # -- Error classification --

    @abstractmethod
    def is_unreachable_error(self, error: Exception) -> bool:
        """Return True if the error indicates the backend service is unreachable."""
        ...

    @abstractmethod
    def is_load_blocked_error(self, error: Exception) -> bool:
        """Return True if model loading failed due to resource conflict."""
        ...

    @property
    def unreachable_message(self) -> str:
        """User-facing message when the backend is unreachable."""
        return f"{self.display_name} is not reachable. Please start {self.display_name} and try again."

    @property
    def model_already_loaded_message(self) -> str:
        """User-facing message when model loading is blocked by an existing model."""
        return (
            f"A model is already loaded in {self.display_name}. "
            f"Unload it before loading another, or use Settings to auto-unload."
        )
