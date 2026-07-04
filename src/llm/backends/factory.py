"""Factory for creating LLM backend instances from configuration."""
from __future__ import annotations

from typing import Optional

from src.llm.backends.base import LLMBackend
from src.llm.backends.types import BackendType


def _create_lmstudio(settings: dict) -> LLMBackend:
    from src.llm.backends.lmstudio_backend import LMStudioBackend

    base_url = settings.get("lmstudio_base_url", "http://localhost:1234")
    return LMStudioBackend(base_url=base_url or "")


def _create_ollama(settings: dict) -> LLMBackend:
    from src.llm.backends.ollama_backend import OllamaBackend

    host = settings.get("ollama_host", "http://localhost:11434")
    return OllamaBackend(host=host)


def _create_vllm(settings: dict) -> LLMBackend:
    from src.llm.backends.vllm_backend import VLLMBackend

    base_url = settings.get("vllm_base_url", "http://localhost:8000")
    api_key = settings.get("vllm_api_key", "")
    return VLLMBackend(base_url=base_url, api_key=api_key)


_BACKEND_CONSTRUCTORS = {
    BackendType.LMSTUDIO: _create_lmstudio,
    BackendType.OLLAMA: _create_ollama,
    BackendType.VLLM: _create_vllm,
}


class BackendFactory:
    """Creates LLMBackend instances from BackendType and settings."""

    @staticmethod
    def create(backend_type: BackendType, settings: Optional[dict] = None) -> LLMBackend:
        constructor = _BACKEND_CONSTRUCTORS.get(backend_type)
        if constructor is None:
            raise ValueError(
                f"No backend registered for {backend_type!r}. "
                f"Available: {[bt.value for bt in _BACKEND_CONSTRUCTORS]}"
            )
        return constructor(settings or {})

    @staticmethod
    def create_from_string(backend_name: str, settings: Optional[dict] = None) -> LLMBackend:
        bt = BackendType.from_string(backend_name)
        return BackendFactory.create(bt, settings)

    @staticmethod
    def available_backends() -> list[BackendType]:
        return list(_BACKEND_CONSTRUCTORS.keys())
