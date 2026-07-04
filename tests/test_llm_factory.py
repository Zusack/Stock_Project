"""Tests for LLM backend factory."""

from src.llm.backends import BackendFactory, BackendType


def test_backend_factory_available():
    backends = BackendFactory.available_backends()
    assert BackendType.LMSTUDIO in backends
    assert BackendType.OLLAMA in backends
    assert BackendType.VLLM in backends
    assert BackendType.GGUF not in backends


def test_backend_factory_create_lmstudio():
    backend = BackendFactory.create(
        BackendType.LMSTUDIO,
        {"lmstudio_base_url": "http://localhost:1234"},
    )
    assert backend.display_name == "LM Studio"


def test_backend_factory_create_from_string():
    backend = BackendFactory.create_from_string("ollama", {"ollama_host": "http://localhost:11434"})
    assert "Ollama" in backend.display_name
