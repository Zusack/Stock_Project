"""Tests for HTTP LLM connection probes."""

from unittest.mock import MagicMock, patch

from src.llm.connection_probe import probe_backend


def test_probe_lm_studio_unreachable():
    with patch("src.llm.connection_probe.requests.get") as mock_get:
        mock_get.side_effect = __import__("requests").exceptions.ConnectionError()
        ok, msg = probe_backend(
            "lmstudio",
            {"lmstudio_base_url": "http://localhost:1234"},
            timeout_sec=5,
        )
    assert ok is False
    assert "not reachable" in msg.lower()


def test_probe_lm_studio_lists_models():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [{"id": "model-a"}, {"id": "model-b"}],
    }
    with patch("src.llm.connection_probe.requests.get", return_value=response):
        ok, msg = probe_backend(
            "lmstudio",
            {"lmstudio_base_url": "http://localhost:1234/v1"},
            timeout_sec=5,
        )
    assert ok is True
    assert "2 model" in msg
    assert "model-a" in msg


def test_probe_ollama_lists_models():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "models": [{"name": "llama3:latest"}],
    }
    with patch("src.llm.connection_probe.requests.get", return_value=response):
        ok, msg = probe_backend(
            "ollama",
            {"ollama_host": "http://localhost:11434"},
            timeout_sec=5,
        )
    assert ok is True
    assert "llama3:latest" in msg
