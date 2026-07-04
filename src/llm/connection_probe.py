"""Fast HTTP connectivity probes for LLM backends (Settings → Test connection)."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

from src.llm.backends.types import BackendType


def _normalize_http_base(url: str, *, default: str) -> str:
    base = (url or default).strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base or default


def _format_model_sample(names: list[str], *, limit: int = 3) -> str:
    if not names:
        return ""
    sample = ", ".join(names[:limit])
    suffix = f" (e.g. {sample})"
    if len(names) > limit:
        suffix += f", +{len(names) - limit} more"
    return suffix


def _probe_openai_models(
    base_url: str,
    *,
    label: str,
    timeout_sec: float,
    headers: dict[str, str] | None = None,
) -> tuple[bool, str]:
    base = _normalize_http_base(base_url, default="http://localhost:1234")
    host = urlparse(base).netloc or base
    url = f"{base}/v1/models"
    try:
        resp = requests.get(url, timeout=timeout_sec, headers=headers or {})
    except requests.exceptions.ConnectTimeout:
        return False, f"Timed out after {timeout_sec:.0f}s connecting to {label} at {host}."
    except requests.exceptions.ReadTimeout:
        return False, f"Timed out after {timeout_sec:.0f}s waiting for {label} at {host}."
    except requests.exceptions.ConnectionError:
        return (
            False,
            f"{label} is not reachable at {host}. Start the server and try again.",
        )
    except requests.exceptions.RequestException as ex:
        return False, f"Could not reach {label}: {ex}"

    if resp.status_code == 401:
        return False, f"{label} rejected the request (HTTP 401). Check API key settings."
    if resp.status_code >= 400:
        return (
            False,
            f"{label} returned HTTP {resp.status_code} from {url}.",
        )

    try:
        payload = resp.json()
    except ValueError:
        return False, f"{label} returned a non-JSON response from {url}."

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return False, f"Connected to {label}, but /v1/models returned an unexpected payload."

    names: list[str] = []
    for item in data:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("model")
        else:
            mid = getattr(item, "id", None) or getattr(item, "model", None)
        if mid:
            names.append(str(mid))

    if not names:
        return False, f"Connected to {label}, but no models were listed at {url}."
    return True, f"Connected — {len(names)} model(s) available{_format_model_sample(names)}."


def _probe_ollama(host: str, *, timeout_sec: float) -> tuple[bool, str]:
    base = _normalize_http_base(host, default="http://localhost:11434")
    host_label = urlparse(base).netloc or base
    url = f"{base}/api/tags"
    try:
        resp = requests.get(url, timeout=timeout_sec)
    except requests.exceptions.Timeout:
        return False, f"Timed out after {timeout_sec:.0f}s connecting to Ollama at {host_label}."
    except requests.exceptions.ConnectionError:
        return (
            False,
            f"Ollama is not reachable at {host_label}. Start Ollama and try again.",
        )
    except requests.exceptions.RequestException as ex:
        return False, f"Could not reach Ollama: {ex}"

    if resp.status_code >= 400:
        return False, f"Ollama returned HTTP {resp.status_code} from {url}."

    try:
        payload = resp.json()
    except ValueError:
        return False, "Ollama returned a non-JSON response."

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return False, "Connected to Ollama, but /api/tags returned an unexpected payload."

    names: list[str] = []
    for item in models:
        if isinstance(item, dict):
            name = item.get("name") or item.get("model")
        else:
            name = getattr(item, "name", None) or getattr(item, "model", None)
        if name:
            names.append(str(name))

    if not names:
        return False, "Connected to Ollama, but no models are installed."
    return True, f"Connected — {len(names)} model(s) available{_format_model_sample(names)}."


def probe_backend(
    backend_name: str,
    settings: dict[str, Any],
    *,
    timeout_sec: float,
) -> tuple[bool, str]:
    """
    Lightweight reachability check using HTTP only (no SDK connect).

    Safe to call from a background thread; respects timeout_sec wall clock.
    """
    timeout = max(3.0, float(timeout_sec))
    bt = BackendType.from_string(backend_name)

    if bt == BackendType.LMSTUDIO:
        return _probe_openai_models(
            settings.get("lmstudio_base_url", "http://localhost:1234"),
            label="LM Studio",
            timeout_sec=timeout,
        )
    if bt == BackendType.OLLAMA:
        return _probe_ollama(
            settings.get("ollama_host", "http://localhost:11434"),
            timeout_sec=timeout,
        )
    if bt == BackendType.VLLM:
        headers: dict[str, str] = {}
        api_key = (settings.get("vllm_api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return _probe_openai_models(
            settings.get("vllm_base_url", "http://localhost:8000"),
            label="vLLM",
            timeout_sec=timeout,
            headers=headers,
        )
    return False, f"Unsupported backend: {backend_name}"
