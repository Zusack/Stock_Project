# src/backends/factory.py
"""
Factory for creating LLM backend instances from configuration.
New backends are registered here -- one import and one dict entry.
"""
from __future__ import annotations

import os
from typing import Optional

from src.backends.base import LLMBackend
from src.backends.types import BackendType


def _create_lmstudio(settings: dict) -> LLMBackend:
    from src.backends.lmstudio_backend import LMStudioBackend
    base_url = settings.get("lmstudio_base_url", "http://localhost:1234")
    return LMStudioBackend(base_url=base_url or "")


def _create_ollama(settings: dict) -> LLMBackend:
    from src.backends.ollama_backend import OllamaBackend
    host = settings.get("ollama_host", "http://localhost:11434")
    return OllamaBackend(host=host)


def _create_gguf(settings: dict) -> LLMBackend:
    runtime_target = (settings.get("gguf_runtime_target", "") or "").strip().lower()
    use_android_runtime = runtime_target == "android" or os.environ.get("LLM_BENCH_PROFILE") == "android"
    android_impl = (settings.get("gguf_android_runtime", "native") or "native").strip().lower()
    include_sub = settings.get("gguf_include_subfolders", "1") in ("1", "true", "yes")
    model_dir = settings.get("gguf_model_dir", "")
    n_gpu = int(settings.get("gguf_n_gpu_layers", -1))
    main_gpu = int(settings.get("gguf_main_gpu", 0))

    if use_android_runtime and android_impl in ("native", "ndk", "jni"):
        from src.backends.native_android_gguf_backend import NativeAndroidGgufBackend

        return NativeAndroidGgufBackend(
            model_dir=model_dir,
            n_gpu_layers=0 if n_gpu < 0 else n_gpu,
            main_gpu=main_gpu,
            include_subfolders=include_sub,
        )
    if use_android_runtime:
        from src.backends.android_gguf_backend import AndroidGGUFBackend as GGUFImpl
    else:
        from src.backends.gguf_backend import GGUFBackend as GGUFImpl
    return GGUFImpl(
        model_dir=model_dir,
        n_gpu_layers=n_gpu,
        main_gpu=main_gpu,
        include_subfolders=include_sub,
    )


def _create_vllm(settings: dict) -> LLMBackend:
    from src.backends.vllm_backend import VLLMBackend
    base_url = settings.get("vllm_base_url", "http://localhost:8000")
    api_key = settings.get("vllm_api_key", "")
    return VLLMBackend(base_url=base_url, api_key=api_key)


# Registry: BackendType -> constructor function
_BACKEND_CONSTRUCTORS = {
    BackendType.LMSTUDIO: _create_lmstudio,
    BackendType.OLLAMA: _create_ollama,
    BackendType.GGUF: _create_gguf,
    BackendType.VLLM: _create_vllm,
}


class BackendFactory:
    """Creates LLMBackend instances from BackendType and settings."""

    @staticmethod
    def create(backend_type: BackendType, settings: Optional[dict] = None) -> LLMBackend:
        """
        Create and return an LLMBackend instance.

        Args:
            backend_type: Which backend to create.
            settings: Dict of backend-specific settings (from database).

        Returns:
            An unconnected LLMBackend. Caller must call .connect() before use.

        Raises:
            ValueError: If backend_type is not registered.
        """
        constructor = _BACKEND_CONSTRUCTORS.get(backend_type)
        if constructor is None:
            raise ValueError(
                f"No backend registered for {backend_type!r}. "
                f"Available: {[bt.value for bt in _BACKEND_CONSTRUCTORS]}"
            )
        return constructor(settings or {})

    @staticmethod
    def create_from_string(backend_name: str, settings: Optional[dict] = None) -> LLMBackend:
        """Convenience: create a backend from a string name (e.g. 'ollama')."""
        bt = BackendType.from_string(backend_name)
        return BackendFactory.create(bt, settings)

    @staticmethod
    def available_backends() -> list[BackendType]:
        """Return list of all registered backend types."""
        return list(_BACKEND_CONSTRUCTORS.keys())
