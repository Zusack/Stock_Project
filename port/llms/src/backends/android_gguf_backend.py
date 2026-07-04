"""
Android-oriented GGUF adapter.

This class keeps the existing GGUF contract but enforces conservative defaults
for mobile memory/thermal limits. It is selected by BackendFactory when the
runtime target is android.
"""

from __future__ import annotations

from src.backends.gguf_backend import GGUFBackend
from shared_core.config import get_active_profile
from shared_core.mobile_capabilities import inspect_device_capabilities


class AndroidGGUFBackend(GGUFBackend):
    def connect(self) -> None:
        capability = inspect_device_capabilities()
        if not capability.meets_minimum_ram:
            raise RuntimeError(
                "Android GGUF runtime requires at least 4 GB RAM for stable local inference."
            )
        super().connect()

    def load_model(self, identifier: str, config: dict):
        profile = get_active_profile()
        runtime_cfg = dict(config or {})
        runtime_cfg.setdefault("gpu_layers", 0)
        runtime_cfg["context_length"] = min(
            int(runtime_cfg.get("context_length", profile.default_context_tokens)),
            profile.default_context_tokens,
        )
        return super().load_model(identifier=identifier, config=runtime_cfg)
