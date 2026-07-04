"""
Single place for user-visible model labels including backend (provider).
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from src.llm.backends.types import BackendType


def _short_name(display_name: Optional[str], llm_identifier: Optional[str]) -> str:
    d = (display_name or "").strip()
    if d:
        return d
    ident = (llm_identifier or "").strip()
    if not ident:
        return "N/A"
    if "/" in ident:
        return ident.split("/")[-1]
    return ident


def backend_display_name(backend_type: Optional[str]) -> str:
    bt = (backend_type or "lmstudio").strip().lower() or "lmstudio"
    try:
        return BackendType.from_string(bt).display_name()
    except ValueError:
        return backend_type or bt


def model_card_label(
    display_name: Optional[str],
    llm_identifier: Optional[str],
    backend_type: Optional[str],
) -> str:
    """Short name plus provider, e.g. \"Llama 3 (Ollama)\"."""
    short = _short_name(display_name, llm_identifier)
    src = backend_display_name(backend_type)
    return f"{short} ({src})"


def model_tooltip_full_name(
    display_name: Optional[str],
    llm_identifier: Optional[str],
    backend_type: Optional[str],
) -> str:
    """Full identifier line for tooltips: \"org/model — Ollama\"."""
    ident = (llm_identifier or "").strip()
    base = ident or _short_name(display_name, llm_identifier)
    src = backend_display_name(backend_type)
    return f"{base} — {src}"


def model_card_label_from_mapping(m: Mapping[str, Any]) -> str:
    return model_card_label(
        m.get("display_name"),
        m.get("llm_identifier"),
        m.get("backend_type"),
    )


def model_row_from_benchmark_export(row: Mapping[str, Any]) -> str:
    """Row from get_full_benchmark_results after BackendType column is added."""
    return model_card_label(
        row.get("DisplayName"),
        row.get("Model"),
        row.get("BackendType"),
    )


def is_provider_detected(m: Mapping[str, Any]) -> bool:
    """
    True if the model is present: API backends via Quick Scan (sys_is_available);
    GGUF via model_path file existence (ignores stale available_on_disk for API backends).
    """
    bt = (m.get("backend_type") or "lmstudio").strip().lower()
    if bt == "gguf":
        p = (m.get("model_path") or "").strip()
        return bool(p and os.path.isfile(p))
    return bool(m.get("sys_is_available"))


def raw_data_model_key(row: Mapping[str, Any]) -> str:
    """
    Stable key for analytics / exports from a benchmark result row.
    Uses provider suffix when BackendType is present (new rows); else DisplayName/Model only.
    """
    dn = row.get("DisplayName")
    mid = row.get("Model")
    bt = row.get("BackendType")
    if bt is not None and str(bt).strip() != "":
        return model_card_label(dn, mid, bt)
    return ((dn or mid) or "").strip() or "Unknown"
