# src/backends/gguf_metadata.py
"""
Minimal GGUF file metadata reader (no numpy/gguf dependency).
Extracts architecture, context length, and parameter count for Model Manager display.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
# Value type enum (GGUFValueType, matches llama.cpp)
GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16 = 0, 1, 2, 3
GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_FLOAT64 = 4, 5, 6, 7
GGUF_BOOL, GGUF_STRING, GGUF_ARRAY = 8, 9, 10
GGUF_UINT64, GGUF_INT64 = 11, 12


@dataclass
class GGUFMeta:
    """Metadata read from a GGUF file (best-effort)."""
    architecture: str = ""
    context_length: int = 0
    parameter_count: int = 0
    vision: bool = False  # inferred from architecture or file type
    raw: dict = None  # all read KV pairs for debugging

    def __post_init__(self):
        if self.raw is None:
            self.raw = {}


def _read_string(f) -> str:
    n, = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vt: int) -> tuple[int, object]:
    """Read one value of given type; returns (bytes consumed, value)."""
    if vt == GGUF_STRING:
        n, = struct.unpack("<Q", f.read(8))
        data = f.read(n)
        return 8 + n, data.decode("utf-8", errors="replace")
    if vt == GGUF_UINT32:
        x, = struct.unpack("<I", f.read(4))
        return 4, x
    if vt == GGUF_INT32:
        x, = struct.unpack("<i", f.read(4))
        return 4, x
    if vt == GGUF_UINT64:
        x, = struct.unpack("<Q", f.read(8))
        return 8, x
    if vt == GGUF_INT64:
        x, = struct.unpack("<q", f.read(8))
        return 8, x
    if vt == GGUF_FLOAT32:
        f.read(4)
        return 4, None
    if vt == GGUF_FLOAT64:
        f.read(8)
        return 8, None
    if vt == GGUF_BOOL:
        f.read(1)
        return 1, None
    if vt in (GGUF_UINT8, GGUF_INT8):
        f.read(1)
        return 1, None
    if vt in (GGUF_UINT16, GGUF_INT16):
        f.read(2)
        return 2, None
    if vt == GGUF_ARRAY:
        elem_type, = struct.unpack("<I", f.read(4))
        count, = struct.unpack("<Q", f.read(8))
        consumed = 12
        for _ in range(count):
            c, _ = _read_value(f, elem_type)
            consumed += c
        return consumed, None
    # unknown type: skip 4 bytes as fallback (conservative)
    f.read(4)
    return 4, None


def read_gguf_metadata(path: str | Path) -> GGUFMeta | None:
    """
    Read metadata from a GGUF file without loading the model.
    Returns GGUFMeta with architecture, context_length, parameter_count, vision;
    returns None on any error (invalid file, I/O, etc.).
    """
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".gguf":
        return None
    meta = GGUFMeta()
    try:
        with open(path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                return None
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                vt = struct.unpack("<I", f.read(4))[0]
                consumed, value = _read_value(f, vt)
                meta.raw[key] = value
                if key == "general.architecture" and value:
                    meta.architecture = str(value).strip()
                elif key in ("llama.context_length", "llama.rope.context_length", "context_length") and value is not None:
                    try:
                        meta.context_length = int(value)
                    except (TypeError, ValueError):
                        pass
                elif key in ("llama.parameter_count", "general.parameter_count") and value is not None:
                    meta.parameter_count = int(value)
                elif key == "general.file_type" and value:
                    s = str(value).lower()
                    if "mmproj" in s or "clip" in s:
                        meta.vision = True
            # Infer vision from architecture name
            arch_lower = meta.architecture.lower()
            if any(x in arch_lower for x in ("llava", "llava15", "bakllava", "minicpm-v", "phi-3.5-vision", "vision")):
                meta.vision = True
            # If we didn't get context_length from metadata, try fallback from raw
            if meta.context_length <= 0:
                for k in ("llama.rope.context_length", "llama.context_length"):
                    v = meta.raw.get(k)
                    if v is not None:
                        try:
                            meta.context_length = int(v)
                            break
                        except (TypeError, ValueError):
                            pass
    except Exception:
        return None
    return meta


def format_parameter_count(n: int) -> str:
    """Format e.g. 3200000000 -> '3.2B', 700000000 -> '700M'."""
    if n <= 0:
        return ""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B".replace(".0B", "B")
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)
