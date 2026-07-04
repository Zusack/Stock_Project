"""
Load NDK-built libllama_bench_native and llama.cpp dependencies via ctypes.

Set LLAMA_BENCH_NATIVE_LIB_DIR to the directory containing libllama.so, libggml*.so,
and libllama_bench_native.so (e.g. platform/android/native/_build/bin on Linux dev).

On Android, point this at the directory where JNI libs are extracted (see docs).
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import CFUNCTYPE, c_char_p, c_float, c_int, c_void_p
from pathlib import Path
from typing import Any, Callable, Optional

_lib: Optional[ctypes.CDLL] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_embedded_android_python() -> bool:
    """True when running the Flet / Serious Python app on a device (not desktop dev)."""
    if (os.environ.get("LLM_BENCH_PROFILE") or "").strip().lower() == "android":
        return True
    # Typical env vars on Android embedded runtimes
    if os.environ.get("ANDROID_ARGUMENT") or os.environ.get("ANDROID_ROOT"):
        return True
    # sys.getandroidapilevel exists on some Android Python builds
    return hasattr(sys, "getandroidapilevel")


def default_native_lib_dir() -> Optional[Path]:
    env = (os.environ.get("LLAMA_BENCH_NATIVE_LIB_DIR") or "").strip()
    if env:
        return Path(env)

    # Never auto-use CMake _build/ from the repo on Android: those .so files are almost always
    # host builds (e.g. x86_64 Linux) and will fail with EM_X86_64 vs EM_AARCH64 on arm64 devices.
    if _is_embedded_android_python():
        from shared_core.android_native_discovery import find_native_lib_dir

        found = find_native_lib_dir()
        if found is not None:
            return found
        return None

    dev = _project_root() / "platform" / "android" / "native" / "_build" / "bin"
    if dev.is_dir():
        return dev
    dev2 = _project_root() / "platform" / "android" / "native" / "_build"
    if (dev2 / "libllama_bench_native.so").is_file():
        return dev2
    return None


def _preload_posix_dependencies(lib_dir: Path) -> None:
    # Order matters: ggml stack then llama, then our bridge.
    names = (
        "libggml-base.so",
        "libggml.so",
        "libggml-cpu.so",
        "libllama.so",
    )
    flags = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    strict = _is_embedded_android_python()
    for n in names:
        p = lib_dir / n
        if not p.is_file():
            if strict and n in ("libggml-base.so", "libggml.so", "libggml-cpu.so", "libllama.so"):
                raise OSError(
                    f"Missing required native library {n} under {lib_dir}. "
                    "Rebuild with scripts/android/build_native.sh android and repackage the APK."
                )
            continue
        try:
            ctypes.CDLL(str(p), mode=flags)
        except OSError as e:
            if strict:
                raise OSError(
                    f"dlopen failed for {p}: {e}. "
                    "On Android, all of libggml*.so and libllama.so must sit in the same folder as "
                    "libllama_bench_native.so (see platform/android/prebuilt/arm64-v8a)."
                ) from e
            # Desktop: optional preload failures are non-fatal (layout varies by llama.cpp version).


def _prepend_ld_library_path(lib_dir: Path) -> None:
    """Help Android's linker find DT_NEEDED deps when RUNPATH is absent on older builds."""
    if not _is_embedded_android_python():
        return
    d = str(lib_dir)
    old = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [d] + [x for x in old.split(os.pathsep) if x and x != d]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)


def load_native_library(lib_dir: Optional[Path] = None) -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    d = lib_dir or default_native_lib_dir()
    if not d or not d.is_dir():
        if _is_embedded_android_python():
            raise OSError(
                "Native ARM64 libraries are not installed for this app. "
                "Host CMake _build/*.so (x86_64) cannot run on Android. "
                "Build with the Android NDK for arm64-v8a (see docs/ANDROID_NATIVE_BUILD.md), "
                "then copy libggml*.so, libllama.so, and libllama_bench_native.so into "
                "platform/android/prebuilt/arm64-v8a/ (packaged with the app) or jniLibs/arm64-v8a. "
                "Optionally set LLAMA_BENCH_NATIVE_LIB_DIR to that folder at runtime."
            )
        raise OSError(
            "Native inference libraries not found. Set LLAMA_BENCH_NATIVE_LIB_DIR to the folder "
            "containing libllama_bench_native.so (and libllama.so / libggml*.so), or build with "
            "platform/android/native/CMakeLists.txt"
        )

    bridge = d / "libllama_bench_native.so"
    lib_dir = d
    if not bridge.is_file():
        bridge = d.parent / "libllama_bench_native.so"
        if bridge.is_file():
            lib_dir = bridge.parent
    if not bridge.is_file():
        raise OSError(f"libllama_bench_native.so not found under {d}")

    _prepend_ld_library_path(lib_dir)
    if os.name != "nt":
        _preload_posix_dependencies(lib_dir)

    _lib = ctypes.CDLL(str(bridge))
    _lib.lb_init.argtypes = []
    _lib.lb_init.restype = c_int

    _lib.lb_shutdown.argtypes = []
    _lib.lb_shutdown.restype = None

    _lib.lb_load_model.argtypes = [c_char_p, c_int, c_int]
    _lib.lb_load_model.restype = c_int

    _lib.lb_unload_model.argtypes = []
    _lib.lb_unload_model.restype = None

    CHUNK = CFUNCTYPE(None, c_void_p, c_int, c_void_p)
    _lib.lb_complete_stream.argtypes = [c_char_p, c_int, c_float, CHUNK, c_void_p]
    _lib.lb_complete_stream.restype = c_int

    _lib.lb_last_error.argtypes = []
    _lib.lb_last_error.restype = c_char_p

    setattr(_lib, "_CHUNK_CB", CHUNK)
    return _lib


def lb_init() -> int:
    return int(load_native_library().lb_init())


def lb_shutdown() -> None:
    if _lib is not None:
        _lib.lb_shutdown()


def lb_load_model(path: str, n_ctx: int, n_gpu_layers: int) -> int:
    lib = load_native_library()
    b = path.encode("utf-8")
    return int(lib.lb_load_model(b, int(n_ctx), int(n_gpu_layers)))


def lb_unload_model() -> None:
    if _lib is not None:
        _lib.lb_unload_model()


def lb_last_error() -> str:
    if _lib is None:
        return ""
    p = _lib.lb_last_error()
    if not p:
        return ""
    return p.decode("utf-8", errors="replace")


def lb_complete_stream(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    on_chunk: Callable[[bytes], None],
) -> int:
    lib = load_native_library()
    CHUNK = getattr(lib, "_CHUNK_CB")

    def _cb(data: Any, byte_len: int, _user: Any) -> None:
        if not data or byte_len <= 0:
            return
        raw = ctypes.string_at(data, int(byte_len))
        on_chunk(bytes(raw))

    cb = CHUNK(_cb)
    pr = prompt.encode("utf-8")
    rc = int(
        lib.lb_complete_stream(
            pr,
            int(max_new_tokens),
            float(temperature),
            cb,
            None,
        )
    )
    return rc
