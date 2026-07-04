# src/backends/gguf_backend.py
"""
GGUF backend implementation using llama-cpp-python in a subprocess.
Each model load spawns a child process (gguf_worker.py) that loads the GGUF file.
When the model is unloaded, the child process is killed, guaranteeing VRAM release.

- Metadata: Max context, parameter count, and architecture are read from each .gguf
  file for the Model Manager table (see gguf_metadata.py).
- Vision: Supported for LLaVA-style models. Use a vision-capable .gguf and optionally
  a companion .mmproj (or -mmproj.gguf); the backend auto-detects mmproj and enables
  image_comprehension prompts. Image content is passed via create_chat_completion.
- Tool-use: Not implemented; models with function-calling (e.g. Llama 3.1+, Functionary)
  would require a dedicated chat format and grammar. Raised as NotImplementedError.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator, Optional

from src.backends.base import LLMBackend
from src.backends.gguf_metadata import read_gguf_metadata, format_parameter_count
from src.backends.types import (
    BackendType,
    CancellableStream,
    LLMHandle,
    ModelInfo,
    StreamChunk,
    ToolResult,
)


class _WorkerProcess:
    """Manages a single gguf_worker.py subprocess."""

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self._msg_id = 0
        self._lock = threading.Lock()

    def _stderr_snapshot(self) -> str:
        """Read stderr from the worker process (call only after process has exited)."""
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            return (self.proc.stderr.read() or "")[:2000]
        except Exception:
            return ""

    def start(self):
        worker_path = os.path.join(os.path.dirname(__file__), "gguf_worker.py")
        if not os.path.isfile(worker_path):
            raise FileNotFoundError(
                f"GGUF worker script not found: {worker_path}. "
                "Ensure the application is run from the project root."
            )
        # Use project root as cwd so subprocess has same environment
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(worker_path)))
        self.proc = subprocess.Popen(
            [sys.executable, worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=project_root,
        )
        # If the process exits immediately (e.g. import error), surface stderr
        time.sleep(0.15)
        if self.proc.poll() is not None:
            err = self._stderr_snapshot() or "(no stderr)"
            raise RuntimeError(
                "GGUF worker process exited immediately. "
                "Ensure llama-cpp-python is installed in the same environment as this app. "
                f"Worker stderr:\n{err}"
            )

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def send(self, msg: dict) -> dict:
        """Send a command and wait for the response."""
        with self._lock:
            if not self.is_alive():
                raise RuntimeError("GGUF worker process is not running.")
            self._msg_id += 1
            msg["id"] = self._msg_id
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            return self._read_response(self._msg_id)

    def send_streaming(self, msg: dict) -> Iterator[dict]:
        """Send a command and yield streaming responses until stream_end."""
        with self._lock:
            if not self.is_alive():
                raise RuntimeError("GGUF worker process is not running.")
            self._msg_id += 1
            msg_id = self._msg_id
            msg["id"] = msg_id
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self._stderr_snapshot()
                msg = "GGUF worker process ended unexpectedly."
                if err:
                    msg += f" Worker stderr:\n{err}"
                raise RuntimeError(msg)
            try:
                resp = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if resp.get("id") != msg_id:
                continue
            if resp.get("error"):
                raise RuntimeError(resp["error"])
            if resp.get("stream_end"):
                return
            yield resp

    def _read_response(self, expected_id: int) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self._stderr_snapshot()
                msg = "GGUF worker process ended unexpectedly."
                if err:
                    msg += f" Worker stderr:\n{err}"
                raise RuntimeError(msg)
            try:
                resp = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if resp.get("id") == expected_id:
                if resp.get("error"):
                    raise RuntimeError(resp["error"])
                return resp

    def kill(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


class GGUFBackend(LLMBackend):
    """LLM backend loading GGUF files via llama-cpp-python in a subprocess."""

    def __init__(self, model_dir: str = "", n_gpu_layers: int = -1, main_gpu: int = 0, include_subfolders: bool = True):
        self._model_dir = model_dir
        self._n_gpu_layers = n_gpu_layers
        self._main_gpu = main_gpu
        self._include_subfolders = include_subfolders
        self._worker: Optional[_WorkerProcess] = None

    # -- Identity --

    @property
    def backend_type(self) -> BackendType:
        return BackendType.GGUF

    # -- Connection lifecycle --

    def connect(self) -> None:
        # Verify llama-cpp-python is importable (try in main process)
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            from shared_core.config import get_active_profile

            if get_active_profile().name == "android":
                raise ImportError(
                    "GGUF on Android is not available in this build: llama-cpp-python is not bundled "
                    "(PyPI has no installable wheels for Android/arm64 in the APK pip step). "
                    "On-device inference needs a custom native build (e.g. NDK + vendored llama.cpp). "
                    "For GGUF on PC: pip install -r requirements-desktop.txt"
                ) from None
            raise ImportError(
                "The 'llama-cpp-python' package is not installed. "
                "Install with: pip install llama-cpp-python. "
                "For NVIDIA GPU: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python (see project docs)."
            ) from None

    def disconnect(self) -> None:
        if self._worker is not None:
            try:
                self._worker.send({"cmd": "exit"})
            except Exception:
                pass
            self._worker.kill()
            self._worker = None

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    # -- Model management --

    def list_available_models(self) -> list[ModelInfo]:
        if not self._model_dir or not os.path.isdir(self._model_dir):
            return []
        models = []
        if self._include_subfolders:
            candidates = (os.path.join(root, f) for root, _dirs, files in os.walk(self._model_dir) for f in files)
        else:
            candidates = (os.path.join(self._model_dir, f) for f in os.listdir(self._model_dir)
                         if os.path.isfile(os.path.join(self._model_dir, f)))
        for full_path in candidates:
            if not full_path.lower().endswith(".gguf"):
                continue
            fname = os.path.basename(full_path)
            # Exclude mmproj/projection .gguf files: they are vision encoders, not standalone LLMs
            if "mmproj" in fname.lower():
                continue
            rel_path = os.path.relpath(full_path, self._model_dir)
            size = os.path.getsize(full_path)
            display = fname.replace(".gguf", "").replace("-", " ").replace("_", " ")
            # Read metadata from the GGUF file for context, params, architecture, vision
            meta = read_gguf_metadata(full_path)
            max_ctx = 0
            params_str = ""
            arch = ""
            vision = False
            if meta:
                max_ctx = meta.context_length or 0
                params_str = format_parameter_count(meta.parameter_count) if meta.parameter_count else ""
                arch = meta.architecture or ""
                vision = meta.vision
            # Vision: also set if a companion .mmproj exists (LLaVA-style)
            if not vision:
                base = full_path[:-5]  # strip .gguf
                for ext in (".mmproj", "-mmproj.gguf"):
                    if os.path.isfile(base + ext):
                        vision = True
                        break
            models.append(ModelInfo(
                identifier=full_path,
                backend_type=BackendType.GGUF,
                display_name=display,
                model_key=rel_path,
                format="GGUF",
                size_bytes=size,
                vision=vision,
                trained_for_tool_use=False,
                max_context_length=max_ctx,
                params_string=params_str,
                architecture=arch,
                model_path=full_path,
            ))
        models.sort(key=lambda m: m.display_name.lower())
        return models

    def list_loaded_models(self) -> list[str]:
        if self._worker is not None and self._worker.is_alive():
            return ["(GGUF model loaded in subprocess)"]
        return []

    def load_model(self, identifier: str, config: dict) -> LLMHandle:
        if self._worker is not None:
            self._worker.kill()
        self._worker = _WorkerProcess()
        self._worker.start()

        ping_resp = self._worker.send({"cmd": "ping"})
        if not ping_resp.get("ok"):
            raise RuntimeError("GGUF worker failed to start.")

        load_msg = {
            "cmd": "load",
            "model_path": identifier,
            "n_ctx": config.get("context_length", 8192),
            "n_gpu_layers": config.get("gpu_layers", self._n_gpu_layers),
            "main_gpu": config.get("main_gpu", self._main_gpu),
        }
        chat_format = config.get("chat_format")
        if chat_format:
            load_msg["chat_format"] = chat_format
        mmproj_path = config.get("mmproj_path")
        if not mmproj_path and identifier.lower().endswith(".gguf"):
            base = identifier[:-5]
            for p in (base + ".mmproj", base + "-mmproj.gguf"):
                if os.path.isfile(p):
                    mmproj_path = p
                    break
        if mmproj_path:
            load_msg["mmproj_path"] = mmproj_path

        load_resp = self._worker.send(load_msg)
        if load_resp.get("error"):
            err = load_resp["error"] or "Unknown load error"
            err_lower = err.lower()
            hint_parts = []
            if "mmproj" in identifier.lower() or "failed to load model from file" in err_lower:
                hint_parts.append("This may be a companion vision projection (mmproj) file, not a standalone LLM — use the main model .gguf instead (the one without -mmproj in the name).")
            if "unsupported" in err_lower or "architecture" in err_lower or "not supported" in err_lower:
                hint_parts.append("The model architecture may not be supported by this build of llama-cpp-python.")
            if not hint_parts and ("failed to load" in err_lower or "error" in err_lower):
                hint_parts.append("The file may be a companion mmproj (use the main LLM .gguf), or the architecture may not be supported by your llama-cpp-python build.")
            hint = " " + " ".join(hint_parts) if hint_parts else ""
            raise RuntimeError(f"{err}.{hint}")
        return LLMHandle(
            native=self._worker,
            identifier=identifier,
            backend_type=BackendType.GGUF,
            config=config,
        )

    def load_model_alongside(self, identifier: str, config: dict) -> LLMHandle:
        return self.load_model(identifier, config)

    def unload_model(self, handle: LLMHandle, stream: Optional[CancellableStream] = None) -> bool:
        if stream is not None:
            try:
                stream.cancel()
                stream.close()
            except Exception:
                pass
        if self._worker is not None:
            self._worker.kill()
            self._worker = None
        if handle is not None:
            handle.native = None
        return True

    def unload_all_models(self) -> tuple[bool, Optional[str]]:
        if self._worker is not None:
            self._worker.kill()
            self._worker = None
        return (True, None)

    # -- Generation --

    def complete_stream(
        self, handle: LLMHandle, prompt: str, config: dict
    ) -> CancellableStream:
        worker: _WorkerProcess = handle.native
        if worker is None or not worker.is_alive():
            raise RuntimeError("GGUF worker is not running.")

        msg = {
            "cmd": "complete",
            "prompt": prompt,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 4096),
            "stream": True,
        }

        stream_iter = worker.send_streaming(msg)

        def _iter():
            for resp in stream_iter:
                yield StreamChunk(
                    content=resp.get("content"),
                    stop_reason=resp.get("stop_reason"),
                )

        return CancellableStream(iterator=_iter())

    def chat_stream(
        self,
        handle: LLMHandle,
        messages: list[dict],
        config: dict,
        images: Optional[list[Any]] = None,
    ) -> CancellableStream:
        worker: _WorkerProcess = handle.native
        if worker is None or not worker.is_alive():
            raise RuntimeError("GGUF worker is not running.")

        chat_messages = []
        for msg in messages:
            entry = {"role": msg["role"], "content": msg["content"]}
            if msg["role"] == "user" and images:
                content_parts = [{"type": "text", "text": msg["content"]}]
                for img_data in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    })
                entry["content"] = content_parts
            chat_messages.append(entry)

        send_msg = {
            "cmd": "chat",
            "messages": chat_messages,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 4096),
            "stream": True,
        }

        stream_iter = worker.send_streaming(send_msg)

        def _iter():
            for resp in stream_iter:
                yield StreamChunk(
                    content=resp.get("content"),
                    stop_reason=resp.get("stop_reason"),
                )

        return CancellableStream(iterator=_iter())

    def act_with_tools(
        self,
        handle: LLMHandle,
        messages: list[dict],
        tools: list[Callable],
        config: dict,
        on_message: Optional[Callable] = None,
        on_prediction_fragment: Optional[Callable] = None,
    ) -> ToolResult:
        raise NotImplementedError(
            "Tool calling with GGUF backend requires a model with function-calling support "
            "and specific chat format. This feature is experimental."
        )

    # -- Utilities --

    def prepare_image(self, path: str) -> Any:
        if not os.path.exists(path):
            raise ValueError(f"Image file not found: {path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def tokenize(self, handle: LLMHandle, text: str) -> list[int]:
        worker: _WorkerProcess = handle.native
        if worker is None or not worker.is_alive():
            word_count = len(text.split())
            return list(range(int(word_count * 1.33)))
        try:
            resp = worker.send({"cmd": "tokenize", "text": text})
            return resp.get("tokens", [])
        except Exception:
            word_count = len(text.split())
            return list(range(int(word_count * 1.33)))

    # -- Capability queries --

    def supports_vision(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return False

    # -- Error classification --

    def is_unreachable_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return "worker" in msg and ("not running" in msg or "ended" in msg)

    def is_load_blocked_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return "out of memory" in msg or "oom" in msg or "failed to load" in msg

    @property
    def unreachable_message(self) -> str:
        return (
            "GGUF worker process is not running. The model may have crashed or "
            "llama-cpp-python may not be installed correctly."
        )
