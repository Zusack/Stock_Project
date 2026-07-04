#!/usr/bin/env python3
# src/backends/gguf_worker.py
"""
Subprocess worker for GGUF model inference via llama-cpp-python.
Communicates with the parent process via JSON messages over stdin/stdout.

This runs in a separate process so that all GPU memory is released when
the process exits -- working around llama-cpp-python's known VRAM leak.

Protocol (JSON-RPC-like, one JSON object per line):
  Parent -> Worker: {"cmd": "...", "id": N, ...}
  Worker -> Parent: {"id": N, "ok": true, ...} or {"id": N, "error": "..."}
  Streaming:        {"id": N, "stream": true, "content": "...", "stop_reason": null}
                    {"id": N, "stream_end": true, "stop_reason": "stop"}
"""
from __future__ import annotations

import os
import sys

# Remove this script's directory from sys.path so it does not shadow the stdlib
# 'types' module (this package has src/backends/types.py which would be found first).
_worker_dir = os.path.dirname(os.path.abspath(__file__))
if _worker_dir in sys.path:
    sys.path.remove(_worker_dir)

import base64
import json
import traceback


def _send(obj: dict):
    """Send a JSON message to the parent process."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _recv() -> dict:
    """Receive a JSON message from the parent process."""
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line.strip())


def main():
    _llm = None
    _chat_handler = None

    while True:
        try:
            msg = _recv()
        except (json.JSONDecodeError, EOFError):
            break

        cmd = msg.get("cmd")
        msg_id = msg.get("id", 0)

        try:
            if cmd == "ping":
                _send({"id": msg_id, "ok": True, "version": "1.0"})

            elif cmd == "load":
                from llama_cpp import Llama

                model_path = msg["model_path"]
                n_ctx = msg.get("n_ctx", 8192)
                n_gpu_layers = msg.get("n_gpu_layers", -1)
                main_gpu = msg.get("main_gpu", 0)
                chat_format = msg.get("chat_format", None)
                mmproj_path = msg.get("mmproj_path", None)

                kwargs = {
                    "model_path": model_path,
                    "n_ctx": n_ctx,
                    "n_gpu_layers": n_gpu_layers,
                    "main_gpu": main_gpu,
                    "verbose": False,
                }
                if chat_format:
                    kwargs["chat_format"] = chat_format

                if mmproj_path and os.path.isfile(mmproj_path):
                    try:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        _chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                        kwargs["chat_handler"] = _chat_handler
                    except ImportError:
                        pass

                _llm = Llama(**kwargs)
                _send({"id": msg_id, "ok": True})

            elif cmd == "unload":
                if _llm is not None:
                    del _llm
                    _llm = None
                _chat_handler = None
                _send({"id": msg_id, "ok": True})

            elif cmd == "complete":
                if _llm is None:
                    _send({"id": msg_id, "error": "No model loaded"})
                    continue
                prompt = msg["prompt"]
                temperature = msg.get("temperature", 0.7)
                max_tokens = msg.get("max_tokens", 4096)
                stream = msg.get("stream", True)

                if stream:
                    gen = _llm(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                    for chunk in gen:
                        choices = chunk.get("choices", [{}])
                        text = choices[0].get("text", "") if choices else ""
                        finish = choices[0].get("finish_reason") if choices else None
                        _send({
                            "id": msg_id, "stream": True,
                            "content": text,
                            "stop_reason": finish,
                        })
                    _send({"id": msg_id, "stream_end": True, "stop_reason": "stop"})
                else:
                    result = _llm(prompt, max_tokens=max_tokens, temperature=temperature)
                    text = result["choices"][0]["text"] if result.get("choices") else ""
                    _send({"id": msg_id, "ok": True, "text": text})

            elif cmd == "chat":
                if _llm is None:
                    _send({"id": msg_id, "error": "No model loaded"})
                    continue
                messages = msg["messages"]
                temperature = msg.get("temperature", 0.7)
                max_tokens = msg.get("max_tokens", 4096)
                stream = msg.get("stream", True)

                if stream:
                    gen = _llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                    for chunk in gen:
                        choices = chunk.get("choices", [{}])
                        delta = choices[0].get("delta", {}) if choices else {}
                        content = delta.get("content", "")
                        finish = choices[0].get("finish_reason") if choices else None
                        _send({
                            "id": msg_id, "stream": True,
                            "content": content,
                            "stop_reason": finish,
                        })
                    _send({"id": msg_id, "stream_end": True, "stop_reason": "stop"})
                else:
                    result = _llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    content = ""
                    if result.get("choices"):
                        content = result["choices"][0].get("message", {}).get("content", "")
                    _send({"id": msg_id, "ok": True, "text": content})

            elif cmd == "tokenize":
                if _llm is None:
                    _send({"id": msg_id, "error": "No model loaded"})
                    continue
                text = msg["text"]
                tokens = _llm.tokenize(text.encode("utf-8"))
                _send({"id": msg_id, "ok": True, "tokens": tokens})

            elif cmd == "memory":
                import psutil
                proc = psutil.Process(os.getpid())
                mem_info = proc.memory_info()
                _send({
                    "id": msg_id, "ok": True,
                    "rss_gb": round(mem_info.rss / (1024 ** 3), 3),
                    "vms_gb": round(mem_info.vms / (1024 ** 3), 3),
                })

            elif cmd == "exit":
                _send({"id": msg_id, "ok": True})
                break

            else:
                _send({"id": msg_id, "error": f"Unknown command: {cmd}"})

        except Exception as e:
            _send({"id": msg_id, "error": str(e), "traceback": traceback.format_exc()})

    if _llm is not None:
        del _llm
    sys.exit(0)


if __name__ == "__main__":
    main()
