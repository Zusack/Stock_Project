"""Regression tests for LLMManager lock handling."""

import threading

from src.llm.manager import LLMManager


def test_connect_does_not_deadlock_on_first_call(monkeypatch):
    """connect() used to call reset() while holding a non-reentrant lock."""
    mgr = LLMManager()
    mgr.reset()

    class _FakeBackend:
        def connect(self) -> None:
            pass

        def disconnect(self) -> None:
            pass

        def list_available_models(self):
            return []

    monkeypatch.setattr(
        "src.llm.manager.BackendFactory.create_from_string",
        lambda *_a, **_k: _FakeBackend(),
    )

    done = threading.Event()
    err: list[Exception] = []

    def _work():
        try:
            mgr.connect()
        except Exception as ex:
            err.append(ex)
        finally:
            done.set()

    threading.Thread(target=_work, daemon=True).start()
    assert done.wait(2.0), "LLMManager.connect() deadlocked"
    assert not err
