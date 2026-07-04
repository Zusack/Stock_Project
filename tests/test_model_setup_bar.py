"""Tests for Dashboard / Assistant model status bar."""

from src.views.components.model_setup_bar import resolve_model_status


def test_resolve_hidden_when_local_ai_disabled():
    assert resolve_model_status(
        lm_studio_enabled=False,
        setup_mode=False,
        connection_ok=True,
        loaded_model="",
        configured_model="gpt",
        backend_label="LM Studio",
    ) is None


def test_resolve_unreachable_does_not_show_configured_as_active():
    status = resolve_model_status(
        lm_studio_enabled=True,
        setup_mode=False,
        connection_ok=False,
        loaded_model="",
        configured_model="openai/gpt-oss-20b",
        backend_label="LM Studio",
    )
    assert status is not None
    assert status.tone == "error"
    assert "not running" in status.message.lower()
    assert "gpt-oss" not in status.message


def test_resolve_connected_shows_configured_model():
    status = resolve_model_status(
        lm_studio_enabled=True,
        setup_mode=False,
        connection_ok=True,
        loaded_model="",
        configured_model="openai/gpt-oss-20b",
        backend_label="LM Studio",
    )
    assert status is not None
    assert status.tone == "success"
    assert "Active model: openai/gpt-oss-20b" == status.message


def test_resolve_loaded_model_while_checking():
    status = resolve_model_status(
        lm_studio_enabled=True,
        setup_mode=False,
        connection_ok=None,
        loaded_model="live-model",
        configured_model="",
        backend_label="LM Studio",
    )
    assert status is not None
    assert status.tone == "success"
    assert "live-model" in status.message


def test_resolve_disconnected_after_load():
    status = resolve_model_status(
        lm_studio_enabled=True,
        setup_mode=False,
        connection_ok=False,
        loaded_model="live-model",
        configured_model="",
        backend_label="LM Studio",
    )
    assert status is not None
    assert status.tone == "error"
    assert "disconnected" in status.message.lower()
