"""Tests for service-layer infrastructure."""

from __future__ import annotations

from src.services.event_bus import EventBus
from src.services.global_control_service import GlobalControlService


def test_event_bus_subscribe_emit():
    bus = EventBus()
    seen: list[str] = []

    def handler(**kwargs) -> None:
        seen.append(kwargs.get("msg", ""))

    bus.subscribe("ping", handler)
    bus.emit("ping", msg="hello")
    assert seen == ["hello"]
    bus.unsubscribe("ping", handler)
    bus.emit("ping", msg="ignored")
    assert seen == ["hello"]


def test_global_control_singleton():
    a = GlobalControlService()
    b = GlobalControlService()
    assert a is b


def test_settings_store_round_trip():
    from src.services.settings_store import settings_store

    key = "_test_round_trip_key"
    store = settings_store()
    store.set_setting(key, "bar")
    assert store.get_setting(key) == "bar"
    store.set_setting(key, None)
