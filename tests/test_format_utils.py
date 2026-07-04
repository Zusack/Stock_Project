"""Shared formatting helpers."""

from datetime import datetime

from src.utils.format_utils import format_timestamp, natural_sort_key


def test_format_timestamp_datetime():
    ts = datetime(2026, 6, 28, 14, 30)
    assert format_timestamp(ts) == "2026-06-28 14:30"


def test_format_timestamp_none():
    assert format_timestamp(None) == "N/A"


def test_natural_sort_key_orders_numeric_suffixes():
    items = ["AAPL10", "AAPL2", "AAPL1"]
    assert sorted(items, key=natural_sort_key) == ["AAPL1", "AAPL2", "AAPL10"]
