"""Smoke tests for shared UI theme/table helpers."""

from __future__ import annotations

import flet as ft

from src.views.components.tables import filter_rows, sort_rows
from src.views.theme import InputStyles, ThemeHelper


def test_input_styles_apply_text_field():
    field = ft.TextField(label="Test")
    InputStyles.apply_text_field(None, field)
    assert field.border_color == ThemeHelper.input_border(None)
    assert field.filled is True
    assert field.fill_color is not None


def test_table_filter_and_sort():
    rows = [
        ft.DataRow(cells=[ft.DataCell(ft.Text("AAPL")), ft.DataCell(ft.Text("10"))]),
        ft.DataRow(cells=[ft.DataCell(ft.Text("MSFT")), ft.DataCell(ft.Text("2"))]),
    ]
    filtered = filter_rows(rows, "ms")
    assert len(filtered) == 1
    sorted_rows = sort_rows(filtered, 1, ascending=True, numeric=True)
    cell0 = sorted_rows[0].cells[0].content
    assert isinstance(cell0, ft.Text)
    assert cell0.value == "MSFT"
