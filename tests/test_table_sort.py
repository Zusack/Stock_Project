"""Table sort helpers."""

from src.views.components.tables import _parse_sort_number, sort_rows
import flet as ft


def test_parse_sort_number_ratio():
    assert _parse_sort_number("5/7") == 5.0
    assert _parse_sort_number("12/7") == 12.0


def test_sort_rank_numeric():
    rows = [
        ft.DataRow(cells=[ft.DataCell(ft.Text(str(n)))]) for n in (10, 2, 1)
    ]
    sorted_rows = sort_rows(rows, 0, ascending=True, numeric=True)
    values = [r.cells[0].content.value for r in sorted_rows]
    assert values == ["1", "2", "10"]
