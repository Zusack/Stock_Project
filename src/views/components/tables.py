"""Themed DataTable helpers, sorting, search, and responsive wrappers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import flet as ft

from src.views.theme import InputStyles, ThemeHelper


_PLACEHOLDER_COLUMN_LABEL = "—"


def placeholder_data_columns(
    label: str = _PLACEHOLDER_COLUMN_LABEL,
) -> list[ft.DataColumn]:
    """Flet requires at least one visible DataColumn; use this when clearing a table."""
    return [ft.DataColumn(ft.Text(label))]


def themed_data_table(
    page,
    *,
    columns: list[ft.DataColumn],
    rows: list[ft.DataRow] | None = None,
    placeholder_label: str = _PLACEHOLDER_COLUMN_LABEL,
    **extra,
) -> ft.DataTable:
    """Build a DataTable using shared theme styling (borders, heading row, sizes)."""
    if not columns:
        columns = placeholder_data_columns(placeholder_label)
    kwargs = ThemeHelper.results_summary_data_table_kwargs(page)
    kwargs.update(extra)
    return ft.DataTable(columns=columns, rows=rows or [], **kwargs)


def bordered_table_wrap(
    page,
    table: ft.DataTable,
    *,
    expand: bool = False,
) -> ft.Container:
    """Horizontal scroll only — avoids nested vertical scroll inside a tab."""
    return ft.Container(
        content=ft.Row([table], scroll=ft.ScrollMode.AUTO, expand=expand),
        padding=4,
        border=ft.border.all(1, ThemeHelper.border_default(page)),
        border_radius=6,
        expand=expand,
        alignment=ft.alignment.top_left,
    )


def data_column(
    label: str,
    *,
    tooltip: str | None = None,
    numeric: bool = False,
    on_sort: Callable[[ft.DataColumnSortEvent], None] | None = None,
) -> ft.DataColumn:
    """Header column with optional tooltip and sort handler."""
    return ft.DataColumn(
        ft.Text(label, weight=ft.FontWeight.W_600),
        tooltip=tooltip or label,
        numeric=numeric,
        on_sort=on_sort,
    )


def extract_cell_text(cell: ft.DataCell) -> str:
    content = cell.content
    if isinstance(content, ft.Text):
        return (content.value or "").strip()
    if isinstance(content, ft.Container) and isinstance(content.content, ft.Text):
        return (content.content.value or "").strip()
    if isinstance(content, ft.Row):
        parts: list[str] = []
        for c in content.controls:
            if isinstance(c, ft.Text):
                parts.append((c.value or "").strip())
            elif isinstance(c, ft.IconButton):
                parts.append((c.tooltip or "").strip())
        return " ".join(p for p in parts if p)
    return str(content or "").strip()


def extract_row_values(row: ft.DataRow) -> list[str]:
    return [extract_cell_text(c) for c in row.cells]


def filter_rows(rows: list[ft.DataRow], query: str) -> list[ft.DataRow]:
    q = (query or "").strip().lower()
    if not q:
        return list(rows)
    out: list[ft.DataRow] = []
    for row in rows:
        blob = " ".join(extract_row_values(row)).lower()
        if q in blob:
            out.append(row)
    return out


def _parse_sort_number(text: str) -> float:
    raw = (text or "").strip()
    ratio = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*\d+", raw)
    if ratio:
        return float(ratio.group(1))
    cleaned = re.sub(r"[^\d.\-+eE]", "", raw)
    if not cleaned or cleaned in ("-", "+", ".", "-.", "+."):
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def sort_rows(
    rows: list[ft.DataRow],
    column_index: int,
    *,
    ascending: bool = True,
    numeric: bool = False,
) -> list[ft.DataRow]:
    if not rows or column_index < 0:
        return list(rows)

    def sort_key(row: ft.DataRow):
        values = extract_row_values(row)
        raw = values[column_index] if column_index < len(values) else ""
        if numeric:
            return _parse_sort_number(raw)
        return raw.lower()

    return sorted(rows, key=sort_key, reverse=not ascending)


@dataclass
class TableViewState:
    """Client-side table search + sort state."""

    all_rows: list[ft.DataRow]
    search_query: str = ""
    sort_column_index: int | None = None
    sort_ascending: bool = True
    sort_numeric: bool = False

    def visible_rows(self) -> list[ft.DataRow]:
        rows = filter_rows(self.all_rows, self.search_query)
        if self.sort_column_index is not None:
            rows = sort_rows(
                rows,
                self.sort_column_index,
                ascending=self.sort_ascending,
                numeric=self.sort_numeric,
            )
        return rows


class InteractiveTablePanel:
    """
    Search field + bordered, scrollable DataTable with optional column sorting.
    """

    def __init__(
        self,
        page,
        columns: list[ft.DataColumn],
        *,
        search_label: str = "Search table",
        search_hint: str = "Filter visible rows…",
        expand: bool = True,
        empty_message: str = "No rows to display.",
        table_kwargs: dict | None = None,
    ):
        self.page = page
        self._empty_message = empty_message
        self._state = TableViewState(all_rows=[])
        self._numeric_columns: set[int] = set()

        self.search_field = InputStyles.text_field(
            page,
            label=search_label,
            hint_text=search_hint,
            expand=True,
            on_change=self._on_search,
            tooltip="Type to filter rows in the current table.",
        )
        self.table = themed_data_table(
            page,
            columns=columns,
            rows=[],
            **(table_kwargs or {}),
        )
        self._wrap = bordered_table_wrap(page, self.table, expand=expand)

        for i, col in enumerate(columns):
            if getattr(col, "numeric", False):
                self._numeric_columns.add(i)

    @property
    def control(self) -> ft.Column:
        return ft.Column(
            [self.search_field, ft.Container(content=self._wrap, expand=True)],
            spacing=8,
            expand=True,
        )

    def set_rows(self, rows: list[ft.DataRow]) -> None:
        self._state.all_rows = list(rows)
        self._apply()

    def _apply(self) -> None:
        visible = self._state.visible_rows()
        if not visible and self._state.all_rows:
            visible = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                f"No rows match “{self._state.search_query}”.",
                                color=ThemeHelper.text_muted(self.page),
                                italic=True,
                            )
                        )
                    ]
                    + [ft.DataCell(ft.Text(""))] * max(0, len(self.table.columns) - 1)
                )
            ]
        elif not visible:
            visible = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                self._empty_message,
                                color=ThemeHelper.text_muted(self.page),
                                italic=True,
                            )
                        )
                    ]
                    + [ft.DataCell(ft.Text(""))] * max(0, len(self.table.columns) - 1)
                )
            ]
        self.table.rows = visible
        if self._state.sort_column_index is not None:
            self.table.sort_column_index = self._state.sort_column_index
            self.table.sort_ascending = self._state.sort_ascending

    def _on_search(self, e: ft.ControlEvent) -> None:
        self._state.search_query = (e.control.value or "").strip()
        self._apply()
        try:
            self.table.update()
        except RuntimeError:
            pass

    def on_column_sort(self, e: ft.DataColumnSortEvent) -> None:
        ci = e.column_index
        if self._state.sort_column_index == ci:
            self._state.sort_ascending = not self._state.sort_ascending
        else:
            self._state.sort_column_index = ci
            self._state.sort_ascending = e.ascending
        self._state.sort_numeric = ci in self._numeric_columns
        self._apply()
        try:
            self.table.update()
        except RuntimeError:
            pass

    def wire_sortable_columns(self) -> None:
        """Attach ``on_sort`` to every column that does not already define it."""
        new_cols: list[ft.DataColumn] = []
        for i, col in enumerate(self.table.columns):
            if col.on_sort is None:
                new_cols.append(
                    ft.DataColumn(
                        col.label,
                        tooltip=col.tooltip,
                        numeric=col.numeric,
                        on_sort=self.on_column_sort,
                    )
                )
            else:
                new_cols.append(col)
            if col.numeric:
                self._numeric_columns.add(i)
        self.table.columns = new_cols

    def refresh_theme(self) -> None:
        InputStyles.apply_text_field(self.page, self.search_field)
        kwargs = ThemeHelper.results_summary_data_table_kwargs(self.page)
        for key, val in kwargs.items():
            setattr(self.table, key, val)
        self._wrap.border = ft.border.all(1, ThemeHelper.border_default(self.page))
