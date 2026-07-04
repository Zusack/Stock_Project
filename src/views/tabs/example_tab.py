"""
Example sub-tab. Use as a starting point for any view embedded inside another
view's tab strip (e.g. multiple analytical views inside an Analysis view).

It is a plain ft.Column — sub-tabs don't need to subclass BaseView because
their parent already takes care of pubsub and scheduling.
"""
from __future__ import annotations

import flet as ft

from src.views.components.cards import SelectableMetricCard
from src.views.components.layouts import SectionHeader
from src.views.theme import ThemeHelper


class ExampleTab(ft.Column):
    """Minimal sub-tab body showing a section header and a row of metric cards."""

    def __init__(self, page: ft.Page):
        super().__init__(expand=True, scroll=ft.ScrollMode.AUTO, spacing=12)
        self.page_ref = page

        self.metric_a = SelectableMetricCard(page, title="Metric A", value="—", subtitle="placeholder")
        self.metric_b = SelectableMetricCard(page, title="Metric B", value="—", subtitle="placeholder", accent="teal")
        self.metric_c = SelectableMetricCard(page, title="Metric C", value="—", subtitle="placeholder", accent="amber")

        self.controls = [
            SectionHeader("Example metrics", page_ref=page, icon=ft.Icons.QUERY_STATS),
            ft.Row([self.metric_a, self.metric_b, self.metric_c], spacing=12, wrap=True),
            ft.Container(
                content=ft.Text(
                    "Replace this body with your own visualizations, tables, or controls.",
                    color=ThemeHelper.text_muted(page),
                    size=12,
                ),
                padding=12,
            ),
        ]
