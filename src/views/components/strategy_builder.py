"""No-code strategy rule builder for Strategy Backtests tab."""

from __future__ import annotations

from typing import Callable

import flet as ft

from src.analysis.strategy_spec import (
    COMPARATORS,
    INDICATOR_DEFAULTS,
    INDICATORS,
    RiskExits,
    Rule,
    RuleGroup,
    RuleTarget,
    StrategySpec,
    spec_to_plain_english,
    validate_spec,
)
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper

_INDICATOR_OPTIONS = [ft.dropdown.Option(i, i) for i in INDICATORS]
_COMPARATOR_OPTIONS = [ft.dropdown.Option(c, c.replace("_", " ").title()) for c in COMPARATORS]


class _RuleRow(ft.Container):
    def __init__(self, page: ft.Page, rule: Rule | None, on_remove: Callable[[], None]):
        super().__init__()
        self.page_ref = page
        rule = rule or Rule(indicator="PRICE", comparator="above", target=RuleTarget(kind="value", value=0))
        self.indicator_dd = InputStyles.dropdown(
            page, label="Indicator", width=160, value=rule.indicator, options=list(_INDICATOR_OPTIONS),
        )
        self.comparator_dd = InputStyles.dropdown(
            page, label="Comparator", width=140, value=rule.comparator, options=list(_COMPARATOR_OPTIONS),
        )
        self.target_kind = InputStyles.dropdown(
            page, label="Target type", width=120, value=rule.target.kind,
            options=[ft.dropdown.Option("value", "Value"), ft.dropdown.Option("indicator", "Indicator")],
        )
        self.target_value = InputStyles.text_field(
            page, label="Value", width=80, value=str(rule.target.value or ""),
        )
        self.target_indicator = InputStyles.dropdown(
            page, label="Target indicator", width=160, value=rule.target.indicator or "SMA",
            options=list(_INDICATOR_OPTIONS),
        )
        self.period_field = InputStyles.text_field(
            page, label="Period", width=70,
            value=str(rule.params.get("period", INDICATOR_DEFAULTS.get(rule.indicator, {}).get("period", 14))),
        )
        self.remove_btn = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE, on_click=lambda e: on_remove(), tooltip="Remove rule",
        )
        self.content = ft.Row(
            [
                self.indicator_dd, self.period_field, self.comparator_dd,
                self.target_kind, self.target_value, self.target_indicator, self.remove_btn,
            ],
            wrap=True, spacing=8,
        )

    def to_rule(self) -> Rule:
        ind = self.indicator_dd.value or "PRICE"
        params = dict(INDICATOR_DEFAULTS.get(ind, {}))
        try:
            if self.period_field.value:
                params["period"] = int(float(self.period_field.value))
        except ValueError:
            pass
        target_kind = self.target_kind.value or "value"
        if target_kind == "indicator":
            target = RuleTarget(
                kind="indicator",
                indicator=self.target_indicator.value or "SMA",
                params={"period": int(params.get("period", 50))},
            )
        else:
            try:
                val = float(self.target_value.value or 0)
            except ValueError:
                val = 0.0
            target = RuleTarget(kind="value", value=val)
        return Rule(
            indicator=ind,
            params=params,
            comparator=self.comparator_dd.value or "above",
            target=target,
        )


class StrategyBuilder(ft.Container):
    """Row-based rule editor with entry/exit groups and risk exits."""

    def __init__(
        self,
        page: ft.Page,
        *,
        on_change: Callable[[], None] | None = None,
        on_ai_request: Callable[[str], None] | None = None,
    ):
        super().__init__(expand=True)
        self.page_ref = page
        self._on_change = on_change
        self._on_ai_request = on_ai_request
        self._entry_rows: list[_RuleRow] = []
        self._exit_rows: list[_RuleRow] = []

        self.name_field = InputStyles.text_field(
            page, label="Strategy name", value="My Strategy", expand=True,
            tooltip="Name for saving and comparing this custom strategy.",
        )
        self.entry_logic = InputStyles.dropdown(
            page, label="Entry logic", width=120, value="and",
            options=[ft.dropdown.Option("and", "AND"), ft.dropdown.Option("or", "OR")],
        )
        self.exit_logic = InputStyles.dropdown(
            page, label="Exit logic", width=120, value="or",
            options=[ft.dropdown.Option("and", "AND"), ft.dropdown.Option("or", "OR")],
        )
        self.stop_loss = InputStyles.text_field(page, label="Stop loss %", width=100, value="8", tooltip="e.g. 8 = 8% loss")
        self.take_profit = InputStyles.text_field(page, label="Take profit %", width=100, value="20")
        self.trailing_stop = InputStyles.text_field(page, label="Trailing stop %", width=110, value="")
        self.max_hold = InputStyles.text_field(page, label="Max hold (days)", width=110, value="")
        self.apply_costs = ft.Switch(label="Apply transaction costs", value=True)
        self.preview_text = ft.Text(
            "", size=12, italic=True, color=ThemeHelper.text_muted(page), selectable=True,
        )
        self.validation_text = ft.Text("", size=11, color=ThemeHelper.chart_named(page, "loss"))
        self.ai_prompt = InputStyles.text_field(
            page,
            label="Describe strategy in plain English (AI)",
            hint_text="e.g. Buy when RSI is below 30 and price is above the 200-day average",
            expand=True,
            multiline=True,
            min_lines=2,
            max_lines=4,
            tooltip="Requires Local AI enabled. The model will draft rules for you to review.",
        )
        self._entry_host = ft.Column(spacing=6)
        self._exit_host = ft.Column(spacing=6)

        self.content = ft.Column(
            [
                self.name_field,
                ft.Text("Entry rules", weight=ft.FontWeight.W_600, size=13),
                self.entry_logic,
                self._entry_host,
                ft.TextButton("Add entry rule", icon=ft.Icons.ADD, on_click=lambda e: self._add_entry()),
                ft.Divider(),
                ft.Text("Exit rules (optional)", weight=ft.FontWeight.W_600, size=13),
                self.exit_logic,
                self._exit_host,
                ft.TextButton("Add exit rule", icon=ft.Icons.ADD, on_click=lambda e: self._add_exit()),
                ft.Divider(),
                ft.Text("Risk exits", weight=ft.FontWeight.W_600, size=13),
                ft.Row([self.stop_loss, self.take_profit, self.trailing_stop, self.max_hold], wrap=True),
                self.apply_costs,
                self.preview_text,
                self.validation_text,
                ft.Row(
                    [
                        self.ai_prompt,
                        ft.ElevatedButton(
                            "Draft with AI",
                            icon=ft.Icons.AUTO_AWESOME,
                            style=ButtonStyles.primary(),
                            on_click=self._on_draft_ai,
                        ),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.END,
                ),
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )
        self._add_entry()

    def _refresh_preview(self) -> None:
        spec = self.build_spec()
        self.preview_text.value = spec_to_plain_english(spec)
        errors = validate_spec(spec)
        self.validation_text.value = "; ".join(errors) if errors else ""
        if self._on_change:
            self._on_change()
        try:
            self.preview_text.update()
            self.validation_text.update()
        except RuntimeError:
            pass

    def _add_entry(self, rule: Rule | None = None) -> None:
        row_ref: list[_RuleRow] = []

        def on_remove() -> None:
            if row_ref:
                self._remove_entry(row_ref[0])

        row = _RuleRow(self.page_ref, rule, on_remove=on_remove)
        row_ref.append(row)
        self._entry_rows.append(row)
        self._entry_host.controls.append(row)
        self._refresh_preview()

    def _remove_entry(self, row: _RuleRow) -> None:
        if row in self._entry_rows:
            self._entry_rows.remove(row)
            self._entry_host.controls.remove(row)
            self._refresh_preview()

    def _add_exit(self, rule: Rule | None = None) -> None:
        row_ref: list[_RuleRow] = []

        def on_remove() -> None:
            if row_ref:
                self._remove_exit(row_ref[0])

        row = _RuleRow(self.page_ref, rule, on_remove=on_remove)
        row_ref.append(row)
        self._exit_rows.append(row)
        self._exit_host.controls.append(row)
        self._refresh_preview()

    def _remove_exit(self, row: _RuleRow) -> None:
        if row in self._exit_rows:
            self._exit_rows.remove(row)
            self._exit_host.controls.remove(row)
            self._refresh_preview()

    def _parse_pct_field(self, text: str) -> float | None:
        text = (text or "").strip()
        if not text:
            return None
        try:
            v = float(text)
            return v / 100.0 if v > 1 else v
        except ValueError:
            return None

    def build_spec(self) -> StrategySpec:
        entry = RuleGroup(
            logic=self.entry_logic.value or "and",
            rules=[r.to_rule() for r in self._entry_rows],
        )
        exit_rules = None
        if self._exit_rows:
            exit_rules = RuleGroup(
                logic=self.exit_logic.value or "or",
                rules=[r.to_rule() for r in self._exit_rows],
            )
        risk = RiskExits(
            stop_loss_pct=self._parse_pct_field(self.stop_loss.value),
            take_profit_pct=self._parse_pct_field(self.take_profit.value),
            trailing_stop_pct=self._parse_pct_field(self.trailing_stop.value),
            max_holding_days=int(self.max_hold.value) if (self.max_hold.value or "").strip().isdigit() else None,
        )
        return StrategySpec(
            name=self.name_field.value or "Custom Strategy",
            description=spec_to_plain_english(
                StrategySpec(name="", entry=entry, exit_rules=exit_rules, risk=risk)
            ),
            entry=entry,
            exit_rules=exit_rules,
            risk=risk,
            apply_costs=self.apply_costs.value,
            engine="unified",
        )

    def load_spec(self, spec: StrategySpec) -> None:
        self.name_field.value = spec.name
        self.entry_logic.value = spec.entry.logic
        self.exit_logic.value = (spec.exit_rules.logic if spec.exit_rules else "or")
        self.apply_costs.value = spec.apply_costs
        if spec.risk.stop_loss_pct is not None:
            self.stop_loss.value = str(spec.risk.stop_loss_pct * 100)
        if spec.risk.take_profit_pct is not None:
            self.take_profit.value = str(spec.risk.take_profit_pct * 100)
        self._entry_rows.clear()
        self._entry_host.controls.clear()
        self._exit_rows.clear()
        self._exit_host.controls.clear()
        for rule in spec.entry.rules:
            self._add_entry(rule)
        if not spec.entry.rules:
            self._add_entry()
        if spec.exit_rules:
            for rule in spec.exit_rules.rules:
                self._add_exit(rule)
        self._refresh_preview()

    def _on_draft_ai(self, e) -> None:
        if self._on_ai_request and self.ai_prompt.value:
            self._on_ai_request(self.ai_prompt.value.strip())
