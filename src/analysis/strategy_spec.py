"""Strategy specification model for unified backtesting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

INDICATORS = (
    "SMA",
    "EMA",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "BOLLINGER_UPPER",
    "BOLLINGER_LOWER",
    "BOLLINGER_MID",
    "ATR",
    "ROC",
    "VOLUME_AVG",
    "PRICE",
    "HIGH_52W",
    "LOW_52W",
    "MARKET_PRICE",
    "MARKET_SMA",
)

COMPARATORS = (
    "above",
    "below",
    "crosses_above",
    "crosses_below",
    "within_pct",
)

INDICATOR_DEFAULTS: dict[str, dict[str, float | int]] = {
    "SMA": {"period": 50},
    "EMA": {"period": 20},
    "RSI": {"period": 14},
    "MACD": {"fast": 12, "slow": 26},
    "MACD_SIGNAL": {"fast": 12, "slow": 26, "signal": 9},
    "BOLLINGER_UPPER": {"period": 20, "num_std": 2},
    "BOLLINGER_LOWER": {"period": 20, "num_std": 2},
    "BOLLINGER_MID": {"period": 20, "num_std": 2},
    "ATR": {"period": 14},
    "ROC": {"period": 10},
    "VOLUME_AVG": {"period": 50},
    "PRICE": {},
    "HIGH_52W": {"period": 252},
    "LOW_52W": {"period": 252},
    "MARKET_PRICE": {},
    "MARKET_SMA": {"period": 50},
}


@dataclass
class RuleTarget:
    """Compare against a numeric value or another indicator series."""

    kind: str = "value"  # value | indicator
    value: float | None = None
    indicator: str | None = None
    params: dict[str, float | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "indicator": self.indicator,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RuleTarget:
        if not data:
            return cls()
        return cls(
            kind=str(data.get("kind") or "value"),
            value=data.get("value"),
            indicator=data.get("indicator"),
            params=dict(data.get("params") or {}),
        )


@dataclass
class Rule:
    indicator: str
    params: dict[str, float | int] = field(default_factory=dict)
    comparator: str = "above"
    target: RuleTarget = field(default_factory=RuleTarget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator,
            "params": dict(self.params),
            "comparator": self.comparator,
            "target": self.target.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        indicator = str(data.get("indicator") or "PRICE").upper()
        params = dict(data.get("params") or INDICATOR_DEFAULTS.get(indicator, {}))
        return cls(
            indicator=indicator,
            params=params,
            comparator=str(data.get("comparator") or "above").lower(),
            target=RuleTarget.from_dict(data.get("target")),
        )


@dataclass
class RuleGroup:
    logic: str = "and"  # and | or
    rules: list[Rule] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "logic": self.logic,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RuleGroup:
        if not data:
            return cls()
        rules = [Rule.from_dict(r) for r in (data.get("rules") or [])]
        return cls(logic=str(data.get("logic") or "and").lower(), rules=rules)


@dataclass
class RiskExits:
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    max_holding_days: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RiskExits:
        if not data:
            return cls()
        return cls(
            stop_loss_pct=data.get("stop_loss_pct"),
            take_profit_pct=data.get("take_profit_pct"),
            trailing_stop_pct=data.get("trailing_stop_pct"),
            max_holding_days=data.get("max_holding_days"),
        )


@dataclass
class StrategySpec:
    name: str
    description: str = ""
    entry: RuleGroup = field(default_factory=RuleGroup)
    exit_rules: RuleGroup | None = None
    risk: RiskExits = field(default_factory=RiskExits)
    apply_costs: bool = True
    market_ticker: str | None = None
    preset_id: str | None = None
    engine: str = "unified"  # unified | canslim | buy_hold

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "entry": self.entry.to_dict(),
            "exit_rules": self.exit_rules.to_dict() if self.exit_rules else None,
            "risk": self.risk.to_dict(),
            "apply_costs": self.apply_costs,
            "market_ticker": self.market_ticker,
            "preset_id": self.preset_id,
            "engine": self.engine,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategySpec:
        exit_raw = data.get("exit_rules")
        return cls(
            name=str(data.get("name") or "Custom Strategy"),
            description=str(data.get("description") or ""),
            entry=RuleGroup.from_dict(data.get("entry")),
            exit_rules=RuleGroup.from_dict(exit_raw) if exit_raw else None,
            risk=RiskExits.from_dict(data.get("risk")),
            apply_costs=bool(data.get("apply_costs", True)),
            market_ticker=data.get("market_ticker"),
            preset_id=data.get("preset_id"),
            engine=str(data.get("engine") or "unified"),
        )

    @classmethod
    def from_json(cls, text: str) -> StrategySpec:
        return cls.from_dict(json.loads(text))


def validate_spec(spec: StrategySpec) -> list[str]:
    """Return human-readable validation errors (empty if valid)."""
    errors: list[str] = []
    if not spec.name.strip():
        errors.append("Strategy name is required.")
    if spec.engine == "unified" and not spec.entry.rules:
        errors.append("At least one entry rule is required.")
    for group_name, group in (("entry", spec.entry), ("exit", spec.exit_rules)):
        if group is None:
            continue
        if group.logic not in ("and", "or"):
            errors.append(f"{group_name} logic must be 'and' or 'or'.")
        for i, rule in enumerate(group.rules):
            if rule.indicator not in INDICATORS:
                errors.append(f"{group_name} rule {i + 1}: unknown indicator '{rule.indicator}'.")
            if rule.comparator not in COMPARATORS:
                errors.append(f"{group_name} rule {i + 1}: unknown comparator '{rule.comparator}'.")
            if rule.target.kind == "indicator" and rule.target.indicator not in INDICATORS:
                errors.append(
                    f"{group_name} rule {i + 1}: unknown target indicator '{rule.target.indicator}'."
                )
            if rule.target.kind == "value" and rule.target.value is None:
                if rule.comparator not in ("crosses_above", "crosses_below"):
                    errors.append(f"{group_name} rule {i + 1}: numeric target value is required.")
    if spec.risk.stop_loss_pct is not None and spec.risk.stop_loss_pct <= 0:
        errors.append("Stop loss must be positive.")
    if spec.risk.take_profit_pct is not None and spec.risk.take_profit_pct <= 0:
        errors.append("Take profit must be positive.")
    return errors


def rule_to_plain_english(rule: Rule) -> str:
    """Single rule as a readable phrase."""
    params = rule.params or INDICATOR_DEFAULTS.get(rule.indicator, {})
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    ind_label = f"{rule.indicator}({param_str})" if param_str else rule.indicator
    if rule.target.kind == "indicator":
        t_params = rule.target.params or INDICATOR_DEFAULTS.get(rule.target.indicator or "", {})
        t_str = ", ".join(f"{k}={v}" for k, v in t_params.items())
        target_label = (
            f"{rule.target.indicator}({t_str})" if t_str else str(rule.target.indicator)
        )
    elif rule.comparator == "within_pct":
        target_label = f"{rule.target.value}%"
    else:
        target_label = str(rule.target.value)
    comp_map = {
        "above": "is above",
        "below": "is below",
        "crosses_above": "crosses above",
        "crosses_below": "crosses below",
        "within_pct": "is within",
    }
    comp = comp_map.get(rule.comparator, rule.comparator)
    return f"{ind_label} {comp} {target_label}"


def spec_to_plain_english(spec: StrategySpec) -> str:
    """Full strategy as readable sentences."""
    if spec.engine == "buy_hold":
        return f"{spec.name}: buy and hold for the entire period."
    if spec.engine == "canslim":
        return f"{spec.name}: CANSLM growth rules with configured stop/take-profit exits."
    if not spec.entry.rules:
        return f"{spec.name}: no entry rules defined."
    joiner = " AND " if spec.entry.logic == "and" else " OR "
    entry_text = joiner.join(rule_to_plain_english(r) for r in spec.entry.rules)
    parts = [f"Buy when {entry_text}."]
    if spec.exit_rules and spec.exit_rules.rules:
        ex_join = " AND " if spec.exit_rules.logic == "and" else " OR "
        exit_text = ex_join.join(rule_to_plain_english(r) for r in spec.exit_rules.rules)
        parts.append(f"Sell when {exit_text}.")
    risk_bits: list[str] = []
    if spec.risk.stop_loss_pct:
        risk_bits.append(f"stop loss {spec.risk.stop_loss_pct:.0%}")
    if spec.risk.take_profit_pct:
        risk_bits.append(f"take profit {spec.risk.take_profit_pct:.0%}")
    if spec.risk.trailing_stop_pct:
        risk_bits.append(f"trailing stop {spec.risk.trailing_stop_pct:.0%}")
    if spec.risk.max_holding_days:
        risk_bits.append(f"max hold {spec.risk.max_holding_days} days")
    if risk_bits:
        parts.append("Risk exits: " + ", ".join(risk_bits) + ".")
    return " ".join(parts)


def parse_spec_from_llm_json(text: str) -> tuple[StrategySpec | None, str | None]:
    """Parse LLM output into StrategySpec; returns (spec, error)."""
    from src.analysis.ai.narrative import parse_narrative_json

    payload = parse_narrative_json(text)
    if payload is None:
        try:
            payload = json.loads(text.strip())
        except json.JSONDecodeError:
            return None, "Could not parse strategy JSON from model output."
    if not isinstance(payload, dict):
        return None, "Strategy JSON must be an object."
    try:
        spec = StrategySpec.from_dict(payload)
    except (TypeError, ValueError) as exc:
        return None, f"Invalid strategy structure: {exc}"
    errors = validate_spec(spec)
    if errors:
        return None, "; ".join(errors)
    return spec, None
