"""Versioned CANSLIM / O'Neil rule thresholds for signals, backtests, and leaderboard."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

RULE_SET_VERSION = "oneil_v1"
SCORE_VERSION_V1 = "v1"
SCORE_VERSION_V2 = "v2"


@dataclass(frozen=True)
class EarningsRules:
    c_yoy_threshold: float = 0.20
    c_valid_days: int = 63
    a_valid_days: int = 126
    min_quarters: int = 4
    c_proxy_6m_return: float = 0.20
    a_proxy_1y_return: float = 0.0


@dataclass(frozen=True)
class TechnicalRules:
    near_high_pct: float = 0.85
    volume_surge_mult: float = 1.5
    breakout_volume_mult: float = 1.40
    rs_lookback_days: int = 126
    sma50_days: int = 50
    sma10w_days: int = 50


@dataclass(frozen=True)
class CupWithHandleRules:
    cup_min_weeks: int = 7
    cup_max_weeks: int = 65
    cup_min_depth_pct: float = 0.12
    cup_max_depth_pct: float = 0.50
    handle_min_days: int = 5
    handle_max_depth_pct: float = 0.20
    handle_max_pct_below_cup_high: float = 0.15
    pivot_buffer_pct: float = 0.001
    breakout_volume_mult: float = 1.40


@dataclass(frozen=True)
class EntryRules:
    require_pattern: bool = False
    require_volume_on_breakout: bool = True
    buy_zone_max_above_pivot_pct: float = 0.05
    allow_follow_on: bool = True


@dataclass(frozen=True)
class ExitRules:
    stop_loss_pct: float = -0.08
    take_profit_pct: float = 0.25
    profit_zone_min_pct: float = 0.20
    profit_zone_max_pct: float = 0.25
    use_below_sma50_exit: bool = True
    use_market_downtrend_exit: bool = True
    round_trip_min_gain_pct: float = 0.10
    round_trip_exit_if_below_entry: bool = True


@dataclass(frozen=True)
class LeaderboardWeights:
    canslim_setup: float = 0.35
    pattern_quality: float = 0.20
    relative_strength: float = 0.20
    volume_profile: float = 0.15
    news_sentiment: float = 0.10


@dataclass(frozen=True)
class LeaderboardWeightsV2:
    """Balanced multi-factor composite weights (v2)."""

    momentum_technical: float = 0.35
    quality: float = 0.20
    value: float = 0.15
    risk_regime: float = 0.20
    sentiment_catalyst: float = 0.10


@dataclass(frozen=True)
class CanslimRuleSet:
    version: str = RULE_SET_VERSION
    earnings: EarningsRules = field(default_factory=EarningsRules)
    technical: TechnicalRules = field(default_factory=TechnicalRules)
    cup_with_handle: CupWithHandleRules = field(default_factory=CupWithHandleRules)
    entry: EntryRules = field(default_factory=EntryRules)
    exit: ExitRules = field(default_factory=ExitRules)
    leaderboard: LeaderboardWeights = field(default_factory=LeaderboardWeights)
    leaderboard_v2: LeaderboardWeightsV2 = field(default_factory=LeaderboardWeightsV2)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def spec_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


_DEFAULT_RULE_SET = CanslimRuleSet()


def get_rule_set(version: str | None = None) -> CanslimRuleSet:
    """Return rule set for version; only one version implemented today."""
    if version is None or version == RULE_SET_VERSION:
        return _DEFAULT_RULE_SET
    if version == "legacy":
        return CanslimRuleSet(
            version="legacy",
            entry=EntryRules(require_pattern=False, require_volume_on_breakout=False),
            exit=ExitRules(use_below_sma50_exit=True, use_market_downtrend_exit=True),
        )
    return _DEFAULT_RULE_SET


def rule_set_from_config(
    *,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    require_pattern: bool | None = None,
    version: str | None = None,
) -> CanslimRuleSet:
    """Build rule set with optional overrides from app settings."""
    base = get_rule_set(version)
    exit_rules = base.exit
    entry_rules = base.entry
    if stop_loss is not None or take_profit is not None:
        exit_rules = ExitRules(
            stop_loss_pct=stop_loss if stop_loss is not None else exit_rules.stop_loss_pct,
            take_profit_pct=take_profit if take_profit is not None else exit_rules.take_profit_pct,
            profit_zone_min_pct=exit_rules.profit_zone_min_pct,
            profit_zone_max_pct=exit_rules.profit_zone_max_pct,
            use_below_sma50_exit=exit_rules.use_below_sma50_exit,
            use_market_downtrend_exit=exit_rules.use_market_downtrend_exit,
            round_trip_min_gain_pct=exit_rules.round_trip_min_gain_pct,
            round_trip_exit_if_below_entry=exit_rules.round_trip_exit_if_below_entry,
        )
    if require_pattern is not None:
        entry_rules = EntryRules(
            require_pattern=require_pattern,
            require_volume_on_breakout=entry_rules.require_volume_on_breakout,
            buy_zone_max_above_pivot_pct=entry_rules.buy_zone_max_above_pivot_pct,
            allow_follow_on=entry_rules.allow_follow_on,
        )
    return CanslimRuleSet(
        version=base.version,
        earnings=base.earnings,
        technical=base.technical,
        cup_with_handle=base.cup_with_handle,
        entry=entry_rules,
        exit=exit_rules,
        leaderboard=base.leaderboard,
        leaderboard_v2=base.leaderboard_v2,
    )
