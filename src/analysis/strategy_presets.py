"""Built-in strategy presets as editable StrategySpec objects."""

from __future__ import annotations

from src.analysis.strategy_spec import RiskExits, Rule, RuleGroup, RuleTarget, StrategySpec

PRESET_CATALOG: dict[str, dict[str, str]] = {
    "golden_cross": {
        "title": "Golden Cross",
        "summary": "Stay long while the 50-day SMA is above the 200-day SMA; exit when it falls below (death cross regime).",
        "category": "trend",
    },
    "trend_200": {
        "title": "200-Day Trend",
        "summary": "Stay invested while price is above the 200-day moving average.",
        "category": "trend",
    },
    "rsi_dip": {
        "title": "RSI Dip Buy",
        "summary": "Buy while RSI is below 30 (oversold); sell when RSI rises above 50.",
        "category": "mean_reversion",
    },
    "macd_cross": {
        "title": "MACD Crossover",
        "summary": "Buy when MACD crosses above its signal line; sell on bearish cross.",
        "category": "momentum",
    },
    "bollinger_reversion": {
        "title": "Bollinger Reversion",
        "summary": "Buy at the lower band; sell at the upper band.",
        "category": "mean_reversion",
    },
    "market_filtered": {
        "title": "Market-Filtered Trend",
        "summary": "Stock trend strategy active only when the market index is above its SMA.",
        "category": "regime",
    },
    "buy_hold": {
        "title": "Buy & Hold",
        "summary": "Passive benchmark — buy at period start and hold.",
        "category": "benchmark",
    },
    "canslim": {
        "title": "CANSLM",
        "summary": "O'Neil-style growth entries with stop-loss and take-profit exits.",
        "category": "fundamental",
    },
}


def golden_cross_preset(
    fast: int = 50,
    slow: int = 200,
) -> StrategySpec:
    # Level (above/below), not edge-only crosses: matches classic SMA trend
    # following and the app's strat_golden_cross. Cross-only entry misses any
    # window that starts already in a golden-cross regime.
    return StrategySpec(
        name="Golden Cross",
        description=PRESET_CATALOG["golden_cross"]["summary"],
        preset_id="golden_cross",
        entry=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="SMA",
                    params={"period": fast},
                    comparator="above",
                    target=RuleTarget(kind="indicator", indicator="SMA", params={"period": slow}),
                ),
            ],
        ),
        exit_rules=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="SMA",
                    params={"period": fast},
                    comparator="below",
                    target=RuleTarget(kind="indicator", indicator="SMA", params={"period": slow}),
                ),
            ],
        ),
    )


def trend_200_preset(period: int = 200) -> StrategySpec:
    return StrategySpec(
        name="200-Day Trend",
        description=PRESET_CATALOG["trend_200"]["summary"],
        preset_id="trend_200",
        entry=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="PRICE",
                    comparator="above",
                    target=RuleTarget(kind="indicator", indicator="SMA", params={"period": period}),
                ),
            ],
        ),
        exit_rules=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="PRICE",
                    comparator="below",
                    target=RuleTarget(kind="indicator", indicator="SMA", params={"period": period}),
                ),
            ],
        ),
    )


def rsi_dip_preset(
    rsi_period: int = 14,
    buy_below: float = 30.0,
    sell_above: float = 50.0,
) -> StrategySpec:
    # Level thresholds with hysteresis (long while oversold, flat once
    # recovered), matching strat_rsi_dip. Edge-only crosses miss any window
    # that begins already below the buy threshold.
    return StrategySpec(
        name="RSI Dip Buy",
        description=PRESET_CATALOG["rsi_dip"]["summary"],
        preset_id="rsi_dip",
        entry=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="RSI",
                    params={"period": rsi_period},
                    comparator="below",
                    target=RuleTarget(kind="value", value=buy_below),
                ),
            ],
        ),
        exit_rules=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="RSI",
                    params={"period": rsi_period},
                    comparator="above",
                    target=RuleTarget(kind="value", value=sell_above),
                ),
            ],
        ),
    )


def macd_cross_preset(
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> StrategySpec:
    return StrategySpec(
        name="MACD Crossover",
        description=PRESET_CATALOG["macd_cross"]["summary"],
        preset_id="macd_cross",
        entry=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="MACD",
                    params={"fast": fast, "slow": slow},
                    comparator="crosses_above",
                    target=RuleTarget(
                        kind="indicator",
                        indicator="MACD_SIGNAL",
                        params={"fast": fast, "slow": slow, "signal": signal},
                    ),
                ),
            ],
        ),
        exit_rules=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="MACD",
                    params={"fast": fast, "slow": slow},
                    comparator="crosses_below",
                    target=RuleTarget(
                        kind="indicator",
                        indicator="MACD_SIGNAL",
                        params={"fast": fast, "slow": slow, "signal": signal},
                    ),
                ),
            ],
        ),
    )


def bollinger_reversion_preset(period: int = 20, num_std: float = 2.0) -> StrategySpec:
    params = {"period": period, "num_std": num_std}
    return StrategySpec(
        name="Bollinger Reversion",
        description=PRESET_CATALOG["bollinger_reversion"]["summary"],
        preset_id="bollinger_reversion",
        entry=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="PRICE",
                    comparator="crosses_below",
                    target=RuleTarget(kind="indicator", indicator="BOLLINGER_LOWER", params=params),
                ),
            ],
        ),
        exit_rules=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="PRICE",
                    comparator="crosses_above",
                    target=RuleTarget(kind="indicator", indicator="BOLLINGER_UPPER", params=params),
                ),
            ],
        ),
    )


def market_filtered_preset(
    stock_sma: int = 50,
    market_sma: int = 100,
    market_ticker: str | None = None,
) -> StrategySpec:
    return StrategySpec(
        name="Market-Filtered Trend",
        description=PRESET_CATALOG["market_filtered"]["summary"],
        preset_id="market_filtered",
        market_ticker=market_ticker,
        entry=RuleGroup(
            logic="and",
            rules=[
                Rule(
                    indicator="PRICE",
                    comparator="above",
                    target=RuleTarget(kind="indicator", indicator="SMA", params={"period": stock_sma}),
                ),
                Rule(
                    indicator="MARKET_PRICE",
                    comparator="above",
                    target=RuleTarget(
                        kind="indicator", indicator="MARKET_SMA", params={"period": market_sma},
                    ),
                ),
            ],
        ),
        exit_rules=RuleGroup(
            logic="or",
            rules=[
                Rule(
                    indicator="PRICE",
                    comparator="below",
                    target=RuleTarget(kind="indicator", indicator="SMA", params={"period": stock_sma}),
                ),
                Rule(
                    indicator="MARKET_PRICE",
                    comparator="below",
                    target=RuleTarget(
                        kind="indicator", indicator="MARKET_SMA", params={"period": market_sma},
                    ),
                ),
            ],
        ),
    )


def buy_hold_preset() -> StrategySpec:
    return StrategySpec(
        name="Buy & Hold",
        description=PRESET_CATALOG["buy_hold"]["summary"],
        preset_id="buy_hold",
        engine="buy_hold",
    )


def canslim_preset(
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.25,
) -> StrategySpec:
    return StrategySpec(
        name="CANSLM",
        description=PRESET_CATALOG["canslim"]["summary"],
        preset_id="canslim",
        engine="canslim",
        risk=RiskExits(stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct),
    )


def get_preset(preset_id: str, **kwargs) -> StrategySpec:
    """Factory for preset specs with optional parameter overrides."""
    factories = {
        "golden_cross": golden_cross_preset,
        "trend_200": trend_200_preset,
        "rsi_dip": rsi_dip_preset,
        "macd_cross": macd_cross_preset,
        "bollinger_reversion": bollinger_reversion_preset,
        "market_filtered": market_filtered_preset,
        "buy_hold": buy_hold_preset,
        "canslim": canslim_preset,
    }
    fn = factories.get(preset_id)
    if fn is None:
        raise ValueError(f"Unknown preset: {preset_id}")
    if preset_id == "buy_hold":
        return fn()
    return fn(**kwargs)


def list_preset_ids() -> list[str]:
    return list(PRESET_CATALOG.keys())
