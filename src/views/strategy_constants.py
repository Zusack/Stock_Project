"""Strategy backtest metadata shown in the Strategy Backtests tab."""

from __future__ import annotations

STRATEGY_INFO: dict[str, dict[str, str]] = {
    "general": {
        "title": "General Rules",
        "summary": (
            "Compares four rule-based return multipliers across your universe: "
            "buy & hold, golden cross (50/200-day simple moving averages), "
            "200-day trend, and RSI dip buying."
        ),
        "strengths": (
            "Simple, transparent rules; useful for screening which style fits your "
            "database; easy to explain to stakeholders."
        ),
        "weaknesses": (
            "Lagging signals; no transaction costs or slippage; can overfit one historical "
            "period; ignores fundamentals and position sizing."
        ),
        "tooltip": (
            "Compare buy & hold, golden cross, 200-day trend, and RSI dip rules. "
            "Best for quick style screening."
        ),
        "run_tooltip": (
            "Run all four rule styles on your database and compare which approach "
            "would have performed best over the selected period."
        ),
    },
    "hybrid": {
        "title": "Hybrid (market filter)",
        "summary": (
            "Runs a stock simple moving average trend strategy alone, then again only "
            "when a market index (e.g. ^DJI) is also above its trend — measuring whether "
            "a regime filter improves results."
        ),
        "strengths": (
            "Widely used industry idea (trade with the market tide); may cut drawdowns "
            "in bear markets; separates stock skill from macro timing."
        ),
        "weaknesses": (
            "Whipsaws in sideways markets; misses fast rebounds; results depend heavily "
            "on which index you use as the market proxy."
        ),
        "tooltip": (
            "Stock trend vs trend + market filter. Tests if macro timing helps. "
            "Sensitive to market proxy choice."
        ),
        "run_tooltip": (
            "Backtest a stock simple moving average strategy with and without a market "
            "uptrend filter to see if timing the broad market improves returns."
        ),
    },
    "technical": {
        "title": "Technical (MACD vs Bollinger)",
        "summary": (
            "For each ticker, backtests MACD crossover (trend) and Bollinger Band "
            "mean-reversion, then ranks names by alpha vs buy & hold."
        ),
        "strengths": (
            "Uses familiar institutional indicators; contrasts trend-following vs "
            "mean-reversion; highlights tickers where technical rules beat passive hold."
        ),
        "weaknesses": (
            "Parameter-sensitive; noisy in choppy ranges; no fundamentals; "
            "past indicator edge may not persist out of sample."
        ),
        "tooltip": (
            "MACD vs Bollinger per ticker, ranked by alpha. "
            "Classic technicals — weak in sideways markets."
        ),
        "run_tooltip": (
            "Simulate MACD trend-following and Bollinger mean-reversion per ticker, "
            "then rank symbols by outperformance vs buy-and-hold."
        ),
    },
    "canslim": {
        "title": "CANSLIM backtest",
        "summary": (
            "Simulates William O'Neil–style growth entries: quarterly/annual EPS growth, "
            "new highs, leadership vs market, volume surge, and market uptrend, with "
            "configurable stop-loss and take-profit exits."
        ),
        "strengths": (
            "Disciplined growth framework combining fundamentals and technical triggers; "
            "explicit risk rules; aligns with widely taught CANSLIM methodology."
        ),
        "weaknesses": (
            "Strongest in bull markets; earnings data can lag; rule thresholds are "
            "simplified vs live CANSLIM; survivorship and delisting not fully modeled."
        ),
        "tooltip": (
            "O'Neil-style growth rules with stop/take-profit simulation. "
            "Bull-market friendly; earnings timing matters."
        ),
        "run_tooltip": (
            "Simulate CANSLIM-style entries and rulebook exits (stop loss, take profit, "
            "below 50-day simple moving average, market off) using your stored "
            "fundamentals and prices."
        ),
    },
}
