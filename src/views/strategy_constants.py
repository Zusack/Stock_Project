"""Strategy backtest metadata, tooltips, and user guidance."""

from __future__ import annotations

from src.analysis.strategy_presets import PRESET_CATALOG

STRATEGY_INFO: dict[str, dict[str, str]] = {
    pid: {
        "title": meta["title"],
        "summary": meta["summary"],
        "category": meta.get("category", ""),
        "tooltip": meta["summary"],
        "run_tooltip": f"Run the {meta['title']} preset on your selected tickers and period.",
    }
    for pid, meta in PRESET_CATALOG.items()
}

METRIC_TOOLTIPS: dict[str, str] = {
    "total_return_pct": "Total percentage gain or loss over the backtest period. Does not annualize.",
    "cagr_pct": "Compound Annual Growth Rate — average yearly return if compounded. Above 10% is strong for equities.",
    "sharpe": "Risk-adjusted return (excess return per unit of volatility). Above 1.0 is generally good; below 0 means losing money per unit risk.",
    "sortino": "Like Sharpe but only penalizes downside volatility. Higher is better.",
    "max_drawdown_pct": "Largest peak-to-trough decline in portfolio value. Lower is better; above 25% is painful for most investors.",
    "win_rate_pct": "Percentage of closed trades that were profitable. High win rate alone does not guarantee profitability.",
    "profit_factor": "Gross profits divided by gross losses. Above 1.0 means net profitable; above 1.5 is solid.",
    "exposure_pct": "Fraction of time the strategy held a position (was invested vs cash).",
    "alpha_pct": "Excess return vs the benchmark index over the same period.",
    "beta": "Sensitivity to benchmark moves. Beta near 1 moves with the market; below 1 is less volatile.",
    "trade_count": "Number of round-trip trades executed in the simulation.",
}

CONTROL_TOOLTIPS: dict[str, str] = {
    "period": "Calendar days ending at the latest date in your database (minimum 30).",
    "capital": "Starting cash for the simulation; fractional shares are allowed.",
    "tickers": "Limit to specific symbols, or leave empty to use the universe selector.",
    "universe": "Symbol set when the ticker filter is empty.",
    "benchmark": "Index used for alpha/beta and the benchmark equity overlay (e.g. ^GSPC).",
    "costs": "Apply slippage, spread, and per-trade fees from Settings → Backtest defaults.",
    "compare": "Run 2–5 strategies on the same tickers and period for side-by-side metrics.",
}

GUIDED_EMPTY_STATE = (
    "Welcome to Strategy Backtests\n\n"
    "1. Pick tickers (or choose a universe) in the sidebar.\n"
    "2. Select a preset, build custom rules, or compare strategies.\n"
    "3. Click Run Backtest to see metrics, charts, and trade log.\n\n"
    "Tip: Enable Local AI in Settings for natural-language strategy drafting and results analysis."
)

RESULTS_HELP_TEXT = (
    "How to read backtest results\n\n"
    "Equity curve: Shows portfolio value over time. Compare the strategy line to buy-and-hold "
    "and the benchmark to see if timing rules added value.\n\n"
    "Drawdown: How far the portfolio fell from its prior peak. Deep or long drawdowns mean "
    "the strategy would have been hard to stick with emotionally.\n\n"
    "Sharpe & profit factor: Risk-adjusted quality measures. A strategy can have high returns "
    "but poor Sharpe if volatility was extreme.\n\n"
    "Monthly returns: Green months are gains, red are losses. Look for consistency vs a few "
    "lucky months.\n\n"
    "Important caveats: Backtests use historical data only. Survivorship bias, delisting, "
    "earnings timing, and liquidity are simplified. Past performance does not guarantee "
    "future results. Avoid overfitting by testing multiple periods and out-of-sample tickers."
)

MODE_TOOLTIPS = {
    "presets": "Ready-made strategies (Golden Cross, RSI, MACD, etc.) with editable parameters.",
    "builder": "Build custom entry/exit rules without code, or draft rules with AI.",
    "compare": "Run multiple strategies side-by-side on the same universe.",
}

LLM_STRATEGY_SCHEMA_HINT = (
    'Reply with JSON only matching StrategySpec: {"name":"...", "description":"...", '
    '"entry":{"logic":"and|or","rules":[{"indicator":"RSI","params":{"period":14},'
    '"comparator":"crosses_below","target":{"kind":"value","value":30}}]}, '
    '"exit_rules":{...}, "risk":{"stop_loss_pct":0.08,"take_profit_pct":0.20}, '
    '"apply_costs":true, "engine":"unified"}'
)
