# Golden Cross Rule Specification

Classic dual moving-average trend following. Machine defaults live in
`src.analysis.strategy_presets.golden_cross_preset`. The reference
implementation is `src.analysis.general.strat_golden_cross`.

## Rules

| Rule | Definition |
|------|------------|
| Fast average | 50-day simple moving average of adjusted close (configurable) |
| Slow average | 200-day simple moving average of adjusted close (configurable) |
| Long regime | Fast SMA **is above** slow SMA |
| Flat regime | Fast SMA **is below** slow SMA |
| Entry | Next-bar open after the long regime is true while flat |
| Exit | Next session when the flat regime becomes true (or end of period) |

A “golden cross” is the transition into the long regime; a “death cross” is the
transition into the flat regime. The backtest holds the **regime**, not only the
single crossover bar.

## Why not edge-only `crosses_above` / `crosses_below`?

Edge-only entry stays in cash for any lookback window that begins *after* the
cross already happened. Example: WDC over a recent ~1y window had SMA50 > SMA200
on every day and **zero** fresh crosses → 0% return while buy-and-hold captured
the full run.

State-based `above` / `below` matches how the method is usually backtested and
matches `strat_golden_cross` in `general.py`.

## Warmup

SMA(200) needs ~200 prior trading days. The unified engine loads indicator
warmup history before the lookback window, simulates positions on that longer
slice, then reports equity and trades only inside the requested period.

## Related presets

- **200-Day Trend** uses the same level semantics: long while price is above the
  200-day SMA.
- Event-driven presets (RSI dip, MACD crossover, Bollinger) still use
  `crosses_*` because those strategies are defined by the edge, not the level.
