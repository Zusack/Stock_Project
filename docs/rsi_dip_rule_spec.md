# RSI Dip Buy Rule Specification

Classic RSI mean-reversion dip buy. Machine defaults live in
`src.analysis.strategy_presets.rsi_dip_preset`. The reference implementation is
`src.analysis.general.strat_rsi_dip`.

## Rules

| Rule | Definition |
|------|------------|
| RSI | Wilder-style RSI on adjusted close (default period 14) |
| Buy threshold | RSI **is below** 30 (configurable `buy_below`) |
| Sell threshold | RSI **is above** 50 (configurable `sell_above`) |
| Entry | Next-bar open while flat and RSI is below the buy threshold |
| Exit | When RSI is above the sell threshold (or end of period) |
| Hysteresis | Between 30 and 50, keep the prior regime (stay long if long, stay flat if flat) |

## Why level thresholds instead of `crosses_*`?

Edge-only `crosses_below` stays flat for any lookback that begins *already*
oversold (RSI already under 30, no fresh cross in-window). Level `below` /
`above` matches `strat_rsi_dip` and the usual “buy dips / sell recoveries”
backtest.

## When 0% is correct

If RSI never reaches the buy threshold in the window, the strategy correctly
takes **no trades** and returns **0%** (cash). Example: WDC over a recent ~1y
window had RSI min ≈ 35 — a strong uptrend with no classic oversold dip — so
RSI Dip Buy stays flat while buy-and-hold and Golden Cross (already in an
uptrend regime) can post large gains.

That is a property of the method on that tape, not a silent engine failure.

## Warmup

RSI(14) needs a short prior history. The unified engine loads indicator warmup
before the lookback window (same path as Golden Cross).

## Parameters

Strategy Backtest UI can override:

- `rsi_period` (default 14)
- `buy_below` (default 30)
- `sell_above` (default 50)

Raising `buy_below` (e.g. 40) produces more entries on strong trends; that is a
different strategy than the classic 30/50 dip buy.
