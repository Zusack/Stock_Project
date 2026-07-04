# CANSLIM Rule Specification (`oneil_v1`)

Machine-readable defaults live in `src/analysis/canslim_rulebook.py`. This document maps O'Neil / IBD-style concepts to implementation thresholds.

## Buy rules

| Rule | Threshold | Source concept |
|------|-----------|----------------|
| Quarterly EPS growth (C) | YoY > 20% for 63 days after report | CAN SLIM |
| Annual EPS growth (A) | Positive YoY for 126 days after report | CAN SLIM |
| New highs (N) | Price >= 85% of 52-week high | CAN SLIM |
| Leadership (L) | 6-month return > market 6-month return | CAN SLIM |
| Market (M) | Index above 50-day simple moving average | IBD market direction |
| Supply (S) | Volume > 1.5x 50-day average | Volume surge |
| Cup-with-handle (optional) | Valid base + pivot breakout | How to Make Money in Stocks |
| Buy zone | Up to 5% above pivot | IBD buy range |
| Breakout volume | >= 40% above 50-day avg | O'Neil breakout confirmation |

## Sell rules

| Rule | Threshold | Priority |
|------|-----------|----------|
| Stop loss | -7% to -8% from entry | 1 |
| Take profit | +20% to +25% from entry | 2 |
| Round-trip | Gave up 10%+ gain, now below entry | 3 |
| Below 50-day simple moving average | Close below 50-day line | 4 |
| Market downtrend | Index below 50-day simple moving average | 5 |

## Cup-with-handle geometry

- Cup: 7–65 weeks, depth 12–50%, U-shaped (not V)
- Handle: 5+ days, depth <= 20%, in upper half of cup
- Pivot: Handle high (+ small buffer)
- Breakout: Close above pivot with volume expansion

## Reproducibility

Set `rule_set_version` in backtest results. Hash: `CanslimRuleSet().spec_hash()`.
