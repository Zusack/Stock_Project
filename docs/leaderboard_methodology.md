# Leaderboard Methodology

The **Leaderboard** tab provides an IBD-style *proxy* ranking using only free data in your local database. It does not replicate proprietary IBD RS ratings or proprietary list membership.

## Composite score versions

### v2 (default) — balanced multi-factor

| Bucket | Weight | Inputs |
|--------|--------|--------|
| Momentum / Technical | 35% | RS, pattern quality, volume, near-highs, CANSLIM pass count (percentile-ranked vs universe) |
| Quality | 20% | ROE, profit margins, debt/equity |
| Value | 15% | PEG, P/E vs sector peers |
| Risk / Regime | 20% | Volatility, market regime alignment, risk flags |
| Sentiment / Catalyst | 10% | Headlines (bounded; optional local LLM via LM Studio) |

Scores are percentile-normalized across the scored universe so tickers compare fairly in the same run. A **data confidence** factor pulls uncertain rows toward neutral when fundamentals or news are sparse.

### v1 (legacy)

| Component | Weight |
|-----------|--------|
| CANSLIM setup | 35% (n/6 on C/A/N/S/L/M) |
| Pattern quality | 20% |
| Relative strength | 20% |
| Volume profile | 15% |
| News sentiment | 10% |

Toggle v1/v2 in **Settings → Leaderboard composite version**.

## Action scores

| Column | Use |
|--------|-----|
| **Buy Ready** (0–100) | Sort when deploying cash — rewards setup, momentum, quality, regime |
| **Sell Press** (0–100) | Sort when trimming holdings — trend breaks, weak RS, negative news |

## Letter grades

**Value**, **Quality**, **Momentum**, and **Risk Gr** show A–F ratings derived from each factor bucket (0–100).

## UI columns

| Column | Meaning |
|--------|---------|
| Rank | Position within the current segment |
| Symbol | Ticker |
| Industry | From `stock_profiles` |
| Price | Last adjusted close |
| Composite | Weighted 0–100 (v2 default) |
| Buy Ready / Sell Press | Action scores |
| Value / Quality / Momentum / Risk Gr | Letter grades |
| Conf | Data confidence (0–100 display) |
| CANSLIM | Pass count on C/A/N/S/L/M as `n/6` |
| RS vs Mkt | Six-month stock minus market (%) |
| Vol vs 50d | Latest volume / 50-day average |
| % of 52w High | Price vs 52-week high |
| Setup / Pattern | CANSLIM setup and cup-with-handle |
| Risk | Below 50-day simple moving average, weak market, or far from highs |

## Segments

- **All scored** — full universe by composite
- **Candidates** — setup pass, by composite
- **Buy Ready** — by buy readiness
- **Sell Press** — by sell pressure
- **Watchlist** — focus watchlist
- **Breakouts** — setup + pattern
- **Risk Flags** — flagged names

## Future: LM Studio

Enable **LM Studio** in Settings for richer headline analysis. Sentiment contribution is capped so LLM output cannot dominate the composite alone.

## Data requirements

Run **Research Universe** ingest for `stock_history`, `fundamentals`, `stock_profiles`, and optional `stock_news`.
