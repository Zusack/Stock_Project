# Capability Scorecard — Stock Analyzer vs Professional-Grade Tools

Scoring: **0** absent · **1** partial · **2** production-ready for personal research use.

| Capability | Current | Target (this plan) | Professional reference |
|------------|---------|-------------------|------------------------|
| **Data ingest & coverage** | 2 | 2 | Bloomberg, FactSet |
| **Price history depth** | 2 | 2 | Daily EOD sufficient for CANSLIM |
| **Fundamentals breadth** | 1 | 2 | Expanded EAV + profiles |
| **Market regime / breadth** | 0 | 2 | IBD market direction, sector rotation |
| **CANSLIM / pattern signals** | 2 | 2 | IBD-style proxy |
| **Multi-strategy backtests** | 2 | 2 | TradingView, Amibroker |
| **Ranked guidance (conservative)** | 1 | 2 | IBD Leaderboard, TC2000 |
| **Explainability** | 0 | 2 | Driver lists per ticker |
| **News intelligence** | 1 | 2 | Headlines + sentiment scoring |
| **Portfolio construction** | 0 | 2 | Risk parity, sector caps |
| **Daily monitoring / alerts** | 0 | 2 | Email/push in pro tools |
| **Signal quality governance** | 0 | 2 | Slippage, calibration metrics |
| **Real-time (sub-daily)** | 0 | 0 | Deferred; daily cadence OK |
| **Broker integration** | 0 | 0 | Out of scope |

## Module map (implemented)

| Module | Role |
|--------|------|
| `src/analysis/intelligence_schema.py` | SQLite tables for guidance, alerts, quality |
| `src/analysis/data_quality.py` | Ingest health metrics per symbol |
| `src/analysis/market_context.py` | Regime, breadth, sector ETF trends |
| `src/analysis/news_signals.py` | Headline sentiment from DB + AI/rule fallback |
| `src/analysis/guidance.py` | Ranked bands: Monitor → High Priority |
| `src/analysis/portfolio_intelligence.py` | Position sizing, sector caps, templates |
| `src/services/daily_monitor.py` | EOD scan, alerts, digest |
| `src/analysis/signal_quality.py` | Slippage/fees, calibration snapshots |

## Remaining gaps (future UI/performance plan)

- Polished dedicated Guidance / Portfolio / Alerts tabs
- Sub-second live quotes and Level-2
- Survivorship-bias-free institutional backtests
- Multi-user cloud sync
