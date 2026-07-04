# Stock Analyzer

Local stock research desktop app built with **Flet 0.85.1**. Ingest Yahoo Finance data into SQLite, run CANSLIM scoring, strategy backtests, and optimization tools.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Tabs

| Tab | Purpose |
|-----|---------|
| Dashboard | Market overview, index chart, movers, news feed |
| Portfolio | Holdings, trades, performance, rule-based insights |
| Watchlists | Named watchlists, quotes, focus-symbol ingest |
| Stock Detail | Quote, chart, fundamentals, news, CANSLIM, insights |
| Compare | Normalized multi-ticker performance |
| Strategy Backtests | General, hybrid, technical, CANSLIM backtest |
| Leaderboard | IBD-style composite rankings, drill-down |
| Optimization | Focused momentum grid, volatility/ATR |
| Live | Finnhub stream + intraday chart |
| Data Management | Symbol registry, ingest, maintenance |
| Settings | Paths, ingest, intraday, backtest, LM Studio, theme, logging |

See [`docs/symbol_pools.md`](docs/symbol_pools.md) for focus vs research universe behavior and ingest scopes.

## Layout

- `src/` — Flet UI (`views/`, `services/`, `analysis/`, `patterns/`, `news_sources/`, `ai/`)
- `apps/desktop/` — Flet entry shim
- `scripts/` — CLI helpers (ingest, daily scan, benchmarks)
- `docs/canslim_rule_spec.md` — Versioned CANSLIM/O'Neil thresholds
- `docs/leaderboard_methodology.md` — Leaderboard scoring methodology
- `data/settings.json` — persisted UI settings (created on first save)

## Market intelligence (breadth & depth)

| Capability | Module / entry |
|------------|----------------|
| Capability gap scorecard | [`docs/capability_scorecard.md`](docs/capability_scorecard.md) |
| Data quality per ingest | `src/analysis/data_quality.py` |
| Market regime & sector context | `src/analysis/market_context.py` |
| Ranked guidance (Monitor → High Priority) | `src/analysis/guidance.py` |
| Portfolio templates (growth / balanced / defensive) | `src/analysis/portfolio_intelligence.py` |
| Daily scan + alerts + digest | `src/services/daily_monitor.py`, `scripts/run_daily_scan.py` |
| Backtest slippage & calibration | `src/analysis/signal_quality.py`, Settings → Backtest defaults |

**Daily scan (CLI):**

```bash
python scripts/run_daily_scan.py
```

**Data ingest (CLI):**

```bash
python data_ingest.py
```

Dashboard shows **market regime** and **high-priority guidance count** after the first scan.

## Performance

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for tuning on multi-core / high-RAM machines and `scripts/perf_benchmark.py` for timings.

## Database locking

The app uses SQLite **WAL mode**, a **30s busy timeout**, and **short-lived connections** (always closed via `db_connection()`). On startup, `ensure_db_ready()` runs a passive WAL checkpoint to recover cleanly after a crash. On normal exit, a passive checkpoint merges WAL pages.

If ingest and analysis run at the same time, you may still see brief waits (by design) rather than immediate errors. A reboot clears OS-level locks from dead processes; WAL recovery handles leftover `-wal` state.
