# Performance Tuning Guide

This app is optimized for **multi-core CPU** and **large RAM** hosts. Core analytics stay on CPU (pandas/numpy); use **LM Studio GPU** for AI inference, not in-process CUDA.

## Recommended settings (high-end workstation)

| Setting | Suggested value | Location |
|---------|-----------------|----------|
| Analysis worker count | 8–16 (≤ CPU cores) | Settings → Performance |
| Use parallel processing | On | Settings |
| SQLite cache (MB) | 256–1024 | Settings |
| SQLite mmap (MB) | 256–1024 | Settings |
| Dashboard cache (sec) | 30–60 | Settings |
| Leaderboard auto-refresh | Off (use Refresh button) | Settings |
| Parallel ingest | On | Settings → Ingest |
| Ingest price workers | 4–8 | Settings |
| Ingest fundamentals workers | 2–4 | Settings |
| Yahoo min interval (sec) | 0.5–0.75 | Settings |

## Benchmarks

Run repeatable timings (includes SLA checks and recent DB op telemetry):

```bash
python scripts/perf_benchmark.py --db market_data.db --tickers AAPL,MSFT,NVDA
python scripts/perf_benchmark.py --db market_data.db --skip-backtest
python scripts/perf_benchmark.py --db market_data.db --benchmark-registry
```

Target SLAs (P95 ceilings) are defined in `src/analysis/db_perf.py` (`OPERATION_SLAS`).

## What was optimized

1. **Cup-with-handle** — numpy array scan (no per-bar DataFrame copies).
2. **Leaderboard** — multiprocessing, bulk fundamentals/history preload, segment filter from cache.
3. **CANSLIM backtest** — bulk DB preload + tuned worker pool.
4. **Strategy engines** — `worker_count` respected via `parallel_exec`.
5. **SQLite** — WAL + configurable cache/mmap/temp_store.
6. **Ingest** — atomic run-item claims, multiple fundamentals workers.
7. **UI** — dashboard TTL cache, leaderboard progress throttling, no default tab-open full scan.
8. **Optimization** — filtered/chunked DB loads, numpy momentum backtest, live progress/ETA/results, cancel support.
9. **Database** — batched dead-ticker removal, consolidated `count_summary`, query indexes, bulk price/headline loads, lock-aware UI banners and fetch error surfacing.
10. **Observability** — `DB_PERF` slow-query logging, per-run maintenance summaries, `get_recent_db_ops()` for benchmarks.

## GPU note

An NVIDIA GPU helps **LM Studio** headline sentiment when enabled in Settings. It does not accelerate daily-bar backtests in this codebase.

## Optimization tab (Focused Momentum / Volatility)

| Setting | Suggested value | Notes |
|---------|-----------------|-------|
| Analysis worker count | 8–16 (≤ CPU cores) | Settings → Performance |
| Use parallel processing | On | Required for full-universe runs |
| Ticker filter | Use for exploratory runs | 10 tickers completes in seconds vs hours for full DB |

**Focused Momentum** grid-searches 108 parameter combos per ticker (6 windows × 6 buy × 3 sell thresholds). A full database (~3,000+ tickers) is CPU-heavy even with optimizations; use a ticker filter first, then widen.

**Volatility / ATR** loads OHLCV per ticker; filtered loads avoid scanning the entire `stock_history` table.

The Optimization tab shows live progress (%), ticker counts, ETA, elapsed time, incremental results, and a **Stop** button that preserves partial results.

**Order-of-magnitude runtime** (varies by hardware and bar count):

- 10 filtered tickers, parallel on: under 1 minute
- 500 tickers, 8 workers: tens of minutes to a few hours
- Full universe (~3,000+ tickers), 16 workers: several hours (not days) with progress visible throughout
