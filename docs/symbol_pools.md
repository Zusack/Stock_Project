# Symbol pools: focus watchlist and research universe

The app uses a single SQLite registry (`watchlist_tickers`) with a `pool` column:

| Pool | Max | Ingest | Primary use |
|------|-----|--------|-------------|
| `focus` | 20 (configurable) | Daily — smart update | Guidance, daily scan, default leaderboard scoring |
| `universe` | Unlimited | Weekly — smart update | Backtests, breadth sampling, bulk research |
| Archived (`skip_ingest=1`) | — | Skipped | Dead/delisted symbols kept for audit |

## Rules

- **One row per symbol.** Focus symbols use `pool='focus'` but still participate in universe ingest and backtests.
- **Removing from focus** sets `pool='universe'`; price history is retained.
- **CSV import** (`tickerList.csv`) seeds **universe only**, never focus.
- **Adding to focus** also ensures the symbol exists in the registry (upsert).

## Ingest scopes

```bash
# Focus only (fast, daily)
python scripts/ingest_focus.py

# Full active universe (slow, weekly)
python data_ingest.py --scope universe
```

In the UI:

- **Watchlist** tab → Run focus ingest
- **Research Universe** tab → Start universe ingest

Focus symbols are processed first when running a universe ingest.

## App defaults

- Daily scan and guidance: focus symbols when `daily_scan_scope = focus`
- Leaderboard refresh: scores focus by default; use **Score full universe** for the slow path
- Strategy backtests: universe dropdown when the ticker filter is empty
- Dashboard: Focus N/20 and Universe M counts

## Cron suggestions

```cron
# Daily focus (example 6:30 AM weekdays)
30 6 * * 1-5 cd /path/to/Stock_Project && python scripts/ingest_focus.py

# Weekly universe (example Saturday 11 PM)
0 23 * * 6 cd /path/to/Stock_Project && python data_ingest.py --scope universe
```
