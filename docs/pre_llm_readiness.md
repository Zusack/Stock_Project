# Pre-LLM Readiness Review

Last updated: 2026-06-28 — end of pre-AI refactor workflow.

## Status: ready for local LLM integration

The refactor workflow is complete. **137 tests pass**, `python main.py` starts cleanly, and AI seams are wired without import cycles.

## Ready capabilities

| Capability | Module | Notes |
|------------|--------|-------|
| Headline sentiment | `src/analysis/ai/adapter.py` | Toggle via Settings → Local AI |
| Ticker context payload | `src/analysis/ai/context.py` | `assemble_ticker_context()` |
| Signal advisor | `src/analysis/ai/advisor.py` | Auto-loads leaderboard/guidance rows |
| Insight persistence | `src/analysis/intelligence_schema.py` | `ai_insights` table |
| LM Studio client | `src/analysis/ai/lmstudio_client.py` | Retry, JSON validation, streaming |
| Settings | Settings → Local AI section | Separate from backtest defaults |

## What was completed in this workflow

### Stability & structure
- Dead code removed (`stock_suite/`, Archive, stale root scripts)
- `ticker_cleanup` migrated to `src/analysis/`
- Circular import fixed: `ai/__init__.py` exports adapter only; lazy imports in advisor/context
- Heavy work moved off UI thread: ingest, leaderboard, portfolio CRUD, watchlist add, settings CSV import, compare/live refresh patterns

### Extractions (Phase 3)
- `src/services/ingest_runner.py` — universe ingest + ETA helpers
- `src/analysis/leaderboard_runner.py` — full leaderboard build
- `src/views/strategy_runners.py` — per-strategy backtest runners
- `src/views/strategy_constants.py`, `settings_sections.py`, `components/symbol_detail_dialog.py`
- `src/services/config_groups.py` — setting key domain documentation

### Shared helpers (Phase 2a)
- `format_utils.py`: `format_number`, `slice_price_df`, `format_timestamp`, `natural_sort_key`
- `ui_helpers.py`: `parse_symbols`, `navigate_to_stock_detail`

### Tests
- AI context/insights, services, stock_config, format_utils, ingest ETA, leaderboard refresh

## Architecture notes

- **Import cycle:** Import `advisor` and `context` from their modules directly, not from `src.analysis.ai`.
- **CANSLIM scores:** Interactive `analyze_canslim` uses 7 letters; leaderboard uses 6 pass columns. Context payload includes `canslim_note`.
- **RuntimeError guards:** `live_view` and similar views intentionally swallow `RuntimeError` on control `.update()` when controls are not yet mounted — this is expected Flet behavior, not silent failure.
- **Config groups:** Use `config_groups.py` when adding new AI-related settings keys.

## Suggested next steps for LLM integration

1. **Prompt templates** — Versioned files under `src/analysis/ai/prompts/` for headline vs narrative vs chat modes.
2. **Context caching** — Compare `context_hash` against `load_latest_ai_insight()` to skip redundant LM Studio calls when data is unchanged.
3. **Streaming UI** — Wire `LMStudioAdapter.stream_text()` into a chat panel or progressive insight display on Stock Detail.
4. **Structured output validation** — Extend the `_validate_analysis_json` pattern to narrative JSON with pydantic or jsonschema.
5. **Rate limiting** — Queue headline analysis during bulk leaderboard scoring to avoid saturating local GPU.
6. **Observability** — Log provider, model, latency, and token counts per `ai_insights` row for tuning.

## Optional follow-ups (non-blocking)

These do not block LLM work:

- Extract the Data Ingest maintenance dialog (~190 lines) into `src/views/components/db_maintenance_dialog.py`.
- Further shrink `leaderboard_view.py` and `settings_view.py` by moving cache UI and settings section builders.
- Split `StockConfig` into nested accessor objects using `config_groups.py` keys.
- Route remaining chart tooltip currency formatting through `format_utils.format_currency`.
- Add debug logging to `news_feed.open_external_url` fallback path (currently falls back to `webbrowser` silently).

## Verification checklist

```bash
python -m pytest tests/ -q          # expect 139+ passing
python main.py                      # app window opens
```

Manual smoke test before LLM work:
1. Enable LM Studio in Settings → Local AI (host, model, toggle on).
2. Run Data Ingest or refresh Leaderboard rankings.
3. Open Stock Detail for a scored ticker — AI insight should reference real guidance, not "No guidance data".
4. Toggle headline AI on Dashboard/news — sentiment badges should appear when LM Studio is reachable.
