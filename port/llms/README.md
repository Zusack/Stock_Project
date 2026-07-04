# LLM Backend Port (`port/llms/`)

A generous, **verbatim** copy of the LLM communication, management, and
execution layer from the LLM Benchmark Evaluator project, staged for reuse in
a new **stock-analysis / research chat** application.

This bundle is intentionally over-inclusive: it contains the core LLM plumbing
plus database, orchestration, and benchmark-specific artifacts so that a
follow-up pass (human or AI) can keep exactly what's useful and delete the
rest. Nothing here has been rewritten — every file is byte-for-byte identical
to the source so its behavior and internal dependencies stay intact.

> Companion bundle: the UI skeleton lives in `port/foundation/`. Where the two
> overlap (e.g. `event_bus.py`, `global_control_service.py`, `theme_settings.py`,
> `ui_health_monitor.py`, `logger_utils.py`, `format_utils.py`), read the
> "Overlap with `port/foundation/`" section below before merging.

---

## Layout & import model

Files are copied under `port/llms/src/` mirroring the original package tree.
Every module uses **absolute imports rooted at `src.`** (e.g.
`from src.backends.types import BackendType`, `from src.database.manager import
DatabaseManager`). That means:

- To use this code with imports **unchanged**, drop the contents of
  `port/llms/src/` into your new project's `src/` package (or add
  `port/llms/` to `sys.path` so `src` resolves).
- If your new project uses a different root package name, do a project-wide
  rename of the `src.` import prefix.

```
port/llms/
├── README.md                          # this file
├── requirements-desktop.reference.txt # parent project's full dep list (reference)
└── src/
    ├── controller.py                  # BenchmarkController orchestration (execution reference)
    ├── backends/                      # <-- CORE: the LLM provider abstraction
    ├── database/                      # persistence (SQLite); optional but handy
    ├── services/                      # engines + app services (mixed relevance)
    └── utils/                         # cross-cutting helpers (mixed relevance)
```

---

## The core you almost certainly want

### `src/backends/` — the LLM provider abstraction (primary deliverable)

A clean, uniform interface over four providers. This is the heart of the port
and is fully self-contained (backend modules only import each other).

| File | Role |
|------|------|
| `types.py` | `BackendType`, `StreamChunk`, `ModelInfo`, `LLMHandle`, `ToolResult`, `CancellableStream` — the normalized types every backend produces/consumes. |
| `base.py` | `LLMBackend` ABC: `connect/disconnect`, `list_available_models`, `load_model`, `chat_stream`, `complete_stream`, `act_with_tools`, `tokenize`, capability queries, error classification. |
| `factory.py` | `BackendFactory.create(BackendType, settings)` / `create_from_string("ollama", settings)`. One place to register backends. |
| `lmstudio_backend.py` | **LM Studio** (uses the `lmstudio` SDK). Full vision + tool-calling support. |
| `ollama_backend.py` | **Ollama** (uses the `ollama` package). |
| `vllm_backend.py` | **vLLM** / any OpenAI-compatible server (uses the `openai` client). |
| `gguf_backend.py`, `gguf_worker.py`, `gguf_metadata.py` | Local **GGUF** via `llama-cpp-python` in a subprocess. Drop if you only use remote servers. |
| `android_gguf_backend.py`, `native_android_gguf_backend.py`, `native_bridge.py` | Android/NDK GGUF runtime. **Almost certainly droppable** for a desktop stock app. |

`factory.py` imports the GGUF/Android backends **lazily** (inside functions), so
you can delete those files and, as long as you never request
`BackendType.GGUF`, nothing breaks. If you want the factory to only advertise
remote backends, trim `_BACKEND_CONSTRUCTORS`.

Minimal remote-only backend set:
```
backends/__init__.py  types.py  base.py  factory.py
backends/lmstudio_backend.py  backends/ollama_backend.py  backends/vllm_backend.py
```
(Trim the GGUF imports from `factory.py` if you delete those files.)

### Backend-relevant utils (`src/utils/`)

| File | Keep? | Why |
|------|-------|-----|
| `metadata_utils.py` | ✅ | `get_all_downloaded_llms(backend, settings)` — model discovery via the factory. Thin, provider-agnostic. |
| `lmstudio_utils.py` | ✅ if using LM Studio | LM Studio-specific helpers. |
| `stream_utils.py` | ✅ | `iterate_stream_with_thermal_checks`, `JobTimeoutError`, `ErrorCallInterrupted` — resilient streaming with cancel/timeout. The thermal hooks are optional; the timeout/cancel logic is broadly useful for chat. |
| `repetition_utils.py` | ✅ | `detect_garbage_response`, `strip_thinking_blocks` — clean up model output (strip `<think>` blocks, catch degenerate loops). Useful for chat. |
| `model_display_label.py` | ✅ | Friendly model naming / provider detection. Imports `BackendType`. |
| `pre_load_check.py` | ➖ | Backend readiness + optional "unload everything first" before a run. Depends on DB + `GlobalControlService`. Useful pattern; trim if not needed. |
| `progress_utils.py` | ➖ | Progress calculator for multi-step runs. Benchmark-oriented. |
| `prompt_utils.py` | ➖ | Loads prompt files from a folder — benchmark prompt catalog. Repurpose or drop. |
| `memory_utils.py` | ➖ | CPU/GPU/RAM/VRAM + power sensors (psutil/NVML). Only needed for hardware monitoring / thermal safety. |
| `system_utils.py` | ➖ | `SystemScanner` (hardware fingerprint / system id). Benchmark fleet feature. |
| `logger_utils.py` | ⚠️ | Rotating logger. **NOTE:** imports `ROOT_DIR` from `src.database.manager`, so it drags in the DB package. `port/foundation/` already ships a DB-free logger — prefer that one and delete this copy. |
| `format_utils.py` | ⚠️ | Duplicate of the foundation copy. Keep one. |

---

## Optional: `src/database/` (persistence)

A self-contained SQLite layer (internal imports are relative: `.core`,
`.schema`, …). `DatabaseManager` is a mixin of repositories.

| File | Role |
|------|------|
| `manager.py` | `DatabaseManager` facade, `ROOT_DIR`, `DEFAULT_DB_FILE`, `BACKEND_SETTINGS_DEFAULTS`, and `get_backend_settings()` — the **glue** that feeds `BackendFactory` its connection settings. |
| `core.py` | Connection, WAL, row factory. |
| `schema.py` | Table definitions + migrations (models, prompts, systems, responses, evaluations, settings, audit profiles). Heavily benchmark-shaped. |
| `settings.py` | `get_setting/set_setting` key-value store. |
| `llms.py`, `prompts.py`, `systems.py`, `audit_profiles.py` | Domain repositories. |
| `results/` | `reader.py` / `writer.py` / `cleanup.py` + `is_real_benchmark_error`, incompatible-response helpers. **Benchmark results storage** — the schema is about respondents/evaluators/ratings, not stock data. |

**For a stock chat app**, the valuable, reusable bits are:
- The `get_setting/set_setting` pattern and `get_backend_settings()` glue.
- `BACKEND_SETTINGS_DEFAULTS` (the canonical default host URLs/keys per backend).

The `llms`/`prompts`/`results`/`systems`/`evaluations` schema is
benchmark-specific — you'll likely design your own tables (articles, tickers,
chat sessions, messages). Keep this package as a **reference** for structure, or
lift only `manager.py` + `settings.py` + `core.py`.

> Note: `port/foundation/` already provides a lightweight JSON `SettingsStore`
> with the same `get_setting/set_setting` API. If you adopt that, you can point
> the backend factory at it instead of SQLite by passing a settings dict into
> `BackendFactory.create(...)`. Decide on **one** settings source to avoid drift.

---

## Optional: `src/services/` (engines & app services)

Mixed relevance. Split into three tiers:

### Tier A — execution plumbing worth keeping
| File | Why |
|------|-----|
| `generation_engine.py` | The **execution engine**: loads a model, runs `chat_stream`/`complete_stream`, handles cancel/timeout, records output. The closest thing to "how do I actually run a prompt through a backend." Note it imports `ScoringEngine`, `event_bus`, `tool_resolver`, DB, and several utils — see deps below. |
| `tool_resolver.py` | Maps tool definitions to Python callables for tool-calling (`act_with_tools`). **Directly reusable** for a stock app: register tools like `get_quote`, `fetch_news`, `get_financials`. Currently imports `ROOT_DIR` from the DB package for custom-tool script loading. |
| `model_pipeline_status.py` | Classifies model readiness/errors. Small. |
| `event_bus.py` | App-wide pub/sub. **Duplicate of the foundation copy** — keep one. |
| `global_control_service.py` | Run/stop/pause/skip signalling used by the engines. The **parent's full version** (with process control) — richer than the trimmed one in `port/foundation/`. If you keep `generation_engine.py`, you likely need this version. |

### Tier B — likely drop for a stock app (benchmark/eval specific)
| File | Why it's here / why you'd drop it |
|------|-----|
| `evaluation_engine.py` | LLM-as-judge scoring of other models' answers. Reusable only if you want a "critic" model; otherwise drop. |
| `scoring_engine.py` | NLP metric scoring (ROUGE/BERTScore/etc.). **`generation_engine.py` imports this**, so keep it if you keep the engine — or stub it out. |
| `audit_engine.py`, `deep_audit_engine.py` | 4-pass VRAM/RAM measurement of models. Pure hardware benchmarking. |
| `thermal_monitor.py` | Temperature/power safety that pauses/stops runs. Needs `memory_utils`. Optional safety layer for long local-GGUF runs; irrelevant for remote APIs. |
| `export_service.py` | Excel/report export of benchmark results. |
| `analytics/` | Leaderboards, proficiency, scatter mappings — benchmark results analytics. Not LLM comms. |
| `theme_settings.py`, `ui_health_monitor.py` | UI concerns — duplicates of `port/foundation/`. Keep the foundation copies. |

### `src/controller.py`
`BenchmarkController` — the top-level orchestrator that wires every engine
together (generation → evaluation → audit) on a worker thread. Included as an
**execution/orchestration reference**. You'll write your own controller for
"user asks a question → fetch news/quotes via tools → stream an answer," but
this shows the threading + `event_bus` + stop-signal pattern to emulate.

---

## Dependency map (top-level `src.` imports)

Use this to figure out the minimal closure when you keep a given file.

```
backends/*            -> backends only (self-contained)

utils/metadata_utils  -> backends
utils/model_display_label -> backends.types
utils/logger_utils    -> database.manager (ROOT_DIR)         # coupling! prefer foundation's logger
utils/pre_load_check  -> database.manager, services.global_control_service,
                         utils.system_utils, backends
utils/progress_utils  -> utils.model_display_label

database/*            -> self-contained (relative imports only)

services/generation_engine -> utils.memory_utils, utils.system_utils,
                              utils.stream_utils, utils.repetition_utils,
                              services.scoring_engine, services.event_bus,
                              services.tool_resolver, backends.types,
                              database.manager
services/evaluation_engine -> services.global_control_service, utils.stream_utils,
                              utils.repetition_utils, services.event_bus, backends.types
services/tool_resolver     -> database.manager (ROOT_DIR)
services/model_pipeline_status -> database.results, utils.model_display_label
services/thermal_monitor   -> database.manager, utils.memory_utils,
                              services.global_control_service, services.event_bus
services/audit_engine      -> utils.memory_utils, utils.system_utils, services.event_bus
services/deep_audit_engine -> utils.memory_utils, services.event_bus,
                              services.global_control_service
services/export_service    -> database.manager (ROOT_DIR), utils.model_display_label

controller.py         -> database.manager, utils.metadata_utils, utils.logger_utils,
                         services.{event_bus, global_control_service, thermal_monitor,
                         generation_engine, evaluation_engine, audit_engine,
                         deep_audit_engine}, backends
```

### Suggested minimal "LLM comms only" closure
If all you want is *talk to LM Studio / Ollama / vLLM and stream chat with tools*:
```
src/backends/            (drop android_* and native_bridge; optionally drop gguf_*)
src/utils/metadata_utils.py
src/utils/stream_utils.py
src/utils/repetition_utils.py
src/utils/model_display_label.py
src/services/event_bus.py            (or reuse port/foundation's)
src/services/tool_resolver.py        (rework ROOT_DIR dependency)
```
`tool_resolver.py` and `logger_utils.py` are the only "core-ish" files that reach
into the DB package (both only for `ROOT_DIR`); swap that for a plain path
constant to fully decouple from `database/`.

---

## Python package dependencies (per backend)

From `requirements-desktop.reference.txt`:

| Backend | pip package(s) |
|---------|----------------|
| LM Studio | `lmstudio==1.5.0` |
| Ollama | `ollama>=0.4.0` |
| vLLM (OpenAI-compatible) | `openai>=1.0.0` |
| GGUF (local) | `llama-cpp-python==0.3.16` (CPU/GPU build flags apply) |
| Hardware monitoring (memory/thermal) | `psutil>=5.9.0` (+ NVML tooling for GPU) |
| Model hub downloads | `huggingface_hub>=0.20.0,<1.0` |

The scoring/evaluation stack (`transformers`, `torch`, `nltk`, `rouge-score`,
`bert-score`) is only needed if you keep `scoring_engine.py` /
`evaluation_engine.py`. For a stock chat app you can almost certainly omit it.

---

## Overlap with `port/foundation/`

These files exist in **both** bundles. Keep one authoritative copy per concern:

| Concern | `port/foundation/` version | `port/llms/` version | Recommendation |
|---------|---------------------------|----------------------|----------------|
| Settings | JSON `SettingsStore` (DB-free) | `database/settings.py` (SQLite) | Pick one settings source; pass a dict into `BackendFactory` either way. |
| Logger | DB-free `logger_utils.py` | `logger_utils.py` (imports `database.manager`) | Prefer the foundation logger; delete this copy. |
| `event_bus.py` | trimmed | full (same behavior) | Either; keep one. |
| `global_control_service.py` | trimmed (tab/loading only) | full (run/stop/pause/skip) | If you keep the engines, use the **full** version here. |
| `theme_settings.py`, `ui_health_monitor.py` | present | duplicates | Keep foundation copies. |
| `format_utils.py` | extended (currency/percent) | original | Keep foundation copy. |

---

## Best-practice notes for adopting this foundation

1. **Decide the backend boundary first.** For remote-only (LM Studio / Ollama /
   vLLM), delete the GGUF + Android files and trim `factory.py`. Everything else
   in `backends/` is provider-agnostic.
2. **Decouple settings.** The factory takes a plain `settings` dict, so you do
   **not** need the SQLite layer to talk to models. Feed it from
   `port/foundation/`'s JSON `SettingsStore` (use `BACKEND_SETTINGS_DEFAULTS`
   from `database/manager.py` as your default keys).
3. **Keep the normalized types.** Build your app against `StreamChunk`,
   `ModelInfo`, `LLMHandle`, and `CancellableStream` rather than raw SDK objects
   — that's what makes swapping providers painless.
4. **Reuse `tool_resolver` for your domain tools.** Replace the demo tools
   (`add`, `get_time`, …) with `fetch_news`, `get_quote`, `get_financials`, etc.
   Its DB-`ROOT_DIR` dependency is only for loading external tool scripts; swap
   it for a config path.
5. **Write a slim controller.** `controller.py` is a benchmark orchestrator;
   emulate its threading + `event_bus` + stop-signal pattern, but for a
   chat/news-analysis flow rather than benchmark sweeps.
6. **Prune aggressively after choosing.** This bundle errs on the side of
   including too much. Once you've picked your set, delete the rest so the new
   project's dependency graph stays clean.
```
