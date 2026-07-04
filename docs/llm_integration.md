# LLM Integration

This app integrates local LLMs via the **Assistant** tab and **Settings → Local AI**.

## Quick start (LM Studio)

1. Install and start [LM Studio](https://lmstudio.ai/).
2. Load a chat model in LM Studio.
3. In the app: **Settings → Local AI**
   - Enable **LM Studio (local AI)**
   - Set **Backend** to `LM Studio`
   - Set **LM Studio base URL** (default `http://localhost:1234/v1` — the SDK strips `/v1` automatically)
   - Set **Default chat model** to your model identifier
   - Click **Test connection**, then **Save AI settings**
4. Open the **Assistant** tab for chat, agents, model management, research, and insights.

## Inference settings (Temperature, Max tokens, Context length)

These values are sent **with each request** from this app (Assistant tab, agents). They apply when the app loads or calls a model via the SDK. They do not change LM Studio's global UI defaults for models you load manually in LM Studio.

| Setting | Meaning |
|---------|---------|
| **Temperature** | Randomness per generation (0 = focused, higher = more varied). |
| **Max tokens** | Maximum **output** tokens in a single reply (caps response length). |
| **Context length** | **Input** window when loading a model and trimming chat history (should be ≤ model limit). |

**Context length vs Max tokens:** Context length is how much conversation + prompt the model can *read*; max tokens is how much it may *write* in one response.

## Request timeout

**Request timeout (sec)** is the maximum wait for backend connection, model listing, and a single LLM response before the app aborts with an error. It is **not** an idle unload timer and does not tell LM Studio when to unload models.

## Backends

| Backend | Settings |
|---------|----------|
| LM Studio | `lmstudio_base_url`, `llm_chat_model` |
| Ollama | `ollama_host` |
| vLLM | `vllm_base_url`, `vllm_api_key` |

Python packages: `lmstudio`, `ollama`, `openai` (see `requirements.txt`).

## Agents

Built-in agents (seeded on first run):

- **Chat** — general Q&A with market data tools
- **News Analyst** — reads headlines, records clarifying questions
- **Researcher** — answers open questions via web scrape
- **Strategist** — buy/sell guidance using CANSLIM and leaderboard data

Configure **Agent autonomy** in Settings:

- `manual` — no tools
- `confirm` — tool calls (default)
- `auto` — full tool loop

## Web research

When enabled, agents can call `web_search_lite` and `web_fetch`. Only domains in the **Allowed domains** list are fetched. Defaults include Yahoo Finance, SEC, Reuters, etc.

## Data storage

LLM data is stored in `market_data.db`:

- `llm_conversations`, `llm_messages` — chat history
- `llm_agents`, `llm_agent_runs` — agent configs and runs
- `llm_open_questions`, `llm_research_findings` — research queue
- `ai_insights` — saved analyses (shared with headline AI)

## Headline analysis

When Local AI is enabled, `get_ai_adapter()` uses the SDK backend (`SDKLMStudioAdapter`) with fallback to the HTTP adapter and rule-based analysis.
