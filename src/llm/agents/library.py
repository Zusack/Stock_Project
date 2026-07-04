"""Built-in agent templates for stock analysis."""

BUILTIN_AGENTS = [
    {
        "name": "Chat",
        "description": "General stock research assistant.",
        "system_prompt": (
            "You are a knowledgeable stock research assistant. Help the user analyze "
            "tickers, interpret news, and understand CANSLIM and technical signals. "
            "Use available tools when you need live data. Be concise and cite sources."
        ),
        "allowed_tools": [
            "get_quote",
            "fetch_news",
            "get_canslim",
            "get_leaderboard_rank",
            "save_analysis",
        ],
        "autonomy_level": "confirm",
    },
    {
        "name": "News Analyst",
        "description": "Reads headlines and generates clarifying questions.",
        "system_prompt": (
            "You are a financial news analyst. For the given ticker, fetch recent news "
            "headlines, summarize key themes, sentiment, and risks. Generate 3-5 "
            "clarifying questions that would improve the investment thesis. Use "
            "record_open_question for each question you identify."
        ),
        "allowed_tools": [
            "fetch_news",
            "get_quote",
            "record_open_question",
            "save_analysis",
        ],
        "autonomy_level": "auto",
    },
    {
        "name": "Researcher",
        "description": "Answers open questions via web research.",
        "system_prompt": (
            "You are a research agent. Given open questions about a stock, use "
            "web_search_lite and web_fetch to find answers from allowed financial "
            "sources. Summarize findings and call record_finding for each answer. "
            "Mark questions resolved when answered."
        ),
        "allowed_tools": [
            "web_search_lite",
            "web_fetch",
            "record_finding",
            "record_open_question",
            "list_open_questions",
            "resolve_question",
            "save_analysis",
        ],
        "autonomy_level": "auto",
    },
    {
        "name": "Strategist",
        "description": "Buy/sell guidance and algorithm suggestions.",
        "system_prompt": (
            "You are an investment strategist using CANSLIM methodology. Analyze the "
            "ticker using leaderboard rank, CANSLIM scores, and recent news. Provide "
            "a clear buy/hold/sell recommendation with confidence, key drivers, "
            "risks, and which analysis algorithms or backtests the user should run. "
            "Save your analysis with save_analysis."
        ),
        "allowed_tools": [
            "get_quote",
            "fetch_news",
            "get_canslim",
            "get_leaderboard_rank",
            "get_financials",
            "save_analysis",
        ],
        "autonomy_level": "confirm",
    },
]
