"""SQLite schema for LLM conversations, agents, and research."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.analysis.db import db_connection, ingest_write_lock

LLM_TABLES = [
    "llm_conversations",
    "llm_messages",
    "llm_agents",
    "llm_agent_runs",
    "llm_open_questions",
    "llm_research_findings",
]


def ensure_llm_schema(db_path: str) -> None:
    """Create LLM tables if missing and seed built-in agents."""
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS llm_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    agent_id INTEGER,
                    model TEXT,
                    backend_type TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_llm_conv_created
                    ON llm_conversations(created_at);

                CREATE TABLE IF NOT EXISTS llm_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES llm_conversations(id)
                );
                CREATE INDEX IF NOT EXISTS idx_llm_msg_conv
                    ON llm_messages(conversation_id, created_at);

                CREATE TABLE IF NOT EXISTS llm_agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    system_prompt TEXT NOT NULL,
                    model TEXT,
                    backend_type TEXT,
                    allowed_tools_json TEXT,
                    autonomy_level TEXT DEFAULT 'confirm',
                    enabled INTEGER DEFAULT 1,
                    is_builtin INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS llm_agent_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id INTEGER NOT NULL,
                    ticker TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    summary TEXT,
                    tokens INTEGER DEFAULT 0,
                    error TEXT,
                    FOREIGN KEY (agent_id) REFERENCES llm_agents(id)
                );
                CREATE INDEX IF NOT EXISTS idx_llm_runs_agent
                    ON llm_agent_runs(agent_id, started_at);

                CREATE TABLE IF NOT EXISTS llm_open_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    source_ref TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_llm_questions_status
                    ON llm_open_questions(status, created_at);

                CREATE TABLE IF NOT EXISTS llm_research_findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_id INTEGER,
                    ticker TEXT,
                    url TEXT,
                    snippet TEXT,
                    answer TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (question_id) REFERENCES llm_open_questions(id)
                );
                """
            )
            conn.commit()
    _seed_builtin_agents(db_path)


def _seed_builtin_agents(db_path: str) -> None:
    from src.llm.agents.library import BUILTIN_AGENTS

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            for agent in BUILTIN_AGENTS:
                row = conn.execute(
                    "SELECT id FROM llm_agents WHERE name = ? AND is_builtin = 1",
                    (agent["name"],),
                ).fetchone()
                if row:
                    continue
                conn.execute(
                    """
                    INSERT INTO llm_agents
                    (name, description, system_prompt, model, backend_type,
                     allowed_tools_json, autonomy_level, enabled, is_builtin,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                    """,
                    (
                        agent["name"],
                        agent.get("description", ""),
                        agent["system_prompt"],
                        agent.get("model", ""),
                        agent.get("backend_type", ""),
                        json.dumps(agent.get("allowed_tools", [])),
                        agent.get("autonomy_level", "confirm"),
                        now,
                        now,
                    ),
                )
            conn.commit()
