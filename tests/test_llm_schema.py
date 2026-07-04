"""Tests for LLM schema and conversation helpers."""

import os
import tempfile

from src.analysis.llm_schema import ensure_llm_schema
from src.analysis.llm_store import (
    create_conversation,
    list_agents,
    record_open_question,
)
from src.llm.conversation import trim_messages


def test_ensure_llm_schema_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        ensure_llm_schema(db)
        ensure_llm_schema(db)
        agents = list_agents(db)
        names = {a["name"] for a in agents}
        assert "Chat" in names
        assert "News Analyst" in names


def test_conversation_and_questions():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        ensure_llm_schema(db)
        cid = create_conversation(db, title="Test chat", model="local")
        assert cid > 0
        qid = record_open_question(db, ticker="AAPL", question="What is the catalyst?")
        assert qid > 0


def test_trim_messages_keeps_system():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "a" * 10000},
        {"role": "assistant", "content": "b" * 10000},
        {"role": "user", "content": "recent"},
    ]
    trimmed = trim_messages(msgs, max_chars=5000)
    assert trimmed[0]["role"] == "system"
    assert any(m.get("content") == "recent" for m in trimmed)
