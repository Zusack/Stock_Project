"""Repository helpers for LLM persistence tables."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.analysis.db import db_connection, ingest_write_lock


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def create_conversation(
    db_path: str,
    *,
    title: str,
    agent_id: int | None = None,
    model: str = "",
    backend_type: str = "",
) -> int:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            cur = conn.execute(
                """
                INSERT INTO llm_conversations (title, created_at, agent_id, model, backend_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (title, _now(), agent_id, model, backend_type),
            )
            conn.commit()
            return int(cur.lastrowid)


def list_conversations(db_path: str, *, limit: int = 50) -> list[dict]:
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at, agent_id, model, backend_type
            FROM llm_conversations ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "agent_id": r[3],
            "model": r[4],
            "backend_type": r[5],
        }
        for r in rows
    ]


def add_message(
    db_path: str,
    *,
    conversation_id: int,
    role: str,
    content: str,
    tool_calls: list | None = None,
) -> int:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            cur = conn.execute(
                """
                INSERT INTO llm_messages
                (conversation_id, role, content, tool_calls_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(tool_calls) if tool_calls else None,
                    _now(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)


def load_messages(db_path: str, conversation_id: int) -> list[dict]:
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(
            """
            SELECT role, content, tool_calls_json, created_at
            FROM llm_messages WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "role": r[0],
            "content": r[1],
            "tool_calls": json.loads(r[2]) if r[2] else None,
            "created_at": r[3],
        })
    return out


def list_agents(db_path: str, *, enabled_only: bool = False) -> list[dict]:
    sql = """
        SELECT id, name, description, system_prompt, model, backend_type,
               allowed_tools_json, autonomy_level, enabled, is_builtin
        FROM llm_agents
    """
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY is_builtin DESC, name ASC"
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(sql).fetchall()
    return [_agent_row(r) for r in rows]


def get_agent(db_path: str, agent_id: int) -> dict | None:
    with db_connection(db_path, readonly=True) as conn:
        row = conn.execute(
            """
            SELECT id, name, description, system_prompt, model, backend_type,
                   allowed_tools_json, autonomy_level, enabled, is_builtin
            FROM llm_agents WHERE id = ?
            """,
            (agent_id,),
        ).fetchone()
    return _agent_row(row) if row else None


def _agent_row(r) -> dict:
    return {
        "id": r[0],
        "name": r[1],
        "description": r[2],
        "system_prompt": r[3],
        "model": r[4],
        "backend_type": r[5],
        "allowed_tools": json.loads(r[6] or "[]"),
        "autonomy_level": r[7],
        "enabled": bool(r[8]),
        "is_builtin": bool(r[9]),
    }


def save_agent(db_path: str, agent: dict) -> int:
    now = _now()
    tools_json = json.dumps(agent.get("allowed_tools", []))
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            if agent.get("id"):
                conn.execute(
                    """
                    UPDATE llm_agents SET
                        name=?, description=?, system_prompt=?, model=?,
                        backend_type=?, allowed_tools_json=?, autonomy_level=?,
                        enabled=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        agent["name"],
                        agent.get("description", ""),
                        agent["system_prompt"],
                        agent.get("model", ""),
                        agent.get("backend_type", ""),
                        tools_json,
                        agent.get("autonomy_level", "confirm"),
                        1 if agent.get("enabled", True) else 0,
                        now,
                        agent["id"],
                    ),
                )
                conn.commit()
                return int(agent["id"])
            cur = conn.execute(
                """
                INSERT INTO llm_agents
                (name, description, system_prompt, model, backend_type,
                 allowed_tools_json, autonomy_level, enabled, is_builtin,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    agent["name"],
                    agent.get("description", ""),
                    agent["system_prompt"],
                    agent.get("model", ""),
                    agent.get("backend_type", ""),
                    tools_json,
                    agent.get("autonomy_level", "confirm"),
                    1 if agent.get("enabled", True) else 0,
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)


def delete_agent(db_path: str, agent_id: int) -> bool:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            row = conn.execute(
                "SELECT is_builtin FROM llm_agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if not row or row[0]:
                return False
            conn.execute("DELETE FROM llm_agents WHERE id = ?", (agent_id,))
            conn.commit()
            return True


def create_agent_run(
    db_path: str,
    *,
    agent_id: int,
    ticker: str = "",
) -> int:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            cur = conn.execute(
                """
                INSERT INTO llm_agent_runs
                (agent_id, ticker, status, started_at)
                VALUES (?, ?, 'running', ?)
                """,
                (agent_id, ticker.upper() if ticker else "", _now()),
            )
            conn.commit()
            return int(cur.lastrowid)


def finish_agent_run(
    db_path: str,
    run_id: int,
    *,
    status: str,
    summary: str = "",
    tokens: int = 0,
    error: str = "",
) -> None:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                UPDATE llm_agent_runs SET
                    status=?, finished_at=?, summary=?, tokens=?, error=?
                WHERE id=?
                """,
                (status, _now(), summary, tokens, error, run_id),
            )
            conn.commit()


def list_agent_runs(db_path: str, *, limit: int = 30) -> list[dict]:
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.agent_id, a.name, r.ticker, r.status,
                   r.started_at, r.finished_at, r.summary, r.tokens, r.error
            FROM llm_agent_runs r
            LEFT JOIN llm_agents a ON a.id = r.agent_id
            ORDER BY r.started_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "agent_id": r[1],
            "agent_name": r[2],
            "ticker": r[3],
            "status": r[4],
            "started_at": r[5],
            "finished_at": r[6],
            "summary": r[7],
            "tokens": r[8],
            "error": r[9],
        }
        for r in rows
    ]


def record_open_question(
    db_path: str,
    *,
    ticker: str,
    question: str,
    source_ref: str = "",
) -> int:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            cur = conn.execute(
                """
                INSERT INTO llm_open_questions
                (ticker, question, status, source_ref, created_at)
                VALUES (?, ?, 'open', ?, ?)
                """,
                (ticker.upper(), question, source_ref, _now()),
            )
            conn.commit()
            return int(cur.lastrowid)


def list_open_questions(
    db_path: str,
    *,
    status: str | None = None,
    ticker: str | None = None,
    limit: int = 50,
) -> list[dict]:
    sql = """
        SELECT id, ticker, question, status, source_ref, created_at, resolved_at
        FROM llm_open_questions WHERE 1=1
    """
    params: list[Any] = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker.upper())
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": r[0],
            "ticker": r[1],
            "question": r[2],
            "status": r[3],
            "source_ref": r[4],
            "created_at": r[5],
            "resolved_at": r[6],
        }
        for r in rows
    ]


def resolve_question(db_path: str, question_id: int) -> None:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            conn.execute(
                """
                UPDATE llm_open_questions SET status='resolved', resolved_at=?
                WHERE id=?
                """,
                (_now(), question_id),
            )
            conn.commit()


def record_finding(
    db_path: str,
    *,
    question_id: int | None,
    ticker: str,
    url: str,
    snippet: str,
    answer: str,
) -> int:
    with ingest_write_lock():
        with db_connection(db_path, readonly=False) as conn:
            cur = conn.execute(
                """
                INSERT INTO llm_research_findings
                (question_id, ticker, url, snippet, answer, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (question_id, ticker.upper(), url, snippet[:2000], answer, _now()),
            )
            conn.commit()
            return int(cur.lastrowid)


def list_findings(db_path: str, *, ticker: str | None = None, limit: int = 50) -> list[dict]:
    sql = """
        SELECT id, question_id, ticker, url, snippet, answer, created_at
        FROM llm_research_findings WHERE 1=1
    """
    params: list[Any] = []
    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker.upper())
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": r[0],
            "question_id": r[1],
            "ticker": r[2],
            "url": r[3],
            "snippet": r[4],
            "answer": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


def list_ai_insights(db_path: str, *, ticker: str | None = None, limit: int = 50) -> list[dict]:
    sql = """
        SELECT ticker, created_at, provider, model, kind, payload_json
        FROM ai_insights WHERE 1=1
    """
    params: list[Any] = []
    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker.upper())
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with db_connection(db_path, readonly=True) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "ticker": r[0],
            "created_at": r[1],
            "provider": r[2],
            "model": r[3],
            "kind": r[4],
            "payload": json.loads(r[5] or "{}"),
        }
        for r in rows
    ]
