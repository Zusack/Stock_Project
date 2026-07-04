"""Conversation message models and trimming."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMMessage:
    role: str
    content: str
    tool_calls: list | None = None

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class LLMConversation:
    id: int | None = None
    title: str = "New chat"
    agent_id: int | None = None
    model: str = ""
    backend_type: str = ""
    messages: list[LLMMessage] = field(default_factory=list)

    def to_api_messages(self, *, system_prompt: str = "") -> list[dict]:
        out: list[dict] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        out.extend(m.to_dict() for m in self.messages)
        return out


def trim_messages(
    messages: list[dict],
    *,
    max_chars: int = 24_000,
) -> list[dict]:
    """Keep system message and trim oldest user/assistant turns to fit budget."""
    if not messages:
        return []
    system_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= max_chars:
        return messages
    trimmed = list(rest)
    while trimmed and total > max_chars:
        removed = trimmed.pop(0)
        total -= len(str(removed.get("content", "")))
    return system_msgs + trimmed
