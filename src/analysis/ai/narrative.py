"""Parse LLM narrative JSON for ticker insights."""

from __future__ import annotations

import json
import re


def extract_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def parse_narrative_json(text: str) -> dict | None:
    """Return dict when payload looks like ticker narrative insight JSON."""
    data = extract_json_object(text)
    if not isinstance(data, dict):
        return None
    if data.get("headline") or data.get("detail"):
        return data
    return None
