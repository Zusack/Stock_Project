"""
Tool Resolver: Maps rubric tool_definition_json to Python callables for LM Studio act().
LM Studio's act() expects Python functions with type hints and docstrings.
Supports built-in registry and optional tool_script_path for custom tools.
"""
import importlib.util
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable

from src.database.manager import ROOT_DIR

# Built-in tools (function names match what the model sees; docstrings are passed to the model)
def add(a: int, b: int) -> int:
    """Given two numbers a and b, returns the sum of them."""
    return a + b


def get_time() -> str:
    """Returns the current time in HH:MM:SS format (UTC)."""
    return datetime.utcnow().strftime("%H:%M:%S")


def format_time(s: str) -> str:
    """Formats a time string as HH:MM. Expects HH:MM:SS or similar."""
    if not s or ":" not in s:
        return s
    parts = s.strip().split(":")
    if len(parts) >= 2:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    return s


def if_else(condition: str, x: int, y: int) -> int:
    """Returns x if condition evaluates to true, else y. Condition must be a boolean expression using comparison operators (e.g., '35 > 28')."""
    import re
    try:
        # Parse simple "a op b" patterns (numbers and >, <, >=, <=, ==, !=)
        m = re.match(r"^\s*(\d+)\s*(>=|<=|==|!=|>|<)\s*(\d+)\s*$", (condition or "").strip())
        if not m:
            return y
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        result = (op == ">" and a > b) or (op == "<" and a < b) or (op == ">=" and a >= b) or (op == "<=" and a <= b) or (op == "==" and a == b) or (op == "!=" and a != b)
        return x if result else y
    except Exception:
        return y


def get_ai_news(limit: int = 5, language: str = "en") -> list:
    """Fetches top AI-related headlines from a news API. Returns a JSON array of objects with fields: title, source, url. Use limit (max 10) to control count."""
    # Deterministic mock data for reproducible benchmarking (no external API)
    mock = [
        {"title": "AI Breakthrough in Language Models", "source": "TechNews", "url": "https://example.com/1"},
        {"title": "New Guidelines for AI Safety Released", "source": "ScienceDaily", "url": "https://example.com/2"},
        {"title": "Open-Source LLM Surpasses Benchmark", "source": "AI Weekly", "url": "https://example.com/3"},
    ]
    return mock[: min(max(1, limit), 10)]


def format_headlines(items: list) -> str:
    """Formats raw news items into a human-readable list: one headline per line, ending with 'Total: N'."""
    if not items:
        return "Total: 0"
    lines = []
    for i, obj in enumerate(items[:10], 1):
        title = obj.get("title", str(obj)) if isinstance(obj, dict) else str(obj)
        source = obj.get("source", "") if isinstance(obj, dict) else ""
        prefix = f"[{source}] " if source else ""
        lines.append(f'{i}. {prefix}"{title}"')
    return "\n".join(lines) + f"\nTotal: {len(items)}"


def fetch_weather(city: str) -> str:
    """Retrieves current weather data for a given city (e.g., 'London' or 'Paris'). Returns raw weather string with temperature."""
    # Deterministic mock for reproducible benchmarking (no external API)
    city_lower = (city or "").strip().lower()
    mock_temps = {"paris": 21, "london": 18, "tokyo": 24, "new york": 22}
    temp = mock_temps.get(city_lower, 20)
    return f"Weather for {city}: temp={temp}C, conditions=Partly cloudy"


def extract_temp(data: str) -> int:
    """Extracts the temperature in degrees Celsius from raw weather data."""
    import re
    m = re.search(r"temp[=:]?\s*([+-]?\d+)", data or "", re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*°?C", data or "", re.I)
    return int(m.group(1)) if m else 0


# Registry: tool name (from rubric JSON) -> Python callable
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "add": add,
    "get_time": get_time,
    "format_time": format_time,
    "if_else": if_else,
    "get_ai_news": get_ai_news,
    "format_headlines": format_headlines,
    "fetch_weather": fetch_weather,
    "extract_temp": extract_temp,
}

_log = logging.getLogger(__name__)


def _validate_tool_script_path(raw: str, tools_dir: str) -> str:
    """
    Validate tool_script_path. Reject corrupted values (e.g. prompt text captured
    by regex bug) that would cause path.join to produce invalid paths.
    """
    if not raw:
        return ""
    # Reject if it looks like rubric content (section headers, multiline prompt text)
    if "\n" in raw or "##" in raw or len(raw) > 200:
        _log.debug("Rejecting invalid tool_script_path (contains newline/##/too long): %r", raw[:80])
        return ""
    # Must be a .py file path under data/tools
    if not raw.endswith(".py"):
        return ""
    path = os.path.normpath(os.path.abspath(raw if os.path.isabs(raw) else os.path.join(ROOT_DIR, raw)))
    tools_abs = os.path.normpath(os.path.abspath(tools_dir))
    if not path.startswith(tools_abs):
        return ""
    return raw


def _load_tools_from_script(script_path: str, tools_dir: str) -> dict[str, Callable[..., Any]]:
    """
    Load tools from a Python module. Path must be under tools_dir (data/tools/) for security.
    Expects module to export TOOLS = [func1, func2] or define callables matching expected names.
    """
    if not script_path or not script_path.strip():
        return {}
    path = os.path.normpath(os.path.abspath(script_path))
    tools_dir_abs = os.path.normpath(os.path.abspath(tools_dir))
    if not path.startswith(tools_dir_abs) or not path.endswith(".py"):
        _log.warning("Tool script path %r must be a .py file under %r", script_path, tools_dir)
        return {}
    if not os.path.isfile(path):
        _log.warning("Tool script not found: %s", path)
        return {}
    try:
        spec = importlib.util.spec_from_file_location("_benchmark_tool_module", path)
        if not spec or not spec.loader:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        _log.warning("Failed to load tool script %s: %s", path, e)
        return {}
    result: dict[str, Callable[..., Any]] = {}
    if hasattr(module, "TOOLS") and isinstance(module.TOOLS, (list, tuple)):
        for item in module.TOOLS:
            if callable(item) and hasattr(item, "__name__"):
                result[item.__name__] = item
    else:
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj) and hasattr(obj, "__name__"):
                result[name] = obj
    return result


def resolve_tools(prompt_data: dict) -> list[Callable[..., Any]]:
    """
    Resolve tool definitions from prompt_data to Python callables for LM Studio act().

    Args:
        prompt_data: Dict with 'tool_definition_json' (JSON string) and optionally
                     'tool_script_path' for custom .py modules under data/tools/.

    Returns:
        List of Python callables to pass to llm.act(). Empty if no tools resolved.
    """
    tools: list[Callable[..., Any]] = []
    raw = (prompt_data.get("tool_definition_json") or "").strip()
    if not raw:
        return tools

    try:
        tool_defs = json.loads(raw)
    except json.JSONDecodeError as e:
        _log.warning("Invalid tool_definition_json: %s", e)
        return tools

    if not isinstance(tool_defs, list):
        _log.warning("tool_definition_json must be a list, got %s", type(tool_defs).__name__)
        return tools

    # Build effective registry: built-in + script tools (script overrides)
    registry = dict(TOOL_REGISTRY)
    raw_path = (prompt_data.get("tool_script_path") or "").strip()
    tools_dir = os.path.join(ROOT_DIR, "data", "tools")
    # Validate: reject corrupted paths (e.g. prompt text captured by regex bug)
    tool_script_path = _validate_tool_script_path(raw_path, tools_dir)
    if tool_script_path:
        script_path = tool_script_path
        if not os.path.isabs(script_path):
            script_path = os.path.join(ROOT_DIR, script_path)
        script_tools = _load_tools_from_script(script_path, tools_dir)
        registry.update(script_tools)

    for item in tool_defs:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name or not isinstance(name, str):
            continue
        name = name.strip()
        if not name:
            continue
        if name in registry:
            tools.append(registry[name])
        else:
            _log.warning("Unknown tool name %r (not in registry or script). Skipping.", name)

    return tools
