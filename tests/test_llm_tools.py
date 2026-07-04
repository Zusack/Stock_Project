"""Tests for LLM tool registry."""

from src.llm.tools.registry import TOOL_REGISTRY, all_tool_names, resolve_tools


def test_all_tool_names():
    names = all_tool_names()
    assert "get_quote" in names
    assert "web_fetch" in names
    assert "save_analysis" in names


def test_resolve_tools():
    tools = resolve_tools(["get_quote", "fetch_news", "nonexistent"])
    assert len(tools) == 2


def test_tool_registry_callable():
    for name, fn in TOOL_REGISTRY.items():
        assert callable(fn), f"{name} is not callable"
