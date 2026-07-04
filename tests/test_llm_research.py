"""Tests for web research allowlist and caps."""

from unittest.mock import MagicMock, patch

from src.llm.research import WebResearch


def test_domain_allowlist_blocks_unknown():
    wr = WebResearch()
    wr.allowlist = {"finance.yahoo.com", "sec.gov"}
    assert wr.is_allowed("https://finance.yahoo.com/news/foo")
    assert not wr.is_allowed("https://evil.example.com/page")


@patch("src.llm.research.requests.get")
def test_fetch_url_respects_allowlist(mock_get):
    wr = WebResearch()
    wr.allowlist = {"finance.yahoo.com"}
    result = wr.fetch_url("https://evil.example.com/page")
    assert "error" in result
    mock_get.assert_not_called()


@patch("src.llm.research.requests.get")
def test_fetch_url_extracts_text(mock_get):
    wr = WebResearch()
    wr.allowlist = {"example.com"}
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.encoding = "utf-8"
    mock_resp.iter_content = lambda **kw: [b"<html><title>Test</title><body><p>Hello world</p></body></html>"]
    mock_get.return_value = mock_resp
    result = wr.fetch_url("https://example.com/article")
    assert result.get("title") == "Test"
    assert "Hello world" in result.get("text", "")
