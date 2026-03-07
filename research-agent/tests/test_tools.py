"""
Unit tests for the tools layer (SearchTool, ScraperTool).
All network calls are mocked so tests run offline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# SearchTool
# ---------------------------------------------------------------------------

class TestSearchTool:
    def test_search_returns_results(self, event_loop):
        """SearchTool.search should return a list of dicts from DDGS."""
        from src.tools.search_tool import SearchTool

        fake_results = [
            {"title": "Test", "url": "https://example.com", "body": "body text"},
        ]

        with patch("src.tools.search_tool._search_sync", return_value=fake_results):
            tool = SearchTool(max_results=5)
            tool._last_call = 0.0  # skip rate-limit wait

            results = event_loop.run_until_complete(tool.search("test query"))

        assert results == fake_results

    def test_search_handles_exception(self, event_loop):
        """SearchTool.search should return [] when the underlying search raises."""
        from src.tools.search_tool import SearchTool

        with patch("src.tools.search_tool._search_sync", side_effect=RuntimeError("network")):
            tool = SearchTool()
            tool._last_call = 0.0

            results = event_loop.run_until_complete(tool.search("fail query"))

        assert results == []


# ---------------------------------------------------------------------------
# ScraperTool
# ---------------------------------------------------------------------------

class TestScraperTool:
    def test_scrape_returns_text(self, event_loop):
        """ScraperTool.scrape should return text after JS rendering."""
        from src.tools.scraper_tool import ScraperTool

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.inner_text = AsyncMock(return_value="Hello world\n  \nMore text")

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)
        mock_browser.close = AsyncMock()

        mock_pw = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.scraper_tool.async_playwright", return_value=mock_ctx):
            tool = ScraperTool()
            result = event_loop.run_until_complete(tool.scrape("https://example.com"))

        assert result == "Hello world\nMore text"

    def test_scrape_returns_none_on_failure(self, event_loop):
        """ScraperTool.scrape should return None when Playwright raises."""
        from src.tools.scraper_tool import ScraperTool

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=Exception("browser launch failed"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.scraper_tool.async_playwright", return_value=mock_ctx):
            tool = ScraperTool()
            result = event_loop.run_until_complete(tool.scrape("https://bad.url"))

        assert result is None


# ---------------------------------------------------------------------------
# SearchLogger
# ---------------------------------------------------------------------------

class TestSearchLogger:
    def setup_method(self):
        """Reset SearchLogger state before each test."""
        from src.tools.search_tool import SearchLogger
        SearchLogger.close()

    def teardown_method(self):
        """Ensure the logger is closed after each test."""
        from src.tools.search_tool import SearchLogger
        SearchLogger.close()

    def test_disabled_by_default(self, event_loop, tmp_path):
        """SearchLogger should not write anything unless enabled."""
        from src.tools.search_tool import SearchTool, SearchLogger

        log_path = tmp_path / "search.jsonl"
        fake_results = [{"title": "T", "url": "https://example.com", "body": "b"}]

        with patch("src.tools.search_tool._search_sync", return_value=fake_results):
            tool = SearchTool()
            tool._last_call = 0.0
            event_loop.run_until_complete(tool.search("test query"))

        assert not log_path.exists()

    def test_writes_entry_when_enabled(self, event_loop, tmp_path):
        """SearchLogger should write a JSONL entry for each search when enabled."""
        import json
        from src.tools.search_tool import SearchTool, SearchLogger

        log_path = tmp_path / "search.jsonl"
        SearchLogger.enable(str(log_path))

        fake_results = [
            {"title": "T1", "url": "https://example.com/page", "body": "b1"},
            {"title": "T2", "url": "https://other.org/post", "body": "b2"},
        ]

        with patch("src.tools.search_tool._search_sync", return_value=fake_results):
            tool = SearchTool()
            tool._last_call = 0.0
            event_loop.run_until_complete(tool.search("RL policy gradient"))

        SearchLogger.close()

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["query"] == "RL policy gradient"
        assert entry["result_count"] == 2
        assert "example.com" in entry["domains"]
        assert "other.org" in entry["domains"]
        assert "ts" in entry

    def test_multiple_searches_append(self, event_loop, tmp_path):
        """Each search call appends a separate line."""
        import json
        from src.tools.search_tool import SearchTool, SearchLogger

        log_path = tmp_path / "search.jsonl"
        SearchLogger.enable(str(log_path))

        fake_results = [{"title": "T", "url": "https://example.com", "body": "b"}]

        with patch("src.tools.search_tool._search_sync", return_value=fake_results):
            tool = SearchTool()
            tool._last_call = 0.0
            event_loop.run_until_complete(tool.search("query one"))
            tool._last_call = 0.0
            event_loop.run_until_complete(tool.search("query two"))

        SearchLogger.close()

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["query"] == "query one"
        assert json.loads(lines[1])["query"] == "query two"
