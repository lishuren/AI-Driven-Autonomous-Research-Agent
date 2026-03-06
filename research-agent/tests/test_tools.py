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
        """ScraperTool.scrape should return extracted text."""
        from src.tools.scraper_tool import ScraperTool

        with patch("src.tools.scraper_tool._fetch_sync", return_value="Hello world"):
            tool = ScraperTool()
            result = event_loop.run_until_complete(tool.scrape("https://example.com"))

        assert result == "Hello world"

    def test_scrape_returns_none_on_failure(self, event_loop):
        """ScraperTool.scrape should return None when fetch fails."""
        from src.tools.scraper_tool import ScraperTool

        with patch("src.tools.scraper_tool._fetch_sync", return_value=None):
            tool = ScraperTool()
            result = event_loop.run_until_complete(tool.scrape("https://bad.url"))

        assert result is None
