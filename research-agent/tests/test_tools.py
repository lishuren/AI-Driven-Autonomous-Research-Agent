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
    def setup_method(self):
        """Reset process-scoped flags before each test."""
        import src.tools.search_tool as st_mod
        st_mod._tavily_quota_exhausted = False
        st_mod._cjk_no_results_warned = False

    def test_search_returns_results(self, event_loop):
        """SearchTool.search should return results from Tavily."""
        from src.tools.search_tool import SearchTool

        fake_results = [
            {"title": "Test", "url": "https://example.com", "body": "body text"},
        ]

        with patch("src.tools.search_tool._tavily_search_sync", return_value=fake_results):
            tool = SearchTool(max_results=5)
            tool._last_call = 0.0  # skip rate-limit wait

            results = event_loop.run_until_complete(tool.search("test query"))

        assert results == fake_results

    def test_search_handles_exception(self, event_loop):
        """SearchTool.search should return [] when Tavily raises."""
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import SearchTool

        with patch.object(st_mod, "_tavily_search_sync", side_effect=RuntimeError("network")):
            tool = SearchTool()
            tool._last_call = 0.0

            results = event_loop.run_until_complete(tool.search("fail query"))

        assert results == []

    def test_contains_cjk_detects_chinese(self):
        """_contains_cjk should return True for Chinese characters."""
        from src.tools.search_tool import _contains_cjk
        assert _contains_cjk("用户数量") is True
        assert _contains_cjk("TRPG market analysis") is False
        assert _contains_cjk("mixed query 用户") is True
        assert _contains_cjk("한국어") is True   # Korean
        assert _contains_cjk("") is False

    def test_tavily_disabled_when_no_api_key(self, monkeypatch):
        """_tavily_search_sync should return [] immediately when key is absent."""
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import _tavily_search_sync

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        st_mod._tavily_quota_exhausted = False
        assert _tavily_search_sync("test query", 5) == []

    def test_tavily_quota_exhausted_disables_tavily(self, monkeypatch):
        """_tavily_search_sync should return [] and set flag on 401/429/432."""
        import urllib.error
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import _tavily_search_sync

        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        st_mod._tavily_quota_exhausted = False

        http_error = urllib.error.HTTPError(url="", code=401, msg="Unauthorized", hdrs=None, fp=None)
        with patch("urllib.request.urlopen", side_effect=http_error):
            result = _tavily_search_sync("test query", 5)

        assert result == []
        assert st_mod._tavily_quota_exhausted is True

        with patch("urllib.request.urlopen", side_effect=AssertionError("should not call")):
            assert _tavily_search_sync("another", 5) == []

    def test_tavily_432_disables_tavily(self, monkeypatch):
        """HTTP 432 (usage limit exceeded) should set quota exhausted flag."""
        import urllib.error
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import _tavily_search_sync

        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        st_mod._tavily_quota_exhausted = False

        http_error = urllib.error.HTTPError(url="", code=432, msg="", hdrs=None, fp=None)
        with patch("urllib.request.urlopen", side_effect=http_error):
            result = _tavily_search_sync("test query", 5)

        assert result == []
        assert st_mod._tavily_quota_exhausted is True

        # Subsequent calls must short-circuit without hitting the network
        with patch("urllib.request.urlopen", side_effect=AssertionError("should not call")):
            assert _tavily_search_sync("another", 5) == []

    def test_tavily_returns_normalised_results(self, monkeypatch):
        """_tavily_search_sync should normalise Tavily's 'content' field to 'body'."""
        import json as _json
        import io
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import _tavily_search_sync

        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        st_mod._tavily_quota_exhausted = False

        payload = {"results": [
            {"title": "T1", "url": "https://example.com/1", "content": "body text"},
            {"title": "T2", "url": "https://example.com/2", "content": "more text"},
        ]}
        fake_resp = MagicMock()
        fake_resp.read.return_value = _json.dumps(payload).encode()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=fake_resp):
            results = _tavily_search_sync("query", 5)

        assert len(results) == 2
        assert results[0]["body"] == "body text"
        assert results[1]["url"] == "https://example.com/2"

    def test_cjk_no_results_warns_about_api_key(self, event_loop, monkeypatch):
        """SearchTool.search should emit a one-time WARNING when CJK returns 0 results
        and TAVILY_API_KEY is not set."""
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import SearchTool

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        st_mod._cjk_no_results_warned = False

        with patch.object(st_mod, "_tavily_search_sync", return_value=[]), \
             patch.object(st_mod, "_contains_cjk", return_value=True), \
             patch.object(st_mod.logger, "warning") as mock_warn:
            tool = SearchTool()
            tool._last_call = 0.0
            results = event_loop.run_until_complete(tool.search("中国TRPG市场"))

        assert results == []
        joined = " ".join(str(call) for call in mock_warn.call_args_list)
        assert "TAVILY_API_KEY" in joined

        # Second call must NOT warn again
        st_mod._cjk_no_results_warned = True
        with patch.object(st_mod, "_tavily_search_sync", return_value=[]), \
             patch.object(st_mod, "_contains_cjk", return_value=True), \
             patch.object(st_mod.logger, "warning") as mock_warn2:
            tool2 = SearchTool()
            tool2._last_call = 0.0
            event_loop.run_until_complete(tool2.search("日本語クエリ"))
        key_warns = [c for c in mock_warn2.call_args_list if "TAVILY_API_KEY" in str(c)]
        assert len(key_warns) == 0


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

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
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

        with patch("src.tools.search_tool._tavily_search_sync", return_value=fake_results):
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

        with patch("src.tools.search_tool._tavily_search_sync", return_value=fake_results):
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

        with patch("src.tools.search_tool._tavily_search_sync", return_value=fake_results):
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


# ---------------------------------------------------------------------------
# Tavily Extract
# ---------------------------------------------------------------------------

class TestTavilyExtract:
    def setup_method(self):
        import src.tools.search_tool as st_mod
        st_mod._tavily_quota_exhausted = False
        st_mod._budget = None

    def test_extract_returns_content(self, event_loop):
        """SearchTool.extract should return extracted content for URLs.."""
        from src.tools.search_tool import SearchTool

        fake_resp = {
            "results": [
                {"url": "https://a.com", "raw_content": "Page A text"},
                {"url": "https://b.com", "raw_content": "Page B text"},
            ],
            "usage": {"credits": 1},
        }

        with patch("src.tools.search_tool._tavily_extract_sync", return_value=[
            {"url": "https://a.com", "content": "Page A text"},
            {"url": "https://b.com", "content": "Page B text"},
        ]):
            tool = SearchTool()
            results = event_loop.run_until_complete(
                tool.extract(["https://a.com", "https://b.com"], "test query")
            )

        assert len(results) == 2
        assert results[0]["url"] == "https://a.com"
        assert results[0]["content"] == "Page A text"

    def test_extract_skips_when_budget_exhausted(self, event_loop):
        """Extract should return [] when budget is exhausted."""
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import SearchTool
        from src.budget import BudgetTracker

        bt = BudgetTracker(max_credits=0.0)
        st_mod._budget = bt

        tool = SearchTool()
        results = event_loop.run_until_complete(
            tool.extract(["https://a.com"], "query")
        )
        assert results == []


# ---------------------------------------------------------------------------
# Search budget integration
# ---------------------------------------------------------------------------

class TestSearchBudget:
    def setup_method(self):
        import src.tools.search_tool as st_mod
        st_mod._tavily_quota_exhausted = False
        st_mod._budget = None

    def teardown_method(self):
        import src.tools.search_tool as st_mod
        st_mod._budget = None

    def test_search_skips_when_budget_exhausted(self, event_loop):
        """Search should return [] when budget is exhausted."""
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import SearchTool
        from src.budget import BudgetTracker

        bt = BudgetTracker(max_queries=0)
        st_mod._budget = bt

        tool = SearchTool()
        tool._last_call = 0.0
        results = event_loop.run_until_complete(tool.search("test"))
        assert results == []

    def test_search_records_credits(self, event_loop, monkeypatch):
        """Search should record credit usage in budget tracker."""
        import json as _json
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import SearchTool
        from src.budget import BudgetTracker

        bt = BudgetTracker(max_credits=100.0)
        st_mod._budget = bt
        # Ensure Tavily is considered available
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        st_mod._tavily_quota_exhausted = False

        # Mock at HTTP level so _tavily_search_sync runs and records credits
        fake_api_response = _json.dumps({
            "results": [
                {"title": "T", "url": "https://example.com", "content": "text"}
            ],
            "usage": {"credits": 1},
        }).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = fake_api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.tools.search_tool.urllib.request.urlopen", return_value=mock_resp):
            tool = SearchTool()
            tool._last_call = 0.0
            event_loop.run_until_complete(tool.search("test query"))

        assert bt.queries_used >= 1


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    def test_detect_english(self):
        from src.tools.search_tool import _detect_language
        assert _detect_language("stock trading strategies") == "en"

    def test_detect_chinese(self):
        from src.tools.search_tool import _detect_language
        assert _detect_language("在线TRPG市场分析") == "zh"

    def test_detect_mixed_mostly_english(self):
        from src.tools.search_tool import _detect_language
        assert _detect_language("AI research on neural networks") == "en"

    def test_detect_empty(self):
        from src.tools.search_tool import _detect_language
        assert _detect_language("") == "en"


# ---------------------------------------------------------------------------
# Scraper robots.txt & no-scrape
# ---------------------------------------------------------------------------

class TestScraperFlags:
    def teardown_method(self):
        import src.tools.scraper_tool as sc_mod
        sc_mod._no_scrape = False
        sc_mod._respect_robots = True

    def test_no_scrape_returns_none(self, event_loop):
        """When _no_scrape is True, scrape should return None immediately."""
        from src.tools.scraper_tool import ScraperTool, set_no_scrape
        set_no_scrape(True)

        tool = ScraperTool()
        result = event_loop.run_until_complete(tool.scrape("https://example.com"))
        assert result is None

    def test_check_robots_txt_allows(self):
        """robots.txt check should return True when allowed."""
        from src.tools.scraper_tool import _check_robots_txt, set_respect_robots
        set_respect_robots(True)

        # Mock the RobotFileParser
        with patch("src.tools.scraper_tool._robots_cache", {"https://example.com": None}):
            assert _check_robots_txt("https://example.com/page") is True

    def test_check_robots_txt_disabled(self):
        """When respect_robots is False, always returns True."""
        from src.tools.scraper_tool import _check_robots_txt, set_respect_robots
        set_respect_robots(False)
        assert _check_robots_txt("https://example.com/page") is True


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def setup_method(self):
        import src.tools.search_tool as st_mod
        st_mod._dry_run = False
        st_mod._tavily_quota_exhausted = False

    def teardown_method(self):
        import src.tools.search_tool as st_mod
        st_mod._dry_run = False

    def test_dry_run_search_returns_empty(self, event_loop):
        """In dry-run mode, search() returns [] without calling Tavily."""
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import SearchTool, set_dry_run

        set_dry_run(True)
        with patch("src.tools.search_tool._tavily_search_sync") as mock_search:
            tool = SearchTool()
            tool._last_call = 0.0
            results = event_loop.run_until_complete(tool.search("test query"))

        assert results == []
        mock_search.assert_not_called()

    def test_dry_run_extract_returns_empty(self, event_loop):
        """In dry-run mode, extract() returns [] without calling Tavily."""
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import SearchTool, set_dry_run

        set_dry_run(True)
        with patch("src.tools.search_tool._tavily_extract_sync") as mock_extract:
            tool = SearchTool()
            results = event_loop.run_until_complete(
                tool.extract(["https://example.com"], "query")
            )

        assert results == []
        mock_extract.assert_not_called()

    def test_dry_run_does_not_change_budget(self, event_loop):
        """Dry-run search should not touch the budget tracker."""
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import SearchTool, set_dry_run
        from src.budget import BudgetTracker

        bt = BudgetTracker(max_credits=100.0)
        st_mod._budget = bt
        set_dry_run(True)

        tool = SearchTool()
        tool._last_call = 0.0
        event_loop.run_until_complete(tool.search("test"))

        assert bt.queries_used == 0
        assert bt.credits_used == 0.0

    def test_set_dry_run_toggles(self):
        """set_dry_run() can enable and disable dry-run mode."""
        import src.tools.search_tool as st_mod
        from src.tools.search_tool import set_dry_run

        set_dry_run(True)
        assert st_mod._dry_run is True
        set_dry_run(False)
        assert st_mod._dry_run is False


# ---------------------------------------------------------------------------
# Tavily account credits fetch
# ---------------------------------------------------------------------------

class TestFetchAccountCredits:
    def test_returns_none_when_no_key(self, event_loop, monkeypatch):
        """fetch_account_credits returns None when TAVILY_API_KEY is absent."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        from src.tools.search_tool import fetch_account_credits
        result = event_loop.run_until_complete(fetch_account_credits())
        assert result is None

    def test_returns_credit_info_on_success(self, event_loop, monkeypatch):
        """fetch_account_credits parses the /usage response correctly."""
        import json as _json
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

        fake_response = _json.dumps({
            "used": 123,
            "limit": 1000,
            "remaining": 877,
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        from src.tools.search_tool import fetch_account_credits
        with patch("src.tools.search_tool.urllib.request.urlopen", return_value=mock_resp):
            result = event_loop.run_until_complete(fetch_account_credits())

        assert result is not None
        assert result["credits_used"] == 123
        assert result["credits_limit"] == 1000
        assert result["credits_remaining"] == 877

    def test_returns_none_on_404(self, event_loop, monkeypatch):
        """fetch_account_credits returns None when /usage returns 404."""
        import urllib.error
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

        from src.tools.search_tool import fetch_account_credits
        err = urllib.error.HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None)
        with patch("src.tools.search_tool.urllib.request.urlopen", side_effect=err):
            result = event_loop.run_until_complete(fetch_account_credits())

        assert result is None

    def test_returns_none_on_network_error(self, event_loop, monkeypatch):
        """fetch_account_credits returns None on connection failure."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

        from src.tools.search_tool import fetch_account_credits
        with patch("src.tools.search_tool.urllib.request.urlopen",
                   side_effect=OSError("connection refused")):
            result = event_loop.run_until_complete(fetch_account_credits())

        assert result is None
