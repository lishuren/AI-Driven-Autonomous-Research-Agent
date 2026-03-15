"""
Unit tests for the optional Scrapling hub-page backend (hub_scraper_tool).
All network calls are mocked so tests run offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_hub_mod():
    """Reset hub_scraper_tool module-level state to safe defaults."""
    import src.tools.hub_scraper_tool as hst
    hst._budget = None


# ---------------------------------------------------------------------------
# is_hub_url heuristics
# ---------------------------------------------------------------------------

class TestIsHubUrl:
    @pytest.mark.parametrize("url,title,expected", [
        # Hub URL path segments
        ("https://example.com/category/tech", "", True),
        ("https://example.com/archive/2025", "", True),
        ("https://example.com/tag/python", "", True),
        ("https://example.com/topics", "", True),
        ("https://example.com/results?q=ai", "", True),
        ("https://example.com/directory/tools", "", True),
        ("https://example.com/listing", "", True),
        ("https://example.com/feed", "", True),
        ("https://example.com/sitemap", "", True),
        ("https://example.com/browse/science", "", True),
        # Root domain (no path) is treated as hub
        ("https://example.com", "", True),
        ("https://example.com/", "", True),
        # Hub via title keywords
        ("https://example.com/posts", "All Posts", True),
        ("https://example.com/posts", "Latest Articles", True),
        ("https://example.com/posts", "Recent Posts", True),
        ("https://example.com/archive", "Archive Overview", True),
        ("https://example.com/foo", "Search Results for AI", True),
        # Non-hub pages
        ("https://example.com/article/deep-learning-2025", "Deep Learning in 2025", False),
        ("https://example.com/blog/quantum-computing", "Quantum Computing Explained", False),
        ("https://example.com/reports/annual-report", "Annual Report 2025", False),
        ("https://example.com/paper/transformer-attention", "Attention Is All You Need", False),
    ])
    def test_is_hub_url_cases(self, url, title, expected):
        from src.tools.hub_scraper_tool import is_hub_url
        assert is_hub_url(url, title) is expected


# ---------------------------------------------------------------------------
# fetch_hub_detail gate checks (no Scrapling import needed)
# ---------------------------------------------------------------------------

class TestFetchHubDetailGates:
    def setup_method(self):
        _reset_hub_mod()

    def teardown_method(self):
        _reset_hub_mod()

    def test_returns_none_when_scrapling_missing(self, event_loop):
        """fetch_hub_detail returns (None, None) when Scrapling raises on hub fetch."""
        import src.tools.hub_scraper_tool as hst

        mock_fetcher = MagicMock()
        mock_fetcher.get.side_effect = RuntimeError("connection refused")

        with patch.object(hst, "_ScraplingFetcher", mock_fetcher):
            result = event_loop.run_until_complete(
                hst.fetch_hub_detail("https://example.com/category", "AI research")
            )
        assert result == (None, None)

    def test_returns_none_on_hub_fetch_error(self, event_loop):
        """fetch_hub_detail returns (None, None) when Scrapling raises on hub fetch."""
        import src.tools.hub_scraper_tool as hst

        mock_fetcher = MagicMock()
        mock_fetcher.get.side_effect = RuntimeError("connection refused")

        with patch.object(hst, "_ScraplingFetcher", mock_fetcher):
            result = event_loop.run_until_complete(
                hst.fetch_hub_detail("https://example.com/category", "AI research")
            )
        assert result == (None, None)

    def test_returns_none_when_no_links_found(self, event_loop):
        """fetch_hub_detail returns (None, None) when hub page has no candidate links."""
        import src.tools.hub_scraper_tool as hst

        mock_hub_page = MagicMock()
        mock_hub_page.css.return_value = []  # no <a> elements

        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = mock_hub_page

        with patch.object(hst, "_ScraplingFetcher", mock_fetcher):
            result = event_loop.run_until_complete(
                hst.fetch_hub_detail("https://example.com/category", "AI research")
            )
        assert result == (None, None)


# ---------------------------------------------------------------------------
# fetch_hub_detail happy path
# ---------------------------------------------------------------------------

class TestFetchHubDetailSuccess:
    def setup_method(self):
        _reset_hub_mod()

    def teardown_method(self):
        _reset_hub_mod()

    def _make_anchor(self, href: str, text: str) -> MagicMock:
        anchor = MagicMock()
        anchor.attrib = {"href": href}
        anchor.css.return_value = [text]
        # Make css("::text").getall() return list of text
        css_result = MagicMock()
        css_result.getall.return_value = [text]
        anchor.css.return_value = css_result
        return anchor

    def test_returns_detail_url_and_text(self, event_loop):
        """fetch_hub_detail returns (detail_url, text) for a valid hub page."""
        import src.tools.hub_scraper_tool as hst

        # Hub page contains one good candidate link matching the query
        anchor = MagicMock()
        anchor.attrib = {"href": "/article/ai-research-2025"}
        link_css = MagicMock()
        link_css.getall.return_value = ["AI Research Results 2025"]
        anchor.css.return_value = link_css

        mock_hub_page = MagicMock()
        mock_hub_page.css.return_value = [anchor]

        # Detail page text
        detail_css = MagicMock()
        detail_css.getall.return_value = ["Deep insights about AI research.", "More content here."]
        mock_detail_page = MagicMock()
        mock_detail_page.css.return_value = detail_css

        call_count = [0]
        def fake_get(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_hub_page
            return mock_detail_page

        mock_fetcher = MagicMock()
        mock_fetcher.get.side_effect = fake_get

        with patch.object(hst, "_ScraplingFetcher", mock_fetcher):
            detail_url, text = event_loop.run_until_complete(
                hst.fetch_hub_detail("https://example.com/category", "AI research")
            )

        assert detail_url == "https://example.com/article/ai-research-2025"
        assert text is not None
        assert "Deep insights" in text

    def test_same_domain_filter_excludes_external_links(self, event_loop):
        """Links from a different domain are excluded; returns (None, None)."""
        import src.tools.hub_scraper_tool as hst

        # Only link is cross-domain
        anchor = MagicMock()
        anchor.attrib = {"href": "https://otherdomain.com/article"}
        link_css = MagicMock()
        link_css.getall.return_value = ["Great article"]
        anchor.css.return_value = link_css

        mock_hub_page = MagicMock()
        mock_hub_page.css.return_value = [anchor]

        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = mock_hub_page

        with patch.object(hst, "_ScraplingFetcher", mock_fetcher):
            result = event_loop.run_until_complete(
                hst.fetch_hub_detail("https://example.com/category", "AI research")
            )
        assert result == (None, None)

    def test_noise_links_excluded(self, event_loop):
        """Links containing noise patterns (login, account, etc.) are excluded."""
        import src.tools.hub_scraper_tool as hst

        noise_anchors = []
        for noise_href in ["/login", "/signup", "/account/profile", "/terms", "/privacy"]:
            anchor = MagicMock()
            anchor.attrib = {"href": noise_href}
            link_css = MagicMock()
            link_css.getall.return_value = ["Link text"]
            anchor.css.return_value = link_css
            noise_anchors.append(anchor)

        mock_hub_page = MagicMock()
        mock_hub_page.css.return_value = noise_anchors

        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = mock_hub_page

        with patch.object(hst, "_ScraplingFetcher", mock_fetcher):
            result = event_loop.run_until_complete(
                hst.fetch_hub_detail("https://example.com/category", "AI research")
            )
        assert result == (None, None)


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------

class TestFetchHubDetailBudget:
    def setup_method(self):
        _reset_hub_mod()

    def teardown_method(self):
        _reset_hub_mod()

    def test_budget_exhausted_skips_call(self, event_loop):
        """When BudgetTracker is exhausted, fetch_hub_detail returns (None, None)."""
        import src.tools.hub_scraper_tool as hst
        from src.budget import BudgetTracker

        bt = BudgetTracker(max_credits=0.0)  # immediately exhausted
        hst._budget = bt

        mock_fetcher = MagicMock()

        with patch.object(hst, "_ScraplingFetcher", mock_fetcher):
            result = event_loop.run_until_complete(
                hst.fetch_hub_detail("https://example.com/category", "AI research")
            )

        assert result == (None, None)
        mock_fetcher.get.assert_not_called()

    def test_budget_records_credits_on_success(self, event_loop):
        """Successful fetch_hub_detail records 2 credits (hub + detail fetches)."""
        import src.tools.hub_scraper_tool as hst
        from src.budget import BudgetTracker

        bt = BudgetTracker(max_credits=10.0)
        hst._budget = bt

        anchor = MagicMock()
        anchor.attrib = {"href": "/article/test"}
        link_css = MagicMock()
        link_css.getall.return_value = ["AI test article"]
        anchor.css.return_value = link_css

        mock_hub_page = MagicMock()
        mock_hub_page.css.return_value = [anchor]

        detail_css = MagicMock()
        detail_css.getall.return_value = ["Test content here."]
        mock_detail_page = MagicMock()
        mock_detail_page.css.return_value = detail_css

        call_count = [0]
        def fake_get(url, **kwargs):
            call_count[0] += 1
            return mock_hub_page if call_count[0] == 1 else mock_detail_page

        mock_fetcher = MagicMock()
        mock_fetcher.get.side_effect = fake_get

        with patch.object(hst, "_ScraplingFetcher", mock_fetcher):
            event_loop.run_until_complete(
                hst.fetch_hub_detail("https://example.com/category", "AI research")
            )

        assert bt.credits_used == 2.0


# ---------------------------------------------------------------------------
# set_scrapling / set_budget setters
# ---------------------------------------------------------------------------

class TestSetters:
    def setup_method(self):
        _reset_hub_mod()

    def teardown_method(self):
        _reset_hub_mod()

    def test_set_budget_attaches_tracker(self):
        import src.tools.hub_scraper_tool as hst
        from src.budget import BudgetTracker
        assert hst._budget is None
        bt = BudgetTracker()
        hst.set_budget(bt)
        assert hst._budget is bt
