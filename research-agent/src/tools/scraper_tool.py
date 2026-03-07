"""
scraper_tool.py – Playwright-based web scraper.

Fetches a URL using a headless Chromium browser, waits for JavaScript rendering,
and returns cleaned plain-text content.  Handles both static pages and
JavaScript-heavy (SPA / React) applications.
"""

from __future__ import annotations

import logging
from typing import Optional

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

_PAGE_TIMEOUT = 30_000  # milliseconds — covers slow JS-heavy pages
_MAX_CONTENT_CHARS = 20_000


class ScraperTool:
    """Playwright-backed scraper that renders JavaScript before extracting text."""

    async def scrape(self, url: str) -> Optional[str]:
        """Fetch *url*, wait for JS rendering, and return plain-text content.

        Returns *None* on any failure.
        """
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                try:
                    page = await browser.new_page()
                    await page.goto(url, timeout=_PAGE_TIMEOUT, wait_until="networkidle")
                    content = await page.inner_text("body")
                finally:
                    await browser.close()

            lines = [line.strip() for line in content.splitlines() if line.strip()]
            text = "\n".join(lines)[:_MAX_CONTENT_CHARS]
            logger.info("Scraped %d chars from %r.", len(text), url)
            return text or None

        except Exception as exc:
            logger.warning("Playwright scrape failed for %r: %s", url, exc)
            return None
