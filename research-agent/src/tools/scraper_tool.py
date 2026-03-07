"""
scraper_tool.py – Playwright-based web scraper.

Fetches a URL using a headless Chromium browser, waits for JavaScript rendering,
and returns cleaned plain-text content.  Handles both static pages and
JavaScript-heavy (SPA / React) applications. Includes retry logic for transient
network failures.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

_PAGE_TIMEOUT = 15_000  # milliseconds — fail fast on blocked/slow sites
_MAX_CONTENT_CHARS = 20_000
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds


def _is_transient_scrape_error(exc: Exception) -> bool:
    """Determine if a Playwright error is transient (retry-worthy) or permanent."""
    error_str = str(exc).lower()
    
    # Transient errors
    transient_keywords = {
        "timeout", "connection", "refused", "reset",
        "temporarily unavailable", "dns", "econnreset",
    }
    
    # Permanent errors
    permanent_keywords = {
        "404", "403", "401", "not found", "forbidden", "unauthorized",
        "net::err_invalid_url", "net::err_file_not_found",
    }
    
    if any(kw in error_str for kw in permanent_keywords):
        return False
    if any(kw in error_str for kw in transient_keywords):
        return True
    
    # Check exception type
    transient_types = (ConnectionError, TimeoutError, OSError)
    return isinstance(exc, transient_types)


class ScraperTool:
    """Playwright-backed scraper that renders JavaScript before extracting text."""

    async def scrape(self, url: str) -> Optional[str]:
        """Fetch *url*, wait for JS rendering, and return plain-text content.

        Retries on transient network failures with exponential backoff.
        Returns *None* on permanent failures or max retries exceeded.
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            try:
                async with async_playwright() as pw:
                    browser = await pw.chromium.launch(headless=True)
                    try:
                        page = await browser.new_page()
                        await page.goto(url, timeout=_PAGE_TIMEOUT, wait_until="domcontentloaded")
                        content = await page.inner_text("body")
                    finally:
                        await browser.close()

                lines = [line.strip() for line in content.splitlines() if line.strip()]
                text = "\n".join(lines)[:_MAX_CONTENT_CHARS]
                logger.info("Scraped %d chars from %r.", len(text), url)
                return text or None
            
            except Exception as exc:
                last_error = exc
                is_transient = _is_transient_scrape_error(exc)
                
                if not is_transient or attempt >= _RETRY_MAX_ATTEMPTS:
                    # Permanent error or max retries reached
                    logger.warning(
                        "Playwright scrape failed for %r (attempt %d/%d, %s): %s",
                        url,
                        attempt,
                        _RETRY_MAX_ATTEMPTS,
                        "transient" if is_transient else "permanent",
                        exc,
                    )
                    break
                
                # Transient error - retry with exponential backoff
                backoff = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.1 * backoff)
                wait_time = backoff + jitter
                logger.info(
                    "Transient scrape error for %r (attempt %d/%d), retrying in %.1fs: %s",
                    url,
                    attempt,
                    _RETRY_MAX_ATTEMPTS,
                    wait_time,
                    exc,
                )
                await asyncio.sleep(wait_time)
        
        logger.warning("Playwright scrape exhausted all %d attempts for %r", _RETRY_MAX_ATTEMPTS, url)
        return None
