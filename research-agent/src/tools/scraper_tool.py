"""
scraper_tool.py – Playwright-based web scraper.

Fetches a URL using a headless Chromium browser, waits for JavaScript rendering,
and returns cleaned plain-text content.  Handles both static pages and
JavaScript-heavy (SPA / React) applications. Includes retry logic for transient
network failures, robots.txt advisory checks, and user-agent rotation.
"""

from __future__ import annotations

import asyncio
import logging
import random
import urllib.parse
import urllib.robotparser
from typing import Optional

from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

_PAGE_TIMEOUT = 15_000  # milliseconds — fail fast on blocked/slow sites
_MAX_CONTENT_CHARS = 20_000
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds

# Random delay range (seconds) before each Playwright scrape
_SCRAPE_DELAY_MIN = 0.5
_SCRAPE_DELAY_MAX = 2.0

# User-agent rotation pool
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

# Module-level flag — set via set_respect_robots(); default False (opt-in)
_respect_robots: bool = False

# Cache robots.txt parsers per domain to avoid re-fetching
_robots_cache: dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}


def set_respect_robots(enabled: bool) -> None:
    """Enable or disable the robots.txt advisory check."""
    global _respect_robots
    _respect_robots = enabled


def _check_robots_txt(url: str) -> bool:
    """Return True if *url* is allowed by the site's robots.txt (advisory).

    Returns True (allow) on any error — this is advisory only.
    """
    if not _respect_robots:
        return True

    parsed = urllib.parse.urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    if domain not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{domain}/robots.txt")
        try:
            rp.read()
            _robots_cache[domain] = rp
        except Exception:
            # On any error, allow scraping
            _robots_cache[domain] = None

    rp = _robots_cache.get(domain)
    if rp is None:
        return True

    allowed = rp.can_fetch("*", url)
    if not allowed:
        logger.info("robots.txt disallows %r (advisory — proceeding anyway).", url)
    return allowed


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
        Returns *None* on permanent failures or when max retries are exceeded.
        """
        # Advisory robots.txt check
        _check_robots_txt(url)

        last_error: Optional[Exception] = None
        ua = random.choice(_USER_AGENTS)

        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            try:
                # Random delay to avoid aggressive crawling patterns
                await asyncio.sleep(random.uniform(_SCRAPE_DELAY_MIN, _SCRAPE_DELAY_MAX))

                async with async_playwright() as pw:
                    browser = await pw.chromium.launch(headless=True)
                    try:
                        context = await browser.new_context(user_agent=ua)
                        page = await context.new_page()
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
