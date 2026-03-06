"""
scraper_tool.py – BeautifulSoup-based web scraper.

Fetches a URL and returns cleaned plain-text content.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 15
_MAX_CONTENT_CHARS = 20_000
_USER_AGENT = (
    "Mozilla/5.0 (compatible; AutonomousResearchAgent/1.0; +https://github.com)"
)


def _fetch_sync(url: str) -> Optional[str]:
    """Synchronously fetch *url* and return extracted plain text."""
    headers = {"User-Agent": _USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %r: %s", url, exc)
        return None

    soup = BeautifulSoup(response.text, "lxml")

    # Remove navigation, scripts, and style noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content = "\n".join(lines)
    return content[:_MAX_CONTENT_CHARS]


class ScraperTool:
    """Async-friendly wrapper that extracts plain text from web pages."""

    async def scrape(self, url: str) -> Optional[str]:
        """Fetch *url* and return its text content, or *None* on failure."""
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, _fetch_sync, url)
        if content:
            logger.info("Scraped %d chars from %r.", len(content), url)
        return content
