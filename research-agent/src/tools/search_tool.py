"""
search_tool.py – DuckDuckGo search wrapper.

Returns a list of result dicts: {'title': str, 'url': str, 'body': str}.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_RATE_LIMIT_MIN = 2.0
_RATE_LIMIT_MAX = 5.0
_DEFAULT_MAX_RESULTS = 5


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Perform a synchronous DuckDuckGo text search."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for query %r: %s", query, exc)
        return []


class SearchTool:
    """Async-friendly wrapper around DuckDuckGo search."""

    def __init__(self, max_results: int = _DEFAULT_MAX_RESULTS) -> None:
        self.max_results = max_results
        self._last_call: float = 0.0

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search for *query* and return a list of result dicts.

        Applies randomised rate-limiting between consecutive calls to reduce
        the risk of IP blocks.
        """
        elapsed = time.monotonic() - self._last_call
        sleep_time = random.uniform(_RATE_LIMIT_MIN, _RATE_LIMIT_MAX)
        if elapsed < sleep_time:
            await asyncio.sleep(sleep_time - elapsed)

        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None, _search_sync, query, self.max_results
            )
        except Exception as exc:
            logger.warning("Search executor error for query %r: %s", query, exc)
            results = []
        self._last_call = time.monotonic()
        logger.info("Search for %r returned %d results.", query, len(results))
        return results
