"""
search_tool.py – DuckDuckGo search wrapper.

Returns a list of result dicts: {'title': str, 'url': str, 'body': str}.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import warnings
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RATE_LIMIT_MIN = 2.0
_RATE_LIMIT_MAX = 5.0
_DEFAULT_MAX_RESULTS = 5
_SEARCH_BACKENDS = ("lite", "html", "bing")


def _normalise_results(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert provider-specific result dictionaries into a stable schema."""
    normalised: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue

        # `ddgs` uses `href`; some older payloads already use `url`.
        url = str(item.get("url") or item.get("href") or "")
        title = str(item.get("title") or item.get("heading") or "")
        body = str(item.get("body") or item.get("snippet") or item.get("content") or "")

        if not (url or title or body):
            continue
        normalised.append({"title": title, "url": url, "body": body})
    return normalised


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Perform a synchronous DuckDuckGo text search."""
    try:
        using_new_ddgs = False
        try:
            from ddgs import DDGS
            using_new_ddgs = True
        except Exception:
            from duckduckgo_search import DDGS

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*renamed to `ddgs`.*",
                category=RuntimeWarning,
            )
            with DDGS() as ddgs:
                if using_new_ddgs:
                    # `ddgs` exposes a different backend set and `auto` is usually the
                    # most reliable cross-network option.
                    return _normalise_results(
                        list(ddgs.text(query, max_results=max_results))
                    )

                last_error: Optional[Exception] = None
                for backend in _SEARCH_BACKENDS:
                    try:
                        results = list(
                            ddgs.text(query, max_results=max_results, backend=backend)
                        )
                    except TypeError:
                        # Some DDGS implementations do not accept a `backend` kwarg.
                        results = list(ddgs.text(query, max_results=max_results))
                    except Exception as exc:
                        last_error = exc
                        logger.info(
                            "Search backend %r failed for query %r: %s",
                            backend,
                            query,
                            exc,
                        )
                        continue

                    normalised = _normalise_results(results)
                    if normalised:
                        return normalised

                if last_error is not None:
                    raise last_error
                return []
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
