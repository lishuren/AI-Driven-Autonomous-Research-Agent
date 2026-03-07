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
# Backends to try in order. "brave" uses Google and is blocked by robot detection; skip it.
_SEARCH_BACKENDS = ("lite", "html", "bing")
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds


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


def _is_transient_error(exc: Exception) -> bool:
    """Determine if an error is transient (retry-worthy) or permanent."""
    error_str = str(exc).lower()
    
    # Transient errors: network, timeout, temporary unavailability
    transient_keywords = {
        "timeout", "connection", "refused", "reset", "host unreachable",
        "temporarily unavailable", "429", "503", "dns", "resolved",
    }
    
    # Permanent errors: client errors, invalid input
    permanent_keywords = {
        "404", "403", "401", "invalid", "not found", "forbidden",
        "unauthorized", "malformed", "typeerror",
    }
    
    if any(kw in error_str for kw in permanent_keywords):
        return False
    if any(kw in error_str for kw in transient_keywords):
        return True
    
    # Check exception type
    transient_types = (
        ConnectionError, TimeoutError, OSError, PermissionError
    )
    return isinstance(exc, transient_types)


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Perform a synchronous DuckDuckGo text search with retry logic."""
    last_error: Optional[Exception] = None
    
    for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
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
                        # Iterate through safe backends explicitly — avoid "brave" which
                        # routes through Google and triggers robot-detection blocks.
                        backend_error = None
                        for backend in _SEARCH_BACKENDS:
                            try:
                                results = list(
                                    ddgs.text(query, max_results=max_results, backend=backend)
                                )
                            except TypeError:
                                results = list(ddgs.text(query, max_results=max_results))
                            except Exception as exc:
                                backend_error = exc
                                logger.info(
                                    "Search backend %r failed for query %r: %s",
                                    backend, query, exc,
                                )
                                continue

                            normalised = _normalise_results(results)
                            if normalised:
                                return normalised

                        if backend_error is not None:
                            raise backend_error
                        return []

                    backend_error: Optional[Exception] = None
                    for backend in _SEARCH_BACKENDS:
                        try:
                            results = list(
                                ddgs.text(query, max_results=max_results, backend=backend)
                            )
                        except TypeError:
                            # Some DDGS implementations do not accept a `backend` kwarg.
                            results = list(ddgs.text(query, max_results=max_results))
                        except Exception as exc:
                            backend_error = exc
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

                    if backend_error is not None:
                        raise backend_error
                    return []
        
        except Exception as exc:
            last_error = exc
            is_transient = _is_transient_error(exc)
            
            if not is_transient or attempt >= _RETRY_MAX_ATTEMPTS:
                # Permanent error or max retries reached
                logger.warning(
                    "DuckDuckGo search failed for query %r (attempt %d/%d, %s): %s",
                    query,
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
                "Transient search error for query %r (attempt %d/%d), retrying in %.1fs: %s",
                query,
                attempt,
                _RETRY_MAX_ATTEMPTS,
                wait_time,
                exc,
            )
            time.sleep(wait_time)
    
    logger.warning("DuckDuckGo search exhausted all %d attempts for query %r", _RETRY_MAX_ATTEMPTS, query)
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
