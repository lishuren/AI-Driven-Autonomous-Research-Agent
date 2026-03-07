"""
search_tool.py – Web search wrapper.

Primary backend: DuckDuckGo (no API key needed).
Optional fallback: Bing Web Search API — activated automatically when the
environment variable ``BING_SEARCH_API_KEY`` is set.  Bing is only called
when DuckDuckGo returns zero results.  Once Bing returns HTTP 403 or 429
(quota exceeded) it is disabled for the rest of the process lifetime so no
further requests are wasted.

Returns a list of result dicts: {'title': str, 'url': str, 'body': str}.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional search query logger
# ---------------------------------------------------------------------------

class SearchLogger:
    """Optional JSONL logger for search queries and result domains.

    Disabled by default.  Call ``SearchLogger.enable(path)`` once at startup
    (e.g. from main.py when --search-log is passed) to turn it on.

    Each line written is a JSON object::

        {
            "ts": "2026-03-07T15:23:54",
            "query": "Westworld S3 episode guide",
            "result_count": 4,
            "domains": ["imdb.com", "wikipedia.org", ...]
        }
    """

    _file: Optional[io.TextIOWrapper] = None

    @classmethod
    def enable(cls, path: str) -> None:
        """Open *path* for append and start logging.  Safe to call multiple times."""
        if cls._file is not None:
            return  # already enabled
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cls._file = log_path.open("a", encoding="utf-8", buffering=1)
        logger.info("Search query log enabled: %s", log_path)

    @classmethod
    def log(cls, query: str, results: list[dict[str, Any]]) -> None:
        """Write one log entry.  No-op when disabled."""
        if cls._file is None:
            return
        domains: list[str] = []
        for r in results:
            url = r.get("url", "")
            if url:
                parsed = urllib.parse.urlparse(url)
                host = parsed.netloc.lstrip("www.")
                if host and host not in domains:
                    domains.append(host)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query,
            "result_count": len(results),
            "domains": domains,
        }
        cls._file.write(json.dumps(entry) + "\n")

    @classmethod
    def close(cls) -> None:
        """Flush and close the log file."""
        if cls._file is not None:
            cls._file.flush()
            cls._file.close()
            cls._file = None

_RATE_LIMIT_MIN = 2.0
_RATE_LIMIT_MAX = 5.0
_DEFAULT_MAX_RESULTS = 5
# Backends for the legacy `duckduckgo_search` package.
_SEARCH_BACKENDS = ("lite", "html", "bing")
# Backends for the newer `ddgs` package.  "duckduckgo" is the primary.
# Candidates tried and rejected:
#   "auto"   — fires all engines in parallel; Google/Brave/Yahoo time out
#               (~40 s) when behind the GFW; mojeek returns 403.
#   "yandex" — consistently returns a CAPTCHA page (HTTP 200) for CJK
#               queries; ddgs parses 0 results and raises DDGSException.
# The ddgs built-in Bing engine has disabled=True and cannot be used here.
# For reliable CJK coverage set BING_SEARCH_API_KEY (see _bing_search_sync).
_SEARCH_BACKENDS_DDGS = ("duckduckgo",)
_SEARCH_BACKENDS_DDGS_CJK = ("duckduckgo",)  # same — no free engine beats DDG
# DDGS connection timeout (seconds).  10 s is enough for a single DDG request.
_DDGS_TIMEOUT_DEFAULT = 10
_DDGS_TIMEOUT_CJK = _DDGS_TIMEOUT_DEFAULT
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds


# URL substrings that indicate a captcha or block page rather than a real result.
_CAPTCHA_URL_MARKERS = (
    "showcaptcha",
    "/captcha",
    "?captcha",
    "&captcha",
    "validatecaptcha",
    "robot",
    "challenge",
)

# Unicode ranges that indicate CJK (Chinese/Japanese/Korean) content.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (most common Chinese/Japanese)
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B
    (0xAC00, 0xD7AF),   # Hangul Syllables (Korean)
    (0x3040, 0x30FF),   # Hiragana + Katakana (Japanese)
)


def _contains_cjk(text: str) -> bool:
    """Return True if *text* contains at least one CJK/Hangul/Kana character."""
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _CJK_RANGES):
            return True
    return False


# ---------------------------------------------------------------------------
# Bing Web Search API (optional fallback)
# ---------------------------------------------------------------------------

_BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
_BING_TIMEOUT = 10  # seconds
_bing_quota_exhausted: bool = False  # set to True on 403/429; process-scoped
_cjk_no_results_warned: bool = False  # emit the API-key hint at most once


def _bing_search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Call Bing Web Search API and return normalised results.

    Returns [] immediately if:
    - ``BING_SEARCH_API_KEY`` is not set in the environment.
    - The quota has already been exhausted this process run.
    - Any HTTP or network error occurs (logged, not raised).
    """
    global _bing_quota_exhausted
    if _bing_quota_exhausted:
        return []

    api_key = os.environ.get("BING_SEARCH_API_KEY", "").strip()
    if not api_key:
        return []

    market = "zh-CN" if _contains_cjk(query) else "en-US"
    params = urllib.parse.urlencode({
        "q": query,
        "count": max_results,
        "mkt": market,
        "responseFilter": "Webpages",
    })
    url = f"{_BING_ENDPOINT}?{params}"
    req = urllib.request.Request(url, headers={"Ocp-Apim-Subscription-Key": api_key})

    try:
        with urllib.request.urlopen(req, timeout=_BING_TIMEOUT) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            _bing_quota_exhausted = True
            logger.warning(
                "Bing search quota exhausted (HTTP %d) — Bing disabled for this run.",
                exc.code,
            )
        else:
            logger.warning("Bing search HTTP error for query %r: %s", query, exc)
        return []
    except Exception as exc:
        logger.warning("Bing search error for query %r: %s", query, exc)
        return []

    raw: list[dict[str, Any]] = []
    for page in payload.get("webPages", {}).get("value", []):
        raw.append({
            "title": page.get("name", ""),
            "url": page.get("url", ""),
            "body": page.get("snippet", ""),
        })
    results = _normalise_results(raw)
    if results:
        logger.info("Bing fallback returned %d results for query %r.", len(results), query)
    return results


# ---------------------------------------------------------------------------
# Tavily Search API (optional fallback, tried before Bing)
# ---------------------------------------------------------------------------

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_TAVILY_TIMEOUT = 15  # seconds — Tavily is slightly slower than Bing
_tavily_quota_exhausted: bool = False  # set to True on 401/429; process-scoped


def _tavily_search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Call Tavily Search API and return normalised results.

    Tavily is purpose-built for AI agents and handles CJK content well.
    Free tier: 1,000 requests/month — no credit card required.
    Sign up at https://app.tavily.com and export the key as ``TAVILY_API_KEY``.

    Returns [] immediately if:
    - ``TAVILY_API_KEY`` is not set in the environment.
    - The quota has already been exhausted this process run.
    - Any HTTP or network error occurs (logged, not raised).
    """
    global _tavily_quota_exhausted
    if _tavily_quota_exhausted:
        return []

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }).encode()
    req = urllib.request.Request(
        _TAVILY_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TAVILY_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 429):
            _tavily_quota_exhausted = True
            logger.warning(
                "Tavily search quota exhausted (HTTP %d) — Tavily disabled for this run.",
                exc.code,
            )
        else:
            logger.warning("Tavily search HTTP error for query %r: %s", query, exc)
        return []
    except Exception as exc:
        logger.warning("Tavily search error for query %r: %s", query, exc)
        return []

    raw: list[dict[str, Any]] = []
    for item in data.get("results", []):
        raw.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "body": item.get("content", ""),
        })
    results = _normalise_results(raw)
    if results:
        logger.info("Tavily fallback returned %d results for query %r.", len(results), query)
    return results


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

        # Drop results whose URL is a captcha / robot-challenge page.
        url_lower = url.lower()
        if any(marker in url_lower for marker in _CAPTCHA_URL_MARKERS):
            logger.info("Filtered captcha/block URL from results: %s", url[:80])
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

    # CJK queries need a locale hint so DuckDuckGo returns relevant results.
    is_cjk = _contains_cjk(query)
    region: Optional[str] = "cn-zh" if is_cjk else None
    # CJK auto-backend probes multiple search engines in parallel — needs more time.
    ddgs_timeout = _DDGS_TIMEOUT_CJK if is_cjk else _DDGS_TIMEOUT_DEFAULT

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
                with DDGS(timeout=ddgs_timeout) as ddgs:
                    if using_new_ddgs:
                        # Use the ddgs-specific backend list — its backend names differ
                        # from duckduckgo_search, and passing an unknown name causes the
                        # library to silently fall back to "auto" (all engines, very slow).
                        # CJK queries use an extended list: try "duckduckgo" first, then
                        # "auto" (Google/Brave/Yahoo/Bing in parallel) if no results.
                        backends = _SEARCH_BACKENDS_DDGS_CJK if is_cjk else _SEARCH_BACKENDS_DDGS
                        backend_error = None
                        for backend in backends:
                            try:
                                kwargs: dict[str, Any] = {
                                    "max_results": max_results,
                                    "backend": backend,
                                }
                                if region:
                                    kwargs["region"] = region
                                results = list(ddgs.text(query, **kwargs))
                            except TypeError:
                                # Older DDGS versions may not accept region/backend kwargs.
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
                            kwargs_legacy: dict[str, Any] = {
                                "max_results": max_results,
                                "backend": backend,
                            }
                            if region:
                                kwargs_legacy["region"] = region
                            results = list(ddgs.text(query, **kwargs_legacy))
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
    """Async-friendly wrapper around DuckDuckGo search with API fallbacks.

    Fallback chain when DuckDuckGo returns zero results:
    1. Tavily Search API — if ``TAVILY_API_KEY`` is set (recommended for CJK).
    2. Bing Web Search API — if ``BING_SEARCH_API_KEY`` is set.

    Both APIs offer a free tier of ~1,000 queries/month with no credit card.
    Either key (or both) can be set independently.
    """

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

        # API fallbacks — tried in order when DDG returned nothing.
        # 1. Tavily (purpose-built for AI agents, strong CJK coverage)
        if not results:
            try:
                results = await loop.run_in_executor(
                    None, _tavily_search_sync, query, self.max_results
                )
            except Exception as exc:
                logger.warning("Tavily fallback executor error for query %r: %s", query, exc)
                results = []

        # 2. Bing Web Search API
        if not results:
            try:
                results = await loop.run_in_executor(
                    None, _bing_search_sync, query, self.max_results
                )
            except Exception as exc:
                logger.warning("Bing fallback executor error for query %r: %s", query, exc)
                results = []

        self._last_call = time.monotonic()
        logger.info("Search for %r returned %d results.", query, len(results))
        SearchLogger.log(query, results)

        # One-time advisory when every backend returned nothing and no API
        # keys are configured.  Most actionable signal in the log.
        if not results and _contains_cjk(query):
            global _cjk_no_results_warned
            if not _cjk_no_results_warned:
                _cjk_no_results_warned = True
                tavily_set = bool(os.environ.get("TAVILY_API_KEY", "").strip())
                bing_set = bool(os.environ.get("BING_SEARCH_API_KEY", "").strip())
                if not tavily_set and not bing_set:
                    logger.warning(
                        "CJK search returned 0 results and no API fallback keys are "
                        "set.  DuckDuckGo may be rate-limiting this host for "
                        "Chinese/Japanese/Korean queries.  Set one of:\n"
                        "  TAVILY_API_KEY     — https://app.tavily.com (recommended, "
                        "purpose-built for AI agents)\n"
                        "  BING_SEARCH_API_KEY — Azure Cognitive Services\n"
                        "Both offer ~1,000 free queries/month with no credit card."
                    )

        return results
