"""
search_tool.py – Web search wrapper.

Backend: Tavily Search API (requires ``TAVILY_API_KEY`` env var or
``--tavily-key`` CLI argument).  Returns a list of result dicts:
``{'title': str, 'url': str, 'body': str}``.
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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.budget import BudgetTracker

from src.config_loader import get_filters_config

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


_cjk_no_results_warned: bool = False  # emit the API-key hint at most once


def _detect_language(text: str) -> str:
    """Detect the primary language of *text*.

    Returns ``"zh"`` when >30 % of characters are CJK, else ``"en"``.
    """
    if not text:
        return "en"
    cjk_count = sum(
        1 for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES)
    )
    return "zh" if cjk_count / len(text) > 0.30 else "en"


# ---------------------------------------------------------------------------
# Tavily Search API (optional fallback, tried before Bing)
# ---------------------------------------------------------------------------

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_TAVILY_TIMEOUT = 15  # seconds — Tavily is slightly slower than Bing
_tavily_quota_exhausted: bool = False  # set to True on 401/429; process-scoped

# Module-level budget tracker — set via set_budget() before searches begin.
_budget: Optional["BudgetTracker"] = None

# Dry-run mode — set via set_dry_run(); all searches return [] without HTTP calls.
_dry_run: bool = False


def set_budget(budget: "BudgetTracker") -> None:
    """Attach a :class:`BudgetTracker` so every search records credit usage."""
    global _budget
    _budget = budget


def set_dry_run(enabled: bool = True) -> None:
    """Enable or disable dry-run mode.

    When enabled, :meth:`SearchTool.search` and :meth:`SearchTool.extract`
    return ``[]`` immediately without making any HTTP calls to Tavily.
    Use this for ``--dry-run`` / ``--estimate-credits`` mode to estimate
    query costs via LLM decomposition only.
    """
    global _dry_run
    _dry_run = enabled


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

    body: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_usage": True,
        "include_raw_content": "markdown",
    }
    # Region hint for Chinese queries
    if _detect_language(query) == "zh":
        body["country"] = "china"

    payload = json.dumps(body).encode()
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
        if exc.code in (401, 429, 432):
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

    # Track API credit usage via the budget tracker
    usage = data.get("usage", {})
    credits = usage.get("credits", 1)  # default to 1 credit per search
    if _budget is not None:
        _budget.record_query(credits=credits)

    raw: list[dict[str, Any]] = []
    for item in data.get("results", []):
        entry: dict[str, Any] = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "body": item.get("content", ""),
        }
        # Tavily returns raw_content when include_raw_content is set
        raw_content = item.get("raw_content") or ""
        if raw_content:
            entry["raw_content"] = raw_content
        raw.append(entry)
    results = _normalise_results(raw)
    if results:
        logger.info("Tavily returned %d results for query %r.", len(results), query)
    return results


def _normalise_results(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert provider-specific result dictionaries into a stable schema."""
    normalised: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue

        url = str(item.get("url") or item.get("href") or "")
        title = str(item.get("title") or item.get("heading") or "")
        body = str(item.get("body") or item.get("snippet") or item.get("content") or "")

        if not (url or title or body):
            continue

        # Drop results whose URL is a captcha / robot-challenge page.
        url_lower = url.lower()
        if any(marker in url_lower for marker in get_filters_config()["captcha_url_markers"]):
            logger.info("Filtered captcha/block URL from results: %s", url[:80])
            continue

        entry = {"title": title, "url": url, "body": body}
        # Preserve raw_content from Tavily if present
        raw_content = item.get("raw_content", "")
        if raw_content:
            entry["raw_content"] = raw_content
        normalised.append(entry)
    return normalised


# ---------------------------------------------------------------------------
# Tavily Extract API
# ---------------------------------------------------------------------------

_TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"
_TAVILY_EXTRACT_TIMEOUT = 20  # seconds — extraction is slower than search


def _tavily_extract_sync(
    urls: list[str], query: str = "",
) -> list[dict[str, Any]]:
    """Call Tavily Extract API and return extracted content for each URL.

    Returns a list of dicts: ``{'url': str, 'content': str}``.
    Costs 1 credit per 5 URLs. Returns [] on error or missing key.
    """
    global _tavily_quota_exhausted
    if _tavily_quota_exhausted:
        return []

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    body: dict[str, Any] = {
        "api_key": api_key,
        "urls": urls[:20],  # Tavily allows up to 20 URLs per call
        "include_usage": True,
        "extract_depth": "basic",
        "format": "markdown",
    }
    if query:
        body["query"] = query

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        _TAVILY_EXTRACT_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TAVILY_EXTRACT_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 429, 432):
            _tavily_quota_exhausted = True
            logger.warning(
                "Tavily extract quota exhausted (HTTP %d).", exc.code,
            )
        else:
            logger.warning("Tavily extract HTTP error: %s", exc)
        return []
    except Exception as exc:
        logger.warning("Tavily extract error: %s", exc)
        return []

    # Track credit usage
    usage = data.get("usage", {})
    credits = usage.get("credits", len(urls) / 5.0)
    if _budget is not None:
        _budget.record_query(credits=credits)

    results: list[dict[str, Any]] = []
    for item in data.get("results", []):
        content = item.get("raw_content") or item.get("content") or ""
        url = item.get("url", "")
        if content and url:
            results.append({"url": url, "content": content})

    logger.info(
        "Tavily Extract returned content for %d / %d URLs.", len(results), len(urls),
    )
    return results


# ---------------------------------------------------------------------------
# Tavily account credit usage
# ---------------------------------------------------------------------------

_TAVILY_USAGE_ENDPOINT = "https://api.tavily.com/usage"


def _fetch_account_credits_sync() -> Optional[dict[str, Any]]:
    """Call the Tavily /usage endpoint and return account credit info.

    Returns a dict such as::

        {
            "credits_used":      123,
            "credits_limit":    1000,
            "credits_remaining": 877,
        }

    Returns ``None`` if the API key is absent, the endpoint is unavailable,
    or any network/HTTP error occurs.  This is a best-effort call — the
    response format may vary by Tavily plan.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None

    req = urllib.request.Request(
        f"{_TAVILY_USAGE_ENDPOINT}?api_key={urllib.parse.quote(api_key)}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        # Normalise varying field names across Tavily API versions
        result: dict[str, Any] = {}
        for src_key, dst_key in (
            ("used", "credits_used"),
            ("credits_used", "credits_used"),
            ("limit", "credits_limit"),
            ("credits_limit", "credits_limit"),
            ("remaining", "credits_remaining"),
            ("credits_remaining", "credits_remaining"),
        ):
            if src_key in data and dst_key not in result:
                result[dst_key] = data[src_key]
        return result if result else data  # return raw if no known fields matched
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 405, 501):
            logger.debug(
                "Tavily /usage endpoint not available (HTTP %d) — "
                "account balance will not be shown.",
                exc.code,
            )
        else:
            logger.debug("Tavily /usage HTTP error: %s", exc)
        return None
    except Exception as exc:
        logger.debug("Tavily /usage fetch failed: %s", exc)
        return None


async def fetch_account_credits() -> Optional[dict[str, Any]]:
    """Async wrapper around :func:`_fetch_account_credits_sync`.

    Fetches current Tavily account credit usage from the ``/usage`` endpoint.
    Returns the credit dict or ``None`` if unavailable.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_account_credits_sync)


class SearchTool:
    """Async-friendly wrapper around the Tavily Search API.

    Requires ``TAVILY_API_KEY`` environment variable (or ``--tavily-key`` CLI
    arg).  When the key is missing, searches return ``[]`` with a warning.
    """

    def __init__(self, max_results: int = _DEFAULT_MAX_RESULTS) -> None:
        self.max_results = max_results
        self._last_call: float = 0.0

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search for *query* via Tavily and return a list of result dicts.

        Applies randomised rate-limiting between consecutive calls.
        Returns [] immediately if the budget is exhausted or in dry-run mode.
        """
        # Dry-run mode — no HTTP calls, just return empty results
        if _dry_run:
            logger.debug("Dry-run: skipping search for %r.", query)
            return []

        # Budget guard — refuse to spend credits when limit reached
        if _budget is not None and not _budget.can_query():
            logger.info("Budget exhausted — skipping search for %r.", query)
            return []

        elapsed = time.monotonic() - self._last_call
        sleep_time = random.uniform(_RATE_LIMIT_MIN, _RATE_LIMIT_MAX)
        if elapsed < sleep_time:
            await asyncio.sleep(sleep_time - elapsed)

        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None, _tavily_search_sync, query, self.max_results
            )
        except Exception as exc:
            logger.warning("Search executor error for query %r: %s", query, exc)
            results = []

        self._last_call = time.monotonic()
        logger.info("Search for %r returned %d results.", query, len(results))
        SearchLogger.log(query, results)

        # One-time advisory when Tavily returned nothing and no key is set.
        if not results and _contains_cjk(query):
            global _cjk_no_results_warned
            if not _cjk_no_results_warned:
                _cjk_no_results_warned = True
                tavily_set = bool(os.environ.get("TAVILY_API_KEY", "").strip())
                if not tavily_set:
                    logger.warning(
                        "CJK search returned 0 results and TAVILY_API_KEY is not "
                        "set.  Set it via --tavily-key or the environment variable.\n"
                        "  Sign up at https://app.tavily.com (1,000 free queries/month)."
                    )

        return results

    async def extract(self, urls: list[str], query: str = "") -> list[dict[str, Any]]:
        """Extract full page content via Tavily Extract API.

        Returns a list of ``{'url': str, 'content': str}`` dicts.
        Skips the call when the budget is exhausted or in dry-run mode.
        """
        if _dry_run:
            logger.debug("Dry-run: skipping Tavily Extract for %d URLs.", len(urls))
            return []

        if _budget is not None and not _budget.can_query():
            logger.info("Budget exhausted — skipping Tavily Extract.")
            return []
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, _tavily_extract_sync, urls, query,
            )
        except Exception as exc:
            logger.warning("Tavily Extract executor error: %s", exc)
            return []
