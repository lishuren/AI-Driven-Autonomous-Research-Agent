"""
hub_scraper_tool.py – Optional Scrapling-backed hub-page content extractor.

When a Tavily search result points to a landing page, index, directory, or
listing page (rather than the actual article/report), this tool:
  1. Fetches the hub page with Scrapling's sync ``Fetcher``.
  2. Extracts same-domain candidate links.
  3. Scores each link by keyword overlap with the research query.
  4. Fetches the best-scoring detail page.
  5. Returns ``(detail_url, extracted_text)`` for downstream summarisation.

This is an opt-in feature disabled by default:
  Install:  pip install "scrapling[fetchers]" && scrapling install

The module follows the same module-level flag/setter pattern as
``scraper_tool.py`` (``_no_scrape`` / ``set_no_scrape``), and budget tracking
follows the same pattern as ``search_tool.py`` (``_budget`` / ``set_budget``).
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.budget import BudgetTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Scrapling import (same graceful-degradation pattern as Playwright)
# ---------------------------------------------------------------------------

try:
    from scrapling.fetchers import Fetcher as _ScraplingFetcher
    _HAS_SCRAPLING = True
except ImportError:  # pragma: no cover
    _ScraplingFetcher = None  # type: ignore[assignment]
    _HAS_SCRAPLING = False
    logger.info("scrapling not installed — hub scraper disabled.")

# Budget tracker reference — set via set_budget()
_budget: Optional["BudgetTracker"] = None

# Content cap per page (match existing _PER_PAGE_CAP × 5 for full-page text)
_MAX_HUB_CONTENT_CHARS = 20_000

# ---------------------------------------------------------------------------
# Hub-page heuristics
# ---------------------------------------------------------------------------

_HUB_PATH_SEGMENTS = frozenset({
    "index", "category", "categories", "archive", "archives",
    "tag", "tags", "topics", "topic", "search", "results",
    "directory", "listing", "listings", "feed", "feeds",
    "hub", "sitemap", "overview", "explore", "browse",
    "news",  # root /news/ is typically a hub
})

_HUB_TITLE_KEYWORDS = frozenset({
    "index", "category", "categories", "archive", "archives",
    "tag", "directory", "search results", "latest", "all posts",
    "recent posts", "overview", "browse", "topics", "listing",
    "sitemap", "home page", "homepage",
})

# Anchor hrefs containing any of these are navigation noise, not content
_LINK_EXCLUDE_SUBSTRINGS = frozenset({
    "login", "signin", "sign-in", "signup", "sign-up", "register",
    "logout", "account", "profile", "privacy", "terms", "cookie",
    "advertise", "about", "contact", "careers", "sitemap",
    "javascript:", "mailto:", "#",
})


def set_budget(budget: "BudgetTracker") -> None:
    """Attach a :class:`~src.budget.BudgetTracker` to record hub-fetch credits."""
    global _budget
    _budget = budget


def is_hub_url(url: str, title: str = "", snippet: str = "") -> bool:
    """Return True when *url* / *title* look like a hub, index, or listing page.

    Uses deterministic URL and title heuristics only — no LLM involved.
    Fast enough to run inline for every unresolved URL before any fetch.
    """
    parsed = urllib.parse.urlparse(url.lower())
    path_parts = {p for p in parsed.path.split("/") if p}

    # URL path segment match
    if path_parts & _HUB_PATH_SEGMENTS:
        return True

    # Root domain with no significant path → homepage / hub
    if not parsed.path.strip("/"):
        return True

    # Title keyword match
    title_lower = title.lower()
    if any(kw in title_lower for kw in _HUB_TITLE_KEYWORDS):
        return True

    return False


# ---------------------------------------------------------------------------
# Scrapling-backed detail fetcher
# ---------------------------------------------------------------------------

def _fetch_hub_detail_sync(
    url: str,
    query: str,
) -> tuple[Optional[str], Optional[str]]:
    """Synchronous implementation — run via ``loop.run_in_executor``.

    1. Fetch the hub page with Scrapling ``Fetcher.get()``.
    2. Extract all ``<a>`` elements; keep same-domain, non-noise links.
    3. Score each by query-keyword overlap in link text + href.
    4. Fetch the best-scoring detail page.
    5. Return ``(detail_url, text)`` or ``(None, None)`` on any error.
    """
    # Budget guard — pre-flight (hub fetch costs 1 credit)
    if _budget is not None and not _budget.can_query():
        logger.info("Budget exhausted — skipping hub scrape for %r.", url)
        return None, None

    # --- Fetch hub page ---
    try:
        hub_page = _ScraplingFetcher.get(url, stealthy_headers=True)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("Scrapling hub fetch failed for %r: %s", url, exc)
        return None, None

    if _budget is not None:
        _budget.record_query(credits=1)

    # --- Extract and score candidate links ---
    parsed_hub = urllib.parse.urlparse(url)
    hub_domain = parsed_hub.netloc.lower()
    query_words = frozenset(w.lower() for w in query.split() if len(w) > 2)

    best_url: Optional[str] = None
    best_score: int = -1

    try:
        anchors = hub_page.css("a")
    except Exception:
        anchors = []

    for anchor in anchors:
        try:
            href = (anchor.attrib.get("href") or "").strip()
        except Exception:
            continue

        if not href:
            continue

        # Resolve relative URLs and normalise
        resolved = urllib.parse.urljoin(url, href)
        parsed_link = urllib.parse.urlparse(resolved)

        # Same-domain enforcement
        if parsed_link.netloc.lower() != hub_domain:
            continue

        # HTTP(S) only
        if parsed_link.scheme not in ("http", "https"):
            continue

        # Exclude navigation / account noise
        href_lower = resolved.lower()
        if any(excl in href_lower for excl in _LINK_EXCLUDE_SUBSTRINGS):
            continue

        # Do not loop back to the hub page itself
        if resolved.rstrip("/") == url.rstrip("/"):
            continue

        # Score by query keyword overlap in link text + href
        try:
            link_text = " ".join(anchor.css("::text").getall()).lower()
        except Exception:
            link_text = ""
        combined = link_text + " " + href_lower
        score = sum(1 for w in query_words if w in combined)

        if score > best_score:
            best_score = score
            best_url = resolved

    if best_url is None:
        logger.info("No suitable detail link found on hub page %r.", url)
        return None, None

    # Budget guard for the second (detail) fetch
    if _budget is not None and not _budget.can_query():
        logger.info("Budget exhausted — skipping detail fetch for %r.", best_url)
        return None, None

    # --- Fetch detail page ---
    try:
        detail_page = _ScraplingFetcher.get(best_url, stealthy_headers=True)  # type: ignore[attr-defined]
        text_parts: list[str] = detail_page.css("*::text").getall()
        text = "\n".join(t.strip() for t in text_parts if t.strip())
        text = text[:_MAX_HUB_CONTENT_CHARS]
    except Exception as exc:
        logger.warning("Scrapling detail fetch failed for %r: %s", best_url, exc)
        return None, None

    if _budget is not None:
        _budget.record_query(credits=1)

    logger.info(
        "Hub %r → detail %r (%d chars extracted).", url, best_url, len(text)
    )
    return best_url, text or None


async def fetch_hub_detail(
    url: str,
    query: str,
) -> tuple[Optional[str], Optional[str]]:
    """Async wrapper around :func:`_fetch_hub_detail_sync`.

    Fetches *url* as a hub page, follows the best same-domain detail link,
    and returns ``(detail_url, extracted_text)`` or ``(None, None)`` when:
    - No suitable detail link is found
    - Any network or parsing error occurs
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_hub_detail_sync, url, query)
