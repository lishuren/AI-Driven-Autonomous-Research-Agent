"""
researcher.py – Search & Scrape agent.

For a given task query:
1. Searches using SearchTool (Tavily).
2. Uses Tavily raw_content (markdown) when available.
3. Falls back to Tavily Extract for URLs lacking raw_content.
4. Falls back to Playwright scraping when Extract is unavailable.
5. Passes the combined raw text to Ollama for a concise summary.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from src.llm_client import generate_text
from src.prompt_loader import load_prompt
from src.tools.scraper_tool import ScraperTool
from src.tools.search_tool import SearchTool, _detect_language

logger = logging.getLogger(__name__)

_SUMMARISE_PROMPT_FILE = "researcher_summarise.txt"

_SCRAPE_TOP_N = 4
_MIN_RAW_CONTENT_CHARS = 500  # threshold for "sufficient" Tavily raw_content
_PER_PAGE_CAP = 4000


class ResearcherAgent:
    """Searches the web, scrapes pages, and produces a factual summary."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
        max_search_results: int = _SCRAPE_TOP_N,
        user_prompt: Optional[str] = None,
        llm_provider: str = "ollama",
        llm_api_key: Optional[str] = None,
        prompt_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.max_search_results = max_search_results
        self._user_prompt = user_prompt
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._summarise_prompt = load_prompt(_SUMMARISE_PROMPT_FILE, prompt_dir)

        self._search = SearchTool(max_results=max_search_results)
        self._scraper = ScraperTool()

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Synchronous LLM call; returns the model's text response."""
        return generate_text(
            prompt,
            self.model,
            self.ollama_base_url,
            provider=self._llm_provider,
            api_key=self._llm_api_key,
            timeout=180,
        )

    async def research(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute search + content acquisition + summarise for *task*.

        Content acquisition flow:
        1. Use Tavily ``raw_content`` (markdown) when the search already returned
           sufficient page text (≥ ``_MIN_RAW_CONTENT_CHARS`` per result).
        2. For URLs lacking raw_content, try Tavily Extract (1 credit / 5 URLs).
        3. Fall back to Playwright scraping for any remaining URLs.
        4. If everything fails, use search-result snippets.

        Returns a dict with keys: ``subtopic``, ``query``, ``summary``,
        ``source_urls``, ``raw_content``.
        """
        query: str = task.get("query", task.get("subtopic", ""))
        subtopic: str = task.get("subtopic", query)

        logger.info("Researching: %r", query)

        # 1. Search
        results = await self._search.search(query)
        if not results:
            logger.warning("No search results for %r.", query)
            return {
                "subtopic": subtopic,
                "query": query,
                "summary": "",
                "source_urls": [],
                "raw_content": "",
            }

        # 2. Content acquisition (conditional scraping)
        source_urls: list[str] = []
        raw_parts: list[str] = []
        urls_needing_content: list[str] = []

        # 2a. Check Tavily raw_content first
        for r in results:
            url = r.get("url", "")
            if not url:
                continue
            raw_content = r.get("raw_content", "")
            if raw_content and len(raw_content) >= _MIN_RAW_CONTENT_CHARS:
                source_urls.append(url)
                raw_parts.append(
                    f"--- Source: {url} ---\n{raw_content[:_PER_PAGE_CAP]}"
                )
            else:
                urls_needing_content.append(url)

        # 2b. Try Tavily Extract for URLs lacking raw_content
        if urls_needing_content:
            try:
                extracted = await self._search.extract(urls_needing_content, query)
                extracted_urls: set[str] = set()
                for item in extracted:
                    eurl = item.get("url", "")
                    content = item.get("content", "")
                    if content and eurl:
                        source_urls.append(eurl)
                        raw_parts.append(
                            f"--- Source: {eurl} ---\n{content[:_PER_PAGE_CAP]}"
                        )
                        extracted_urls.add(eurl)
                # Remaining URLs that Extract didn't cover
                urls_needing_content = [
                    u for u in urls_needing_content if u not in extracted_urls
                ]
            except Exception as exc:
                logger.warning("Tavily Extract failed: %s", exc)

        # 2c. Fall back to Playwright for remaining URLs
        if urls_needing_content:
            scrape_tasks = [self._scraper.scrape(url) for url in urls_needing_content]
            scraped = await asyncio.gather(*scrape_tasks, return_exceptions=True)
            for url, content in zip(urls_needing_content, scraped):
                if isinstance(content, str) and content:
                    source_urls.append(url)
                    raw_parts.append(
                        f"--- Source: {url} ---\n{content[:_PER_PAGE_CAP]}"
                    )

        raw_content = "\n\n".join(raw_parts) or "\n".join(
            r.get("body", "") for r in results
        )

        # 3. Summarise with Ollama
        user_context = (
            f"User instructions:\n{self._user_prompt}\n"
            if self._user_prompt else ""
        )
        lang = _detect_language(query)
        language_hint = (
            "Write the summary in Chinese (中文)." if lang == "zh" else ""
        )
        prompt = self._summarise_prompt.format(
            task=subtopic, raw_content=raw_content[:12000],
            user_context=user_context,
            language_hint=language_hint,
        )
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, self._call_ollama, prompt)

        if not summary:
            # Fallback: use search result snippets
            summary = "\n".join(r.get("body", "") for r in results[:3])
            logger.warning("Falling back to raw snippets for %r.", query)

        return {
            "subtopic": subtopic,
            "query": query,
            "summary": summary,
            "source_urls": source_urls,
            "raw_content": raw_content,
        }
