"""
researcher.py – Search & Scrape agent.

For a given task query:
1. Searches using SearchTool (DuckDuckGo).
2. Scrapes the top N result URLs with ScraperTool.
3. Passes the combined raw text to Ollama for a concise technical summary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Any, Optional

from src.tools.scraper_tool import ScraperTool
from src.tools.search_tool import SearchTool

logger = logging.getLogger(__name__)

_SUMMARISE_PROMPT = """You are a senior software engineer writing technical research notes.

Task: {task}

Below is raw scraped content from multiple web pages.  Your job is to distil this into
a concise (≤500 words) TECHNICAL summary that:
- Retains all mathematical formulas (LaTeX notation preferred).
- Lists every required Python library.
- Describes the algorithm in step-by-step pseudocode.
- Includes any relevant code snippets verbatim.
- DISCARDS marketing copy, opinions, and redundant prose.

Raw content:
{raw_content}

Technical summary:"""

_SCRAPE_TOP_N = 4


class ResearcherAgent:
    """Searches the web, scrapes pages, and produces a technical summary."""

    def __init__(
        self,
        model: str = "llama3",
        ollama_base_url: str = "http://localhost:11434",
        max_search_results: int = _SCRAPE_TOP_N,
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.max_search_results = max_search_results

        self._search = SearchTool(max_results=max_search_results)
        self._scraper = ScraperTool()

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Synchronous Ollama call; returns the model's text response."""
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                return data.get("response", "")
        except Exception as exc:
            logger.warning("Ollama summarisation failed: %s", exc)
            return None

    async def research(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute search + scrape + summarise for *task*.

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

        # 2. Scrape top N results concurrently
        urls = [r["url"] for r in results if r.get("url")]
        scrape_tasks = [self._scraper.scrape(url) for url in urls]
        scraped = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        source_urls = []
        raw_parts: list[str] = []
        for url, content in zip(urls, scraped):
            if isinstance(content, str) and content:
                source_urls.append(url)
                raw_parts.append(f"--- Source: {url} ---\n{content[:4000]}")

        raw_content = "\n\n".join(raw_parts) or "\n".join(
            r.get("body", "") for r in results
        )

        # 3. Summarise with Ollama
        prompt = _SUMMARISE_PROMPT.format(task=subtopic, raw_content=raw_content[:12000])
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
