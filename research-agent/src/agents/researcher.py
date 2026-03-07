"""
researcher.py – Search & Scrape agent.

For a given task query:
1. Searches using SearchTool (DuckDuckGo).
2. Scrapes the top N result URLs with ScraperTool.
3. Passes the combined raw text to Ollama for a concise summary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from src.tools.scraper_tool import ScraperTool
from src.tools.search_tool import SearchTool

logger = logging.getLogger(__name__)

_SUMMARISE_PROMPT = """You are a research assistant producing concise, factual notes.

Task: {task}
{user_context}
Below is raw scraped content from multiple web pages.  Write a concise summary
(≤500 words) that directly answers the task.  Rules:
- Report only facts that are actually present in the raw content below.
- Do NOT invent, extrapolate, or hallucinate any details not in the source.
- Match the style to the topic: for technical topics include formulas, algorithms,
  and code only when genuinely present in the source; for general/entertainment
  topics write plain prose describing what the sources say.
- If the sources do not contain useful information for the task, explicitly say so.
- DISCARD marketing copy, opinions, and redundant prose.

Raw content:
{raw_content}

Summary:"""

_SCRAPE_TOP_N = 4


class ResearcherAgent:
    """Searches the web, scrapes pages, and produces a factual summary."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
        max_search_results: int = _SCRAPE_TOP_N,
        user_prompt: Optional[str] = None,
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.max_search_results = max_search_results
        self._user_prompt = user_prompt

        self._search = SearchTool(max_results=max_search_results)
        self._scraper = ScraperTool()

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Synchronous Ollama call; returns the model's text response."""
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.ollama_base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                return data.get("response", "")
        except urllib.error.HTTPError as exc:
            error_text = ""
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="ignore"))
                if isinstance(payload, dict):
                    error_text = str(payload.get("error", ""))
            except Exception:
                error_text = ""

            if error_text:
                logger.warning("Ollama summarisation failed (%s): %s", exc.code, error_text)
            else:
                logger.warning("Ollama summarisation failed (%s): %s", exc.code, exc.reason)
            return None
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
        user_context = (
            f"User instructions:\n{self._user_prompt}\n"
            if self._user_prompt else ""
        )
        prompt = _SUMMARISE_PROMPT.format(
            task=subtopic, raw_content=raw_content[:12000],
            user_context=user_context,
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
