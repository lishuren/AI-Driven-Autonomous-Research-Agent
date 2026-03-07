"""
planner.py – Topic decomposition agent.

Breaks a high-level research topic into concrete, specific technical
questions suitable for targeted web searches.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """You are an expert research decomposition specialist.

Your task: Decompose the user's topic into exactly 5 distinct, specific search queries.
Each query must be DIRECT and to the point - do NOT embellish with phrases like 
"quality source", "detailed analysis", "comprehensive overview", etc.

RULES (critical):
1. Search queries should be SHORT and FOCUSED - just the core search terms.
2. Do NOT add adjectives like "detailed", "comprehensive", "quality", "advanced".
3. Do NOT add phrases like "in research", "from sources", "for analysis".
4. Do NOT reframe or expand the user's topic unless they explicitly asked for it.
5. Queries should work as-is in a search engine with minimal words.

GOOD queries (SHORT, FOCUSED):
  - "Westworld season 3 plot"
  - "RSI indicator formula Python"
  - "neural network reinforcement learning"

BAD queries (EMBELLISHED, VERBOSE):
  - "Detailed neural network architectures for quality analysis"
  - "comprehensive Westworld research from authoritative sources"
  - "in-depth study of RSI mathematical foundations"

Topic: {topic}

Already researched: {known_topics}

Respond ONLY with this JSON structure, no extra text:
[
  {{"subtopic": "<short name>", "query": "<short search query>"}},
  {{"subtopic": "<short name>", "query": "<short search query>"}},
  {{"subtopic": "<short name>", "query": "<short search query>"}},
  {{"subtopic": "<short name>", "query": "<short search query>"}},
  {{"subtopic": "<short name>", "query": "<short search query>"}}
]
"""

_FOLLOWUP_PROMPT = """You are an expert research planner.

The Critic Agent REJECTED this research due to gaps:
Topic: {topic}
Gaps: {gaps}

Generate ONE short, focused follow-up search query that addresses ONLY the gaps.
Do NOT add embellishments like "detailed", "quality", "comprehensive", "in research".
Keep the query SHORT - just the core search terms needed.

Respond ONLY with this JSON:
{{"subtopic": "{topic} (follow-up)", "query": "<short focused search query>"}}
"""


def _call_ollama(prompt: str, model: str, base_url: str) -> Optional[str]:
    """Synchronous call to Ollama REST API; returns the response text."""
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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
            logger.warning("Ollama call failed (%s): %s", exc.code, error_text)
        else:
            logger.warning("Ollama call failed (%s): %s", exc.code, exc.reason)
        return None
    except Exception as exc:
        logger.warning("Ollama call failed: %s", exc)
        return None


class PlannerAgent:
    """Decomposes high-level topics into concrete search tasks."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url

    def _clean_query(self, query: str) -> str:
        """Remove unnecessary embellishments from LLM-generated queries."""
        # Words/phrases to remove (case-insensitive)
        embellishments = {
            "detailed", "detailed analysis", "comprehensive", "comprehensive overview",
            "in-depth", "in-depth study", "quality", "quality source", "quality sources",
            "authoritative source", "authoritative sources", "from research",
            "in research", "for analysis", "for deeper understanding",
            "advanced study", "scholarly", "academic research", "expert guide",
            "step by step", "step-by-step",
        }
        
        result = query.lower()
        # Remove embellishments
        for emb in embellishments:
            result = result.replace(f" {emb} ", " ").replace(f"{emb} ", "").replace(f" {emb}", "")
        
        # Clean up multiple spaces
        while "  " in result:
            result = result.replace("  ", " ")
        
        return result.strip().capitalize() if result else query

    def _parse_json(self, text: Optional[str]) -> Any:
        """Extract the first JSON structure from *text*."""
        if not text:
            return None
        # Find first '[' or '{'
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        return None

    def _fallback_tasks(self, topic: str) -> list[dict[str, str]]:
        """Return default tasks when the LLM is unavailable."""
        # Generate topic-agnostic fallback queries
        keywords = [
            f"{topic} overview summary",
            f"{topic} main concepts key points",
            f"{topic} examples use cases",
            f"{topic} advantages disadvantages comparison",
            f"{topic} best practices tips tutorial",
        ]
        return [{"subtopic": topic, "query": q} for q in keywords]

    async def decompose(
        self, topic: str, known_topics: Optional[list[str]] = None
    ) -> list[dict[str, str]]:
        """Return a list of ``{'subtopic': …, 'query': …}`` dicts."""
        known = ", ".join(known_topics or []) or "none"
        prompt = _DECOMPOSE_PROMPT.format(topic=topic, known_topics=known)

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, _call_ollama, prompt, self.model, self.ollama_base_url
        )

        tasks = self._parse_json(raw)
        if isinstance(tasks, list) and tasks:
            # Clean all queries to remove embellishments
            for task in tasks:
                if isinstance(task, dict) and "query" in task:
                    task["query"] = self._clean_query(task["query"])
            logger.info("Planner generated %d tasks for topic %r.", len(tasks), topic)
            return tasks

        logger.warning(
            "Planner LLM returned unusable output; using fallback tasks for %r.", topic
        )
        return self._fallback_tasks(topic)

    async def refine(self, topic: str, gaps: str) -> dict[str, str]:
        """Generate a targeted follow-up task to fill the identified *gaps*."""
        prompt = _FOLLOWUP_PROMPT.format(topic=topic, gaps=gaps)

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, _call_ollama, prompt, self.model, self.ollama_base_url
        )

        task = self._parse_json(raw)
        if isinstance(task, dict) and "query" in task:
            # Clean query to remove embellishments
            task["query"] = self._clean_query(task["query"])
            logger.info("Planner refined task for %r: %s", topic, task["query"])
            return task

        # Fallback: build a simple query from topic and gaps
        return {
            "subtopic": f"{topic} (refined)",
            "query": self._clean_query(f"{topic} {gaps}"),
        }
