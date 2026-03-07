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

Your task: Given the high-level topic below, decompose it into exactly 5 distinct,
specific sub-topics or research questions. Each query should be directly relevant 
to the topic as stated by the user - do NOT add extra context or reframe the topic 
unless necessary for clarity.

Guidelines:
- If the topic is technical, focus on implementation, formulas, and algorithms.
- If the topic is general knowledge, focus on facts, history, or key concepts.
- If the topic is about a TV series, focus on plot, characters, episodes, and events.
- Stay faithful to the user's intent - don't add dimensions they didn't ask for.

GOOD example for "Westworld S3 and S4": "Westworld season 3 plot summary main events"
BAD example: "machine learning model Python implementation for predicting Westworld season ratings"

GOOD example for "RSI trading": "RSI indicator mathematical formula Python pandas"
BAD example: "What is RSI?"

Topic: {topic}

Already researched topics (avoid repeating): {known_topics}

Respond ONLY with a JSON array, no commentary:
[
  {{"subtopic": "<name>", "query": "<specific search query>"}},
  ...
]
"""

_FOLLOWUP_PROMPT = """You are an expert research planner.

The Critic Agent has REJECTED the following research and identified gaps:
Topic: {topic}
Gaps identified: {gaps}

Generate ONE highly specific follow-up search query that would directly address
these gaps and provide the missing technical details.

Respond ONLY with a JSON object:
{{"subtopic": "{topic} (refined)", "query": "<specific follow-up search query>"}}
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
            logger.info("Planner refined task for %r: %s", topic, task["query"])
            return task

        # Fallback: build a query from the gaps description
        return {
            "subtopic": f"{topic} (refined)",
            "query": f"{topic} {gaps} python implementation formula",
        }
