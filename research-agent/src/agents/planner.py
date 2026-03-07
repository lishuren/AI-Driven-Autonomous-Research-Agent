"""
planner.py – Topic decomposition agent.

Breaks a high-level research topic into concrete search tasks.
Supports:
  - Pre-search vocabulary grounding (real web terms fed into prompts)
  - Feedback loop (successful/failed query examples passed back in)
  - Retrospective re-plan when the agent is stuck
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.tools.search_tool import SearchTool

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """You are a search query generator. Convert a topic into 5 search queries.

Rules:
- Each query must use ONLY words already present in the topic, plus these search helpers if needed:
  season, episode, cast, plot, summary, review, release, explained, how, why, list, vs,
  history, tutorial, formula, algorithm, code, example, install, python, wiki, imdb, rating
- DO NOT invent new words or concepts not found in the topic.
- DO NOT add adjectives or qualifiers the user did not write.
- Queries must be 2-6 words maximum.

Examples:
  Topic "Westworld TV Series S3 and S4"  →  "Westworld S3 plot", "Westworld S4 cast"
  Topic "RSI trading indicator"          →  "RSI indicator formula", "RSI trading python"
  Topic "React framework hooks"          →  "React hooks tutorial", "React framework example"

Topic: {topic}
Already researched: {known_topics}
{vocab_section}
{feedback_section}
Respond ONLY with valid JSON, no other text:
[
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}},
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}},
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}},
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}},
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}}
]
"""

_RETROSPECTIVE_PROMPT = """You are a search query generator. The research agent is STUCK — all previous queries for this topic failed to produce useful results.

Your task: Generate 5 completely DIFFERENT search queries that approach the topic from a fresh angle.

Topic: {topic}
Failed queries so far: {failed_queries}

Rules:
- Each query must use ONLY words already present in the topic, plus these helpers if needed:
  season, episode, cast, plot, summary, review, release, explained, how, why, list, vs,
  history, tutorial, formula, algorithm, code, example, install, python, wiki, imdb
- Queries must be 2-6 words maximum.
- Every query MUST be different from the failed ones above.

Respond ONLY with valid JSON, no other text:
[
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}},
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}},
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}},
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}},
  {{"subtopic": "<short name>", "query": "<2-6 word query>"}}
]
"""

_FOLLOWUP_PROMPT = """You are a search query generator.

The previous research had gaps: {gaps}
Original topic: {topic}

Generate ONE follow-up search query using only words from the topic + gap description.
Keep the query to 2-6 words. No abstract nouns. No embellishments.

Respond ONLY with valid JSON:
{{"subtopic": "{topic} (follow-up)", "query": "<2-6 word query>"}}
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


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "with",
    "on", "at", "by", "from", "is", "are", "was", "be", "as", "it",
    "its", "this", "that", "how", "what", "which", "who", "s",
}


class PlannerAgent:
    """Decomposes high-level topics into concrete search tasks."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
        search_tool: Optional["SearchTool"] = None,
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url
        self._search_tool = search_tool

    async def _pre_search_vocab(self, topic: str) -> list[str]:
        """Do a quick search for the topic and extract real vocabulary from results.

        Returns up to 20 meaningful words found in search titles/snippets.
        These are fed into the Planner prompt so queries are grounded in real
        web vocabulary rather than LLM-invented abstract nouns.
        """
        if self._search_tool is None:
            return []
        try:
            results = await self._search_tool.search(topic)
        except Exception as exc:
            logger.warning("Pre-search vocab lookup failed for %r: %s", topic, exc)
            return []

        words: dict[str, int] = {}
        for r in results[:3]:
            text = f"{r.get('title', '')} {r.get('body', '')[:120]}"
            for raw_word in re.split(r"\W+", text):
                word = raw_word.lower().strip()
                if len(word) >= 3 and word not in _STOPWORDS:
                    words[word] = words.get(word, 0) + 1

        # Return most-frequent words (real vocabulary from the web)
        sorted_words = sorted(words, key=lambda w: words[w], reverse=True)
        vocab = sorted_words[:20]
        if vocab:
            logger.info("Pre-search vocab for %r: %s", topic, ", ".join(vocab))
        return vocab

    def _clean_query(self, query: str, topic: str = "") -> str:
        """Strip LLM-invented filler words from a generated query.

        Two-pass approach:
        1. Remove known filler phrases/words (fast blocklist).
        2. If the query still looks bloated (>8 words), drop any word that
           neither appears in the original topic nor in an allowed helper vocab.
        """
        # Pass 1: explicit blocklist of common LLM embellishments
        _FILLER = {
            "detailed", "detail", "comprehensive", "in-depth", "indepth",
            "quality", "authoritative", "scholarly", "academic", "advanced",
            "expert", "analytical", "analysis", "narrative", "elements",
            "architectural", "thematic", "contextual", "technological",
            "mechanisms", "paradigm", "strategies", "exploration", "overview",
            "framework", "implementation", "foundational", "fundamental",
            "examination", "investigation", "perspective", "aspects",
            "concepts", "principles", "dynamics", "insights", "approach",
            "methodology", "techniques", "components", "structure",
            "from", "research", "sources", "authoritative", "source",
            "study", "guide", "deeper", "understanding",
        }
        words = query.split()
        words = [w for w in words if w.lower().rstrip("s") not in _FILLER and w.lower() not in _FILLER]

        # Pass 2: if still >8 words, keep only topic words + allowed helpers
        if len(words) > 8 and topic:
            topic_words = set(topic.lower().split())
            _ALLOWED_HELPERS = {
                "season", "episode", "cast", "plot", "characters", "summary",
                "review", "release", "explained", "how", "why", "list", "vs",
                "history", "tutorial", "formula", "algorithm", "code",
                "example", "python", "date", "trailer", "ending", "finale",
                "recap", "explained", "wiki", "imdb", "rating", "s3", "s4",
            }
            words = [
                w for w in words
                if w.lower() in topic_words or w.lower() in _ALLOWED_HELPERS
            ]

        result = " ".join(words).strip()
        return result if result else query

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
        self,
        topic: str,
        known_topics: Optional[list[str]] = None,
        good_examples: Optional[list[str]] = None,
        bad_examples: Optional[list[str]] = None,
    ) -> list[dict[str, str]]:
        """Return a list of ``{'subtopic': …, 'query': …}`` dicts.

        Args:
            topic: The high-level research topic.
            known_topics: Already-researched subtopics to avoid repeating.
            good_examples: Query strings that previously produced PROCEED results.
            bad_examples: Query strings that previously exhausted all retries.
        """
        known = ", ".join(known_topics or []) or "none"

        # Phase 1 — pre-search vocabulary grounding
        vocab = await self._pre_search_vocab(topic)
        vocab_section = (
            f"Real vocabulary from web search results for this topic: {', '.join(vocab)}\n"
            "Prefer these words in your queries — they appear in real search results."
            if vocab else ""
        )

        # Phase 2 — feedback from previous cycles
        feedback_parts: list[str] = []
        if good_examples:
            sample = good_examples[-3:]  # last 3 successes
            feedback_parts.append(f"Queries that worked well (mimic these patterns): {sample}")
        if bad_examples:
            sample = bad_examples[-3:]  # last 3 failures
            feedback_parts.append(f"Queries that failed (avoid these patterns): {sample}")
        feedback_section = "\n".join(feedback_parts)

        prompt = _DECOMPOSE_PROMPT.format(
            topic=topic,
            known_topics=known,
            vocab_section=vocab_section,
            feedback_section=feedback_section,
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, _call_ollama, prompt, self.model, self.ollama_base_url
        )

        tasks = self._parse_json(raw)
        if isinstance(tasks, list) and tasks:
            for task in tasks:
                if isinstance(task, dict) and "query" in task:
                    task["query"] = self._clean_query(task["query"], topic)
            logger.info("Planner generated %d tasks for topic %r.", len(tasks), topic)
            return tasks

        logger.warning(
            "Planner LLM returned unusable output; using fallback tasks for %r.", topic
        )
        return self._fallback_tasks(topic)

    async def decompose_retrospective(
        self,
        topic: str,
        failed_queries: list[str],
    ) -> list[dict[str, str]]:
        """Phase 3 — re-plan from scratch when the agent is stuck.

        Uses a special prompt that instructs the LLM to approach the topic
        from a completely different angle, explicitly showing what has failed.
        """
        logger.info(
            "Retrospective re-plan for %r after %d failed queries.",
            topic, len(failed_queries),
        )
        prompt = _RETROSPECTIVE_PROMPT.format(
            topic=topic,
            failed_queries=", ".join(failed_queries[-10:]) or "none",
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, _call_ollama, prompt, self.model, self.ollama_base_url
        )

        tasks = self._parse_json(raw)
        if isinstance(tasks, list) and tasks:
            for task in tasks:
                if isinstance(task, dict) and "query" in task:
                    task["query"] = self._clean_query(task["query"], topic)
            logger.info(
                "Retrospective planner generated %d tasks for topic %r.",
                len(tasks), topic,
            )
            return tasks

        logger.warning(
            "Retrospective planner returned unusable output; using fallback for %r.", topic
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
            task["query"] = self._clean_query(task["query"], topic)
            logger.info("Planner refined task for %r: %s", topic, task["query"])
            return task

        # Fallback: build a simple query from topic and gaps
        return {
            "subtopic": f"{topic} (refined)",
            "query": self._clean_query(f"{topic} {gaps}", topic),
        }
