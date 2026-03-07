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
  Topic "Westworld TV Series S3 and S4"  →  "Westworld season 3 plot", "Westworld season 4 cast"
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
{user_context}
Generate ONE follow-up search query using only words from the topic + gap description.
Keep the query to 2-6 words. No abstract nouns. No embellishments.

Respond ONLY with valid JSON:
{{"subtopic": "{topic} (follow-up)", "query": "<2-6 word query>"}}
"""

_ANALYZE_PROMPT = """You are a research topic analyzer.

Given a topic and an optional initial summary, decide whether this topic:
- Can be researched DIRECTLY with a few simple web searches ("leaf" topic), OR
- Is COMPLEX and should be broken into smaller sub-topics for thorough research.

Examples of LEAF topics: "RSI indicator formula", "Python asyncio tutorial", "GraphRAG installation guide"
Examples of COMPLEX topics: "Design a World Extraction Service", "Stock Trading Strategies", "Build a recommendation engine"

Topic: {topic}
{description_section}
{summary_section}
{user_context}
Respond ONLY with valid JSON:
{{"is_leaf": true | false, "reasoning": "<one sentence explanation>"}}
"""

_HIERARCHICAL_DECOMPOSE_PROMPT = """You are a research planner that breaks a complex topic into sub-topics.

Topic: {topic}
{description_section}
{user_context}
Already known sub-topics (do NOT repeat these): {known_subtopics}
{vocab_section}

Rules:
- Generate 3-7 sub-topics that together cover the main topic comprehensively.
- Sub-topics must be non-overlapping and specific.
- Each sub-topic needs a short name, a 2-6 word search query, a priority (1-10, higher = more important), and a one-sentence description.
- DO NOT invent abstract or vague sub-topics. Be concrete.
- Search queries must use real words from the topic.

Respond ONLY with valid JSON:
[
  {{"name": "<short name>", "query": "<2-6 word query>", "priority": <1-10>, "description": "<one sentence>"}},
  ...
]
"""

_CONSOLIDATION_PROMPT = """You are a research consolidator. Synthesize multiple research findings into a unified summary.

Parent topic: {parent_topic}
{user_context}

Child research findings:
{child_summaries}

Write a comprehensive consolidated summary (≤800 words) that:
1. Integrates all child findings into a coherent narrative.
2. Highlights connections and dependencies between sub-topics.
3. Identifies any remaining gaps or contradictions.
4. Provides a clear overall picture of the parent topic.

Consolidated summary:"""

_RESTRUCTURE_PROMPT = """You are a research planner reviewing a research graph that has gaps.

Current research outline:
{graph_outline}

Identified gaps or issues:
{gaps}
{user_context}

Suggest changes to improve coverage. You can:
- Add new sub-topics under existing parents.
- Note which areas need more depth.

Respond ONLY with valid JSON:
[
  {{"action": "add", "parent_name": "<existing parent topic name>", "name": "<new sub-topic>", "query": "<2-6 word query>", "priority": <1-10>}},
  ...
]
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
        user_prompt: Optional[str] = None,
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url
        self._search_tool = search_tool
        self._user_prompt = user_prompt

    @property
    def _user_context(self) -> str:
        """Return the user prompt section for injection into LLM prompts."""
        if self._user_prompt:
            return f"User instructions:\n{self._user_prompt}\n"
        return ""

    async def _pre_search_vocab(self, topic: str) -> list[str]:
        """Do a quick search for the topic and extract real vocabulary from results.

        Returns up to 20 meaningful words found in search titles/snippets.
        These are fed into the Planner prompt so queries are grounded in real
        web vocabulary rather than LLM-invented abstract nouns.
        """
        if self._search_tool is None:
            return []
        # Truncate multi-line / requirements-file topics to a short search query.
        # DuckDuckGo cannot handle hundreds of words; take the first non-empty
        # line and cap it at 150 characters so the lookup is actually useful.
        first_line = next(
            (line.strip() for line in topic.split("\n") if line.strip()), topic
        )
        search_query = first_line[:150]
        try:
            results = await self._search_tool.search(search_query)
        except Exception as exc:
            logger.warning("Pre-search vocab lookup failed for %r: %s", search_query, exc)
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
        prompt = _FOLLOWUP_PROMPT.format(
            topic=topic, gaps=gaps, user_context=self._user_context,
        )

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

    # ------------------------------------------------------------------
    # Hierarchical decomposition methods
    # ------------------------------------------------------------------

    async def analyze(self, topic: str, initial_summary: str = "", description: str = "") -> dict[str, Any]:
        """Decide whether *topic* is a simple leaf or needs sub-topic decomposition.

        Returns ``{'is_leaf': bool, 'reasoning': str}``.
        """
        summary_section = (
            f"Initial research summary:\n{initial_summary[:2000]}\n"
            if initial_summary else ""
        )
        description_section = (
            f"Description: {description}\n" if description else ""
        )
        prompt = _ANALYZE_PROMPT.format(
            topic=topic,
            summary_section=summary_section,
            description_section=description_section,
            user_context=self._user_context,
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, _call_ollama, prompt, self.model, self.ollama_base_url
        )

        result = self._parse_json(raw)
        if isinstance(result, dict) and "is_leaf" in result:
            logger.info("Analyze %r: is_leaf=%s (%s)", topic, result["is_leaf"], result.get("reasoning", ""))
            return result

        # Fallback: assume complex if topic has many words or multiple sentences
        is_leaf = len(topic.split()) <= 6 and "\n" not in topic
        return {"is_leaf": is_leaf, "reasoning": "fallback heuristic (word count)"}

    async def decompose_hierarchical(
        self,
        topic: str,
        description: str = "",
        known_subtopics: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Break a complex topic into 3-7 prioritized sub-topics.

        Returns a list of dicts with keys: ``name``, ``query``, ``priority``,
        ``description``.
        """
        known = ", ".join(known_subtopics or []) or "none"
        description_section = (
            f"Description: {description}\n" if description else ""
        )

        # Vocabulary grounding
        vocab = await self._pre_search_vocab(topic)
        vocab_section = (
            f"Real vocabulary from web search results: {', '.join(vocab)}\n"
            "Prefer these words in your queries."
            if vocab else ""
        )

        prompt = _HIERARCHICAL_DECOMPOSE_PROMPT.format(
            topic=topic,
            description_section=description_section,
            user_context=self._user_context,
            known_subtopics=known,
            vocab_section=vocab_section,
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
                    task.setdefault("priority", 5)
                    task.setdefault("description", "")
                    task.setdefault("name", task.get("query", topic))
            logger.info(
                "Hierarchical planner generated %d sub-topics for %r.",
                len(tasks), topic,
            )
            return tasks

        # Fallback: generate 3 generic sub-topics
        logger.warning(
            "Hierarchical planner returned unusable output for %r; using fallback.", topic
        )
        return [
            {"name": f"{topic} overview", "query": f"{topic} overview", "priority": 8, "description": "General overview"},
            {"name": f"{topic} key concepts", "query": f"{topic} concepts", "priority": 6, "description": "Core concepts"},
            {"name": f"{topic} examples", "query": f"{topic} examples", "priority": 4, "description": "Practical examples"},
        ]

    async def consolidate_summaries(
        self,
        parent_topic: str,
        child_summaries: list[tuple[str, str]],
    ) -> str:
        """Synthesize child findings into a unified parent summary.

        *child_summaries* is a list of ``(name, summary)`` tuples.
        Returns the consolidated summary text.
        """
        parts = []
        for name, summary in child_summaries:
            parts.append(f"### {name}\n{summary}")
        child_text = "\n\n".join(parts)

        prompt = _CONSOLIDATION_PROMPT.format(
            parent_topic=parent_topic,
            user_context=self._user_context,
            child_summaries=child_text[:10000],
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, _call_ollama, prompt, self.model, self.ollama_base_url
        )

        if raw and raw.strip():
            logger.info("Consolidated summary for %r: %d chars.", parent_topic, len(raw))
            return raw.strip()

        # Fallback: concatenate child summaries
        logger.warning("Consolidation LLM failed for %r; concatenating summaries.", parent_topic)
        return child_text

    async def suggest_restructure(
        self,
        graph_outline: str,
        gaps: str,
    ) -> list[dict[str, Any]]:
        """Suggest graph modifications to fill identified *gaps*.

        Returns a list of dicts with keys: ``action``, ``parent_name``, ``name``,
        ``query``, ``priority``.
        """
        prompt = _RESTRUCTURE_PROMPT.format(
            graph_outline=graph_outline,
            gaps=gaps,
            user_context=self._user_context,
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, _call_ollama, prompt, self.model, self.ollama_base_url
        )

        result = self._parse_json(raw)
        if isinstance(result, list):
            logger.info("Restructure suggestions: %d changes.", len(result))
            return result

        logger.warning("Restructure LLM returned unusable output.")
        return []
