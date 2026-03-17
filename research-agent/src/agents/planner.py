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
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.tools.search_tool import SearchTool

from src.config_loader import get_filters_config
from src.llm_client import generate_text
from src.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# CJK Unicode ranges — mirrors search_tool._contains_cjk to avoid circular import.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0xAC00, 0xD7AF),
    (0x3040, 0x30FF),
)


def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _CJK_RANGES):
            return True
    return False

_DECOMPOSE_PROMPT_FILE = "planner_decompose.md"
_RETROSPECTIVE_PROMPT_FILE = "planner_retrospective.md"
_FOLLOWUP_PROMPT_FILE = "planner_followup.md"
_ANALYZE_PROMPT_FILE = "planner_analyze.md"
_HIERARCHICAL_DECOMPOSE_PROMPT_FILE = "planner_hierarchical_decompose.md"
_CONSOLIDATION_PROMPT_FILE = "planner_consolidate.md"
_RESTRUCTURE_PROMPT_FILE = "planner_restructure.md"


class PlannerAgent:
    """Decomposes high-level topics into concrete search tasks."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
        search_tool: Optional["SearchTool"] = None,
        user_prompt: Optional[str] = None,
        llm_provider: str = "ollama",
        llm_api_key: Optional[str] = None,
        prompt_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url
        self._search_tool = search_tool
        self._user_prompt = user_prompt
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._decompose_prompt = load_prompt(_DECOMPOSE_PROMPT_FILE, prompt_dir)
        self._retrospective_prompt = load_prompt(_RETROSPECTIVE_PROMPT_FILE, prompt_dir)
        self._followup_prompt = load_prompt(_FOLLOWUP_PROMPT_FILE, prompt_dir)
        self._analyze_prompt = load_prompt(_ANALYZE_PROMPT_FILE, prompt_dir)
        self._hierarchical_decompose_prompt = load_prompt(
            _HIERARCHICAL_DECOMPOSE_PROMPT_FILE,
            prompt_dir,
        )
        self._consolidation_prompt = load_prompt(_CONSOLIDATION_PROMPT_FILE, prompt_dir)
        self._restructure_prompt = load_prompt(_RESTRUCTURE_PROMPT_FILE, prompt_dir)

    @property
    def _user_context(self) -> str:
        """Return the user prompt section for injection into LLM prompts."""
        if self._user_prompt:
            return f"User instructions:\n{self._user_prompt}\n"
        return ""

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Synchronous LLM call used by planner operations."""
        return generate_text(
            prompt,
            self.model,
            self.ollama_base_url,
            provider=self._llm_provider,
            api_key=self._llm_api_key,
            timeout=120,
        )

    async def _pre_search_vocab(self, topic: str) -> list[str]:
        """Do a quick search for the topic and extract real vocabulary from results.

        Returns up to 20 meaningful words found in search titles/snippets.
        These are fed into the Planner prompt so queries are grounded in real
        web vocabulary rather than LLM-invented abstract nouns.
        """
        if self._search_tool is None:
            return []
        # Truncate multi-line / requirements-file topics to a short search query.
        # Search APIs cannot handle hundreds of words; take the first non-empty
        # line and cap it at 150 characters so the lookup is actually useful.
        first_line = next(
            (line.strip() for line in topic.split("\n") if line.strip()), topic
        )
        # Strip date stamps and separator noise (mirrors _make_search_query in
        # agent_manager) to clean up date suffixes and separator characters.
        search_query = re.sub(r"[（(]\d{4}[-/]\d{2}[-/]\d{2}[）)]", "", first_line)
        search_query = re.sub(r"[（(]\d{4}[）)]", "", search_query)
        search_query = re.sub(r"\s*[-–—|]+\s*", " ", search_query).strip()[:150]
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
                if len(word) >= 3 and word not in get_filters_config()["stopwords"]:
                    words[word] = words.get(word, 0) + 1

        # Return most-frequent words (real vocabulary from the web)
        sorted_words = sorted(words, key=lambda w: words[w], reverse=True)
        vocab = sorted_words[:20]
        if vocab:
            logger.info("Pre-search vocab for %r: %s", topic, ", ".join(vocab))
        return vocab

    @staticmethod
    def _split_camel_case(token: str) -> str:
        """Convert CamelCase / PascalCase tokens to space-separated lowercase words.

        Examples::
            'TechProviderIntegration' → 'tech provider integration'
            'APIIntegration'          → 'api integration'
            'already spaced'          → 'already spaced'
        """
        # Insert a space before every uppercase letter that follows a lowercase
        # letter or before an uppercase letter followed by a lowercase letter
        # (handles sequences like 'APIIntegration' → 'API Integration').
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', token)
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
        return s.lower()

    def _clean_query(self, query: str, topic: str = "") -> str:
        """Strip LLM-invented filler words from a generated query.

        Three-pass approach:
        0. Split any CamelCase / PascalCase tokens into separate words.
        1. Remove known filler phrases/words (fast blocklist).
        2. If the query still looks bloated (>8 words), drop any word that
           neither appears in the original topic nor in an allowed helper vocab.
        """
        # Pass 0: expand CamelCase tokens (e.g. 'TechProviderIntegration' → 'tech provider integration')
        words_raw = query.split()
        words_raw = [w if ' ' in w else self._split_camel_case(w) for w in words_raw]
        query = ' '.join(words_raw)

        # Pass 1: explicit blocklist of common LLM embellishments
        filler_words = get_filters_config()["filler_words"]
        words = query.split()
        words = [w for w in words if w.lower().rstrip("s") not in filler_words and w.lower() not in filler_words]

        # Pass 2: if still >8 words, keep only topic words + allowed helpers.
        # Skip entirely for CJK queries — Chinese/Japanese/Korean words are not
        # space-delimited so topic_words splitting doesn't work, and stripping
        # them leaves an empty query.
        if len(words) > 8 and topic and not _contains_cjk("".join(words)):
            topic_words = set(topic.lower().split())
            allowed_helpers = get_filters_config()["allowed_query_helpers"]
            words = [
                w for w in words
                if w.lower() in topic_words or w.lower() in allowed_helpers
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

        prompt = self._decompose_prompt.format(
            topic=topic,
            known_topics=known,
            vocab_section=vocab_section,
            feedback_section=feedback_section,
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._call_llm, prompt)

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
        prompt = self._retrospective_prompt.format(
            topic=topic,
            failed_queries=", ".join(failed_queries[-10:]) or "none",
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._call_llm, prompt)

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

    async def refine(self, topic: str, gaps: str, main_topic: str = "") -> dict[str, str]:
        """Generate a targeted follow-up task to fill the identified *gaps*."""
        prompt = self._followup_prompt.format(
            topic=topic, gaps=gaps, main_topic=main_topic or topic,
            user_context=self._user_context,
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._call_llm, prompt)

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

    async def analyze(self, topic: str, initial_summary: str = "", description: str = "", main_topic: str = "") -> dict[str, Any]:
        """Decide whether *topic* is a simple leaf or needs sub-topic decomposition.

        Returns ``{'is_leaf': bool, 'relevance': str, 'reasoning': str}``.
        """
        summary_section = (
            f"Initial research summary:\n{initial_summary[:2000]}\n"
            if initial_summary else ""
        )
        description_section = (
            f"Description: {description}\n" if description else ""
        )
        prompt = self._analyze_prompt.format(
            topic=topic,
            main_topic=main_topic or topic,
            summary_section=summary_section,
            description_section=description_section,
            user_context=self._user_context,
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._call_llm, prompt)

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
        main_topic: str = "",
        max_children: int = 5,
    ) -> list[dict[str, Any]]:
        """Break a complex topic into prioritized sub-topics.

        *max_children* controls how many sub-topics the LLM is asked to
        generate (adaptive based on budget).

        Returns a list of dicts with keys: ``name``, ``query``, ``priority``,
        ``description``.
        """
        # Clamp to sensible bounds
        max_children = max(1, min(max_children, 7))
        children_label = f"{max(2, max_children - 1)}-{max_children}"

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

        prompt = self._hierarchical_decompose_prompt.format(
            topic=topic,
            main_topic=main_topic or topic,
            description_section=description_section,
            user_context=self._user_context,
            known_subtopics=known,
            vocab_section=vocab_section,
            max_children=children_label,
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._call_llm, prompt)

        tasks = self._parse_json(raw)
        if isinstance(tasks, list) and tasks:
            # Build a short context prefix for CJK topics so that bare generic
            # phrases like "用户数量" become "在线TRPG 用户数量" (searchable).
            cjk_prefix = ""
            if _contains_cjk(topic):
                # Take up to the first 8 meaningful chars of the topic as context.
                clean_topic = re.sub(r"[（(][^）)]*[）)]", "", topic).strip()
                cjk_prefix = clean_topic[:8].strip()

            for task in tasks:
                if isinstance(task, dict) and "query" in task:
                    task["query"] = self._clean_query(task["query"], topic)
                    # For CJK: if the query is short and doesn't already contain
                    # domain context, prepend the topic prefix.
                    q = task["query"]
                    if (
                        cjk_prefix
                        and _contains_cjk(q)
                        and len(q.replace(" ", "")) <= 6
                        and cjk_prefix not in q
                    ):
                        task["query"] = f"{cjk_prefix} {q}".strip()
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

        prompt = self._consolidation_prompt.format(
            parent_topic=parent_topic,
            user_context=self._user_context,
            child_summaries=child_text[:10000],
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._call_llm, prompt)

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
        prompt = self._restructure_prompt.format(
            graph_outline=graph_outline,
            gaps=gaps,
            user_context=self._user_context,
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._call_llm, prompt)

        result = self._parse_json(raw)
        if isinstance(result, list):
            logger.info("Restructure suggestions: %d changes.", len(result))
            return result

        logger.warning("Restructure LLM returned unusable output.")
        return []
