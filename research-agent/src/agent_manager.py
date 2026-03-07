"""
agent_manager.py – Orchestrates Planner, Researcher, and Critic.

Provides:
* ``AgentManager.run_graph`` – builds a hierarchical topic graph, researches
  leaf nodes, and consolidates findings bottom-up.
* ``AgentManager.run_cycle`` – executes one research cycle for a single task.
* ``AgentManager.generate_report`` – consolidates approved findings into Markdown.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from pathlib import Path
from typing import Any, Optional

from src.agents.critic import CriticAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.database.knowledge_base import KnowledgeBase
from src.tools.search_tool import SearchTool
from src.topic_graph import TopicGraph, TopicNode, MAX_DEPTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Patterns that make terrible search queries: ISO dates in full-width or ASCII
# parens, standalone parenthesised years, and redundant separators like " - ".
_DATE_PATTERNS = re.compile(
    r"[（(]\d{4}[-/]\d{2}[-/]\d{2}[）)]"   # （2026-03-06） or (2026-03-06)
    r"|[（(]\d{4}[）)]"                      # （2026） or (2026)
)
_SEPARATOR_RE = re.compile(r"\s*[-–—|]+\s*")


def _make_search_query(name: str) -> str:
    """Derive a clean, short search query from a topic name.

    Removes date stamps (e.g. ``（2026-03-06）``) and normalises separators
    (``-``, ``–``, ``—``, ``|``) to spaces so phrases like
    ``在线 TRPG 市场分析 - 中国市场（2026-03-06）`` become
    ``在线 TRPG 市场分析 中国市场``.
    """
    q = _DATE_PATTERNS.sub("", name)
    q = _SEPARATOR_RE.sub(" ", q)
    return q.strip()

_REPORT_TEMPLATE = """\
# {topic}

{graph_outline_section}
{findings}
{technical_sections}
## Sources
{sources}
"""

_MAX_REJECT_RETRIES = 3
_MAX_CONSECUTIVE_FAILURES = 5


class AgentManager:
    """Coordinates the Planner → Researcher → Critic pipeline.

    Supports two modes of operation:
    1. **Graph mode** (default for complex topics): Builds a hierarchical topic
       graph, recursively decomposes, researches leaf nodes, and consolidates.
    2. **Flat mode** (simple topics): Falls back to the original flat task queue
       when the Planner determines the topic is a simple leaf.
    """

    def __init__(
        self,
        topic: str,
        title: Optional[str] = None,
        user_prompt: Optional[str] = None,
        model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
        reports_dir: str = "data/reports",
        db_path: str = "data/research.db",
    ) -> None:
        self.topic = topic
        self._title = title if title is not None else topic
        self._user_prompt = user_prompt
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1 — pass SearchTool to Planner for pre-search vocabulary grounding
        _search_tool = SearchTool(max_results=3)
        self._planner = PlannerAgent(
            model=model,
            ollama_base_url=ollama_base_url,
            search_tool=_search_tool,
            user_prompt=user_prompt,
        )
        self._researcher = ResearcherAgent(
            model=model, ollama_base_url=ollama_base_url,
            user_prompt=user_prompt,
        )
        self._critic = CriticAgent(
            model=model, ollama_base_url=ollama_base_url,
            user_prompt=user_prompt,
        )
        self._kb = KnowledgeBase(db_path=db_path)

        # Topic graph (built during build_graph)
        self._graph: Optional[TopicGraph] = None

        # Flat fallback queue (used for simple topics or legacy mode)
        self._task_queue: deque[dict[str, str]] = deque()
        self._approved: list[dict[str, Any]] = []

        # Phase 2 — track query-level feedback across cycles
        self._successful_queries: list[str] = []
        self._failed_queries: list[str] = []

        # Phase 3 — stuck detection
        self._consecutive_failures: int = 0

    async def init(self) -> None:
        """Initialise the knowledge base."""
        await self._kb.init()

    async def close(self) -> None:
        """Shut down the knowledge base."""
        await self._kb.close()

    # ------------------------------------------------------------------
    # Queue Management (flat mode — kept for backward compatibility)
    # ------------------------------------------------------------------

    async def populate_queue(
        self,
        *,
        is_retrospective: bool = False,
    ) -> None:
        """Ask the Planner to seed the task queue with initial subtopics.

        On normal runs passes accumulated feedback examples so the Planner
        can learn from prior successes/failures (Phase 2).  On retrospective
        runs (Phase 3) uses the dedicated retrospective prompt instead.
        """
        known = await self._kb.get_all_topics()
        if is_retrospective:
            tasks = await self._planner.decompose_retrospective(
                self.topic,
                failed_queries=self._failed_queries,
            )
        else:
            tasks = await self._planner.decompose(
                self.topic,
                known_topics=known,
                good_examples=self._successful_queries or None,
                bad_examples=self._failed_queries or None,
            )
        for task in tasks:
            self._task_queue.append(task)
        logger.info("Task queue seeded with %d tasks.", len(tasks))

    def has_tasks(self) -> bool:
        return len(self._task_queue) > 0

    # ------------------------------------------------------------------
    # Hierarchical Graph Construction
    # ------------------------------------------------------------------

    async def build_graph(self) -> TopicGraph:
        """Build the hierarchical topic graph by recursive decomposition.

        Phase A: Quick research on root for context.
        Phase B: Analyze → decompose recursively to MAX_DEPTH.
        """
        # Root name should always come from the topic text (first line), not
        # the title (which is a filename suited for report output, not search).
        root_name = self.topic.split("\n")[0][:100] if "\n" in self.topic else self.topic
        # Strip leading Markdown heading markers (e.g. "# Heading" → "Heading")
        root_name = re.sub(r"^#+\s*", "", root_name).strip()
        # Derive a clean search query by removing date stamps and normalising
        # separators — e.g. "在线 TRPG 市场分析 - 中国市场（2026-03-06）"
        # becomes "在线 TRPG 市场分析 中国市场".
        root_query = _make_search_query(root_name)
        graph = TopicGraph(root_name=root_name, root_query=root_query)
        self._graph = graph

        logger.info("Building topic graph for %r ...", root_name)

        # Phase A — quick initial research on root topic for context
        root_task = {"subtopic": root_name, "query": root_query}
        root_result = await self._researcher.research(root_task)
        root_summary = root_result.get("summary", "")
        if root_summary:
            graph.root.summary = root_summary
            graph.root.source_urls = root_result.get("source_urls", [])

        # Phase B — analyze and decompose recursively
        await self._decompose_node(graph, graph.root.id, root_summary)

        logger.info(
            "Topic graph built: %d nodes.\n%s",
            graph.node_count(), graph.get_outline(),
        )
        return graph

    async def _decompose_node(
        self,
        graph: TopicGraph,
        node_id: str,
        context_summary: str = "",
    ) -> None:
        """Recursively decompose a node into sub-topics if it's complex."""
        node = graph.get_node(node_id)
        if node is None:
            return

        if node.depth >= MAX_DEPTH:
            graph.mark_leaf(node_id)
            logger.info("Max depth reached for %r — marking as leaf.", node.name)
            return

        graph.mark_analyzing(node_id)

        # Use the node's search query (enriched form) for planner calls when
        # available — it contains domain context stripped from the display name
        # (e.g. "在线TRPG 市场规模" rather than bare "市场规模").
        planner_topic = node.query if node.query.strip() else node.name

        # Ask Planner if this topic is simple (leaf) or complex
        analysis = await self._planner.analyze(
            planner_topic,
            initial_summary=context_summary,
            description=node.description,
        )

        if analysis.get("is_leaf", False):
            graph.mark_leaf(node_id)
            node.status = "pending"  # ready for research
            logger.info("Leaf topic: %r — %s", node.name, analysis.get("reasoning", ""))
            return

        # Complex topic: decompose into sub-topics
        known = graph.get_all_researched_names()
        children_names = [c.name for c in graph.get_children(node_id)]
        known.extend(children_names)

        sub_topics = await self._planner.decompose_hierarchical(
            planner_topic,
            description=node.description,
            known_subtopics=known,
        )

        if not sub_topics:
            graph.mark_leaf(node_id)
            node.status = "pending"
            return

        for st in sub_topics:
            child = graph.add_node(
                name=st.get("name", st.get("query", "")),
                query=st.get("query", ""),
                parent_id=node_id,
                priority=st.get("priority", 5),
                description=st.get("description", ""),
            )
            # Recursively analyze each child
            await self._decompose_node(graph, child.id)

    # ------------------------------------------------------------------
    # Graph-Based Research Orchestration
    # ------------------------------------------------------------------

    async def run_graph(self) -> Optional[dict[str, Any]]:
        """Process the next available node in the topic graph.

        Tries leaf research first, then consolidation.  Returns the approved
        finding dict, or None if nothing was processed.
        """
        if self._graph is None:
            return None

        # Priority 1: Research pending leaf nodes
        ready = self._graph.get_ready_for_research()
        if ready:
            node = ready[0]
            return await self._research_node(node)

        # Priority 2: Consolidate parent nodes
        consolidatable = self._graph.get_ready_for_consolidation()
        if consolidatable:
            node = consolidatable[0]
            return await self._consolidate_node(node)

        return None

    def has_graph_work(self) -> bool:
        """Return True if there are still nodes to research or consolidate."""
        if self._graph is None:
            return False
        if self._graph.is_complete():
            return False
        ready = self._graph.get_ready_for_research()
        consolidatable = self._graph.get_ready_for_consolidation()
        return bool(ready or consolidatable)

    async def _research_node(self, node: TopicNode) -> Optional[dict[str, Any]]:
        """Research a single leaf node using the Researcher → Critic pipeline."""
        assert self._graph is not None

        self._graph.mark_researching(node.id)
        task = {"subtopic": node.name, "query": node.query}
        reject_count = 0

        while reject_count < _MAX_REJECT_RETRIES:
            result = await self._researcher.research(task)

            # Deduplication
            if result["summary"] and await self._kb.is_duplicate(result["summary"]):
                logger.info("Duplicate content for %r – skipping.", node.name)
                self._graph.mark_researched(node.id, "(duplicate — skipped)", [])
                return None

            # Critic review
            verdict = await self._critic.review(
                task["subtopic"], result["summary"], topic=self._title,
            )

            if verdict.get("status") == "PROCEED":
                source_url = (
                    result["source_urls"][0] if result["source_urls"] else None
                )
                await self._kb.save(
                    topic=node.name,
                    content=result["summary"],
                    source_url=source_url,
                )

                finding = {
                    "subtopic": node.name,
                    "query": task["query"],
                    "summary": result["summary"],
                    "source_urls": result["source_urls"],
                    "node_id": node.id,
                }
                self._approved.append(finding)

                self._graph.mark_researched(
                    node.id, result["summary"], result["source_urls"],
                )

                # Feedback tracking
                self._successful_queries.append(task["query"])
                self._consecutive_failures = 0

                logger.info("Approved (graph): %r", node.name)
                return finding

            # Critic rejected — refine and retry
            missing = verdict.get("missing", "unspecified details")
            logger.info(
                "Rejected %r (attempt %d/%d). Missing: %s",
                node.name, reject_count + 1, _MAX_REJECT_RETRIES, missing,
            )
            refined_task = await self._planner.refine(node.name, missing)
            task = refined_task
            reject_count += 1

        # Exhausted retries
        self._failed_queries.append(task["query"])
        self._consecutive_failures += 1
        self._graph.mark_failed(node.id)
        logger.warning("Node %r exhausted retries.", node.name)

        # Phase 3 — stuck detection across the graph
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "%d consecutive failures — triggering graph restructure.",
                self._consecutive_failures,
            )
            self._consecutive_failures = 0
            await self._restructure_graph()

        return None

    async def _consolidate_node(self, node: TopicNode) -> Optional[dict[str, Any]]:
        """Consolidate child summaries into a parent summary."""
        assert self._graph is not None

        children = self._graph.get_children(node.id)
        child_summaries: list[tuple[str, str]] = []
        for child in children:
            summary = child.consolidated_summary or child.summary or ""
            if summary and summary != "(duplicate — skipped)":
                child_summaries.append((child.name, summary))

        if not child_summaries:
            self._graph.mark_consolidated(node.id, "")
            return None

        consolidated = await self._planner.consolidate_summaries(
            node.name, child_summaries,
        )

        # Critic reviews consolidated summary
        verdict = await self._critic.review(
            node.name, consolidated, topic=self._title,
        )

        if verdict.get("status") == "PROCEED":
            self._graph.mark_consolidated(node.id, consolidated)

            finding = {
                "subtopic": f"{node.name} (consolidated)",
                "query": node.query,
                "summary": consolidated,
                "source_urls": [],
                "node_id": node.id,
            }
            self._approved.append(finding)
            logger.info("Consolidated: %r", node.name)
            return finding

        # Consolidation rejected — try restructuring then accept what we have
        missing = verdict.get("missing", "gaps in consolidated summary")
        logger.info("Consolidated review rejected for %r: %s", node.name, missing)
        await self._restructure_graph_for_node(node, missing)

        # Mark consolidated anyway to avoid infinite loops
        self._graph.mark_consolidated(node.id, consolidated)
        return None

    async def _restructure_graph(self) -> None:
        """Request graph restructuring from the Planner after stuck detection."""
        if self._graph is None:
            return
        outline = self._graph.get_outline()
        gaps = f"Last {len(self._failed_queries)} queries failed: {', '.join(self._failed_queries[-5:])}"

        suggestions = await self._planner.suggest_restructure(outline, gaps)
        self._apply_restructure_suggestions(suggestions)

    async def _restructure_graph_for_node(
        self, node: TopicNode, gaps: str,
    ) -> None:
        """Request targeted restructuring for a specific node."""
        if self._graph is None:
            return
        outline = self._graph.get_outline()
        suggestions = await self._planner.suggest_restructure(outline, gaps)
        self._apply_restructure_suggestions(suggestions)

    def _apply_restructure_suggestions(
        self, suggestions: list[dict[str, Any]],
    ) -> None:
        """Apply restructure suggestions to the graph."""
        if self._graph is None:
            return
        for suggestion in suggestions:
            if suggestion.get("action") != "add":
                continue
            parent_name = suggestion.get("parent_name", "")
            parent = self._graph.find_by_name(parent_name)
            if parent is None:
                parent = self._graph.root
            try:
                new_node = self._graph.add_node(
                    name=suggestion.get("name", ""),
                    query=suggestion.get("query", ""),
                    parent_id=parent.id,
                    priority=suggestion.get("priority", 5),
                )
                self._graph.mark_leaf(new_node.id)
                new_node.status = "pending"
                logger.info(
                    "Restructure: added new leaf %r under %r.",
                    new_node.name, parent.name,
                )
            except (ValueError, KeyError) as exc:
                logger.warning("Restructure suggestion failed: %s", exc)

    # ------------------------------------------------------------------
    # Core Cycle (flat mode — kept for backward compatibility)
    # ------------------------------------------------------------------

    async def run_cycle(self) -> Optional[dict[str, Any]]:
        """Pop one task, research it, critique it, and handle the verdict.

        Returns the approved finding dict, or *None* if the task was rejected
        after max retries.
        """
        if not self._task_queue:
            logger.warning("run_cycle called with empty queue.")
            return None

        task = self._task_queue.popleft()
        reject_count = 0

        while reject_count < _MAX_REJECT_RETRIES:
            # Research
            result = await self._researcher.research(task)

            # Deduplication
            if result["summary"] and await self._kb.is_duplicate(result["summary"]):
                logger.info("Duplicate content for %r – skipping.", task["subtopic"])
                return None

            # Critic review
            verdict = await self._critic.review(task["subtopic"], result["summary"], topic=self._title)

            if verdict.get("status") == "PROCEED":
                # Save to knowledge base
                source_url = (
                    result["source_urls"][0] if result["source_urls"] else None
                )
                await self._kb.save(
                    topic=task["subtopic"],
                    content=result["summary"],
                    source_url=source_url,
                )

                finding = {
                    "subtopic": task["subtopic"],
                    "query": task["query"],
                    "summary": result["summary"],
                    "source_urls": result["source_urls"],
                }
                self._approved.append(finding)
                logger.info("Approved: %r", task["subtopic"])

                # Phase 2 — record success, reset stuck counter
                self._successful_queries.append(task["query"])
                self._consecutive_failures = 0

                # Ask Planner for derived follow-up tasks.
                # Use _title (short) rather than the full topic text so that
                # requirements-file runs don't pass hundreds of words into
                # every follow-up LLM call.
                follow_ups = await self._planner.decompose(
                    f"{self._title}: {task['subtopic']}",
                    known_topics=await self._kb.get_all_topics(),
                )
                for follow_up in follow_ups[:2]:  # limit to 2 derived tasks
                    self._task_queue.append(follow_up)

                return finding

            # Critic rejected – refine and retry
            missing = verdict.get("missing", "unspecified details")
            logger.info(
                "Rejected %r (attempt %d/%d). Missing: %s",
                task["subtopic"],
                reject_count + 1,
                _MAX_REJECT_RETRIES,
                missing,
            )
            refined_task = await self._planner.refine(task["subtopic"], missing)
            task = refined_task
            reject_count += 1

        # Phase 2 — record failure
        self._failed_queries.append(task["query"])

        # Phase 3 — stuck detection: if too many consecutive failures, re-plan
        self._consecutive_failures += 1
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "%d consecutive failures — triggering retrospective re-plan.",
                self._consecutive_failures,
            )
            self._consecutive_failures = 0
            self._task_queue.clear()
            await self.populate_queue(is_retrospective=True)

        logger.warning(
            "Task %r exhausted retries without passing the Critic.", task["subtopic"]
        )
        return None

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def generate_report(self) -> Path:
        """Consolidate all approved findings into a Markdown report."""
        safe_name = re.sub(r"[^\w\-]", "_", self._title.lower())
        report_path = self.reports_dir / f"{safe_name}.md"

        # Graph outline section
        graph_outline_section = ""
        if self._graph is not None:
            outline = self._graph.get_outline()
            if outline:
                graph_outline_section = f"## Research Plan\n\n```\n{outline}\n```\n"

        finding_sections: list[str] = []
        formulas: list[str] = []
        deps: set[str] = set()
        sources: list[str] = []

        for finding in self._approved:
            summary = finding.get("summary", "")
            subtopic = finding["subtopic"]
            finding_sections.append(f"### {subtopic}\n\n{summary}")

            # Extract LaTeX-style formulas (technical topics)
            for match in re.finditer(r"\$\$.*?\$\$|\$[^$\n]+\$", summary):
                formulas.append(match.group())

            # Extract Python library names (technical topics)
            for match in re.finditer(r"\b(import|from)\s+([\w.]+)", summary):
                deps.add(match.group(2).split(".")[0])

            sources.extend(finding.get("source_urls", []))

        # Build the findings block
        if finding_sections:
            findings_block = "## Findings\n\n" + "\n\n---\n\n".join(finding_sections)
        else:
            findings_block = "## Findings\n\n_No approved findings yet._"

        # Only include technical sections when there is actual content
        technical_parts: list[str] = []
        if formulas:
            technical_parts.append("## Math/Formulas\n\n" + "\n".join(formulas))
        if deps:
            technical_parts.append(
                "## Dependencies\n\n"
                + "\n".join(f"- `{d}`" for d in sorted(deps))
            )
        technical_sections = ("\n\n".join(technical_parts) + "\n\n") if technical_parts else ""

        content = _REPORT_TEMPLATE.format(
            topic=self._title,
            graph_outline_section=graph_outline_section,
            findings=findings_block,
            technical_sections=technical_sections,
            sources="\n".join(f"- {s}" for s in sources) or "_No sources._",
        )

        report_path.write_text(content, encoding="utf-8")
        logger.info("Report written to %s", report_path)
        return report_path
