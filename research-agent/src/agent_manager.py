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
import json
import logging
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.agents.critic import CriticAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.budget import BudgetTracker
from src.database.knowledge_base import KnowledgeBase
from src.tools.search_tool import SearchTool, set_budget
from src.tools.hub_scraper_tool import set_budget as set_hub_budget
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

    Removes date stamps (e.g. ``(2026-03-06)``) and normalises separators
    (``-``, ``–``, ``—``, ``|``) to spaces.
    """
    q = _DATE_PATTERNS.sub("", name)
    q = _SEPARATOR_RE.sub(" ", q)
    return q.strip()


def _build_inline_refs(source_urls: list[str]) -> str:
    """Return a Markdown inline-references string for *source_urls*.

    Returns an empty string when no URLs are provided.
    Example output: ``"\\n\\n*Sources: [1](https://a.com), [2](https://b.com)*"``
    """
    if not source_urls:
        return ""
    links = ", ".join(
        f"[{i + 1}]({url})" for i, url in enumerate(source_urls)
    )
    return f"\n\n*Sources: {links}*"


_REPORT_TEMPLATE = """\
# {topic}

> **Session elapsed:** {elapsed_str}

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
        llm_provider: str = "ollama",
        llm_api_key: Optional[str] = None,
        prompt_dir: Optional[str] = None,
        reports_dir: str = "data/reports",
        db_path: str = "data/research.db",
        max_depth: int = MAX_DEPTH,
        max_queries: Optional[int] = None,
        max_nodes: Optional[int] = None,
        max_credits: Optional[float] = None,
        warn_threshold: float = 0.80,
        task_json_path: Optional[Path] = None,
    ) -> None:
        self.topic = topic
        self._title = title if title is not None else topic
        self._user_prompt = user_prompt
        self._max_depth = max_depth
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._session_start: float = time.monotonic()

        # Path for task.json persistence (optional — None disables auto-save)
        self._task_json_path: Optional[Path] = task_json_path

        # Budget tracker — attach to the search module so every query is recorded
        self.budget = BudgetTracker(
            max_queries=max_queries,
            max_nodes=max_nodes,
            max_credits=max_credits,
            warn_threshold=warn_threshold,
        )
        set_budget(self.budget)
        set_hub_budget(self.budget)

        # Phase 1 — pass SearchTool to Planner for pre-search vocabulary grounding
        _search_tool = SearchTool(max_results=3)
        self._planner = PlannerAgent(
            model=model,
            ollama_base_url=ollama_base_url,
            search_tool=_search_tool,
            user_prompt=user_prompt,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            prompt_dir=prompt_dir,
        )
        self._researcher = ResearcherAgent(
            model=model, ollama_base_url=ollama_base_url,
            user_prompt=user_prompt,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            prompt_dir=prompt_dir,
        )
        self._critic = CriticAgent(
            model=model, ollama_base_url=ollama_base_url,
            user_prompt=user_prompt,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key,
            prompt_dir=prompt_dir,
        )
        self._kb = KnowledgeBase(db_path=db_path)

        # Topic graph (built during build_graph)
        self._graph: Optional[TopicGraph] = None

        # BFS depth cursor (initialised in build_graph; pre-set here for restore)
        self._current_research_depth: int = 0

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

    def _save_tree_json(self) -> None:
        """Save the topic graph as a hierarchical JSON tree alongside the report.

        Excludes nodes with no summary or consolidated_summary.
        Also triggers a task.json auto-save if a path was configured.
        """
        if self._graph is None:
            return
        safe_name = re.sub(r"[^\w\-]", "_", self._title.lower())
        json_path = self.reports_dir / f"{safe_name}.json"
        tree = self._graph.to_tree_dict(exclude_empty=True)
        json_path.write_text(
            json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info("Topic tree JSON saved to %s", json_path)

        # Persist task state for crash/quota recovery
        if self._task_json_path is not None:
            self.save_task()

    # ------------------------------------------------------------------
    # Task Persistence (save / restore)
    # ------------------------------------------------------------------

    def save_task(self, status: str = "in_progress") -> None:
        """Persist full task state to the configured task_json_path.

        Serializes the graph (if any), approved findings, and all tracking
        counters so the session can be resumed exactly where it left off.

        *status* should be ``"in_progress"`` for mid-run auto-saves and
        ``"completed"`` when the session finishes normally.
        """
        if self._task_json_path is None:
            return
        state: dict[str, Any] = {
            "version": 1,
            "topic": self.topic,
            "title": self._title,
            "user_prompt": self._user_prompt,
            "current_research_depth": self._current_research_depth,
            "approved": self._approved,
            "successful_queries": self._successful_queries,
            "failed_queries": self._failed_queries,
            "consecutive_failures": self._consecutive_failures,
            "status": status,
            "last_saved_iso": datetime.now(timezone.utc).isoformat(),
        }
        if self._graph is not None:
            state["graph"] = self._graph.to_dict()
        self._task_json_path.parent.mkdir(parents=True, exist_ok=True)
        self._task_json_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.debug("Task state saved to %s  [%s]", self._task_json_path, status)

    def restore_task(self, task_json_path: Path) -> bool:
        """Load and restore task state from *task_json_path*.

        Applies the saved graph, depth cursor, approved findings, and
        feedback counters so the agent resumes exactly where it left off.
        Nodes whose status was ``"researching"`` or ``"analyzing"`` are reset
        to ``"pending"`` so interrupted work is retried cleanly.

        Returns ``True`` if state was successfully restored, ``False`` if
        the file was missing, corrupt, or belongs to a completed session that
        should be re-generated.
        """
        if not task_json_path.is_file():
            return False
        try:
            state = json.loads(task_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not load task state from %s: %s — starting fresh.",
                task_json_path, exc,
            )
            return False

        if state.get("version", 1) > 1:
            logger.warning(
                "task.json version %s is newer than supported; ignoring.",
                state.get("version"),
            )
            return False

        prev_status = state.get("status", "in_progress")
        last_saved = state.get("last_saved_iso", "unknown")

        if prev_status == "completed":
            logger.info(
                "Restoring completed session from %s (saved %s).",
                task_json_path, last_saved,
            )
        else:
            logger.info(
                "Resuming in-progress session from %s (saved %s).",
                task_json_path, last_saved,
            )

        # Restore flat-mode tracking state
        self._approved = state.get("approved", [])
        self._successful_queries = state.get("successful_queries", [])
        self._failed_queries = state.get("failed_queries", [])
        self._consecutive_failures = state.get("consecutive_failures", 0)
        self._current_research_depth = state.get("current_research_depth", 0)

        # Restore graph (if present)
        if "graph" in state:
            self._graph = TopicGraph.from_dict(state["graph"])
            # Reset interrupted nodes so they will be retried
            for node in self._graph.get_all_nodes():
                if node.status in ("researching", "analyzing"):
                    node.status = "pending"
            logger.info(
                "Restored graph with %d nodes at depth %d.",
                self._graph.node_count(),
                self._current_research_depth,
            )

            # If the session was saved as "completed" but there are still pending
            # leaf nodes (e.g. from a post-consolidation restructure that ran out
            # of budget), reset the root and their direct parents back to
            # "completed" so the next run can research and re-consolidate them.
            if prev_status == "completed":
                pending_nodes = [
                    n for n in self._graph.get_all_nodes()
                    if n.status == "pending"
                ]
                if pending_nodes:
                    nodes_by_id = {n.id: n for n in self._graph.get_all_nodes()}
                    to_reset: set[str] = {self._graph.root.id}
                    for pn in pending_nodes:
                        for pid in pn.parent_ids:
                            to_reset.add(pid)
                    reset_count = 0
                    for nid in to_reset:
                        anc = nodes_by_id.get(nid)
                        if anc and anc.status == "consolidated":
                            anc.status = "completed"
                            reset_count += 1
                    if reset_count:
                        logger.info(
                            "Found %d pending nodes in a completed session — "
                            "reset %d consolidated ancestors to allow further research.",
                            len(pending_nodes),
                            reset_count,
                        )

        return True

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
        """Build the initial topic graph (root + first layer of children).

        Phase A: Quick research on root for context.
        Phase B: Analyze root → decompose into depth-1 children only.
        Deeper layers are created lazily during ``run_graph()`` (BFS).
        """
        root_name = self.topic.split("\n")[0][:100] if "\n" in self.topic else self.topic
        root_name = re.sub(r"^#+\s*", "", root_name).strip()
        root_query = _make_search_query(root_name)
        graph = TopicGraph(root_name=root_name, root_query=root_query)
        self._graph = graph
        self._current_research_depth = 0

        logger.info("Building topic graph for %r ...", root_name)

        # Phase A — quick initial research on root topic for context
        root_task = {"subtopic": root_name, "query": root_query}
        root_result = await self._researcher.research(root_task)
        root_summary = root_result.get("summary", "")
        if root_summary:
            graph.root.summary = root_summary
            graph.root.source_urls = root_result.get("source_urls", [])

        # Phase B — decompose root into depth-1 children only (no recursion)
        await self._decompose_node(graph, graph.root.id, root_summary)

        self._save_tree_json()
        logger.info(
            "Topic graph built: %d nodes.\n%s",
            graph.node_count(), graph.get_outline(),
        )
        return graph

    def _adaptive_max_children(self) -> int:
        """Return the recommended number of children based on remaining budget.

        Returns 0 when budget is critically low (force leaf).
        """
        frac = self.budget.budget_fraction_remaining()
        if frac < 0.10:
            return 0   # force leaf — almost no budget left
        if frac < 0.25:
            return 2
        if frac < 0.50:
            return 3
        return 5

    async def _decompose_node(
        self,
        graph: TopicGraph,
        node_id: str,
        context_summary: str = "",
    ) -> None:
        """Decompose a node into sub-topics (single level, no recursion).

        Children are created at the next depth level but are not themselves
        decomposed here — that happens lazily in ``run_graph()`` (BFS).
        The number of children is adaptive based on remaining budget.
        """
        node = graph.get_node(node_id)
        if node is None:
            return

        if node.depth >= self._max_depth:
            graph.mark_leaf(node_id)
            node.status = "pending"
            logger.info("Max depth reached for %r — marking as leaf.", node.name)
            return

        graph.mark_analyzing(node_id)

        planner_topic = node.query if node.query.strip() else node.name

        # Ask Planner if this topic is simple (leaf) or complex
        analysis = await self._planner.analyze(
            planner_topic,
            initial_summary=context_summary,
            description=node.description,
            main_topic=self.topic,
        )

        # Relevance gate: low-relevance topics become leaves immediately
        relevance = analysis.get("relevance", "high")
        if relevance == "low":
            graph.mark_leaf(node_id)
            node.status = "pending"
            logger.info(
                "Low relevance topic: %r — forcing leaf (not decomposing).", node.name,
            )
            return

        if analysis.get("is_leaf", False):
            graph.mark_leaf(node_id)
            node.status = "pending"  # ready for research
            logger.info("Leaf topic: %r — %s", node.name, analysis.get("reasoning", ""))
            return

        # Complex topic: decompose into sub-topics (one level only)
        # Adaptive: reduce branching when budget is running low
        max_children = self._adaptive_max_children()
        if max_children == 0:
            graph.mark_leaf(node_id)
            node.status = "pending"
            logger.info("Budget critical — forcing %r to leaf.", node.name)
            return

        known = graph.get_all_researched_names()
        children_names = [c.name for c in graph.get_children(node_id)]
        known.extend(children_names)

        sub_topics = await self._planner.decompose_hierarchical(
            planner_topic,
            description=node.description,
            known_subtopics=known,
            main_topic=self.topic,
            max_children=max_children,
        )

        if not sub_topics:
            graph.mark_leaf(node_id)
            node.status = "pending"
            return

        added_children = 0
        for st in sub_topics:
            if not self.budget.can_create_node():
                logger.info("Node budget exhausted — stopping decomposition of %r.", node.name)
                break
            try:
                graph.add_node(
                    name=st.get("name", st.get("query", "")),
                    query=st.get("query", ""),
                    parent_id=node_id,
                    priority=st.get("priority", 5),
                    description=st.get("description", ""),
                )
            except ValueError as exc:
                logger.warning("Skipping invalid sub-topic under %r: %s", node.name, exc)
                continue
            added_children += 1
            self.budget.record_node()

        if added_children == 0:
            graph.mark_leaf(node_id)
            node.status = "pending"
            logger.info("No valid sub-topics for %r — forcing leaf.", node.name)

    # ------------------------------------------------------------------
    # Graph-Based Research Orchestration
    # ------------------------------------------------------------------

    async def run_graph(self) -> Optional[dict[str, Any]]:
        """Process the next available node in the topic graph (BFS layer-by-layer).

        Processes nodes at ``_current_research_depth`` first:
        1. Research pending leaf nodes at this depth.
        2. Analyze+decompose pending non-leaf nodes at this depth.
        When all nodes at the current depth are done, advance to the next depth.
        Finally, consolidate parents bottom-up.

        Returns the approved finding dict, or None if nothing was processed.
        """
        if self._graph is None:
            return None

        depth = self._current_research_depth

        # --- Step 1: Process nodes at current depth ---
        nodes_at_depth = self._graph.get_nodes_at_depth(depth)

        # 1a: Research pending leaf nodes at current depth
        pending_leaves = sorted(
            [n for n in nodes_at_depth if n.is_leaf and n.status == "pending"],
            key=lambda n: n.priority, reverse=True,
        )
        if pending_leaves and not self.budget.is_exhausted():
            return await self._research_node(pending_leaves[0])

        # 1a.5: Research decomposed non-leaf nodes that have no direct summary yet.
        # This ensures every layer is searched even when the session timer expires
        # before deeper leaves are reached.  The node's own query is searched now;
        # bottom-up consolidation will later synthesise children summaries on top.
        unsearched_nonleaf = sorted(
            [
                n for n in nodes_at_depth
                if not n.is_leaf
                and n.status == "analyzing"
                and not n.summary
            ],
            key=lambda n: n.priority, reverse=True,
        )
        if unsearched_nonleaf and not self.budget.is_exhausted():
            return await self._research_node(unsearched_nonleaf[0])

        # 1b: Analyze+decompose pending non-leaf nodes at current depth.
        # Nodes at exactly _max_depth are included: _decompose_node() handles them
        # by calling mark_leaf() immediately (no LLM calls), making them researchable.
        pending_nonleaf = [
            n for n in nodes_at_depth
            if not n.is_leaf and n.status == "pending"
        ]
        if pending_nonleaf:
            node = pending_nonleaf[0]
            await self._decompose_node(self._graph, node.id)
            return None  # decomposition doesn't produce a finding

        # --- Step 2: Check if current depth is complete ---
        all_done = all(
            n.status in ("completed", "consolidated", "failed", "analyzing")
            and (n.is_leaf or n.children_ids)  # non-leaves must have been decomposed
            for n in nodes_at_depth
        ) if nodes_at_depth else True

        if all_done and depth < self._graph.max_depth_present():
            self._current_research_depth = depth + 1
            logger.info("Layer %d complete — advancing to layer %d.", depth, depth + 1)
            # Merge similar pending nodes at the new depth to reduce redundancy
            merged = self._graph.merge_similar_nodes(depth=depth + 1)
            if merged:
                logger.info("Merged %d similar nodes at depth %d.", merged, depth + 1)
            return None  # will process next depth on next call

        # --- Step 3: Consolidate parents bottom-up ---
        consolidatable = self._graph.get_ready_for_consolidation()
        if consolidatable:
            return await self._consolidate_node(consolidatable[0])

        # --- Step 4: Safety net — orphan pending leaves at non-current depths ---
        # Nodes can become stranded if restructuring resets a cross-referenced
        # node's status to pending while leaving its depth unchanged (e.g. depth 1
        # when _current_research_depth has already advanced to 2).
        orphan_pending = sorted(
            [
                n for n in self._graph.get_all_nodes()
                if n.is_leaf and n.status == "pending" and n.depth != depth
            ],
            key=lambda n: n.priority, reverse=True,
        )
        if orphan_pending and not self.budget.is_exhausted():
            logger.warning(
                "Orphan pending leaf %r at depth %d (current depth=%d) — rescuing.",
                orphan_pending[0].name, orphan_pending[0].depth, depth,
            )
            return await self._research_node(orphan_pending[0])

        return None

    def progress_summary(self) -> str:
        """Return a one-line human-readable progress string for heartbeat logging."""
        if self._graph is None:
            queued = len([t for t in getattr(self, "_queue", [])]) if hasattr(self, "_queue") else 0
            return f"flat-mode queue={queued}  approved={len(self._approved)}"
        nodes = self._graph.get_all_nodes()
        done  = sum(1 for n in nodes if n.status in ("completed", "consolidated"))
        pend  = sum(1 for n in nodes if n.status == "pending")
        active = next((n.name for n in nodes if n.status == "researching"), None)
        consolidating = next((n.name for n in nodes if n.status == "analyzing"), None)
        working = f"  working={active!r}" if active else (f"  consolidating={consolidating!r}" if consolidating else "")
        return (
            f"approved={len(self._approved)}  done={done}/{len(nodes)}"
            f"  pending={pend}  depth={self._current_research_depth}{working}"
        )

    def extend_graph_for_deeper_research(self) -> int:
        """Re-open deepest nodes for further decomposition when _max_depth has grown.

        Called after restoring a completed session with a higher ``--max-depth``
        than the original run.  Returns the number of nodes re-opened (0 if there
        is nothing to extend).
        """
        if self._graph is None:
            return 0

        current_max = self._graph.max_depth_present()
        if current_max >= self._max_depth:
            return 0  # already at or beyond target depth

        # Re-open every node at the current deepest level so _decompose_node will
        # expand it into a new generation of children.
        deepest_nodes = self._graph.get_nodes_at_depth(current_max)
        reopened = 0
        for node in deepest_nodes:
            if node.status in ("completed", "consolidated", "failed"):
                node.is_leaf = False
                node.status = "pending"
                reopened += 1

        if reopened == 0:
            return 0

        # Walk back up the ancestor chain: reset "consolidated" → "completed" so
        # those nodes are picked up for re-consolidation once the new leaves are done.
        for depth in range(current_max - 1, -1, -1):
            for node in self._graph.get_nodes_at_depth(depth):
                if node.status == "consolidated":
                    node.status = "completed"

        # Also reset root (depth 0 is included above, but be explicit)
        if self._graph.root.status == "consolidated":
            self._graph.root.status = "completed"

        self._current_research_depth = current_max
        logger.info(
            "Extending graph: re-opened %d node(s) at depth %d for decomposition "
            "(new max depth: %d).",
            reopened, current_max, self._max_depth,
        )
        return reopened

    def has_graph_work(self) -> bool:
        """Return True if there are still nodes to research, decompose, or consolidate."""
        if self._graph is None:
            return False
        if self._graph.is_complete():
            return False

        # Budget exhaustion — only consolidation work may remain
        budget_ok = not self.budget.is_exhausted()

        # Check all depths for pending work
        for depth in range(self._graph.max_depth_present() + 1):
            nodes = self._graph.get_nodes_at_depth(depth)
            for n in nodes:
                if n.is_leaf and n.status == "pending" and budget_ok:
                    return True
                if not n.is_leaf and n.status == "pending":
                    return True
                if not n.is_leaf and n.status == "analyzing" and not n.summary and budget_ok:
                    return True

        # Check for consolidation work
        if self._graph.get_ready_for_consolidation():
            return True

        return False

    async def _research_node(self, node: TopicNode) -> Optional[dict[str, Any]]:
        """Research a single leaf node using the Researcher → Critic pipeline."""
        assert self._graph is not None

        # Budget guard — stop researching when credits exhausted
        if self.budget.is_exhausted():
            logger.info("Budget exhausted — skipping research for %r.", node.name)
            return None

        self._graph.mark_researching(node.id)
        task = {"subtopic": node.name, "query": node.query}
        reject_count = 0
        pending_total = sum(
            1 for n in self._graph.get_all_nodes() if n.status in ("pending", "researching")
        )
        done_total = sum(
            1 for n in self._graph.get_all_nodes()
            if n.status in ("completed", "consolidated")
        )
        logger.info(
            "[Researching] %r  (done=%d, pending≈%d, depth=%d)",
            node.name, done_total, pending_total, node.depth,
        )

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
                self._save_tree_json()

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
            refined_task = await self._planner.refine(node.name, missing, main_topic=self._title)
            task = refined_task
            reject_count += 1

        # Exhausted retries
        self._failed_queries.append(task["query"])
        self._consecutive_failures += 1
        self._graph.mark_failed(node.id)
        self._save_tree_json()
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
            self._save_tree_json()
            return None

        logger.info("[Consolidating] %r  (%d child summaries)", node.name, len(child_summaries))
        consolidated = await self._planner.consolidate_summaries(
            node.name, child_summaries,
        )

        # Critic reviews consolidated summary
        verdict = await self._critic.review(
            node.name, consolidated, topic=self._title,
        )

        if verdict.get("status") == "PROCEED":
            self._graph.mark_consolidated(node.id, consolidated)
            self._save_tree_json()

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
        self._save_tree_json()
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
        add_suggestions = [
            suggestion for suggestion in suggestions
            if suggestion.get("action") == "add"
        ]
        add_suggestions.sort(key=lambda s: s.get("priority", 5), reverse=True)

        budget_used = 1.0 - self.budget.budget_fraction_remaining()
        status_counts = self._graph.get_status_counts()
        unresolved_count = (
            status_counts.get("pending", 0)
            + status_counts.get("analyzing", 0)
            + status_counts.get("failed", 0)
        )

        if not self.budget.can_create_node():
            max_additions = 0
        elif budget_used >= 0.75 or unresolved_count >= 12:
            max_additions = 1
        elif budget_used >= 0.50 or unresolved_count >= 8:
            max_additions = 2
        else:
            max_additions = 3

        if len(add_suggestions) > max_additions:
            logger.info(
                "Restructure: limiting add suggestions from %d to %d "
                "(budget_used=%.0f%%, unresolved=%d).",
                len(add_suggestions), max_additions, budget_used * 100, unresolved_count,
            )

        for suggestion in add_suggestions[:max_additions]:
            node_name = suggestion.get("name", "")
            # Skip if a node with this name already exists — resetting an
            # existing node's status would strand it at its original depth,
            # causing run_graph to spin without making progress.
            if self._graph.find_by_name(node_name) is not None:
                logger.info("Restructure: skipping already-existing node %r.", node_name)
                continue
            parent_name = suggestion.get("parent_name", "")
            parent = self._graph.find_by_name(parent_name)
            if parent is None:
                parent = self._graph.root
            try:
                new_node = self._graph.add_node(
                    name=node_name,
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
            refined_task = await self._planner.refine(task["subtopic"], missing, main_topic=self._title)
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

    def generate_report(self, elapsed_seconds: Optional[float] = None) -> Path:
        """Consolidate all approved findings into a Markdown report.

        When a topic graph is available the report mirrors the hierarchy:
        depth-0 (root) → ``##``, depth-1 → ``###``, depth-2+ → ``####``.
        Falls back to a flat list of approved findings otherwise.

        *elapsed_seconds* — if supplied, overrides the internal session clock
        (useful for the final report call which has an accurate wall-clock value).
        """
        safe_name = re.sub(r"[^\w\-]", "_", self._title.lower())
        report_path = self.reports_dir / f"{safe_name}.md"

        formulas: list[str] = []
        deps: set[str] = set()
        sources: list[str] = []

        def _extract_technical(summary: str) -> None:
            for match in re.finditer(r"\$\$.*?\$\$|\$[^$\n]+\$", summary):
                formulas.append(match.group())
            for match in re.finditer(r"\b(import|from)\s+([\w.]+)", summary):
                deps.add(match.group(2).split(".")[0])

        # ── graph-based hierarchical report ──────────────────────────
        if self._graph is not None:
            findings_block = self._graph_findings_block(
                self._graph, _extract_technical, sources,
            )
        # ── flat fallback ────────────────────────────────────────────
        elif self._approved:
            sections: list[str] = []
            for finding in self._approved:
                summary = finding.get("summary", "")
                source_urls = finding.get("source_urls", [])
                inline_refs = _build_inline_refs(source_urls)
                sections.append(
                    f"### {finding['subtopic']}\n\n{summary}{inline_refs}"
                )
                _extract_technical(summary)
                sources.extend(source_urls)
            findings_block = "## Findings\n\n" + "\n\n---\n\n".join(sections)
        else:
            findings_block = "## Findings\n\n_No approved findings yet._"

        # Graph outline section
        graph_outline_section = ""
        if self._graph is not None:
            outline = self._graph.get_outline(report_mode=True)
            if outline:
                counts = self._graph.get_status_counts()
                unresolved_parts = []
                for status in ("pending", "analyzing", "failed"):
                    count = counts.get(status, 0)
                    if count:
                        unresolved_parts.append(f"{count} {status}")
                unresolved_line = ""
                if unresolved_parts:
                    unresolved_line = (
                        "\n_Unresolved graph state retained in task data: "
                        + ", ".join(unresolved_parts)
                        + "._\n"
                    )
                graph_outline_section = f"## Research Plan\n\n```\n{outline}\n```{unresolved_line}\n"

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

        secs = elapsed_seconds if elapsed_seconds is not None else (time.monotonic() - self._session_start)
        mins, whole_secs = divmod(int(secs), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            elapsed_str = f"{hours}h {mins:02d}m {whole_secs:02d}s"
        elif mins:
            elapsed_str = f"{mins}m {whole_secs:02d}s"
        else:
            elapsed_str = f"{whole_secs}s"

        content = _REPORT_TEMPLATE.format(
            topic=self._title,
            elapsed_str=elapsed_str,
            graph_outline_section=graph_outline_section,
            findings=findings_block,
            technical_sections=technical_sections,
            sources="\n".join(f"- {s}" for s in sources) or "_No sources._",
        )

        report_path.write_text(content, encoding="utf-8")
        logger.info("Report written to %s", report_path)
        return report_path

    # ------------------------------------------------------------------
    # Report helper — walk graph hierarchy
    # ------------------------------------------------------------------

    @staticmethod
    def _graph_findings_block(
        graph: TopicGraph,
        extract_fn: Any,
        sources: list[str],
    ) -> str:
        """Build the Findings section by DFS-walking the topic graph.

        Heading depth follows node depth: depth-1 children → ``###``,
        depth-2 → ``####``, etc.  The root node's own summary (if any)
        is emitted as intro prose directly under ``## Findings``.
        Each section includes inline source reference links when available.
        """
        lines: list[str] = []

        visited_ids: set[str] = set()

        def _walk(node: TopicNode, *, is_root: bool = False) -> None:
            if node.id in visited_ids:
                return
            visited_ids.add(node.id)

            summary = node.consolidated_summary or node.summary or ""
            has_content = summary and summary != "(duplicate — skipped)"

            if has_content:
                inline_refs = _build_inline_refs(node.source_urls)
                if is_root:
                    # Root summary becomes intro prose under ## Findings
                    lines.append(f"{summary}{inline_refs}")
                else:
                    heading_level = min(node.depth + 2, 6)  # depth 1→###, 2→####
                    prefix = "#" * heading_level
                    heading_text = node.name.strip() or node.query.strip() or f"[Unnamed-{node.id[:6]}]"
                    lines.append(
                        f"{prefix} {heading_text}\n\n{summary}{inline_refs}"
                    )
                extract_fn(summary)
                sources.extend(node.source_urls)

            for child in graph.get_children(node.id):
                _walk(child)

        _walk(graph.root, is_root=True)

        if lines:
            return "## Findings\n\n" + "\n\n---\n\n".join(lines)
        return "## Findings\n\n_No approved findings yet._"
