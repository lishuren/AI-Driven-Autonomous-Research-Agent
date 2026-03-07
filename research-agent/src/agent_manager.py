"""
agent_manager.py – Orchestrates Planner, Researcher, and Critic.

Provides:
* ``AgentManager.run_cycle`` – executes one full research cycle for a task.
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

logger = logging.getLogger(__name__)

_REPORT_TEMPLATE = """\
# {topic}

{findings}
{technical_sections}
## Sources
{sources}
"""

_MAX_REJECT_RETRIES = 3
_MAX_CONSECUTIVE_FAILURES = 5


class AgentManager:
    """Coordinates the Planner → Researcher → Critic pipeline."""

    def __init__(
        self,
        topic: str,
        title: Optional[str] = None,
        model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
        reports_dir: str = "data/reports",
        db_path: str = "data/research.db",
    ) -> None:
        self.topic = topic
        self._title = title if title is not None else topic
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1 — pass SearchTool to Planner for pre-search vocabulary grounding
        _search_tool = SearchTool(max_results=3)
        self._planner = PlannerAgent(
            model=model,
            ollama_base_url=ollama_base_url,
            search_tool=_search_tool,
        )
        self._researcher = ResearcherAgent(model=model, ollama_base_url=ollama_base_url)
        self._critic = CriticAgent(model=model, ollama_base_url=ollama_base_url)
        self._kb = KnowledgeBase(db_path=db_path)

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
    # Queue Management
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
    # Core Cycle
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
            verdict = await self._critic.review(task["subtopic"], result["summary"])

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

                # Ask Planner for derived follow-up tasks
                follow_ups = await self._planner.decompose(
                    f"{self.topic}: {task['subtopic']}",
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
            findings=findings_block,
            technical_sections=technical_sections,
            sources="\n".join(f"- {s}" for s in sources) or "_No sources._",
        )

        report_path.write_text(content, encoding="utf-8")
        logger.info("Report written to %s", report_path)
        return report_path
