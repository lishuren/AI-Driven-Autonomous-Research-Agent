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

logger = logging.getLogger(__name__)

_REPORT_TEMPLATE = """\
# {topic}

## Implementation Logic
{logic_steps}

## Math/Formulas
{formulas}

## Dependencies
{dependencies}

## Sources
{sources}
"""

_MAX_REJECT_RETRIES = 3


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

        self._planner = PlannerAgent(model=model, ollama_base_url=ollama_base_url)
        self._researcher = ResearcherAgent(model=model, ollama_base_url=ollama_base_url)
        self._critic = CriticAgent(model=model, ollama_base_url=ollama_base_url)
        self._kb = KnowledgeBase(db_path=db_path)

        self._task_queue: deque[dict[str, str]] = deque()
        self._approved: list[dict[str, Any]] = []

    async def init(self) -> None:
        """Initialise the knowledge base."""
        await self._kb.init()

    async def close(self) -> None:
        """Shut down the knowledge base."""
        await self._kb.close()

    # ------------------------------------------------------------------
    # Queue Management
    # ------------------------------------------------------------------

    async def populate_queue(self) -> None:
        """Ask the Planner to seed the task queue with initial subtopics."""
        known = await self._kb.get_all_topics()
        tasks = await self._planner.decompose(self.topic, known_topics=known)
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

        logic_steps = []
        formulas: list[str] = []
        deps: set[str] = set()
        sources: list[str] = []

        for i, finding in enumerate(self._approved, start=1):
            summary = finding.get("summary", "")
            logic_steps.append(f"{i}. **{finding['subtopic']}** – {summary[:300]}…")

            # Extract LaTeX-style formulas
            for match in re.finditer(r"\$\$.*?\$\$|\$[^$\n]+\$", summary):
                formulas.append(match.group())

            # Extract library names
            for match in re.finditer(
                r"\b(import|from)\s+([\w.]+)", summary
            ):
                deps.add(match.group(2).split(".")[0])

            sources.extend(finding.get("source_urls", []))

        content = _REPORT_TEMPLATE.format(
            topic=self._title,
            logic_steps="\n".join(logic_steps) or "No approved steps yet.",
            formulas="\n".join(formulas) or "_No formulas extracted._",
            dependencies="\n".join(f"- `{d}`" for d in sorted(deps))
            or "_No dependencies extracted._",
            sources="\n".join(f"- {s}" for s in sources) or "_No sources._",
        )

        report_path.write_text(content, encoding="utf-8")
        logger.info("Report written to %s", report_path)
        return report_path
