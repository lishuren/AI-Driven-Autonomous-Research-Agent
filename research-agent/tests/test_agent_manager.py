"""
Integration tests for AgentManager (all external calls mocked).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def tmp_reports(tmp_path):
    return str(tmp_path / "reports")


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "research.db")


class TestAgentManager:
    def _build_manager(self, topic, reports_dir, db_path):
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            manager = AgentManager(
                topic=topic,
                reports_dir=reports_dir,
                db_path=db_path,
            )
        return manager

    def test_init_creates_reports_dir(self, event_loop, tmp_reports, tmp_db):
        from src.agent_manager import AgentManager
        from src.database.knowledge_base import KnowledgeBase

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"):
            manager = AgentManager(
                topic="Test",
                reports_dir=tmp_reports,
                db_path=tmp_db,
            )

        # Use real KnowledgeBase with tmp_db
        from src.database.knowledge_base import KnowledgeBase
        manager._kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(manager.init())

        assert Path(tmp_reports).exists()
        event_loop.run_until_complete(manager.close())

    def test_run_cycle_approved(self, event_loop, tmp_reports, tmp_db):
        from src.agent_manager import AgentManager
        from src.database.knowledge_base import KnowledgeBase

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"):
            manager = AgentManager(
                topic="RSI",
                reports_dir=tmp_reports,
                db_path=tmp_db,
            )

        manager._kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(manager._kb.init())

        # Seed task queue manually
        manager._task_queue.append({"subtopic": "RSI", "query": "RSI formula python"})

        # Mock researcher result
        manager._researcher.research = AsyncMock(return_value={
            "subtopic": "RSI",
            "query": "RSI formula python",
            "summary": "Step 1: import pandas\n$RSI = 100 - 100/(1+RS)$\nfrom pandas import DataFrame",
            "source_urls": ["https://example.com"],
            "raw_content": "raw",
        })

        # Mock critic: PROCEED
        manager._critic.review = AsyncMock(return_value={
            "status": "PROCEED",
            "checks": {"logical_steps": True, "math_formulas": True, "python_libraries": True},
            "missing": "",
        })

        # Mock planner follow-ups
        manager._planner.decompose = AsyncMock(return_value=[
            {"subtopic": "RSI advanced", "query": "RSI advanced python"},
        ])

        finding = event_loop.run_until_complete(manager.run_cycle())

        assert finding is not None
        assert finding["subtopic"] == "RSI"
        assert len(manager._approved) == 1

        event_loop.run_until_complete(manager._kb.close())

    def test_run_cycle_rejected_then_refined(self, event_loop, tmp_reports, tmp_db):
        from src.agent_manager import AgentManager
        from src.database.knowledge_base import KnowledgeBase

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"):
            manager = AgentManager(
                topic="MACD",
                reports_dir=tmp_reports,
                db_path=tmp_db,
            )

        manager._kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(manager._kb.init())

        manager._task_queue.append({"subtopic": "MACD", "query": "MACD formula"})

        # Researcher always returns a weak summary
        manager._researcher.research = AsyncMock(return_value={
            "subtopic": "MACD",
            "query": "MACD formula",
            "summary": "MACD is an indicator.",
            "source_urls": [],
            "raw_content": "",
        })

        # Critic always rejects
        manager._critic.review = AsyncMock(return_value={
            "status": "REJECT",
            "checks": {"logical_steps": False, "math_formulas": False, "python_libraries": False},
            "missing": "Missing formula and steps",
        })

        # Planner refine returns a new task
        manager._planner.refine = AsyncMock(return_value={
            "subtopic": "MACD (refined)",
            "query": "MACD formula python pandas EMA",
        })
        manager._planner.decompose = AsyncMock(return_value=[])

        result = event_loop.run_until_complete(manager.run_cycle())

        # Should be None after exhausting retries
        assert result is None

        event_loop.run_until_complete(manager._kb.close())

    def test_generate_report(self, tmp_reports, tmp_db):
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            manager = AgentManager(
                topic="RSI Strategy",
                reports_dir=tmp_reports,
                db_path=tmp_db,
            )

        manager._approved = [
            {
                "subtopic": "RSI formula",
                "query": "RSI formula python",
                "summary": "1. import pandas\n$RSI = 100 - 100/(1+RS)$",
                "source_urls": ["https://example.com"],
            }
        ]

        report_path = manager.generate_report()

        assert report_path.exists()
        content = report_path.read_text()
        assert "RSI Strategy" in content
        assert "RSI formula" in content

    def test_generate_report_uses_title_for_filename(self, tmp_reports, tmp_db):
        """When a title is provided it is used for the report file name and heading."""
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            manager = AgentManager(
                topic="## Research\nFull spec…\n\n## Output\nPython code.",
                title="my_spec",
                reports_dir=tmp_reports,
                db_path=tmp_db,
            )

        manager._approved = []
        report_path = manager.generate_report()

        assert report_path.name == "my_spec.md"
        content = report_path.read_text()
        assert "my_spec" in content
        # The long spec text should NOT appear as the report heading
        assert "## Research" not in content.split("\n")[0]
