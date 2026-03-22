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


class TestMakeSearchQuery:
    def test_strips_iso_date_fullwidth_parens(self):
        from src.agent_manager import _make_search_query
        assert _make_search_query("在线 TRPG 市场分析 - 中国市场（2026-03-06）") == "在线 TRPG 市场分析 中国市场"

    def test_strips_iso_date_ascii_parens(self):
        from src.agent_manager import _make_search_query
        assert _make_search_query("Market analysis (2026-03-06)") == "Market analysis"

    def test_normalises_dash_separator(self):
        from src.agent_manager import _make_search_query
        assert _make_search_query("Topic A - Sub B") == "Topic A Sub B"

    def test_no_change_for_plain_query(self):
        from src.agent_manager import _make_search_query
        assert _make_search_query("online TRPG China market") == "online TRPG China market"

    def test_strips_year_only_parens(self):
        from src.agent_manager import _make_search_query
        assert _make_search_query("Annual report（2025）") == "Annual report"


class TestAgentManager:
    def _build_manager(self, topic, reports_dir, db_path, **kwargs):
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            manager = AgentManager(
                topic=topic,
                reports_dir=reports_dir,
                db_path=db_path,
                **kwargs,
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
        manager = self._build_manager("RSI Strategy", tmp_reports, tmp_db)

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
        # Topic-neutral structure: findings section, no hardcoded technical headings
        assert "## Findings" in content
        # Technical sections only appear when content exists
        assert "## Math/Formulas" in content   # formula present in summary
        assert "## Dependencies" in content    # 'import pandas' in summary
        assert "## Implementation Logic" not in content

    def test_generate_report_uses_title_for_filename(self, tmp_reports, tmp_db):
        """When a title is provided it is used for the report file name and heading."""
        manager = self._build_manager(
            "## Research\nFull spec…\n\n## Output\nPython code.",
            tmp_reports, tmp_db, title="my_spec",
        )

        manager._approved = []
        report_path = manager.generate_report()

        assert report_path.name == "my_spec.md"
        content = report_path.read_text()
        assert "my_spec" in content
        # The long spec text should NOT appear as the report heading
        assert "## Research" not in content.split("\n")[0]

    # ------------------------------------------------------------------
    # Phase 2: feedback tracking
    # ------------------------------------------------------------------

    def test_run_cycle_tracks_successful_query(self, event_loop, tmp_reports, tmp_db):
        """Approved results append the query to _successful_queries."""
        from src.agent_manager import AgentManager
        from src.database.knowledge_base import KnowledgeBase

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"):
            manager = AgentManager(topic="RSI", reports_dir=tmp_reports, db_path=tmp_db)

        manager._kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(manager._kb.init())

        manager._task_queue.append({"subtopic": "RSI", "query": "RSI formula python"})
        manager._researcher.research = AsyncMock(return_value={
            "subtopic": "RSI", "query": "RSI formula python",
            "summary": "RSI details", "source_urls": [], "raw_content": "",
        })
        manager._critic.review = AsyncMock(return_value={
            "status": "PROCEED", "checks": {}, "missing": "",
        })
        manager._planner.decompose = AsyncMock(return_value=[])

        event_loop.run_until_complete(manager.run_cycle())

        assert "RSI formula python" in manager._successful_queries
        assert manager._consecutive_failures == 0
        event_loop.run_until_complete(manager._kb.close())

    def test_run_cycle_tracks_failed_query(self, event_loop, tmp_reports, tmp_db):
        """Exhausted-retry tasks append the last query to _failed_queries."""
        from src.agent_manager import AgentManager
        from src.database.knowledge_base import KnowledgeBase

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"):
            manager = AgentManager(topic="MACD", reports_dir=tmp_reports, db_path=tmp_db)

        manager._kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(manager._kb.init())

        manager._task_queue.append({"subtopic": "MACD", "query": "MACD formula"})
        manager._researcher.research = AsyncMock(return_value={
            "subtopic": "MACD", "query": "MACD formula",
            "summary": "weak", "source_urls": [], "raw_content": "",
        })
        manager._critic.review = AsyncMock(return_value={
            "status": "REJECT", "checks": {}, "missing": "missing",
        })
        manager._planner.refine = AsyncMock(return_value={
            "subtopic": "MACD (refined)", "query": "MACD refined query",
        })
        manager._planner.decompose = AsyncMock(return_value=[])
        manager._planner.decompose_retrospective = AsyncMock(return_value=[])

        event_loop.run_until_complete(manager.run_cycle())

        assert len(manager._failed_queries) == 1
        assert manager._consecutive_failures == 1
        event_loop.run_until_complete(manager._kb.close())

    # ------------------------------------------------------------------
    # Phase 3: stuck detection
    # ------------------------------------------------------------------

    def test_run_cycle_triggers_retrospective_when_stuck(self, event_loop, tmp_reports, tmp_db):
        """After _MAX_CONSECUTIVE_FAILURES, a retrospective re-plan is triggered."""
        from src.agent_manager import AgentManager, _MAX_CONSECUTIVE_FAILURES
        from src.database.knowledge_base import KnowledgeBase

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"):
            manager = AgentManager(topic="Bollinger", reports_dir=tmp_reports, db_path=tmp_db)

        manager._kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(manager._kb.init())

        # Pre-load failure counter so one more failure tips it over the threshold
        manager._consecutive_failures = _MAX_CONSECUTIVE_FAILURES - 1

        manager._task_queue.append({"subtopic": "B", "query": "bollinger bands"})
        manager._researcher.research = AsyncMock(return_value={
            "subtopic": "B", "query": "bollinger bands",
            "summary": "weak", "source_urls": [], "raw_content": "",
        })
        manager._critic.review = AsyncMock(return_value={
            "status": "REJECT", "checks": {}, "missing": "x",
        })
        manager._planner.refine = AsyncMock(return_value={
            "subtopic": "B refined", "query": "bollinger bands refined",
        })
        new_tasks = [{"subtopic": "B new", "query": "bollinger new angle"}]
        manager._planner.decompose_retrospective = AsyncMock(return_value=new_tasks)

        event_loop.run_until_complete(manager.run_cycle())

        # Counter should have been reset and retrospective called
        assert manager._consecutive_failures == 0
        manager._planner.decompose_retrospective.assert_called_once()
        event_loop.run_until_complete(manager._kb.close())


class TestGraphReport:
    """Tests for graph-based generate_report and _save_tree_json."""

    def _build_manager(self, topic, reports_dir, db_path, **kwargs):
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            manager = AgentManager(
                topic=topic,
                reports_dir=reports_dir,
                db_path=db_path,
                **kwargs,
            )
        return manager

    def test_generate_report_uses_graph_hierarchy(self, tmp_path):
        """When a graph exists, generate_report walks it hierarchically."""
        from src.topic_graph import TopicGraph

        manager = self._build_manager("AI Research", str(tmp_path / "reports"), str(tmp_path / "db"))
        g = TopicGraph(root_name="AI Research", root_query="AI")
        g.root.summary = "AI overview prose"
        c1 = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        c2 = g.add_node(name="Vision", query="CV", parent_id=g.root.id)
        g.mark_leaf(c1.id)
        g.mark_leaf(c2.id)
        g.mark_researched(c1.id, "NLP is about language.", ["https://nlp.example.com"])
        g.mark_researched(c2.id, "Vision is about images.", ["https://cv.example.com"])
        g.mark_consolidated(g.root.id, "AI consolidated summary")
        manager._graph = g

        report_path = manager.generate_report()
        content = report_path.read_text()

        assert "# AI Research" in content
        assert "## Findings" in content
        # Root summary appears as intro (no heading for root)
        assert "AI consolidated summary" in content
        # Children get ### headings (depth 1 → ###)
        assert "### NLP" in content
        assert "### Vision" in content
        assert "NLP is about language." in content
        assert "https://nlp.example.com" in content

    def test_save_tree_json(self, tmp_path):
        """_save_tree_json creates a JSON file alongside the report."""
        from src.topic_graph import TopicGraph

        manager = self._build_manager("AI Research", str(tmp_path / "reports"), str(tmp_path / "db"))
        g = TopicGraph(root_name="AI Research", root_query="AI")
        g.root.summary = "AI overview"
        c = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        g.mark_leaf(c.id)
        g.mark_researched(c.id, "NLP summary", [])
        manager._graph = g

        manager._save_tree_json()

        json_files = list(manager.reports_dir.glob("*.json"))
        assert len(json_files) == 1
        tree = json.loads(json_files[0].read_text())
        assert tree["name"] == "AI Research"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["name"] == "NLP"

    def test_generate_report_no_graph_flat_fallback(self, tmp_path):
        """Without a graph, flat _approved list is used."""
        manager = self._build_manager("Flat Topic", str(tmp_path / "reports"), str(tmp_path / "db"))
        manager._approved = [
            {
                "subtopic": "Sub A",
                "query": "sub a query",
                "summary": "Summary of A.",
                "source_urls": ["https://a.example.com"],
            }
        ]

        report_path = manager.generate_report()
        content = report_path.read_text()

        assert "# Flat Topic" in content
        assert "### Sub A" in content
        assert "Summary of A." in content


class TestBudgetIntegration:
    """Tests for budget integration in AgentManager."""

    def _build_manager(self, topic, reports_dir, db_path, **kwargs):
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            manager = AgentManager(
                topic=topic,
                reports_dir=reports_dir,
                db_path=db_path,
                **kwargs,
            )
        return manager

    def test_budget_tracker_created(self, tmp_path):
        """AgentManager should create a BudgetTracker with provided limits."""
        manager = self._build_manager(
            "Test", str(tmp_path / "reports"), str(tmp_path / "db"),
            max_queries=10, max_nodes=20, max_credits=100.0,
        )
        assert manager.budget._max_queries == 10
        assert manager.budget._max_nodes == 20
        assert manager.budget._max_credits == 100.0

    def test_budget_default_unlimited(self, tmp_path):
        """Default budget should be unlimited."""
        manager = self._build_manager(
            "Test", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        assert manager.budget.can_query()
        assert manager.budget.can_create_node()
        assert not manager.budget.is_exhausted()

    def test_has_graph_work_false_when_budget_exhausted(self, tmp_path):
        """has_graph_work should return False for research when budget exhausted."""
        from src.topic_graph import TopicGraph

        manager = self._build_manager(
            "Test", str(tmp_path / "reports"), str(tmp_path / "db"),
            max_queries=0,
        )
        g = TopicGraph(root_name="Test", root_query="test")
        c = g.add_node(name="Child", query="child", parent_id=g.root.id)
        g.mark_leaf(c.id)
        # Mark the root as researched so it doesn't show as pending decomposition work
        g.mark_researched(g.root.id, summary="root summary")
        manager._graph = g

        # Budget is exhausted (max_queries=0), pending leaf should not count
        assert not manager.has_graph_work()

    def test_adaptive_max_children_full_budget(self, tmp_path):
        """With full budget, should return 5."""
        manager = self._build_manager(
            "Test", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        assert manager._adaptive_max_children() == 5

    def test_adaptive_max_children_low_budget(self, tmp_path):
        """With <25% budget, should return 2."""
        manager = self._build_manager(
            "Test", str(tmp_path / "reports"), str(tmp_path / "db"),
            max_queries=100,
        )
        # Use up 80% of queries
        for _ in range(80):
            manager.budget.record_query(credits=0)
        assert manager._adaptive_max_children() == 2

    def test_adaptive_max_children_critical_budget(self, tmp_path):
        """With <10% budget remaining, should return 0 (force leaf)."""
        manager = self._build_manager(
            "Test", str(tmp_path / "reports"), str(tmp_path / "db"),
            max_queries=100,
        )
        for _ in range(95):
            manager.budget.record_query(credits=0)
        assert manager._adaptive_max_children() == 0


class TestTaskJsonSaveRestore:
    """Tests for AgentManager.save_task() and restore_task()."""

    def _build_manager(self, topic, reports_dir, db_path, **kwargs):
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            return AgentManager(topic=topic, reports_dir=reports_dir,
                                db_path=db_path, **kwargs)

    def test_save_task_writes_json_file(self, tmp_path):
        task_path = tmp_path / "task.json"
        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
            task_json_path=task_path,
        )
        manager.save_task()
        assert task_path.is_file()
        data = json.loads(task_path.read_text(encoding="utf-8"))
        assert data["topic"] == "AI Research"
        assert data["status"] == "in_progress"
        assert "last_saved_iso" in data
        assert data["version"] == 1

    def test_save_task_writes_completed_status(self, tmp_path):
        task_path = tmp_path / "task.json"
        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
            task_json_path=task_path,
        )
        manager.save_task(status="completed")
        data = json.loads(task_path.read_text(encoding="utf-8"))
        assert data["status"] == "completed"

    def test_save_task_includes_graph_when_present(self, tmp_path):
        from src.topic_graph import TopicGraph

        task_path = tmp_path / "task.json"
        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
            task_json_path=task_path,
        )
        manager._graph = TopicGraph(root_name="AI Research", root_query="AI")
        manager.save_task()
        data = json.loads(task_path.read_text(encoding="utf-8"))
        assert "graph" in data
        assert "root_id" in data["graph"]
        assert "nodes" in data["graph"]

    def test_save_task_no_path_does_nothing(self, tmp_path):
        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        # Should not raise and should not create any file
        manager.save_task()
        assert not (tmp_path / "task.json").exists()

    def test_restore_task_returns_false_when_file_missing(self, tmp_path):
        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        result = manager.restore_task(tmp_path / "nonexistent.json")
        assert result is False

    def test_restore_task_returns_false_for_corrupt_json(self, tmp_path):
        bad = tmp_path / "task.json"
        bad.write_text("{corrupt", encoding="utf-8")
        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        result = manager.restore_task(bad)
        assert result is False

    def test_restore_task_restores_approved_and_queries(self, tmp_path):
        task_path = tmp_path / "task.json"
        state = {
            "version": 1,
            "topic": "AI Research",
            "title": "AI Research",
            "user_prompt": None,
            "current_research_depth": 1,
            "approved": [{"subtopic": "NLP", "query": "nlp", "summary": "Natural language processing overview.", "source_urls": []}],
            "successful_queries": ["nlp"],
            "failed_queries": ["bad"],
            "consecutive_failures": 2,
            "status": "in_progress",
            "last_saved_iso": "2026-01-01T00:00:00+00:00",
        }
        task_path.write_text(json.dumps(state), encoding="utf-8")

        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        result = manager.restore_task(task_path)

        assert result is True
        assert manager._approved == state["approved"]
        assert manager._successful_queries == ["nlp"]
        assert manager._failed_queries == ["bad"]
        assert manager._consecutive_failures == 2
        assert manager._current_research_depth == 1

    def test_restore_task_restores_graph(self, tmp_path):
        from src.topic_graph import TopicGraph

        # Build and save a graph
        task_path = tmp_path / "task.json"
        g = TopicGraph(root_name="AI", root_query="AI")
        child = g.add_node(name="NLP", query="nlp", parent_id=g.root.id)
        g.mark_leaf(child.id)
        g.mark_researched(child.id, "NLP summary", ["https://example.com"])

        state = {
            "version": 1,
            "topic": "AI",
            "title": "AI",
            "user_prompt": None,
            "current_research_depth": 1,
            "approved": [],
            "successful_queries": [],
            "failed_queries": [],
            "consecutive_failures": 0,
            "status": "in_progress",
            "last_saved_iso": "2026-01-01T00:00:00+00:00",
            "graph": g.to_dict(),
        }
        task_path.write_text(json.dumps(state), encoding="utf-8")

        manager = self._build_manager(
            "AI", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        result = manager.restore_task(task_path)

        assert result is True
        assert manager._graph is not None
        assert manager._graph.node_count() == 2
        restored_child = manager._graph.find_by_name("NLP")
        assert restored_child is not None
        assert restored_child.summary == "NLP summary"

    def test_restore_task_resets_researching_nodes_to_pending(self, tmp_path):
        """Nodes in 'researching' status must be reset so they are retried."""
        from src.topic_graph import TopicGraph

        task_path = tmp_path / "task.json"
        g = TopicGraph(root_name="AI", root_query="AI")
        child = g.add_node(name="NLP", query="nlp", parent_id=g.root.id)
        g.mark_leaf(child.id)
        g.mark_researching(child.id)  # interrupted mid-research

        state = {
            "version": 1, "topic": "AI", "title": "AI", "user_prompt": None,
            "current_research_depth": 0, "approved": [], "successful_queries": [],
            "failed_queries": [], "consecutive_failures": 0,
            "status": "in_progress", "last_saved_iso": "2026-01-01T00:00:00+00:00",
            "graph": g.to_dict(),
        }
        task_path.write_text(json.dumps(state), encoding="utf-8")

        manager = self._build_manager(
            "AI", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        manager.restore_task(task_path)

        restored_child = manager._graph.find_by_name("NLP")
        assert restored_child.status == "pending"  # reset from 'researching'

    def test_restore_task_returns_true_for_completed_status(self, tmp_path):
        """Completed tasks are restored (graph available for report gen) but return True."""
        task_path = tmp_path / "task.json"
        state = {
            "version": 1, "topic": "AI", "title": "AI", "user_prompt": None,
            "current_research_depth": 0, "approved": [], "successful_queries": [],
            "failed_queries": [], "consecutive_failures": 0,
            "status": "completed", "last_saved_iso": "2026-01-01T00:00:00+00:00",
        }
        task_path.write_text(json.dumps(state), encoding="utf-8")
        manager = self._build_manager(
            "AI", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        result = manager.restore_task(task_path)
        assert result is True

    def test_save_restore_roundtrip(self, tmp_path):
        """save_task() followed by restore_task() preserves all state."""
        from src.topic_graph import TopicGraph

        task_path = tmp_path / "task.json"
        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
            task_json_path=task_path,
        )
        manager._graph = TopicGraph(root_name="AI Research", root_query="AI")
        manager._current_research_depth = 1
        manager._approved = [{"subtopic": "ML", "summary": "summary", "source_urls": []}]
        manager._successful_queries = ["ml query"]
        manager._failed_queries = ["bad query"]
        manager._consecutive_failures = 1
        manager.save_task()

        # Create fresh manager and restore
        manager2 = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        result = manager2.restore_task(task_path)

        assert result is True
        assert manager2._current_research_depth == 1
        assert manager2._approved[0]["subtopic"] == "ML"
        assert manager2._successful_queries == ["ml query"]
        assert manager2._failed_queries == ["bad query"]
        assert manager2._consecutive_failures == 1
        assert manager2._graph is not None


class TestInlineSourceLinks:
    """Tests for inline source reference links in generated reports."""

    def _build_manager(self, topic, reports_dir, db_path):
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            return AgentManager(topic=topic, reports_dir=reports_dir, db_path=db_path)

    def test_build_inline_refs_empty(self):
        from src.agent_manager import _build_inline_refs
        assert _build_inline_refs([]) == ""

    def test_build_inline_refs_single(self):
        from src.agent_manager import _build_inline_refs
        result = _build_inline_refs(["https://example.com"])
        assert "*Sources:" in result
        assert "[1](https://example.com)" in result

    def test_build_inline_refs_multiple(self):
        from src.agent_manager import _build_inline_refs
        result = _build_inline_refs(["https://a.com", "https://b.com"])
        assert "[1](https://a.com)" in result
        assert "[2](https://b.com)" in result

    def test_graph_report_includes_inline_source_links(self, tmp_path):
        """Reports generated from a graph include inline source links per section."""
        from src.topic_graph import TopicGraph
        from src.agent_manager import AgentManager

        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        g = TopicGraph(root_name="AI Research", root_query="AI")
        child = g.add_node(name="NLP", query="nlp", parent_id=g.root.id)
        g.mark_leaf(child.id)
        g.mark_researched(
            child.id,
            "Natural language processing overview.",
            ["https://nlp.example.com"],
        )
        manager._graph = g

        report_path = manager.generate_report()
        content = report_path.read_text(encoding="utf-8")

        assert "[1](https://nlp.example.com)" in content

    def test_flat_report_includes_inline_source_links(self, tmp_path):
        """Flat-mode reports (no graph) also include inline source links."""
        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        manager._approved = [
            {
                "subtopic": "Machine Learning",
                "query": "ML basics",
                "summary": "Overview of ML.",
                "source_urls": ["https://ml.example.com"],
            }
        ]

        report_path = manager.generate_report()
        content = report_path.read_text(encoding="utf-8")

        assert "[1](https://ml.example.com)" in content

    def test_report_no_inline_refs_when_no_sources(self, tmp_path):
        """Sections with no source URLs have no inline *Sources:* text."""
        from src.topic_graph import TopicGraph

        manager = self._build_manager(
            "AI Research", str(tmp_path / "reports"), str(tmp_path / "db"),
        )
        g = TopicGraph(root_name="AI Research", root_query="AI")
        child = g.add_node(name="NLP", query="nlp", parent_id=g.root.id)
        g.mark_leaf(child.id)
        g.mark_researched(child.id, "NLP summary.", [])  # no sources
        manager._graph = g

        report_path = manager.generate_report()
        content = report_path.read_text(encoding="utf-8")

        assert "*Sources:" not in content


class TestRestructureSkipsExistingNodes:
    """_apply_restructure_suggestions must not reset already-completed nodes."""

    def _build_manager(self, tmp_path):
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            return AgentManager(
                topic="Trading",
                reports_dir=str(tmp_path / "reports"),
                db_path=str(tmp_path / "db"),
            )

    def test_existing_node_not_reset_to_pending(self, tmp_path):
        """A restructure suggestion for a completed node must be skipped — the
        node's status must not be reverted to 'pending'."""
        from src.topic_graph import TopicGraph

        manager = self._build_manager(tmp_path)
        g = TopicGraph(root_name="Trading", root_query="trading")
        child = g.add_node(name="RSI Strategy", query="rsi strategy", parent_id=g.root.id)
        g.mark_leaf(child.id)
        g.mark_researched(child.id, "RSI overview.", [])
        manager._graph = g

        # Planner suggests re-adding the same node
        suggestions = [
            {"action": "add", "name": "RSI Strategy", "query": "rsi q", "parent_name": "Trading", "priority": 5},
        ]
        manager._apply_restructure_suggestions(suggestions)

        # Node must remain completed, not be reverted to pending
        node = g.find_by_name("RSI Strategy")
        assert node is not None
        assert node.status == "completed"

    def test_new_node_is_added_normally(self, tmp_path):
        """Suggestions for genuinely new node names are still applied."""
        from src.topic_graph import TopicGraph

        manager = self._build_manager(tmp_path)
        g = TopicGraph(root_name="Trading", root_query="trading")
        manager._graph = g

        suggestions = [
            {"action": "add", "name": "MACD Divergence", "query": "macd divergence", "parent_name": "Trading", "priority": 5},
        ]
        manager._apply_restructure_suggestions(suggestions)

        node = g.find_by_name("MACD Divergence")
        assert node is not None
        assert node.is_leaf is True
        assert node.status == "pending"

    def test_mixed_suggestions_only_new_added(self, tmp_path):
        """Mixed batch: existing node skipped, new node added."""
        from src.topic_graph import TopicGraph

        manager = self._build_manager(tmp_path)
        g = TopicGraph(root_name="Trading", root_query="trading")
        existing = g.add_node(name="Bollinger Bands", query="bollinger", parent_id=g.root.id)
        g.mark_leaf(existing.id)
        g.mark_researched(existing.id, "BB summary.", [])
        manager._graph = g

        suggestions = [
            {"action": "add", "name": "Bollinger Bands", "query": "bb2", "parent_name": "Trading", "priority": 5},
            {"action": "add", "name": "VWAP Strategy", "query": "vwap", "parent_name": "Trading", "priority": 5},
        ]
        manager._apply_restructure_suggestions(suggestions)

        # Bollinger Bands must remain completed
        bb = g.find_by_name("Bollinger Bands")
        assert bb.status == "completed"

        # VWAP was genuinely new and must have been added as pending leaf
        vwap = g.find_by_name("VWAP Strategy")
        assert vwap is not None
        assert vwap.status == "pending"
        assert vwap.is_leaf is True


class TestOrphanPendingLeafRescue:
    """run_graph step 4: pending leaf nodes at non-current depth are rescued."""

    def _build_manager(self, tmp_path):
        from src.agent_manager import AgentManager

        with patch("src.agent_manager.PlannerAgent"), \
             patch("src.agent_manager.ResearcherAgent"), \
             patch("src.agent_manager.CriticAgent"), \
             patch("src.agent_manager.KnowledgeBase"):
            return AgentManager(
                topic="Trading",
                reports_dir=str(tmp_path / "reports"),
                db_path=str(tmp_path / "db"),
            )

    def test_orphan_leaf_at_lower_depth_is_researched(self, tmp_path, event_loop):
        """A pending leaf stranded at depth 1 while current_depth=2 must not be skipped."""
        from src.topic_graph import TopicGraph

        manager = self._build_manager(tmp_path)
        g = TopicGraph(root_name="Trading", root_query="trading")

        # Depth-1 orphan leaf — pending, will be missed by BFS depth=2 scan
        orphan = g.add_node(name="Orphan Leaf", query="orphan query", parent_id=g.root.id)
        g.mark_leaf(orphan.id)

        # Depth-1 non-leaf with one depth-2 child, both already done
        # so: no pending leaves at depth 2, no consolidation candidates
        depth1_parent = g.add_node(name="Parent Node", query="parent", parent_id=g.root.id)
        depth2_child = g.add_node(name="Deep Child", query="deep", parent_id=depth1_parent.id)
        g.mark_leaf(depth2_child.id)
        g.mark_researched(depth2_child.id, "deep summary", [])
        g.mark_consolidated(depth1_parent.id, "consolidated parent")

        manager._graph = g
        manager._current_research_depth = 2  # loop advanced past orphan's depth

        # Stub _research_node to record the node it was called with
        researched_nodes = []

        async def fake_research(node):
            researched_nodes.append(node.name)
            return None

        manager._research_node = fake_research

        event_loop.run_until_complete(manager.run_graph())

        assert "Orphan Leaf" in researched_nodes
