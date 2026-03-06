"""
Unit tests for agent modules (Planner, Researcher, Critic).
Ollama and network calls are fully mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class TestPlannerAgent:
    def test_decompose_parses_valid_llm_response(self, event_loop):
        from src.agents.planner import PlannerAgent

        llm_output = json.dumps([
            {"subtopic": "RSI", "query": "RSI formula python pandas"},
            {"subtopic": "MACD", "query": "MACD calculation numpy"},
        ])

        with patch("src.agents.planner._call_ollama", return_value=llm_output):
            agent = PlannerAgent()
            tasks = event_loop.run_until_complete(
                agent.decompose("Stock Trading Strategies")
            )

        assert len(tasks) == 2
        assert tasks[0]["subtopic"] == "RSI"
        assert "query" in tasks[0]

    def test_decompose_falls_back_on_bad_response(self, event_loop):
        from src.agents.planner import PlannerAgent

        with patch("src.agents.planner._call_ollama", return_value="not json"):
            agent = PlannerAgent()
            tasks = event_loop.run_until_complete(agent.decompose("Topic"))

        assert len(tasks) == 5  # fallback generates 5 tasks

    def test_decompose_falls_back_on_none_response(self, event_loop):
        from src.agents.planner import PlannerAgent

        with patch("src.agents.planner._call_ollama", return_value=None):
            agent = PlannerAgent()
            tasks = event_loop.run_until_complete(agent.decompose("Topic"))

        assert len(tasks) == 5

    def test_refine_returns_task(self, event_loop):
        from src.agents.planner import PlannerAgent

        llm_output = json.dumps(
            {"subtopic": "RSI (refined)", "query": "RSI Wilder smoothing formula python"}
        )

        with patch("src.agents.planner._call_ollama", return_value=llm_output):
            agent = PlannerAgent()
            task = event_loop.run_until_complete(
                agent.refine("RSI", "Missing Wilder smoothing formula")
            )

        assert "query" in task

    def test_refine_falls_back_on_bad_response(self, event_loop):
        from src.agents.planner import PlannerAgent

        with patch("src.agents.planner._call_ollama", return_value=None):
            agent = PlannerAgent()
            task = event_loop.run_until_complete(
                agent.refine("RSI", "Missing formula")
            )

        assert "query" in task
        assert "RSI" in task["query"]


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class TestCriticAgent:
    def test_review_proceed(self, event_loop):
        from src.agents.critic import CriticAgent

        verdict = json.dumps({
            "status": "PROCEED",
            "checks": {
                "logical_steps": True,
                "math_formulas": True,
                "python_libraries": True,
            },
            "missing": "",
        })

        with patch("src.agents.critic.CriticAgent._call_ollama", return_value=verdict):
            agent = CriticAgent()
            result = event_loop.run_until_complete(
                agent.review("RSI", "Step 1: import pandas. Formula: $RSI = 100 - \\frac{100}{1+RS}$")
            )

        assert result["status"] == "PROCEED"

    def test_review_reject(self, event_loop):
        from src.agents.critic import CriticAgent

        verdict = json.dumps({
            "status": "REJECT",
            "checks": {
                "logical_steps": True,
                "math_formulas": False,
                "python_libraries": False,
            },
            "missing": "Missing math formula and library list",
        })

        with patch("src.agents.critic.CriticAgent._call_ollama", return_value=verdict):
            agent = CriticAgent()
            result = event_loop.run_until_complete(
                agent.review("RSI", "RSI is a momentum indicator.")
            )

        assert result["status"] == "REJECT"
        assert "missing" in result

    def test_review_empty_summary_rejects(self, event_loop):
        from src.agents.critic import CriticAgent

        agent = CriticAgent()
        result = event_loop.run_until_complete(agent.review("RSI", ""))

        assert result["status"] == "REJECT"

    def test_review_heuristic_fallback(self, event_loop):
        from src.agents.critic import CriticAgent

        with patch("src.agents.critic.CriticAgent._call_ollama", return_value=None):
            agent = CriticAgent()
            # Provide a summary rich enough for heuristics to PROCEED
            summary = (
                "1. First import pandas\n"
                "2. Then calculate $RSI = 100 - 100/(1+RS)$\n"
                "from pandas import DataFrame"
            )
            result = event_loop.run_until_complete(agent.review("RSI", summary))

        assert result["status"] in ("PROCEED", "REJECT")  # heuristic result
        assert "checks" in result


# ---------------------------------------------------------------------------
# ResearcherAgent
# ---------------------------------------------------------------------------

class TestResearcherAgent:
    def _make_researcher(self):
        from src.agents.researcher import ResearcherAgent

        with patch("src.tools.search_tool.SearchTool"), \
             patch("src.tools.scraper_tool.ScraperTool"):
            return ResearcherAgent()

    def test_research_returns_dict(self, event_loop):
        from src.agents.researcher import ResearcherAgent
        from unittest.mock import AsyncMock

        with patch("src.tools.search_tool.SearchTool") as MockSearch, \
             patch("src.tools.scraper_tool.ScraperTool") as MockScraper:

            instance = ResearcherAgent.__new__(ResearcherAgent)
            instance.model = "llama3"
            instance.ollama_base_url = "http://localhost:11434"
            instance.max_search_results = 4

            mock_search = MagicMock()
            mock_search.search = AsyncMock(return_value=[
                {"url": "https://example.com", "body": "RSI content"}
            ])
            mock_scraper = MagicMock()
            mock_scraper.scrape = AsyncMock(return_value="scraped RSI content")
            instance._search = mock_search
            instance._scraper = mock_scraper

            with patch.object(instance, "_call_ollama", return_value="Technical summary"):
                result = event_loop.run_until_complete(
                    instance.research({"subtopic": "RSI", "query": "RSI formula python"})
                )

        assert "subtopic" in result
        assert "summary" in result
        assert "source_urls" in result

    def test_research_handles_empty_search(self, event_loop):
        from src.agents.researcher import ResearcherAgent
        from unittest.mock import AsyncMock

        instance = ResearcherAgent.__new__(ResearcherAgent)
        instance.model = "llama3"
        instance.ollama_base_url = "http://localhost:11434"
        instance.max_search_results = 4

        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=[])
        instance._search = mock_search
        instance._scraper = MagicMock()

        result = event_loop.run_until_complete(
            instance.research({"subtopic": "RSI", "query": "RSI formula"})
        )

        assert result["summary"] == ""
        assert result["source_urls"] == []
