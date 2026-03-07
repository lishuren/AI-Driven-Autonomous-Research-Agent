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

    def test_decompose_with_good_and_bad_examples(self, event_loop):
        """decompose() accepts good/bad example queries and still returns tasks."""
        from src.agents.planner import PlannerAgent

        llm_output = json.dumps([
            {"subtopic": "RSI", "query": "RSI indicator python"},
        ])

        with patch("src.agents.planner._call_ollama", return_value=llm_output):
            agent = PlannerAgent()
            tasks = event_loop.run_until_complete(
                agent.decompose(
                    "Stock Trading",
                    good_examples=["RSI formula pandas"],
                    bad_examples=["vague market sentiment"],
                )
            )

        assert len(tasks) == 1
        assert tasks[0]["subtopic"] == "RSI"

    def test_decompose_retrospective_returns_tasks(self, event_loop):
        """decompose_retrospective() uses the retrospective prompt and returns tasks."""
        from src.agents.planner import PlannerAgent

        llm_output = json.dumps([
            {"subtopic": "Bollinger alternative", "query": "bollinger bands width formula"},
        ])

        with patch("src.agents.planner._call_ollama", return_value=llm_output):
            agent = PlannerAgent()
            tasks = event_loop.run_until_complete(
                agent.decompose_retrospective(
                    "Bollinger Bands",
                    failed_queries=["bollinger bands", "bollinger trading signal"],
                )
            )

        assert len(tasks) == 1
        assert "subtopic" in tasks[0]

    def test_decompose_retrospective_falls_back_on_bad_response(self, event_loop):
        """decompose_retrospective() falls back to 5 tasks on unparsable LLM output."""
        from src.agents.planner import PlannerAgent

        with patch("src.agents.planner._call_ollama", return_value="not json"):
            agent = PlannerAgent()
            tasks = event_loop.run_until_complete(
                agent.decompose_retrospective("Topic", failed_queries=["q1"])
            )

        assert len(tasks) == 5

    def test_pre_search_vocab_returns_words(self, event_loop):
        """_pre_search_vocab() extracts real words from search result titles/snippets."""
        from src.agents.planner import PlannerAgent
        from unittest.mock import AsyncMock

        mock_search_tool = MagicMock()
        mock_search_tool.search = AsyncMock(return_value=[
            {"title": "RSI Indicator Python Tutorial", "body": "Calculate RSI using pandas"},
        ])

        agent = PlannerAgent(search_tool=mock_search_tool)
        vocab = event_loop.run_until_complete(agent._pre_search_vocab("RSI indicator"))

        assert isinstance(vocab, list)
        assert len(vocab) > 0
        # common words from the title/body should appear
        assert any(w in vocab for w in ["rsi", "indicator", "python", "pandas", "calculate"])

    def test_pre_search_vocab_returns_empty_on_error(self, event_loop):
        """_pre_search_vocab() returns [] when the search tool raises an exception."""
        from src.agents.planner import PlannerAgent
        from unittest.mock import AsyncMock

        mock_search_tool = MagicMock()
        mock_search_tool.search = AsyncMock(side_effect=RuntimeError("search failed"))

        agent = PlannerAgent(search_tool=mock_search_tool)
        vocab = event_loop.run_until_complete(agent._pre_search_vocab("RSI"))

        assert vocab == []

    def test_pre_search_vocab_returns_empty_without_tool(self, event_loop):
        """_pre_search_vocab() returns [] when no search_tool is provided."""
        from src.agents.planner import PlannerAgent

        agent = PlannerAgent()
        vocab = event_loop.run_until_complete(agent._pre_search_vocab("RSI"))

        assert vocab == []


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
            instance._user_prompt = None

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
        instance._user_prompt = None

        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value=[])
        instance._search = mock_search
        instance._scraper = MagicMock()

        result = event_loop.run_until_complete(
            instance.research({"subtopic": "RSI", "query": "RSI formula"})
        )

        assert result["summary"] == ""
        assert result["source_urls"] == []
