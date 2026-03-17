"""
Tests for main.py – specifically the _parse_duration helper and
the argument-parsing logic around --hours / --duration / --requirements-file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.main import _parse_duration, _parse_requirements_file


class TestParseDuration:
    """Unit tests for _parse_duration."""

    # --- valid inputs ---

    def test_seconds_only(self):
        assert _parse_duration("30s") == 30.0

    def test_seconds_long_form(self):
        assert _parse_duration("45sec") == 45.0

    def test_minutes_only(self):
        assert _parse_duration("10m") == 600.0

    def test_minutes_long_form(self):
        assert _parse_duration("10min") == 600.0

    def test_minutes_plural(self):
        assert _parse_duration("90mins") == 5400.0

    def test_hours_only(self):
        assert _parse_duration("1h") == 3600.0

    def test_hours_long_form(self):
        assert _parse_duration("2hrs") == 7200.0

    def test_hours_and_minutes(self):
        assert _parse_duration("1h30m") == 5400.0

    def test_hours_minutes_seconds(self):
        assert _parse_duration("1h30m45s") == pytest.approx(5445.0)

    def test_case_insensitive(self):
        assert _parse_duration("1H30M") == 5400.0

    def test_fractional_hours(self):
        assert _parse_duration("0.5h") == 1800.0

    def test_fractional_minutes(self):
        assert _parse_duration("1.5m") == pytest.approx(90.0)

    def test_whitespace_stripped(self):
        assert _parse_duration("  10m  ") == 600.0

    # --- invalid inputs ---

    def test_empty_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_duration("")

    def test_zero_duration_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="greater than zero"):
            _parse_duration("0m")

    def test_invalid_unit_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_duration("5x")

    def test_plain_number_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_duration("10")


class TestArgParseDuration:
    """Tests for the CLI argument parsing of --hours / --duration."""

    def _parse(self, extra_args: list[str]) -> argparse.Namespace:
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "Test"] + extra_args):
            return _parse_args()

    def test_default_duration_is_none(self):
        args = self._parse([])
        assert args.hours is None
        assert args.duration is None

    def test_hours_flag(self):
        args = self._parse(["--hours", "2"])
        assert args.hours == 2.0
        assert args.duration is None

    def test_duration_flag_minutes(self):
        args = self._parse(["--duration", "10m"])
        assert args.duration == 600.0
        assert args.hours is None

    def test_duration_flag_hours(self):
        args = self._parse(["--duration", "1h"])
        assert args.duration == 3600.0

    def test_duration_flag_combined(self):
        args = self._parse(["--duration", "1h30m"])
        assert args.duration == 5400.0

    def test_hours_and_duration_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse(["--hours", "1", "--duration", "10m"])

    def test_no_topic_or_requirements_file_raises(self):
        from src.main import _parse_args

        with patch("sys.argv", ["prog"]):
            with pytest.raises(SystemExit):
                _parse_args()

    def test_topic_and_requirements_file_mutually_exclusive(self, tmp_path):
        req_file = tmp_path / "spec.md"
        req_file.write_text("research detail", encoding="utf-8")
        from src.main import _parse_args

        with patch(
            "sys.argv",
            ["prog", "--topic", "Test", "--requirements-file", str(req_file)],
        ):
            with pytest.raises(SystemExit):
                _parse_args()


class TestRequirementsFile:
    """Tests for --requirements-file argument parsing and loading."""

    def test_requirements_file_stored_in_namespace(self, tmp_path):
        req_file = tmp_path / "my_spec.md"
        req_file.write_text("research spec content", encoding="utf-8")
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--requirements-file", str(req_file)]):
            args = _parse_args()

        assert args.requirements_file == str(req_file)
        assert args.topic is None

    def test_parse_requirements_file_ignores_prompt_section(self, tmp_path):
        req_file = tmp_path / "spec.md"
        req_file.write_text(
            "## Topic\nRSI trading\n\n"
            "## Prompt\nThis should be ignored.\n\n"
            "## Output Expectations\nNeed formulas and examples.\n",
            encoding="utf-8",
        )

        topic, title, user_prompt = _parse_requirements_file(req_file)

        assert topic == "RSI trading"
        assert title == "spec"
        assert user_prompt == "Need formulas and examples."

    def test_parse_requirements_file_ignores_prompt_and_keeps_other_context(self, tmp_path):
        req_file = tmp_path / "spec.md"
        req_file.write_text(
            "## Topic\nRSI trading\n\n"
            "## Prompt\nIgnore this prompt block.\n\n"
            "## Research Detail\nInvestigate the formula.\n\n"
            "## Output Expectations\nNeed examples.\n",
            encoding="utf-8",
        )

        topic, title, user_prompt = _parse_requirements_file(req_file)

        assert topic == "RSI trading"
        assert title == "spec"
        assert user_prompt == "Investigate the formula.\n\nNeed examples."


class TestMaxDepthArg:
    """Tests for --max-depth CLI argument."""

    def _parse(self, extra_args: list[str]) -> argparse.Namespace:
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "Test"] + extra_args):
            return _parse_args()

    def test_default_max_depth(self):
        args = self._parse([])
        assert args.max_depth == 3

    def test_custom_max_depth(self):
        args = self._parse(["--max-depth", "5"])
        assert args.max_depth == 5


class TestBudgetArgs:
    """Tests for --max-queries, --max-nodes, --max-credits-spend CLI args."""

    def _parse(self, extra_args: list[str]) -> argparse.Namespace:
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "Test"] + extra_args):
            return _parse_args()

    def test_max_queries_default_none(self):
        args = self._parse([])
        assert args.max_queries is None

    def test_max_queries_set(self):
        args = self._parse(["--max-queries", "50"])
        assert args.max_queries == 50

    def test_max_nodes_default_none(self):
        args = self._parse([])
        assert args.max_nodes is None

    def test_max_nodes_set(self):
        args = self._parse(["--max-nodes", "100"])
        assert args.max_nodes == 100

    def test_max_credits_spend_default_none(self):
        args = self._parse([])
        assert args.max_credits_spend is None

    def test_max_credits_spend_set(self):
        args = self._parse(["--max-credits-spend", "5.5"])
        assert args.max_credits_spend == 5.5


class TestScrapingArgs:
    """Tests for the --respect-robots CLI arg."""

    def _parse(self, extra_args: list[str]) -> argparse.Namespace:
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "Test"] + extra_args):
            return _parse_args()

    def test_respect_robots_default_false(self):
        args = self._parse([])
        assert args.respect_robots is False

    def test_respect_robots_flag(self):
        args = self._parse(["--respect-robots"])
        assert args.respect_robots is True


class TestBudgetEnvFallback:
    """Tests that budget CLI args fall back to env vars when not specified."""

    def test_max_queries_env_fallback(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_MAX_QUERIES", "42")
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "Test"]):
            args = _parse_args()

        # CLI arg is None → env var should be used in main()
        assert args.max_queries is None
        # Verify the env helper works
        val = os.environ.get("RESEARCH_MAX_QUERIES", "").strip()
        assert int(val) == 42

    def test_max_queries_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_MAX_QUERIES", "42")
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "Test", "--max-queries", "10"]):
            args = _parse_args()

        assert args.max_queries == 10

    def test_max_credits_env_fallback(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_MAX_CREDITS", "3.5")
        val = os.environ.get("RESEARCH_MAX_CREDITS", "").strip()
        assert float(val) == 3.5

    def test_respect_robots_env_bool_parsing(self, monkeypatch):
        """Verify the _bool_env logic for RESEARCH_RESPECT_ROBOTS."""
        for true_val in ("1", "true", "yes", "True", "YES"):
            monkeypatch.setenv("RESEARCH_RESPECT_ROBOTS", true_val)
            raw = os.environ.get("RESEARCH_RESPECT_ROBOTS", "").strip().lower()
            assert raw in ("1", "true", "yes")

        for false_val in ("0", "false", "no", "False", "NO"):
            monkeypatch.setenv("RESEARCH_RESPECT_ROBOTS", false_val)
            raw = os.environ.get("RESEARCH_RESPECT_ROBOTS", "").strip().lower()
            assert raw in ("0", "false", "no")
    """Tests for --tavily-key CLI argument."""

    def _parse(self, extra_args: list[str]) -> argparse.Namespace:
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "Test"] + extra_args):
            return _parse_args()

    def test_tavily_key_stored(self):
        args = self._parse(["--tavily-key", "tvly-test123"])
        assert args.tavily_key == "tvly-test123"

    def test_tavily_key_default_none(self):
        args = self._parse([])
        assert args.tavily_key is None

    def test_topic_stored_in_namespace(self):
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "RSI Strategy"]):
            args = _parse_args()

        assert args.topic == "RSI Strategy"
        assert args.requirements_file is None

    def test_main_reads_requirements_file(self, tmp_path, monkeypatch):
        """main() should read the file and pass its content as the topic."""
        req_file = tmp_path / "spec.md"
        # No section markers — backward compatible: entire file is topic
        spec_content = "Analyse RSI momentum indicators for trading."
        req_file.write_text(spec_content, encoding="utf-8")

        captured: dict = {}

        async def fake_run(topic, duration_seconds, title=None, user_prompt=None, **kwargs):
            captured["topic"] = topic
            captured["title"] = title
            captured["user_prompt"] = user_prompt

        monkeypatch.setattr("src.main.run", fake_run)
        monkeypatch.setattr("src.main.Path.mkdir", lambda *a, **kw: None)

        with patch(
            "sys.argv",
            ["prog", "--requirements-file", str(req_file), "--duration", "1s"],
        ):
            from src.main import main
            main()

        assert captured["topic"] == spec_content
        assert captured["title"] == "spec"
        assert captured["user_prompt"] is None

    def test_main_requirements_file_heading_becomes_topic(self, tmp_path, monkeypatch):
        """When a requirements file starts with a # heading and has no section markers,
        the heading text becomes the topic and the full content becomes the user_prompt."""
        req_file = tmp_path / "market-analysis.md"
        heading = "Chinese Online TRPG Market Analysis"
        spec_content = f"# {heading}\n\nDetailed analysis content goes here.\n\nMore context."
        req_file.write_text(spec_content, encoding="utf-8")

        captured: dict = {}

        async def fake_run(topic, duration_seconds, title=None, user_prompt=None, **kwargs):
            captured["topic"] = topic
            captured["title"] = title
            captured["user_prompt"] = user_prompt

        monkeypatch.setattr("src.main.run", fake_run)
        monkeypatch.setattr("src.main.Path.mkdir", lambda *a, **kw: None)

        with patch(
            "sys.argv",
            ["prog", "--requirements-file", str(req_file), "--duration", "1s"],
        ):
            from src.main import main
            main()

        assert captured["topic"] == heading
        assert captured["title"] == "market-analysis"
        assert captured["user_prompt"] == spec_content

    def test_main_requirements_file_topic_section_passes_remaining_context(self, tmp_path, monkeypatch):
        req_file = tmp_path / "spec.md"
        req_file.write_text(
            "## Topic\nRSI trading\n\n"
            "## Research Detail\nInvestigate the formula.\n\n"
            "## Prompt\nOld inline prompt.\n",
            encoding="utf-8",
        )

        captured: dict = {}

        async def fake_run(topic, duration_seconds, title=None, user_prompt=None, **kwargs):
            captured["topic"] = topic
            captured["title"] = title
            captured["user_prompt"] = user_prompt

        monkeypatch.setattr("src.main.run", fake_run)
        monkeypatch.setattr("src.main.Path.mkdir", lambda *a, **kw: None)

        with patch(
            "sys.argv",
            ["prog", "--requirements-file", str(req_file), "--duration", "1s"],
        ):
            from src.main import main
            main()

        assert captured["topic"] == "RSI trading"
        assert captured["title"] == "spec"
        assert captured["user_prompt"] == "Investigate the formula."

    def test_main_missing_requirements_file_exits(self, tmp_path, monkeypatch):
        """main() should exit with an error if the file does not exist."""
        missing = tmp_path / "nonexistent.md"

        with patch(
            "sys.argv",
            ["prog", "--requirements-file", str(missing), "--duration", "1s"],
        ):
            with pytest.raises(SystemExit):
                from src.main import main
                main()


class TestMainDurationResolution:
    """Tests that main() resolves --duration / --hours / default correctly."""

    def _run_main_duration(self, extra_args: list[str]) -> float:
        """Return the duration_seconds passed to run()."""
        from src.main import _parse_args, _DEFAULT_HOURS

        with patch("sys.argv", ["prog", "--topic", "Test"] + extra_args):
            args = _parse_args()

        if args.duration is not None:
            return args.duration
        if args.hours is not None:
            return args.hours * 3600
        return _DEFAULT_HOURS * 3600

    def test_no_flags_uses_default(self):
        assert self._run_main_duration([]) == 8 * 3600

    def test_hours_flag_converts_to_seconds(self):
        assert self._run_main_duration(["--hours", "1"]) == 3600.0

    def test_duration_minutes(self):
        assert self._run_main_duration(["--duration", "10m"]) == 600.0

    def test_duration_seconds(self):
        assert self._run_main_duration(["--duration", "90s"]) == 90.0

    def test_duration_combined(self):
        assert self._run_main_duration(["--duration", "1h30m"]) == 5400.0


class TestProgressiveReportSave:
    """Tests that the run() loop saves the report after every approved finding."""

    def test_report_saved_after_each_finding(self, event_loop, tmp_path):
        """generate_report() should be called once per approved finding."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.main import run

        reports_dir = str(tmp_path / "reports")
        db_path = str(tmp_path / "research.db")

        # Build a minimal AgentManager mock that yields one finding then stops.
        mock_manager = MagicMock()
        mock_manager.init = AsyncMock()
        mock_manager.close = AsyncMock()
        mock_manager.build_graph = AsyncMock()
        mock_manager.has_graph_work = MagicMock(side_effect=[True, True, False])
        mock_manager.has_tasks = MagicMock(return_value=False)
        mock_manager.populate_queue = AsyncMock()

        finding = {"subtopic": "RL policy", "query": "policy gradient", "source_urls": []}
        # First call returns a finding; second returns None.
        mock_manager.run_graph = AsyncMock(side_effect=[finding, None])
        mock_manager.generate_report = MagicMock(
            return_value=tmp_path / "reports" / "test.md"
        )

        with patch("src.main.AgentManager", return_value=mock_manager), \
             patch("src.main._list_ollama_models", return_value=[]):
            event_loop.run_until_complete(
                run(
                    topic="Reinforcement Learning",
                    duration_seconds=2,
                    reports_dir=reports_dir,
                    db_path=db_path,
                )
            )

        # generate_report() called: initial + after graph outline + after finding + finally
        assert mock_manager.generate_report.call_count >= 3

    def test_report_saved_on_interrupt(self, event_loop, tmp_path):
        """generate_report() is called in the finally block even on KeyboardInterrupt."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.main import run

        reports_dir = str(tmp_path / "reports")
        db_path = str(tmp_path / "research.db")

        mock_manager = MagicMock()
        mock_manager.init = AsyncMock()
        mock_manager.close = AsyncMock()
        mock_manager.build_graph = AsyncMock(side_effect=KeyboardInterrupt)
        mock_manager.has_graph_work = MagicMock(return_value=False)
        mock_manager.has_tasks = MagicMock(return_value=False)
        mock_manager.populate_queue = AsyncMock()
        mock_manager.generate_report = MagicMock(
            return_value=tmp_path / "reports" / "test.md"
        )

        with patch("src.main.AgentManager", return_value=mock_manager), \
             patch("src.main._list_ollama_models", return_value=[]):
            with pytest.raises(KeyboardInterrupt):
                event_loop.run_until_complete(
                    run(
                        topic="Reinforcement Learning",
                        duration_seconds=60,
                        reports_dir=reports_dir,
                        db_path=db_path,
                    )
                )

        # generate_report called at start (placeholder) and in finally block
        assert mock_manager.generate_report.call_count >= 2


class TestDryRunArgs:
    """Tests for --dry-run, --estimate-credits, and --warn-credits CLI args."""

    def _parse(self, extra_args: list[str]) -> argparse.Namespace:
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "Test"] + extra_args):
            return _parse_args()

    def test_dry_run_default_false(self):
        args = self._parse([])
        assert args.dry_run is False

    def test_dry_run_flag(self):
        args = self._parse(["--dry-run"])
        assert args.dry_run is True

    def test_warn_credits_default(self):
        args = self._parse([])
        assert args.warn_credits == 0.80

    def test_warn_credits_custom(self):
        args = self._parse(["--warn-credits", "0.6"])
        assert args.warn_credits == pytest.approx(0.6)

    def test_warn_credits_disabled(self):
        """Setting warn-credits to 1.0 effectively disables warnings."""
        args = self._parse(["--warn-credits", "1.0"])
        assert args.warn_credits == pytest.approx(1.0)


class TestLLMArgs:
    def _parse(self, extra_args: list[str]) -> argparse.Namespace:
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "Test"] + extra_args):
            return _parse_args()

    def test_llm_provider_default_none(self):
        args = self._parse([])
        assert args.llm_provider is None

    def test_llm_provider_accepts_siliconflow(self):
        args = self._parse(["--llm-provider", "siliconflow"])
        assert args.llm_provider == "siliconflow"

    def test_llm_url_stored(self):
        args = self._parse(["--llm-url", "https://api.siliconflow.cn/v1"])
        assert args.llm_url == "https://api.siliconflow.cn/v1"

    def test_llm_api_key_stored(self):
        args = self._parse(["--llm-api-key", "sk-test"])
        assert args.llm_api_key == "sk-test"

    def test_prompt_dir_stored(self):
        args = self._parse(["--prompt-dir", "/tmp/prompts"])
        assert args.prompt_dir == "/tmp/prompts"

    def test_main_requires_api_key_for_online_provider(self):
        with patch("sys.argv", ["prog", "--topic", "Test", "--llm-provider", "siliconflow"]):
            from src.main import main
            with pytest.raises(SystemExit):
                main()

    def test_main_passes_online_llm_and_prompt_dir_to_run(self, monkeypatch):
        captured: dict = {}

        async def fake_run(topic, duration_seconds, title=None, user_prompt=None, **kwargs):
            captured["topic"] = topic
            captured["duration_seconds"] = duration_seconds
            captured["kwargs"] = kwargs

        monkeypatch.setattr("src.main.run", fake_run)
        monkeypatch.setattr("src.main.Path.mkdir", lambda *a, **kw: None)

        with patch(
            "sys.argv",
            [
                "prog",
                "--topic", "RSI Strategy",
                "--duration", "1s",
                "--llm-provider", "siliconflow",
                "--llm-url", "https://api.siliconflow.cn/v1",
                "--llm-api-key", "sk-test",
                "--prompt-dir", "/tmp/prompts",
            ],
        ):
            from src.main import main
            main()

        assert captured["topic"] == "RSI Strategy"
        assert captured["kwargs"]["ollama_base_url"] == "https://api.siliconflow.cn/v1"
        assert captured["kwargs"]["llm_provider"] == "openai"
        assert captured["kwargs"]["llm_api_key"] == "sk-test"
        assert captured["kwargs"]["prompt_dir"] == "/tmp/prompts"


class TestEstimateRun:
    """Tests for the estimate_run() dry-run estimation function."""

    def test_estimate_run_uses_dry_run_mode(self, event_loop, tmp_path):
        """estimate_run() activates dry-run mode so no Tavily calls are made."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.main import estimate_run

        calls: list[bool] = []

        def spy_set_dry_run(enabled: bool = True) -> None:
            calls.append(enabled)

        mock_manager = MagicMock()
        mock_manager.init = AsyncMock()
        mock_manager.close = AsyncMock()
        mock_manager.build_graph = AsyncMock()
        mock_manager._graph = MagicMock()
        mock_manager._graph._nodes = {}

        # Patch the module-level import target inside estimate_run()
        with patch("src.tools.search_tool.set_dry_run", spy_set_dry_run), \
             patch("src.main.AgentManager", return_value=mock_manager), \
             patch("src.main._list_ollama_models", return_value=[]):
            event_loop.run_until_complete(
                estimate_run(topic="Test topic", model="qwen2.5:7b")
            )

        assert True in calls  # set_dry_run(True) was called

    def test_estimate_run_counts_leaf_nodes(self, event_loop, tmp_path, capsys):
        """estimate_run() counts childless depth>0 nodes as leaves to research."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.main import estimate_run
        from src.topic_graph import TopicNode

        # Depth-1 nodes with no children — as produced by build_graph() in dry-run
        # (is_leaf is NOT set because they are never analyzed)
        leaf1 = TopicNode(id="n1", name="Sub A", query="Sub A", depth=1, is_leaf=False)
        leaf2 = TopicNode(id="n2", name="Sub B", query="Sub B", depth=1, is_leaf=False)
        root = TopicNode(id="n0", name="Root", query="Root", depth=0, is_leaf=False,
                         children_ids=["n1", "n2"])

        mock_manager = MagicMock()
        mock_manager.init = AsyncMock()
        mock_manager.close = AsyncMock()
        mock_manager.build_graph = AsyncMock()
        mock_manager._graph = MagicMock()
        mock_manager._graph._nodes = {root.id: root, leaf1.id: leaf1, leaf2.id: leaf2}
        mock_manager._graph.max_depth_present = MagicMock(return_value=1)
        mock_manager._graph.get_nodes_at_depth = MagicMock(
            side_effect=lambda d: [root] if d == 0 else [leaf1, leaf2]
        )

        with patch("src.tools.search_tool.set_dry_run"), \
             patch("src.main.AgentManager", return_value=mock_manager), \
             patch("src.main._list_ollama_models", return_value=[]):
            event_loop.run_until_complete(
                estimate_run(topic="Test topic", model="qwen2.5:7b")
            )

        out = capsys.readouterr().out
        assert "2" in out  # 2 leaf nodes mentioned somewhere in estimate output


class TestTopicDir:
    """Tests for --topic-dir argument and _parse_topic_dir helper."""

    def _parse(self, extra_args: list[str]) -> "argparse.Namespace":
        from src.main import _parse_args

        with patch("sys.argv", ["prog"] + extra_args):
            return _parse_args()

    # --- argument parsing ---

    def test_topic_dir_stored_in_namespace(self, tmp_path):
        folder = tmp_path / "my-research"
        folder.mkdir()
        args = self._parse(["--topic-dir", str(folder)])
        assert args.topic_dir == str(folder)

    def test_topic_dir_exclusive_with_topic(self, tmp_path):
        folder = tmp_path / "my-research"
        folder.mkdir()
        with pytest.raises(SystemExit):
            self._parse(["--topic", "Test", "--topic-dir", str(folder)])

    def test_topic_dir_exclusive_with_requirements_file(self, tmp_path):
        folder = tmp_path / "my-research"
        folder.mkdir()
        req = tmp_path / "spec.md"
        req.write_text("content")
        with pytest.raises(SystemExit):
            self._parse(["--requirements-file", str(req), "--topic-dir", str(folder)])

    # --- _parse_topic_dir ---

    def test_parse_topic_dir_uses_requirements_md(self, tmp_path):
        from src.main import _parse_topic_dir

        folder = tmp_path / "project"
        folder.mkdir()
        (folder / "requirements.md").write_text("## Topic\nAI Agents\n", encoding="utf-8")

        topic, title, user_prompt, prompt_dir, config_dir = _parse_topic_dir(folder)
        assert topic == "AI Agents"
        assert title == "project"
        assert prompt_dir is None
        assert config_dir is None

    def test_parse_topic_dir_uses_topic_md_over_other_md(self, tmp_path):
        from src.main import _parse_topic_dir

        folder = tmp_path / "project"
        folder.mkdir()
        (folder / "topic.md").write_text("## Topic\nML Research\n", encoding="utf-8")
        (folder / "other.md").write_text("## Topic\nUnrelated\n", encoding="utf-8")

        topic, title, user_prompt, prompt_dir, config_dir = _parse_topic_dir(folder)
        assert topic == "ML Research"

    def test_parse_topic_dir_falls_back_to_first_md(self, tmp_path):
        from src.main import _parse_topic_dir

        folder = tmp_path / "project"
        folder.mkdir()
        (folder / "analysis.md").write_text("## Topic\nMarket Research\n", encoding="utf-8")

        topic, title, user_prompt, prompt_dir, config_dir = _parse_topic_dir(folder)
        assert topic == "Market Research"

    def test_parse_topic_dir_falls_back_to_folder_name(self, tmp_path):
        from src.main import _parse_topic_dir

        folder = tmp_path / "my-research-topic"
        folder.mkdir()

        topic, title, user_prompt, prompt_dir, config_dir = _parse_topic_dir(folder)
        assert topic == "my-research-topic"
        assert title == "my-research-topic"
        assert user_prompt is None
        assert config_dir is None

    def test_parse_topic_dir_detects_prompts_subfolder(self, tmp_path):
        from src.main import _parse_topic_dir

        folder = tmp_path / "project"
        folder.mkdir()
        prompts_dir = folder / "prompts"
        prompts_dir.mkdir()
        (folder / "requirements.md").write_text("AI Research", encoding="utf-8")

        topic, title, user_prompt, prompt_dir, config_dir = _parse_topic_dir(folder)
        assert prompt_dir == str(prompts_dir)

    def test_parse_topic_dir_no_prompts_returns_none_prompt_dir(self, tmp_path):
        from src.main import _parse_topic_dir

        folder = tmp_path / "project"
        folder.mkdir()
        (folder / "requirements.md").write_text("AI Research", encoding="utf-8")

        topic, title, user_prompt, prompt_dir, config_dir = _parse_topic_dir(folder)
        assert prompt_dir is None

    def test_parse_topic_dir_detects_config_subfolder(self, tmp_path):
        from src.main import _parse_topic_dir

        folder = tmp_path / "project"
        folder.mkdir()
        config_sub = folder / "config"
        config_sub.mkdir()
        (folder / "requirements.md").write_text("AI Research", encoding="utf-8")

        topic, title, user_prompt, prompt_dir, config_dir = _parse_topic_dir(folder)
        assert config_dir == str(config_sub)

    def test_parse_topic_dir_no_config_returns_none_config_dir(self, tmp_path):
        from src.main import _parse_topic_dir

        folder = tmp_path / "project"
        folder.mkdir()
        (folder / "requirements.md").write_text("AI Research", encoding="utf-8")

        topic, title, user_prompt, prompt_dir, config_dir = _parse_topic_dir(folder)
        assert config_dir is None

    def test_parse_topic_dir_title_is_folder_name(self, tmp_path):
        from src.main import _parse_topic_dir

        folder = tmp_path / "stock-trading"
        folder.mkdir()
        (folder / "requirements.md").write_text(
            "## Topic\nStock Trading Strategies\n", encoding="utf-8"
        )

        topic, title, user_prompt, prompt_dir, config_dir = _parse_topic_dir(folder)
        assert title == "stock-trading"
        assert topic == "Stock Trading Strategies"

    # --- main() integration ---

    def test_main_topic_dir_missing_folder_exits(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with patch("sys.argv", ["prog", "--topic-dir", str(missing), "--duration", "1s"]):
            with pytest.raises(SystemExit):
                from src.main import main
                main()

    def test_main_topic_dir_creates_output_folder_and_passes_paths(
        self, tmp_path, monkeypatch
    ):
        """main() creates <folder>/output/ and passes derived paths to run()."""
        folder = tmp_path / "my-project"
        folder.mkdir()
        (folder / "requirements.md").write_text("AI Agents research", encoding="utf-8")

        captured: dict = {}

        async def fake_run(topic, duration_seconds, **kwargs):
            captured["topic"] = topic
            captured["reports_dir"] = kwargs.get("reports_dir")
            captured["db_path"] = kwargs.get("db_path")
            captured["task_json_path"] = kwargs.get("task_json_path")

        monkeypatch.setattr("src.main.run", fake_run)

        with patch("sys.argv", ["prog", "--topic-dir", str(folder), "--duration", "1s"]):
            from src.main import main
            main()

        expected_output = folder / "output"
        assert captured["reports_dir"] == str(expected_output / "reports")
        assert captured["db_path"] == str(expected_output / "research.db")
        assert captured["task_json_path"] == str(expected_output / "task.json")

    def test_main_topic_dir_uses_prompts_subfolder(self, tmp_path, monkeypatch):
        """When <folder>/prompts/ exists, main() passes it as prompt_dir."""
        folder = tmp_path / "project"
        folder.mkdir()
        (folder / "requirements.md").write_text("Research topic", encoding="utf-8")
        prompts_dir = folder / "prompts"
        prompts_dir.mkdir()

        captured: dict = {}

        async def fake_run(topic, duration_seconds, **kwargs):
            captured["prompt_dir"] = kwargs.get("prompt_dir")

        monkeypatch.setattr("src.main.run", fake_run)

        with patch("sys.argv", ["prog", "--topic-dir", str(folder), "--duration", "1s"]):
            from src.main import main
            main()

        assert captured["prompt_dir"] == str(prompts_dir)

    def test_main_topic_dir_prompt_dir_not_overridden_by_subfolder(
        self, tmp_path, monkeypatch
    ):
        """Explicit --prompt-dir takes precedence over <folder>/prompts/."""
        folder = tmp_path / "project"
        folder.mkdir()
        (folder / "requirements.md").write_text("Research topic", encoding="utf-8")
        (folder / "prompts").mkdir()  # subfolder exists but should be ignored

        captured: dict = {}

        async def fake_run(topic, duration_seconds, **kwargs):
            captured["prompt_dir"] = kwargs.get("prompt_dir")

        monkeypatch.setattr("src.main.run", fake_run)

        with patch(
            "sys.argv",
            [
                "prog",
                "--topic-dir", str(folder),
                "--prompt-dir", "/my/custom/prompts",
                "--duration", "1s",
            ],
        ):
            from src.main import main
            main()

        assert captured["prompt_dir"] == "/my/custom/prompts"


class TestTaskJsonPersistence:
    """Tests for task.json auto-save and restore via run()."""

    def test_run_passes_task_json_path_to_manager(self, event_loop, tmp_path):
        """run() wires task_json_path into AgentManager."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.main import run

        reports_dir = str(tmp_path / "reports")
        db_path = str(tmp_path / "research.db")
        task_file = str(tmp_path / "task.json")

        mock_manager = MagicMock()
        mock_manager.init = AsyncMock()
        mock_manager.close = AsyncMock()
        mock_manager.build_graph = AsyncMock()
        mock_manager.has_graph_work = MagicMock(return_value=False)
        mock_manager.has_tasks = MagicMock(return_value=False)
        mock_manager.populate_queue = AsyncMock()
        mock_manager.restore_task = MagicMock(return_value=False)
        mock_manager.generate_report = MagicMock(return_value=tmp_path / "r.md")
        mock_manager._graph = None
        mock_manager.budget = MagicMock()
        mock_manager.budget.is_exhausted = MagicMock(return_value=False)
        mock_manager.budget.summary = MagicMock(return_value="ok")

        captured_kwargs: dict = {}

        def spy_init(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_manager

        with patch("src.main.AgentManager", side_effect=spy_init), \
             patch("src.main._list_ollama_models", return_value=[]):
            event_loop.run_until_complete(
                run(
                    topic="AI Agents",
                    duration_seconds=1,
                    reports_dir=reports_dir,
                    db_path=db_path,
                    task_json_path=task_file,
                )
            )

        from pathlib import Path
        assert captured_kwargs.get("task_json_path") == Path(task_file)

    def test_run_restores_task_when_file_exists(self, event_loop, tmp_path):
        """run() calls restore_task() when task_json_path points to an existing file."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.main import run

        reports_dir = str(tmp_path / "reports")
        db_path = str(tmp_path / "research.db")
        task_file = tmp_path / "task.json"
        task_file.write_text('{"version": 1, "status": "in_progress"}', encoding="utf-8")

        mock_manager = MagicMock()
        mock_manager.init = AsyncMock()
        mock_manager.close = AsyncMock()
        mock_manager.build_graph = AsyncMock()
        mock_manager.has_graph_work = MagicMock(return_value=False)
        mock_manager.has_tasks = MagicMock(return_value=False)
        mock_manager.populate_queue = AsyncMock()
        mock_manager.restore_task = MagicMock(return_value=True)
        mock_manager.generate_report = MagicMock(return_value=tmp_path / "r.md")
        mock_manager._graph = None
        mock_manager.budget = MagicMock()
        mock_manager.budget.is_exhausted = MagicMock(return_value=False)
        mock_manager.budget.summary = MagicMock(return_value="ok")
        mock_manager.save_task = MagicMock()

        with patch("src.main.AgentManager", return_value=mock_manager), \
             patch("src.main._list_ollama_models", return_value=[]):
            event_loop.run_until_complete(
                run(
                    topic="AI Agents",
                    duration_seconds=1,
                    reports_dir=reports_dir,
                    db_path=db_path,
                    task_json_path=str(task_file),
                )
            )

        # restore_task() should have been called with the path
        mock_manager.restore_task.assert_called_once()
        # build_graph() should NOT have been called when restore succeeds
        mock_manager.build_graph.assert_not_called()

    def test_run_builds_graph_when_no_saved_state(self, event_loop, tmp_path):
        """run() calls build_graph() when task_json_path is absent."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.main import run

        reports_dir = str(tmp_path / "reports")
        db_path = str(tmp_path / "research.db")

        mock_manager = MagicMock()
        mock_manager.init = AsyncMock()
        mock_manager.close = AsyncMock()
        mock_manager.build_graph = AsyncMock()
        mock_manager.has_graph_work = MagicMock(return_value=False)
        mock_manager.has_tasks = MagicMock(return_value=False)
        mock_manager.populate_queue = AsyncMock()
        mock_manager.restore_task = MagicMock(return_value=False)
        mock_manager.generate_report = MagicMock(return_value=tmp_path / "r.md")
        mock_manager._graph = None
        mock_manager.budget = MagicMock()
        mock_manager.budget.is_exhausted = MagicMock(return_value=False)
        mock_manager.budget.summary = MagicMock(return_value="ok")
        mock_manager.save_task = MagicMock()

        with patch("src.main.AgentManager", return_value=mock_manager), \
             patch("src.main._list_ollama_models", return_value=[]):
            event_loop.run_until_complete(
                run(
                    topic="AI Agents",
                    duration_seconds=1,
                    reports_dir=reports_dir,
                    db_path=db_path,
                    task_json_path=None,
                )
            )

        # restore_task() should NOT be called (no task_json_path)
        mock_manager.restore_task.assert_not_called()
        mock_manager.build_graph.assert_called_once()
