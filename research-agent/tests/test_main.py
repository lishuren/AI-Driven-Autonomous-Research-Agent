"""
Tests for main.py – specifically the _parse_duration helper and
the argument-parsing logic around --hours / --duration / --requirements-file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.main import _parse_duration


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

    def test_topic_stored_in_namespace(self):
        from src.main import _parse_args

        with patch("sys.argv", ["prog", "--topic", "RSI Strategy"]):
            args = _parse_args()

        assert args.topic == "RSI Strategy"
        assert args.requirements_file is None

    def test_main_reads_requirements_file(self, tmp_path, monkeypatch):
        """main() should read the file and pass its content as the topic."""
        req_file = tmp_path / "spec.md"
        spec_content = "## Research\nAnalyse RSI.\n\n## Output\nPython code."
        req_file.write_text(spec_content, encoding="utf-8")

        captured: dict = {}

        async def fake_run(topic, duration_seconds, title=None, **kwargs):
            captured["topic"] = topic
            captured["title"] = title

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
        mock_manager.has_tasks = MagicMock(return_value=True)
        mock_manager.populate_queue = AsyncMock()

        finding = {"subtopic": "RL policy", "query": "policy gradient", "source_urls": []}
        # First call returns a finding; subsequent calls return None to idle out.
        mock_manager.run_cycle = AsyncMock(side_effect=[finding, None, None, None, None])
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

        # generate_report() called: once for the finding + once in the finally block
        assert mock_manager.generate_report.call_count >= 2

    def test_report_saved_on_interrupt(self, event_loop, tmp_path):
        """generate_report() is called in the finally block even on KeyboardInterrupt."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.main import run

        reports_dir = str(tmp_path / "reports")
        db_path = str(tmp_path / "research.db")

        mock_manager = MagicMock()
        mock_manager.init = AsyncMock()
        mock_manager.close = AsyncMock()
        mock_manager.has_tasks = MagicMock(return_value=True)
        mock_manager.populate_queue = AsyncMock()
        mock_manager.run_cycle = AsyncMock(side_effect=KeyboardInterrupt)
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

        mock_manager.generate_report.assert_called_once()
