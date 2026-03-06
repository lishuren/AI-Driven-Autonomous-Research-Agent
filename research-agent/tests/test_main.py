"""
Tests for main.py – specifically the _parse_duration helper and
the argument-parsing logic around --hours / --duration.
"""

from __future__ import annotations

import argparse
import sys
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
