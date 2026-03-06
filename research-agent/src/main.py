"""
main.py – Entry point for the Autonomous Research Agent.

Usage:
    python -m src.main --topic "Stock Trading Strategies" --hours 8
    python -m src.main --topic "Stock Trading Strategies" --duration 10m
    python -m src.main --requirements-file requirements.md --duration 1h30m

The ``--requirements-file`` option accepts a plain-text or Markdown file that
contains the full research specification, including research details and output
expectations.  This is the recommended approach for complex, multi-section
requirements that are too long to fit comfortably on the command line.

The controller implements a robust asyncio event loop that:
- Runs for the specified duration (default 8 hours).
- Handles all exceptions gracefully (log → sleep → retry).
- Applies randomised rate-limiting between search requests.
- Saves incremental Markdown reports throughout the run.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

from src.agent_manager import AgentManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

_DEFAULT_HOURS = 8
_ERROR_SLEEP_SECONDS = 5
_CYCLE_SLEEP_MIN = 2.0
_CYCLE_SLEEP_MAX = 5.0
_QUEUE_REFRESH_EVERY = 10  # refill queue every N approved findings


def _parse_duration(value: str) -> float:
    """Parse a human-readable duration string and return total seconds.

    Supported formats (case-insensitive):
        ``30s``       → 30 seconds
        ``10m``       → 10 minutes
        ``1h``        → 1 hour
        ``1h30m``     → 1 hour 30 minutes
        ``1h30m45s``  → 1 hour 30 minutes 45 seconds
        ``90min``     → 90 minutes
        ``2hrs``      → 2 hours

    Raises:
        argparse.ArgumentTypeError: if the string cannot be parsed.
    """
    pattern = re.compile(
        r"^(?:(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h))?"
        r"(?:(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m))?"
        r"(?:(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s))?$",
        re.IGNORECASE,
    )
    match = pattern.match(value.strip())
    if not match or not any(match.groups()):
        raise argparse.ArgumentTypeError(
            f"Invalid duration {value!r}. "
            "Use formats like '10m', '1h', '1h30m', '90s'."
        )
    hours = float(match.group(1) or 0)
    minutes = float(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise argparse.ArgumentTypeError(
            f"Duration must be greater than zero, got {value!r}."
        )
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent – runs for a set duration."
    )
    topic_group = parser.add_mutually_exclusive_group(required=True)
    topic_group.add_argument(
        "--topic",
        type=str,
        default=None,
        help="High-level research topic as inline text "
             "(e.g. 'Stock Trading Strategies'). "
             "Use --requirements-file for longer specifications.",
    )
    topic_group.add_argument(
        "--requirements-file",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a plain-text or Markdown file containing the full "
             "research specification (research details and output expectations). "
             "The file name stem is used as the report title.",
    )
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Number of hours to run. If neither --hours nor --duration is specified, "
             f"defaults to {_DEFAULT_HOURS} hours. Cannot be used together with --duration.",
    )
    duration_group.add_argument(
        "--duration",
        type=_parse_duration,
        default=None,
        metavar="DURATION",
        help="How long to run, as a human-readable string "
             "(e.g. '10m', '1h', '1h30m', '90s'). "
             "Cannot be used together with --hours.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Ollama model name (default: llama3).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="data/reports",
        help="Directory for Markdown reports (default: data/reports).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/research.db",
        help="SQLite database path (default: data/research.db).",
    )
    return parser.parse_args()


async def run(
    topic: str,
    duration_seconds: float,
    title: Optional[str] = None,
    model: str = "llama3",
    ollama_base_url: str = "http://localhost:11434",
    reports_dir: str = "data/reports",
    db_path: str = "data/research.db",
) -> None:
    """Main async loop – runs for *duration_seconds* seconds."""
    start_time = time.monotonic()
    end_time = start_time + duration_seconds
    cycles_completed = 0
    approved_count = 0

    manager = AgentManager(
        topic=topic,
        title=title,
        model=model,
        ollama_base_url=ollama_base_url,
        reports_dir=reports_dir,
        db_path=db_path,
    )

    await manager.init()
    logger.info(
        "Research session started. Topic: %r  Duration: %.1f h",
        topic,
        duration_seconds / 3600,
    )

    try:
        # Seed the initial task queue
        await manager.populate_queue()

        while time.monotonic() < end_time:
            # Refresh the queue if it runs dry
            if not manager.has_tasks():
                logger.info("Task queue empty – requesting new tasks from Planner.")
                try:
                    await manager.populate_queue()
                except Exception as exc:
                    logger.error("Planner populate failed: %s", exc)
                    await asyncio.sleep(_ERROR_SLEEP_SECONDS)
                    continue

            # Run one research cycle
            try:
                finding = await manager.run_cycle()
                if finding:
                    approved_count += 1
                    logger.info(
                        "Approved finding #%d: %r", approved_count, finding["subtopic"]
                    )

                    # Periodically refresh the queue
                    if approved_count % _QUEUE_REFRESH_EVERY == 0:
                        await manager.populate_queue()

            except Exception as exc:
                logger.error(
                    "Cycle error (will retry in %ds): %s", _ERROR_SLEEP_SECONDS, exc,
                    exc_info=True,
                )
                await asyncio.sleep(_ERROR_SLEEP_SECONDS)
                continue

            cycles_completed += 1

            # Rate limiting – randomised sleep between cycles
            sleep_time = random.uniform(_CYCLE_SLEEP_MIN, _CYCLE_SLEEP_MAX)
            remaining = end_time - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(sleep_time, remaining))

    finally:
        # Always generate the final report
        report_path = manager.generate_report()
        await manager.close()
        elapsed = time.monotonic() - start_time
        logger.info(
            "Research session complete. Cycles: %d  Approved findings: %d  "
            "Elapsed: %.1f min  Report: %s",
            cycles_completed,
            approved_count,
            elapsed / 60,
            report_path,
        )


def main() -> None:
    args = _parse_args()
    if args.duration is not None:
        duration = args.duration
    elif args.hours is not None:
        duration = args.hours * 3600
    else:
        duration = _DEFAULT_HOURS * 3600

    # Resolve topic and title from --topic or --requirements-file
    if args.requirements_file is not None:
        req_path = Path(args.requirements_file)
        if not req_path.is_file():
            sys.exit(f"Requirements file not found: {args.requirements_file!r}")
        topic = req_path.read_text(encoding="utf-8").strip()
        report_title: Optional[str] = req_path.stem
    else:
        topic = args.topic
        report_title = None

    # Ensure report and data directories exist relative to CWD
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)
    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        run(
            topic=topic,
            title=report_title,
            duration_seconds=duration,
            model=args.model,
            ollama_base_url=args.ollama_url,
            reports_dir=args.reports_dir,
            db_path=args.db_path,
        )
    )


if __name__ == "__main__":
    main()
