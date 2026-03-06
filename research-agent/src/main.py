"""
main.py – Entry point for the Autonomous Research Agent.

Usage:
    python -m src.main --topic "Stock Trading Strategies" --hours 8

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
import sys
import time
from pathlib import Path

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent – runs for a set duration."
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="High-level research topic (e.g. 'Stock Trading Strategies').",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=_DEFAULT_HOURS,
        help=f"How many hours to run (default: {_DEFAULT_HOURS}).",
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
    duration = args.hours * 3600

    # Ensure report and data directories exist relative to CWD
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)
    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        run(
            topic=args.topic,
            duration_seconds=duration,
            model=args.model,
            ollama_base_url=args.ollama_url,
            reports_dir=args.reports_dir,
            db_path=args.db_path,
        )
    )


if __name__ == "__main__":
    main()
