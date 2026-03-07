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
import json
import logging
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from src.agent_manager import AgentManager
from src.tools.search_tool import SearchLogger

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
_OLLAMA_TAGS_TIMEOUT_SECONDS = 10


def _list_ollama_models(base_url: str) -> list[str]:
    """Return local Ollama model names from /api/tags."""
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/tags",
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=_OLLAMA_TAGS_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read())

    models = payload.get("models", [])
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for model in models:
        if isinstance(model, dict) and isinstance(model.get("name"), str):
            names.append(model["name"])
    return names


def _resolve_model_name(requested_model: str, available_models: list[str]) -> str:
    """Resolve a usable Ollama model name from available local models."""
    if requested_model in available_models:
        return requested_model

    # If no explicit tag was provided (e.g. `llama3`), try family match first.
    if ":" not in requested_model:
        for model_name in available_models:
            if model_name.split(":", 1)[0] == requested_model:
                return model_name

    if available_models:
        return available_models[0]
    return requested_model


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
        default="qwen2.5:7b",
        help="Requested Ollama model name (default: qwen2.5:7b). "
             "If unavailable locally, runtime falls back to an installed model.",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base data directory. If specified, overrides --reports-dir and --db-path "
             "defaults (e.g., --data-dir /custom/path sets reports to /custom/path/reports "
             "and db to /custom/path/research.db).",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="data/reports",
        help="Directory for Markdown reports (default: data/reports). "
             "Overridden by --data-dir if specified.",
    )
    parser.add_argument(
        "--search-log",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a JSONL file for logging every search query, result count, "
             "and result domains. Disabled by default. "
             "Example: --search-log data/search.jsonl",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/research.db",
        help="SQLite database path (default: data/research.db). "
             "Overridden by --data-dir if specified.",
    )
    return parser.parse_args()


def _parse_requirements_file(path: Path) -> tuple[str, str, Optional[str]]:
    """Parse a requirements file and extract topic, title, and optional user prompt.

    Supports section markers:
    - ``## Prompt`` — Text between this heading and the next ``##`` heading is
      extracted as the user prompt (injected into all LLM calls).
    - ``## Topic`` — Text after this heading becomes the research topic.
    - If no section markers are found, the entire file content is used as the
      topic (backward compatible).

    Returns ``(topic, title, user_prompt)``.
    """
    raw = path.read_text(encoding="utf-8").strip()
    title = path.stem

    # Try to extract sections based on ## headings
    import re as _re
    sections: dict[str, str] = {}
    current_section: Optional[str] = None
    current_lines: list[str] = []
    non_section_lines: list[str] = []

    for line in raw.split("\n"):
        heading_match = _re.match(r"^##\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = heading_match.group(1).strip().lower()
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)
        else:
            non_section_lines.append(line)

    # Save last section
    if current_section is not None:
        sections[current_section] = "\n".join(current_lines).strip()

    user_prompt: Optional[str] = sections.get("prompt") or None

    if "topic" in sections:
        topic = sections["topic"]
    elif sections:
        # Has section markers but no ## Topic — use non-section text + all
        # non-prompt sections as topic
        parts = ["\n".join(non_section_lines).strip()]
        for key, val in sections.items():
            if key != "prompt" and val:
                parts.append(val)
        topic = "\n\n".join(p for p in parts if p)
    else:
        # No section markers at all — backward compat: entire file is topic
        topic = raw

    if not topic:
        topic = raw

    return topic, title, user_prompt


async def run(
    topic: str,
    duration_seconds: float,
    title: Optional[str] = None,
    user_prompt: Optional[str] = None,
    model: str = "qwen2.5:7b",
    ollama_base_url: str = "http://localhost:11434",
    reports_dir: str = "data/reports",
    db_path: str = "data/research.db",
) -> None:
    """Main async loop – runs for *duration_seconds* seconds."""
    start_time = time.monotonic()
    end_time = start_time + duration_seconds
    cycles_completed = 0
    approved_count = 0

    resolved_model = model
    try:
        available_models = _list_ollama_models(ollama_base_url)
        if available_models:
            selected_model = _resolve_model_name(model, available_models)
            if selected_model != model:
                logger.warning(
                    "Requested Ollama model %r not found; using %r. "
                    "Pull it with: ollama pull %s",
                    model,
                    selected_model,
                    model,
                )
            resolved_model = selected_model
        else:
            logger.warning(
                "No local Ollama models found at %s. Pull one with: ollama pull %s",
                ollama_base_url,
                model,
            )
    except Exception as exc:
        logger.warning(
            "Could not query Ollama models at %s: %s. Continuing with model %r.",
            ollama_base_url,
            exc,
            model,
        )

    manager = AgentManager(
        topic=topic,
        title=title,
        user_prompt=user_prompt,
        model=resolved_model,
        ollama_base_url=ollama_base_url,
        reports_dir=reports_dir,
        db_path=db_path,
    )

    await manager.init()
    logger.info(
        "Research session started. Topic: %r  Duration: %.1f h  Model: %r",
        topic,
        duration_seconds / 3600,
        resolved_model,
    )

    try:
        # Write an initial placeholder report so the file exists immediately.
        manager.generate_report()

        # Build the hierarchical topic graph (recursive decomposition)
        logger.info("Building topic graph ...")
        try:
            await manager.build_graph()
        except Exception as exc:
            logger.error("Graph build failed: %s — falling back to flat mode.", exc)
            await manager.populate_queue()

        manager.generate_report()  # Update report with graph outline

        while time.monotonic() < end_time:
            # Prefer graph-based orchestration; fall back to flat queue
            if manager.has_graph_work():
                try:
                    finding = await manager.run_graph()
                except Exception as exc:
                    logger.error(
                        "Graph cycle error (will retry in %ds): %s",
                        _ERROR_SLEEP_SECONDS, exc, exc_info=True,
                    )
                    await asyncio.sleep(_ERROR_SLEEP_SECONDS)
                    continue
            elif manager.has_tasks():
                try:
                    finding = await manager.run_cycle()
                except Exception as exc:
                    logger.error(
                        "Cycle error (will retry in %ds): %s",
                        _ERROR_SLEEP_SECONDS, exc, exc_info=True,
                    )
                    await asyncio.sleep(_ERROR_SLEEP_SECONDS)
                    continue
            else:
                # Both graph and queue exhausted — we're done
                logger.info("All research work complete.")
                break

            if finding:
                approved_count += 1
                logger.info(
                    "Approved finding #%d: %r  (%.0f min remaining)",
                    approved_count,
                    finding["subtopic"],
                    (end_time - time.monotonic()) / 60,
                )

                # Progressive report save
                try:
                    manager.generate_report()
                except Exception as report_exc:
                    logger.warning("Progressive report save failed: %s", report_exc)

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

    # Resolve topic, title, and optional user prompt
    user_prompt: Optional[str] = None
    if args.requirements_file is not None:
        req_path = Path(args.requirements_file)
        if not req_path.is_file():
            sys.exit(f"Requirements file not found: {args.requirements_file!r}")
        topic, report_title, user_prompt = _parse_requirements_file(req_path)
    else:
        topic = args.topic
        report_title = None

    # If --data-dir is provided, override reports-dir and db-path defaults
    reports_dir = args.reports_dir
    db_path = args.db_path
    if args.data_dir is not None:
        reports_dir = str(Path(args.data_dir) / "reports")
        db_path = str(Path(args.data_dir) / "research.db")

    # Ensure report and data directories exist relative to CWD
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Enable optional search query log before the run starts
    search_log_path: Optional[str] = None
    if args.data_dir is not None:
        search_log_path = str(Path(args.data_dir) / "search.jsonl")
    if args.search_log is not None:
        search_log_path = args.search_log
    if search_log_path is not None:
        SearchLogger.enable(search_log_path)

    try:
        asyncio.run(
            run(
                topic=topic,
                title=report_title,
                user_prompt=user_prompt,
                duration_seconds=duration,
                model=args.model,
                ollama_base_url=args.ollama_url,
                reports_dir=reports_dir,
                db_path=db_path,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        SearchLogger.close()


if __name__ == "__main__":
    main()
