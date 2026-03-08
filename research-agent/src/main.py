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
import os
import random
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional


def _load_dotenv(env_path: Optional[Path] = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (stdlib only).

    Lines starting with ``#`` and blank lines are ignored.  Existing
    environment variables are *not* overwritten, so shell exports take
    precedence.  The file is looked for next to this module's package root
    (``research-agent/.env``) unless *env_path* is supplied explicitly.
    """
    if env_path is None:
        # src/main.py lives at <root>/src/main.py; .env is at <root>/.env
        env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    with env_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

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
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum number of topic depth levels (default: 3 = root + 2 child layers). "
             "Internally stored as max_depth - 1.",
    )
    parser.add_argument(
        "--tavily-key",
        type=str,
        default=None,
        metavar="KEY",
        help="Tavily Search API key. Overrides the TAVILY_API_KEY environment "
             "variable / .env file if provided.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of search queries per session. "
             "Env: RESEARCH_MAX_QUERIES (default: unlimited).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of topic-graph nodes to create. "
             "Env: RESEARCH_MAX_NODES (default: unlimited).",
    )
    parser.add_argument(
        "--max-credits-spend",
        type=float,
        default=None,
        metavar="CREDITS",
        help="Maximum Tavily API credits to spend per session. "
             "Env: RESEARCH_MAX_CREDITS (default: unlimited).",
    )
    parser.add_argument(
        "--respect-robots",
        action="store_true",
        default=None,
        help="Enable advisory robots.txt checks before Playwright scraping. "
             "Env: RESEARCH_RESPECT_ROBOTS (default: True).",
    )
    parser.add_argument(
        "--no-respect-robots",
        action="store_true",
        default=False,
        help="Disable advisory robots.txt checks.",
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        default=False,
        help="Disable Playwright scraping entirely. Content will come from "
             "Tavily raw_content and Extract only. "
             "Env: RESEARCH_NO_SCRAPE.",
    )
    # Dry-run and credit estimation
    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run planner decompositions to build the topic graph without executing "
             "any Tavily searches, then print an estimated credit cost and exit.",
    )
    dry_run_group.add_argument(
        "--estimate-credits",
        action="store_true",
        default=False,
        help="Alias for --dry-run.  Builds the graph and prints a credit estimate.",
    )
    parser.add_argument(
        "--warn-credits",
        type=float,
        default=0.80,
        metavar="FRACTION",
        help="Warn when this fraction (0.0–1.0) of any budget limit is consumed. "
             "Default: 0.80 (warn at 80%%). Set to 1.0 to disable warnings.",
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
        # No section markers at all.
        # If the file starts with a Markdown heading (# …), treat the heading
        # text as the concise searchable topic and the full file content as
        # background context (user_prompt) so all LLM calls are grounded in the
        # document without using the file-name as a search query.
        first_line = raw.split("\n")[0].strip()
        heading_match = _re.match(r"^#{1,3}\s+(.+)$", first_line)
        if heading_match:
            topic = heading_match.group(1).strip()
            user_prompt = raw  # full document is background context
        else:
            # Backward compat: entire file is topic
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
    max_depth: int = 2,
    max_queries: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_credits: Optional[float] = None,
    warn_threshold: float = 0.80,
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
        max_depth=max_depth,
        max_queries=max_queries,
        max_nodes=max_nodes,
        max_credits=max_credits,
        warn_threshold=warn_threshold,
    )

    await manager.init()

    # Fetch and log Tavily account credit balance at session start (best-effort)
    from src.tools.search_tool import fetch_account_credits
    account_info = await fetch_account_credits()
    if account_info:
        remaining = account_info.get("credits_remaining")
        limit = account_info.get("credits_limit")
        used = account_info.get("credits_used")
        if remaining is not None:
            logger.info(
                "Tavily account: %s credits remaining (of %s; %s used).",
                remaining, limit, used,
            )
            # Warn immediately if account balance is already low
            if max_credits is None and limit and remaining / limit < (1.0 - warn_threshold):
                logger.warning(
                    "Tavily account is %.0f%% consumed (%s/%s credits used) before "
                    "this session. Consider setting --max-credits-spend.",
                    (1 - remaining / limit) * 100, used, limit,
                )
    else:
        logger.debug("Tavily account usage endpoint unavailable — using local credit tracking only.")

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

        deadline_logged = False
        budget_logged = False
        while True:
            past_deadline = time.monotonic() >= end_time

            # Graceful stop when budget is exhausted
            if manager.budget.is_exhausted() and not budget_logged:
                budget_logged = True
                logger.info(
                    "Budget exhausted — finishing remaining consolidation work. %s",
                    manager.budget.summary(),
                )

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
                # Both graph and queue exhausted — we're done regardless of time
                logger.info("All research work complete.")
                break

            if finding:
                approved_count += 1
                remaining = end_time - time.monotonic()
                if remaining > 0:
                    logger.info(
                        "Approved finding #%d: %r  (%.0f min remaining)",
                        approved_count,
                        finding["subtopic"],
                        remaining / 60,
                    )
                else:
                    logger.info(
                        "Approved finding #%d: %r  (deadline passed — press Ctrl+C to stop)",
                        approved_count,
                        finding["subtopic"],
                    )

                # Progressive report save
                try:
                    manager.generate_report()
                except Exception as report_exc:
                    logger.warning("Progressive report save failed: %s", report_exc)

            # Log once when we pass the deadline but keep going
            if past_deadline and not deadline_logged:
                logger.info(
                    "Specified duration reached but research is still in progress. "
                    "Continuing until complete — press Ctrl+C to stop now."
                )
                deadline_logged = True

            cycles_completed += 1

            # Rate limiting — always sleep the full interval (no clamp to deadline)
            sleep_time = random.uniform(_CYCLE_SLEEP_MIN, _CYCLE_SLEEP_MAX)
            await asyncio.sleep(sleep_time)

    finally:
        # Always generate the final report
        report_path = manager.generate_report()
        await manager.close()
        elapsed = time.monotonic() - start_time
        budget_info = manager.budget.summary()
        logger.info(
            "Research session complete. Cycles: %d  Approved findings: %d  "
            "Elapsed: %.1f min  Report: %s  Budget: %s",
            cycles_completed,
            approved_count,
            elapsed / 60,
            report_path,
            budget_info,
        )


async def estimate_run(
    topic: str,
    title: Optional[str] = None,
    user_prompt: Optional[str] = None,
    model: str = "qwen2.5:7b",
    ollama_base_url: str = "http://localhost:11434",
    max_depth: int = 2,
) -> None:
    """Build the topic graph without searches and print a credit cost estimate.

    Uses LLM decomposition (Ollama) only — all Tavily API calls are suppressed.
    The graph structure determines how many leaf nodes need to be researched,
    which gives the estimated search credit cost.

    Credit estimates:
    - **Conservative**: 1 credit per leaf (1 search, no extract, first-try success)
    - **Typical**: 1.5 credits per leaf (1 search + Tavily Extract overhead)
    - **High**: 4 credits per leaf (search + extract + up to 2 retries)
    """
    from src.tools.search_tool import set_dry_run, fetch_account_credits

    set_dry_run(True)
    logger.info("Dry-run mode: Tavily searches suppressed — building graph with LLM only.")

    # Fetch Tavily account quota before suppressing calls (best-effort).
    account_credits = await fetch_account_credits()

    resolved_model = model
    try:
        available_models = _list_ollama_models(ollama_base_url)
        if available_models:
            resolved_model = _resolve_model_name(model, available_models)
    except Exception:
        pass  # proceed with requested model name; connection failure tolerated

    dry_run_dir = tempfile.mkdtemp(prefix="research-dry-run-")
    manager = AgentManager(
        topic=topic,
        title=title or topic,
        user_prompt=user_prompt,
        model=resolved_model,
        ollama_base_url=ollama_base_url,
        reports_dir=dry_run_dir,
        db_path=":memory:",
        max_depth=max_depth,
    )
    await manager.init()

    try:
        await manager.build_graph()
    except Exception as exc:
        logger.warning("Dry-run: graph build incomplete — estimate may be partial: %s", exc)
    finally:
        await manager.close()

    graph = manager._graph
    if graph is None:
        print("\nCould not build topic graph — no estimate available.\n")
        return

    all_nodes = list(graph._nodes.values())
    # Use terminal nodes (no children, not the root) as the estimation basis.
    # In dry-run, depth-1 children are never analyzed so is_leaf is not set;
    # counting childless non-root nodes gives the correct leaf count.
    leaf_nodes = [n for n in all_nodes if len(n.children_ids) == 0 and n.depth > 0]
    # Fall back to is_leaf flag if a deeper graph was somehow built.
    if not leaf_nodes:
        leaf_nodes = [n for n in all_nodes if n.is_leaf]
    leaf_count = len(leaf_nodes)
    leaf_ids = {n.id for n in leaf_nodes}
    total_nodes = len(all_nodes)

    # Credit cost model (per leaf node):
    # - 1 Tavily Search = 1 credit
    # - Tavily Extract for top-3 URLs = ~3/5 = 0.6 credits overhead
    # - Retry (Critic REJECT + refine + re-search): additional 1–2 credits each
    root_context_credits = 1.0   # root initial research
    credits_low = root_context_credits + leaf_count * 1.0
    credits_typical = root_context_credits + leaf_count * 1.5
    credits_high = root_context_credits + leaf_count * 4.0

    width = 62
    print()
    print("=" * width)
    print(f"  Credit estimate  —  {topic[:width - 20]}")
    print("=" * width)
    print(f"  Topic graph nodes:      {total_nodes:3d}  (root + {total_nodes - 1} subtopics)")
    print(f"  Leaf nodes to research: {leaf_count:3d}")
    print()
    print("  Estimated Tavily credits:")
    print(f"    Conservative (no retries, no extract): ~{credits_low:.0f}")
    print(f"    Typical     (search + extract):        ~{credits_typical:.0f}")
    print(f"    High        (retries + extract):       ~{credits_high:.0f}")
    print()
    print("  Tavily account quota:")
    if account_credits:
        used = account_credits.get("credits_used")
        limit = account_credits.get("credits_limit")
        remaining = account_credits.get("credits_remaining")
        if remaining is not None and limit:
            pct_used = (used or 0) / limit * 100
            can_run = "YES" if remaining >= credits_typical else "MAYBE (low)" if remaining >= credits_low else "NO (insufficient credits)"
            print(f"    Used:      {used} / {limit}  ({pct_used:.0f}%)")
            print(f"    Remaining: {remaining} credits")
            print(f"    Can run this topic? {can_run}")
        else:
            print(f"    {account_credits}")
    else:
        print("    (unavailable — check TAVILY_API_KEY or visit app.tavily.com)")
    print()
    print("  Tavily pricing reference:")
    print("    Free tier: 1,000 credits/month")
    print("    1 Search  = 1 credit")
    print("    1 Extract = 1 credit per 5 URLs (~0.6 credit overhead/node)")
    print()
    print("  Graph outline:")
    for depth in range(graph.max_depth_present() + 1):
        nodes_at_depth = graph.get_nodes_at_depth(depth)
        indent = "    " + "  " * depth
        for n in nodes_at_depth:
            leaf_marker = " *" if (n.id in leaf_ids or n.is_leaf) else ""
            name_trunc = n.name[:width - len(indent) - 4]
            print(f"  {indent}{name_trunc}{leaf_marker}")
    print()
    print("  * = leaf node (will be researched)")
    print()
    print("  To start researching, run without --dry-run / --estimate-credits.")
    print("  To cap spending, add:  --max-credits-spend <N>")
    print("=" * width)
    print()


def main() -> None:
    args = _parse_args()

    # Dry-run / estimate-credits mode — no searches, just decompose & count nodes
    is_dry_run = args.dry_run or args.estimate_credits

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

    # --tavily-key CLI arg overrides env var / .env
    if args.tavily_key:
        os.environ["TAVILY_API_KEY"] = args.tavily_key

    # Compute internal max_depth (user says 3 levels → internal limit 2)
    max_depth = max(args.max_depth - 1, 0)

    # Resolve budget limits — CLI args override env vars
    def _int_env(name: str) -> Optional[int]:
        val = os.environ.get(name, "").strip()
        return int(val) if val else None

    def _float_env(name: str) -> Optional[float]:
        val = os.environ.get(name, "").strip()
        return float(val) if val else None

    max_queries = args.max_queries if args.max_queries is not None else _int_env("RESEARCH_MAX_QUERIES")
    max_nodes = args.max_nodes if args.max_nodes is not None else _int_env("RESEARCH_MAX_NODES")
    max_credits = args.max_credits_spend if args.max_credits_spend is not None else _float_env("RESEARCH_MAX_CREDITS")

    warn_threshold = args.warn_credits

    # Resolve scraping flags (CLI → env → default)
    from src.tools.scraper_tool import set_respect_robots, set_no_scrape

    def _bool_env(name: str, default: bool) -> bool:
        val = os.environ.get(name, "").strip().lower()
        if val in ("1", "true", "yes"):
            return True
        if val in ("0", "false", "no"):
            return False
        return default

    if args.no_respect_robots:
        set_respect_robots(False)
    elif args.respect_robots is not None:
        set_respect_robots(True)
    else:
        set_respect_robots(_bool_env("RESEARCH_RESPECT_ROBOTS", True))

    no_scrape = args.no_scrape or _bool_env("RESEARCH_NO_SCRAPE", False)
    if no_scrape:
        set_no_scrape(True)
        logger.info("Playwright scraping disabled (--no-scrape).")

    # Dry-run path — estimate credits without running any real searches
    if is_dry_run:
        try:
            asyncio.run(
                estimate_run(
                    topic=topic,
                    title=report_title,
                    user_prompt=user_prompt,
                    model=args.model,
                    ollama_base_url=args.ollama_url,
                    max_depth=max_depth,
                )
            )
        except KeyboardInterrupt:
            logger.info("Dry-run interrupted by user.")
        return

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
                max_depth=max_depth,
                max_queries=max_queries,
                max_nodes=max_nodes,
                max_credits=max_credits,
                warn_threshold=warn_threshold,
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        SearchLogger.close()


if __name__ == "__main__":
    main()
