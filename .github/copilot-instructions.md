# GitHub Copilot Instructions

## Project Overview

This is an **AI-Driven Autonomous Research Agent** — a recursive agentic engine that
autonomously researches diverse topics (technical, general knowledge, entertainment, etc.)
for extended durations (default 8 hours) and produces detailed Markdown reports. It runs
locally against an Ollama LLM and uses DuckDuckGo for live web searches.

---

## Architecture Summary

```
research-agent/src/
├── main.py              # CLI entry point; asyncio loop controller
├── agent_manager.py     # Orchestrates Planner → Researcher → Critic pipeline
├── agents/
│   ├── planner.py       # Decomposes topics into search tasks; cleans queries to prevent embellishment
│   ├── researcher.py    # Search → Scrape → Summarise pipeline
│   └── critic.py        # Quality auditor (PROCEED / REJECT) – flexible for all topic types
├── tools/
│   ├── search_tool.py   # DuckDuckGo async wrapper with rate-limiting and retry logic
│   └── scraper_tool.py  # Playwright headless Chromium scraper with retry on transient errors
└── database/
    └── knowledge_base.py  # SQLite (aiosqlite) + optional ChromaDB vector store
```

### Data flow

```
CLI (main.py)
  └─▶ AgentManager.populate_queue()   ← PlannerAgent.decompose()
        └─▶ AgentManager.run_cycle()
              ├─▶ ResearcherAgent.research()
              │     ├─▶ SearchTool.search()      (DuckDuckGo)
              │     ├─▶ ScraperTool.scrape()     (Playwright / Chromium)
              │     └─▶ Ollama summarise prompt
              ├─▶ KnowledgeBase.is_duplicate()   (ChromaDB / SQLite)
              ├─▶ CriticAgent.review()           (Ollama + heuristic fallback)
              │     ├─▶ PROCEED → KnowledgeBase.save() → approved list
              │     └─▶ REJECT  → PlannerAgent.refine_task() → re-queue (≤3 retries)
              └─▶ AgentManager.generate_report() → Markdown file
```

---

## Key Design Patterns

### Critic Loop (quality gate)
Every research result must pass three checks before being accepted:
1. **Logical Steps / Clear Structure** (organized and coherent)
2. **Specific Details** (concrete facts, not vague)
3. **Task Relevant** (directly addresses the research question)

The Critic intelligently adapts to topic type:
- **Technical topics**: May additionally verify algorithms, formulas, library dependencies
- **General topics**: Focuses on accuracy, specificity, and clarity
- **Mixed topics**: Applies criteria relevant to the content

Failure on any check triggers a refined follow-up query (via `PlannerAgent.refine_task`)
and re-queues the task. Maximum `_MAX_REJECT_RETRIES = 3` attempts per task.

### Ollama integration
All three agents communicate with a **local Ollama server** via a plain
`urllib.request` HTTP call to `POST /api/generate`. No Ollama SDK is used at
the HTTP level — the `ollama` package in `requirements.txt` is available but
the agents call the REST API directly to keep IO in their own executor threads.

- Requested default model: `qwen2.5:7b`
- If the requested model is unavailable locally, runtime falls back to an installed local model and logs a warning
- Default base URL: `http://localhost:11434`
- Timeout: 120 s (Planner), 180 s (Researcher), 120 s (Critic)

### Async strategy
The codebase is `asyncio`-native. Blocking operations (Ollama calls, DuckDuckGo
search, web scraping, ChromaDB) are offloaded to `loop.run_in_executor(None, ...)`.
All public agent methods are `async def`.

### KnowledgeBase (storage)
| Backend  | Purpose | Fallback |
|----------|---------|---------|
| SQLite (`aiosqlite`) | Structured metadata, always present | — |
| ChromaDB (optional) | Semantic deduplication via cosine similarity (threshold 0.85) | Exact-string match |

DB and report paths default to `data/research.db` and `data/reports/`.

### Rate-limiting and Retry Logic
`SearchTool` and `ScraperTool` implement smart retry behavior:
- **Transient errors** (timeout, DNS, connection refused, 429/503) retry up to 3 attempts with exponential backoff (1s, 2s, 4s + jitter)
- **Permanent errors** (404, 403, invalid URL) fail immediately without retry
- Both tools maintain rate-limiting: `SearchTool` enforces 2–5 s random delay between DuckDuckGo requests
- `main.py` additionally sleeps 2–5 s between research cycles and refreshes queue every 10 approved findings (`_QUEUE_REFRESH_EVERY`)

---

## CLI Usage

```bash
# Run from research-agent/ directory
python -m src.main --topic "Stock Trading Strategies" --hours 8
python -m src.main --topic "Transformer Attention Mechanisms" --duration 1h30m
python -m src.main --requirements-file requirements.md --duration 45m
```

Arguments:
- `--topic` / `--requirements-file` (mutually exclusive, one required)
- `--hours` — integer hours (default 8)
- `--duration` — flexible format: `30s`, `10m`, `1h`, `1h30m`, `2hrs`, `90min`
- `--model` — Requested Ollama model name (default `qwen2.5:7b`); falls back to an installed local model if unavailable
- `--ollama-url` — Ollama base URL (default `http://localhost:11434`)
- `--data-dir` — base data directory; if specified, overrides `--reports-dir` and `--db-path` defaults (e.g., `--data-dir /custom/path` uses `/custom/path/reports` and `/custom/path/research.db`)
- `--reports-dir` — output directory for Markdown reports (default `data/reports`); overridden by `--data-dir` if specified
- `--db-path` — path to SQLite database (default `data/research.db`); overridden by `--data-dir` if specified

---

## Testing

Framework: `pytest` + `pytest-asyncio` (mode: `auto`).

```bash
cd research-agent
pytest                   # run all tests
pytest tests/test_agents.py -v
```

Test files mirror the source layout:
- `test_agents.py` — unit tests for Planner, Researcher, Critic
- `test_agent_manager.py` — AgentManager orchestration tests
- `test_knowledge_base.py` — KnowledgeBase CRUD and dedup tests
- `test_tools.py` — SearchTool / ScraperTool tests
- `test_main.py` — CLI argument parsing and main loop tests

---

## Dependencies

| Package | Role |
|---------|------|
| `aiosqlite` | Async SQLite for knowledge base |
| `chromadb` | Optional vector store for semantic dedup |
| `duckduckgo-search` | Web search (no API key required) |
| `playwright` | Headless Chromium scraping — handles JS/SPA pages |
| `aiohttp` | Async HTTP utilities |
| `pytest` + `pytest-asyncio` | Testing |

Python ≥ 3.10 required. Ollama must be running locally before starting the agent.

---

## Code Conventions

- All source files use `from __future__ import annotations` for PEP 563 deferred evaluation.
- Type hints are used throughout; prefer `Optional[T]` over `T | None` for compatibility.
- Logging uses the standard `logging` module; every module gets its own `logger = logging.getLogger(__name__)`.
- Private helpers are prefixed with `_` (e.g. `_call_ollama`, `_parse_json`).
- Constants are `UPPER_SNAKE_CASE` module-level variables prefixed with `_` when internal.
- Avoid adding external HTTP clients or LLM SDKs unless strictly necessary — prefer thin `urllib.request` calls to Ollama.
- Do not change the Planner → Researcher → Critic pipeline ordering or remove the critic quality gate.
- The Planner should generate queries faithful to the user's topic intent without adding unwanted technical framing.
- The Planner includes `_clean_query()` to remove embellishments like "detailed", "quality", "comprehensive" from generated queries.
- The Critic should evaluate content quality based on topic type, not force all topics into a single assessment model.
- When adding new agents, follow the same pattern: `__init__(model, ollama_base_url)`, async public method, `_call_ollama` private helper, graceful fallback when Ollama is unreachable.

---

## Output Format

Approved findings are consolidated by `AgentManager.generate_report()` into a
Markdown file with sections:

```markdown
# <topic>

## Implementation Logic
## Math/Formulas
## Dependencies
## Sources
```

Reports are saved incrementally to `data/reports/` during the run.
