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
├── agent_manager.py     # Orchestrates Planner → Researcher → Critic pipeline (graph + flat modes)
├── topic_graph.py       # TopicNode/TopicGraph DAG for hierarchical recursive research
├── agents/
│   ├── planner.py       # Decomposes topics; analyze/hierarchical decompose; consolidate; restructure
│   ├── researcher.py    # Search → Scrape → Summarise pipeline (topic-neutral factual summary)
│   └── critic.py        # Quality auditor (PROCEED / REJECT) – flexible for all topic types
├── tools/
│   ├── search_tool.py   # DuckDuckGo async wrapper with rate-limiting and retry logic
│   └── scraper_tool.py  # Playwright headless Chromium scraper with retry on transient errors
└── database/
    └── knowledge_base.py  # SQLite (aiosqlite) + optional ChromaDB vector store
```

### Data flow (Graph Mode — default)

```
CLI (main.py)
  └─▶ AgentManager.build_graph()
        ├─▶ ResearcherAgent.research() on root topic (initial context)
        └─▶ AgentManager._decompose_node() (recursive)
              ├─▶ PlannerAgent.analyze()           (leaf vs complex?)
              │     ├─▶ Leaf → mark pending for research
              │     └─▶ Complex → PlannerAgent.decompose_hierarchical()
              │           └─▶ TopicGraph.add_node() for each sub-topic (cross-ref dedup)
              │                 └─▶ _decompose_node() recursively (up to MAX_DEPTH=5)
              └─▶ TopicGraph tracks all nodes, statuses, and priorities
  └─▶ AgentManager.run_graph() loop (while has_graph_work)
        ├─▶ Priority 1: Research pending leaf nodes
        │     └─▶ _research_node()
        │           ├─▶ ResearcherAgent.research()
        │           ├─▶ KnowledgeBase.is_duplicate()
        │           ├─▶ CriticAgent.review()
        │           │     ├─▶ PROCEED → KnowledgeBase.save() → mark_researched
        │           │     └─▶ REJECT  → PlannerAgent.refine() → retry (≤3)
        │           └─▶ On exhaustion → mark_failed, stuck detection
        ├─▶ Priority 2: Consolidate parent nodes (bottom-up)
        │     └─▶ _consolidate_node()
        │           ├─▶ PlannerAgent.consolidate_summaries()
        │           └─▶ CriticAgent.review() → mark_consolidated
        └─▶ AgentManager.generate_report() → Markdown file (progressive)
```

### Data flow (Flat Mode — fallback)

```
CLI (main.py)
  └─▶ AgentManager.populate_queue()   ← PlannerAgent.decompose()  (+ pre-search vocab grounding)
        └─▶ AgentManager.run_cycle()
              ├─▶ ResearcherAgent.research()
              │     ├─▶ SearchTool.search()      (DuckDuckGo)
              │     ├─▶ ScraperTool.scrape()     (Playwright / Chromium)
              │     └─▶ Ollama summarise prompt
              ├─▶ KnowledgeBase.is_duplicate()   (ChromaDB / SQLite)
              ├─▶ CriticAgent.review()           (Ollama + heuristic fallback)
              │     ├─▶ PROCEED → KnowledgeBase.save() → approved list
              │     │         → track _successful_queries, reset _consecutive_failures
              │     └─▶ REJECT  → PlannerAgent.refine_task() → re-queue (≤3 retries)
              │               → on exhaustion: track _failed_queries, incr _consecutive_failures
              │               → if _consecutive_failures ≥ 5: clear queue,
              │                   AgentManager.populate_queue(is_retrospective=True)
              │                   → PlannerAgent.decompose_retrospective()
              └─▶ AgentManager.generate_report() → Markdown file
```

---

## Key Design Patterns

### Hierarchical Topic Graph (Graph Mode)
The default orchestration mode builds a recursive **TopicGraph** (DAG):
- `TopicNode` dataclass tracks: name, query, depth, parent/child IDs, status
  (pending → analyzing → researching → completed → consolidated → failed),
  priority, summary, consolidated_summary, source_urls.
- `TopicGraph` manages node lifecycle, cross-reference dedup (case-insensitive
  name matching), and traversal helpers.
- Maximum recursion depth: `MAX_DEPTH = 5`.
- `AgentManager.build_graph()` does initial root research for context, then
  recursively calls `PlannerAgent.analyze()` (leaf vs complex) and
  `PlannerAgent.decompose_hierarchical()`.
- `AgentManager.run_graph()` processes one node per call: researches pending
  leaves first, then consolidates parent nodes bottom-up.
- `PlannerAgent.consolidate_summaries()` synthesizes child summaries; the
  Critic reviews consolidated content before marking nodes complete.
- Stuck detection triggers `PlannerAgent.suggest_restructure()` which adds new
  leaf nodes to the graph.

### Requirements File Format
The `--requirements-file` option supports section markers:
- `## Prompt` — Text extracted as a user prompt injected into all LLM calls.
- `## Topic` — Text used as the research topic.
- If no section markers are found, the entire file is used as the topic (backward compatible).

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

### Adaptive Research Loop
The agent adapts its direction in three phases:

**Phase 1 — Pre-search vocabulary grounding**
Before generating queries, `PlannerAgent._pre_search_vocab()` runs a single quick DuckDuckGo search for the topic and extracts the most-frequent real words from titles and snippets (stopwords filtered). These are injected into `_DECOMPOSE_PROMPT` via `{vocab_section}` so the LLM uses words that actually appear in web results, not invented abstract nouns.

**Phase 2 — Feedback loop**
`AgentManager` tracks `_successful_queries` (PROCEED outcomes) and `_failed_queries` (exhausted retries). On every call to `populate_queue()` the last 3 of each list are passed to `PlannerAgent.decompose()` as `good_examples` / `bad_examples` and injected into the prompt via `{feedback_section}`, steering the LLM toward patterns that work.

**Phase 3 — Stuck detection + retrospective re-plan**
`AgentManager` maintains `_consecutive_failures`. It resets to 0 on any PROCEED and increments on each exhausted-retry failure. When it reaches `_MAX_CONSECUTIVE_FAILURES = 5`, the queue is cleared and `populate_queue(is_retrospective=True)` calls `PlannerAgent.decompose_retrospective()` which uses `_RETROSPECTIVE_PROMPT` — a dedicated prompt showing the failed queries and asking the LLM to approach the topic from a completely different angle.

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
- `--data-dir` — base data directory; if specified, overrides `--reports-dir`, `--db-path`, and `--search-log` defaults
- `--reports-dir` — output directory for Markdown reports (default `data/reports`); overridden by `--data-dir` if specified
- `--db-path` — path to SQLite database (default `data/research.db`); overridden by `--data-dir` if specified
- `--search-log` — optional path to a JSONL file logging every search query, result count, and result domains (disabled by default; auto-set to `<data-dir>/search.jsonl` when `--data-dir` is used)

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
- `test_topic_graph.py` — TopicGraph / TopicNode tests
- `test_knowledge_base.py` — KnowledgeBase CRUD and dedup tests
- `test_tools.py` — SearchTool / ScraperTool tests
- `test_main.py` — CLI argument parsing and main loop tests
- `test_main.py` — CLI argument parsing and main loop tests

---

## Dependencies

| Package | Role |
|---------|------|
| `aiosqlite` | Async SQLite for knowledge base |
| `chromadb` | Optional vector store for semantic dedup |
| `ddgs` | Primary web search package (replaces `duckduckgo-search`) |
| `duckduckgo-search` | Fallback web search package |
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
- `PlannerAgent` accepts an optional `search_tool` in its constructor; when provided, `_pre_search_vocab()` seeds query generation with real vocabulary from a quick pre-search (Phase 1 of the adaptive loop).
- `PlannerAgent.decompose()` accepts `good_examples` and `bad_examples` lists for feedback-aware planning (Phase 2).
- `PlannerAgent.decompose_retrospective()` is used when the agent is stuck (Phase 3) — it uses a different prompt that explicitly acknowledges failure and requests a fresh angle.
- `AgentManager` tracks `_successful_queries`, `_failed_queries`, and `_consecutive_failures`; stuck detection triggers `populate_queue(is_retrospective=True)` when `_consecutive_failures ≥ _MAX_CONSECUTIVE_FAILURES = 5`.
- The Critic should evaluate content quality based on topic type, not force all topics into a single assessment model.
- When adding new agents, follow the same pattern: `__init__(model, ollama_base_url)`, async public method, `_call_ollama` private helper, graceful fallback when Ollama is unreachable.

---

## Output Format

Approved findings are consolidated by `AgentManager.generate_report()` into a
Markdown file. The report is **saved progressively after every approved finding**
so partial results survive interrupts.

Structure:
```markdown
# <topic>

## Findings

### <subtopic 1>
(full summary prose — style matches topic type)

---

### <subtopic 2>
…

## Math/Formulas        ← only present when formulas appear in summaries
## Dependencies         ← only present when Python imports appear in summaries

## Sources
```

Reports are saved to `data/reports/` (or `--reports-dir` / `--data-dir`).
An optional JSONL search log (`--search-log`) records every query, result count,
and result domains for post-run review.
