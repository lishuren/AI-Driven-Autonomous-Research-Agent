# GitHub Copilot Instructions

## Project Overview

This is an **AI-Driven Autonomous Research Agent** — a recursive agentic engine that
autonomously researches diverse topics (technical, general knowledge, entertainment, etc.)
for extended durations (default 8 hours) and produces detailed Markdown reports. It runs
locally against an Ollama LLM and uses **Tavily** for live web searches.

---

## Architecture Summary

```
research-agent/src/
├── main.py              # CLI entry point; asyncio loop controller
├── agent_manager.py     # Orchestrates Planner → Researcher → Critic pipeline (graph + flat modes)
├── budget.py            # Per-session BudgetTracker (queries, nodes, credits)
├── topic_graph.py       # TopicNode/TopicGraph DAG for hierarchical recursive research
├── agents/
│   ├── planner.py       # Decomposes topics; analyze/hierarchical decompose; consolidate; restructure
│   ├── researcher.py    # Search → Conditional Scrape → Summarise pipeline (topic-neutral)
│   └── critic.py        # Quality auditor (PROCEED / REJECT) – flexible for all topic types
├── tools/
│   ├── search_tool.py   # Tavily Search + Extract API wrapper (sole search backend)
│   └── scraper_tool.py  # Playwright headless Chromium scraper with ethics features
└── database/
    └── knowledge_base.py  # SQLite (aiosqlite) + optional ChromaDB vector store
```

### Data flow (Graph Mode — default)

Graph mode uses **BFS (breadth-first) layer-by-layer** research. The graph
is built lazily: only the root and its immediate children are created during
`build_graph()`. Deeper layers are decomposed on-the-fly during `run_graph()`
after the current depth's nodes have all been researched.

```
CLI (main.py)
  └─▶ AgentManager.build_graph()
        ├─▶ ResearcherAgent.research() on root topic (initial context)
        └─▶ AgentManager._decompose_node() on root (single level only)
              ├─▶ PlannerAgent.analyze(main_topic=…)  (leaf vs complex? + relevance)
              │     ├─▶ Leaf / low-relevance → mark pending for research
              │     └─▶ Complex → PlannerAgent.decompose_hierarchical(main_topic=…)
              │           └─▶ TopicGraph.add_node() for each sub-topic (3-5 children, dedup)
              └─▶ TopicGraph tracks all nodes, statuses, and priorities
  └─▶ AgentManager.run_graph() BFS loop (layer by layer)
        ├─▶ For current depth layer:
        │     ├─▶ Research pending leaf nodes → _research_node()
        │     │     ├─▶ BudgetTracker.can_query() guard (skip if exhausted)
        │     │     ├─▶ ResearcherAgent.research()  (Tavily search → Conditional Scrape → Summarise)
        │     │     ├─▶ KnowledgeBase.is_duplicate()
        │     │     ├─▶ CriticAgent.review()
        │     │     │     ├─▶ PROCEED → KnowledgeBase.save() → mark_researched
        │     │     │     └─▶ REJECT  → PlannerAgent.refine() → retry (≤3)
        │     │     └─▶ On exhaustion → mark_failed, stuck detection
        │     └─▶ Decompose pending non-leaf nodes → _decompose_node() (next layer)
        │           └─▶ _adaptive_max_children() limits branching when budget is low
        ├─▶ Advance to next depth when all nodes at current depth are done
        │     └─▶ TopicGraph.merge_similar_nodes() deduplicates via ChromaDB embeddings
        ├─▶ After all depths: Consolidate parent nodes (bottom-up)
        │     └─▶ _consolidate_node()
        │           ├─▶ PlannerAgent.consolidate_summaries()
        │           └─▶ CriticAgent.review() → mark_consolidated
        ├─▶ _save_tree_json() after every node completion → JSON tree file
        └─▶ AgentManager.generate_report() → hierarchical Markdown file (progressive)
```

### Data flow (Flat Mode — fallback)

```
CLI (main.py)
  └─▶ AgentManager.populate_queue()   ← PlannerAgent.decompose()  (+ pre-search vocab grounding)
        └─▶ AgentManager.run_cycle()
              ├─▶ ResearcherAgent.research()
              │     ├─▶ SearchTool.search()      (Tavily)
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
The default orchestration mode builds a **BFS layer-by-layer** TopicGraph (DAG):
- `TopicNode` dataclass tracks: name, query, depth, parent/child IDs, status
  (pending → analyzing → researching → completed → consolidated → failed),
  priority, summary, consolidated_summary, source_urls.
- `TopicGraph` manages node lifecycle, cross-reference dedup (case-insensitive
  name matching), and traversal helpers (`get_nodes_at_depth()`, `max_depth_present()`).
- Default depth: `MAX_DEPTH = 2` (3 total levels: root + 2). Configurable via `--max-depth` CLI arg.
- **BFS strategy**: `build_graph()` creates only the root and depth-1 children.
  `run_graph()` processes one depth layer at a time — researching all pending
  leaves and decomposing non-leaves before advancing to the next depth.
- **Relevance gate**: `PlannerAgent.analyze()` returns a `relevance` field
  (`high`/`medium`/`low`). Low-relevance sub-topics are forced to leaf status
  and not further decomposed, preventing topic drift.
- Each decomposition generates up to `max_children` sub-topics (default 5, reduced
  adaptively when budget is low via `_adaptive_max_children()`).
- `PlannerAgent.consolidate_summaries()` synthesizes child summaries; the
  Critic reviews consolidated content before marking nodes complete.
- Stuck detection triggers `PlannerAgent.suggest_restructure()` which adds new
  leaf nodes to the graph.
- `merge_similar_nodes(threshold=0.85)` merges semantically duplicate nodes
  using ChromaDB embeddings after each depth advance. Gracefully skips if
  ChromaDB is unavailable.
- `_save_tree_json()` writes a hierarchical JSON tree after every node completion.
- `to_tree_dict(exclude_empty=True)` builds the JSON tree, excluding nodes
  without summaries.

### Requirements File Format
The `--requirements-file` option supports section markers:
- `## Prompt` — Text extracted as a user prompt injected into all LLM calls.
- `## Topic` — Text used as the research topic.
- If no section markers are found and the file starts with a Markdown heading (`#`, `##`, or `###`), the heading text becomes the concise searchable **topic** and the full file content becomes the **user_prompt** (background context for the LLM). This avoids using the filename as a search query when the file is a rich analysis document.
- If no section markers and no leading heading, the entire file is used as the topic (backward compatible).

`build_graph()` derives `root_name` (used as the search query) from the topic's first line with Markdown heading markers stripped — never from the filename title.

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
The codebase is `asyncio`-native. Blocking operations (Ollama calls, Tavily
search, web scraping, ChromaDB) are offloaded to `loop.run_in_executor(None, ...)`.
All public agent methods are `async def`.

### KnowledgeBase (storage)
| Backend  | Purpose | Fallback |
|----------|---------|---------|
| SQLite (`aiosqlite`) | Structured metadata, always present | — |
| ChromaDB (optional) | Semantic deduplication via cosine similarity (threshold 0.85) | Exact-string match |

DB and report paths default to `data/research.db` and `data/reports/`.

### Rate-limiting and Retry Logic
`ScraperTool` implements smart retry behavior:
- **Transient errors** (timeout, DNS, connection refused, 429/503) retry up to 3 attempts with exponential backoff (1s, 2s, 4s + jitter)
- **Permanent errors** (404, 403, invalid URL) fail immediately without retry
- Random delay (0.5–2s) and User-Agent rotation per request (see Scraping Ethics)
- `main.py` additionally sleeps 2–5 s between research cycles and refreshes queue every 10 approved findings (`_QUEUE_REFRESH_EVERY`)

### Search Backend (Tavily)
`SearchTool` uses **Tavily** as the sole search backend. Requires a `TAVILY_API_KEY` environment variable (or `--tavily-key` CLI argument).
- Tavily Search returns structured results with titles, URLs, content snippets,
  and optionally `raw_content` (full page Markdown via `include_raw_content: "markdown"`).
- `include_usage: True` enables credit tracking via the `usage` field in API responses.
- `_tavily_search_sync()` is the single search function; results are normalised
  via `_normalise_results()` and filtered through `_CAPTCHA_URL_MARKERS`.
- `SearchTool.extract(urls, query)` calls the **Tavily Extract API** to fetch
  full page content for specific URLs without Playwright.
- `set_budget(tracker)` attaches a `BudgetTracker` — every search/extract call
  records credits and checks `can_query()` before executing.
- A `SearchLogger` optionally logs every query, result count, and domains to a JSONL file.
- **Dry-run mode**: `set_dry_run(enabled=True)` is a module-level function that suppresses
  all HTTP calls. When active, `search()` and `extract()` return `[]` immediately (before
  any budget guard or rate-limit logic). Used by `estimate_run()` in `main.py`.
- `fetch_account_credits()` is an async helper that issues a best-effort `GET` to the
  Tavily usage endpoint. Returns a dict with `credits_used`/`credits_limit`/`credits_remaining`
  or `None` if the endpoint is unavailable. Used by `run()` for informational account-level
  credit logging at session start. Gracefully degrades on HTTP 404/405/network errors.

**Language detection**: `_detect_language(query)` returns `"zh"` if >30% of
characters are CJK, otherwise `"en"`. Chinese queries set `country: "china"`
in the Tavily payload for region-optimised results. The Researcher adds a
`{language_hint}` ("Respond in Chinese") to the summarise prompt for `zh` queries.

### Content Acquisition (Conditional Scraping)
The `ResearcherAgent` uses a tiered content acquisition strategy:
1. **Tavily raw_content** — If `include_raw_content: "markdown"` yielded ≥500 chars
   per result, use it directly (no scraping needed).
2. **Tavily Extract** — For URLs lacking sufficient raw_content, call the
   Extract API (`extract_depth: "basic"`, `format: "markdown"`) to get full
   page content without a browser.
3. **Playwright** — Headless Chromium fallback for pages that Tavily couldn't
   extract. Skipped when `--no-scrape` is set.
4. **Snippets** — If all above fail, fall back to Tavily search snippets.

This cascading approach minimises Playwright usage and respects `--no-scrape`.

### Budget Controls
`BudgetTracker` (in `src/budget.py`) provides per-session limits:
- `max_queries` — maximum number of search API calls
- `max_nodes` — maximum topic-graph nodes to create
- `max_credits` — maximum Tavily API credits to spend
- `warn_threshold` — log a WARNING once when this fraction (0.0–1.0) of any limit is consumed (default `0.80`).
  Set to `1.0` to disable warnings entirely.

All limits are optional (`None` = unlimited). The tracker exposes:
- `record_query(credits)`, `record_node()` — increment counters
- `can_query()`, `can_create_node()` — pre-flight checks
- `is_exhausted()` — True when any limit is reached
- `used_fraction()` — 0.0–1.0 ratio of budget consumed
- `approaching_limit()` — True when `used_fraction() >= warn_threshold` AND at least one limit is configured
- `summary()` — human-readable status string (includes `warn_threshold` key)

`AgentManager` creates a `BudgetTracker` from CLI/env args and:
- Passes it to `SearchTool` via `set_budget()`
- Guards `_research_node()` and `_decompose_node()` with budget checks
- Uses `_adaptive_max_children()` to reduce branching factor as budget depletes:
  - ≥75% remaining → 5 children
  - ≥50% → 3 children
  - ≥25% → 2 children
  - <25% → 1 child
- Logs a budget summary at session end

### Scraping Ethics & Hardening
`ScraperTool` includes ethics and robustness features:
- **robots.txt advisory**: `_check_robots_txt(url)` uses `urllib.robotparser`
  to check site permissions before scraping. Advisory only — blocked URLs are
  logged as warnings but not skipped by default. Enabled via `--respect-robots`
  (default True). Uses an in-memory cache (`_ROBOTS_CACHE`) to avoid repeated fetches.
- **User-Agent rotation**: 8 realistic desktop User-Agent strings, randomly
  selected per request via `random.choice(_USER_AGENTS)`.
- **Random delay**: 0.5–2s sleep before each scrape to reduce server load.
- **`--no-scrape` flag**: Disables Playwright entirely; content comes from
  Tavily raw_content and Extract only.
- **Optional Playwright**: If `playwright` is not installed, scraping gracefully
  degrades to Tavily-only content acquisition via `_HAS_PLAYWRIGHT` flag.

### Adaptive Research Loop
The agent adapts its direction in three phases:

**Phase 1 — Pre-search vocabulary grounding**
Before generating queries, `PlannerAgent._pre_search_vocab()` runs a single quick Tavily search for the topic and extracts the most-frequent real words from titles and snippets (stopwords filtered). These are injected into `_DECOMPOSE_PROMPT` via `{vocab_section}` so the LLM uses words that actually appear in web results, not invented abstract nouns.

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
- `--max-depth` — maximum topic graph depth (default 3, i.e. root + 2 levels of sub-topics)
- `--tavily-key` — Tavily API key; overrides the `TAVILY_API_KEY` environment variable
- `--max-queries N` — maximum search API calls per session (env: `RESEARCH_MAX_QUERIES`; default: unlimited)
- `--max-nodes N` — maximum topic-graph nodes to create (env: `RESEARCH_MAX_NODES`; default: unlimited)
- `--max-credits-spend CREDITS` — maximum Tavily API credits to spend (env: `RESEARCH_MAX_CREDITS`; default: unlimited)
- `--warn-credits FRACTION` — warn when this fraction of any budget limit is consumed (default: `0.80`; set `1.0` to disable)
- `--respect-robots` / `--no-respect-robots` — enable or disable advisory robots.txt checks (env: `RESEARCH_RESPECT_ROBOTS`; default: True)
- `--no-scrape` — disable Playwright scraping entirely (env: `RESEARCH_NO_SCRAPE`; default: False)
- `--dry-run` — build the topic graph with LLM only (no Tavily searches), print a credit cost estimate, and exit
- `--estimate-credits` — alias for `--dry-run` (`--dry-run` and `--estimate-credits` are mutually exclusive)

### Environment Variable Fallbacks

All budget and scraping CLI flags fall back to environment variables when not specified:

| CLI Flag | Environment Variable | Type | Default |
|----------|---------------------|------|---------|
| `--max-queries` | `RESEARCH_MAX_QUERIES` | int | unlimited |
| `--max-nodes` | `RESEARCH_MAX_NODES` | int | unlimited |
| `--max-credits-spend` | `RESEARCH_MAX_CREDITS` | float | unlimited |
| `--respect-robots` | `RESEARCH_RESPECT_ROBOTS` | bool | `True` |
| `--no-scrape` | `RESEARCH_NO_SCRAPE` | bool | `False` |
| `--tavily-key` | `TAVILY_API_KEY` | str | — |

CLI flags always take precedence over environment variables. Helper functions
`_int_env()`, `_float_env()`, and `_bool_env()` parse the env var values.

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
- `test_agent_manager.py` — AgentManager orchestration tests (budget integration, adaptive depth)
- `test_budget.py` — BudgetTracker unit tests (limits, tracking, exhaustion, summary, warn_threshold, approaching_limit)
- `test_topic_graph.py` — TopicGraph / TopicNode tests (including node merge by similarity)
- `test_knowledge_base.py` — KnowledgeBase CRUD and dedup tests
- `test_tools.py` — SearchTool / ScraperTool tests (budget guard, extract, robots.txt, no-scrape, language detection, dry-run mode, account credits fetch)
- `test_main.py` — CLI argument parsing, env var fallbacks, budget/scraping/dry-run args, `estimate_run()`, and main loop tests

---

## Dependencies

| Package | Role |
|---------|------|
| `aiosqlite` | Async SQLite for knowledge base |
| `chromadb` | Optional vector store for semantic dedup |
| `tavily-python` | Tavily web search API client (sole search backend) |
| `playwright` | Headless Chromium scraping — handles JS/SPA pages |
| `aiohttp` | Async HTTP utilities |
| `pytest` + `pytest-asyncio` | Testing |

Python ≥ 3.10 required. Ollama must be running locally before starting the agent.
A **Tavily API key** is required — set `TAVILY_API_KEY` in the environment or pass `--tavily-key` on the CLI.

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
- `BudgetTracker` is created in `AgentManager.__init__()` and wired to `SearchTool` via `set_budget()`. Always check `can_query()` / `can_create_node()` before consuming resources.
- `BudgetTracker.warn_threshold` (default `0.80`) triggers a one-time WARNING log when usage crosses the threshold. Use `approaching_limit()` to check programmatically. Pass `warn_threshold=1.0` to disable.
- Dry-run / cost estimation: call `set_dry_run(True)` from `src.tools.search_tool` to suppress all Tavily HTTP calls. `main.py`'s `estimate_run()` uses this pattern — it builds the full topic graph via LLM decompositions, counts leaf nodes, and prints a credit cost estimate. `estimate_run()` uses `db_path=":memory:"` and a `tempfile.mkdtemp()` reports dir to avoid side effects.
- Scraping preferences (`_respect_robots`, `_no_scrape`) are set via module-level functions `set_respect_robots()` and `set_no_scrape()` in `scraper_tool.py` — called once at startup from `main()`.
- `PlannerAgent.decompose_hierarchical()` accepts a `max_children` parameter; `AgentManager._adaptive_max_children()` adjusts this based on remaining budget.
- `_detect_language()` in `search_tool.py` returns `"zh"` for CJK-heavy queries. The Researcher uses this to add a language hint to the summarise prompt.

---

## Output Format

Approved findings are consolidated by `AgentManager.generate_report()` into a
Markdown file. The report is **saved progressively after every approved finding**
so partial results survive interrupts.

### Markdown Report
When a topic graph exists, the report mirrors the graph hierarchy:
- Root summary appears as intro prose under `## Findings`.
- Depth-1 children get `###` headings.
- Depth-2 children get `####` headings.
- Falls back to a flat `### <subtopic>` list when no graph is present.

Structure:
```markdown
# <topic>

## Research Plan
(graph outline)

## Findings

(root summary prose)

### <depth-1 subtopic A>
(summary)

#### <depth-2 subtopic A.1>
(summary)

---

### <depth-1 subtopic B>
…

## Math/Formulas        ← only present when formulas appear in summaries
## Dependencies         ← only present when Python imports appear in summaries

## Sources
```

### JSON Tree
A hierarchical JSON tree is saved alongside the Markdown report as
`<safe_name>.json`. It is updated after every node completion (research,
consolidation, or failure). Empty nodes (no summary) are excluded by default.

Reports and JSON trees are saved to `data/reports/` (or `--reports-dir` / `--data-dir`).
An optional JSONL search log (`--search-log`) records every query, result count,
and result domains for post-run review.
