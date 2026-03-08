# AI-Driven Autonomous Research Agent

A **Recursive Agentic Research Engine** that autonomously researches technical
topics for extended periods (default: 8 hours) and produces *Code-Ready*
Markdown specifications.

Need a complete setup walkthrough from checkout to first query? See
`USER_GUIDE.md`.

## Architecture

```
research-agent/
├── src/
│   ├── main.py              # Entry point – async loop controller
│   ├── agent_manager.py     # Orchestrates Planner / Researcher / Critic
│   ├── budget.py            # Per-session BudgetTracker (queries, nodes, credits)
│   ├── topic_graph.py       # TopicNode / TopicGraph DAG for hierarchical research
│   ├── agents/
│   │   ├── planner.py       # Topic decomposition (Ollama)
│   │   ├── researcher.py    # Search → Conditional Scrape → Summarise
│   │   └── critic.py        # Quality auditor (PROCEED / REJECT)
│   ├── tools/
│   │   ├── search_tool.py   # Tavily Search + Extract API wrapper
│   │   └── scraper_tool.py  # Playwright headless Chromium scraper
│   └── database/
│       └── knowledge_base.py  # SQLite + optional ChromaDB vector store
├── data/
│   └── reports/             # Generated Markdown reports + JSON trees
├── tests/                   # pytest unit/integration tests
├── requirements.txt
└── pyproject.toml
```

## Core Philosophy: The Critic Loop

Every piece of research must pass a **Critic Agent** review:

1. **Logical Steps / Clear Structure** — organized and coherent?
2. **Specific Details** — concrete facts, not vague?
3. **Task Relevant** — directly addresses the research question?

The Critic adapts intelligently to the topic type: technical topics may also
check for algorithms and library dependencies; general topics focus on accuracy
and clarity.  If any check fails, the system generates a more specific
follow-up query and retries automatically (up to 3 retries) — it never accepts
vague summaries.

## Quick Start

### 1. Prerequisites

* Python 3.10+
* [Ollama](https://ollama.ai/) running locally (`ollama serve`)
* At least one downloaded model, e.g. `ollama pull qwen2.5:7b` or `ollama pull llama3`
* **Tavily API key** — set `TAVILY_API_KEY` in your environment or pass `--tavily-key`. Sign up at [tavily.com](https://tavily.com) (free tier: 1,000 credits/month)

To check your current credit balance and usage history at any time:

```bash
python3 research-agent/check_tavily_usage.py
```

This reads `TAVILY_API_KEY` from the environment or from `research-agent/.env`, shows plan name, credits used/remaining, a per-type breakdown (Search / Extract / Crawl), and a local history table that accumulates across runs.

### 2. Install

```bash
cd research-agent
pip install -r requirements.txt
```

### 3. Run

```bash
# Research "Stock Trading Strategies" for 8 hours (inline topic)
python -m src.main --topic "Stock Trading Strategies"

# Use a requirements file with full research spec and output expectations
python -m src.main --requirements-file requirements.md

# Short 30-minute run with a different model
python -m src.main --topic "Reinforcement Learning" --hours 0.5 --model mistral

# Requirements file with a custom duration
python -m src.main --requirements-file requirements.md --duration 1h30m

# Explicit model selection (recommended)
python -m src.main --topic "Reinforcement Learning" --duration 30m --model qwen2.5:7b

# Dry-run: build topic graph and estimate Tavily credit cost — no real searches
python -m src.main --topic "Machine Learning" --dry-run
# or equivalently:
python -m src.main --topic "Machine Learning" --estimate-credits

# Cap credit spend and warn when 70% of budget is consumed
python -m src.main --topic "Blockchain" --max-credits-spend 50 --warn-credits 0.70
```

Model resolution behavior:
- If `--model` is available locally, it is used directly.
- If `--model` is missing (for example `qwen2.5:7b` is requested but not pulled),
  the app automatically falls back to an available local model and logs a warning.
- To avoid ambiguity, pass `--model` explicitly.

### Model Guide

Check which models are installed locally:

```bash
ollama list
```

Run with an explicit model:

```bash
python -m src.main --topic "Reinforcement Learning" --duration 30m --model qwen2.5:7b
python -m src.main --topic "Reinforcement Learning" --duration 30m --model llama3
python -m src.main --topic "Reinforcement Learning" --duration 30m --model mistral
```

Typical model differences:

| Model | Typical strengths | Typical trade-offs | Good fit |
|------|-------------------|--------------------|----------|
| `qwen2.5:7b` | Strong coding output, clear technical explanations, good speed/quality balance | Can still miss edge-case depth on highly niche topics | Default for most research runs |
| `llama3` | Strong general reasoning and broad instruction following | May not be installed locally by default | Good all-round choice if already pulled |
| `mistral` | Usually fast with lower resource usage | Summaries may be shorter/less detailed on complex math-heavy tasks | Faster exploratory runs |

Notes:
- Model quality/speed depends on hardware and quantization level.
- For stable reproducibility, pin the model explicitly with `--model`.

#### Requirements File Format

For complex research tasks, create a plain-text or Markdown file that contains
the full specification.  The file name stem (e.g. `requirements` for
`requirements.md`) is used as the report title and output file name.

```markdown
## Research Detail
Research algorithmic trading strategies based on the RSI indicator.
Cover the mathematical formula, Python implementation using pandas/numpy,
back-testing methodology, and known pitfalls.

## Output Expectations
- Python code that calculates RSI from OHLCV data.
- Step-by-step pseudocode for the trading signal logic.
- LaTeX-formatted mathematical formulas.
- List of all required Python libraries.
- References to authoritative sources.
```

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--topic` | *(required\*)* | High-level research topic as inline text |
| `--requirements-file` | *(required\*)* | Path to a file with the full research specification |
| `--hours` | `8` | How long to run |
| `--duration` | — | Human-readable duration (e.g. `10m`, `1h30m`, `90s`) |
| `--model` | `qwen2.5:7b` | Requested Ollama model name. Falls back to any installed local model if unavailable. |
| `--ollama-url` | `http://localhost:11434` | Ollama base URL |
| `--tavily-key` | *(env `TAVILY_API_KEY`)* | Tavily API key for web search |
| `--data-dir` | — | Base data directory; overrides `--reports-dir`, `--db-path`, and `--search-log` defaults |
| `--reports-dir` | `data/reports` | Output directory for Markdown reports |
| `--db-path` | `data/research.db` | SQLite database path |
| `--search-log` | — | Path to a JSONL file logging every search query and result domains |
| `--max-depth` | `3` | Maximum topic graph depth (root + N levels of sub-topics) |
| **Budget controls** | | |
| `--max-queries` | unlimited | Maximum Tavily API calls per session |
| `--max-nodes` | unlimited | Maximum topic-graph nodes to create |
| `--max-credits-spend` | unlimited | Maximum Tavily credits to spend |
| `--warn-credits` | `0.80` | Warn (log) when this fraction of any budget limit is consumed. Set to `1.0` to silence warnings. |
| **Dry-run / estimation** | | |
| `--dry-run` | `False` | Build the topic graph with LLM only (no Tavily searches), print a credit cost estimate, and exit. |
| `--estimate-credits` | `False` | Alias for `--dry-run`. |
| **Scraping controls** | | |
| `--respect-robots` | `True` | Honour advisory `robots.txt` checks before scraping |
| `--no-respect-robots` | — | Disable `robots.txt` advisory checks |
| `--no-scrape` | `False` | Disable Playwright entirely; content comes from Tavily only |

\* Exactly one of `--topic` or `--requirements-file` is required.

#### Environment Variable Fallbacks

Budget and scraping flags also read from environment variables when not passed on the CLI:

| CLI Flag | Environment Variable | Default |
|----------|---------------------|---------|
| `--tavily-key` | `TAVILY_API_KEY` | — |
| `--max-queries` | `RESEARCH_MAX_QUERIES` | unlimited |
| `--max-nodes` | `RESEARCH_MAX_NODES` | unlimited |
| `--max-credits-spend` | `RESEARCH_MAX_CREDITS` | unlimited |
| `--respect-robots` | `RESEARCH_RESPECT_ROBOTS` | `True` |
| `--no-scrape` | `RESEARCH_NO_SCRAPE` | `False` |

## Output Format

Reports are saved to `data/reports/<topic>.md`:

```markdown
# Stock Trading Strategies

## Implementation Logic
1. **RSI** – Step 1: import pandas…

## Math/Formulas
$$ RSI = 100 - \frac{100}{1 + RS} $$

## Dependencies
- `numpy`
- `pandas`

## Sources
- https://…
```

## Running Tests

```bash
cd research-agent
python -m pytest tests/ -v
```

## Technical Stack

| Component | Library |
|-----------|---------|
| LLM | [Ollama](https://ollama.ai/) (local) |
| Search | [Tavily](https://tavily.com/) Search + Extract API |
| Scraping | [Playwright](https://playwright.dev/python/) (headless Chromium, optional) |
| Vector store | [ChromaDB](https://www.trychroma.com/) *(optional, for semantic dedup)* |
| Structured DB | SQLite via [aiosqlite](https://github.com/omnilib/aiosqlite) |
| Concurrency | Python `asyncio` |
