# AI-Driven Autonomous Research Agent

A **Recursive Agentic Research Engine** that autonomously researches technical
topics for extended periods (default: 8 hours) and produces *Code-Ready*
Markdown specifications.

Need a complete setup walkthrough from checkout to first query? See
`USER_GUIDE.md`.

## Architecture

```
research-agent/
├── prompts/                 # Customisable prompt templates
├── config/
│   └── filters.json         # Bundled filter config (stopwords, hub patterns, etc.)
├── src/
│   ├── main.py              # Entry point – async loop controller
│   ├── agent_manager.py     # Orchestrates Planner / Researcher / Critic
│   ├── llm_client.py        # Ollama + OpenAI-compatible LLM wrapper
│   ├── prompt_loader.py     # Prompt template loader with override support
│   ├── config_loader.py     # Filter config loader with per-task override support
│   ├── budget.py            # Per-session BudgetTracker (queries, nodes, credits)
│   ├── topic_graph.py       # TopicNode / TopicGraph DAG for hierarchical research
│   ├── agents/
│   │   ├── planner.py       # Topic decomposition
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
* Either:
  * [Ollama](https://ollama.ai/) running locally (`ollama serve`), or
  * an OpenAI-compatible online LLM endpoint such as [SiliconFlow](https://www.siliconflowcn.com/)
* A model available from your chosen provider
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

# Use a folder as a self-contained research task (recommended for complex projects)
python -m src.main --topic-dir ./my-research/ --duration 2h

# Short 30-minute run with a different model
python -m src.main --topic "Reinforcement Learning" --hours 0.5 --model mistral

# Requirements file with a custom duration
python -m src.main --requirements-file requirements.md --duration 1h30m

# Explicit model selection (recommended)
python -m src.main --topic "Reinforcement Learning" --duration 30m --model qwen2.5:7b

# Use SiliconFlow (OpenAI-compatible online LLM)
python -m src.main --topic "Reinforcement Learning" --duration 30m \
  --llm-provider siliconflow \
  --llm-url https://api.siliconflow.cn/v1 \
  --llm-api-key "$SILICONFLOW_API_KEY" \
  --model Qwen/Qwen2.5-7B-Instruct

# Dry-run: build topic graph and estimate Tavily credit cost — no real searches
python -m src.main --topic "Machine Learning" --dry-run

# Cap credit spend and warn when 70% of budget is consumed
python -m src.main --topic "Blockchain" --max-credits-spend 50 --warn-credits 0.70
```

Model resolution behavior:
- If `--model` is available locally, it is used directly.
- If `--model` is missing (for example `qwen2.5:7b` is requested but not pulled),
  the app automatically falls back to an available local model and logs a warning.
- To avoid ambiguity, pass `--model` explicitly.
- For online providers, the requested model name is passed through unchanged.

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

Use an online OpenAI-compatible provider such as SiliconFlow:

```bash
export SILICONFLOW_API_KEY="sk-..."
python -m src.main --topic "Reinforcement Learning" --duration 30m \
  --llm-provider siliconflow \
  --llm-url https://api.siliconflow.cn/v1 \
  --llm-api-key "$SILICONFLOW_API_KEY" \
  --model Qwen/Qwen2.5-7B-Instruct
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

#### Topic Directory (`--topic-dir`)

Pass an entire folder as a self-contained research task.  The agent reads the
topic from the folder, writes all output inside it, and can **resume
automatically** after a crash or quota interruption.

**Folder layout:**

```
my-research/
├── requirements.md      ← topic file (or topic.md, or any .md)
├── prompts/             ← optional: prompt template overrides
│   └── planner_decompose.md
├── config/              ← optional: filter config overrides
│   └── filters.json
└── output/              ← auto-created on first run
    ├── reports/         ← generated Markdown reports + JSON trees
    ├── research.db      ← SQLite knowledge base
    ├── task.json        ← persistent progress checkpoint
    └── search.jsonl     ← optional search query log
```

**Topic file resolution order** (first match wins):
1. `requirements.md`
2. `topic.md`
3. First `.md` file found alphabetically in the folder
4. Folder name used as-is if no `.md` file exists

**Example commands:**

```bash
# Create a topic folder
mkdir my-research
echo "## Topic\nQuantum Computing" > my-research/requirements.md

# First run — builds graph and starts researching
python -m src.main --topic-dir ./my-research/ --duration 2h

# Re-run after a quota/crash interruption — resumes from task.json automatically
python -m src.main --topic-dir ./my-research/ --duration 2h
```

The `prompts/` subfolder, when present, is automatically used as `--prompt-dir`.
An explicit `--prompt-dir` on the CLI always takes precedence over the folder's
`prompts/` subfolder.

The `config/` subfolder, when present, is automatically used as `--config-dir` to
override filter settings (stopwords, filler words, hub patterns, etc.).  An
explicit `--config-dir` on the CLI always takes precedence.

#### Crash Recovery / Progress Persistence (`task.json`)

When a data directory is active (via `--topic-dir` or `--data-dir`), the agent
writes a `task.json` checkpoint after every completed research node.  If the
run is interrupted — power off, network loss, API quota exhaustion — simply
re-run the **same command** and the agent will:

1. Detect the existing `task.json`.
2. Restore the full topic graph and all approved findings.
3. Reset any nodes that were mid-research (`researching` / `analyzing`) back to
   `pending` so they are retried cleanly.
4. Continue from where it left off, skipping the graph-build phase entirely.

The final `task.json` contains `"status": "completed"` once the graph is fully
consolidated.  Re-running a completed session regenerates the report without
performing new searches.

```bash
# With --data-dir, task.json lives at <data-dir>/task.json
python -m src.main --topic "AI Safety" --data-dir ./ai-safety-data/ --duration 3h

# After interruption — same command resumes from checkpoint
python -m src.main --topic "AI Safety" --data-dir ./ai-safety-data/ --duration 3h
```

#### Requirements File Format

For complex research tasks, create a plain-text or Markdown file that contains
the full specification. Prompt templates now live in `research-agent/prompts/`
or a custom `--prompt-dir`, so the requirements file only needs the research
topic and supporting requirements. The file name stem (e.g. `requirements` for
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
| `--topic-dir` | *(required\*)* | Path to a folder containing a requirements file; all output goes to `<folder>/output/`; re-running resumes from `task.json` |
| `--hours` | `8` | Hard time limit in hours. Research stops and the report is saved when the duration expires. |
| `--duration` | — | Hard time limit in human-readable format (e.g. `10m`, `1h30m`, `90s`). Overrides `--hours`. Research stops and the report is saved when the duration expires. If research finishes before the deadline, the agent logs a suggestion to re-run with a higher `--max-depth`. |
| `--model` | `qwen2.5:7b` | Model name for the selected LLM provider |
| `--llm-provider` | `ollama` | LLM backend: `ollama`, `openai`, or `siliconflow` |
| `--llm-url` | provider-specific | Base URL for the selected online/local LLM endpoint |
| `--llm-api-key` | — | API key for OpenAI-compatible online providers |
| `--ollama-url` | `http://localhost:11434` | Ollama base URL when `--llm-provider=ollama` |
| `--prompt-dir` | bundled prompts | Directory containing prompt template overrides |
| `--config-dir` | — | Directory containing a `filters.json` that overrides bundled filter defaults (stopwords, filler words, hub patterns, etc.) |
| `--tavily-key` | *(env `TAVILY_API_KEY`)* | Tavily API key for web search |
| `--data-dir` | — | Base data directory; overrides `--reports-dir`, `--db-path`, and `--search-log` defaults; also enables `task.json` persistence |
| `--reports-dir` | `data/reports` | Output directory for Markdown reports |
| `--db-path` | `data/research.db` | SQLite database path |
| `--search-log` | — | Path to a JSONL file logging every search query and result domains |
| `--max-depth` | `3` | Number of decomposition levels below the root (root + N sub-topic tiers). Increase this when research finishes before `--duration` ends — each extra level adds significantly more nodes and research time. |
| **Budget controls** | | |
| `--max-queries` | unlimited | Maximum Tavily API calls per session |
| `--max-nodes` | unlimited | Maximum topic-graph nodes to create |
| `--max-credits-spend` | unlimited | Maximum Tavily credits to spend |
| `--warn-credits` | `0.80` | Warn (log) when this fraction of any budget limit is consumed. Set to `1.0` to silence warnings. |
| **Dry-run / estimation** | | |
| `--dry-run` | `False` | Build the topic graph with LLM only (no Tavily searches), print a credit cost estimate, and exit. |
| **Scraping controls** | | |
| `--respect-robots` | `False` | Opt in to advisory `robots.txt` checks before scraping |

\* Exactly one of `--topic`, `--requirements-file`, or `--topic-dir` is required.

#### Environment Variable Fallbacks

Budget and scraping flags also read from environment variables when not passed on the CLI:

| CLI Flag | Environment Variable | Default |
|----------|---------------------|---------|
| `--tavily-key` | `TAVILY_API_KEY` | — |
| `--llm-provider` | `RESEARCH_LLM_PROVIDER` | `ollama` |
| `--llm-url` | `RESEARCH_LLM_URL` | provider-specific |
| `--llm-api-key` | `RESEARCH_LLM_API_KEY` / `SILICONFLOW_API_KEY` | — |
| `--prompt-dir` | `RESEARCH_PROMPT_DIR` | bundled prompts |
| `--config-dir` | `RESEARCH_CONFIG_DIR` | — |
| `--max-queries` | `RESEARCH_MAX_QUERIES` | unlimited |
| `--max-nodes` | `RESEARCH_MAX_NODES` | unlimited |
| `--max-credits-spend` | `RESEARCH_MAX_CREDITS` | unlimited |
| `--respect-robots` | `RESEARCH_RESPECT_ROBOTS` | `False` |

All variables can be placed in a `research-agent/.env` file (copied from
`research-agent/.env.example`) rather than exported as shell variables.

#### Prompt Templates

Bundled prompt templates live under `research-agent/prompts/`. To customise the
planner, researcher, or critic instructions without editing Python code, copy
that directory and point the CLI at your version:

```bash
cp -R research-agent/prompts /tmp/my-prompts
python -m src.main --topic "AI Safety" --prompt-dir /tmp/my-prompts
```

Only the `.md` prompt files you override need to exist in your custom
directory; missing files fall back to the bundled defaults.

#### Filter Configuration

Bundled filter settings (stopwords, filler words, CAPTCHA URL markers, hub
page patterns, and link exclusions) live in `research-agent/config/filters.json`.
To override specific settings for a particular run, copy the file to a custom
directory and pass that path on the CLI:

```bash
cp -R research-agent/config /tmp/my-config
# Edit /tmp/my-config/filters.json — only the keys you change need to be present
python -m src.main --topic "AI Safety" --config-dir /tmp/my-config
```

Only the keys present in your override file are merged on top of the bundled
defaults; all other keys retain their bundled values.

When using `--topic-dir`, place a `config/filters.json` inside the topic folder
for automatic per-task filter customization — no extra CLI flag needed.

## Output Format

Reports are saved to `data/reports/<topic>.md` (or `<folder>/output/reports/` when using `--topic-dir`):

```markdown
# Stock Trading Strategies

## Implementation Logic
1. **RSI** – Step 1: import pandas…

   *Sources: [1](https://investopedia.com/rsi) [2](https://arxiv.org/abs/...)*

## Math/Formulas
$$ RSI = 100 - \frac{100}{1 + RS} $$

## Dependencies
- `numpy`
- `pandas`

## Sources
- https://investopedia.com/rsi
- https://arxiv.org/abs/...
```

Each finding section includes **inline source reference links** (`*Sources: [1](url)*`)
directly after the summary text, so readers can follow citations without scrolling
to the bottom.  The global `## Sources` list at the end is preserved for a complete
reference overview.

## Running Tests

```bash
cd research-agent
python -m pytest tests/ -v
```

## Technical Stack

| Component | Library |
|-----------|---------|
| LLM | [Ollama](https://ollama.ai/) or any OpenAI-compatible endpoint such as SiliconFlow |
| Search | [Tavily](https://tavily.com/) Search + Extract API |
| Scraping | [Playwright](https://playwright.dev/python/) (headless Chromium, optional) |
| Vector store | [ChromaDB](https://www.trychroma.com/) *(optional, for semantic dedup)* |
| Structured DB | SQLite via [aiosqlite](https://github.com/omnilib/aiosqlite) |
| Concurrency | Python `asyncio` |
