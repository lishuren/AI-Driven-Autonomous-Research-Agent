# AI-Driven Autonomous Research Agent

A **Recursive Agentic Research Engine** that autonomously researches technical
topics for extended periods (default: 8 hours) and produces *Code-Ready*
Markdown specifications.

## Architecture

```
research-agent/
├── src/
│   ├── main.py              # Entry point – 8-hour loop controller
│   ├── agent_manager.py     # Orchestrates Planner / Researcher / Critic
│   ├── agents/
│   │   ├── planner.py       # Topic decomposition (Ollama)
│   │   ├── researcher.py    # Search → Scrape → Summarise
│   │   └── critic.py        # Code-readiness auditor (the Gatekeeper)
│   ├── tools/
│   │   ├── search_tool.py   # DuckDuckGo search wrapper
│   │   └── scraper_tool.py  # Playwright headless Chromium scraper
│   └── database/
│       └── knowledge_base.py  # SQLite + optional ChromaDB vector store
├── data/
│   └── reports/             # Generated Markdown reports
├── tests/                   # pytest unit/integration tests
├── requirements.txt
└── pyproject.toml
```

## Core Philosophy: The Critic Loop

Every piece of research must pass a **Critic Agent** review:

1. **Logical Steps** present?
2. **Mathematical Formulas** present?
3. **Python Library Dependencies** listed?

If any check fails the system generates a more specific follow-up query and
retries automatically – it never accepts vague summaries.

## Quick Start

### 1. Prerequisites

* Python 3.10+
* [Ollama](https://ollama.ai/) running locally (`ollama serve`)
* A downloaded model, e.g. `ollama pull llama3`

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
```

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
| `--duration` | — | Human-readable duration (e.g. `10m`, `1h30m`) |
| `--model` | `llama3` | Ollama model name |
| `--ollama-url` | `http://localhost:11434` | Ollama base URL |
| `--reports-dir` | `data/reports` | Output directory for Markdown reports |
| `--db-path` | `data/research.db` | SQLite database path |

\* Exactly one of `--topic` or `--requirements-file` is required.

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
| Search | [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) |
| Scraping | [Playwright](https://playwright.dev/python/) (headless Chromium) |
| Vector store | [ChromaDB](https://www.trychroma.com/) *(optional)* |
| Structured DB | SQLite via [aiosqlite](https://github.com/omnilib/aiosqlite) |
| Concurrency | Python `asyncio` |
