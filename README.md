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
* At least one downloaded model, e.g. `ollama pull qwen2.5:7b` or `ollama pull llama3`

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
| `--duration` | — | Human-readable duration (e.g. `10m`, `1h30m`) |
| `--model` | `qwen2.5:7b` | Requested Ollama model name. If unavailable locally, runtime falls back to an installed model and logs a warning. |
| `--ollama-url` | `http://localhost:11434` | Ollama base URL |
| `--data-dir` | — | Base data directory. If specified, overrides `--reports-dir` and `--db-path` defaults (e.g., `--data-dir /custom/path` uses `/custom/path/reports` and `/custom/path/research.db`) |
| `--reports-dir` | `data/reports` | Output directory for Markdown reports (overridden by `--data-dir` if specified) |
| `--db-path` | `data/research.db` | SQLite database path (overridden by `--data-dir` if specified) |

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
