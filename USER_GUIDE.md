# User Guide: Checkout to First Query

This guide walks you from a fresh checkout to running your first research query with a local Ollama model.

## 1. Checkout the code

```bash
git clone <your-repo-url>
cd AI-Driven-Autonomous-Research-Agent/research-agent
```

If you already have the repository, just `cd` into `research-agent/`.

## 2. Set up Python and project dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

Notes:
- Python 3.10+ is required.
- On some Linux distributions, Playwright may also need OS packages. If Chromium install fails, run:

```bash
python -m playwright install --with-deps chromium
```

## 3. Install Ollama (Linux)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify installation:

```bash
ollama --version
```

## 4. Start Ollama

Run Ollama in a separate terminal:

```bash
ollama serve
```

Keep it running while you use the research agent.

## 5. Download a model

Pull at least one model. Example:

```bash
ollama pull qwen2.5:7b
```

Optional (if you also want an alternate model):

```bash
ollama pull llama3
```

Optional sanity check against Ollama directly:

```bash
ollama run qwen2.5:7b "In one sentence, explain RSI in trading."
```

## 6. Set up your Tavily API key

The agent uses **Tavily** for web search.  Get a free API key at
[app.tavily.com](https://app.tavily.com) (free tier: 1,000 credits/month) and
export it before running:

```bash
export TAVILY_API_KEY="tvly-YOUR-API-KEY-HERE"
```

Or pass it inline:

```bash
python -m src.main --topic "..." --tavily-key tvly-YOUR-API-KEY-HERE
```

Without a key the agent will exit immediately with an error.

### Checking your credit balance

After setting up the key you can verify it and see how many credits remain:

```bash
python3 check_tavily_usage.py
```

Example output:

```
API key  : tvly-dev...XXXX
Checking usage...

──────────────────────────────────────────────────
  Tavily Credit Usage  (source: /usage endpoint)
──────────────────────────────────────────────────
  Plan      : Researcher
  Used      :          0
  Limit     :      1,000
  Remaining :      1,000
  Progress  : [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.0%
  Breakdown :  Search: 0  │  Extract: 0  │  Crawl: 0
  Pay-as-go :          0
──────────────────────────────────────────────────
```

Each run appends a snapshot to `data/tavily_usage_history.jsonl`, so re-running the script over time builds a credit-consumption trend table with delta annotations (`+N` / `-N` per run).

## 7. Run your first research query

From `research-agent/`:

```bash
python -m src.main --topic "Stock Trading Strategies" --duration 10m --model qwen2.5:7b
```

You can also use a requirements file instead of inline topic text:

```bash
python -m src.main --requirements-file requirements.md --duration 30m --model qwen2.5:7b
```

Important:
- Use exactly one of `--topic` or `--requirements-file`.
- Default Ollama URL is `http://localhost:11434`.
- The CLI default request is `--model qwen2.5:7b`.
- If the requested model is not installed, the app automatically falls back to a local installed model and logs a warning.
- To avoid accidental model changes between runs, pass `--model` explicitly.
- To use a custom data directory, specify `--data-dir /your/path` (reports go to `/your/path/reports` and db to `/your/path/research.db`).

### Duration and depth

`--duration` (or `--hours`) is a **hard time limit**. When it expires, the agent stops before starting any new research cycle and saves the report as-is — nothing is lost.

`--max-depth` controls how many decomposition tiers are created below the root (default: `3`, meaning root + 3 levels of sub-topics). More depth = more nodes = more research time.

Two scenarios you will encounter:

| Scenario | Log message | What to do |
|---|---|---|
| Duration expires while still researching | `Duration reached — stopping research and generating report.` | Increase `--duration`, or accept the partial report |
| Research finishes before duration | `All research work complete (N min remaining). For deeper research re-run with --max-depth X` | Re-run with the suggested `--max-depth` value |

Example — run finished in 10 min of a 1h session:
```bash
# First run finished early at depth 3
python -m src.main --topic "World Extraction Service" --duration 1h --max-depth 3

# Re-run one tier deeper to fill the remaining time
python -m src.main --topic "World Extraction Service" --duration 1h --max-depth 4
```

## Estimating costs before running

Before committing to a full research run, use `--dry-run` (or the equivalent
`--estimate-credits`) to build the topic graph with the LLM only — **no Tavily
searches are executed** — and print an estimated credit cost:

```bash
python -m src.main --topic "Quantum Computing" --dry-run
# or
python -m src.main --topic "Quantum Computing" --estimate-credits
```

Example output:

```
==============================================================
  Credit estimate  —  Quantum Computing
==============================================================
  Topic graph nodes:       11  (root + 10 subtopics)
  Leaf nodes to research:   8

  Estimated Tavily credits:
    Conservative (no retries):    9 credits
    Typical (some retries):      13 credits
    Pessimistic (max retries):   33 credits
...
==============================================================
```

This only uses LLM (Ollama) calls.  No Tavily credits are consumed.

## Budget controls

Avoid unexpected API costs by setting per-session limits:

```bash
# Cap at 50 Tavily credits per session
python -m src.main --topic "AI Safety" --max-credits-spend 50

# Cap at 100 search API calls
python -m src.main --topic "AI Safety" --max-queries 100

# Cap the number of topic-graph nodes created
python -m src.main --topic "AI Safety" --max-nodes 30

# Warn (log a warning) when 70% of the credit budget is consumed (default: 80%)
python -m src.main --topic "AI Safety" --max-credits-spend 50 --warn-credits 0.70

# Disable the warning entirely
python -m src.main --topic "AI Safety" --max-credits-spend 50 --warn-credits 1.0
```

All budget flags also read from environment variables (useful in `.env` files or
docker-compose):

| CLI flag | Environment variable | Default |
|----------|---------------------|---------|
| `--max-queries` | `RESEARCH_MAX_QUERIES` | unlimited |
| `--max-nodes` | `RESEARCH_MAX_NODES` | unlimited |
| `--max-credits-spend` | `RESEARCH_MAX_CREDITS` | unlimited |

At session end the agent logs a budget summary showing credits used, queries
made, and nodes created.

## Scraping controls

```bash
# Disable Playwright entirely (content from Tavily only, no browser required)
python -m src.main --topic "AI Safety" --no-scrape

# Disable robots.txt advisory checks (scraping still works, just skips the check)
python -m src.main --topic "AI Safety" --no-respect-robots
```

These also have environment variable equivalents:

| CLI flag | Environment variable | Default |
|----------|---------------------|---------|
| `--no-scrape` | `RESEARCH_NO_SCRAPE` | `False` |
| `--respect-robots` / `--no-respect-robots` | `RESEARCH_RESPECT_ROBOTS` | `True` |

## Model Guide: How to Specify and Choose a Model

1. Check local models:

```bash
ollama list
```

2. Run with a specific model:

```bash
python -m src.main --topic "Stock Trading Strategies" --duration 10m --model qwen2.5:7b
```

3. Quick comparison runs (same topic, different model):

```bash
python -m src.main --topic "Reinforcement Learning" --duration 10m --model qwen2.5:7b
python -m src.main --topic "Reinforcement Learning" --duration 10m --model llama3
python -m src.main --topic "Reinforcement Learning" --duration 10m --model mistral
```

Model differences (typical):

| Model | Typical strengths | Typical trade-offs | Best use case |
|------|-------------------|--------------------|---------------|
| `qwen2.5:7b` | Strong technical writing and code-ready structure | Can still be brief on very deep specialist areas | Daily default for this project |
| `llama3` | Good general reasoning and instruction following | Might not be installed locally on fresh setups | General-purpose long runs |
| `mistral` | Fast responses with lower resource needs | Often less detail for formulas/pseudocode-heavy topics | Fast trial runs and quick iteration |

Tip:
- Use the same topic and duration when comparing models, then compare reports in `data/reports/`.

## 8. (Optional) Using a custom data directory

By default, all reports and databases go to `data/reports/` and `data/research.db`.
If you want to use a different base directory, use `--data-dir`:

```bash
python -m src.main --topic "Stock Trading Strategies" --duration 10m --model qwen2.5:7b --data-dir /my/custom/data
```

This will store reports in `/my/custom/data/reports/` and the database in `/my/custom/data/research.db`.

You can also override individual paths if needed:

```bash
# Use default reports but custom db location
python -m src.main --topic "Reinforcement Learning" --duration 10m --db-path /tmp/research.db

# Use custom reports but default db
python -m src.main --topic "Reinforcement Learning" --duration 10m --reports-dir /my/reports
```

## 9. Find the generated report

Reports are written to:

```text
data/reports/
```

Example:

```bash
ls data/reports
```

The report filename is a sanitized version of the topic or requirements filename stem.

## 10. Common issues

1. Ollama connection errors:

```bash
curl http://localhost:11434/api/tags
```

If this fails, confirm `ollama serve` is running.

2. `ollama: command not found`:
- Restart your shell after installation.
- Check your PATH includes Ollama's install location.

3. Playwright/browser errors:

```bash
python -m playwright install --with-deps chromium
```

5. Missing Tavily API key:

```bash
export TAVILY_API_KEY="tvly-YOUR-API-KEY-HERE"
```

Or pass it with `--tavily-key tvly-YOUR-API-KEY-HERE`.

6. Missing Python packages:

```bash
pip install -r requirements.txt
```
