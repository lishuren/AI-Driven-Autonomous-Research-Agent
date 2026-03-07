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

## 6. Run your first research query

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

## 7. Find the generated report

Reports are written to:

```text
data/reports/
```

Example:

```bash
ls data/reports
```

The report filename is a sanitized version of the topic or requirements filename stem.

## 8. Common issues

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

4. Missing Python packages:

```bash
pip install -r requirements.txt
```
