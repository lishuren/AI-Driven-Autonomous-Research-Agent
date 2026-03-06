"""
critic.py – Quality Assurance / Code-Readiness auditor.

Reviews research summaries and decides whether they contain enough detail
to write executable code (PROCEED) or need more specific follow-up (REJECT).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CRITIC_PROMPT = """You are a Senior Software Engineer performing a strict code-readiness audit.

Review the following research summary for the task: "{task}"

Does the summary contain:
1. Logical Steps / Algorithm? (Yes/No)
2. Mathematical Formulas? (Yes/No)
3. Python Library Dependencies? (Yes/No)

Decision rules:
- If ALL three checks are YES → output status: PROCEED
- If ANY check is NO → output status: REJECT and list EXACTLY what is missing

Respond ONLY with JSON (no extra text):
{{
  "status": "PROCEED" | "REJECT",
  "checks": {{
    "logical_steps": true | false,
    "math_formulas": true | false,
    "python_libraries": true | false
  }},
  "missing": "<concise description of what is missing, or empty string if PROCEED>"
}}

Research summary:
{summary}
"""

_HEURISTIC_FORMULA_PATTERNS = [
    r"\$\$.*?\$\$",        # LaTeX display math
    r"\$[^$]+\$",          # LaTeX inline math
    r"[a-zA-Z]\s*=\s*[-+]?\d",      # formula-like assignment (right side starts with digit/sign)
    r"\b(formula|equation|algorithm|pseudocode|formula:)\b",
]

_HEURISTIC_LIBRARY_PATTERNS = [
    r"\b(import|from)\s+\w+",
    r"\b(pandas|numpy|scipy|sklearn|tensorflow|torch|ta-lib|requests|aiohttp)\b",
]


def _heuristic_check(summary: str) -> dict[str, Any]:
    """Quick regex-based code-readiness check used as Ollama fallback."""
    has_steps = bool(re.search(r"(\d+\.\s|\bstep\b|\bfirst\b|\bthen\b)", summary, re.I))
    has_formulas = any(re.search(p, summary, re.I) for p in _HEURISTIC_FORMULA_PATTERNS)
    has_libs = any(re.search(p, summary, re.I) for p in _HEURISTIC_LIBRARY_PATTERNS)

    missing_parts = []
    if not has_steps:
        missing_parts.append("step-by-step algorithm")
    if not has_formulas:
        missing_parts.append("mathematical formulas")
    if not has_libs:
        missing_parts.append("Python library dependencies")

    status = "PROCEED" if not missing_parts else "REJECT"
    return {
        "status": status,
        "checks": {
            "logical_steps": has_steps,
            "math_formulas": has_formulas,
            "python_libraries": has_libs,
        },
        "missing": ", ".join(missing_parts),
    }


class CriticAgent:
    """Evaluates whether a research summary is ready for code generation."""

    def __init__(
        self,
        model: str = "llama3",
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url

    def _call_ollama(self, prompt: str) -> Optional[str]:
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data.get("response", "")
        except Exception as exc:
            logger.warning("Ollama critic call failed: %s", exc)
            return None

    def _parse_verdict(self, text: Optional[str]) -> Optional[dict[str, Any]]:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                verdict = json.loads(text[start : end + 1])
                if "status" in verdict:
                    return verdict
            except json.JSONDecodeError:
                pass
        return None

    async def review(self, task: str, summary: str) -> dict[str, Any]:
        """Evaluate *summary* for code-readiness.

        Returns a dict with at minimum:
        ``{'status': 'PROCEED'|'REJECT', 'missing': str}``.
        """
        if not summary.strip():
            return {
                "status": "REJECT",
                "checks": {
                    "logical_steps": False,
                    "math_formulas": False,
                    "python_libraries": False,
                },
                "missing": "Empty summary – no content was retrieved.",
            }

        prompt = _CRITIC_PROMPT.format(task=task, summary=summary[:6000])

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._call_ollama, prompt)

        verdict = self._parse_verdict(raw)
        if verdict:
            logger.info(
                "Critic verdict for %r: %s (missing: %s)",
                task,
                verdict.get("status"),
                verdict.get("missing", ""),
            )
            return verdict

        # Fallback to heuristic check
        logger.warning("Using heuristic critic for %r.", task)
        return _heuristic_check(summary)
