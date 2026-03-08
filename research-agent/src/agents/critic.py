"""
critic.py – Quality Assurance auditor.

Reviews research summaries and decides whether they contain enough detail,
specificity, and clarity to be accepted as high-quality research findings.
Handles both technical and non-technical topics flexibly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CRITIC_PROMPT = """You are a Senior Research Quality Auditor.

Original research topic: "{topic}"
Subtask being evaluated: "{task}"
{user_context}
ASSESSMENT RULES:
- For TECHNICAL topics (code, algorithms, math): Check for logical steps, formulas, and library dependencies
- For GENERAL topics (history, facts, entertainment, news): Check for detailed, specific information with proper structure
- For MIXED topics: Apply relevant criteria from both categories

Evaluate:
1. Logical Steps / Clear Structure? (explanation or algorithm present)
2. Specific Details? (concrete facts, not vague)
3. Relevant to Subtask? (directly addresses the SUBTASK being evaluated — it does NOT need to answer
   the entire original topic; sub-topics contribute a focused piece of the larger research)

Decision rules:
- If ALL three checks are YES → output status: PROCEED
- If ANY check is NO → output status: REJECT and list what is missing

Respond ONLY with JSON (no extra text):
{{
  "status": "PROCEED" | "REJECT",
  "checks": {{
    "logical_steps": true | false,
    "specific_details": true | false,
    "task_relevant": true | false
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
    """Quick regex-based quality check used as Ollama fallback."""
    # Check for logical structure/steps (numbered lists, procedural markers, clear sections)
    has_steps = bool(re.search(r"(\d+\.\s|step\s*\d|first|then|next|finally)", summary, re.I))
    
    # Check for specific details (numbers, names, dates, technical terms)
    has_specifics = bool(re.search(r"(\d{1,4}|[A-Z][a-z]+|:\s|—|•|★)", summary))
    
    # Check if content is substantive (not too short, has some detail)
    is_substantive = len(summary) > 150 and summary.count(" ") > 20
    
    task_relevant = is_substantive  # if substantive and specific, likely relevant

    missing_parts = []
    if not has_steps:
        missing_parts.append("clear structure/organization")
    if not has_specifics:
        missing_parts.append("specific details")
    if not task_relevant:
        missing_parts.append("sufficient depth/content")

    status = "PROCEED" if not missing_parts else "REJECT"
    return {
        "status": status,
        "checks": {
            "logical_steps": has_steps,
            "specific_details": has_specifics,
            "task_relevant": task_relevant,
        },
        "missing": ", ".join(missing_parts),
    }


class CriticAgent:
    """Evaluates whether a research summary meets quality standards for acceptance."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
        user_prompt: Optional[str] = None,
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url
        self._user_prompt = user_prompt

    def _call_ollama(self, prompt: str) -> Optional[str]:
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.ollama_base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data.get("response", "")
        except urllib.error.HTTPError as exc:
            error_text = ""
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="ignore"))
                if isinstance(payload, dict):
                    error_text = str(payload.get("error", ""))
            except Exception:
                error_text = ""

            if error_text:
                logger.warning("Ollama critic call failed (%s): %s", exc.code, error_text)
            else:
                logger.warning("Ollama critic call failed (%s): %s", exc.code, exc.reason)
            return None
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

    async def review(self, task: str, summary: str, topic: str = "") -> dict[str, Any]:
        """Evaluate *summary* for research quality.

        *topic* is the original user research topic (e.g. "Westworld TV Series").
        When provided it anchors the relevance check so off-topic content from
        ambiguous queries (e.g. "S3" resolved to Amazon S3 instead of Season 3)
        is correctly rejected.

        Returns a dict with at minimum:
        ``{'status': 'PROCEED'|'REJECT', 'missing': str}``.
        """
        if not summary.strip():
            return {
                "status": "REJECT",
                "checks": {
                    "logical_steps": False,
                    "specific_details": False,
                    "task_relevant": False,
                },
                "missing": "Empty summary – no content was retrieved.",
            }

        prompt = _CRITIC_PROMPT.format(
            topic=topic or task, task=task, summary=summary[:6000],
            user_context=(
                f"User instructions:\n{self._user_prompt}\n"
                if self._user_prompt else ""
            ),
        )

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
