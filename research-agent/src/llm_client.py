from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
_DEFAULT_ONLINE_LLM_BASE_URL = "https://api.siliconflow.cn/v1"


def normalize_provider(provider: Optional[str]) -> str:
    """Normalise supported provider aliases to a stable internal name."""
    value = (provider or "ollama").strip().lower()
    if value in {"openai-compatible", "openai_compatible", "siliconflow", "online"}:
        return "openai"
    return value


def default_base_url(provider: Optional[str]) -> str:
    """Return the default base URL for *provider*."""
    if normalize_provider(provider) == "openai":
        return _DEFAULT_ONLINE_LLM_BASE_URL
    return _DEFAULT_OLLAMA_BASE_URL


def _extract_error_message(exc: urllib.error.HTTPError) -> str:
    """Best-effort extraction of an API error message from an HTTPError."""
    try:
        payload = json.loads(exc.read().decode("utf-8", errors="ignore"))
        if isinstance(payload, dict):
            if isinstance(payload.get("error"), str):
                return payload["error"]
            if isinstance(payload.get("message"), str):
                return payload["message"]
    except Exception:
        pass
    return str(exc.reason)


def _read_json_response(req: urllib.request.Request, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload if isinstance(payload, dict) else {}


def generate_text(
    prompt: str,
    model: str,
    base_url: str,
    *,
    provider: str = "ollama",
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> Optional[str]:
    """Generate text from the configured LLM provider."""
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "ollama":
        payload = json.dumps(
            {"model": model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            data = _read_json_response(req, timeout=timeout)
            response = data.get("response")
            return response if isinstance(response, str) else ""
        except urllib.error.HTTPError as exc:
            logger.warning(
                "Ollama call failed (%s): %s",
                exc.code,
                _extract_error_message(exc),
            )
            return None
        except Exception as exc:
            logger.warning("Ollama call failed: %s", exc)
            return None

    if normalized_provider == "openai":
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            data = _read_json_response(req, timeout=timeout)
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                return ""
            choice = choices[0]
            if not isinstance(choice, dict):
                return ""
            message = choice.get("message")
            if not isinstance(message, dict):
                return ""
            content = message.get("content")
            return content if isinstance(content, str) else ""
        except urllib.error.HTTPError as exc:
            logger.warning(
                "OpenAI-compatible LLM call failed (%s): %s",
                exc.code,
                _extract_error_message(exc),
            )
            return None
        except Exception as exc:
            logger.warning("OpenAI-compatible LLM call failed: %s", exc)
            return None

    logger.warning("Unsupported LLM provider %r.", provider)
    return None
