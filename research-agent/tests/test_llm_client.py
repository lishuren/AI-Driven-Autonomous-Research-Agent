from __future__ import annotations

import json
from unittest.mock import patch

from src.llm_client import generate_text, normalize_provider


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_normalize_provider_maps_siliconflow_to_openai():
    assert normalize_provider("siliconflow") == "openai"


def test_generate_text_uses_openai_compatible_chat_completions():
    captured: dict = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {"choices": [{"message": {"content": "online summary"}}]}
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = generate_text(
            "Explain RSI",
            "Qwen/Qwen2.5-7B-Instruct",
            "https://api.siliconflow.cn/v1",
            provider="siliconflow",
            api_key="sk-test",
        )

    assert result == "online summary"
    assert captured["url"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["payload"]["messages"][0]["content"] == "Explain RSI"


def test_generate_text_uses_ollama_generate_endpoint():
    captured: dict = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"response": "local summary"})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = generate_text(
            "Explain RSI",
            "qwen2.5:7b",
            "http://localhost:11434",
            provider="ollama",
        )

    assert result == "local summary"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"]["prompt"] == "Explain RSI"
