"""
Unit tests for KnowledgeBase (SQLite-only; ChromaDB mocked out).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_research.db")


class TestKnowledgeBase:
    def test_init_creates_schema(self, event_loop, tmp_db):
        from src.database.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(kb.init())

        # DB file should exist
        assert Path(tmp_db).exists()
        event_loop.run_until_complete(kb.close())

    def test_save_and_get_recent(self, event_loop, tmp_db):
        from src.database.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(kb.init())

        record_id = event_loop.run_until_complete(
            kb.save(
                topic="RSI",
                content="RSI technical summary",
                code_snippet="rs = avg_gain / avg_loss",
                source_url="https://example.com",
            )
        )

        assert record_id  # should be a non-empty UUID string
        records = event_loop.run_until_complete(kb.get_recent(limit=5))
        assert len(records) == 1
        assert records[0]["topic"] == "RSI"
        assert records[0]["content"] == "RSI technical summary"

        event_loop.run_until_complete(kb.close())

    def test_get_all_topics(self, event_loop, tmp_db):
        from src.database.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(kb.init())

        event_loop.run_until_complete(kb.save("RSI", "content A"))
        event_loop.run_until_complete(kb.save("MACD", "content B"))
        event_loop.run_until_complete(kb.save("RSI", "content C"))

        topics = event_loop.run_until_complete(kb.get_all_topics())
        assert sorted(topics) == ["MACD", "RSI"]

        event_loop.run_until_complete(kb.close())

    def test_is_duplicate_false_for_new_content(self, event_loop, tmp_db):
        from src.database.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(kb.init())

        is_dup = event_loop.run_until_complete(
            kb.is_duplicate("completely new content")
        )
        assert is_dup is False

        event_loop.run_until_complete(kb.close())

    def test_is_duplicate_true_for_saved_content(self, event_loop, tmp_db):
        from src.database.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(db_path=tmp_db)
        event_loop.run_until_complete(kb.init())

        content = "RSI (Relative Strength Index) is a momentum oscillator"
        event_loop.run_until_complete(kb.save("RSI", content))

        is_dup = event_loop.run_until_complete(
            kb.is_duplicate(content[:200])
        )
        assert is_dup is True

        event_loop.run_until_complete(kb.close())
