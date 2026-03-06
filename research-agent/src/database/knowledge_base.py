"""
knowledge_base.py – Persistent storage layer.

Combines:
* SQLite (via aiosqlite) for structured metadata records.
* ChromaDB (optional) for semantic-similarity deduplication.

If ChromaDB is unavailable, the system degrades gracefully to SQLite-only
and uses exact-string deduplication instead of vector similarity.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research (
    id          TEXT PRIMARY KEY,
    topic       TEXT NOT NULL,
    content     TEXT NOT NULL,
    code_snippet TEXT,
    source_url  TEXT,
    timestamp   TEXT NOT NULL
);
"""

_SIMILARITY_THRESHOLD = 0.85  # cosine-distance threshold for ChromaDB


class KnowledgeBase:
    """Async knowledge base backed by SQLite and (optionally) ChromaDB."""

    def __init__(self, db_path: str = "data/research.db") -> None:
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._chroma_collection: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Create tables and connect to storage backends."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(_SCHEMA)
        await self._db.commit()

        # Optional ChromaDB
        try:
            import chromadb  # optional dependency – graceful degradation if missing

            loop = asyncio.get_event_loop()
            client = await loop.run_in_executor(
                None,
                lambda: chromadb.PersistentClient(
                    path=str(Path(self._db_path).parent / "chroma")
                ),
            )
            self._chroma_collection = await loop.run_in_executor(
                None,
                lambda: client.get_or_create_collection("research"),
            )
            logger.info("ChromaDB vector store initialised.")
        except Exception as exc:
            logger.warning(
                "ChromaDB unavailable (%s); falling back to SQLite-only deduplication.",
                exc,
            )

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def save(
        self,
        topic: str,
        content: str,
        code_snippet: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> str:
        """Persist an approved research finding; returns its UUID."""
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        assert self._db is not None
        await self._db.execute(
            "INSERT INTO research (id, topic, content, code_snippet, source_url, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (record_id, topic, content, code_snippet, source_url, timestamp),
        )
        await self._db.commit()

        # Also store embedding in ChromaDB for deduplication
        if self._chroma_collection is not None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._chroma_collection.add(
                        documents=[content],
                        ids=[record_id],
                        metadatas=[{"topic": topic, "source_url": source_url or ""}],
                    ),
                )
            except Exception as exc:
                logger.warning("ChromaDB save failed: %s", exc)

        logger.info("Saved research record %s for topic %r.", record_id, topic)
        return record_id

    # ------------------------------------------------------------------
    # Read / Query
    # ------------------------------------------------------------------

    async def is_duplicate(self, content: str) -> bool:
        """Return *True* if semantically similar content already exists."""
        if self._chroma_collection is not None:
            try:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: self._chroma_collection.query(
                        query_texts=[content],
                        n_results=1,
                    ),
                )
                distances = results.get("distances", [[]])[0]
                if distances and distances[0] < (1 - _SIMILARITY_THRESHOLD):
                    return True
            except Exception as exc:
                logger.warning("ChromaDB query failed: %s", exc)

        # Fallback: substring deduplication
        assert self._db is not None
        snippet = content[:200]
        async with self._db.execute(
            "SELECT 1 FROM research WHERE content LIKE ? LIMIT 1",
            (f"%{snippet}%",),
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None

    async def get_all_topics(self) -> list[str]:
        """Return a deduplicated list of all researched topics."""
        assert self._db is not None
        async with self._db.execute("SELECT DISTINCT topic FROM research") as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recently saved records."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT id, topic, content, code_snippet, source_url, timestamp "
            "FROM research ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
        columns = ["id", "topic", "content", "code_snippet", "source_url", "timestamp"]
        return [dict(zip(columns, row)) for row in rows]
