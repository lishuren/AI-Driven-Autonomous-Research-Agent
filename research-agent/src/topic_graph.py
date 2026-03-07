"""
topic_graph.py – Hierarchical topic graph for recursive research.

Provides:
* ``TopicNode`` – A single node in the research topic hierarchy.
* ``TopicGraph`` – A DAG (directed acyclic graph) that tracks topics,
  sub-topics, cross-references, research status, and consolidation.

The graph supports recursive decomposition up to ``MAX_DEPTH`` levels,
deduplicates topics by name (case-insensitive), and provides traversal
helpers for the research orchestration loop.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Valid node statuses (progression order)
_VALID_STATUSES = {"pending", "analyzing", "researching", "completed", "consolidated", "failed"}

MAX_DEPTH = 5


@dataclass
class TopicNode:
    """A single node in the research topic hierarchy."""

    id: str
    name: str
    query: str
    depth: int = 0
    parent_ids: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)
    status: str = "pending"
    priority: int = 5
    summary: Optional[str] = None
    consolidated_summary: Optional[str] = None
    source_urls: list[str] = field(default_factory=list)
    is_leaf: bool = False
    retry_count: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON persistence."""
        return {
            "id": self.id,
            "name": self.name,
            "query": self.query,
            "depth": self.depth,
            "parent_ids": list(self.parent_ids),
            "children_ids": list(self.children_ids),
            "status": self.status,
            "priority": self.priority,
            "summary": self.summary,
            "consolidated_summary": self.consolidated_summary,
            "source_urls": list(self.source_urls),
            "is_leaf": self.is_leaf,
            "retry_count": self.retry_count,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopicNode":
        """Deserialize from a plain dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            query=data["query"],
            depth=data.get("depth", 0),
            parent_ids=list(data.get("parent_ids", [])),
            children_ids=list(data.get("children_ids", [])),
            status=data.get("status", "pending"),
            priority=data.get("priority", 5),
            summary=data.get("summary"),
            consolidated_summary=data.get("consolidated_summary"),
            source_urls=list(data.get("source_urls", [])),
            is_leaf=data.get("is_leaf", False),
            retry_count=data.get("retry_count", 0),
            description=data.get("description", ""),
        )


class TopicGraph:
    """A directed acyclic graph of research topics.

    Supports recursive decomposition, cross-reference deduplication,
    status tracking, and BFS traversal.
    """

    def __init__(self, root_name: str, root_query: str = "") -> None:
        self._nodes: dict[str, TopicNode] = {}
        root = TopicNode(
            id=str(uuid.uuid4()),
            name=root_name,
            query=root_query or root_name,
            depth=0,
        )
        self._nodes[root.id] = root
        self._root_id: str = root.id

    @property
    def root(self) -> TopicNode:
        return self._nodes[self._root_id]

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        query: str,
        parent_id: str,
        priority: int = 5,
        depth: Optional[int] = None,
        description: str = "",
    ) -> TopicNode:
        """Create a child node under *parent_id*, or add a cross-reference edge.

        If a node with the same name (case-insensitive) already exists,
        a cross-reference edge is added from *parent_id* to the existing
        node instead of creating a duplicate. Returns the (possibly
        pre-existing) node.

        Raises ``ValueError`` if *parent_id* is unknown or *depth* exceeds
        ``MAX_DEPTH``.
        """
        parent = self._nodes.get(parent_id)
        if parent is None:
            raise ValueError(f"Unknown parent node: {parent_id!r}")

        node_depth = depth if depth is not None else parent.depth + 1
        if node_depth > MAX_DEPTH:
            raise ValueError(
                f"Depth {node_depth} exceeds MAX_DEPTH ({MAX_DEPTH}) "
                f"for node {name!r}"
            )

        # Cross-reference deduplication: existing node with same name?
        existing = self.find_by_name(name)
        if existing is not None:
            # Add cross-reference edge if not already linked
            if parent_id not in existing.parent_ids:
                existing.parent_ids.append(parent_id)
            if existing.id not in parent.children_ids:
                parent.children_ids.append(existing.id)
            logger.info(
                "Cross-reference: linked existing node %r under parent %r.",
                name, parent.name,
            )
            return existing

        node = TopicNode(
            id=str(uuid.uuid4()),
            name=name,
            query=query,
            depth=node_depth,
            parent_ids=[parent_id],
            priority=priority,
            description=description,
        )
        self._nodes[node.id] = node
        parent.children_ids.append(node.id)
        return node

    def get_node(self, node_id: str) -> Optional[TopicNode]:
        return self._nodes.get(node_id)

    def find_by_name(self, name: str) -> Optional[TopicNode]:
        """Find a node by case-insensitive name match."""
        lower = name.lower().strip()
        for node in self._nodes.values():
            if node.name.lower().strip() == lower:
                return node
        return None

    def get_children(self, node_id: str) -> list[TopicNode]:
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def mark_analyzing(self, node_id: str) -> None:
        node = self._nodes[node_id]
        node.status = "analyzing"

    def mark_researching(self, node_id: str) -> None:
        node = self._nodes[node_id]
        node.status = "researching"

    def mark_researched(
        self,
        node_id: str,
        summary: str,
        source_urls: Optional[list[str]] = None,
    ) -> None:
        node = self._nodes[node_id]
        node.summary = summary
        node.source_urls = source_urls or []
        node.status = "completed"

    def mark_consolidated(self, node_id: str, consolidated_summary: str) -> None:
        node = self._nodes[node_id]
        node.consolidated_summary = consolidated_summary
        node.status = "consolidated"

    def mark_leaf(self, node_id: str) -> None:
        node = self._nodes[node_id]
        node.is_leaf = True

    def mark_failed(self, node_id: str) -> None:
        node = self._nodes[node_id]
        node.status = "failed"

    def increment_retry(self, node_id: str) -> int:
        node = self._nodes[node_id]
        node.retry_count += 1
        return node.retry_count

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_ready_for_research(self) -> list[TopicNode]:
        """Return leaf nodes with status ``'pending'``, sorted by priority descending."""
        return sorted(
            [
                n for n in self._nodes.values()
                if n.is_leaf and n.status == "pending"
            ],
            key=lambda n: n.priority,
            reverse=True,
        )

    def get_ready_for_consolidation(self) -> list[TopicNode]:
        """Return non-leaf nodes whose children are all completed/consolidated.

        Nodes already consolidated or with no children are excluded.
        """
        ready = []
        for node in self._nodes.values():
            if node.is_leaf:
                continue
            if node.status in ("consolidated",):
                continue
            children = self.get_children(node.id)
            if not children:
                continue
            if all(c.status in ("completed", "consolidated", "failed") for c in children):
                ready.append(node)
        return sorted(ready, key=lambda n: n.depth, reverse=True)  # deepest first

    def is_complete(self) -> bool:
        """Return True when the root node has been consolidated."""
        return self.root.status == "consolidated"

    def get_all_researched_names(self) -> list[str]:
        """Return names of all completed or consolidated nodes (for dedup)."""
        return [
            n.name for n in self._nodes.values()
            if n.status in ("completed", "consolidated")
        ]

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_all_nodes(self) -> list[TopicNode]:
        """Return all nodes in BFS order from root."""
        visited: list[TopicNode] = []
        queue: deque[str] = deque([self._root_id])
        seen: set[str] = set()
        while queue:
            nid = queue.popleft()
            if nid in seen or nid not in self._nodes:
                continue
            seen.add(nid)
            node = self._nodes[nid]
            visited.append(node)
            for cid in node.children_ids:
                if cid not in seen:
                    queue.append(cid)
        return visited

    def node_count(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Outline / Display
    # ------------------------------------------------------------------

    def get_outline(self) -> str:
        """Return an indented text outline of the graph for report headers."""
        lines: list[str] = []
        self._outline_recurse(self._root_id, 0, lines, set())
        return "\n".join(lines)

    def _outline_recurse(
        self, node_id: str, indent: int, lines: list[str], seen: set[str]
    ) -> None:
        if node_id in seen or node_id not in self._nodes:
            return
        seen.add(node_id)
        node = self._nodes[node_id]
        prefix = "  " * indent + "- "
        status_tag = f" [{node.status}]"
        lines.append(f"{prefix}{node.name}{status_tag}")
        for cid in node.children_ids:
            self._outline_recurse(cid, indent + 1, lines, seen)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self._root_id,
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopicGraph":
        """Restore a graph from a serialized dict."""
        root_id = data["root_id"]
        nodes_data = data["nodes"]
        # Create a minimal instance and populate manually
        root_data = nodes_data[root_id]
        graph = cls.__new__(cls)
        graph._nodes = {}
        graph._root_id = root_id
        for nid, nd in nodes_data.items():
            graph._nodes[nid] = TopicNode.from_dict(nd)
        return graph

    def save_json(self, path: Path) -> None:
        """Persist the graph to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Topic graph saved to %s", path)

    @classmethod
    def load_json(cls, path: Path) -> "TopicGraph":
        """Load a graph from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
