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

MAX_DEPTH = 2


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

        Raises ``ValueError`` if *parent_id* is unknown.

        Depth is not enforced here — the caller (``AgentManager._decompose_node``)
        is responsible for honouring ``self._max_depth``.
        """
        parent = self._nodes.get(parent_id)
        if parent is None:
            raise ValueError(f"Unknown parent node: {parent_id!r}")

        node_depth = depth if depth is not None else parent.depth + 1

        # Cross-reference deduplication: existing node with same name?
        existing = self.find_by_name(name)
        if existing is not None:
            # Guard: skip if this would create a self-loop or a cycle.
            # A cycle occurs when parent_id is already reachable from existing
            # (i.e. existing is an ancestor of parent).
            if existing.id == parent_id or self._is_ancestor(existing.id, parent_id):
                logger.warning(
                    "Cross-reference skipped: linking %r under %r would create a cycle.",
                    name, parent.name,
                )
                return existing
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

    def _is_ancestor(self, candidate_id: str, node_id: str) -> bool:
        """Return True if *candidate_id* is an ancestor of *node_id* (BFS)."""
        visited: set[str] = set()
        queue = list(self._nodes[node_id].parent_ids) if node_id in self._nodes else []
        while queue:
            pid = queue.pop()
            if pid == candidate_id:
                return True
            if pid in visited:
                continue
            visited.add(pid)
            if pid in self._nodes:
                queue.extend(self._nodes[pid].parent_ids)
        return False

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

    def get_nodes_at_depth(self, depth: int) -> list[TopicNode]:
        """Return all nodes at a specific depth level."""
        return [n for n in self._nodes.values() if n.depth == depth]

    def max_depth_present(self) -> int:
        """Return the maximum depth of any node currently in the graph."""
        if not self._nodes:
            return 0
        return max(n.depth for n in self._nodes.values())

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

    def to_tree_dict(self, exclude_empty: bool = True) -> dict[str, Any]:
        """Return a hierarchical tree dict rooted at the root node.

        When *exclude_empty* is True (default), nodes with no summary and
        no consolidated_summary are omitted from the output.
        """
        return self._node_to_tree(self._root_id, set(), exclude_empty)

    def _node_to_tree(
        self, node_id: str, seen: set[str], exclude_empty: bool,
    ) -> Optional[dict[str, Any]]:
        if node_id in seen or node_id not in self._nodes:
            return None
        seen.add(node_id)
        node = self._nodes[node_id]

        has_content = bool(node.summary or node.consolidated_summary)

        children: list[dict[str, Any]] = []
        for cid in node.children_ids:
            child_tree = self._node_to_tree(cid, seen, exclude_empty)
            if child_tree is not None:
                children.append(child_tree)

        # Skip this node if it has no content and no children with content
        if exclude_empty and not has_content and not children:
            return None

        entry: dict[str, Any] = {
            "name": node.name,
            "depth": node.depth,
            "status": node.status,
        }
        if node.summary:
            entry["summary"] = node.summary
        if node.consolidated_summary:
            entry["consolidated_summary"] = node.consolidated_summary
        if node.source_urls:
            entry["source_urls"] = node.source_urls
        if children:
            entry["children"] = children
        return entry

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

    # ------------------------------------------------------------------
    # Node Merging by Semantic Similarity
    # ------------------------------------------------------------------

    def merge_similar_nodes(
        self,
        threshold: float = 0.85,
        depth: Optional[int] = None,
    ) -> int:
        """Merge pending nodes that are semantically similar.

        Nodes whose ``name + " " + query`` text are cosine-similar above
        *threshold* (using ChromaDB embeddings) are merged: the
        higher-priority node absorbs the other.

        Only nodes at the given *depth* (or all pending nodes if depth is
        ``None``) are considered. Already-completed or failed nodes are
        skipped.

        Returns the number of merges performed.
        """
        try:
            import chromadb
        except ImportError:
            logger.info("ChromaDB not available — skipping node merge.")
            return 0

        # Collect candidate pending nodes
        candidates = [
            n for n in self._nodes.values()
            if n.status == "pending"
            and (depth is None or n.depth == depth)
        ]
        if len(candidates) < 2:
            return 0

        # Build texts for embedding
        texts = [f"{n.name} {n.query}" for n in candidates]
        ids = [n.id for n in candidates]

        try:
            client = chromadb.EphemeralClient()
            col = client.get_or_create_collection(
                "node_merge", metadata={"hnsw:space": "cosine"},
            )
            col.add(documents=texts, ids=ids)
        except Exception as exc:
            logger.warning("Node merge embedding failed: %s", exc)
            return 0

        merged: set[str] = set()
        merge_count = 0

        for i, node in enumerate(candidates):
            if node.id in merged:
                continue
            try:
                results = col.query(query_texts=[texts[i]], n_results=5)
            except Exception:
                continue

            result_ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for rid, dist in zip(result_ids, distances):
                if rid == node.id or rid in merged:
                    continue
                similarity = 1.0 - dist  # cosine distance → similarity
                if similarity < threshold:
                    continue

                other = self._nodes.get(rid)
                if other is None or other.status != "pending":
                    continue

                # Merge: keep the higher-priority node
                keep, drop = (node, other) if node.priority >= other.priority else (other, node)

                # Redirect parents of dropped node
                for pid in drop.parent_ids:
                    parent = self._nodes.get(pid)
                    if parent is not None:
                        if drop.id in parent.children_ids:
                            parent.children_ids.remove(drop.id)
                        if keep.id not in parent.children_ids:
                            parent.children_ids.append(keep.id)
                        if pid not in keep.parent_ids:
                            keep.parent_ids.append(pid)

                # Combine descriptions
                if drop.description and drop.description not in (keep.description or ""):
                    keep.description = f"{keep.description or ''}; {drop.description}"

                # Remove the dropped node
                del self._nodes[drop.id]
                merged.add(drop.id)
                merge_count += 1
                logger.info(
                    "Merged node %r into %r (similarity %.2f).",
                    drop.name, keep.name, similarity,
                )

        # Clean up ephemeral collection
        try:
            client.delete_collection("node_merge")
        except Exception:
            pass

        return merge_count
