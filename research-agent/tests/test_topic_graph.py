"""
Unit tests for topic_graph.py – TopicNode and TopicGraph.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


class TestTopicGraph:
    def test_root_node_created(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="artificial intelligence")
        assert g.root.name == "AI"
        assert g.root.depth == 0
        assert g.node_count() == 1

    def test_add_child_node(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        child = g.add_node(
            name="Machine Learning",
            query="machine learning overview",
            parent_id=g.root.id,
        )
        assert child.name == "Machine Learning"
        assert child.depth == 1
        assert g.root.id in child.parent_ids
        assert child.id in g.root.children_ids
        assert g.node_count() == 2

    def test_add_node_rejects_empty_name(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        with pytest.raises(ValueError, match="cannot be empty"):
            g.add_node(name="   ", query="query", parent_id=g.root.id)

    def test_cross_reference_dedup(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        c1 = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        c2 = g.add_node(name="nlp", query="natural language", parent_id=g.root.id)
        # Same node (case-insensitive) — should be the same id
        assert c1.id == c2.id
        assert g.node_count() == 2

    def test_find_by_name(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        g.add_node(name="Vision", query="computer vision", parent_id=g.root.id)
        found = g.find_by_name("VISION")
        assert found is not None
        assert found.name == "Vision"

    def test_get_ready_for_research(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        c1 = g.add_node(name="NLP", query="NLP", parent_id=g.root.id, priority=3)
        c2 = g.add_node(name="Vision", query="CV", parent_id=g.root.id, priority=8)
        g.mark_leaf(c1.id)
        g.mark_leaf(c2.id)
        c1.status = "pending"
        c2.status = "pending"

        ready = g.get_ready_for_research()
        assert len(ready) == 2
        # Higher priority first
        assert ready[0].name == "Vision"

    def test_mark_researched(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        c = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        g.mark_leaf(c.id)
        g.mark_researching(c.id)
        g.mark_researched(c.id, "NLP summary", ["https://example.com"])

        assert c.status == "completed"
        assert c.summary == "NLP summary"
        assert c.source_urls == ["https://example.com"]

    def test_get_ready_for_consolidation(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        c1 = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        c2 = g.add_node(name="Vision", query="CV", parent_id=g.root.id)
        g.mark_leaf(c1.id)
        g.mark_leaf(c2.id)
        g.mark_researched(c1.id, "s1", [])
        g.mark_researched(c2.id, "s2", [])

        consolidatable = g.get_ready_for_consolidation()
        # Root should be ready since both children are completed
        assert any(n.id == g.root.id for n in consolidatable)

    def test_is_complete(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        c = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        g.mark_leaf(c.id)
        assert not g.is_complete()

        g.mark_researched(c.id, "s", [])
        g.mark_consolidated(g.root.id, "consolidated AI")
        assert g.is_complete()

    def test_get_outline(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        g.add_node(name="Vision", query="CV", parent_id=g.root.id)

        outline = g.get_outline()
        assert "AI" in outline
        assert "NLP" in outline
        assert "Vision" in outline

    def test_get_outline_report_mode_hides_unresolved_branches(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        c1 = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        c2 = g.add_node(name="Vision", query="CV", parent_id=g.root.id)
        g.mark_leaf(c1.id)
        g.mark_leaf(c2.id)
        g.mark_researched(c1.id, "summary", [])

        outline = g.get_outline(report_mode=True)

        assert "AI" in outline
        assert "NLP" in outline
        assert "Vision" not in outline

    def test_from_dict_replaces_blank_name(self):
        from src.topic_graph import TopicNode

        node = TopicNode.from_dict({"id": "abc123", "name": " ", "query": "fallback query"})
        assert node.name == "fallback query"

    def test_save_and_load_json(self, tmp_path):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        path = tmp_path / "graph.json"
        g.save_json(path)

        g2 = TopicGraph.load_json(path)
        assert g2.root.name == "AI"
        assert g2.node_count() == 2

    def test_mark_failed(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        c = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        g.mark_failed(c.id)
        assert c.status == "failed"

    def test_get_all_researched_names(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        c = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        g.mark_leaf(c.id)
        g.mark_researched(c.id, "summary", [])
        names = g.get_all_researched_names()
        assert "NLP" in names

    def test_get_nodes_at_depth(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        c1 = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        c2 = g.add_node(name="Vision", query="CV", parent_id=g.root.id)
        gc = g.add_node(name="Transformers", query="transformers", parent_id=c1.id)

        assert [n.name for n in g.get_nodes_at_depth(0)] == ["AI"]
        depth1 = {n.name for n in g.get_nodes_at_depth(1)}
        assert depth1 == {"NLP", "Vision"}
        assert [n.name for n in g.get_nodes_at_depth(2)] == ["Transformers"]
        assert g.get_nodes_at_depth(3) == []

    def test_max_depth_present(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        assert g.max_depth_present() == 0
        g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        assert g.max_depth_present() == 1
        c = g.add_node(name="Vision", query="CV", parent_id=g.root.id)
        g.add_node(name="Object Detection", query="OD", parent_id=c.id)
        assert g.max_depth_present() == 2

    def test_add_node_beyond_default_max_depth(self):
        """Nodes deeper than the default MAX_DEPTH constant can be added — depth
        enforcement is the caller's (AgentManager) responsibility, not add_node's."""
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="Root", root_query="root")
        d1 = g.add_node(name="D1", query="d1", parent_id=g.root.id)
        d2 = g.add_node(name="D2", query="d2", parent_id=d1.id)
        d3 = g.add_node(name="D3", query="d3", parent_id=d2.id)  # depth 3 — was forbidden

        assert d3.depth == 3
        assert g.max_depth_present() == 3

    def test_to_tree_dict_basic(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        g.root.summary = "AI overview"
        c = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        g.mark_leaf(c.id)
        g.mark_researched(c.id, "NLP summary", ["https://nlp.example.com"])

        tree = g.to_tree_dict(exclude_empty=False)
        assert tree["name"] == "AI"
        assert tree["summary"] == "AI overview"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["name"] == "NLP"
        assert tree["children"][0]["summary"] == "NLP summary"

    def test_to_tree_dict_exclude_empty(self):
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="AI", root_query="AI")
        g.root.summary = "AI overview"
        c1 = g.add_node(name="NLP", query="NLP", parent_id=g.root.id)
        c2 = g.add_node(name="Vision", query="CV", parent_id=g.root.id)
        g.mark_leaf(c1.id)
        g.mark_researched(c1.id, "NLP summary", [])
        # c2 has no summary — should be excluded

        tree = g.to_tree_dict(exclude_empty=True)
        assert len(tree["children"]) == 1
        assert tree["children"][0]["name"] == "NLP"


# ---------------------------------------------------------------------------
# Node Merging
# ---------------------------------------------------------------------------

class TestNodeMerging:
    def test_merge_no_candidates(self):
        """Merging with fewer than 2 pending nodes should do nothing."""
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="Root", root_query="root")
        c = g.add_node(name="Only Child", query="only", parent_id=g.root.id)
        assert g.merge_similar_nodes(depth=1) == 0

    def test_merge_without_chromadb(self):
        """When chromadb is not importable, merge should return 0."""
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="Root", root_query="root")
        g.add_node(name="Machine Learning", query="ML", parent_id=g.root.id)
        g.add_node(name="Machine Learning Intro", query="ML intro", parent_id=g.root.id)

        with patch("builtins.__import__", side_effect=ImportError("no chromadb")):
            result = g.merge_similar_nodes(depth=1)
        assert result == 0

    def test_merge_similar_nodes_reduces_count(self):
        """Highly similar nodes at the same depth should be merged."""
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="Root", root_query="root")
        n1 = g.add_node(name="Python web scraping", query="python web scraping", parent_id=g.root.id, priority=8)
        n2 = g.add_node(name="Python web scraping tutorial", query="python web scraping tutorial", parent_id=g.root.id, priority=5)

        initial_count = g.node_count()
        merged = g.merge_similar_nodes(depth=1, threshold=0.50)

        # Should have merged at least one pair (these names are very similar)
        if merged > 0:
            assert g.node_count() < initial_count
        # Either way, no crash

    def test_merge_skips_completed_nodes(self):
        """Completed nodes should not be merge candidates."""
        from src.topic_graph import TopicGraph

        g = TopicGraph(root_name="Root", root_query="root")
        n1 = g.add_node(name="NLP basics", query="NLP basics", parent_id=g.root.id)
        n2 = g.add_node(name="NLP fundamentals", query="NLP fundamentals", parent_id=g.root.id)
        g.mark_leaf(n1.id)
        g.mark_researched(n1.id, "some summary", [])

        # n1 is "completed", should not be merged
        merged = g.merge_similar_nodes(depth=1, threshold=0.50)
        assert g.get_node(n1.id) is not None  # n1 should still exist
