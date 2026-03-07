"""
Unit tests for topic_graph.py – TopicNode and TopicGraph.
"""

from __future__ import annotations

import json

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
