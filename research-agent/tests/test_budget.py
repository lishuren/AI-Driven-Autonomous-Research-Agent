"""
Unit tests for BudgetTracker (src/budget.py).
"""

from __future__ import annotations

import pytest

from src.budget import BudgetTracker


class TestBudgetTracker:
    # ------------------------------------------------------------------
    # Basic construction
    # ------------------------------------------------------------------

    def test_unlimited_budget(self):
        bt = BudgetTracker()
        assert bt.can_query()
        assert bt.can_create_node()
        assert not bt.is_exhausted()
        assert bt.remaining_credits() is None
        assert bt.budget_fraction_remaining() == 1.0

    def test_summary_initial(self):
        bt = BudgetTracker(max_queries=10, max_nodes=20, max_credits=100.0)
        s = bt.summary()
        assert s["queries_used"] == 0
        assert s["max_queries"] == 10
        assert s["nodes_created"] == 0
        assert s["max_nodes"] == 20
        assert s["credits_used"] == 0
        assert s["max_credits"] == 100.0
        assert s["budget_fraction_remaining"] == 1.0

    # ------------------------------------------------------------------
    # Query tracking
    # ------------------------------------------------------------------

    def test_record_query(self):
        bt = BudgetTracker(max_queries=3)
        bt.record_query(credits=1)
        assert bt.queries_used == 1
        assert bt.credits_used == 1.0
        assert bt.can_query()

    def test_query_limit_reached(self):
        bt = BudgetTracker(max_queries=2)
        bt.record_query()
        bt.record_query()
        assert not bt.can_query()
        assert bt.is_exhausted()

    def test_credit_limit_reached(self):
        bt = BudgetTracker(max_credits=5.0)
        bt.record_query(credits=3.0)
        assert bt.can_query()
        bt.record_query(credits=3.0)
        assert not bt.can_query()
        assert bt.is_exhausted()

    def test_remaining_credits(self):
        bt = BudgetTracker(max_credits=10.0)
        bt.record_query(credits=4.0)
        assert bt.remaining_credits() == 6.0
        bt.record_query(credits=7.0)
        assert bt.remaining_credits() == 0.0

    # ------------------------------------------------------------------
    # Node tracking
    # ------------------------------------------------------------------

    def test_record_node(self):
        bt = BudgetTracker(max_nodes=5)
        for _ in range(4):
            bt.record_node()
        assert bt.can_create_node()
        bt.record_node()
        assert not bt.can_create_node()

    def test_node_limit_no_effect_on_query(self):
        bt = BudgetTracker(max_nodes=1)
        bt.record_node()
        assert not bt.can_create_node()
        assert bt.can_query()  # query budget is separate

    # ------------------------------------------------------------------
    # Budget fraction
    # ------------------------------------------------------------------

    def test_budget_fraction_queries(self):
        bt = BudgetTracker(max_queries=10)
        bt.record_query()
        bt.record_query()
        assert bt.budget_fraction_remaining() == pytest.approx(0.8)

    def test_budget_fraction_multiple_limits(self):
        bt = BudgetTracker(max_queries=10, max_credits=100.0)
        bt.record_query(credits=80.0)
        # queries: 9/10 remaining = 0.9
        # credits: 20/100 remaining = 0.2
        # min(0.9, 0.2) = 0.2
        assert bt.budget_fraction_remaining() == pytest.approx(0.2)

    def test_budget_fraction_unlimited(self):
        bt = BudgetTracker()
        bt.record_query(credits=999)
        assert bt.budget_fraction_remaining() == 1.0

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_zero_limits(self):
        bt = BudgetTracker(max_queries=0, max_credits=0.0, max_nodes=0)
        assert not bt.can_query()
        assert not bt.can_create_node()
        assert bt.is_exhausted()

    # ------------------------------------------------------------------
    # Warn threshold
    # ------------------------------------------------------------------

    def test_approaching_limit_false_when_unlimited(self):
        """Unlimited budget should never approach a limit."""
        bt = BudgetTracker()
        bt.record_query(credits=999)
        assert not bt.approaching_limit()

    def test_approaching_limit_below_threshold(self):
        bt = BudgetTracker(max_credits=100.0, warn_threshold=0.80)
        bt.record_query(credits=50.0)  # 50% used — below 80%
        assert not bt.approaching_limit()

    def test_approaching_limit_at_threshold(self):
        bt = BudgetTracker(max_credits=100.0, warn_threshold=0.80)
        bt.record_query(credits=80.0)  # exactly at 80%
        assert bt.approaching_limit()

    def test_approaching_limit_above_threshold(self):
        bt = BudgetTracker(max_credits=100.0, warn_threshold=0.80)
        bt.record_query(credits=90.0)
        assert bt.approaching_limit()

    def test_warn_threshold_in_summary(self):
        bt = BudgetTracker(max_credits=100.0, warn_threshold=0.75)
        s = bt.summary()
        assert s["warn_threshold"] == 0.75

    def test_warn_logged_once(self, caplog):
        """Warning is logged exactly once when the threshold is crossed."""
        import logging
        bt = BudgetTracker(max_credits=100.0, warn_threshold=0.80)
        with caplog.at_level(logging.WARNING, logger="src.budget"):
            bt.record_query(credits=85.0)   # crosses 80% → warning
            bt.record_query(credits=5.0)    # already warned — no second warning
        warn_records = [r for r in caplog.records if "Approaching budget limit" in r.message]
        assert len(warn_records) == 1

    def test_no_warn_when_threshold_is_one(self, caplog):
        """Threshold of 1.0 disables the warning."""
        import logging
        bt = BudgetTracker(max_credits=100.0, warn_threshold=1.0)
        with caplog.at_level(logging.WARNING, logger="src.budget"):
            bt.record_query(credits=99.0)
        warn_records = [r for r in caplog.records if "Approaching budget limit" in r.message]
        assert len(warn_records) == 0
