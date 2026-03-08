"""
budget.py – Per-session budget tracker for API credits and node creation.

Provides a simple in-memory counter that the research loop checks before
making Tavily API calls or creating new topic-graph nodes.  When any
configured limit is reached the tracker signals exhaustion so the main
loop can stop gracefully.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BudgetTracker:
    """In-memory per-session budget tracker.

    Parameters
    ----------
    max_queries : int or None
        Maximum number of search queries allowed (None = unlimited).
    max_nodes : int or None
        Maximum number of topic-graph nodes that may be created (None = unlimited).
    max_credits : float or None
        Maximum Tavily API credits to spend (None = unlimited).
    warn_threshold : float
        Fraction of any limit at which a WARNING is logged (default 0.80 = 80%).
        Set to 1.0 to disable warnings.
    """

    def __init__(
        self,
        max_queries: Optional[int] = None,
        max_nodes: Optional[int] = None,
        max_credits: Optional[float] = None,
        warn_threshold: float = 0.80,
    ) -> None:
        self._max_queries = max_queries
        self._max_nodes = max_nodes
        self._max_credits = max_credits
        self._warn_threshold = warn_threshold

        self._queries_used: int = 0
        self._nodes_created: int = 0
        self._credits_used: float = 0.0
        self._limit_warning_logged: bool = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_query(self, credits: float = 1.0) -> None:
        """Record one search query and the credits it consumed."""
        self._queries_used += 1
        self._credits_used += credits
        if self._max_credits is not None and self._credits_used >= self._max_credits:
            logger.warning(
                "Credit budget exhausted: %.1f / %.1f credits used.",
                self._credits_used, self._max_credits,
            )
        if self._max_queries is not None and self._queries_used >= self._max_queries:
            logger.warning(
                "Query budget exhausted: %d / %d queries used.",
                self._queries_used, self._max_queries,
            )
        # Warn once when approaching the limit
        if not self._limit_warning_logged and self.approaching_limit():
            self._limit_warning_logged = True
            frac_used = 1.0 - self.budget_fraction_remaining()
            logger.warning(
                "Approaching budget limit (%.0f%% used) — %d queries, "
                "%.1f credits consumed so far.",
                frac_used * 100,
                self._queries_used,
                self._credits_used,
            )

    def record_node(self) -> None:
        """Record one new topic-graph node."""
        self._nodes_created += 1
        if self._max_nodes is not None and self._nodes_created >= self._max_nodes:
            logger.warning(
                "Node budget exhausted: %d / %d nodes created.",
                self._nodes_created, self._max_nodes,
            )

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def can_query(self) -> bool:
        """Return True if another search query is allowed."""
        if self._max_queries is not None and self._queries_used >= self._max_queries:
            return False
        if self._max_credits is not None and self._credits_used >= self._max_credits:
            return False
        return True

    def can_create_node(self) -> bool:
        """Return True if another topic-graph node may be created."""
        if self._max_nodes is not None and self._nodes_created >= self._max_nodes:
            return False
        return True

    def is_exhausted(self) -> bool:
        """Return True if any budget limit has been reached."""
        return not self.can_query()

    def approaching_limit(self) -> bool:
        """Return True when budget usage is at or above the warn threshold.

        Always returns False when no limits are configured (unlimited budget).
        """
        frac_remaining = self.budget_fraction_remaining()
        if frac_remaining == 1.0 and self._max_queries is None and self._max_credits is None:
            return False  # truly unlimited — nothing to warn about
        return (1.0 - frac_remaining) >= self._warn_threshold

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def queries_used(self) -> int:
        return self._queries_used

    @property
    def nodes_created(self) -> int:
        return self._nodes_created

    @property
    def credits_used(self) -> float:
        return self._credits_used

    def remaining_credits(self) -> Optional[float]:
        """Return remaining credits, or None if unlimited."""
        if self._max_credits is None:
            return None
        return max(0.0, self._max_credits - self._credits_used)

    def budget_fraction_remaining(self) -> float:
        """Return the fraction of budget remaining (0.0–1.0).

        When no limits are configured returns 1.0 (fully available).
        When multiple limits exist, returns the *minimum* fraction.
        """
        fractions: list[float] = []
        if self._max_queries is not None and self._max_queries > 0:
            fractions.append(
                max(0.0, 1.0 - self._queries_used / self._max_queries)
            )
        if self._max_credits is not None and self._max_credits > 0:
            fractions.append(
                max(0.0, 1.0 - self._credits_used / self._max_credits)
            )
        if self._max_nodes is not None and self._max_nodes > 0:
            fractions.append(
                max(0.0, 1.0 - self._nodes_created / self._max_nodes)
            )
        return min(fractions) if fractions else 1.0

    def summary(self) -> dict:
        """Return a dict summarising current budget usage."""
        return {
            "queries_used": self._queries_used,
            "max_queries": self._max_queries,
            "nodes_created": self._nodes_created,
            "max_nodes": self._max_nodes,
            "credits_used": round(self._credits_used, 2),
            "max_credits": self._max_credits,
            "budget_fraction_remaining": round(self.budget_fraction_remaining(), 3),
            "warn_threshold": self._warn_threshold,
        }
