"""pytest configuration for the research-agent test suite."""

import asyncio
import pytest


@pytest.fixture
def event_loop():
    """Function-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
