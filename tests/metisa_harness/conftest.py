"""Shared fixtures for the Metisa Harness test modules."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metisa_harness.models import (
    AgentDefinition,
)

from ._harness_helpers import (
    _completion,
)


@pytest.fixture
def agent() -> AgentDefinition:
    """Provide a valid agent definition for execution tests."""
    return AgentDefinition(
        name="Assistant",
        description="Test agent",
        model="test-model",
        instructions="Answer briefly.",
    )


@pytest.fixture(autouse=True)
def http_client_factory() -> Iterator[MagicMock]:
    """Prevent real HTTP clients from being opened and expose their configuration."""
    with patch("metisa_harness._chat_completions.httpx2.AsyncClient") as factory:
        yield factory


@pytest.fixture
def broker_endpoint() -> Iterator[tuple[MagicMock, AsyncMock]]:
    """Provide a mocked SDK client for broker completions."""
    client = AsyncMock()
    client.chat.completions.create.return_value = _completion("Broker answer.")

    with patch("metisa_harness._chat_completions.AsyncOpenAI") as factory:
        factory.return_value.__aenter__.return_value = client
        yield factory, client
