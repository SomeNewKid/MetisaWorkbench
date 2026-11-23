"""Tests for model alias routing and broker errors."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError

from metisa_harness.harness import AgentHarness
from metisa_harness.models import AgentDefinition

_BROKER_BASE_URL = "http://metisa-llm-broker:8000/v1"


@pytest.mark.parametrize("model_alias", ["granite-4-1b", "gpt-4o"])
def test_run_uses_broker_for_configured_model_alias(
    agent: AgentDefinition,
    broker_endpoint: tuple[MagicMock, AsyncMock],
    http_client_factory: MagicMock,
    model_alias: str,
) -> None:
    """Local and online aliases go unchanged to the same broker endpoint."""
    factory, client = broker_endpoint
    definition = replace(agent, model=model_alias)

    result = asyncio.run(
        AgentHarness.run(definition, run_identifier="run-test", input="Hi")
    )

    assert result.final_output == "Broker answer."
    factory.assert_called_once()
    assert factory.call_args.kwargs["base_url"] == _BROKER_BASE_URL
    assert factory.call_args.kwargs["http_client"] is http_client_factory.return_value
    http_client_factory.assert_called_once_with(trust_env=False)
    client.models.list.assert_not_awaited()
    client.chat.completions.create.assert_awaited_once()
    assert client.chat.completions.create.await_args.kwargs["model"] == model_alias


@pytest.mark.parametrize("status_code", [None, 404, 503])
def test_run_propagates_broker_completion_error(
    agent: AgentDefinition,
    broker_endpoint: tuple[MagicMock, AsyncMock],
    status_code: int | None,
) -> None:
    """Broker connection and HTTP errors propagate without provider fallback."""
    factory, client = broker_endpoint
    request = httpx2.Request("POST", f"{_BROKER_BASE_URL}/chat/completions")
    error: APIConnectionError | APIStatusError
    if status_code is None:
        error = APIConnectionError(request=request)
    else:
        error = APIStatusError(
            "Broker request failed.",
            response=httpx2.Response(status_code, request=request),
            body=None,
        )
    client.chat.completions.create.side_effect = error

    with pytest.raises(type(error)) as caught:
        asyncio.run(AgentHarness.run(agent, run_identifier="run-test", input="Hi"))

    assert caught.value is error
    factory.assert_called_once()
    assert factory.call_args.kwargs["base_url"] == _BROKER_BASE_URL
    client.models.list.assert_not_awaited()
    client.chat.completions.create.assert_awaited_once()
