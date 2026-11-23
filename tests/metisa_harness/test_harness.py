"""Tests for the agent loop, model injection, and multi-turn behavior."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metisa_harness.harness import AgentHarness
from metisa_harness.models import (
    AgentDefinition,
    LlmClient,
    LlmMessage,
    RunResult,
    ToolCall,
)

from ._harness_helpers import (
    _answer,
    _completion,
    _local_tool,
    _request,
    _ScriptedLlm,
    _server,
)


@pytest.mark.parametrize("response", ["Hello, Mary.", " Answer.\n"])
def test_run_uses_supplied_llm(agent: AgentDefinition, response: str) -> None:
    """A single-turn client receives messages and returns an assistant message."""
    llm = _ScriptedLlm(_answer(response))

    with patch("metisa_harness.harness.get_llm") as resolver:
        result = asyncio.run(
            AgentHarness.run(
                agent, run_identifier="run-test", input="Hello.", llm=llm, max_turns=1
            )
        )

    resolver.assert_not_called()
    assert llm.calls == [([LlmMessage(role="user", content="Hello.")], [])]
    assert result == RunResult(
        final_output=response,
        messages=(LlmMessage(role="user", content="Hello."), _answer(response)),
    )


def test_run_resolves_llm_when_not_supplied(agent: AgentDefinition) -> None:
    """Client discovery receives the agent when no client is injected."""
    llm = _ScriptedLlm(_answer("Resolved response."))

    with patch("metisa_harness.harness.get_llm", return_value=llm) as resolver:
        result = asyncio.run(
            AgentHarness.run(agent, run_identifier="run-test", input="Hello.")
        )

    resolver.assert_awaited_once_with(agent)
    assert llm.calls == [([LlmMessage(role="user", content="Hello.")], [])]
    assert result.final_output == "Resolved response."


def test_run_raises_when_llm_unavailable(agent: AgentDefinition) -> None:
    """Missing client discovery produces a clear error."""
    with patch("metisa_harness.harness.get_llm", return_value=None):
        with pytest.raises(RuntimeError, match="^LLM not available\\.$"):
            asyncio.run(
                AgentHarness.run(agent, run_identifier="run-test", input="Hello.")
            )


def test_run_propagates_llm_error(agent: AgentDefinition) -> None:
    """Client errors propagate rather than producing a successful result."""
    error = ConnectionError("Broker unavailable.")
    llm = _ScriptedLlm(error)

    with pytest.raises(ConnectionError) as caught:
        asyncio.run(
            AgentHarness.run(agent, run_identifier="run-test", input="Hello.", llm=llm)
        )

    assert caught.value is error
    assert llm.calls == [([LlmMessage(role="user", content="Hello.")], [])]


@pytest.mark.parametrize("content", [None, "", " \t\n"])
def test_run_rejects_missing_final_text(
    agent: AgentDefinition, content: str | None
) -> None:
    """Missing or blank text without tool calls is not a completed run."""
    llm = _ScriptedLlm(LlmMessage(role="assistant", content=content))

    with pytest.raises(RuntimeError, match="^The model returned no final text\\.$"):
        asyncio.run(
            AgentHarness.run(agent, run_identifier="run-test", input="Hello.", llm=llm)
        )


@pytest.mark.parametrize("max_turns", [0, -1])
def test_run_rejects_invalid_turn_limit(agent: AgentDefinition, max_turns: int) -> None:
    """An invalid budget is rejected before discovery or model requests."""
    llm = _ScriptedLlm()
    server = _server()
    definition = replace(agent, mcp_clients=[server])

    with pytest.raises(ValueError, match="^max_turns must be at least 1\\.$"):
        asyncio.run(
            AgentHarness.run(
                definition,
                run_identifier="run-test",
                input="Hello.",
                llm=llm,
                max_turns=max_turns,
            )
        )

    assert llm.calls == []
    server.list_tools.assert_not_awaited()


def test_run_preserves_history_across_batches_and_repeated_tool_calls(
    agent: AgentDefinition,
) -> None:
    """Multiple calls and later turns retain order, bindings, text, and call IDs."""
    events: list[str] = []
    first = _local_tool("first", {"value": 10}, events)
    second = _local_tool("second", {"value": 20}, events)
    batch = LlmMessage(
        role="assistant",
        content="Checking both sources.",
        tool_calls=[
            ToolCall(id="a", name="first", arguments='{"value": 1}'),
            ToolCall(id="b", name="second", arguments='{"value": 2}'),
        ],
    )
    followup = _request(ToolCall(id="c", name="first", arguments='{"value": 3}'))
    llm = _ScriptedLlm(batch, followup, _answer("Done."))
    definition = replace(agent, tools=[first, second])

    result = asyncio.run(
        AgentHarness.run(
            definition,
            run_identifier="run-test",
            input="Compare.",
            llm=llm,
            max_turns=3,
        )
    )

    assert result.final_output == "Done."
    assert events == ["first:1", "second:2", "first:3"]
    initial = [LlmMessage(role="user", content="Compare.")]
    after_batch = initial + [
        batch,
        LlmMessage(role="tool", content='{"value": 10}', tool_call_id="a"),
        LlmMessage(role="tool", content='{"value": 20}', tool_call_id="b"),
    ]
    assert [messages for messages, _ in llm.calls] == [
        initial,
        after_batch,
        after_batch
        + [
            followup,
            LlmMessage(role="tool", content='{"value": 10}', tool_call_id="c"),
        ],
    ]
    assert result.messages == tuple(
        after_batch
        + [
            followup,
            LlmMessage(role="tool", content='{"value": 10}', tool_call_id="c"),
            _answer("Done."),
        ]
    )


def test_run_returns_a_detached_history_snapshot(agent: AgentDefinition) -> None:
    """Retained client references cannot alter returned messages or tool-call lists."""
    events: list[str] = []
    function = _local_tool("lookup", {"value": 42}, events)
    request = _request(ToolCall(id="a", name="lookup", arguments='{"value": 1}'))
    answer = _answer("Done.")
    llm = AsyncMock(spec=LlmClient)
    llm.complete.side_effect = [request, answer]
    expected = (
        LlmMessage(role="user", content="Lookup."),
        deepcopy(request),
        LlmMessage(role="tool", content='{"value": 42}', tool_call_id="a"),
        answer,
    )

    result = asyncio.run(
        AgentHarness.run(
            replace(agent, tools=[function]),
            run_identifier="run-test",
            input="Lookup.",
            llm=llm,
        )
    )

    assert result.messages == expected
    retained_context = llm.complete.await_args_list[0].kwargs["model_context"]
    retained_messages = retained_context.messages
    retained_messages.clear()
    request.tool_calls.clear()

    assert result.messages == expected


def test_run_keeps_broker_alias_through_tool_round_trip(
    agent: AgentDefinition,
    broker_endpoint: tuple[MagicMock, AsyncMock],
) -> None:
    """Keep using the configured broker alias after executing a tool."""
    factory, client = broker_endpoint
    events: list[str] = []
    function = _local_tool("lookup", {"value": 42}, events)
    requested = ToolCall(id="local-call", name="lookup", arguments='{"value": 1}')
    client.chat.completions.create.side_effect = [
        _completion(None, calls=[requested], finish_reason="tool_calls"),
        _completion("The value is 42."),
    ]
    definition = replace(agent, model="granite-4-1b", tools=[function])

    result = asyncio.run(
        AgentHarness.run(
            definition, run_identifier="run-test", input="Lookup.", max_turns=2
        )
    )

    assert result.final_output == "The value is 42."
    assert events == ["lookup:1"]
    assert client.chat.completions.create.await_count == 2
    first, second = client.chat.completions.create.await_args_list
    assert first.kwargs["model"] == second.kwargs["model"] == "granite-4-1b"
    assert first.kwargs["tool_choice"] == second.kwargs["tool_choice"] == "auto"
    assert [invocation.kwargs["base_url"] for invocation in factory.call_args_list] == [
        "http://metisa-llm-broker:8000/v1",
        "http://metisa-llm-broker:8000/v1",
    ]
    assert second.kwargs["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "local-call",
        "content": '{"value": 42}',
    }
