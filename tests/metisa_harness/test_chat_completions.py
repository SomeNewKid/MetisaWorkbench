"""Tests for Chat Completions requests, responses, and tool-message conversion."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult
from openai import omit
from openai.types.chat import ChatCompletion, ChatCompletionMessageCustomToolCall

from metisa_common.mcp_helper import create_tool
from metisa_harness._chat_completions import OpenAiLlmClient
from metisa_harness.harness import AgentHarness
from metisa_harness.models import (
    AgentDefinition,
    ToolCall,
)

from ._harness_helpers import (
    _completion,
    _get_current_date_and_time,
    _local_tool,
    _remote_tool,
    _server,
)


@pytest.mark.parametrize("include_tools", [False, True])
def test_run_sends_openai_tool_definitions(
    agent: AgentDefinition,
    broker_endpoint: tuple[MagicMock, AsyncMock],
    include_tools: bool,
) -> None:
    """The adapter translates metadata and omits options when no tools exist."""
    remote = _remote_tool("remote")
    server = _server(remote)
    definition = replace(
        agent,
        model="gpt-4o",
        tools=[_get_current_date_and_time] if include_tools else [],
        mcp_clients=[server] if include_tools else [],
    )
    _, client = broker_endpoint
    client.chat.completions.create.return_value = _completion("Hello.")

    result = asyncio.run(
        AgentHarness.run(definition, run_identifier="run-test", input="Hello.")
    )

    client.chat.completions.create.assert_awaited_once()
    request = client.chat.completions.create.await_args.kwargs
    assert request["model"] == "gpt-4o"
    assert request["messages"] == [
        {"role": "system", "content": agent.instructions},
        {"role": "user", "content": "Hello."},
    ]
    assert result.final_output == "Hello."
    if include_tools:
        local = create_tool(_get_current_date_and_time)
        reasoning_property = {
            "model_reasoning": {
                "type": "string",
                "description": "Briefly explain the reason for calling this tool.",
            },
        }
        local_schema = local.input_schema | {
            "properties": local.input_schema["properties"] | reasoning_property,
        }
        remote_schema = remote.input_schema | {
            "properties": remote.input_schema["properties"] | reasoning_property,
        }
        assert request["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": local.name,
                    "description": local.description,
                    "parameters": local_schema,
                    "strict": False,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remote",
                    "description": "",
                    "parameters": remote_schema,
                    "strict": False,
                },
            },
        ]
        assert "model_reasoning" not in remote.input_schema["properties"]
        assert request["tool_choice"] == "auto"
        server.call_tool.assert_not_awaited()
    else:
        assert request["tools"] is omit
        assert request["tool_choice"] is omit


def test_run_round_trips_openai_tool_calls_and_results(
    agent: AgentDefinition,
    broker_endpoint: tuple[MagicMock, AsyncMock],
) -> None:
    """Real adapter conversion preserves assistant text, raw arguments, and call IDs."""
    events: list[str] = []
    function = _local_tool("local", {"value": 42}, events)
    server = _server(_remote_tool("remote"))
    server.call_tool.return_value = CallToolResult(
        content=[], structured_content={"value": 9}
    )
    calls = [
        ToolCall(id="local-id", name="local", arguments='{ "value": 7 }'),
        ToolCall(id="remote-id", name="remote", arguments='{"query":"x"}'),
    ]
    _, client = broker_endpoint
    client.chat.completions.create.side_effect = [
        _completion("Checking.", calls=calls, finish_reason="tool_calls"),
        _completion("Done."),
    ]
    definition = replace(agent, model="gpt-4o", tools=[function], mcp_clients=[server])

    result = asyncio.run(
        AgentHarness.run(
            definition, run_identifier="run-test", input="Lookup.", max_turns=2
        )
    )

    assert result.final_output == "Done."
    assert events == ["local:7"]
    server.call_tool.assert_awaited_once_with("remote", {"query": "x"})
    assert client.chat.completions.create.await_count == 2
    first, second = client.chat.completions.create.await_args_list
    assert second.kwargs["messages"] == [
        {"role": "system", "content": agent.instructions},
        {"role": "user", "content": "Lookup."},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": item.id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
                for item in calls
            ],
        },
        {"role": "tool", "tool_call_id": "local-id", "content": '{"value": 42}'},
        {"role": "tool", "tool_call_id": "remote-id", "content": '{"value": 9}'},
    ]
    assert first.kwargs["messages"] == second.kwargs["messages"][:2]
    assert first.kwargs["tools"] == second.kwargs["tools"]
    assert first.kwargs["tool_choice"] == second.kwargs["tool_choice"] == "auto"


@pytest.mark.parametrize(
    ("completion_factory", "error_message"),
    [
        (
            lambda: _completion_without_choices(),
            "OpenAI returned no completion choices",
        ),
        (
            lambda: _completion(None, finish_reason="length"),
            "OpenAI completion ended with: length",
        ),
        (
            lambda: _completion(None, finish_reason="content_filter"),
            "OpenAI completion ended with: content_filter",
        ),
        (
            lambda: _completion(None, finish_reason="tool_calls"),
            "OpenAI returned no tool calls",
        ),
        (lambda: _completion(None), "The model returned no final text"),
        (
            lambda: _completion_with_custom_tool(),
            "Unsupported tool call type",
        ),
    ],
    ids=[
        "no-choices",
        "truncated",
        "filtered",
        "missing-calls",
        "missing-text",
        "custom-tool",
    ],
)
def test_run_rejects_invalid_openai_responses(
    agent: AgentDefinition,
    broker_endpoint: tuple[MagicMock, AsyncMock],
    completion_factory: Callable[[], ChatCompletion],
    error_message: str,
) -> None:
    """Malformed or incomplete responses must not produce a successful run."""
    _, client = broker_endpoint
    client.chat.completions.create.return_value = completion_factory()

    with pytest.raises(RuntimeError, match=error_message):
        asyncio.run(AgentHarness.run(agent, run_identifier="run-test", input="Hi"))


def test_run_omits_tool_choice_when_adapter_disables_it(
    agent: AgentDefinition,
    broker_endpoint: tuple[MagicMock, AsyncMock],
) -> None:
    """The adapter can still advertise tools with tool choice disabled."""
    _, client = broker_endpoint
    definition = replace(agent, tools=[_get_current_date_and_time])
    llm = OpenAiLlmClient(model=definition.model, supports_tool_choice=False)

    asyncio.run(
        AgentHarness.run(definition, run_identifier="run-test", input="Hello.", llm=llm)
    )

    client.chat.completions.create.assert_awaited_once()
    request = client.chat.completions.create.await_args.kwargs
    assert request["tool_choice"] is omit
    assert len(request["tools"]) == 1
    assert request["tools"][0]["function"]["name"] == "_get_current_date_and_time"


def _completion_without_choices() -> ChatCompletion:
    completion = _completion(None)
    completion.choices = []
    return completion


def _completion_with_custom_tool() -> ChatCompletion:
    completion = _completion(None, finish_reason="tool_calls")
    custom_call = ChatCompletionMessageCustomToolCall.model_validate(
        {
            "id": "custom-id",
            "type": "custom",
            "custom": {"name": "custom-tool", "input": "test input"},
        }
    )
    completion.choices[0].message.tool_calls = [custom_call]
    return completion
