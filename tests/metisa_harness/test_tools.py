"""Tests for tool discovery, registration, invocation, and schema validation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import call

import pytest
from jsonschema import ValidationError
from mcp.types import CallToolResult, ListToolsResult

from metisa_common.mcp_helper import create_tool
from metisa_harness.harness import AgentHarness
from metisa_harness.models import (
    AgentDefinition,
    LlmMessage,
    ToolCall,
)

from ._harness_helpers import (
    _answer,
    _get_current_date_and_time,
    _local_tool,
    _remote_tool,
    _request,
    _ScriptedLlm,
    _server,
)


def test_run_combines_local_tools_and_paginated_mcp_tools(
    agent: AgentDefinition,
) -> None:
    """Discovery combines all pages and servers without executing any tools."""
    first, second, third = [_remote_tool(name) for name in ("first", "second", "third")]
    server = _server(first)
    server.list_tools.side_effect = [
        ListToolsResult(tools=[first], next_cursor="page-2"),
        ListToolsResult(tools=[second]),
    ]
    other_server = _server(third)
    definition = replace(
        agent,
        tools=[_get_current_date_and_time],
        mcp_clients=[server, other_server],
    )
    llm = _ScriptedLlm(_answer("No tools needed."))

    asyncio.run(
        AgentHarness.run(
            definition, run_identifier="run-test", prompt="Hello.", llm=llm
        )
    )

    assert llm.calls == [
        (
            [LlmMessage(role="user", content="Hello.")],
            [create_tool(_get_current_date_and_time), first, second, third],
        )
    ]
    assert server.list_tools.await_args_list == [
        call(cursor=None),
        call(cursor="page-2"),
    ]
    other_server.list_tools.assert_awaited_once_with(cursor=None)
    for borrowed in (server, other_server):
        borrowed.call_tool.assert_not_awaited()
        borrowed.__aenter__.assert_not_called()
        borrowed.__aexit__.assert_not_called()


def test_run_accepts_an_empty_mcp_tool_list(agent: AgentDefinition) -> None:
    """An empty server tool list does not prevent a model request."""
    server = _server()
    llm = _ScriptedLlm(_answer("Hello."))

    asyncio.run(
        AgentHarness.run(
            replace(agent, mcp_clients=[server]),
            run_identifier="run-test",
            prompt="Hello.",
            llm=llm,
        )
    )

    assert llm.calls[0][1] == []
    server.list_tools.assert_awaited_once_with(cursor=None)


@pytest.mark.parametrize(
    "source", ["local", "local-and-remote", "remote-pages", "servers"]
)
def test_run_rejects_duplicate_tool_names(agent: AgentDefinition, source: str) -> None:
    """Ambiguous names fail before the model or any tool is invoked."""
    local = _get_current_date_and_time
    duplicate = _remote_tool(local.__name__)
    server = _server(duplicate)
    other_server = _server(duplicate)
    if source == "local":
        definition = replace(agent, tools=[local, local])
    elif source == "local-and-remote":
        definition = replace(agent, tools=[local], mcp_clients=[server])
    elif source == "remote-pages":
        server.list_tools.side_effect = [
            ListToolsResult(tools=[duplicate], next_cursor="page-2"),
            ListToolsResult(tools=[duplicate]),
        ]
        definition = replace(agent, mcp_clients=[server])
    else:
        definition = replace(agent, mcp_clients=[server, other_server])
    llm = _ScriptedLlm()

    with pytest.raises(ValueError, match=f"^Duplicate tool name: {local.__name__}$"):
        asyncio.run(
            AgentHarness.run(
                definition, run_identifier="run-test", prompt="Hello.", llm=llm
            )
        )

    assert llm.calls == []
    server.call_tool.assert_not_awaited()
    other_server.call_tool.assert_not_awaited()


def test_run_propagates_mcp_discovery_error(agent: AgentDefinition) -> None:
    """Discovery failures do not silently remove a server's tools."""
    error = ConnectionError("MCP server unavailable.")
    server = _server()
    server.list_tools.side_effect = error
    llm = _ScriptedLlm()

    with pytest.raises(ConnectionError) as caught:
        asyncio.run(
            AgentHarness.run(
                replace(agent, mcp_clients=[server]),
                run_identifier="run-test",
                prompt="Hi",
                llm=llm,
            )
        )

    assert caught.value is error
    assert llm.calls == []


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_run_invokes_local_tool_and_returns_result_to_model(
    agent: AgentDefinition, asynchronous: bool
) -> None:
    """Both function kinds receive validated keyword arguments and are awaited."""
    events: list[str] = []
    function = _local_tool("lookup", {"value": 42}, events, asynchronous=asynchronous)
    request = _request(ToolCall(id="call-1", name="lookup", arguments='{"value": 7}'))
    llm = _ScriptedLlm(request, _answer("The value is 42."))
    definition = replace(agent, tools=[function])

    result = asyncio.run(
        AgentHarness.run(
            definition,
            run_identifier="run-test",
            prompt="Look it up.",
            llm=llm,
            max_turns=2,
        )
    )

    assert events == ["lookup:7"]
    assert result.final_output == "The value is 42."
    assert llm.calls[0][0] == [LlmMessage(role="user", content="Look it up.")]
    assert llm.calls[1][0] == [
        LlmMessage(role="user", content="Look it up."),
        request,
        LlmMessage(role="tool", content='{"value": 42}', tool_call_id="call-1"),
    ]
    assert result.messages == tuple(llm.calls[1][0]) + (_answer("The value is 42."),)
    assert llm.calls[0][1] == llm.calls[1][1] == [create_tool(function)]


def test_run_routes_mcp_calls_to_the_correct_tool_and_server(
    agent: AgentDefinition,
) -> None:
    """Remote closures retain their server and tool and leave clients open."""
    first, second, third = [_remote_tool(name) for name in ("first", "second", "third")]
    server = _server(first, second)
    other = _server(third)
    server.call_tool.side_effect = [
        CallToolResult(content=[], structured_content={"value": 1}),
        CallToolResult(content=[], structured_content={"value": 2}),
    ]
    other.call_tool.return_value = CallToolResult(
        content=[], structured_content={"value": 3}
    )
    requests = [
        ToolCall(id=name, name=name, arguments='{"query": "' + name + '"}')
        for name in ("first", "second", "third")
    ]
    llm = _ScriptedLlm(_request(*requests), _answer("Done."))
    definition = replace(agent, mcp_clients=[server, other])

    result = asyncio.run(
        AgentHarness.run(
            definition, run_identifier="run-test", prompt="Lookup.", llm=llm
        )
    )

    assert result.final_output == "Done."
    assert server.call_tool.await_args_list == [
        call("first", {"query": "first"}),
        call("second", {"query": "second"}),
    ]
    other.call_tool.assert_awaited_once_with("third", {"query": "third"})
    assert llm.calls[1][0][2:] == [
        LlmMessage(role="tool", content='{"value": 1}', tool_call_id="first"),
        LlmMessage(role="tool", content='{"value": 2}', tool_call_id="second"),
        LlmMessage(role="tool", content='{"value": 3}', tool_call_id="third"),
    ]
    assert result.messages == tuple(llm.calls[1][0]) + (_answer("Done."),)
    for borrowed in (server, other):
        borrowed.__aenter__.assert_not_called()
        borrowed.__aexit__.assert_not_called()


@pytest.mark.parametrize(
    ("name", "arguments", "error_type"),
    [
        ("unknown", "{}", RuntimeError),
        ("lookup", "{", ValueError),
        ("lookup", "[]", ValueError),
        ("lookup", "null", ValueError),
        ("lookup", "{}", ValidationError),
        ("lookup", '{"value": "wrong"}', ValidationError),
        ("lookup", '{"value": 1, "unexpected": 2}', ValidationError),
    ],
)
def test_run_rejects_invalid_tool_requests_before_execution(
    agent: AgentDefinition, name: str, arguments: str, error_type: type[Exception]
) -> None:
    """Unknown tools and invalid JSON or schemas never reach the callable."""
    events: list[str] = []
    function = _local_tool("lookup", {"value": 42}, events)
    llm = _ScriptedLlm(_request(ToolCall(id="bad", name=name, arguments=arguments)))

    with pytest.raises(error_type):
        asyncio.run(
            AgentHarness.run(
                replace(agent, tools=[function]),
                run_identifier="run-test",
                prompt="Hi",
                llm=llm,
            )
        )

    assert events == []
    assert len(llm.calls) == 1


def test_run_propagates_local_execution_error(agent: AgentDefinition) -> None:
    """Execution failures abort the run rather than inventing a tool result."""
    events: list[str] = []
    error = RuntimeError("Local lookup failed.")
    function = _local_tool("lookup", {"value": 42}, events, error=error)
    llm = _ScriptedLlm(
        _request(ToolCall(id="a", name="lookup", arguments='{"value": 1}'))
    )

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            AgentHarness.run(
                replace(agent, tools=[function]),
                run_identifier="run-test",
                prompt="Hi",
                llm=llm,
            )
        )

    assert caught.value is error
    assert events == ["lookup:1"]
    assert len(llm.calls) == 1


def test_run_validates_local_output(agent: AgentDefinition) -> None:
    """A tool result violating its declared output schema is not sent to the LLM."""
    events: list[str] = []
    function = _local_tool("lookup", {"value": "not an integer"}, events)
    llm = _ScriptedLlm(
        _request(ToolCall(id="a", name="lookup", arguments='{"value": 1}'))
    )

    with pytest.raises(ValidationError):
        asyncio.run(
            AgentHarness.run(
                replace(agent, tools=[function]),
                run_identifier="run-test",
                prompt="Hi",
                llm=llm,
            )
        )

    assert events == ["lookup:1"]
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "error_type"),
    [
        (CallToolResult(content=[], is_error=True), RuntimeError),
        (CallToolResult(content=[]), RuntimeError),
        (
            CallToolResult(content=[], structured_content={"value": "wrong"}),
            ValidationError,
        ),
        (ConnectionError("Remote lookup failed."), ConnectionError),
    ],
    ids=[
        "tool-error",
        "missing-structured-result",
        "invalid-output",
        "transport-error",
    ],
)
def test_run_propagates_mcp_tool_failures(
    agent: AgentDefinition,
    outcome: CallToolResult | Exception,
    error_type: type[Exception],
) -> None:
    """Remote errors and invalid outputs cannot become successful model inputs."""
    server = _server(_remote_tool("lookup"))
    if isinstance(outcome, Exception):
        server.call_tool.side_effect = outcome
    else:
        server.call_tool.return_value = outcome
    llm = _ScriptedLlm(
        _request(ToolCall(id="a", name="lookup", arguments='{"query": "x"}'))
    )

    with pytest.raises(error_type):
        asyncio.run(
            AgentHarness.run(
                replace(agent, mcp_clients=[server]),
                run_identifier="run-test",
                prompt="Hi",
                llm=llm,
            )
        )

    server.call_tool.assert_awaited_once_with("lookup", {"query": "x"})
    assert len(llm.calls) == 1
