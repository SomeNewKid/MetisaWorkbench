"""Tests for agent definitions and run-result data models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
from mcp import Client

from metisa_harness.models import (
    AgentDefinition,
    LlmMessage,
    RunResult,
    ToolCall,
)

from ._harness_helpers import (
    _get_current_date_and_time,
)


@pytest.mark.parametrize("field_name", ["name", "model", "instructions"])
@pytest.mark.parametrize("value", ["", " ", "\t", "\n", " \t\r\n "])
def test_agent_definition_rejects_blank_fields(field_name: str, value: str) -> None:
    """Each required field must contain at least one non-whitespace character."""
    fields = {
        "name": "Assistant",
        "description": "Test agent",
        "model": "test-model",
        "instructions": "Answer briefly.",
    }
    fields[field_name] = value

    verb = "are" if field_name == "instructions" else "is"
    with pytest.raises(ValueError, match=f"^Agent {field_name} {verb} required\\.$"):
        AgentDefinition(
            name=fields["name"],
            description=fields["description"],
            model=fields["model"],
            instructions=fields["instructions"],
        )


def test_agent_definition_preserves_supplied_text() -> None:
    """Validation must preserve surrounding whitespace and instruction layout."""
    agent = AgentDefinition(
        name=" Assistant ",
        description=" Test agent ",
        model=" test-model ",
        instructions="\n    Answer briefly.\n    Preserve formatting.\n",
    )

    assert agent.name == " Assistant "
    assert agent.description == " Test agent "
    assert agent.model == " test-model "
    assert agent.instructions == "\n    Answer briefly.\n    Preserve formatting.\n"


@pytest.mark.parametrize(
    "field_name",
    [
        "name",
        "model",
        "instructions",
        "tools",
        "mcp_clients",
        "guardrail_plugins",
        "output_type",
    ],
)
def test_agent_definition_is_frozen(agent: AgentDefinition, field_name: str) -> None:
    """An agent definition's fields cannot be reassigned after construction."""
    with pytest.raises(FrozenInstanceError):
        setattr(agent, field_name, "replacement")


def test_agent_definition_defaults_to_no_tools_or_servers(
    agent: AgentDefinition,
) -> None:
    """Existing callers can omit both optional collections."""
    assert agent.tools == []
    assert agent.mcp_clients == []
    assert agent.guardrail_plugins == ()
    assert agent.output_type is None


def test_agent_definition_defaults_are_independent(agent: AgentDefinition) -> None:
    """Mutating one agent's defaults must not change another agent."""
    other = AgentDefinition(
        name="Other",
        description="Test agent",
        model="test-model",
        instructions="Be brief.",
    )
    server = MagicMock(spec=Client)

    agent.tools.append(_get_current_date_and_time)
    agent.mcp_clients.append(server)

    assert other.tools == []
    assert other.mcp_clients == []


def test_agent_definition_accepts_local_tools_and_mcp_servers() -> None:
    """Callers can supply Python functions and MCP clients together."""
    server = MagicMock(spec=Client)
    definition = AgentDefinition(
        name="Assistant",
        description="Test agent",
        model="test-model",
        instructions="Answer briefly.",
        tools=[_get_current_date_and_time],
        mcp_clients=[server],
    )

    assert definition.tools == [_get_current_date_and_time]
    assert definition.mcp_clients == [server]
    server.list_tools.assert_not_called()


@pytest.mark.parametrize("final_text", ["Hello.", "", " Answer.\n"])
def test_run_result_preserves_final_text(final_text: str) -> None:
    """Results retain the supplied answer without altering its formatting."""
    result = RunResult(final_output=final_text, messages=())

    assert result.final_output == final_text


@pytest.mark.parametrize("field_name", ["final_output", "messages"])
def test_run_result_is_frozen(field_name: str) -> None:
    """Neither the final answer nor the history field can be reassigned."""
    result = RunResult(final_output="Hello.", messages=())

    with pytest.raises(FrozenInstanceError):
        setattr(result, field_name, "Replacement.")


def test_run_result_preserves_conversation_messages() -> None:
    """Results retain message order, tool arguments, result IDs, and final text."""
    tool_call = ToolCall(id="clock-1", name="clock", arguments="{}")
    messages = (
        LlmMessage(role="user", content="What time?"),
        LlmMessage(role="assistant", content=None, tool_calls=[tool_call]),
        LlmMessage(role="tool", content='{"hour": 12}', tool_call_id="clock-1"),
        LlmMessage(role="assistant", content="It is noon."),
    )

    result = RunResult(final_output="It is noon.", messages=messages)

    assert result.messages == messages
    assert result.final_output == result.messages[-1].content


def test_run_result_accepts_empty_history() -> None:
    """The data container preserves an explicitly supplied empty history."""
    result = RunResult(final_output="Hello.", messages=())

    assert result.messages == ()
