"""Verify typed final answers and dictionary guardrails through agent runs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from metisa_harness.guardrails import GuardrailContext, GuardrailTripwire
from metisa_harness.harness import AgentHarness
from metisa_harness.models import AgentDefinition, GuardrailPlugin, ToolCall

from ._harness_helpers import _answer, _completion, _local_tool, _ScriptedLlm


class _Answer(BaseModel):
    """Require a positive integer and an application-approved label."""

    model_config = ConfigDict(extra="forbid", strict=True)

    value: int = Field(gt=0)
    label: str

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        if value != "approved":
            raise ValueError("Label must be approved.")
        return value


def test_run_returns_typed_output(agent: AgentDefinition) -> None:
    """JSON becomes the specified model while conversation content remains text."""
    text = '{"value": 42, "label": "approved"}'
    llm = _ScriptedLlm(_answer(text))

    result = asyncio.run(
        AgentHarness.run(
            replace(agent, output_type=_Answer),
            run_identifier="run-test",
            input="Answer.",
            llm=llm,
        )
    )

    assert isinstance(result.final_output, _Answer)
    assert result.final_output.value == 42
    assert result.final_output.label == "approved"
    assert result.messages[-1].content == text
    assert len(llm.calls) == 1


def test_run_normalizes_dictionary_before_pydantic_validation(
    agent: AgentDefinition,
) -> None:
    """Output guardrails may repair schema-invalid fields before object creation."""
    seen: list[dict[str, object]] = []

    async def _normalize(
        context: GuardrailContext[str | dict[str, object]],
    ) -> str | dict[str, object]:
        assert context.metadata.agent_name == agent.name
        assert context.metadata.tool_name is None
        assert isinstance(context.payload, dict)
        seen.append(dict(context.payload))
        assert context.payload["value"] == "42"
        return {"value": 42, "label": "approved"}

    plugin = GuardrailPlugin(name="normalize", after_run=(_normalize,))
    llm = _ScriptedLlm(_answer('{"value": "42"}'))
    definition = replace(agent, output_type=_Answer, guardrail_plugins=(plugin,))

    result = asyncio.run(
        AgentHarness.run(
            definition, run_identifier="run-test", input="Answer.", llm=llm
        )
    )

    assert seen == [{"value": "42"}]
    assert result.final_output == _Answer(value=42, label="approved")


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"value": "42", "label": "approved"}',
        '{"value": 0, "label": "approved"}',
        '{"value": 42, "label": "unapproved"}',
        '{"value": 42, "label": "approved", "extra": true}',
    ],
    ids=[
        "missing-fields",
        "wrong-type",
        "constraint",
        "custom-validator",
        "extra-field",
    ],
)
def test_run_rejects_invalid_typed_output_without_retry(
    agent: AgentDefinition, text: str
) -> None:
    """Schema and custom validation failures abort rather than request a repair."""
    llm = _ScriptedLlm(_answer(text), _answer('{"value": 42, "label": "approved"}'))

    with pytest.raises(ValidationError):
        asyncio.run(
            AgentHarness.run(
                replace(agent, output_type=_Answer),
                run_identifier="run-test",
                input="Answer.",
                llm=llm,
            )
        )

    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    ("text", "error_type"),
    [
        ("Not JSON", json.JSONDecodeError),
        ("{", json.JSONDecodeError),
        ("[]", TypeError),
        ("null", TypeError),
        ('"text"', TypeError),
        ("42", TypeError),
    ],
)
def test_run_rejects_non_object_output_before_guardrails(
    agent: AgentDefinition, text: str, error_type: type[Exception]
) -> None:
    """Dictionary guardrails never receive malformed JSON or other JSON shapes."""
    guardrail = AsyncMock()
    plugin = GuardrailPlugin(name="observe", after_run=(guardrail,))
    llm = _ScriptedLlm(_answer(text))
    definition = replace(agent, output_type=_Answer, guardrail_plugins=(plugin,))

    with pytest.raises(error_type):
        asyncio.run(
            AgentHarness.run(
                definition, run_identifier="run-test", input="Answer.", llm=llm
            )
        )

    guardrail.assert_not_awaited()
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    ("replacement", "error_type"),
    [
        ("wrong shape", TypeError),
        ({"value": "wrong type", "label": "approved"}, ValidationError),
        ({"value": 42, "label": "unapproved"}, ValidationError),
    ],
)
def test_run_validates_output_guardrail_replacements(
    agent: AgentDefinition,
    replacement: str | dict[str, object],
    error_type: type[Exception],
) -> None:
    """A valid model answer does not exempt guardrail replacements from validation."""
    guardrail = AsyncMock(return_value=replacement)
    plugin = GuardrailPlugin(name="replace", after_run=(guardrail,))
    llm = _ScriptedLlm(_answer('{"value": 42, "label": "approved"}'))
    definition = replace(agent, output_type=_Answer, guardrail_plugins=(plugin,))

    with pytest.raises(error_type):
        asyncio.run(
            AgentHarness.run(
                definition, run_identifier="run-test", input="Answer.", llm=llm
            )
        )

    guardrail.assert_awaited_once()
    assert len(llm.calls) == 1


def test_run_structured_output_tripwire_propagates(agent: AgentDefinition) -> None:
    """An output tripwire on decoded data aborts without constructing a result."""
    error = GuardrailTripwire("Answer blocked.")
    guardrail = AsyncMock(side_effect=error)
    plugin = GuardrailPlugin(name="block", after_run=(guardrail,))
    llm = _ScriptedLlm(_answer('{"value": 42, "label": "approved"}'))
    definition = replace(agent, output_type=_Answer, guardrail_plugins=(plugin,))

    with pytest.raises(GuardrailTripwire) as caught:
        asyncio.run(
            AgentHarness.run(
                definition, run_identifier="run-test", input="Answer.", llm=llm
            )
        )

    assert caught.value is error
    assert guardrail.await_args is not None
    assert guardrail.await_args.args[0].payload == {"value": 42, "label": "approved"}
    assert len(llm.calls) == 1


@pytest.mark.parametrize("use_tool", [False, True])
def test_run_sends_output_schema_to_model_adapter(
    agent: AgentDefinition,
    broker_endpoint: tuple[MagicMock, AsyncMock],
    use_tool: bool,
) -> None:
    """A declared output type changes the actual request and yields a typed answer."""
    _, client = broker_endpoint
    final_response = _completion('{"value": 42, "label": "approved"}')
    responses = [final_response]
    events: list[str] = []
    tool = _local_tool("lookup", {"value": 42}, events)
    if use_tool:
        requested = ToolCall(id="lookup-1", name="lookup", arguments='{"value": 1}')
        responses.insert(
            0, _completion(None, calls=[requested], finish_reason="tool_calls")
        )
    client.chat.completions.create.side_effect = responses
    definition = replace(
        agent,
        model="gpt-4o",
        output_type=_Answer,
        tools=[tool] if use_tool else [],
    )

    result = asyncio.run(
        AgentHarness.run(definition, run_identifier="run-test", input="Answer.")
    )

    assert client.chat.completions.create.await_count == (2 if use_tool else 1)
    for invocation in client.chat.completions.create.await_args_list:
        instructions = invocation.kwargs["messages"][0]["content"]
        assert instructions.startswith(agent.instructions)
        assert "return only JSON" in instructions
        schema_text = instructions[instructions.index("{") :]
        assert json.loads(schema_text) == _Answer.model_json_schema()
    assert events == (["lookup:1"] if use_tool else [])
    assert result.final_output == _Answer(value=42, label="approved")
