"""Exercise registered guardrail plugins through complete agent runs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from jsonschema import ValidationError
from mcp.types import CallToolResult

from metisa_harness.guardrails import GuardrailContext, GuardrailTripwire
from metisa_harness.harness import AgentHarness
from metisa_harness.models import (
    AgentDefinition,
    GuardrailPlugin,
    LlmMessage,
    ModelContext,
    ToolCall,
)

from ._harness_helpers import (
    _answer,
    _local_tool,
    _remote_tool,
    _request,
    _ScriptedLlm,
    _server,
)


def test_run_applies_lifecycle_transformations(agent: AgentDefinition) -> None:
    """Input, context, model-response, and final-output replacements take effect."""
    events: list[str] = []

    async def _before_run(context: GuardrailContext[str]) -> str:
        assert context.metadata.agent_name == agent.name
        assert context.metadata.tool_name is None
        events.append("before_run")
        return context.payload.strip()

    async def _before_model(context: GuardrailContext[ModelContext]) -> ModelContext:
        assert context.metadata.agent_name == agent.name
        events.append("before_model")
        return replace(context.payload, instructions="Use the approved context.")

    async def _after_model(context: GuardrailContext[LlmMessage]) -> LlmMessage:
        assert context.metadata.tool_name is None
        events.append("after_model")
        return replace(context.payload, content="Accepted model answer.")

    async def _after_run(
        context: GuardrailContext[str | dict[str, object]],
    ) -> str | dict[str, object]:
        assert context.payload == "Accepted model answer."
        events.append("after_run")
        return "Public answer."

    plugin = GuardrailPlugin(
        name="lifecycle",
        before_run=(_before_run,),
        before_model=(_before_model,),
        after_model=(_after_model,),
        after_run=(_after_run,),
    )
    llm = AsyncMock()
    llm.complete.return_value = _answer("Raw model answer.")

    result = asyncio.run(
        AgentHarness.run(
            replace(agent, guardrail_plugins=(plugin,)),
            run_identifier="run-test",
            prompt="  Hello.  ",
            llm=llm,
        )
    )

    assert events == ["before_run", "before_model", "after_model", "after_run"]
    sent_context = llm.complete.await_args.kwargs["model_context"]
    assert sent_context.instructions == "Use the approved context."
    assert sent_context.messages[0] == LlmMessage(role="user", content="Hello.")
    assert result.messages[-1].content == "Accepted model answer."
    assert result.final_output == "Public answer."


@pytest.mark.parametrize("source", ["local", "mcp"])
def test_run_guards_each_tool_call(agent: AgentDefinition, source: str) -> None:
    """Injected arguments reach tools and normalized results reach the next model."""
    events: list[str] = []

    async def _before_tool(
        context: GuardrailContext[dict[str, object]],
    ) -> dict[str, object]:
        assert context.metadata.agent_name == agent.name
        assert context.metadata.tool_name == "lookup"
        events.append("before_tool")
        if source == "local":
            return {"value": 7}
        return {"query": "trusted query"}

    async def _after_tool(
        context: GuardrailContext[dict[str, object]],
    ) -> dict[str, object]:
        assert context.metadata.tool_name == "lookup"
        assert context.payload == {"value": 42}
        events.append("after_tool")
        return {"value": 99}

    plugin = GuardrailPlugin(
        name="tool_policy", before_tool=(_before_tool,), after_tool=(_after_tool,)
    )
    server = _server(_remote_tool("lookup"))
    server.call_tool.return_value = CallToolResult(
        content=[], structured_content={"value": 42}
    )
    function = _local_tool("lookup", {"value": 42}, events)
    definition = replace(
        agent,
        tools=[function] if source == "local" else [],
        mcp_clients=[server] if source == "mcp" else [],
        guardrail_plugins=(plugin,),
    )
    # Missing required arguments must be supplied before schema validation.
    request = _request(
        ToolCall(id="first", name="lookup", arguments="{}"),
        ToolCall(id="second", name="lookup", arguments="{}"),
    )
    llm = _ScriptedLlm(request, _answer("Done."))

    result = asyncio.run(
        AgentHarness.run(
            definition, run_identifier="run-test", prompt="Lookup.", llm=llm
        )
    )

    expected_events = ["before_tool"]
    if source == "local":
        expected_events.append("lookup:7")
    expected_events.append("after_tool")
    assert events == expected_events * 2
    outputs = llm.calls[1][0][2:]
    assert [item.tool_call_id for item in outputs] == ["first", "second"]
    assert [json.loads(item.content or "") for item in outputs] == [
        {"value": 99},
        {"value": 99},
    ]
    assert result.final_output == "Done."
    if source == "mcp":
        assert server.call_tool.await_count == 2
        for invocation in server.call_tool.await_args_list:
            assert invocation.args == ("lookup", {"query": "trusted query"})


def test_run_orders_callbacks_and_reuses_plugins(agent: AgentDefinition) -> None:
    """Callbacks compose within and across plugins reusable by separate agents."""
    seen: list[tuple[str, str]] = []

    async def _first(context: GuardrailContext[str]) -> str:
        seen.append((context.metadata.agent_name, context.payload))
        return context.payload + " first"

    async def _second(context: GuardrailContext[str]) -> str:
        return context.payload + " second"

    async def _third(context: GuardrailContext[str]) -> str:
        return context.payload + " third"

    plugins = (
        GuardrailPlugin(name="first", before_run=(_first, _second)),
        GuardrailPlugin(name="second", before_run=(_third,)),
    )
    for name in ("One", "Two"):
        llm = _ScriptedLlm(_answer("Done."))
        definition = replace(agent, name=name, guardrail_plugins=plugins)
        asyncio.run(
            AgentHarness.run(
                definition, run_identifier="run-test", prompt="Start", llm=llm
            )
        )
        assert llm.calls[0][0][0].content == "Start first second third"

    assert seen == [("One", "Start"), ("Two", "Start")]


@pytest.mark.parametrize(
    "stage",
    [
        "before_run",
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "after_run",
    ],
)
def test_run_tripwire_aborts_processing(agent: AgentDefinition, stage: str) -> None:
    """A tripwire propagates unchanged and stops later callbacks, tools, and turns."""
    error = GuardrailTripwire("Blocked by policy.")
    reject = AsyncMock(side_effect=error)
    later = AsyncMock(side_effect=AssertionError("Later guardrail must not run."))
    plugin = GuardrailPlugin(
        name="blocking",
        before_run=(reject, later) if stage == "before_run" else (),
        before_model=(reject, later) if stage == "before_model" else (),
        after_model=(reject, later) if stage == "after_model" else (),
        before_tool=(reject, later) if stage == "before_tool" else (),
        after_tool=(reject, later) if stage == "after_tool" else (),
        after_run=(reject, later) if stage == "after_run" else (),
    )
    events: list[str] = []
    function = _local_tool("lookup", {"value": 42}, events)
    request = _request(
        ToolCall(id="first", name="lookup", arguments='{"value": 1}'),
        ToolCall(id="second", name="lookup", arguments='{"value": 2}'),
    )
    llm = _ScriptedLlm(_answer("Done.") if stage == "after_run" else request)
    definition = replace(agent, tools=[function], guardrail_plugins=(plugin,))

    with pytest.raises(GuardrailTripwire) as caught:
        asyncio.run(
            AgentHarness.run(
                definition, run_identifier="run-test", prompt="Lookup.", llm=llm
            )
        )

    assert caught.value is error
    reject.assert_awaited_once()
    later.assert_not_awaited()
    assert events == (["lookup:1"] if stage == "after_tool" else [])
    assert len(llm.calls) == (0 if stage in ("before_run", "before_model") else 1)


@pytest.mark.parametrize("stage", ["before_tool", "after_tool"])
def test_run_validates_transformed_tool_payload(
    agent: AgentDefinition, stage: str
) -> None:
    """Transformed arguments and results must satisfy the tool schemas."""
    guardrail = AsyncMock(return_value={"value": "invalid"})
    plugin = GuardrailPlugin(
        name="invalid_payload",
        before_tool=(guardrail,) if stage == "before_tool" else (),
        after_tool=(guardrail,) if stage == "after_tool" else (),
    )
    events: list[str] = []
    function = _local_tool("lookup", {"value": 42}, events)
    llm = _ScriptedLlm(
        _request(ToolCall(id="first", name="lookup", arguments='{"value": 1}'))
    )
    definition = replace(agent, tools=[function], guardrail_plugins=(plugin,))

    with pytest.raises(ValidationError):
        asyncio.run(
            AgentHarness.run(
                definition, run_identifier="run-test", prompt="Lookup.", llm=llm
            )
        )

    guardrail.assert_awaited_once()
    assert events == (["lookup:1"] if stage == "after_tool" else [])
    assert len(llm.calls) == 1


@pytest.mark.parametrize("replacement", ["", "  ", {"value": 1}])
def test_run_rejects_invalid_text_from_output_guardrail(
    agent: AgentDefinition, replacement: str | dict[str, object]
) -> None:
    """Text agents retain their nonempty-string contract after guardrails."""
    guardrail = AsyncMock(return_value=replacement)
    plugin = GuardrailPlugin(name="invalid_text", after_run=(guardrail,))
    llm = _ScriptedLlm(_answer("Valid answer."))

    with pytest.raises(TypeError, match="An output guardrail returned invalid text"):
        asyncio.run(
            AgentHarness.run(
                replace(agent, guardrail_plugins=(plugin,)),
                run_identifier="run-test",
                prompt="Hello.",
                llm=llm,
            )
        )
