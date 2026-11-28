"""Verify lifecycle event ordering, metadata, failures, and cancellation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest

from metisa_harness.guardrails import GuardrailTripwire
from metisa_harness.harness import AgentHarness
from metisa_harness.lifecycle import EventKind, LifecycleEvent
from metisa_harness.models import AgentDefinition, GuardrailPlugin, ToolCall

from ._harness_helpers import (
    _answer,
    _local_tool,
    _remote_tool,
    _request,
    _ScriptedLlm,
    _server,
)

_RUN_IDENTIFIER = "run-lifecycle-test"


class _EventCollector:
    """Collect events in delivery order."""

    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []

    async def __call__(self, event: LifecycleEvent) -> None:
        self.events.append(event)

    @property
    def kinds(self) -> list[EventKind]:
        return [event.kind for event in self.events]


def test_run_emits_success_events(agent: AgentDefinition) -> None:
    """A final answer produces ordered events with consistent run metadata."""
    collector = _EventCollector()
    llm = _ScriptedLlm(_answer("Done."))

    result = asyncio.run(
        AgentHarness.run(
            agent,
            run_identifier=_RUN_IDENTIFIER,
            prompt="Hello.",
            llm=llm,
            event_handler=collector,
        )
    )

    assert result.final_output == "Done."
    assert collector.kinds == [
        "run_started",
        "model_started",
        "model_completed",
        "run_completed",
    ]
    assert [event.turn for event in collector.events] == [None, 1, 1, None]
    _assert_metadata(collector.events, agent)


def test_run_emits_tool_events(agent: AgentDefinition) -> None:
    """Multiple tools retain call IDs and model turns in execution order."""
    collector = _EventCollector()
    executions: list[str] = []
    function = _local_tool("lookup", {"value": 7}, executions)
    definition = replace(agent, tools=[function])
    llm = _ScriptedLlm(
        _request(
            ToolCall(id="first", name="lookup", arguments='{"value": 1}'),
            ToolCall(id="second", name="lookup", arguments='{"value": 2}'),
        ),
        _answer("Done."),
    )

    asyncio.run(
        AgentHarness.run(
            definition,
            run_identifier=_RUN_IDENTIFIER,
            prompt="Compare.",
            llm=llm,
            event_handler=collector,
        )
    )

    assert collector.kinds == [
        "run_started",
        "model_started",
        "model_completed",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
        "model_started",
        "model_completed",
        "run_completed",
    ]
    assert [event.turn for event in collector.events] == [
        None,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        None,
    ]
    tool_events = [event for event in collector.events if event.tool_name is not None]
    assert [event.tool_name for event in tool_events] == ["lookup"] * 4
    assert [event.tool_call_id for event in tool_events] == [
        "first",
        "first",
        "second",
        "second",
    ]
    assert executions == ["lookup:1", "lookup:2"]
    _assert_metadata(collector.events, agent)


@pytest.mark.parametrize("stage", ["model", "tool"])
def test_run_emits_failure_and_preserves_exception(
    agent: AgentDefinition, stage: str
) -> None:
    """An operation failure emits no completion and preserves its exception."""
    collector = _EventCollector()
    error = ValueError("Operation failed.")
    expected: list[EventKind] = ["run_started", "model_started"]
    definition = agent
    llm = _ScriptedLlm(error)
    if stage == "tool":
        function = _local_tool("lookup", {}, [], error=error)
        definition = replace(agent, tools=[function])
        llm = _ScriptedLlm(
            _request(ToolCall(id="lookup-1", name="lookup", arguments='{"value": 1}'))
        )
        expected.extend(["model_completed", "tool_started"])

    with pytest.raises(ValueError) as caught:
        asyncio.run(
            AgentHarness.run(
                definition,
                run_identifier=_RUN_IDENTIFIER,
                prompt="Lookup.",
                llm=llm,
                event_handler=collector,
            )
        )

    assert caught.value is error
    assert collector.kinds == [*expected, "run_failed"]
    assert collector.events[-1].error_type == "ValueError"
    _assert_metadata(collector.events, agent)


def test_run_emits_setup_failure(agent: AgentDefinition) -> None:
    """Model discovery failures still produce run start and failure events."""
    collector = _EventCollector()
    error = RuntimeError("Discovery unavailable.")

    with patch("metisa_harness.harness.get_llm", side_effect=error):
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(
                AgentHarness.run(
                    agent,
                    run_identifier=_RUN_IDENTIFIER,
                    prompt="Hello.",
                    event_handler=collector,
                )
            )

    assert caught.value is error
    assert collector.kinds == ["run_started", "run_failed"]
    assert collector.events[-1].error_type == "RuntimeError"


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("before_run", ["run_started"]),
        ("before_model", ["run_started"]),
        ("after_model", ["run_started", "model_started", "model_completed"]),
        (
            "before_tool",
            ["run_started", "model_started", "model_completed", "tool_started"],
        ),
        (
            "after_tool",
            ["run_started", "model_started", "model_completed", "tool_started"],
        ),
        ("after_run", ["run_started", "model_started", "model_completed"]),
    ],
)
def test_run_emits_guardrail_failure(
    agent: AgentDefinition, stage: str, expected: list[EventKind]
) -> None:
    """Guardrail failures respect the model and tool operation boundaries."""
    collector = _EventCollector()
    error = GuardrailTripwire("Rejected.")
    reject = AsyncMock(side_effect=error)
    plugin = replace(GuardrailPlugin(name="reject"), **{stage: (reject,)})
    executions: list[str] = []
    function = _local_tool("lookup", {"value": 1}, executions)
    definition = replace(agent, tools=[function], guardrail_plugins=(plugin,))
    response = _answer("Done.")
    if stage in ("before_tool", "after_tool"):
        response = _request(
            ToolCall(id="lookup-1", name="lookup", arguments='{"value": 1}')
        )
    llm = _ScriptedLlm(response)

    with pytest.raises(GuardrailTripwire) as caught:
        asyncio.run(
            AgentHarness.run(
                definition,
                run_identifier=_RUN_IDENTIFIER,
                prompt="Lookup.",
                llm=llm,
                event_handler=collector,
            )
        )

    assert caught.value is error
    assert collector.kinds == [*expected, "run_failed"]
    assert collector.events[-1].error_type == "GuardrailTripwire"
    assert executions == (["lookup:1"] if stage == "after_tool" else [])


def test_run_emits_turn_limit_failure(agent: AgentDefinition) -> None:
    """Exhausting turns emits failure after the final tool operation."""
    collector = _EventCollector()
    function = _local_tool("lookup", {"value": 1}, [])
    definition = replace(agent, tools=[function])
    llm = _ScriptedLlm(
        _request(ToolCall(id="lookup-1", name="lookup", arguments='{"value": 1}'))
    )

    with pytest.raises(RuntimeError, match="Maximum model turns exceeded"):
        asyncio.run(
            AgentHarness.run(
                definition,
                run_identifier=_RUN_IDENTIFIER,
                prompt="Lookup.",
                llm=llm,
                max_turns=1,
                event_handler=collector,
            )
        )

    assert collector.kinds == [
        "run_started",
        "model_started",
        "model_completed",
        "tool_started",
        "tool_completed",
        "run_failed",
    ]
    assert collector.events[-1].error_type == "RuntimeError"


@pytest.mark.parametrize("stage", ["model", "tool"])
def test_run_emits_cancellation(agent: AgentDefinition, stage: str) -> None:
    """Cancel a real task during a blocked operation without timing sleeps."""
    collector = _EventCollector()

    async def _exercise_cancellation() -> None:
        entered = asyncio.Event()
        released = asyncio.Event()
        interrupted = asyncio.Event()

        async def _block(*args: object, **kwargs: object) -> None:
            entered.set()
            try:
                await released.wait()
            finally:
                interrupted.set()

        llm = AsyncMock()
        definition = agent
        if stage == "model":
            llm.complete.side_effect = _block
        else:
            server = _server(_remote_tool("lookup"))
            server.call_tool.side_effect = _block
            definition = replace(agent, mcp_clients=[server])
            llm.complete.return_value = _request(
                ToolCall(id="lookup-1", name="lookup", arguments='{"query": "item"}')
            )

        task = asyncio.create_task(
            AgentHarness.run(
                definition,
                run_identifier=_RUN_IDENTIFIER,
                prompt="Lookup.",
                llm=llm,
                event_handler=collector,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            assert task.cancelled()
            assert interrupted.is_set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(_exercise_cancellation())

    expected: list[EventKind] = ["run_started", "model_started"]
    if stage == "tool":
        expected.extend(["model_completed", "tool_started"])
    assert collector.kinds == [*expected, "run_cancelled"]
    assert collector.events[-1].error_type is None
    _assert_metadata(collector.events, agent)


@pytest.mark.parametrize("model_fails", [False, True])
def test_run_logs_handler_errors_without_changing_outcome(
    agent: AgentDefinition, caplog: pytest.LogCaptureFixture, model_fails: bool
) -> None:
    """Observer errors neither abort success nor replace an execution error."""
    collector = _EventCollector()
    error = ValueError("Model failed.")
    llm = _ScriptedLlm(error if model_fails else _answer("Done."))

    async def _reject_event(event: LifecycleEvent) -> None:
        await collector(event)
        raise RuntimeError("Observer failed.")

    with caplog.at_level(logging.ERROR, logger="metisa_harness._lifecycle"):
        if model_fails:
            with pytest.raises(ValueError) as caught:
                asyncio.run(
                    AgentHarness.run(
                        agent,
                        run_identifier=_RUN_IDENTIFIER,
                        prompt="Hello.",
                        llm=llm,
                        event_handler=_reject_event,
                    )
                )
            assert caught.value is error
            assert collector.kinds == ["run_started", "model_started", "run_failed"]
        else:
            result = asyncio.run(
                AgentHarness.run(
                    agent,
                    run_identifier=_RUN_IDENTIFIER,
                    prompt="Hello.",
                    llm=llm,
                    event_handler=_reject_event,
                )
            )
            assert result.final_output == "Done."
            assert collector.kinds == [
                "run_started",
                "model_started",
                "model_completed",
                "run_completed",
            ]

    records = [
        record
        for record in caplog.records
        if record.name == "metisa_harness._lifecycle"
    ]
    assert len(records) == len(collector.events)
    for record, event in zip(records, collector.events, strict=True):
        assert record.getMessage() == f"Lifecycle handler failed for {event.kind}"
        assert record.exc_info is not None
        assert isinstance(record.exc_info[1], RuntimeError)


def _assert_metadata(events: list[LifecycleEvent], agent: AgentDefinition) -> None:
    assert events
    assert all(event.run_identifier == _RUN_IDENTIFIER for event in events)
    assert all(event.agent_name == agent.name for event in events)
    elapsed = [event.elapsed_seconds for event in events]
    assert all(value >= 0 for value in elapsed)
    assert elapsed == sorted(elapsed)
    for event in events:
        if event.kind not in ("tool_started", "tool_completed"):
            assert event.tool_name is None
            assert event.tool_call_id is None
        if event.kind != "run_failed":
            assert event.error_type is None
