"""Implement an AI agent harness."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

from opentelemetry import trace

from ._context_manager import FullHistoryContextBuilder
from ._guardrails import apply_guardrails
from ._lifecycle import LifecycleEmitter
from ._model_resolver import get_llm
from ._skills import get_skill_catalog
from ._tools import get_tool_registry, invoke_tool_call
from .guardrails import GuardrailMetadata
from .lifecycle import EventHandler
from .models import (
    AgentDefinition,
    ContextBuilder,
    LlmClient,
    LlmMessage,
    ModelContext,
    RunResult,
)

_TRACER = trace.get_tracer(__name__)


class AgentHarness:
    """Execute an agent workload."""

    @staticmethod
    async def run(
        agent: AgentDefinition,
        prompt: str,
        *,
        run_identifier: str,
        llm: LlmClient | None = None,
        context_builder: ContextBuilder | None = None,
        max_turns: int = 10,
        event_handler: EventHandler | None = None,
    ) -> RunResult:
        """Execute an agent and report its lifecycle."""
        events = LifecycleEmitter(agent.name, run_identifier, event_handler)

        try:
            await events.emit("run_started")

            result = await AgentHarness._run(
                agent,
                input=prompt,
                llm=llm,
                context_builder=context_builder,
                max_turns=max_turns,
                events=events,
            )
        except asyncio.CancelledError:
            await events.emit("run_cancelled")
            raise
        except Exception as error:
            error_type = type(error).__name__
            await events.emit("run_failed", error_type=error_type)
            raise
        else:
            await events.emit("run_completed")
            return result

    @staticmethod
    async def _run(
        agent: AgentDefinition,
        *,
        input: str,
        llm: LlmClient | None = None,
        context_builder: ContextBuilder | None = None,
        max_turns: int = 10,
        events: LifecycleEmitter,
    ) -> RunResult:
        """Run model turns and tools until a final answer is produced."""
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        if llm is None:
            llm = await get_llm(agent)

        if llm is None:
            raise RuntimeError("LLM not available.")

        skill_catalog = get_skill_catalog(agent)

        tool_registry = await get_tool_registry(agent, skill_catalog)

        output_schema = None
        if agent.output_type is not None:
            output_schema = agent.output_type.model_json_schema()

        guardrail_metadata = GuardrailMetadata(
            agent_name=agent.name,
        )

        before_run_guardrails = (
            guardrail
            for plugin in agent.guardrail_plugins
            for guardrail in plugin.before_run
        )

        input = await apply_guardrails(
            before_run_guardrails,
            metadata=guardrail_metadata,
            payload=input,
        )

        messages = [LlmMessage(role="user", content=input)]

        if context_builder is None:
            context_builder = FullHistoryContextBuilder()

        skill_section = ""
        if skill_catalog:
            skill_lines: list[str] = ["Available skills:"]
            skill_lines.extend(
                [f"- {skill.name}: {skill.description}" for skill in skill_catalog]
            )
            skill_section = "\n".join(skill_lines)

        for turn in range(1, max_turns + 1):
            model_context = context_builder.build(
                agent=agent,
                tool_registry=tool_registry,
                messages=messages,
            )
            instructions = model_context.instructions
            if skill_section:
                instructions += f"\n{skill_section}"
            model_context = replace(
                model_context,
                output_schema=output_schema,
                instructions=instructions,
            )

            before_model_guardrails = (
                guardrail
                for plugin in agent.guardrail_plugins
                for guardrail in plugin.before_model
            )

            model_context = await apply_guardrails(
                before_model_guardrails,
                metadata=guardrail_metadata,
                payload=model_context,
            )

            if not isinstance(model_context, ModelContext):
                raise RuntimeError("Invalid type for model context.")

            await events.emit("model_started", turn=turn)

            with _TRACER.start_as_current_span(
                "metisa.model_turn",
                attributes={
                    "metisa.agent.name": agent.name,
                    "metisa.turn": turn,
                    "gen_ai.request.model": agent.model,
                },
                record_exception=False,
            ) as turn_span:
                response = await llm.complete(model_context=model_context)
                turn_span.set_attribute(
                    "metisa.model.tool_call_count",
                    len(response.tool_calls),
                )
                model_span_context = turn_span.get_span_context()

            await events.emit("model_completed", turn=turn)

            after_model_guardrails = (
                guardrail
                for plugin in agent.guardrail_plugins
                for guardrail in plugin.after_model
            )

            response = await apply_guardrails(
                after_model_guardrails,
                metadata=guardrail_metadata,
                payload=response,
            )

            messages.append(response)

            if not response.tool_calls:
                final_text = response.content

                if final_text is None or not final_text.strip():
                    raise RuntimeError("The model returned no final text.")

                if agent.output_type is None:
                    final_output = final_text
                else:
                    final_output = json.loads(final_text)
                    if not isinstance(final_output, dict):
                        raise TypeError(
                            "The model's final output must be a JSON object."
                        )

                after_run_guardrails = (
                    guardrail
                    for plugin in agent.guardrail_plugins
                    for guardrail in plugin.after_run
                )

                final_output = await apply_guardrails(
                    after_run_guardrails,
                    metadata=guardrail_metadata,
                    payload=final_output,
                )

                if agent.output_type is not None:
                    if not isinstance(final_output, dict):
                        raise TypeError("An output guardrail returned the wrong type.")

                    final_output = agent.output_type.model_validate(final_output)
                else:
                    if not isinstance(final_output, str) or not final_output.strip():
                        raise TypeError("An output guardrail returned invalid text.")

                history = tuple(deepcopy(messages))

                return RunResult(final_output=final_output, messages=history)

            model_links = (
                [trace.Link(model_span_context)] if model_span_context.is_valid else []
            )
            for tool_call in response.tool_calls:
                await events.emit(
                    "tool_started",
                    turn=turn,
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                )

                with _TRACER.start_as_current_span(
                    f"execute_tool {tool_call.name}",
                    links=model_links,
                    attributes={
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": tool_call.name,
                        "gen_ai.tool.call.id": tool_call.id,
                        "metisa.turn": turn,
                    },
                    record_exception=False,
                    set_status_on_exception=False,
                ) as tool_span:
                    try:
                        output = await invoke_tool_call(
                            agent,
                            tool_registry,
                            tool_call,
                        )
                    except Exception as error:
                        tool_span.set_attribute("error.type", type(error).__name__)
                        tool_span.set_status(trace.StatusCode.ERROR)
                        raise

                await events.emit(
                    "tool_completed",
                    turn=turn,
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                )

                messages.append(
                    LlmMessage(
                        role="tool",
                        content=output,
                        tool_call_id=tool_call.id,
                    )
                )

        raise RuntimeError("Maximum model turns exceeded.")
