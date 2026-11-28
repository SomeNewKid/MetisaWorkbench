from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from uuid import uuid4

from jsonschema import validate
from mcp import Client
from mcp.types import Tool
from opentelemetry import trace

from metisa_common.mcp_helper import MODEL_REASONING_FIELD_NAME, create_tool

from ._guardrails import apply_guardrails
from .guardrails import GuardrailMetadata
from .models import AgentDefinition, AgentSkill, RegisteredTool, ToolCall


async def invoke_tool_call(
    agent: AgentDefinition,
    tool_registry: dict[str, RegisteredTool],
    tool_call: ToolCall,
) -> str:
    """Invoke the specified tool call using the registered tools."""
    registered = tool_registry.get(tool_call.name)

    if registered is None:
        raise RuntimeError(f"The model requested an unknown tool: {tool_call.name}")

    arguments = json.loads(tool_call.arguments)

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    original_properties = registered.tool.input_schema.get("properties", {})
    if MODEL_REASONING_FIELD_NAME not in original_properties:
        reasoning = arguments.pop(MODEL_REASONING_FIELD_NAME, None)
        if isinstance(reasoning, str):
            reasoning = reasoning.strip()
            if reasoning:
                tool_span = trace.get_current_span()
                tool_span.set_attribute("metisa.tool.model_reasoning", reasoning[:500])

    metadata = GuardrailMetadata(
        agent_name=agent.name,
        tool_name=tool_call.name,
    )

    before_tool_guardrails = (
        guardrail
        for plugin in agent.guardrail_plugins
        for guardrail in plugin.before_tool
    )

    arguments = await apply_guardrails(
        before_tool_guardrails,
        metadata=metadata,
        payload=arguments,
    )

    validate(
        instance=arguments,
        schema=registered.tool.input_schema,
    )

    arguments_json = json.dumps(arguments, ensure_ascii=False)
    tool_span = trace.get_current_span()
    tool_span.set_attribute("metisa.tool.arguments", arguments_json)

    result = await registered.invoke(arguments)

    after_tool_guardrails = (
        guardrail
        for plugin in agent.guardrail_plugins
        for guardrail in plugin.after_tool
    )

    result = await apply_guardrails(
        after_tool_guardrails,
        metadata=metadata,
        payload=result,
    )

    if registered.tool.output_schema is not None:
        validate(
            instance=result,
            schema=registered.tool.output_schema,
        )

    return json.dumps(result, allow_nan=False)


async def get_tool_registry(
    agent: AgentDefinition,
    skill_catalog: list[AgentSkill],
) -> dict[str, RegisteredTool]:
    """Get the tool registry for the specified agent."""
    registered_tools: list[RegisteredTool] = []

    for function in agent.tools:
        registered = _register_local_tool(function)
        registered_tools.append(registered)

    for server in agent.mcp_clients:
        cursor: str | None = None

        while True:
            page = await server.list_tools(cursor=cursor)

            for tool in page.tools:
                registered = _register_mcp_tool(server, tool)
                registered_tools.append(registered)

            cursor = page.next_cursor
            if cursor is None:
                break

    for sub_agent in agent.sub_agents:
        registered = _register_sub_agent(sub_agent)
        registered_tools.append(registered)

    if skill_catalog:
        registered = _register_load_skill_tool(skill_catalog)
        registered_tools.append(registered)

        registered = _register_get_skill_document_tool(skill_catalog)
        registered_tools.append(registered)

        registered = _register_execute_skill_script_tool(skill_catalog)
        registered_tools.append(registered)

    registry: dict[str, RegisteredTool] = {}

    for registered in registered_tools:
        name = registered.tool.name

        if name in registry:
            raise ValueError(f"Duplicate tool name: {name}")

        registry[name] = registered

    return registry


def _register_local_tool(
    function: Callable[..., dict[str, object]],
) -> RegisteredTool:
    tool = create_tool(function)

    async def _invoke(arguments: dict[str, object]) -> dict[str, object]:
        # Run synchronous functions outside the event-loop thread.
        # Calling an async function here returns its corouting.
        result = await asyncio.to_thread(function, **arguments)

        if inspect.isawaitable(result):
            result = await result

        if tool.output_schema is not None:
            validate(instance=result, schema=tool.output_schema)

        return result

    return RegisteredTool(tool=tool, invoke=_invoke)


def _register_mcp_tool(
    server: Client,
    tool: Tool,
) -> RegisteredTool:
    async def _invoke(arguments: dict[str, object]) -> dict[str, object]:
        result = await server.call_tool(tool.name, arguments)

        if result.is_error:
            raise RuntimeError(f"MCP tool failed: {tool.name}")

        payload = result.structured_content
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"MCP tool {tool.name} did not returned structured JSON."
            )

        if tool.output_schema is not None:
            validate(instance=payload, schema=tool.output_schema)

        return payload

    return RegisteredTool(tool=tool, invoke=_invoke)


def _register_sub_agent(
    sub_agent: AgentDefinition,
) -> RegisteredTool:
    if sub_agent.sub_agents:
        raise ValueError(f"Sub-agent {sub_agent.name} cannot delegate to sub-agents.")

    output_type = sub_agent.output_type
    output_schema = dict[str, object]

    if output_type is None:
        output_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
            "additionalProperties": False,
        }
    else:
        output_schema = output_type.model_json_schema(
            mode="serialization",
            by_alias=True,
        )

    tool = Tool(
        name=sub_agent.name,
        description=sub_agent.description,
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or text to pass to this agent.",
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        output_schema=output_schema,
    )

    async def _invoke(arguments: dict[str, object]) -> dict[str, object]:
        # harness.py imports this module, so defer the import until invocation.
        from .harness import AgentHarness

        task = arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("The sub-agent task must be non-empty text.")

        run_identifier = uuid4().hex
        result = await AgentHarness.run(
            sub_agent,
            run_identifier=run_identifier,
            prompt=task,
        )

        answer = result.final_output

        if output_type is None:
            if not isinstance(answer, str):
                raise TypeError(f"Sub-agent {sub_agent.name} returned non-text output.")

            return {
                "answer": answer,
            }

        if not isinstance(answer, output_type):
            raise TypeError(
                f"Sub-agent {sub_agent.name} returned an unexpected model type."
            )

        payload = answer.model_dump(mode="json", by_alias=True)
        if not isinstance(payload, dict):
            raise TypeError(
                f"Sub-agent {sub_agent.name} must serialize to a JSON object."
            )

        return payload

    return RegisteredTool(tool=tool, invoke=_invoke)


def _register_load_skill_tool(
    skill_catalog: list[AgentSkill],
) -> RegisteredTool:
    tool = Tool(
        name="load_skill",
        description="Get the contents of the skill file.",
        input_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill.",
                },
            },
            "required": ["skill_name"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    )

    async def _invoke(arguments: dict[str, object]) -> dict[str, object]:
        skill_name = arguments.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("The skill name is required.")

        for agent_skill in skill_catalog:
            if agent_skill.name == skill_name:
                return {"content": agent_skill.content}

        raise RuntimeError(f"There is no skill named '{skill_name}'.")

    return RegisteredTool(tool=tool, invoke=_invoke)


def _register_get_skill_document_tool(
    skill_catalog: list[AgentSkill],
) -> RegisteredTool:
    tool = Tool(
        name="get_skill_document",
        description="Get the contents of a skill document.",
        input_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill.",
                },
                "document_path": {
                    "type": "string",
                    "description": "The document path relative to the skill folder.",
                },
            },
            "required": ["skill_name", "document_path"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "error": {
                    "type": "object",
                    "properties": {
                        "code": {"const": "document_not_found"},
                        "message": {"type": "string"},
                    },
                    "required": ["code", "message"],
                    "additionalProperties": False,
                },
            },
            "oneOf": [
                {"required": ["content"]},
                {"required": ["error"]},
            ],
            "additionalProperties": False,
        },
    )

    async def _invoke(arguments: dict[str, object]) -> dict[str, object]:
        skill_name = arguments.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("The skill name is required.")

        document_path = arguments.get("document_path")
        if not isinstance(document_path, str) or not document_path.strip():
            raise ValueError("The document path is required.")

        named_skill: AgentSkill | None = None

        for agent_skill in skill_catalog:
            if agent_skill.name == skill_name:
                named_skill = agent_skill

        if not named_skill:
            raise RuntimeError(f"There is no skill named '{skill_name}'.")

        document_location = named_skill.directory.joinpath(document_path).resolve()

        skill_directory = named_skill.directory.resolve()

        if skill_directory == document_location or not document_location.is_relative_to(
            skill_directory
        ):
            raise RuntimeError("Document path was not relative to the skill directory.")

        if not document_location.is_file():
            return {
                "error": {
                    "code": "document_not_found",
                    "message": (
                        f"Skill '{skill_name}' has no document at '{document_path}'."
                    ),
                },
            }

        document_contents = document_location.read_text(encoding="utf-8")

        return {"content": document_contents}

    return RegisteredTool(tool=tool, invoke=_invoke)


def _register_execute_skill_script_tool(
    skill_catalog: list[AgentSkill],
) -> RegisteredTool:
    tool = Tool(
        name="execute_skill_script",
        description="Get the result of executing a skill's Starlark script.",
        input_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill.",
                },
                "script_path": {
                    "type": "string",
                    "description": "The script path relative to the skill folder.",
                },
                "script_parameters": {
                    "type": "object",
                    "description": "The parameters to pass to the script.",
                    "additionalProperties": True,
                },
            },
            "required": ["skill_name", "script_path", "script_parameters"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "error": {
                    "type": "object",
                    "properties": {
                        "code": {"const": "script_not_found"},
                        "message": {"type": "string"},
                    },
                    "required": ["code", "message"],
                    "additionalProperties": False,
                },
            },
            "oneOf": [
                {"required": ["result"]},
                {"required": ["error"]},
            ],
            "additionalProperties": False,
        },
    )

    async def _invoke(arguments: dict[str, object]) -> dict[str, object]:
        skill_name = arguments.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("The skill name is required.")

        script_path = arguments.get("script_path")
        if not isinstance(script_path, str) or not script_path.strip():
            raise ValueError("The script path is required.")

        script_parameters = arguments.get("script_parameters")
        if not isinstance(script_parameters, dict):
            raise ValueError("The script parameters are required.")

        named_skill: AgentSkill | None = None

        for agent_skill in skill_catalog:
            if agent_skill.name == skill_name:
                named_skill = agent_skill

        if not named_skill:
            raise RuntimeError(f"There is no skill named '{skill_name}'.")

        script_location = named_skill.directory.joinpath(script_path).resolve()

        skill_directory = named_skill.directory.resolve()

        if skill_directory == script_location or not script_location.is_relative_to(
            skill_directory
        ):
            raise RuntimeError("Script path was not relative to the skill directory.")

        if not script_location.is_file():
            return {
                "error": {
                    "code": "script_not_found",
                    "message": (
                        f"Skill '{skill_name}' has no script at '{script_path}'."
                    ),
                },
            }

        script_contents = script_location.read_text(encoding="utf-8")
        script_result = await _execute_script(script_contents, script_parameters)

        return {"result": script_result}

    async def _execute_script(
        script_contents: str,
        script_parameters: dict[str, object],
    ) -> str:
        from metisa_common.clients.mcp_client import get_mcp_server_tool_result

        tool_result = await asyncio.to_thread(
            get_mcp_server_tool_result,
            "execute_starlark_code",
            params={
                "starlark_code": script_contents,
                "script_parameters": script_parameters,
            },
        )

        result = tool_result.get("result")
        if not isinstance(result, str):
            raise RuntimeError("Script result was not a string.")

        return result

    return RegisteredTool(tool=tool, invoke=_invoke)
