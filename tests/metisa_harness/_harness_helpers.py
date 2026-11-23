"""Reusable functions and scripted model responses for harness tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

from mcp import Client
from mcp.types import ListToolsResult, Tool
from openai.types.chat import ChatCompletion

from metisa_common.mcp_helper import with_output_schema
from metisa_harness.models import (
    LlmMessage,
    ModelContext,
    ToolCall,
)


class _ScriptedLlm:
    """Return scripted responses and snapshot history before the harness mutates it."""

    def __init__(self, *responses: LlmMessage | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[LlmMessage], list[Tool]]] = []

    async def complete(self, *, model_context: ModelContext) -> LlmMessage:
        self.calls.append((deepcopy(model_context.messages), list(model_context.tools)))
        if not self.responses:
            raise AssertionError("Unexpected model request.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _answer(text: str) -> LlmMessage:
    return LlmMessage(role="assistant", content=text)


def _request(*calls: ToolCall) -> LlmMessage:
    return LlmMessage(role="assistant", content=None, tool_calls=list(calls))


def _server(*tools: Tool) -> MagicMock:
    server = MagicMock(spec=Client)
    server.list_tools = AsyncMock(return_value=ListToolsResult(tools=list(tools)))
    server.call_tool = AsyncMock()
    return server


def _local_tool(
    name: str,
    result: object,
    events: list[str],
    *,
    asynchronous: bool = False,
    error: Exception | None = None,
) -> Callable[..., object]:
    def _execute(value: int) -> object:
        events.append(f"{name}:{value}")
        if error is not None:
            raise error
        return result

    async def _execute_async(value: int) -> object:
        await asyncio.sleep(0)
        return _execute(value)

    function = _execute_async if asynchronous else _execute
    function.__name__ = name
    return with_output_schema(_value_schema())(function)


def _value_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }


@with_output_schema(
    {
        "type": "object",
        "properties": {"datetime": {"type": "string"}},
        "required": ["datetime"],
        "additionalProperties": False,
    }
)
def _get_current_date_and_time() -> dict[str, object]:
    """Return the current date and time."""
    raise AssertionError("Tool discovery must not execute local functions.")


def _remote_tool(name: str) -> Tool:
    return Tool(
        name=name,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema=_value_schema(),
    )


def _completion(
    text: str | None,
    *,
    calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> ChatCompletion:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {
                            "id": item.id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                        }
                        for item in calls or []
                    ],
                },
            }
        ],
    }
    completion = ChatCompletion.model_validate(payload)
    completion._request_id = "req_test"
    return completion
