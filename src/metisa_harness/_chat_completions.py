from __future__ import annotations

import json
import logging

import httpx2
from mcp.types import Tool
from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)

from .models import (
    LlmMessage,
    ModelContext,
    ToolCall,
)

_LOGGER = logging.getLogger(__name__)


class OpenAiLlmClient:
    """Adapt an OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        trust_env: bool = True,
        supports_tool_choice: bool = True,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._trust_env = trust_env
        self._supports_tool_choice = supports_tool_choice

    async def complete(
        self,
        *,
        model_context: ModelContext,
    ) -> LlmMessage:
        tool_definitions = await self._create_tool_definitions(model_context.tools)

        instructions = model_context.instructions

        if model_context.output_schema is not None:
            schema_json = json.dumps(model_context.output_schema, ensure_ascii=False)
            instructions += (
                "\n\n"
                "When giving your final answer, return only JSON "
                "matching the following following schema. "
                "Do not include Markdown fences or  explanatory prose."
                "\n"
                f"{schema_json}"
            )

        request_messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": instructions,
            }
        ]
        converted_messages = self._create_messages(model_context.messages)
        request_messages.extend(converted_messages)

        tool_choice_label = "omitted"
        if tool_definitions and self._supports_tool_choice:
            tool_choice_label = "auto"

        request_details = {
            "model": self._model,
            "base_url": self._base_url,
            "messages": request_messages,
            "tools": tool_definitions,
            "tool_choice": tool_choice_label,
        }

        _LOGGER.debug(
            "LLM request: %s",
            json.dumps(request_details, ensure_ascii=False, indent=2),
        )

        async with AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            http_client=httpx2.AsyncClient(trust_env=self._trust_env),
        ) as client:
            completion = await client.chat.completions.create(
                model=self._model,
                messages=request_messages,
                tools=tool_definitions if tool_definitions else omit,
                tool_choice=(
                    "auto" if tool_definitions and self._supports_tool_choice else omit
                ),
            )

            _LOGGER.debug(
                "LLM response request_id=%s: %s",
                completion._request_id,
                completion.model_dump_json(),
            )

        if not completion.choices:
            raise RuntimeError("OpenAI returned no completion choices.")

        choice = completion.choices[0]

        if choice.finish_reason not in ("stop", "tool_calls"):
            raise RuntimeError(f"OpenAI completion ended with: {choice.finish_reason}")

        message = choice.message
        tool_calls: list[ToolCall] = []

        for requested in message.tool_calls or []:
            if requested.type != "function":
                raise RuntimeError("Unsupported tool call type.")

            tool_calls.append(
                ToolCall(
                    id=requested.id,
                    name=requested.function.name,
                    arguments=requested.function.arguments,
                )
            )

        if choice.finish_reason == "tool_calls" and not tool_calls:
            raise RuntimeError("OpenAI returned no tool calls.")

        return LlmMessage(
            role="assistant",
            content=message.content,
            tool_calls=tool_calls,
        )

    @staticmethod
    async def _create_tool_definitions(
        tools: list[Tool],
    ) -> list[ChatCompletionFunctionToolParam]:
        definitions: list[ChatCompletionFunctionToolParam] = []

        for tool in tools:
            definition: ChatCompletionFunctionToolParam = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                    "strict": False,
                },
            }
            definitions.append(definition)

        return definitions

    @staticmethod
    def _create_messages(
        messages: list[LlmMessage],
    ) -> list[ChatCompletionMessageParam]:
        converted: list[ChatCompletionMessageParam] = []

        for message in messages:
            if message.role == "user":
                converted.append(
                    {
                        "role": "user",
                        "content": message.content or "",
                    }
                )

            elif message.role == "assistant":
                converted.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": tool_call.arguments,
                                },
                            }
                            for tool_call in message.tool_calls
                        ],
                    }
                )

            else:
                if message.tool_call_id is None:
                    raise ValueError("Tool results require a tool call ID.")

                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content or "",
                    }
                )

        return converted
