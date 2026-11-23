"""Manages the building of model contexts."""

from __future__ import annotations

from .models import (
    AgentDefinition,
    ContextBuilder,
    LlmMessage,
    ModelContext,
    RegisteredTool,
)


class FullHistoryContextBuilder(ContextBuilder):
    """Use the full history to build the model context."""

    def build(
        self,
        *,
        agent: AgentDefinition,
        tool_registry: dict[str, RegisteredTool],
        messages: list[LlmMessage],
    ) -> ModelContext:
        """Build the context for presentation to the model."""
        tools = [registered.tool for registered in tool_registry.values()]
        return ModelContext(
            instructions=agent.instructions,
            messages=messages,
            tools=tools,
        )
