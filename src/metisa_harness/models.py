"""Data models for the Metisa Harness module."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from mcp import Client
from mcp.types import Tool
from pydantic import BaseModel

from .guardrails import Guardrail


@dataclass(frozen=True)
class AgentDefinition:
    """Define an agent."""

    name: str
    description: str
    model: str
    instructions: str
    tools: list[Callable[..., dict[str, object]]] = field(default_factory=list)
    mcp_clients: list[Client] = field(default_factory=list)
    sub_agents: list[AgentDefinition] = field(default_factory=list)
    skills_directory: Path | None = None
    guardrail_plugins: tuple[GuardrailPlugin, ...] = ()

    output_type: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        """Validate the initialized Agent Definition."""
        if not self.name.strip():
            raise ValueError("Agent name is required.")
        if not self.model.strip():
            raise ValueError("Agent model is required.")
        if not self.instructions.strip():
            raise ValueError("Agent instructions are required.")


@dataclass(frozen=True)
class RunResult:
    """Contain the final answer and conversation from a completed run."""

    final_output: str | BaseModel
    messages: tuple[LlmMessage, ...]


@dataclass(frozen=True)
class ToolCall:
    """Represent a model's request to invoke a tool."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class RegisteredTool:
    """Represent a tool registered with the Metisa Harness."""

    tool: Tool
    invoke: Callable[[dict[str, object]], Awaitable[dict[str, object]]]


@dataclass(frozen=True)
class LlmMessage:
    """Represent a conversation message exchanged with a model."""

    role: Literal["user", "assistant", "tool"]
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


class LlmClient(Protocol):
    """Provide model responses."""

    async def complete(
        self,
        *,
        model_context: ModelContext,
    ) -> LlmMessage:
        """Return one model response, possibly containing tool calls."""
        ...


@dataclass(frozen=True)
class ModelContext:
    """Represent the context presented to the model for its next decision."""

    instructions: str
    messages: list[LlmMessage]
    tools: list[Tool]
    output_schema: dict[str, object] | None = None


class ContextBuilder(Protocol):
    """Construct the context presented to the model for its next decision."""

    def build(
        self,
        *,
        agent: AgentDefinition,
        tool_registry: dict[str, RegisteredTool],
        messages: list[LlmMessage],
    ) -> ModelContext:
        """Build the context for presentation to the model."""
        ...


@dataclass(frozen=True)
class GuardrailPlugin:
    """Package related guardrails for reuse across agents."""

    name: str

    before_run: tuple[Guardrail[str], ...] = ()
    after_run: tuple[Guardrail[str | dict[str, object]], ...] = ()

    before_model: tuple[Guardrail[ModelContext], ...] = ()
    after_model: tuple[Guardrail[LlmMessage], ...] = ()

    before_tool: tuple[Guardrail[dict[str, object]], ...] = ()
    after_tool: tuple[Guardrail[dict[str, object]], ...] = ()


@dataclass
class AgentSkill:
    """Represent a single agent skill."""

    name: str
    description: str
    content: str
    directory: Path
