"""Guardrails for the Metisa Harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

_Payload = TypeVar("_Payload")


class GuardrailTripwire(RuntimeError):
    """Abort an agent run because a guardrail rejected processing."""


@dataclass(frozen=True)
class GuardrailMetadata:
    """Describe the execution associated with a guardrail."""

    agent_name: str
    tool_name: str | None = None


@dataclass(frozen=True)
class GuardrailContext(Generic[_Payload]):
    """Present metadata and payload to a guardrail."""

    metadata: GuardrailMetadata
    payload: _Payload


class Guardrail(Protocol[_Payload]):
    """Validate or transform a payload, or raise a tripwire."""

    async def __call__(
        self,
        context: GuardrailContext[_Payload],
    ) -> _Payload:
        """Return the accepted payload or raise a GuardrailTripwire."""
        ...
