"""Describe observable events in an agent run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

EventKind = Literal[
    "run_started",
    "run_completed",
    "run_failed",
    "run_cancelled",
    "model_started",
    "model_completed",
    "tool_started",
    "tool_completed",
]


@dataclass(frozen=True)
class LifecycleEvent:
    """Describe a point in an agent run's execution."""

    kind: EventKind
    run_identifier: str
    agent_name: str
    elapsed_seconds: float
    turn: int | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    error_type: str | None = None


class EventHandler(Protocol):
    """Observe lifecycle events."""

    async def __call__(self, event: LifecycleEvent) -> None:
        """Handle a lifecycle event."""
        ...
