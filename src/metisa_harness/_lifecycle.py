"""Dispatch lifecycle events to an optional observer."""

from __future__ import annotations

import logging
import time

from .lifecycle import EventHandler, EventKind, LifecycleEvent

_LOGGER = logging.getLogger(__name__)


class LifecycleEmitter:
    """Create and deliver events for one run."""

    def __init__(
        self,
        agent_name: str,
        run_identifier: str,
        event_handler: EventHandler | None,
    ) -> None:
        self._agent_name = agent_name
        self._run_identifier = run_identifier
        self._event_handler = event_handler
        self._started_at = time.monotonic()

    async def emit(
        self,
        kind: EventKind,
        *,
        turn: int | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        error_type: str | None = None,
    ) -> None:
        """Deliver an event."""
        if self._event_handler is None:
            return

        lifecycle_event = LifecycleEvent(
            kind=kind,
            agent_name=self._agent_name,
            run_identifier=self._run_identifier,
            elapsed_seconds=time.monotonic() - self._started_at,
            turn=turn,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            error_type=error_type,
        )

        try:
            await self._event_handler(lifecycle_event)
        except Exception:
            _LOGGER.exception("Lifecycle handler failed for %s", kind)
