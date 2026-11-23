"""Apply registered guardrails in sequence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from .guardrails import (
    Guardrail,
    GuardrailContext,
    GuardrailMetadata,
)

_Payload = TypeVar("_Payload")


async def apply_guardrails(
    guardrails: Iterable[Guardrail[_Payload]],
    *,
    metadata: GuardrailMetadata,
    payload: _Payload,
) -> _Payload:
    """Pass a payload through each guardrail in registration order."""
    for guardrail in guardrails:
        context = GuardrailContext(metadata, payload)
        payload = await guardrail(context)

    return payload
