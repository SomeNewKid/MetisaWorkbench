"""Resolve a model alias to a model implementaton."""

from __future__ import annotations

from ._chat_completions import (
    OpenAiLlmClient,
)
from .models import (
    AgentDefinition,
    LlmClient,
)

_BROKER_BASE_URL = "http://metisa-llm-broker:8000/v1"


async def get_llm(
    agent: AgentDefinition,
) -> LlmClient | None:
    """Get the LLM client for the specified agent."""
    return OpenAiLlmClient(
        model=agent.model,
        base_url=_BROKER_BASE_URL,
        api_key="ollama",
        trust_env=False,
    )
