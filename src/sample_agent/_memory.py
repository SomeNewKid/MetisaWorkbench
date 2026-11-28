"""Memory management for the sample agent."""

from __future__ import annotations

import json

import requests

from metisa_common.mcp_helper import with_output_schema
from metisa_common.secrets_helper import load_secrets

_SAVE_MEMORY_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "identifier": {"type": "string"},
    },
    "required": ["identifier"],
    "additionalProperties": False,
}


def get_snapshot() -> str:
    """Get a snapshot of the agent memory database."""
    secrets = load_secrets()
    agent_episode_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_episode_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_episode_endpoint = "http://metisa-agent-memory:8000/snapshot"
    headers = {
        "Accept": "text/plain",
        "Authorization": f"Bearer {agent_episode_token}",
    }

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid
            response = session.get(
                agent_episode_endpoint,
                headers=headers,
                timeout=(5, 240),
            )
            response.raise_for_status()
            snapshot = response.text
            return snapshot
    except Exception as error:
        raise RuntimeError(f"Agent memory returned an error: {error}") from error


def get_episode(
    episode_id: str,
) -> dict[str, object]:
    """Get the identified episode."""
    secrets = load_secrets()
    agent_episode_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_episode_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_episode_endpoint = f"http://metisa-agent-memory:8000/episodes/{episode_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_episode_token}",
    }

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid
            response = session.get(
                agent_episode_endpoint,
                headers=headers,
                timeout=(5, 240),
            )
            response.raise_for_status()
            episode = response.json()
            return episode
    except Exception as error:
        raise RuntimeError(f"Agent memory returned an error: {error}") from error


def save_episode(
    agent_name: str,
    run_id: str,
    user_messages: list[str],
) -> str:
    """Save a single episode."""
    if not agent_name:
        raise ValueError("Agent name is required.")
    if not run_id:
        raise ValueError("Run ID is required.")

    secrets = load_secrets()
    agent_episode_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_episode_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_episode_endpoint = "http://metisa-agent-memory:8000/episodes"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_episode_token}",
    }
    payload = {
        "kind": "user_message",
        "agent_id": agent_name,
        "run_id": run_id,
        "content": {"user_messages": user_messages},
    }

    content: str | None = None
    result: dict[str, object] | None = None

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid
            response = session.post(
                agent_episode_endpoint,
                json=payload,
                headers=headers,
                timeout=(5, 240),
            )
            content = response.text
            response.raise_for_status()
        if not content:
            raise RuntimeError("Agent memory did not return any content.")
    except Exception as error:
        raise RuntimeError(f"Agent memory returned an error: {error}") from error

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("Agent memory returned invalid JSON.") from error

    if result is None:
        raise RuntimeError("Agent memory returned JSON which could not be loaded.")

    identifier = result.get("id", None)

    if identifier is None:
        raise RuntimeError("Agent memory JSON response did not include 'id'.")
    if not isinstance(identifier, str):
        raise RuntimeError("Agent memory JSON 'id' was not a string.")

    return identifier


@with_output_schema(_SAVE_MEMORY_RESULT_SCHEMA)
def save_memory(
    key: str,
    value: str,
) -> dict[str, object]:
    """Save a single fact about the user."""
    if not key:
        raise ValueError("Key is required.")
    if not value:
        raise ValueError("Value is required.")

    secrets = load_secrets()
    agent_memory_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_memory_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_memory_endpoint = "http://metisa-agent-memory:8000/memories"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_memory_token}",
    }
    payload = {
        "subject": key,
        "content": {"fact": value},
        "text": value,
    }

    content: str | None = None
    result: dict[str, object] | None = None

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid
            response = session.post(
                agent_memory_endpoint,
                json=payload,
                headers=headers,
                timeout=(5, 240),
            )
            content = response.text
            response.raise_for_status()
        if not content:
            raise RuntimeError("Agent memory did not return any content.")
    except Exception as error:
        raise RuntimeError(f"Agent memory returned an error: {error}") from error

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("Agent memory returned invalid JSON.") from error

    if result is None:
        raise RuntimeError("Agent memory returned JSON which could not be loaded.")

    identifier = result.get("id", None)

    if identifier is None:
        raise RuntimeError("Agent memory JSON response did not include 'id'.")
    if not isinstance(identifier, str):
        raise RuntimeError("Agent memory JSON 'id' was not a string.")

    return {"identifier": identifier}


def get_all_memories() -> dict[str, str]:
    """Get all memories."""
    secrets = load_secrets()
    agent_memory_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_memory_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_memory_endpoint = "http://metisa-agent-memory:8000/memories"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_memory_token}",
    }
    page_size = 10
    params = {"limit": f"{page_size}"}

    memories: dict[str, str] = {}

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid

            while True:
                response = session.get(
                    agent_memory_endpoint,
                    params=params,
                    headers=headers,
                    timeout=(5, 240),
                )
                response.raise_for_status()
                page = response.json()
                if not isinstance(page, dict):
                    raise RuntimeError("Page was of the wrong type.")

                items = page["items"]
                for item in items:
                    subject = item["subject"]
                    text = item["text"]
                    if not isinstance(subject, str) or not subject.strip():
                        continue
                    if not isinstance(text, str) or not text.strip():
                        continue
                    memories[subject] = text

                next_cursor = page["next_cursor"]
                if next_cursor is None:
                    break
                params["after"] = next_cursor

            return memories

    except Exception as error:
        raise RuntimeError(f"Agent memory returned an error: {error}") from error
