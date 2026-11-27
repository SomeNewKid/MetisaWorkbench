"""Agent Memory tests for the sample agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from uuid import uuid4

import requests

from metisa_common.secrets_helper import (
    load_secrets,
)
from metisa_common.specification_models import (
    Capability,
    MetisaSpecification,
)

from .._helper import (
    print_caution,
    print_disabled,
    print_elapsed_time,
    print_failure,
    print_success,
    print_suite_title,
)

_SKIP_AGENT_MEMORY_TESTS = False


def test_agent_memory(specification: MetisaSpecification):
    """Run code execution tests."""
    print_suite_title("Agent Memory")

    if _SKIP_AGENT_MEMORY_TESTS:
        print_caution("Agent memory capability is being skipped.")
        return

    if Capability.AGENT_MEMORY not in specification.capabilities:
        print_disabled("Agent memory capability not declared.")
        return

    episode_summary = "This is a summary of an episode"
    episode_id: str | None = None
    with print_elapsed_time():
        try:
            episode_id = _save_episode(episode_summary)
            if episode_id:
                print_success("Created episode ID:", episode_id)
            else:
                print_failure("Did not create episode.")
        except Exception as error:
            print_failure("Save episode error:", error)

    if episode_id:
        with print_elapsed_time():
            try:
                episode_details = _get_episode(episode_id)
                if episode_details:
                    episode_details_summary = episode_details.get("summary", None)
                    if episode_details_summary:
                        print_success("Retrieved episode:", episode_details_summary)
                    else:
                        print_failure("Did not retrieve episode details.")
                else:
                    print_failure("Did not retrieve episode.")
            except Exception as error:
                print_failure("Retrieve episode error:", error)
    else:
        print_caution("Skipping retrieve episode test.")

    if episode_id:
        with print_elapsed_time():
            try:
                was_deleted = _delete_episode(episode_id)
                if was_deleted:
                    print_success("Deleted episode:", episode_id)
                else:
                    print_failure("Did not delete episode.")
            except Exception as error:
                print_failure("Delete episode error:", error)
    else:
        print_caution("Skipping delete episode test.")

    if episode_id:
        with print_elapsed_time():
            try:
                episode_details = _get_episode(episode_id)
                if episode_details:
                    print_failure("Retrieved deleted episode.", episode_id)
                else:
                    print_success("Did not retrieve deleted episode.")
            except Exception as error:
                print_failure("Retrieve deleted episode error:", error)
    else:
        print_caution("Skipping re-retrieve episode test.")

    with print_elapsed_time():
        _test_list_episodes()

    with print_elapsed_time():
        _test_expired_episodes()

    print(flush=True)
    favorite_color = "blue"
    memory_id: str | None = None
    with print_elapsed_time():
        try:
            memory_id = _save_memory(favorite_color)
            if memory_id:
                print_success("Created memory ID:", memory_id)
            else:
                print_failure("Did not create memory.")
        except Exception as error:
            print_failure("Save memory error:", error)

    if memory_id:
        with print_elapsed_time():
            try:
                memory_details = _get_memory(memory_id)
                if memory_details:
                    memory_details_fact = memory_details.get("fact", None)
                    memory_details_value = memory_details.get("value", None)
                    if memory_details_fact and memory_details_value:
                        if memory_details_value == favorite_color:
                            print_success(
                                "Retrieved memory:",
                                memory_details_fact,
                                memory_details_value,
                            )
                        else:
                            print_failure(
                                "Memory value does not match favorite color:",
                                memory_details_value,
                                favorite_color,
                            )
                    else:
                        print_failure("Did not retrieve memory details.")
                else:
                    print_failure("Did not retrieve memory.")
            except Exception as error:
                print_failure("Retrieve memory error:", error)
    else:
        print_caution("Skipping retrieve memory test.")

    favorite_color = "red"
    if memory_id:
        with print_elapsed_time():
            try:
                memory_id = _update_memory(memory_id, favorite_color)
                if memory_id:
                    print_success("Updated memory ID:", memory_id)
                else:
                    print_failure("Did not update memory.")
            except Exception as error:
                print_failure("Update memory error:", error)

    if memory_id:
        with print_elapsed_time():
            try:
                memory_details = _get_memory(memory_id)
                if memory_details:
                    memory_details_fact = memory_details.get("fact", None)
                    memory_details_value = memory_details.get("value", None)
                    if memory_details_fact and memory_details_value:
                        if memory_details_value == favorite_color:
                            print_success(
                                "Retrieved memory:",
                                memory_details_fact,
                                memory_details_value,
                            )
                        else:
                            print_failure(
                                "Memory value does not match favorite color:",
                                memory_details_value,
                                favorite_color,
                            )
                    else:
                        print_failure("Did not retrieve memory details.")
                else:
                    print_failure("Did not retrieve memory.")
            except Exception as error:
                print_failure("Retrieve memory error:", error)
    else:
        print_caution("Skipping retrieve memory test.")

    if memory_id:
        with print_elapsed_time():
            try:
                was_deleted = _delete_memory(memory_id)
                if was_deleted:
                    print_success("Deleted memory:", memory_id)
                else:
                    print_failure("Did not delete memory.")
            except Exception as error:
                print_failure("Delete memory error:", error)
    else:
        print_caution("Skipping delete memory test.")

    if memory_id:
        with print_elapsed_time():
            try:
                memory_details = _get_memory(memory_id)
                if memory_details:
                    print_failure("Retrieved deleted memory.", memory_id)
                else:
                    print_success("Did not retrieve deleted memory.")
            except Exception as error:
                print_failure("Retrieve deleted memory error:", error)
    else:
        print_caution("Skipping re-retrieve memory test.")

    with print_elapsed_time():
        _test_list_memories()

    with print_elapsed_time():
        _test_expired_memories()

    with print_elapsed_time():
        _test_memory_valid_to()


def _save_episode(
    episode_summary: str,
    *,
    kind: str = "summary",
) -> str:
    if not episode_summary:
        raise ValueError("Episode summary is required.")

    secrets = load_secrets()
    agent_memory_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_memory_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_memory_endpoint = "http://metisa-agent-memory:8000/episodes"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_memory_token}",
    }
    payload = {
        "kind": kind,
        "agent_id": "sample_agent",
        "content": {"summary": episode_summary},
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

    return identifier


def _get_episode(
    episode_id: str,
) -> dict[str, str] | None:
    if not episode_id:
        raise ValueError("Episode ID is required.")

    secrets = load_secrets()
    agent_memory_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_memory_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_memory_endpoint = f"http://metisa-agent-memory:8000/episodes/{episode_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_memory_token}",
    }

    content: str | None = None
    result: dict[str, object] | None = None

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid
            response = session.get(
                agent_memory_endpoint,
                headers=headers,
                timeout=(5, 240),
            )
            content = response.text
            if response.status_code == 404:  # Not found
                return None
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

    if identifier != episode_id:
        error_message = (
            "Agent memory retrieved a different episode. "
            f"Requested '{episode_id}', "
            f"received '{identifier}', "
        )
        raise RuntimeError(error_message)

    content_json = result.get("content", None)

    if content_json is None:
        raise RuntimeError("Agent memory JSON response did not include 'content'.")
    if not isinstance(content_json, dict):
        raise RuntimeError("Agent memory JSON 'content' was not a dictionary.")

    summary = content_json.get("summary", None)
    if summary is None:
        raise RuntimeError("Agent memory JSON content did not include 'summary'.")
    if not isinstance(summary, str):
        raise RuntimeError("Agent memory JSON 'content.summary' was not a string.")

    return {
        "id": identifier,
        "summary": summary,
    }


def _delete_episode(
    episode_id: str,
) -> bool:
    if not episode_id:
        raise ValueError("Episode ID is required.")

    secrets = load_secrets()
    agent_memory_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_memory_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_memory_endpoint = f"http://metisa-agent-memory:8000/episodes/{episode_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_memory_token}",
    }

    status_code: int | None = None

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid
            response = session.delete(
                agent_memory_endpoint,
                headers=headers,
                timeout=(5, 240),
            )
            response.raise_for_status()
            status_code = response.status_code
    except Exception as error:
        raise RuntimeError(f"Agent memory returned an error: {error}") from error

    return status_code == 204  # No content


def _save_memory(
    favorite_color: str,
) -> str:
    if not favorite_color:
        raise ValueError("Favorite color is required.")

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
        "subject": "user_preferences",
        "content": {"fact": "favorite_color", "value": favorite_color},
        "text": f"The user's favorite color is {favorite_color}.",
        "embedding": [0.25, -0.5, 0.75],
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

    return identifier


def _update_memory(
    memory_id: str,
    favorite_color: str,
) -> str:
    if not memory_id:
        raise ValueError("Memory ID is required.")
    if not favorite_color:
        raise ValueError("Favorite color is required.")

    secrets = load_secrets()
    agent_memory_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_memory_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_memory_endpoint = f"http://metisa-agent-memory:8000/memories/{memory_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_memory_token}",
    }
    payload = {
        "subject": "user_preferences",
        "content": {"fact": "favorite_color", "value": favorite_color},
        "text": f"The user's favorite color is {favorite_color}.",
        "embedding": [0.35, -0.4, 0.65],
    }

    content: str | None = None
    result: dict[str, object] | None = None

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid
            response = session.put(
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

    return identifier


def _get_memory(
    memory_id: str,
) -> dict[str, str] | None:
    if not memory_id:
        raise ValueError("Episode ID is required.")

    secrets = load_secrets()
    agent_memory_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_memory_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_memory_endpoint = f"http://metisa-agent-memory:8000/memories/{memory_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_memory_token}",
    }

    content: str | None = None
    result: dict[str, object] | None = None

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid
            response = session.get(
                agent_memory_endpoint,
                headers=headers,
                timeout=(5, 240),
            )
            content = response.text
            if response.status_code == 404:  # Not found
                return None
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

    if identifier != memory_id:
        error_message = (
            "Agent memory retrieved a different memory. "
            f"Requested '{memory_id}', "
            f"received '{identifier}', "
        )
        raise RuntimeError(error_message)

    content_json = result.get("content", None)

    if content_json is None:
        raise RuntimeError("Agent memory JSON response did not include 'content'.")
    if not isinstance(content_json, dict):
        raise RuntimeError("Agent memory JSON 'content' was not a dictionary.")

    fact = content_json.get("fact", None)
    if fact is None:
        raise RuntimeError("Agent memory JSON content did not include 'fact'.")
    if not isinstance(fact, str):
        raise RuntimeError("Agent memory JSON 'content.fact' was not a string.")

    value = content_json.get("value", None)
    if value is None:
        raise RuntimeError("Agent memory JSON content did not include 'value'.")
    if not isinstance(value, str):
        raise RuntimeError("Agent memory JSON 'content.value' was not a string.")

    return {
        "id": identifier,
        "fact": fact,
        "value": value,
    }


def _delete_memory(
    memory_id: str,
) -> bool:
    if not memory_id:
        raise ValueError("Episode ID is required.")

    secrets = load_secrets()
    agent_memory_token = secrets["METISA_AGENT_MEMORY_TOKEN"]
    if not agent_memory_token:
        raise RuntimeError("METISA_AGENT_MEMORY_TOKEN secret is not set.")

    agent_memory_endpoint = f"http://metisa-agent-memory:8000/memories/{memory_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {agent_memory_token}",
    }

    status_code: int | None = None

    try:
        with requests.Session() as session:
            session.trust_env = False  # do not use Squid
            response = session.delete(
                agent_memory_endpoint,
                headers=headers,
                timeout=(5, 240),
            )
            response.raise_for_status()
            status_code = response.status_code
    except Exception as error:
        raise RuntimeError(f"Agent memory returned an error: {error}") from error

    return status_code == 204  # No content


def _test_list_episodes() -> None:
    episode_ids: list[str] = []
    kind = f"listing-smoke-{uuid4().hex}"

    try:
        first_id = _save_episode("First listing example.", kind=kind)
        episode_ids.append(first_id)
        other_id = _save_episode("An unrelated example.", kind=f"{kind}-other")
        episode_ids.append(other_id)
        second_id = _save_episode("Second listing example.", kind=kind)
        episode_ids.append(second_id)

        token = load_secrets()["METISA_AGENT_MEMORY_TOKEN"]
        endpoint = "http://metisa-agent-memory:8000/episodes"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"kind": kind, "limit": "1"}

        with requests.Session() as session:
            session.trust_env = False
            response = session.get(
                endpoint, headers=headers, params=params, timeout=(5, 240)
            )
            response.raise_for_status()
            first_page = response.json()
            cursor = first_page["next_cursor"]
            if not isinstance(cursor, str) or not cursor:
                raise RuntimeError("Expected a cursor after the first page.")

            params["after"] = cursor
            response = session.get(
                endpoint, headers=headers, params=params, timeout=(5, 240)
            )
            response.raise_for_status()
            second_page = response.json()

        expected = [
            (first_page, first_id, "First listing example."),
            (second_page, second_id, "Second listing example."),
        ]
        for page, episode_id, summary in expected:
            items = page["items"]
            if len(items) != 1 or items[0]["id"] != episode_id:
                raise RuntimeError("Episode list returned the wrong record or count.")
            if items[0]["content"]["summary"] != summary:
                raise RuntimeError("Episode list returned the wrong summary.")

        if second_page["next_cursor"] is not None:
            raise RuntimeError("Expected the second page to be the last page.")

        print_success("Listed episodes by kind across two pages.")
    except Exception as error:
        print_failure("Episode listing error:", error)
    finally:
        for episode_id in episode_ids:
            try:
                if not _delete_episode(episode_id):
                    print_failure("Could not clean up listing episode:", episode_id)
            except Exception as error:
                print_failure("Episode listing cleanup error:", error)


def _test_list_memories() -> None:
    memory_ids: list[str] = []
    subject = f"listing-smoke-{uuid4().hex}"
    endpoint = "http://metisa-agent-memory:8000/memories"

    try:
        token = load_secrets()["METISA_AGENT_MEMORY_TOKEN"]
        headers = {"Authorization": f"Bearer {token}"}
        params = {"subject": subject, "limit": "1"}

        with requests.Session() as session:
            session.trust_env = False

            for record_subject, color in (
                (subject, "blue"),
                (f"{subject}-other", "green"),
                (subject, "red"),
            ):
                payload = {
                    "subject": record_subject,
                    "content": {"fact": "favorite_color", "value": color},
                    "text": f"The user's favorite color is {color}.",
                    "embedding": [0.25, -0.5, 0.75],
                }
                response = session.post(
                    endpoint, headers=headers, json=payload, timeout=(5, 240)
                )
                response.raise_for_status()
                memory_ids.append(response.json()["id"])

            first_id, _, second_id = memory_ids
            expected_pages = [(first_id, "blue"), (second_id, "red")]

            for page_number, (expected_id, color) in enumerate(expected_pages, start=1):
                response = session.get(
                    endpoint, headers=headers, params=params, timeout=(5, 240)
                )
                response.raise_for_status()
                page = response.json()
                items = page["items"]

                if len(items) != 1 or items[0]["id"] != expected_id:
                    raise RuntimeError(f"Page {page_number} returned the wrong memory.")

                expected_content = {"fact": "favorite_color", "value": color}
                if items[0]["content"] != expected_content:
                    raise RuntimeError(
                        f"Page {page_number} returned the wrong content."
                    )

                cursor = page["next_cursor"]
                if page_number == 1:
                    if not isinstance(cursor, str) or not cursor:
                        raise RuntimeError("Expected a cursor after the first page.")
                    params["after"] = cursor
                elif cursor is not None:
                    raise RuntimeError("Expected the second page to be the last page.")

        print_success("Listed memories by subject across two pages.")
    except Exception as error:
        print_failure("Memory listing error:", error)
    finally:
        for memory_id in memory_ids:
            try:
                if not _delete_memory(memory_id):
                    print_failure("Could not clean up listing memory:", memory_id)
            except Exception as error:
                print_failure("Memory listing cleanup error:", error)


def _test_expired_episodes() -> None:
    kind = f"expiry-smoke-{uuid4().hex}"
    _check_expired_records(
        collection="episodes",
        filters={"kind": kind},
        record={"kind": kind, "content": {"summary": "An example episode."}},
        delete_record=_delete_episode,
    )


def _test_expired_memories() -> None:
    subject = f"expiry-smoke-{uuid4().hex}"
    _check_expired_records(
        collection="memories",
        filters={"subject": subject},
        record={
            "subject": subject,
            "content": {"fact": "favorite_color", "value": "blue"},
            "text": "The user's favorite color is blue.",
            "embedding": [0.25, -0.5, 0.75],
        },
        delete_record=_delete_memory,
    )


def _check_expired_records(
    *,
    collection: str,
    filters: dict[str, str],
    record: dict[str, object],
    delete_record: Callable[[str], bool],
) -> None:
    record_ids: list[str] = []
    endpoint = f"http://metisa-agent-memory:8000/{collection}"
    timeout = (5, 240)

    try:
        token = load_secrets()["METISA_AGENT_MEMORY_TOKEN"]
        headers = {"Authorization": f"Bearer {token}"}

        with requests.Session() as session:
            session.trust_env = False

            for expires_at in ("2000-01-01T00:00:00Z", "2099-01-01T00:00:00Z"):
                payload = {**record, "expires_at": expires_at}
                response = session.post(
                    endpoint, headers=headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                record_ids.append(response.json()["id"])

            expired_id, active_id = record_ids

            for record_id, expected_status in ((expired_id, 404), (active_id, 200)):
                response = session.get(
                    f"{endpoint}/{record_id}", headers=headers, timeout=timeout
                )
                if response.status_code != expected_status:
                    raise RuntimeError(
                        f"GET {collection}/{record_id} returned "
                        f"{response.status_code}, expected {expected_status}."
                    )
                if expected_status == 200 and response.json()["id"] != active_id:
                    raise RuntimeError("GET returned the wrong active record.")

            response = session.get(
                endpoint,
                headers=headers,
                params={**filters, "limit": "2"},
                timeout=timeout,
            )
            response.raise_for_status()
            page = response.json()
            listed_ids = [item["id"] for item in page["items"]]
            if listed_ids != [active_id] or page["next_cursor"] is not None:
                raise RuntimeError(
                    f"Expected only the active record with no next page. "
                    f"Listed IDs: {listed_ids}."
                )

        print_success(f"Expired {collection} are hidden from item and list reads.")
    except Exception as error:
        print_failure(f"{collection.capitalize()} expiry error:", error)
    finally:
        for record_id in record_ids:
            try:
                if not delete_record(record_id):
                    print_failure("Could not clean up expiry test record:", record_id)
            except Exception as error:
                print_failure("Expiry test cleanup error:", error)


def _test_memory_valid_to() -> None:
    memory_ids: list[str] = []
    subject = f"valid-to-smoke-{uuid4().hex}"
    endpoint = "http://metisa-agent-memory:8000/memories"
    timeout = (5, 240)
    record = {
        "subject": subject,
        "content": {"fact": "favorite_color", "value": "blue"},
        "text": "The user's favorite color is blue.",
        "embedding": [0.25, -0.5, 0.75],
        "expires_at": "2099-01-01T00:00:00Z",
    }

    try:
        token = load_secrets()["METISA_AGENT_MEMORY_TOKEN"]
        headers = {"Authorization": f"Bearer {token}"}

        with requests.Session() as session:
            session.trust_env = False

            for valid_to in ("2000-01-01T00:00:00Z", "2098-01-01T00:00:00Z", None):
                payload = {**record, "valid_to": valid_to}
                response = session.post(
                    endpoint, headers=headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                memory_ids.append(response.json()["id"])

            historical_id, future_id, open_ended_id = memory_ids
            for memory_id, expected_status in (
                (historical_id, 404),
                (future_id, 200),
                (open_ended_id, 200),
            ):
                response = session.get(
                    f"{endpoint}/{memory_id}", headers=headers, timeout=timeout
                )
                if response.status_code != expected_status:
                    raise RuntimeError(
                        f"GET memory {memory_id} returned {response.status_code}, "
                        f"expected {expected_status}."
                    )
                if expected_status == 200 and response.json()["id"] != memory_id:
                    raise RuntimeError("GET returned the wrong memory.")

            response = session.get(
                endpoint,
                headers=headers,
                params={"subject": subject, "limit": "3"},
                timeout=timeout,
            )
            response.raise_for_status()
            page = response.json()
            listed_ids = [item["id"] for item in page["items"]]
            if sorted(listed_ids) != sorted([future_id, open_ended_id]):
                raise RuntimeError(f"Expected only current memories, got {listed_ids}.")
            if page["next_cursor"] is not None:
                raise RuntimeError("Expected all current memories on one page.")

        if not _delete_memory(historical_id):
            raise RuntimeError("Expected DELETE to remove the historical memory.")
        memory_ids.remove(historical_id)

        print_success("Memory reads respect valid_to and allow historical deletion.")
    except Exception as error:
        print_failure("Memory valid_to error:", error)
    finally:
        for memory_id in memory_ids:
            try:
                if not _delete_memory(memory_id):
                    print_failure("Could not clean up valid_to test memory:", memory_id)
            except Exception as error:
                print_failure("Memory valid_to cleanup error:", error)
