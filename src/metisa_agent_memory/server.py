"""Server for the Prompt Guard HTTP endpoints."""

import json
import os
from dataclasses import asdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import cast
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import jwt
import psycopg
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from psycopg.conninfo import make_conninfo

from metisa_common.screening import screen_content
from metisa_common.secrets_helper import load_secrets

from ._authentication import get_user_sid
from ._models import Episode, JsonObject, Memory
from ._sanitization import sanitize_episode, sanitize_memory
from ._store import MemoryStore

_MAX_REQUEST_BYTES = 1024 * 1024  # 1 MiB
_TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"

_EPISODE_REQUIRED = frozenset(
    {
        "kind",
        "content",
    }
)
_EPISODE_OPTIONAL = frozenset(
    {
        "agent_id",
        "run_id",
        "trace_id",
        "metadata",
        "expires_at",
    }
)

_MEMORY_REQUIRED = frozenset(
    {
        "subject",
        "content",
        "text",
    }
)
_MEMORY_OPTIONAL = frozenset(
    {
        "metadata",
        "valid_to",
        "expires_at",
        "embedding",
    }
)

_STORES: dict[str, MemoryStore] = {}
_lock = Lock()


class _RequestError(Exception):
    def __init__(
        self,
        status: HTTPStatus,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.headers = headers if headers is not None else {}


class _DatabaseSettings:
    def __init__(
        self,
    ) -> None:
        self._lock = Lock()
        self._connection_string: str | None = None

    def get_connection_string(
        self,
    ) -> str:
        """Load database credentials once, after provisioning completes."""
        with self._lock:
            if self._connection_string is not None:
                return self._connection_string

            try:
                secrets = load_secrets()
            except FileNotFoundError as error:
                raise _RequestError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Database credentials are not yet provisioned.",
                    {"Retry-After": "1"},
                ) from error

            password = secrets.get("POSTGRES_PASSWORD")
            if password is None or not password.strip():
                raise ValueError("POSTGRES_PASSWORD secret is required.")

            connection_string = make_conninfo(
                host="metisa-postgres-agent-memory",
                port=5432,
                dbname="agent_memory",
                user="postgres",
                password=password,
            )

            self._connection_string = connection_string
            return connection_string


class _RequestHandler(BaseHTTPRequestHandler):
    # Assigned once during setup; shared configuration is read-only
    _database_settings: _DatabaseSettings
    _run_identifier: str
    _public_key: rsa.RSAPublicKey

    def do_GET(self) -> None:
        self._handle_request()

    def do_POST(self) -> None:
        self._handle_request()

    def do_PUT(self) -> None:
        self._handle_request()

    def do_DELETE(self) -> None:
        self._handle_request()

    def _handle_request(self) -> None:
        self.close_connection = True
        self.connection.settimeout(10)

        try:
            status, payload, headers = self._dispatch()

        except _RequestError as error:
            status = error.status
            payload = {"error": str(error)}
            headers = error.headers

        except TimeoutError:
            status = HTTPStatus.REQUEST_TIMEOUT
            payload = {"error": "Request timed out."}
            headers = {}

        except psycopg.OperationalError:
            status = HTTPStatus.SERVICE_UNAVAILABLE
            payload = {"error": "Database unavailable."}
            headers = {}

        except psycopg.IntegrityError:
            status = HTTPStatus.CONFLICT
            payload = {"error": "The operation conflicts with database constraints."}
            headers = {}

        except psycopg.DataError:
            status = HTTPStatus.UNPROCESSABLE_ENTITY
            payload = {"error": "A supplied value cannot be stored."}
            headers = {}

        except Exception as error:
            # Do not expose database details or credentials to the client.
            self.log_error("Unhandled error: %s", type(error).__name__)
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            payload = {"error": "Internal server error."}
            headers = {}

        self._send_payload(status, payload, headers)

    def _dispatch(
        self,
    ) -> tuple[HTTPStatus, object, dict[str, str]]:
        target = urlsplit(self.path)

        if target.fragment:
            error_message = "URL fragment is not supported"
            raise _RequestError(HTTPStatus.BAD_REQUEST, error_message)

        if target.path == "/health":
            self._require_method({"GET"})
            return HTTPStatus.OK, {"status": "healthy"}, {}

        namespace = self._authenticate()

        if target.path == "/snapshot":
            self._require_method({"GET"})
            if target.query:
                raise _RequestError(
                    HTTPStatus.BAD_REQUEST,
                    "URL query parameters are not supported for snapshots.",
                )

            connection_string = self._database_settings.get_connection_string()
            store = MemoryStore(connection_string, namespace)
            snapshot = store.get_database_snapshot()
            return (
                HTTPStatus.OK,
                snapshot,
                {
                    "Content-Type": _TEXT_CONTENT_TYPE,
                    "Cache-Control": "no-store",
                },
            )

        collection, identifier = self._parse_resource(target.path)

        if identifier is None:
            allowed_methods = {"GET", "POST"}
        elif collection == "episodes":
            allowed_methods = {"GET", "DELETE"}
        else:
            allowed_methods = {"GET", "PUT", "DELETE"}

        self._require_method(allowed_methods)

        connection_string = self._database_settings.get_connection_string()
        store = _get_store(connection_string, namespace)

        record = None

        if self.command == "POST":
            payload = self._read_json()

            if collection == "episodes":
                episode = _episode_from_payload(payload, namespace)
                _screen_episode(episode)
                created = store.create_episode(episode)
            else:
                memory = _memory_from_payload(payload, namespace)
                _screen_memory(memory)
                created = store.create_memory(memory)

            location = f"/{collection}/{created.id}"
            return (
                HTTPStatus.CREATED,
                asdict(created),
                {"Location": location},
            )

        if self.command == "GET":
            try:
                if collection == "episodes":
                    if identifier:
                        if target.query:
                            raise _RequestError(
                                HTTPStatus.BAD_REQUEST,
                                (
                                    "URL query parameters not supported "
                                    "for get-by-id requests."
                                ),
                            )
                        record = store.get_episode(identifier)

                        if record is not None:
                            _prepare_episode_for_read(record)
                    else:
                        kind, limit, after = _parse_episode_list_query(target.query)
                        episodes = store.list_episodes(
                            kind=kind, limit=limit, after=after
                        )
                        page = episodes[:limit]
                        next_cursor = None

                        if len(episodes) > limit:
                            last = page[-1]
                            if last.created_at is None or last.id is None:
                                raise RuntimeError(
                                    "Stored episode is missing its page position."
                                )
                            next_cursor = f"{last.created_at.isoformat()},{last.id}"

                        items = []
                        for episode in page:
                            _prepare_episode_for_read(episode)
                            items.append(asdict(episode))

                        return (
                            HTTPStatus.OK,
                            {
                                "items": items,
                                "next_cursor": next_cursor,
                            },
                            {},
                        )

                else:
                    if identifier:
                        if target.query:
                            raise _RequestError(
                                HTTPStatus.BAD_REQUEST,
                                (
                                    "URL query parameters not supported "
                                    "for get-by-id requests."
                                ),
                            )
                        record = store.get_memory(identifier)
                        if record is not None:
                            _prepare_memory_for_read(record)
                    else:
                        subject, limit, after = _parse_memory_list_query(target.query)
                        memories = store.list_memories(
                            subject=subject, limit=limit, after=after
                        )
                        page = memories[:limit]
                        next_cursor = None

                        if len(memories) > limit:
                            last = page[-1]
                            if last.created_at is None or last.id is None:
                                raise RuntimeError(
                                    "Stored memory is missing its page position."
                                )
                            next_cursor = f"{last.created_at.isoformat()},{last.id}"

                        items = []
                        for memory in page:
                            _prepare_memory_for_read(memory)
                            items.append(asdict(memory))

                        return (
                            HTTPStatus.OK,
                            {
                                "items": items,
                                "next_cursor": next_cursor,
                            },
                            {},
                        )

            except (TypeError, ValueError, RecursionError) as error:
                raise _RequestError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Stored record could not be prepared for retrieval.",
                ) from error

            if record is None:
                raise _RequestError(HTTPStatus.NOT_FOUND, "Record not found.")

            return HTTPStatus.OK, asdict(record), {}

        if self.command == "PUT":
            if target.query:
                raise _RequestError(
                    HTTPStatus.BAD_REQUEST,
                    "URL query parameters not supported for put requests.",
                )
            payload = self._read_json()
            memory = _memory_from_payload(
                payload,
                namespace,
                identifier=identifier,
            )
            _screen_memory(memory)

            try:
                updated = store.update_memory(memory)
            except LookupError as error:
                raise _RequestError(
                    HTTPStatus.NOT_FOUND,
                    "Memory not found.",
                ) from error

            return HTTPStatus.OK, asdict(updated), {}

        if not identifier:
            raise RuntimeError("An deletion request requires an identifier.")
        if target.query:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "URL query parameters not supported for deletion requests.",
            )
        if collection == "episodes":
            deleted = store.delete_episode(identifier)
        else:
            deleted = store.delete_memory(identifier)

        if not deleted:
            raise _RequestError(
                HTTPStatus.NOT_FOUND,
                "Record not found.",
            )

        return HTTPStatus.NO_CONTENT, None, {}

    def _authenticate(
        self,
    ) -> str:
        try:
            user_sid = get_user_sid(self, self._public_key, self._run_identifier)
        except jwt.InvalidTokenError as error:
            raise _RequestError(
                HTTPStatus.UNAUTHORIZED,
                "A valid bearer token is required.",
                {"WWW-Authenticate": 'Bearer realm="metisa-agent-memory"'},
            ) from error

        return f"user:{user_sid}"

    def _parse_resource(
        self,
        path: str,
    ) -> tuple[str, UUID | None]:
        parts = path.split("/")

        if (
            len(parts) not in (2, 3)
            or parts[0] != ""
            or parts[1] not in {"episodes", "memories"}
        ):
            raise _RequestError(
                HTTPStatus.NOT_FOUND,
                "Endpoint not found.",
            )

        collection = parts[1]

        if len(parts) == 2:
            return collection, None

        try:
            identifier = UUID(parts[2])
        except ValueError as error:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "The record ID must be a UUID.",
            ) from error

        return collection, identifier

    def _require_method(
        self,
        allowed: set[str],
    ) -> None:
        if self.command not in allowed:
            raise _RequestError(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "Method not allowed for this endpoint.",
                {"Allow": ", ".join(sorted(allowed))},
            )

    def _read_json(
        self,
    ) -> dict[str, object]:
        if self.headers.get("Transfer-Encoding") is not None:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "Transfer-Encoding is not supported.",
            )

        encoding = self.headers.get("Content-Encoding", "identity")
        if encoding.lower() != "identity":
            raise _RequestError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Compressed request bodies are not supported.",
            )

        if self.headers.get_content_type() != "application/json":
            raise _RequestError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Expected application/json.",
            )

        lengths = self.headers.get_all("Content-Length", [])

        if not lengths:
            raise _RequestError(
                HTTPStatus.LENGTH_REQUIRED,
                "Content-Length is required.",
            )

        if len(lengths) != 1:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "Expected one Content-Length header.",
            )

        length_text = lengths[0].strip()
        if not length_text.isascii() or not length_text.isdecimal():
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "Invalid Content-Length.",
            )

        length = int(length_text)
        if length > _MAX_REQUEST_BYTES:
            raise _RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "Request body is too large.",
            )

        raw_body = self.rfile.read(length)
        if len(raw_body) != length:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "Incomplete request body.",
            )

        try:
            text = raw_body.decode("utf-8")
            payload = json.loads(
                text,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, ValueError, RecursionError) as error:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "Expected valid UTF-8 JSON.",
            ) from error

        if not isinstance(payload, dict):
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "Expected a JSON object.",
            )

        return cast(dict[str, object], payload)

    def _send_payload(
        self,
        status: HTTPStatus,
        payload: object,
        headers: dict[str, str],
    ) -> None:
        headers = headers.copy()
        content_type = headers.pop("Content-Type", "application/json; charset=utf-8")

        if status == HTTPStatus.NO_CONTENT:
            body = b""
        elif content_type == _TEXT_CONTENT_TYPE:
            if not isinstance(payload, str):
                raise TypeError("A plain-text response requires a string.")
            body = payload.encode("utf-8")
        else:
            text = json.dumps(
                payload,
                default=_encode_json_value,
                allow_nan=False,
                ensure_ascii=True,
            )
            body = text.encode("utf-8")

        self.send_response(status)

        if status != HTTPStatus.NO_CONTENT:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))

        self.send_header("Connection", "close")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()

        if body:
            self.wfile.write(body)


def main() -> None:
    """Run the HTTP server until it is stopped."""
    run_identifier = os.environ["METISA_RUN_ID"]
    if not run_identifier:
        raise RuntimeError("Run identifier is required.")

    public_key_path = Path("/sandbox-source/metisa-public-key.pem")
    public_key_pem = public_key_path.read_bytes()
    public_key = serialization.load_pem_public_key(public_key_pem)

    if not isinstance(public_key, rsa.RSAPublicKey):
        raise ValueError("Run verification key must be an RSA public key.")

    _RequestHandler._database_settings = _DatabaseSettings()
    _RequestHandler._run_identifier = run_identifier
    _RequestHandler._public_key = public_key

    with ThreadingHTTPServer(("0.0.0.0", 8000), _RequestHandler) as server:
        server.serve_forever()


# This approach of purging expired records is a little inelegant,
# but it is simple, and is sufficient for the MetisaWorkbench project.
def _get_store(
    connection_string: str,
    namespace: str,
) -> MemoryStore:
    if namespace not in _STORES:
        with _lock:
            if namespace not in _STORES:
                store = MemoryStore(connection_string, namespace)
                store.purge_episodes()
                store.purge_memories()
                _STORES[namespace] = store
    return _STORES[namespace]


def _episode_from_payload(
    payload: dict[str, object],
    namespace: str,
) -> Episode:
    _check_fields(payload, _EPISODE_REQUIRED, _EPISODE_OPTIONAL)

    try:
        episode = Episode(
            id=None,
            namespace=namespace,
            agent_id=cast(str | None, payload.get("agent_id")),
            run_id=cast(str | None, payload.get("run_id")),
            trace_id=cast(str | None, payload.get("trace_id")),
            kind=cast(str, payload["kind"]),
            content=cast(JsonObject, payload["content"]),
            metadata=cast(JsonObject, payload.get("metadata", {})),
            created_at=None,
            expires_at=_parse_datetime(
                payload.get("expires_at"),
                "expires_at",
            ),
        )
        episode.validate()
        sanitize_episode(episode)
        episode.validate()

    except (TypeError, ValueError, RecursionError) as error:
        raise _RequestError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            f"Invalid episode: {error}",
        ) from error

    return episode


def _memory_from_payload(
    payload: dict[str, object],
    namespace: str,
    *,
    identifier: UUID | None = None,
) -> Memory:
    _check_fields(payload, _MEMORY_REQUIRED, _MEMORY_OPTIONAL)

    try:
        memory = Memory(
            id=identifier,
            namespace=namespace,
            subject=cast(str, payload["subject"]),
            content=cast(JsonObject, payload["content"]),
            text=cast(str, payload["text"]),
            embedding=cast(list[float] | None, payload.get("embedding")),
            metadata=cast(JsonObject, payload.get("metadata", {})),
            valid_to=_parse_datetime(payload.get("valid_to"), "valid_to"),
            created_at=None,
            updated_at=None,
            expires_at=_parse_datetime(
                payload.get("expires_at"),
                "expires_at",
            ),
        )
        memory.validate()
        sanitize_memory(memory)
        memory.validate()

    except (TypeError, ValueError, RecursionError) as error:
        raise _RequestError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            f"Invalid memory: {error}",
        ) from error

    return memory


def _check_fields(
    payload: dict[str, object],
    required: frozenset[str],
    optional: frozenset[str],
) -> None:
    supplied = set(payload)

    missing = required - supplied
    if missing:
        names = ", ".join(sorted(missing))
        raise _RequestError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            f"Missing fields: {names}.",
        )

    unknown = supplied - required - optional
    if unknown:
        names = ", ".join(sorted(unknown))
        raise _RequestError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            f"Unsupported fields: {names}.",
        )


def _parse_datetime(
    value: object,
    name: str,
) -> datetime | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO 8601 string or null.")

    parsed = datetime.fromisoformat(value)

    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone.")

    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not valid JSON.")


def _encode_json_value(value: object) -> str:
    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(f"Cannot encode {type(value).__name__} as JSON.")


def _screen_episode(
    episode: Episode,
) -> None:
    screen_content(
        {
            "kind": episode.kind,
            "content": episode.content,
            "metadata": episode.metadata,
        }
    )


def _screen_memory(
    memory: Memory,
) -> None:
    screen_content(
        {
            "subject": memory.subject,
            "content": memory.content,
            "metadata": memory.metadata,
        }
    )
    screen_content(
        {
            "subject": memory.subject,
            "text": memory.text,
        }
    )


def _parse_episode_list_query(
    query: str,
) -> tuple[str | None, int, tuple[datetime, UUID] | None]:
    try:
        fields = parse_qs(
            query,
            keep_blank_values=True,
            strict_parsing=True,
            errors="strict",
            max_num_fields=3,
        )
        for name, values in fields.items():
            if name not in {"kind", "limit", "after"}:
                raise ValueError("Unsupported query parameter.")
            if len(values) != 1:
                raise ValueError("Repeated query parameter.")

        kind = fields.get("kind", [None])[0]
        if kind is not None and (not kind.strip() or "\x00" in kind):
            raise ValueError("Invalid episode kind.")

        limit = int(fields.get("limit", ["10"])[0])
        if not 1 <= limit <= 20:
            raise ValueError("Limit must be between 1 and 20.")

        after = None
        if "after" in fields:
            created_at_text, id_text = fields["after"][0].split(",", maxsplit=1)
            created_at = _parse_datetime(created_at_text, "after")
            if created_at is None:
                raise ValueError("Cursor timestamp is required.")
            after = (created_at, UUID(id_text))

    except ValueError as error:
        raise _RequestError(
            HTTPStatus.BAD_REQUEST,
            "Invalid episode list parameters.",
        ) from error

    return kind, limit, after


def _parse_memory_list_query(
    query: str,
) -> tuple[str | None, int, tuple[datetime, UUID] | None]:
    try:
        fields = parse_qs(
            query,
            keep_blank_values=True,
            strict_parsing=True,
            errors="strict",
            max_num_fields=3,
        )
        for name, values in fields.items():
            if name not in {"subject", "limit", "after"}:
                raise ValueError("Unsupported query parameter.")
            if len(values) != 1:
                raise ValueError("Repeated query parameter.")

        subject = fields.get("subject", [None])[0]
        if subject is not None and (not subject.strip() or "\x00" in subject):
            raise ValueError("Invalid memory subject.")

        limit = int(fields.get("limit", ["10"])[0])
        if not 1 <= limit <= 20:
            raise ValueError("Limit must be between 1 and 20.")

        after = None
        if "after" in fields:
            created_at_text, id_text = fields["after"][0].split(",", maxsplit=1)
            created_at = _parse_datetime(created_at_text, "after")
            if created_at is None:
                raise ValueError("Cursor timestamp is required.")
            after = (created_at, UUID(id_text))

    except ValueError as error:
        raise _RequestError(
            HTTPStatus.BAD_REQUEST,
            "Invalid memory list parameters.",
        ) from error

    return subject, limit, after


def _prepare_episode_for_read(episode: Episode) -> None:
    episode.validate()
    _screen_episode(episode)
    sanitize_episode(episode)
    episode.validate()
    _screen_episode(episode)


def _prepare_memory_for_read(memory: Memory) -> None:
    memory.validate()
    _screen_memory(memory)
    sanitize_memory(memory)
    memory.validate()
    _screen_memory(memory)


if __name__ == "__main__":
    main()
