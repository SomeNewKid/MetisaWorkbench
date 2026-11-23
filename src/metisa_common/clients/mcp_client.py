"""Utility functions for working with the MCP Server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx2
from mcp import Client, types
from mcp.client._transport import TransportStreams
from mcp.client.streamable_http import streamable_http_client
from opentelemetry import trace
from opentelemetry.propagate import inject
from opentelemetry.trace import SpanKind

from metisa_common.secrets_helper import (
    load_secrets,
)

_MCP_SERVER_URL = "http://metisa-mcp-server:8000/mcp"
_TIMEOUT_SECONDS = 300


def get_mcp_server() -> Client:
    """Get an MCP Server client."""
    transport = _open_mcp_transport()

    return Client(
        transport,
        read_timeout_seconds=_TIMEOUT_SECONDS,
    )


def get_mcp_server_tool_result(
    tool_name: str,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Get the result of calling a tool on the MCP Server."""
    tracer = trace.get_tracer("sample_agent.tools")

    with tracer.start_as_current_span(
        f"tool.call {tool_name}",
        kind=SpanKind.CLIENT,
        attributes={"metisa.tool.name": tool_name},
    ):
        return _get_mcp_server_tool_result(tool_name, params)


def _get_mcp_server_tool_result(
    tool_name: str,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Get the result of calling a tool on the MCP Server."""

    async def _call() -> types.CallToolResult:
        async with asyncio.timeout(_TIMEOUT_SECONDS):
            server = get_mcp_server()

            async with server:
                arguments = params if params is not None else {}
                return await server.call_tool(tool_name, arguments)

    result = asyncio.run(_call())

    if result.is_error:
        details = "\n".join(
            content.text
            for content in result.content
            if isinstance(content, types.TextContent)
        )
        raise RuntimeError(details or f"{tool_name} failed.")

    payload = result.structured_content
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a structured JSON object from {tool_name}.")

    return payload


@asynccontextmanager
async def _open_mcp_transport() -> AsyncGenerator[TransportStreams, None]:
    secrets = load_secrets()
    token = secrets.get("METISA_MCP_SERVER_TOKEN")

    if not token:
        raise RuntimeError("METISA_MCP_SERVER_TOKEN secret is not set.")

    headers = {"Authorization": f"Bearer {token}"}
    inject(headers)

    async with httpx2.AsyncClient(
        headers=headers,
        trust_env=False,
        timeout=_TIMEOUT_SECONDS,
    ) as http_client:
        transport = streamable_http_client(
            url=_MCP_SERVER_URL,
            http_client=http_client,
        )

        async with transport as streams:
            yield streams
