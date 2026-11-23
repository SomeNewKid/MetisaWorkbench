"""Command-line interface for the application."""

from __future__ import annotations

from mcp.server import MCPServer

from ._mcp_tools._get_list_of_available_documents_tool import (
    get_list_of_available_documents,
)
from ._mcp_tools._get_semantic_chunks_from_document_tool import (
    get_semantic_chunks_from_document,
)


def get_mcp_server() -> MCPServer:
    """Get an in-memory MCP Server."""
    mcp_server = MCPServer("sample_mcp_server")
    mcp_server.add_tool(
        get_list_of_available_documents,
        structured_output=True,
    )
    mcp_server.add_tool(
        get_semantic_chunks_from_document,
        structured_output=True,
    )
    return mcp_server
