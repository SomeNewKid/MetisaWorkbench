"""Provide helper functions for the Metisa MCP Server."""

from collections.abc import Callable

from mcp.types import Resource, Tool

from metisa_common.mcp_helper import create_tool

# This helper would be unnecessary if we used the built-in MCPServer.
# However, we're using the lower-level Server because this is a learning project.


def create_declared_resource(
    metisa: dict[str, object],
    func: Callable[..., object],
) -> Resource | None:
    """Create an MCP resource definition if metisa.toml declares the resource."""
    declared_resources = _create_string_list(metisa, "resources")

    if not declared_resources:
        return None

    resource_name = func.__name__

    if resource_name not in declared_resources:
        return None

    return create_resource(func)


def create_resource(
    func: Callable[..., object],
) -> Resource:
    """Create an MCP resource definition for the function."""
    resource_name = func.__name__
    resource_description = func.__doc__

    return Resource(
        name=resource_name,
        description=resource_description,
        uri=f"metisa://{resource_name}",
        mime_type="text/plain",
    )


def create_declared_tool(
    metisa: dict[str, object],
    func: Callable[..., object],
) -> Tool | None:
    """Create an MCP tool definition if metisa.toml declares the tool."""
    declared_tools = _create_string_list(metisa, "tools")

    if not declared_tools:
        return None

    tool_name = func.__name__

    if tool_name not in declared_tools:
        return None

    return create_tool(func)


def _create_string_list(
    metisa: dict[str, object],
    field_name: str,
) -> list[str]:
    raw_items = metisa.get(field_name)
    if not raw_items:
        return []

    if not isinstance(raw_items, list):
        return []

    collection: list[str] = []
    for raw_item in raw_items:
        if isinstance(raw_item, str):
            if raw_item:
                collection.append(raw_item)

    return collection
