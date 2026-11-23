"""Provide the tool which uses semantic search to get chunks from a document."""

from __future__ import annotations

import json
import logging

from metisa_common.mcp_helper import with_output_schema

_LOGGER = logging.getLogger(__name__)


_TOOL_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["files"],
    "additionalProperties": False,
}


@with_output_schema(_TOOL_RESULT_SCHEMA)
def get_list_of_available_documents() -> dict[str, object]:
    """List available company documents and their filenames.

    Use this to discover filenames before retrieving document content.
    """
    _LOGGER.debug("get_list_of_available_documents request")

    result: dict[str, object] = {
        "files": [
            "Employee handbook.pdf",
        ]
    }

    _LOGGER.debug(
        "get_list_of_available_documents response: %s",
        json.dumps(result, ensure_ascii=False, indent=2),
    )

    return result
