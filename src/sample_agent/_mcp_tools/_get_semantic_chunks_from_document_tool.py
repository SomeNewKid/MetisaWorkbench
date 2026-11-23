"""Provide the tool which uses semantic search to get chunks from a document."""

from __future__ import annotations

import json
import logging
from typing import Annotated

from pydantic import Field

from metisa_common.mcp_helper import with_output_schema

_LOGGER = logging.getLogger(__name__)


_TOOL_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["chunks"],
    "additionalProperties": False,
}

_COMPANY_VALUES_CHUNK = """
    Values 
    
    At Zava, we strive to create an environment that values hard work, innovation, 
    and collaboration. Our core values serve as the foundation for our success, 
    and they guide our employees in how we should act and interact with each other 
    and our customers. 
    
    Company Values: 
    
    1. Quality: We strive to provide the highest quality services to our customers.
    2. Integrity: We value honesty, respect, and trustworthiness in all things.
    3. Innovation: We encourage creativity and support new ideas in our business.
    4. Teamwork: We believe that by working together we can achieve greater success.
    5. Respect: We treat all our employees, customers, and partners with respect.
    6. Excellence: We strive to exceed expectations and provide excellent service.
    7. Accountability: We take responsibility for our actions.
    8. Community: We are committed to making a positive impact in our communities.
    """

_WHISTLEBLOWER_CHUNK = """
    Reporting Procedures: 
    
    If you witness any activity that you believe to be illegal, unethical, or not in 
    accordance with company policy, it is important that you report it immediately. 
    You can do this by: 
    
    1. Contacting the Human Resources Department.
    2. Emailing the Compliance Officer at compliance@Zava.com.
    3. Calling the Compliance Hotline at 1-800-555-1212.
    
    When making a report, please provide as much detail as possible. 
    This information should include: 
    
    1. The time and date of the incident.
    2. Who was involved.
    """


@with_output_schema(_TOOL_RESULT_SCHEMA)
def get_semantic_chunks_from_document(
    filename: Annotated[str, Field(description="The filename of the document.")],
    prompt: Annotated[str, Field(description="The user's prompt.")],
) -> dict[str, object]:
    """Retrieve document excerpts relevant to the user's prompt.

    Use a filename returned by get_list_of_available_documents.
    """
    _LOGGER.debug("get_semantic_chunks_from_document request")
    chunks: list[str] = []
    if filename == "Employee handbook.pdf":
        if "values" in prompt.lower():
            chunks.append(_COMPANY_VALUES_CHUNK)
        else:
            chunks.append(_WHISTLEBLOWER_CHUNK)

    result: dict[str, object] = {
        "chunks": chunks,
    }

    _LOGGER.debug(
        "get_semantic_chunks_from_document response: %s",
        json.dumps(result, ensure_ascii=False, indent=2),
    )

    return result
