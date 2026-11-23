"""Tools for the sample agent."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

from metisa_common.mcp_helper import with_output_schema

_LOGGER = logging.getLogger(__name__)

_DATE_TIME_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "year": {"type": "integer"},
        "month": {"type": "integer"},
        "day": {"type": "integer"},
        "hour": {"type": "integer"},
        "minutes": {"type": "integer"},
        "seconds": {"type": "integer"},
        "day_of_week": {"type": "integer"},
        "day_of_week_name": {"type": "string"},
        "utc_offset_minutes": {"type": "integer"},
    },
    "required": [
        "year",
        "month",
        "day",
        "hour",
        "minutes",
        "seconds",
        "day_of_week",
        "day_of_week_name",
        "utc_offset_minutes",
    ],
    "additionalProperties": False,
}

_PRODUCT_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "product_code": {"type": "string"},
        "product_name": {"type": "string"},
    },
    "required": ["product_code", "product_name"],
    "additionalProperties": False,
}

_PRODUCT_NAMES = {
    "LAMP": "  desk lamp  ",
    "CHAIR": "  office chair  ",
    "INTERNAL": "  internal prototype   ",
}

_EMAIL_ADDRESS_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "email_address": {"type": "string"},
    },
    "required": ["email_address"],
    "additionalProperties": False,
}

_PHONE_NUMBER_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "phone_number": {"type": "string"},
    },
    "required": ["phone_number"],
    "additionalProperties": False,
}

_ISSUE_REFUND_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "refund_id": {"type": "string"},
    },
    "required": ["refund_id"],
    "additionalProperties": False,
}

_ROBOT_TEMPERATURE_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number"},
    },
    "required": ["temperature"],
    "additionalProperties": False,
}

_ROBOT_BATTERY_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "battery": {"type": "number"},
    },
    "required": ["battery"],
    "additionalProperties": False,
}


@with_output_schema(_EMAIL_ADDRESS_RESULT_SCHEMA)
def get_email_address(
    user: str,
) -> dict[str, object]:
    """Get the email address for the specified user."""
    user_name = user.lower().replace(" ", "-")
    email_address = f"{user_name}@example.com"
    return {
        "email_address": email_address,
    }


@with_output_schema(_PHONE_NUMBER_RESULT_SCHEMA)
def get_phone_number(
    user: str,
) -> dict[str, object]:
    """Get the email address for the specified user."""
    return {
        "phone_number": "555-1234",
    }


@with_output_schema(_DATE_TIME_RESULT_SCHEMA)
def get_current_date_time() -> dict[str, object]:
    """
    Return the current local date, time, day of the week, and UTC offset.

    Day of the week is represented as an integer, where Monday is 1 and Sunday is 7.
    """
    _LOGGER.debug("get_current_date_time request")
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    offset_minutes = int(offset.total_seconds() / 60) if offset else 0

    result: dict[str, object] = {
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minutes": now.minute,
        "seconds": now.second,
        "day_of_week": now.isoweekday(),
        "day_of_week_name": now.strftime("%A"),
        "utc_offset_minutes": offset_minutes,
    }

    _LOGGER.debug(
        "get_current_date_time response: %s",
        json.dumps(result, ensure_ascii=False, indent=2),
    )

    return result


@with_output_schema(_PRODUCT_RESULT_SCHEMA)
def get_product_details(
    product_code: str,
) -> dict[str, object]:
    """Look up a product using its product code."""
    _LOGGER.debug("get_product_details request")

    result: dict[str, object] = {
        "product_code": product_code,
        "product_name": _PRODUCT_NAMES.get(product_code, "  unknown product   "),
    }

    _LOGGER.debug(
        "get_product_details response: %s",
        json.dumps(result, ensure_ascii=False, indent=2),
    )

    return result


@with_output_schema(_ISSUE_REFUND_RESULT_SCHEMA)
def issue_refund(
    refund_amount_cents: int,
) -> dict[str, object]:
    """Issue a refund with the specified refund amount, in cents."""
    refund_id = f"refund_{uuid4().hex}"
    return {
        "refund_id": refund_id,
    }


@with_output_schema(_ROBOT_BATTERY_RESULT_SCHEMA)
def get_robot_battery_charge(
    robot_name: str,
) -> dict[str, object]:
    """Get the battery charge for the named robot."""
    if not robot_name.strip():
        raise ValueError("Robot name is required.")

    charge = 82 if robot_name.lower() == "bolt" else 50
    return {
        "battery": charge,
    }


@with_output_schema(_ROBOT_TEMPERATURE_RESULT_SCHEMA)
def get_robot_motor_temperature(
    robot_name: str,
) -> dict[str, object]:
    """Get the motor temperature for the named robot."""
    if not robot_name.strip():
        raise ValueError("Robot name is required.")

    temperature = 31 if robot_name.lower() == "bolt" else 32
    return {
        "temperature": temperature,
    }
