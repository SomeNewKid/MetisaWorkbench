"""Demonstrate product lookup guardrails."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from metisa_common.terminal_helper import (
    read_user_input,
)
from metisa_harness.guardrails import (
    GuardrailContext,
    GuardrailTripwire,
)
from metisa_harness.models import (
    GuardrailPlugin,
    LlmMessage,
    ModelContext,
)

_PRINT_GUARDRAIL_CHANGES = False


def create_product_guardrail_plugin() -> GuardrailPlugin:
    """Create guardrails for product lookup agent workflows."""
    return GuardrailPlugin(
        name="product_lookup",
        before_run=(_check_length_of_input,),
        after_run=(_append_warning_for_agent_output,),
        before_model=(_disallow_prompt_injection,),
        after_model=(
            _disallow_blocked_response,
            _modify_hello_world_response,
        ),
        before_tool=(
            _normalize_product_arguments,
            _block_internal_product,
            _approve_refund,
        ),
        after_tool=(_normalize_product_result,),
    )


async def _normalize_product_arguments(
    context: GuardrailContext[dict[str, object]],
) -> dict[str, object]:
    arguments = context.payload

    if context.metadata.tool_name != "get_product_details":
        return arguments

    original_code = arguments.get("product_code")
    if not isinstance(original_code, str):
        raise GuardrailTripwire("Product code must be text.")

    normalized_code = original_code.strip().upper()
    if not normalized_code:
        raise GuardrailTripwire("Product code must not be empty.")

    arguments["product_code"] = normalized_code

    if _PRINT_GUARDRAIL_CHANGES:
        print(
            f"[arguments] {original_code!r} -> {normalized_code!r}",
            flush=True,
        )

    return arguments


async def _block_internal_product(
    context: GuardrailContext[dict[str, object]],
) -> dict[str, object]:
    result = context.payload

    if context.metadata.tool_name != "get_product_details":
        return result

    if result.get("product_code") == "INTERNAL":
        if _PRINT_GUARDRAIL_CHANGES:
            print("[tripwire] Internal product lookup blocked.", flush=True)
        raise GuardrailTripwire("Internal products are unavailable to this agent.")

    return result


async def _normalize_product_result(
    context: GuardrailContext[dict[str, object]],
) -> dict[str, object]:
    arguments = context.payload

    if context.metadata.tool_name != "get_product_details":
        return arguments

    result = dict(arguments)
    original_name = result.get("product_name")

    if not isinstance(original_name, str):
        raise GuardrailTripwire("Product result must contain a text name.")

    normalized_name = original_name.strip().title()
    result["product_name"] = normalized_name

    if _PRINT_GUARDRAIL_CHANGES:
        print(
            f"[result] {original_name!r} -> {normalized_name!r}",
            flush=True,
        )

    return result


async def _check_length_of_input(
    context: GuardrailContext[str],
) -> str:
    input = context.payload

    if len(input) > 100:
        raise GuardrailTripwire("Input to an AI agent cannot exceed 100 characters.")

    return input


async def _append_warning_for_agent_output(
    context: GuardrailContext[str | dict[str, object]],
) -> str | dict[str, object]:
    result = context.payload

    if not isinstance(result, str):
        return result

    result += " (AI agents can make mistakes.)"
    return result


async def _disallow_prompt_injection(
    context: GuardrailContext[ModelContext],
) -> ModelContext:
    model_context = context.payload

    for message in model_context.messages:
        if message.role != "user":
            continue

        message_content = message.content
        if not isinstance(message_content, str):
            continue

        if "ignore previous instructions" in message_content.lower():
            raise GuardrailTripwire("Prompt injection detected.")

    return model_context


async def _disallow_blocked_response(
    context: GuardrailContext[LlmMessage],
) -> LlmMessage:
    message = context.payload

    text = message.content

    if text is not None:
        if "[block-response]" in text:
            raise GuardrailTripwire("Demonstration guardrail blocked the response.")

    return message


async def _modify_hello_world_response(
    context: GuardrailContext[LlmMessage],
) -> LlmMessage:
    message = context.payload

    text = message.content

    if text is not None:
        if "ello, world" in text:
            normalized_text = text.replace("ello, world", "ello, cruel world")
            return replace(message, content=normalized_text)

    return message


async def _approve_refund(
    context: GuardrailContext[dict[str, object]],
) -> dict[str, object]:
    arguments = context.payload
    if context.metadata.tool_name != "issue_refund":
        return arguments

    refund_amount_cents = arguments.get("refund_amount_cents")
    if not isinstance(refund_amount_cents, int):
        raise GuardrailTripwire(
            "Refund amount must be specified in cents as an integer."
        )

    if refund_amount_cents <= 0:
        raise GuardrailTripwire("Refund amount must be greater than zero.")

    if refund_amount_cents > 100_00:
        raise GuardrailTripwire("Refund amount cannot exceed $100.00.")

    if refund_amount_cents > 50_00:
        prompt = f"Approve the refund of ${refund_amount_cents / 100:.2f} (y/n)y? "
        answer = await asyncio.to_thread(read_user_input, prompt)
        if answer.lower() not in ("y", "yes"):
            raise GuardrailTripwire("Refund not approved by user.")

    return arguments
