"""Command-line interface for the application."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from textwrap import dedent

from mcp import Client
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider

from metisa_common.clients.mcp_client import get_mcp_server as get_remote_mcp_client
from metisa_common.specification_helper import (
    load_specification,
)
from metisa_common.specification_models import (
    Capability,
    MetisaSpecification,
)
from metisa_common.terminal_helper import (
    read_user_input,
)
from metisa_harness import (
    AgentDefinition,
    AgentHarness,
    LifecycleEvent,
)

from ._guardrails import (
    create_product_guardrail_plugin,
)
from ._helper import (
    print_elapsed_time,
    print_failure,
    print_success,
)
from ._mcp_server import get_mcp_server as get_local_mcp_server
from ._models import (
    ProductDetails,
)
from ._telemetry import (
    initialize_tracing,
)
from ._tools import (
    get_current_date_time,
    get_email_address,
    get_phone_number,
    get_product_details,
    get_robot_battery_charge,
    get_robot_motor_temperature,
    issue_refund,
)

_PRINT_LIFECYCLE_EVENTS = False

_root_directory = Path(__file__).parent
_metisa_json_path = _root_directory / "metisa.toml"
if not _metisa_json_path.is_file():
    raise FileNotFoundError("Sample Agent missing a metisa.toml file.")
_specification: MetisaSpecification = load_specification(_metisa_json_path)


def main(
    argv: list[str] | None = None,
) -> int:
    """Run the sample agent workload."""
    run_id = os.getenv("METISA_RUN_ID")
    if not run_id:
        raise RuntimeError("METISA_RUN_ID environment variable is not set.")

    user_sid = os.getenv("METISA_USER_SID")
    if not user_sid:
        raise RuntimeError("METISA_USER_SID environment variable is not set.")

    prompt = _get_first_argument(argv)

    logging.getLogger("httpx2").setLevel(logging.WARNING)

    provider: TracerProvider | None = None

    if Capability.OTEL_COLLECTOR in _specification.capabilities:
        provider = initialize_tracing()

    # OpenTelemetry supports instrumentation without an initialized SDK provider.
    # Calls such as trace.get_tracer() and start_as_current_span()
    # do not record or export spans until the provider is configured.
    try:
        tracer = trace.get_tracer("sample_agent")

        attributes = {
            "run_id": run_id,
            "user_sid": user_sid,
        }

        with tracer.start_as_current_span(
            "sample_agent.run",
            attributes=attributes,
            context=Context(),
        ):
            asyncio.run(
                _run_agent(
                    _specification,
                    prompt=prompt,
                    run_identifier=run_id,
                )
            )

        return 0

    finally:
        if provider is not None:
            provider.shutdown()


async def _run_agent(
    specification: MetisaSpecification,
    prompt: str | None,
    run_identifier: str,
) -> None:
    # Example prompts:
    # - Look up product code "chair". (Tests guardrails)
    # - How many company values do we have? (Tests RAG)
    # - What is the current time? (Tests tools)
    # - What is the email address and phone number for Claire? (Agents as tools)
    # - Issue a refund for $75.40. (Human in the loop approval)
    #   (over $50 requires approval, and over $100 is automatically rejected)
    # - Can the robot named Bolt enter the race? (Test skills.)
    # - Can the robot named Alice be placed in storage? (Test skills with scripts)
    default_prompt = "Issue a refund for $75.40."

    prompt_count = 0
    if not prompt:
        if Capability.INTERACTIVE not in specification.capabilities:
            prompt = default_prompt
    while not prompt:
        prompt_count += 1
        if prompt_count > 3:
            raise RuntimeError("Too many empty prompts entered.")
        prompt = await asyncio.to_thread(read_user_input, "Enter prompt: ")

    models = [
        "gpt-4o",  # online frontier model
        "granite-4-1b",  # local small model
    ]

    if Capability.LLM_BROKER not in specification.capabilities:
        raise RuntimeError("LLM Broker capability not declared.")
    if not specification.llm_broker:
        raise RuntimeError("LLM Broker not available.")
    local_models = [model for model in specification.llm_broker.local_models]
    online_models = [model for model in specification.llm_broker.online_models]
    available_models = local_models + online_models
    unavailable_models = [model for model in models if model not in available_models]
    if unavailable_models:
        raise RuntimeError(
            f"The following models are unavailable: {unavailable_models}"
        )

    use_mcp_serer = Capability.MCP_SERVER in specification.capabilities

    for model in models:
        await _run_agent_model(
            model,
            prompt,
            run_identifier,
            use_mcp_serer=use_mcp_serer,
        )


async def _run_agent_model(
    model_name: str, prompt: str, run_identifier: str, *, use_mcp_serer: bool = False
) -> None:
    guardrail_plugin = create_product_guardrail_plugin()

    email_agent = AgentDefinition(
        name="email_agent",
        description="Retrieve email addresses for users.",
        model=model_name,
        instructions="You provide the email address for the specified user.",
        tools=[get_email_address],
    )

    phone_agent = AgentDefinition(
        name="phone_agent",
        description="Retrieve phone numbers for users.",
        model=model_name,
        instructions="You provide the phone number for the specified user.",
        tools=[get_phone_number],
    )

    # The MCP session stays open through tool discovery and execution,
    # then closes when the run leaves the stack.
    async with AsyncExitStack() as stack:
        local_mcp_server = get_local_mcp_server()
        local_mcp_client = await stack.enter_async_context(Client(local_mcp_server))
        mcp_clients: list[Client] = [local_mcp_client]

        if use_mcp_serer:
            remote_mcp_client = await stack.enter_async_context(get_remote_mcp_client())
            mcp_clients.append(remote_mcp_client)

        main_agent = AgentDefinition(
            name="sample_agent",
            description="Example agent",  # Description really only used for sub-agents
            model=model_name,
            instructions=dedent("""
                Answer the user's question as simply as possible.
                Provide the answer as plain text, and not as Markdown or JSON.
                Discover the date and time with get_current_date_time.
                Discover available filenames with get_list_of_available_documents,
                then retrieve relevant excerpts with get_semantic_chunks_from_document.
                Check the documents before asking the user to supply a filename.
                For product questions, call get_product_details,
                passing the product code exactly as supplied by the user,
                including its capitalization and surrounding spaces.
                For robot inspections, identify the matching skill and 
                follow its checklist.  If you cannot load the checklist or
                a required reference, stop and explain the failure.
                Do not guess the answer if it is not available in the documents 
                or through the tools.
                If you cannot find the answer, explain what documents were checked and
                which tools were used, before stating that the answer is not available.
                """),
            tools=[
                get_current_date_time,
                get_product_details,
                get_robot_battery_charge,
                get_robot_motor_temperature,
                issue_refund,
            ],
            mcp_clients=mcp_clients,
            sub_agents=[email_agent, phone_agent],
            guardrail_plugins=(guardrail_plugin,),
            skills_directory=_root_directory / "skills",
            # output_type=ProductDetails,
        )

        print()
        print("-" * 20)
        print(model_name)
        print(prompt, flush=True)
        with print_elapsed_time():
            try:
                result = await AgentHarness.run(
                    main_agent,
                    run_identifier=run_identifier,
                    input=prompt,
                    event_handler=_print_lifecycle_event,
                )
                if result.final_output:
                    if main_agent.output_type is ProductDetails:
                        if isinstance(result.final_output, ProductDetails):
                            product_code = result.final_output.product_code
                            product_name = result.final_output.product_name
                            print_success(
                                f"ProductDetails '{product_code}', '{product_name}'"
                            )
                        else:
                            print_failure("Final output of wrong type.")
                    else:
                        print_success(result.final_output)
                else:
                    raise RuntimeError("No final text returned.")
            except Exception as error:
                print_failure(str(error))


async def _print_lifecycle_event(
    event: LifecycleEvent,
) -> None:
    if not _PRINT_LIFECYCLE_EVENTS:
        return

    details = [event.agent_name, event.kind]

    if event.turn is not None:
        details.append(f"turn={event.turn}")

    if event.turn is not None:
        details.append(f"turn={event.turn}")

    if event.tool_name is not None:
        details.append(f"tool={event.tool_name}")

    if event.error_type is not None:
        details.append(f"error={event.error_type}")

    description = " ".join(details)
    print(f"{description} [{event.elapsed_seconds:.3f}s]", flush=True)


def _get_first_argument(
    argv: list[str] | None = None,
) -> str | None:
    """Get the first command-line argument, or an empty string if none are provided."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        return None

    return args[0]
