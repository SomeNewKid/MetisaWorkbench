"""Command-line interface for the application."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path
from textwrap import dedent

from mcp import Client
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel, Field

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
from metisa_harness.models import (
    RunResult,
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
from ._memory import (
    get_all_memories,
    get_episode,
    get_snapshot,
    save_episode,
    save_memory,
)
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

_root_directory = Path(__file__).parent
_metisa_json_path = _root_directory / "metisa.toml"
if not _metisa_json_path.is_file():
    raise FileNotFoundError("Sample Agent missing a metisa.toml file.")
_specification: MetisaSpecification = load_specification(_metisa_json_path)

_USE_HOT_PATH_MEMORIES = True
_USE_REFLECTION_MEMORIES = True
_RECORD_MEMORY_SNAPSHOTS = False


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
        # "granite-4-1b",   # local small model
        # "qwen-3-small",   # local small model
        # "qwen-3-5-small", # local small model
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
            specification,
            model,
            prompt,
            run_identifier,
            use_mcp_serer=use_mcp_serer,
        )


async def _run_agent_model(
    specification: MetisaSpecification,
    model_name: str,
    prompt: str,
    run_identifier: str,
    *,
    use_mcp_serer: bool = False,
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

        use_memory = Capability.AGENT_MEMORY in specification.capabilities

        instructions: list[str] = [
            dedent("""
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

            If you know the user's name, address the user by name in your 
            final response.
            """)
        ]

        tools = [
            get_current_date_time,
            get_product_details,
            get_robot_battery_charge,
            get_robot_motor_temperature,
            issue_refund,
        ]

        if use_memory:
            if _USE_HOT_PATH_MEMORIES:
                tools.append(save_memory)

                instructions.append(
                    dedent("""
                    If the user provides information relevant to agent memory, 
                    such as his or her name or other personal details,
                    save each fact separately using the save_memory tool.
                    """)
                )

            await _record_snapshot("snapshot-before-run.md")

            memories = await asyncio.to_thread(get_all_memories)
            if isinstance(memories, dict) and memories:
                memory_lines = []
                for subject, text in memories.items():
                    if not isinstance(subject, str) or not subject.strip():
                        continue
                    if not isinstance(text, str) or not text.strip():
                        continue
                    memory_lines.append(f"{subject}: {text}")

                if memory_lines:
                    memory_intro = "Here are the memories saved from previous sessions:"
                    memory_lines.insert(0, memory_intro)

                    instructions.append("\n".join(memory_lines))

        complete_instructions = "\n".join(instructions)

        _record_instructions(complete_instructions)

        main_agent = AgentDefinition(
            name="sample_agent",
            description="Example agent",  # Description really only used for sub-agents
            model=model_name,
            instructions=complete_instructions,
            tools=tools,
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
                    prompt=prompt,
                    event_handler=partial(
                        _write_lifecycle_event, model_name=model_name
                    ),
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

                if use_memory:
                    await _record_snapshot("snapshot-before-episodes.md")

                    episode_id = await _save_episode(
                        main_agent.name,
                        run_identifier,
                        result,
                    )
                    if _USE_REFLECTION_MEMORIES:
                        if isinstance(episode_id, str):
                            episode = await asyncio.to_thread(get_episode, episode_id)
                            if not isinstance(episode, dict):
                                raise RuntimeError("Invalid episode format.")
                            await _create_memories_from_episode(
                                run_identifier,
                                episode,
                            )
                            await _record_snapshot("snapshow-after-reflection.md")
                    else:
                        await _record_snapshot("snapshot-without-reflection.md")
            except Exception as error:
                print_failure(str(error))


async def _write_lifecycle_event(
    event: LifecycleEvent,
    *,
    model_name: str,
) -> None:
    output_directory = Path(os.environ["SANDBOX_OUTPUT_DIR"])
    log_directory = output_directory / ".logs" / "metisa"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "lifecycle.txt"

    details = [event.agent_name, event.kind]

    if event.turn is not None:
        details.append(f"turn={event.turn}")

    if event.tool_name is not None:
        details.append(f"tool={event.tool_name}")

    if event.error_type is not None:
        details.append(f"error={event.error_type}")

    description = " ".join(details)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{description} [{event.elapsed_seconds:.3f}s]\n")


def _get_first_argument(
    argv: list[str] | None = None,
) -> str | None:
    """Get the first command-line argument, or an empty string if none are provided."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        return None

    return args[0]


async def _save_episode(
    agent_name: str,
    run_id: str,
    result: RunResult,
) -> str | None:
    user_messages: list[str] = []
    for message in result.messages:
        if message.role != "user":
            continue
        if not isinstance(message.content, str):
            continue
        user_messages.append(message.content)

    if not user_messages:
        return None
    return await asyncio.to_thread(
        save_episode,
        agent_name,
        run_id,
        user_messages,
    )


async def _create_memories_from_episode(
    run_identifier: str,
    episode: dict[str, object],
) -> None:
    content = episode.get("content", None)
    if not isinstance(content, dict):
        raise ValueError("Episode content not in expected list format.")
    messages = content.get("user_messages", None)
    if not isinstance(messages, list):
        raise ValueError("Messages not in expected list format.")
    if not messages:
        return
    instructions = dedent("""
        Your goal is to review a series of messages submitted by a user to an AI agent,
        and determine if there are any facts about the user which should be saved
        as a memory for the AI agent to use in future interactions with the user.

        In particular, there are two types of facts which should be saved as a memory.

        First are personal details about the user, such as his or her name, location,
        contact details, job, and preferences.

        Second are objects which the user refers to by name, such as the name of a pet,
        or the name of a robot, name of best friend, and so on.

        It is not a requirement to find facts if the messages do not contain any facts
        which may be helpful in future interactions with the user.

        Please refer a string dictionary.  The dictionary key is a snake_case 
        description of the fact, such as user_name, user_job, pet_name, robot_name.
        The dictionary value is 
    """)
    memory_agent = AgentDefinition(
        "memory_agent",
        description="",  # only needed for agent as a tool
        model="gpt-4o",
        instructions=instructions,
        output_type=_FactsList,
    )
    s = "s" if len(messages) > 1 else ""
    are = "are" if len(messages) > 1 else "is"
    prompt_lines = [f"Here {are} the message{s} from which to obtain facts."]
    message_count = 0
    for message in messages:
        if not isinstance(message, str):
            raise ValueError("Message was not a string.")
        message_count += 1
        prompt_lines.extend(
            [
                "\n---",
                message,
            ]
        )
    prompt = "\n".join(prompt_lines)
    result = await AgentHarness.run(
        memory_agent,
        prompt=prompt,
        run_identifier=f"{run_identifier}-reflection",
    )
    if not isinstance(result.final_output, _FactsList):
        raise ValueError("Unexpected result format.")
    for fact in result.final_output.facts:
        await asyncio.to_thread(save_memory, fact.key, fact.value)


class _Fact(BaseModel):
    key: str
    value: str


class _FactsList(BaseModel):
    facts: list[_Fact] = Field(max_length=5)


def _record_instructions(
    instructions: str,
) -> None:
    output_dir = os.environ.get("SANDBOX_OUTPUT_DIR", "")
    if not output_dir:
        return

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    instructions_file = output_directory / "instructions.txt"

    instructions_file.write_text(instructions, encoding="utf-8")


async def _record_snapshot(
    filename: str,
) -> None:
    if not _RECORD_MEMORY_SNAPSHOTS:
        return
    
    output_dir = os.environ.get("SANDBOX_OUTPUT_DIR", "")
    if not output_dir:
        return

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    snapshot_file = output_directory / filename

    snapshot = get_snapshot()

    snapshot_file.write_text(snapshot, encoding="utf-8")
