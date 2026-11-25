# Metisa Workbench

Metisa Workbench brings together two Python projects for learning how AI agents
work and how to isolate their workloads.

- **Metisa Harness** runs the model and tool loop for an agent. Following the
  division of responsibilities in the OpenAI Agents SDK, `AgentDefinition`
  describes the model, instructions, tools, and skills, and `AgentHarness`
  executes the run. Start with the [harness sample](#run-the-harness-sample).
- **Metisa Sandbox** runs Python workloads in temporary, hardened Linux
  containers. It reads the workload's `metisa.toml`, builds or reuses Docker
  images, provisions services, checks the sandbox restrictions, and cleans up
  after the run. It uses explicit `docker` CLI commands to make container
  networking, isolation, and Linux hardening easier to study. Start with
  [the sandbox test agent](#run-the-sandbox-test-agent).

Both components use `metisa_common` for shared helpers. The `sample_agent`
workload runs the harness inside the sandbox and sends model requests through
the LLM broker. The `test_agent` workload exercises the sandbox services.

> [!WARNING]
> The sandbox is experimental. Its controls and runtime probes support learning
> and hardening, and containment of hostile code remains unverified.

## Requirements

- **Windows and PowerShell.** The sandbox host CLI uses Windows-specific
  terminal, identity, and Docker Desktop integration. Workloads run in Linux
  containers.
- **Python 3.11 or later.** The Python container images use Python 3.12.
- **For sandbox runs**, use Docker Desktop in Linux container mode with
  `docker` on `PATH`, BuildKit support for named build contexts and build
  mounts, and a container kernel that supports Landlock. The workload
  bootstrap stops if it cannot apply its restrictions.
- Network access for installing dependencies, pulling images, downloading
  models used by some service builds, and making online model calls. The
  workload specification controls internet access inside the sandbox.

Service-heavy sandbox workloads require additional memory, disk space, and
model files. The test agent enables several such services by default.

## Installation

Run these commands from the repository root.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --editable ".[dev]"
```

The `dev` extra installs Ruff, Pyright, and pytest. Omit `[dev]` if development
tools are not needed. Installation includes the project's machine learning and
document-processing dependencies and may take some time.

The setup script provides the same development installation.

```powershell
.\scripts\setup-dev.ps1
```

That script currently contains a developer-specific absolute Python path. Use
the manual commands above on another machine, or adjust the script's
`$HostPython` value for your installation.

Keep the checkout available and run the examples from the repository root.
Sandbox provisioning reads source files and Dockerfiles from the checkout.

## Run the harness sample

Review the [sample specification](src/sample_agent/metisa.toml) before
launching the workload from the repository root.

1. Set `local_models_directory` to your model directory. It must contain the
   [Prompt Guard model files](#run-the-sandbox-test-agent) and the GGUF file
   declared under `[llm_server].models`, at
   `<local_models_directory>/<repo>/<revision>/<filename>`. The checked-in
   configuration starts the local `granite-4-1b` server even though the sample
   CLI selects `gpt-4o` through the broker.
2. Set `OPENAI_API_KEY` in the PowerShell session that launches the sandbox.
   The broker uses it for `gpt-4o`. Set the diagnostic environment value shown
   below so the pre-workload probe passes.
3. Set `SANDBOX_TESTER_MARIADB_CREDENTIALS` in `username,password` format if
   you plan to use the database tool. Using that tool also requires the
   `localnet` capability and a reachable MariaDB backend. For the examples
   below, you can remove the database tool and its secret reference from
   `[mcp_server]`.

```powershell
$env:METISA_PROBE_2 = "Retrieved from host's environment variables"
.\.venv\Scripts\python.exe -m metisa_sandbox sample_agent "What is the time?"
```

The sandbox supplies `METISA_RUN_ID` and `METISA_USER_SID` to the sample.
With the sample's `interactive` capability enabled, omitting the prompt opens
a screened terminal prompt. With `interactive` disabled, the sample defaults
to `Issue a refund for $75.40.`. The refund guardrail requires interactive
approval for that amount.

Try `Look up product code "chair".` to exercise the product tool and its
guardrails, or `How many company values do we have?` to query example handbook
excerpts through the sample's in-memory MCP server. The sample also connects to
the sandbox MCP sidecar when `mcp_server` is enabled. If you enable
`otel_collector`, the collector writes traces to
`.runs/<run-id>/.logs/telemetry/traces.jsonl`.

Try these prompts with the [robot inspection skills](src/sample_agent/skills).

```powershell
.\.venv\Scripts\python.exe -m metisa_sandbox sample_agent "Can the robot named Bolt enter the race?"
.\.venv\Scripts\python.exe -m metisa_sandbox sample_agent "Can the robot named Alice be placed in storage?"
```

Both skills use the battery and motor-temperature tools. The race skill loads
a reference with the race limits. The storage skill loads a checklist and, if
both measurements pass, runs `storage_available.star` to check for a bay.
The sample data gives Bolt passing race measurements and Alice passing storage
measurements with a bay available. To print lifecycle events for model and tool
calls, set `_PRINT_LIFECYCLE_EVENTS` in [the sample CLI](src/sample_agent/cli.py).

## Use the harness from Python

This example uses the default model client. Run it in a sandbox workload with
`metisa_harness` enabled and an LLM broker configured to serve `gpt-4o`. The
client sends Chat Completions requests to `http://metisa-llm-broker:8000/v1`.
Pass `llm=` to `AgentHarness.run()` to supply a model client directly.

```python
import asyncio

from metisa_harness import AgentDefinition, AgentHarness


async def main() -> None:
    agent = AgentDefinition(
        name="example_agent",
        description="Answers users' questions.",
        model="gpt-4o",
        instructions="Answer the user's questions concisely.",
    )
    result = await AgentHarness.run(
        agent,
        run_identifier="example-run",
        input="What is an AI agent harness?",
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

### How a harness run works

1. The harness registers synchronous or asynchronous Python tools, discovers
   tools from connected MCP clients, and exposes `sub_agents` as tools. A
   sub-agent tool starts a child run and returns its final output to the parent.
   Sub-agents can run their own tools and cannot delegate to further sub-agents.
   A skills directory adds tools for loading skills and their supporting files.
2. The model receives the instructions, available tools, and conversation. By
   default each model call receives the run's full conversation history.
3. The harness handles tool requests in order, validates arguments, executes
   the tools, validates results against declared output schemas, and sends the
   results to the model on its next turn.
4. `RunResult` contains the final output and conversation history. Set
   `output_type` on the definition to a Pydantic model for validated structured
   output. The adapter puts the JSON schema in the model instructions, and the
   harness parses and validates the final answer. The default limit is 10 model
   calls. If the last call requests tools, the harness dispatches them before
   raising the turn-limit error.

Add local functions through `tools`. Use type hints and attach a JSON output
schema with `metisa_common.mcp_helper.with_output_schema`. Open MCP clients in
the caller and pass them through `mcp_clients`.

Guardrail plugins can inspect, transform, or reject data at run, model, and tool
boundaries. An optional async `event_handler` receives lifecycle events.
Supply a `context_builder` to customize the conversation sent to the model.

### Add a harness skill

Set `skills_directory=Path("path/to/skills")` on `AgentDefinition`. Give each
skill its own immediate child directory with `name` and `description` in the
YAML front matter of `SKILL.md` and the procedure in the Markdown body. Refer
to supporting files by paths relative to the skill directory. The sample uses
`references/` for documents and `scripts/` for scripts.

The harness uses **progressive disclosure**. The model receives skill names
and descriptions in its instructions and loads the full procedure and
supporting files as needed.

| Tool | Arguments | Result |
| --- | --- | --- |
| `load_skill` | `skill_name` | The body of `SKILL.md` as `content`. |
| `get_skill_document` | `skill_name`, `document_path` | The document text as `content`. |
| `execute_skill_script` | `skill_name`, `script_path`, `script_parameters` | The script return value as a string in `result`. |

The storage example passes `scripts/storage_available.star` and
`{"robot_name": "Alice"}` to the script tool. The harness reads the file from
the skill directory and sends its contents and parameters to the sandbox MCP
tool `execute_starlark_code`. To use skill scripts, enable `mcp_server` and
`code_execution` and include `execute_starlark_code` in `[mcp_server].tools`.

## Run a sandbox workload

Pass the name of a workload package located under `src`.

```powershell
.\.venv\Scripts\python.exe -m metisa_sandbox test_agent
```

To use the installed console entry point, activate the virtual environment.

```powershell
.\.venv\Scripts\Activate.ps1
metisa-workbench test_agent
```

The argument identifies a Python package containing `metisa.toml` and an
executable `__main__.py`. See [the test agent setup](#run-the-sandbox-test-agent)
for the dependencies required by `test_agent`.
If Docker Engine is unavailable, the application offers to start Docker Desktop.
Images are built when the required image reference does not already exist.

The CLI returns the workload's exit status, or a nonzero status when setup or
the pre-workload security probes fail.

### Create a minimal workload

Create the following package for a workload that runs without networking or
optional services.

```text
src/
└── hello_agent/
    ├── __init__.py
    ├── __main__.py
    └── metisa.toml
```

Leave `__init__.py` empty and put the following in `metisa.toml`.

```toml
agent_name = "hello_agent"
capabilities = []
dependencies = []
environs = []
secrets = []
settings = []
```

In `__main__.py`, write a greeting to the supplied output directory.

```python
"""Write a greeting from inside the sandbox."""

import os
from pathlib import Path

output_directory = Path(os.environ["SANDBOX_OUTPUT_DIR"])
output_file = output_directory / "greeting.txt"
output_file.write_text("Hello from Metisa Sandbox!\n", encoding="utf-8")
print("Wrote greeting.txt")
```

Run the workload from the repository root.

```powershell
.\.venv\Scripts\python.exe -m metisa_sandbox hello_agent
```

Docker still needs its build dependencies to construct the image. The empty
capability list disables workload networking at runtime.

## Workload configuration

See [the test agent specification](src/test_agent/metisa.toml) for the browser
configuration and examples of other service settings. The parser and validation
rules are implemented in [`metisa_common`](src/metisa_common).

### Top-level fields

| Field | Purpose |
| --- | --- |
| `agent_name` | Required, nonempty workload name. The CLI argument selects the package to execute. |
| `capabilities` | Array of capabilities requested by the workload. |
| `dependencies` | Python package requirements installed into the workload image. |
| `environs` | Environment declarations in `NAME=value` form. |
| `secrets` | Secret declarations in `NAME=value` form, provisioned separately from environment variables. |
| `settings` | Application settings in `name=value` form, including `local_models_directory`. |

Dependency declarations currently accept package names with `==`, `>=`, or `~=`
and numeric versions, such as `requests==2.34.2`. They do not support every form
of pip requirement, such as URLs, extras, or compound version constraints.

Environment variables and secrets can resolve values from the host process:

```toml
environs = ["AGENT_LABEL=${host:AGENT_LABEL}"]
secrets = ["SERVICE_TOKEN=${host:SERVICE_TOKEN}"]
```

Set the referenced variables in the PowerShell session used to launch the
sandbox. The application does not automatically load a `.env` file. Use host
references for credentials rather than committing credential values in TOML.

### Capabilities and services

| Capability | Behavior |
| --- | --- |
| `network` | Connects the workload to an internal Docker network; otherwise it uses `--network none`. |
| `internet` | Creates Squid-controlled egress using `[squid_proxy]` domain and IP allowlists. Requires `network`. |
| `localnet` | Creates HAProxy routes to the destinations in `[haproxy].backends`. Requires `network`. |
| `interactive` | Enables attached standard input. Requires `mcp_server`; the provided input helper screens text through MCP. |
| `mcp_server` | Provides the tools and resources selected in `[mcp_server]`. Requires `network`, `prompt_guard`, and `entity_guard`. |
| `prompt_guard` | Provides prompt injection checks and text moderation. Started alongside MCP or the LLM broker. |
| `entity_guard` | Provides entity detection and anonymization. Started alongside MCP or the LLM broker. |
| `jina_reader` | Provides web content extraction through Jina Reader. Requires `internet` and `mcp_server`. |
| `playwright_chromium` | Starts a headless Chromium sidecar and permits the workload's bundled Playwright driver. Requires `network` and a compatible Playwright dependency in the workload. Use with `http_server` for browser tests of a local site. |
| `http_server` | Serves an ASP.NET Core 8 application on the internal workload network. Requires `network`. |
| `code_execution` | Provides Starlark execution in a separate Go service. Requires `mcp_server`. |
| `metisa_harness` | Stages the `metisa_harness` Python package in the workload container. Configure `llm_broker` when using the default model client. |
| `llm_server` | Runs a llama.cpp container for each model in `[llm_server].models`. Requires `network`; use with the broker for workload access. |
| `llm_broker` | Routes requests to configured local model aliases and online OpenAI models. Requires `network`, `prompt_guard`, and `entity_guard`. |
| `documents` | Serves files staged from `[documents].local_directory` through Nginx. |
| `docling` | Converts documents to Markdown and chunks. Requires `mcp_server` and `documents`. |
| `qdrant` | Provides a vector database for retrieval. |
| `embeddings` | Indexes document chunks and searches for matching content using Qdrant. Requires `docling`; include `qdrant` for storage. |
| `otel_collector` | Collects OpenTelemetry traces from configured components. |
| `postgres` | Provides PostgreSQL with pgvector and initializes the agent memory schema. Database storage persists across runs. |
| `agent_memory` | Provides authenticated HTTP operations for episodes and memories, scoped to the launching Windows user. Requires `postgres`; content screening also needs Prompt Guard. |

Include the configuration tables and prerequisite capabilities needed by each
service. Use the test agent specification as a reference for combinations of
services. Some service prerequisites must be checked manually.

For example, controlled internet access can be configured with:

```toml
agent_name = "web_agent"
capabilities = ["network", "internet"]
dependencies = ["requests==2.34.2"]

[squid_proxy]
allowed_domains = [".example.com"]
allowed_ip_addresses = []
```

MCP tools include database queries, URL retrieval, text classification and
moderation, anonymization, Starlark execution, document conversion, and semantic
search. Optional tools must be explicitly selected in `[mcp_server].tools`;
the input screening tool is registered automatically.

Interactive workloads can use `metisa_common.terminal_helper.read_user_input`
to read and screen input through MCP. The screening tool checks for prompt
injection and moderation labels and anonymizes detected entities. Input requires
the `interactive` capability and must go through this helper; direct `input()`
calls are disabled.

The LLM broker exposes a restricted, non-streaming subset of the models, chat
completions, and responses APIs. It validates model aliases, request fields,
payload sizes, and response shapes. Streaming, stored responses, and arbitrary
API fields are not supported. It sanitizes supported request and response text
and checks it for prompt injection indicators through Prompt Guard. It rejects
flagged content and returns an error when classification is unavailable.

The agent memory service stores episodes and memories in PostgreSQL, including
supplied memory embeddings. It sanitizes and screens record content and derives
the storage namespace from the authenticated Windows user SID. Its current API
supports creating, retrieving, and deleting episodes, plus creating, retrieving,
updating, and deleting memories. This storage is separate from the Qdrant-based
document retrieval service.

For agent memory, include `network`, `postgres`, and `agent_memory`, and enable
MCP or the LLM broker with their required guard capabilities so that Prompt
Guard is started. The validator does not enforce this complete combination.

### Remote Chromium

Declare the Python client in the workload's `metisa.toml`. The sidecar
currently uses Playwright 1.64.1, and the example pins the client to the same
version.

```toml
agent_name = "browser_agent"
capabilities = ["network", "playwright_chromium"]
dependencies = ["playwright==1.64.1"]
```

From the workload, call `playwright.chromium.connect()` with
`ws://metisa-playwright-chromium:3000/metisa-browser` inside a `sync_playwright()`
context. The Python client starts its bundled Node driver in the workload.
Chromium and its own sandbox run in the sidecar. The workload can use the
normal Playwright page API to navigate, click, and capture screenshots.
`[playwright_chromium].environs` can supply environment variables to the
sidecar, such as `DEBUG=pw:browser` for browser diagnostics.

The [browser test](src/test_agent/test_suite/playwright_chromium.py) opens the
HTTP Server when `http_server` is enabled and renders an inline page otherwise.
Use the configured MCP Server tools to fetch public web content for Prompt
Guard screening.

### HTTP Server

Declare `http_server` alongside `network` in the workload's `metisa.toml`.
The sidecar serves an ASP.NET Core 8 application at
`http://metisa-http-server:8080`. The workload can make HTTP requests to this
URL directly or open it in remote Chromium when `playwright_chromium` is enabled.
Use `[http_server].environs` to set environment variables for the web application.

The [sidecar Dockerfile](src/metisa_sandbox/http_server_sidecar/dockerfile)
copies the published application from `publish/` in its build context into
`/app`. It starts `WebApp.dll` and checks `/hello` for readiness.

## Run the sandbox test agent

[`test_agent`](src/test_agent) exercises the sandbox services, including
network access, database queries, text guards, model requests, documents,
retrieval, interactive input, agent memory, secrets, tracing, and remote browser
automation. The checked-in configuration enables the HTTP Server and Playwright
Chromium. The browser exercise saves `chromium.png` in the run output directory.
If you enable interactive input, its exercise writes `answer.txt` there as well.

1. Review [its TOML file](src/test_agent/metisa.toml), especially the enabled
   capabilities, local paths, allowed destinations, and credentials.
2. Set `local_models_directory` to your model directory. GGUF files must already
   exist under `<directory>/<repo>/<revision>/<filename>`, matching each entry
   in `[llm_server].models`. The application does not download these files.
3. Provide Prompt Guard model directories beneath the same model directory:

   ```text
   meta-llama/Llama-Prompt-Guard-2-22M/11614a155199674a0a95e6602d6ab0417b790ed0/
   KoalaAI/Text-Moderation/838ade0edb66dcffc5532d08ff6ed5c899abfb5c/
   ```

   These paths are currently fixed in the Prompt Guard container manager.
   Each directory must contain the model and tokenizer files needed to load it
   locally. Obtain any required model access before preparing the files.
4. In the PowerShell session that launches the sandbox, set `METISA_PROBE_2`
   to the diagnostic value expected by the environment probe.

   ```powershell
   $env:METISA_PROBE_2 = "Retrieved from host's environment variables"
   ```

   Set `OPENAI_API_KEY` for the broker, `SANDBOX_TESTER_MARIADB_CREDENTIALS`
   for the MCP database tool, and `METISA_POSTGRES_PASSWORD` for PostgreSQL and
   agent memory. The MariaDB credential format is `username,password`. Use the
   password that initialized the volume when reusing an existing PostgreSQL
   volume.
5. Provide the MariaDB service configured through HAProxy, or remove the
   corresponding integration capability and configuration. The sample routes
   port 3306 to `host.docker.internal`.
6. Create and populate the configured document directory, currently `.docs`,
   with the files referenced by the document, Docling, and embeddings exercises
   in [the test suite](src/test_agent/test_suite). The checkout expects you to
   supply these files.

Launch the test agent from the repository root.

```powershell
.\.venv\Scripts\python.exe -m metisa_sandbox test_agent
```

Some service builds download additional models, including Docling artifacts,
embedding models, and the spaCy model used for anonymization. Expect the first
build to take longer than subsequent runs.

## Architecture

The host CLI loads the specification and creates a unique run context. It uses
explicit Docker commands to build images and create, connect, start, inspect,
stop, and remove containers.

The configured services communicate through these networks:

```text
Workload container
    └─ internal workload network
       ├─ Playwright Chromium
       ├─ HTTP Server
       └─ MCP, LLM broker, agent memory, and telemetry

Internal backend network ─ supporting services and proxies
Egress network ─ Squid and HAProxy routes to external destinations
```

Only the networks and services needed for the configured workload are created.
MCP, the LLM broker, agent memory, and the telemetry collector attach to both
the workload and backend networks. Agent memory accesses PostgreSQL on the
backend network. Squid handles permitted internet access, and HAProxy handles
configured local network routes. Services use Docker network aliases on
unpublished ports.

Playwright Chromium and the HTTP Server join the internal workload network.
The workload connects to the browser over WebSocket, and Chromium sends page
requests to the HTTP Server over that same network. Metisa waits for both
configured sidecars to pass their health checks before starting the workload.
See [the network diagram](network.txt) for the full topology.

Inside the workload container, startup follows this sequence:

```text
Secrets provisioning
    -> Landlock bootstrap
    -> security probes
    -> Python workload package
```

Failed secrets provisioning, Landlock setup, or security probes prevents the
workload from starting.

## Security model

The implementation combines several controls:

- Non-root container identities, dropped Linux capabilities, and
  `no-new-privileges`.
- Read-only root filesystems, read-only workload source mounts, dedicated
  output storage, and restricted memory-backed temporary directories.
- CPU, memory, process, file descriptor, and file size limits, with
  service-specific settings.
- A custom workload seccomp profile and Landlock filesystem allow rules.
- A separate seccomp profile for Chromium's namespace sandbox. The workload
  keeps its Landlock rules and has a 256 MiB limit with `playwright_chromium`
  to accommodate the local driver. The default workload limit is 128 MiB.
- Isolated Python startup, disabled bytecode writes, and runtime guards for
  selected imports, writable code locations, process spawning, sockets, and
  hardware access.
- Runtime probes covering container identity, filesystem and network access,
  Linux capabilities, resource limits, seccomp, Landlock, and Python behavior.
  The Chromium network probe checks that the browser cannot navigate to
  `http://example.com` and uses the HTTP Server as a local control when enabled.
- Encrypted staged secrets, with decryption keys delivered separately through
  stdin to a provisioning process and stored in a restricted tmpfs directory.
- MCP and agent memory authentication using separate, one-hour bearer tokens
  signed with a fresh RSA key for each run. Tokens identify the launching
  Windows user, sandbox run, and intended service; services verify the signature,
  audience, run, and expiry.
- Content sanitization and prompt injection screening for MCP results, supported
  LLM broker text, and agent memory records.

The host provisions `METISA_MCP_SERVER_TOKEN` and `METISA_AGENT_MEMORY_TOKEN`
as workload secrets when their respective services are enabled. The shared MCP
client supplies its token automatically; direct agent memory HTTP requests must
send the corresponding token in an `Authorization: Bearer <token>` header.

The Playwright WebSocket grants browser control to clients that can reach it
on the sidecar's attached networks. It has no application-level authentication.

The host Docker installation, Linux kernel, images, dependencies, and configured
external services remain part of the trusted environment. Build-time dependency
installation is separate from the workload's runtime restrictions.

## Output and cleanup

Runs are stored relative to the directory from which the CLI is launched:

```text
.runs/
└── run-<timestamp>/
    ├── output/       # Workload-generated files
    └── .logs/        # Probe, Landlock, and service logs
```

The workload receives `SANDBOX_OUTPUT_DIR=/sandbox-output` for writing retained
files. Normal cleanup removes containers, networks, staged source and secret
files, and scratch storage, while keeping output and collected logs. Cleanup
also runs when setup or execution raises an error. Docker images remain cached
for reuse.

PostgreSQL data is retained in the Docker volume `metisa-agent-memory`,
independently of per-run output and logs. Normal cleanup removes the PostgreSQL
container but preserves this volume. PostgreSQL uses the fixed container name
`metisa-postgres-agent-memory`, so runs using it cannot execute concurrently.

A forced termination or unavailable Docker daemon can leave resources behind.
Inspect the CLI's cleanup errors and use `docker ps -a` and `docker network ls`
to identify the affected run before removing its resources.

## Development

Install the development dependencies as described above, then run:

```powershell
.\scripts\check.ps1
```

The script runs Ruff formatting, Ruff linting, source docstring checks, Pyright,
and pytest. The formatting step can modify Python files.

To run only unit tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Unit tests cover the harness run loop, tools, guardrails, model resolution, and
structured output, as well as specification validation, MCP helpers, agent
memory, text guards, and security probe logic. Security probes run inside each
sandbox. The test agent provides additional service integration exercises.

### Repository layout

| Path | Contents |
| --- | --- |
| `src/metisa_harness/` | Agent definitions, run loop, model adapter, tool dispatch, skills, guardrails, and lifecycle events. |
| `src/metisa_sandbox/` | Host CLI, Docker helpers, lifecycle management, Dockerfiles, and sidecar provisioning. |
| `src/metisa_sandbox/playwright_chromium_sidecar/` | Browser image, Playwright WebSocket server and health check, and Chromium seccomp profile. |
| `src/metisa_sandbox/http_server_sidecar/` | ASP.NET Core runtime image, HTTP health check, and HTTP Server provisioning. |
| `src/metisa_common/` | Shared specification, tool schema, terminal, secrets, screening, and client utilities. |
| `src/metisa_landlock/` | Linux filesystem restrictions and workload bootstrap. |
| `src/metisa_runner/` | Probe execution and handoff to the workload. |
| `src/metisa_probes/` | Container, kernel, and Python security checks. |
| `src/metisa_mcp_server/` | Configured MCP tools and resources. |
| `src/metisa_llm_broker/` | Restricted local and online model request routing. |
| `src/metisa_agent_memory/` | Authenticated, user-scoped episode and memory storage, sanitization, and screening. |
| `src/metisa_prompt_guard/`, `src/metisa_entity_guard/` | Text classification, moderation, and anonymization services. |
| `src/metisa_docling/`, `src/metisa_embeddings/` | Document conversion, indexing, and retrieval services. |
| `src/sample_agent/` | Sandbox-hosted harness sample with example tools, robot inspection skills, guardrails, and an in-memory MCP server. |
| `src/test_agent/` | Sandbox integration workload, TOML configuration, and service exercises. |
| `tests/` | Unit tests. |
| `scripts/` | PowerShell setup and development checks. |

Follow [AGENTS.md](AGENTS.md) for the project's package boundaries, public
interfaces, naming conventions, and Python style. Keep Docker and Linux
configuration explicit so the implementation remains useful for learning.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Sample reports a missing `METISA_RUN_ID` or `METISA_USER_SID` | Launch `sample_agent` through `metisa_sandbox` so the sandbox supplies both values. |
| A harness sample model fails to run | Check the alias in `[llm_broker]` and inspect the broker logs. For `gpt-4o`, verify the host's `OPENAI_API_KEY` and the Squid allowlist. For a configured local alias, verify the GGUF path and inspect the model sidecar logs. |
| Docker command unavailable or Engine unreachable | Check `docker info`, Docker Desktop installation, Linux container mode, and `PATH`. |
| Workload package or specification not found | Run from the repository root with the project's virtual environment; check `src/<package>/metisa.toml`. |
| Invalid specification | Check capability prerequisites, table names, dependency syntax, and environment declarations. |
| Missing host variable | Set the variable referenced by `${host:NAME}` in the launching session. |
| Model file or directory not found | Check `local_models_directory`, exact revisions, filenames, and the fixed Prompt Guard paths. |
| Documents directory not found | Create the configured local directory or change `[documents].local_directory`. |
| Image build fails | Inspect Docker build output, dependency availability, model download access, and disk space. |
| Bootstrap or probes fail | Inspect `.runs/<run>/.logs`, kernel support, and the specific failed restriction or service check. |
| Playwright fails inside `sync_playwright()` | Check `.logs/metisa/stderr.txt` and `state.json` for a driver spawn denial or `OOMKilled`. If the guard changed, ensure the affected workload image was rebuilt. |
| Playwright cannot connect to Chromium | Check `.logs/playwright-chromium/state.json` for the WebSocket health check result, then verify the browser alias, port, and matching Playwright versions. |
| Chromium cannot load the local website | Check that `http_server` is enabled and Chromium is navigating to `http://metisa-http-server:8080`. |
| HTTP Server fails to start | Check `.logs/http-server/state.json` for the exit code and health-check output. Verify that the published files include `WebApp.dll` and that the app serves `/hello` on port `8080`. |
| A service is slow or exits unexpectedly | Inspect its logs and Docker resource allocation; models have different memory and startup requirements. |
| PostgreSQL container name conflict | Check whether another run is using `metisa-postgres-agent-memory`; runs using PostgreSQL must execute sequentially. |
| Agent memory cannot connect to PostgreSQL | Check `METISA_POSTGRES_PASSWORD` against the password used to initialize the retained volume, plus the database and agent memory logs. |
| MCP or agent memory returns an authentication error | Check that the correct service token is supplied and has not expired; tokens are issued for one run and last one hour. |

## License

See [LICENSE](LICENSE) for the GNU General Public License, version 3.
