# Metisa Workbench

Metisa Workbench brings together two Python projects for learning how AI agents
work and how to isolate their workloads.

- **Metisa Harness** runs the model and tool loop for an agent. Following the
  division of responsibilities in the OpenAI Agents SDK, `AgentDefinition`
  describes the model, instructions, tools, and skills, and `AgentHarness`
  executes the run. Start with the [Windows harness sample](#run-the-harness-sample-on-windows).
- **Metisa Sandbox** runs Python workloads in temporary, hardened Linux
  containers. It reads the workload's `metisa.toml`, builds or reuses Docker
  images, provisions services, checks the sandbox restrictions, and cleans up
  after the run. It uses explicit `docker` CLI commands to make container
  networking, isolation, and Linux hardening easier to study. Start with
  [the sandbox test agent](#run-the-sandbox-test-agent).

Both components use `metisa_common` for shared helpers. In this checkout,
`sample_agent` runs on the Windows host with local Ollama and OpenAI clients,
and `test_agent` runs under Docker to exercise the sandbox services. Running
the harness sample inside the sandbox would require porting its Windows-specific
entry point and making the harness package available in the container.

> [!WARNING]
> The sandbox is experimental. Its controls and runtime probes support learning
> and hardening, and containment of hostile code remains unverified.

## Requirements

- **Windows and PowerShell.** The harness sample uses Windows user identity
  APIs. The sandbox host CLI uses Windows-specific terminal, identity, and
  Docker Desktop integration.
- **Python 3.11 or later.** The Python container images use Python 3.12.
- **For the harness sample**, run Ollama on its default local port (`11434`)
  with `granite4.1:3b`, and set `OPENAI_API_KEY` for `gpt-4o`.
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

## Run the harness sample on Windows

With Ollama running and `OPENAI_API_KEY` set in the current PowerShell session,
run the [sample agent](src/sample_agent) directly.

```powershell
ollama pull granite4.1:3b
.\.venv\Scripts\python.exe -m sample_agent "What is the time?"
```

The CLI uses the same prompt with `gpt-4o` and `granite4.1:3b`. Try
`Look up product code "chair".` to exercise the product tool and its guardrails,
or `How many company values do we have?` to query example handbook excerpts
through the sample's in-memory MCP server. The `[mcp_server]` table in
[the sample specification](src/sample_agent/metisa.toml) configures the sandbox
MCP sidecar. With `otel_collector` enabled, the Windows CLI writes traces to
`.runs/<run-id>/.logs/telemetry/traces.json`.

Try these prompts with the [robot inspection skills](src/sample_agent/skills).

```powershell
.\.venv\Scripts\python.exe -m sample_agent "Can the robot named Bolt enter the race?"
.\.venv\Scripts\python.exe -m sample_agent "Can the robot named Alice be placed in storage?"
```

Both skills use the battery and motor-temperature tools. The race skill loads
a reference with the race limits. The storage skill loads a checklist and, if
both measurements pass, runs `storage_available.star` to check for a bay.
The sample data gives Bolt passing race measurements and Alice passing storage
measurements with a bay available. The Alice question is the default if you
omit the prompt. To print lifecycle events for the skill, document, and script
calls, set `_PRINT_LIFECYCLE_EVENTS` in [the sample CLI](src/sample_agent/cli.py).

## Use the harness from Python

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

The default model resolver queries Ollama first, even for the OpenAI aliases
`gpt4o` and `gpt-4o`. An unreachable Ollama server stops model resolution.
Pass `llm=` to `AgentHarness.run()` to supply a model client directly.

### How a harness run works

1. The harness registers synchronous or asynchronous Python tools, discovers
   tools from connected MCP clients, and exposes `sub_agents` as tools. A
   sub-agent tool starts a child run and returns its final output to the parent.
   A skills directory adds tools for loading skills and their supporting files.
2. The model receives the instructions, available tools, and conversation. By
   default each model call receives the run's full conversation history.
3. The harness handles tool requests in order, validates arguments, executes
   the tools, validates results against declared output schemas, and sends the
   results to the model on its next turn.
4. `RunResult` contains the final output and conversation history. Set
   `output_type` on the definition to a Pydantic model for validated structured
   output. The run raises an error if it reaches `max_turns` without a final
   answer. The default limit is 10 turns.

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
`{"robot_name": "Alice"}` to the script tool. The local harness runs the file
with Python `exec()` and calls `main` with the named arguments. Use this runner
with trusted, Python-compatible scripts. The sandbox provides a separate
Starlark service through its `code_execution` capability.

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
| `playwright_chromium` | Starts a headless Chromium sidecar and permits the workload's bundled Playwright driver. Requires `network` and a compatible Playwright dependency in the workload. Add `internet` and a Squid allowlist for public websites. |
| `code_execution` | Provides Starlark execution in a separate Go service. Requires `mcp_server`. |
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

Add `internet` and allow the desired domains or IP addresses in `[squid_proxy]`
to navigate to public websites. Browser requests originate in the sidecar and
use Squid. The sample's
[browser test](src/test_agent/test_suite/playwright_chromium.py) renders an
inline page when `internet` is absent and visits `https://example.com/` when it
is enabled. The test agent allowlist includes `.example.com`.

## Run the sandbox test agent

[`test_agent`](src/test_agent) exercises the sandbox services, including
network access, database queries, text guards, model requests, documents,
retrieval, interactive input, agent memory, secrets, tracing, and remote browser
automation. The checked-in configuration enables public-site browser automation
and the service integrations below. It leaves interactive input disabled. The
browser exercise saves `chromium.png` in the run output directory. If you enable
interactive input, its exercise writes `answer.txt` there as well.

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

The browser sidecar joins the workload network and also joins the backend
network when one is created. In a run with `internet` and `llm_broker`, the
workload connects to the browser on the workload network, and the browser
reaches Squid through the backend network. This preserves the broker
configuration's restriction on direct Squid access from the workload.
Metisa waits for the browser sidecar's WebSocket health check before starting
the workload. See [the network diagram](network.txt) for the full topology.

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
| `src/metisa_common/` | Shared specification, tool schema, terminal, secrets, screening, and client utilities. |
| `src/metisa_landlock/` | Linux filesystem restrictions and workload bootstrap. |
| `src/metisa_runner/` | Probe execution and handoff to the workload. |
| `src/metisa_probes/` | Container, kernel, and Python security checks. |
| `src/metisa_mcp_server/` | Configured MCP tools and resources. |
| `src/metisa_llm_broker/` | Restricted local and online model request routing. |
| `src/metisa_agent_memory/` | Authenticated, user-scoped episode and memory storage, sanitization, and screening. |
| `src/metisa_prompt_guard/`, `src/metisa_entity_guard/` | Text classification, moderation, and anonymization services. |
| `src/metisa_docling/`, `src/metisa_embeddings/` | Document conversion, indexing, and retrieval services. |
| `src/sample_agent/` | Windows harness sample with example tools, robot inspection skills, guardrails, and an in-memory MCP server. |
| `src/test_agent/` | Sandbox integration workload, TOML configuration, and service exercises. |
| `tests/` | Unit tests. |
| `scripts/` | PowerShell setup and development checks. |

Follow [AGENTS.md](AGENTS.md) for the project's package boundaries, public
interfaces, naming conventions, and Python style. Keep Docker and Linux
configuration explicit so the implementation remains useful for learning.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Harness reports `Ollama discovery failed` | Start Ollama and check that the sample can reach it at `localhost:11434`. Discovery also runs for OpenAI aliases. |
| A harness sample model fails to run | Check that `granite4.1:3b` is installed in Ollama and that `OPENAI_API_KEY` is set for `gpt-4o`. |
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
| Chromium reports `ERR_PROXY_CONNECTION_FAILED` | Check the sidecar and Squid network attachments. With `llm_broker` enabled, they must share the backend network. Check `.logs/squid-proxy/stdout.txt` for the browser's `CONNECT` request. |
| A service is slow or exits unexpectedly | Inspect its logs and Docker resource allocation; models have different memory and startup requirements. |
| PostgreSQL container name conflict | Check whether another run is using `metisa-postgres-agent-memory`; runs using PostgreSQL must execute sequentially. |
| Agent memory cannot connect to PostgreSQL | Check `METISA_POSTGRES_PASSWORD` against the password used to initialize the retained volume, plus the database and agent memory logs. |
| MCP or agent memory returns an authentication error | Check that the correct service token is supplied and has not expired; tokens are issued for one run and last one hour. |

## License

See [LICENSE](LICENSE) for the GNU General Public License, version 3.
