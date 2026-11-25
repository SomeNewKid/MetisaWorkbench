"""Manage a HTTP Server sidecar."""

from __future__ import annotations

from pathlib import Path

from metisa_common.specification_helper import (
    generate_image_tag,
    resolve_environs,
)
from metisa_common.specification_models import MetisaSpecification

from .._docker._docker_container import (
    create_docker_container,
)
from .._docker._docker_hardening import (
    harden_docker_container,
)
from .._docker._docker_image import (
    build_docker_image,
    docker_image_exists,
)
from .._sandbox_models import (
    ContainerCreationContext,
    ContainerCreationResult,
    SandboxContainer,
    SandboxContext,
)

_IMAGE_FORMAT_VERSION = 1
_HTTP_SERVER_ALIAS = "metisa-http-server"
_HTTP_SERVER_IMAGE_NAME = "metisa-http-server"
_USE_SQUID_PROXY = True


def create_http_server_definition() -> SandboxContainer:
    """Return the definition of the HTTP Server container."""
    return SandboxContainer(
        "HTTP Server",
        "http_server",
        log_folder="http-server",
        log_files=[],
    )


def create_http_server_container(
    specification: MetisaSpecification,
    sandbox_context: SandboxContext,
    container_creation_context: ContainerCreationContext,
) -> ContainerCreationResult:
    """Create the HTTP Server container and connect it to the Docker network."""
    run_identifier = sandbox_context.run_identifier

    publish_directory = Path(__file__).resolve().with_name("publish")
    entry_assembly = publish_directory / "WebApp.dll"
    if not entry_assembly.is_file():
        raise RuntimeError(
            f"Missing {entry_assembly}. Publish WebApp before starting the sandbox.")

    image_name = _ensure_http_server_image_exists(specification)
    container_name = _create_http_server_container_name(run_identifier)

    if not container_creation_context.network_names:
        raise RuntimeError("No network specified for the HTTP Server container.")
    network_name = container_creation_context.network_names[0]

    environs: list[str] = []

    arguments = [
        "--name",
        container_name,
        "--network",
        network_name,
        "--network-alias",
        _HTTP_SERVER_ALIAS,
    ]

    # Mount the website.
    arguments.extend(
        [
            "--mount",
            f"type=bind,source={publish_directory},target=/app,readonly",
        ]
    )

    # Mount temporary (in memory) filesystems.
    arguments.extend(
        [
            "--tmpfs",
            "/tmp:rw,size=64m,nosuid,nodev,noexec",
        ]
    )

    # Get environment variables
    if specification.http_server is not None:
        resolved_environs = resolve_environs(specification.http_server.environs)
        environs.extend(resolved_environs)

    # Harden the container
    harden_docker_container(
        arguments=arguments,
        environs=environs,
        username="web_server",
        health_start_seconds=360,
        file_size_limit=2 * 1024**4,  # does not allocate 2 TB
    )

    # The image name must be specified last in the arguments list.
    arguments.extend(
        [
            image_name,
        ]
    )

    create_docker_container(arguments)

    return ContainerCreationResult(container_name)


def _ensure_http_server_image_exists(
    specification: MetisaSpecification,
) -> str:
    image_tag = _generate_image_tag(specification)
    image_reference = f"{_HTTP_SERVER_IMAGE_NAME}:{image_tag}"
    if not docker_image_exists(image_reference):
        _create_http_server_image(specification, image_reference)
    return image_reference


def _create_http_server_container_name(
    run_identifier: str,
) -> str:
    return f"http-server-{run_identifier}"


def _create_http_server_image(
    specification: MetisaSpecification,
    image_reference: str,
) -> None:
    dockerfile_location = _get_dockerfile_location()
    build_context_path = dockerfile_location.parent

    build_docker_image(
        image_reference=image_reference,
        dockerfile_path=dockerfile_location,
        build_context_path=build_context_path,
    )


def _get_dockerfile_location() -> Path:
    dockerfile_location = Path(__file__).with_name("dockerfile")
    if not dockerfile_location.exists():
        raise RuntimeError("Dockerfile is not available.")
    return dockerfile_location


def _generate_image_tag(
    specification: MetisaSpecification,
) -> str:
    collection: list[str] = []
    hash = generate_image_tag(collection)
    return f"{_IMAGE_FORMAT_VERSION}-{hash}"
