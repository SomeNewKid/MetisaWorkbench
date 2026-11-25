"""HTTP Server tests for the sample agent."""

from __future__ import annotations

import requests

from metisa_common.specification_models import (
    Capability,
    MetisaSpecification,
)

from .._helper import (
    print_caution,
    print_disabled,
    print_elapsed_time,
    print_failure,
    print_success,
    print_suite_title,
)

_SKIP_HTTP_SERVER_TESTS = False


def test_http_server(specification: MetisaSpecification):
    """Run HTTP Server tests."""
    print_suite_title("HTTP Server")

    if _SKIP_HTTP_SERVER_TESTS:
        print_caution("HTTP Server capability is being skipped.")
        return

    if Capability.HTTP_SERVER not in specification.capabilities:
        print_disabled("HTTP Server capability not declared.")
        return

    with print_elapsed_time():
        try:
            response = requests.get("http://metisa-http-server:8080/hello", timeout=10)
            if response.status_code != 200:
                print_failure("HTTP Server did not return a successful response.")
                return
        except requests.RequestException as e:
            print_failure(f"HTTP Server request failed: {e}")
            return

        content = response.text
        if "hello" in content.lower():
            print_success("HTTP Server returned the expected content.")
        else:
            print_failure("HTTP Server did not return the expected content.")
