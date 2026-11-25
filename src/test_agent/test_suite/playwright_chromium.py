"""Playwright Chromium tests for the sample agent."""

from __future__ import annotations

import os
from pathlib import Path

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

_SKIP_PLAYWRIGHT_CHROMIUM_TESTS = False


def test_playwright_chromium(specification: MetisaSpecification):
    """Run Playwright Chromium tests."""
    print_suite_title("Playwright Chromium")

    if _SKIP_PLAYWRIGHT_CHROMIUM_TESTS:
        print_caution("Playwright Chromium capability is being skipped.")
        return

    if Capability.PLAYWRIGHT_CHROMIUM not in specification.capabilities:
        print_disabled("Playwright Chromium capability not declared.")
        return

    with print_elapsed_time():
        from playwright.sync_api import Browser, expect, sync_playwright

        with sync_playwright() as playwright:
            browser: Browser | None = None
            try:
                browser = playwright.chromium.connect(
                    "ws://metisa-playwright-chromium:3000/metisa-browser",
                    timeout=10_000,
                )

                url = (
                    "http://metisa-http-server:8080/hello"
                    if Capability.HTTP_SERVER in specification.capabilities
                    else None
                )
                context = browser.new_context(
                    offline=not url,
                    viewport={"width": 640, "height": 360},
                )
                try:
                    page = context.new_page()
                    if url:
                        response = page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=10_000,
                        )
                        if response is None or not response.ok:
                            print_failure("HTTP Server navigation failed.")
                            return
                    else:
                        page.set_content("<h1>Remote Chromium ready</h1>")
                        expect(
                            page.get_by_role("heading", name="Remote Chromium ready")
                        ).to_be_visible(timeout=3_000)

                    screenshot = page.screenshot(type="png", timeout=5_000)
                finally:
                    context.close()
            finally:
                if browser is not None:
                    browser.close()

        if not screenshot.startswith(b"\x89PNG\r\n\x1a\n"):
            print_failure("Remote Chromium did not return a PNG screenshot.")
            return

        output_path = Path(os.environ["SANDBOX_OUTPUT_DIR"]) / "chromium.png"
        output_path.write_bytes(screenshot)
        print_success(f"Captured a remote Chromium screenshot at {output_path}.")
