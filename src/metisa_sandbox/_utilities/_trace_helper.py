"""Summarize agent turns from the Collector's OTLP JSONL export."""

import json
from collections import defaultdict
from pathlib import Path
from textwrap import wrap
from typing import Any

_Span = dict[str, Any]
_SpanKey = tuple[str, str]


def write_friendly_trace(telemetry_directory: Path) -> None:
    """Write friendly.txt beside the traces.json file when it exists."""
    trace_path = telemetry_directory / "traces.jsonl"
    try:
        trace_file = trace_path.open(encoding="utf-8")
    except FileNotFoundError:
        return

    spans: list[_Span] = []
    skipped_lines = 0
    with trace_file:
        for line in trace_file:
            if not line.strip():
                continue
            try:
                batch = json.loads(line)
            except json.JSONDecodeError:
                skipped_lines += 1
                continue

            for resource in batch.get("resourceSpans", []):
                for scope in resource.get("scopeSpans", []):
                    spans.extend(scope.get("spans", []))

    lines = _summarize(spans)
    if skipped_lines:
        lines.append(f"Skipped {skipped_lines} invalid JSON line(s).")

    friendly_path = trace_path.with_name("friendly.txt")
    friendly_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(spans: list[_Span]) -> list[str]:
    spans.sort(key=lambda span: int(span["startTimeUnixNano"]))
    model_spans = [span for span in spans if span["name"] == "metisa.model_turn"]
    model_keys = {(span["traceId"], span["spanId"]) for span in model_spans}
    tools_by_model: dict[_SpanKey, list[_Span]] = defaultdict(list)
    unlinked_tools: list[_Span] = []

    for span in spans:
        if not span["name"].startswith("execute_tool "):
            continue

        for link in span.get("links", []):
            key = (link["traceId"], link["spanId"])
            if key in model_keys:
                tools_by_model[key].append(span)
                break
        else:
            unlinked_tools.append(span)

    lines: list[str] = []
    for model in model_spans:
        turn = next(
            (
                attribute["value"]["intValue"]
                for attribute in model.get("attributes", [])
                if attribute["key"] == "metisa.turn"
            ),
            "?",
        )
        turn_number = f"{turn}".rjust(3)
        lines.append(f"Turn {turn_number}   {'Model call':<63}  {_elapsed(model)}")
        tools = tools_by_model[(model["traceId"], model["spanId"])]
        for index, tool in enumerate(tools, start=1):
            branch = "└─" if index == len(tools) else "├─"
            name = tool["name"].removeprefix("execute_tool ")
            lines.append(f"           {branch} {name:<61} {_elapsed(tool)}")
            continuation = "|" if index < len(tools) else " "
            lines.extend(_reason_lines(tool, indent=f"           {continuation}  "))

    if unlinked_tools:
        lines.append("Tool calls with no matching model span")
        for tool in unlinked_tools:
            name = tool["name"].removeprefix("execute_tool ")
            lines.append(f"  {name:<31} {_elapsed(tool)}")

    return lines or ["No agent model turns or tool calls recorded."]


def _elapsed(span: _Span) -> str:
    elapsed_ns = int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])
    seconds = f"{elapsed_ns / 1_000_000_000:.2f}".removeprefix("0")
    duration = f"{seconds} s".rjust(10)

    status = span.get("status", {}).get("code")
    if status in (2, "STATUS_CODE_ERROR"):
        duration += " [error]"
    return duration


def _reason_lines(span: _Span, *, indent: str) -> list[str]:
    for attribute in span.get("attributes", []):
        if attribute["key"] != "metisa.tool.model_reasoning":
            continue

        value = attribute["value"].get("stringValue", "")
        reason = " ".join(value.split())
        if reason:
            return wrap(
                f"Reason: {reason}",
                width=78,
                initial_indent=indent,
                subsequent_indent=indent + "  ",
            )
    return []
