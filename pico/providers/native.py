"""Normalized structures for provider-native tool calling."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class NativeTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    raw_content: list[dict] = field(default_factory=list)

    @property
    def has_tool_calls(self):
        return bool(self.tool_calls)


def _parse_param(spec):
    raw = str(spec).strip()
    if "=" in raw:
        type_name, default_raw = raw.split("=", 1)
        type_name = type_name.strip()
        try:
            default = ast.literal_eval(default_raw.strip())
        except Exception:
            default = default_raw.strip()
        required = False
    else:
        type_name = raw
        default = None
        required = True

    json_type = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "boolean": "boolean",
    }.get(type_name, "string")
    return json_type, default, required


def to_json_schema(tools):
    """Convert pico's compact tool schema strings into JSON Schema tool defs."""
    definitions = []
    for name, tool in tools.items():
        properties = {}
        required = []
        for parameter, spec in tool.get("schema", {}).items():
            json_type, default, is_required = _parse_param(spec)
            prop = {"type": json_type}
            if not is_required:
                prop["default"] = default
            properties[parameter] = prop
            if is_required:
                required.append(parameter)

        input_schema = {"type": "object", "properties": properties}
        if required:
            input_schema["required"] = required
        definitions.append(
            {
                "name": name,
                "description": str(tool.get("description", "")),
                "input_schema": input_schema,
            }
        )
    return definitions


def parse_native_turn(data):
    """Parse an Anthropic-compatible response into a NativeTurn."""
    content = data.get("content", [])
    text_parts = []
    tool_calls = []
    raw_content = []

    for block in content:
        if not isinstance(block, dict):
            continue
        raw_content.append(block)
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
            continue
        if block_type != "tool_use":
            continue

        call_id = str(block.get("id", "")).strip()
        name = str(block.get("name", "")).strip()
        args = block.get("input")
        if not call_id or not name or not isinstance(args, dict):
            raise ValueError("malformed tool_use block")
        tool_calls.append(ToolCall(id=call_id, name=name, args=args))

    return NativeTurn(
        text="\n".join(text_parts).strip(),
        tool_calls=tool_calls,
        stop_reason=str(data.get("stop_reason", "")),
        raw_content=raw_content,
    )
