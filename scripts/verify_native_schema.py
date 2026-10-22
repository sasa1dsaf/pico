"""Verify pico schema -> JSON Schema conversion and native turn parsing."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pico.providers.native import parse_native_turn, to_json_schema


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def main():
    tools = {
        "read_file": {
            "schema": {"path": "str", "start": "int=1", "end": "int=200"},
            "risky": False,
            "description": "Read a file.",
        },
        "remember": {
            "schema": {
                "name": "str",
                "description": "str",
                "content": "str",
                "type": "str='user'",
                "tags": "str=''",
            },
            "risky": False,
            "description": "Save a memory.",
        },
        "flags": {
            "schema": {"enabled": "bool=True", "ratio": "float=0.5"},
            "risky": False,
            "description": "Flag test.",
        },
    }

    definitions = {item["name"]: item for item in to_json_schema(tools)}

    read_schema = definitions["read_file"]["input_schema"]
    check("read_file required path", read_schema["required"] == ["path"])
    check("read_file start type", read_schema["properties"]["start"]["type"] == "integer")
    check("read_file start default", read_schema["properties"]["start"]["default"] == 1)
    check("read_file end default", read_schema["properties"]["end"]["default"] == 200)

    remember_schema = definitions["remember"]["input_schema"]
    check(
        "remember required fields",
        remember_schema["required"] == ["name", "description", "content"],
    )
    check("remember type default", remember_schema["properties"]["type"]["default"] == "user")
    check("remember tags default", remember_schema["properties"]["tags"]["default"] == "")

    flags_schema = definitions["flags"]["input_schema"]
    check("bool maps to boolean", flags_schema["properties"]["enabled"]["type"] == "boolean")
    check("float maps to number", flags_schema["properties"]["ratio"]["type"] == "number")
    check("bool default parsed", flags_schema["properties"]["enabled"]["default"] is True)

    turn = parse_native_turn(
        {
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "x.py"}},
            ],
        }
    )
    check("native turn text", turn.text == "checking")
    check("native turn stop reason", turn.stop_reason == "tool_use")
    check("native turn tool call", turn.tool_calls[0].name == "read_file")
    check("native turn tool args", turn.tool_calls[0].args == {"path": "x.py"})
    check("native turn raw content", len(turn.raw_content) == 2)

    try:
        parse_native_turn(
            {"content": [{"type": "tool_use", "id": "", "name": "read_file", "input": {}}]}
        )
    except ValueError:
        malformed_rejected = True
    else:
        malformed_rejected = False
    check("malformed tool_use rejected", malformed_rejected)

    print("ALL NATIVE SCHEMA CHECKS PASSED")


if __name__ == "__main__":
    main()
