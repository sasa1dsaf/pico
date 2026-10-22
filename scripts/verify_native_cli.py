"""Verify tool-protocol selection and CLI integration."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pico.cli as pico_cli


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def main():
    parser = pico_cli.build_arg_parser()

    with patch.dict(os.environ, {}, clear=True):
        check(
            "auto: deepseek -> native",
            pico_cli._effective_tool_protocol(
                parser.parse_args(["--provider", "deepseek"])
            )
            == "native",
        )
        check(
            "auto: ollama -> text",
            pico_cli._effective_tool_protocol(
                parser.parse_args(["--provider", "ollama"])
            )
            == "text",
        )
        check(
            "explicit text",
            pico_cli._effective_tool_protocol(
                parser.parse_args(["--provider", "deepseek", "--tool-protocol", "text"])
            )
            == "text",
        )
        check(
            "explicit native",
            pico_cli._effective_tool_protocol(
                parser.parse_args(["--provider", "openai", "--tool-protocol", "native"])
            )
            == "native",
        )

    with patch.dict(os.environ, {"PICO_TOOL_PROTOCOL": "text"}, clear=True):
        check(
            "env text",
            pico_cli._effective_tool_protocol(
                parser.parse_args(["--provider", "deepseek"])
            )
            == "text",
        )

    with tempfile.TemporaryDirectory() as td:
        args = parser.parse_args(["--cwd", td, "--provider", "deepseek"])
        with patch.dict(os.environ, {"PICO_DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            with patch("pico.cli.AnthropicCompatibleModelClient"):
                agent = pico_cli.build_agent(args)
        check("build_agent: native for deepseek", agent.tool_protocol == "native")

    with tempfile.TemporaryDirectory() as td:
        args = parser.parse_args(
            ["--cwd", td, "--provider", "deepseek", "--tool-protocol", "text"]
        )
        with patch.dict(os.environ, {"PICO_DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            with patch("pico.cli.AnthropicCompatibleModelClient"):
                agent = pico_cli.build_agent(args)
        check("build_agent: explicit text", agent.tool_protocol == "text")

    print("ALL NATIVE CLI CHECKS PASSED")


if __name__ == "__main__":
    main()
