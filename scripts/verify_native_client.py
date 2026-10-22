"""Verify AnthropicCompatibleModelClient.complete_native parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pico.providers.clients import AnthropicCompatibleModelClient


class FakeResponse:
    headers = {"Content-Type": "application/json"}

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def main():
    client = AnthropicCompatibleModelClient(
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/anthropic",
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
        thinking={"type": "disabled"},
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "检查一下记忆。"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "read_memory",
                        "input": {"name": "user-food-preferences"},
                    },
                ],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        )

    tools = [
        {
            "name": "read_memory",
            "description": "Read memory.",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
    ]
    messages = [{"role": "user", "content": [{"type": "text", "text": "我喜欢啥"}]}]

    with patch("urllib.request.urlopen", fake_urlopen):
        turn = client.complete_native(
            system="You are pico.",
            messages=messages,
            tools=tools,
            max_new_tokens=256,
            tool_choice={"type": "auto"},
        )

    check("native client: stop_reason", turn.stop_reason == "tool_use")
    check("native client: text", turn.text == "检查一下记忆。")
    check("native client: tool call name", turn.tool_calls[0].name == "read_memory")
    check("native client: tool call args", turn.tool_calls[0].args == {"name": "user-food-preferences"})
    check("native client: raw content", len(turn.raw_content) == 2)
    check("native client: system sent", captured["payload"]["system"] == "You are pico.")
    check("native client: tools sent", captured["payload"]["tools"] == tools)
    check("native client: tool_choice sent", captured["payload"]["tool_choice"] == {"type": "auto"})
    check("native client: thinking disabled", captured["payload"]["thinking"] == {"type": "disabled"})
    check("native client: metadata stop_reason", client.last_completion_metadata["stop_reason"] == "tool_use")
    check(
        "native client: metadata content types",
        client.last_completion_metadata["content_block_types"] == ["text", "tool_use"],
    )

    def fake_final_urlopen(request, timeout):
        return FakeResponse(
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "你喜欢吃土豆、豆腐。"}],
                "usage": {"input_tokens": 120, "output_tokens": 10},
            }
        )

    with patch("urllib.request.urlopen", fake_final_urlopen):
        final = client.complete_native(
            system="You are pico.",
            messages=messages,
            tools=tools,
            max_new_tokens=256,
        )
    check("native client: final text", final.text == "你喜欢吃土豆、豆腐。")
    check("native client: no tool calls", final.tool_calls == [])
    check("native client: end_turn", final.stop_reason == "end_turn")

    def fake_malformed_urlopen(request, timeout):
        return FakeResponse(
            {
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "", "name": "read_memory", "input": {}}],
            }
        )

    try:
        with patch("urllib.request.urlopen", fake_malformed_urlopen):
            client.complete_native(
                system="You are pico.",
                messages=messages,
                tools=tools,
                max_new_tokens=256,
            )
    except RuntimeError as exc:
        malformed_rejected = "malformed tool_use" in str(exc)
    else:
        malformed_rejected = False
    check("native client: malformed rejected", malformed_rejected)

    print("ALL NATIVE CLIENT CHECKS PASSED")


if __name__ == "__main__":
    main()
