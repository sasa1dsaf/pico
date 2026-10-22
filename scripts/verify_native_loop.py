"""Verify the native agent loop with a scripted native client."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pico import Pico, SessionStore, WorkspaceContext
from pico.features.durable_memory import MemoryStore
from pico.providers.native import NativeTurn, ToolCall


class FakeNativeClient:
    supports_prompt_cache = False

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        self.last_completion_metadata = {}

    def complete_native(self, system, messages, tools, max_new_tokens, tool_choice=None):
        self.calls.append(
            {
                "system": system,
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "tool_choice": copy.deepcopy(tool_choice),
            }
        )
        turn = self.turns.pop(0)
        self.last_completion_metadata = {
            "stop_reason": turn.stop_reason,
            "content_block_types": [
                str(block.get("type", "")) for block in turn.raw_content
            ],
            "output_tokens": 10,
        }
        return turn


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def make_agent(root, turns):
    client = FakeNativeClient(turns)
    agent = Pico(
        model_client=client,
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".pico" / "sessions"),
        approval_policy="auto",
        tool_protocol="native",
    )
    return agent, client


def make_tool_use(call_id, name, args):
    return {
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": args,
    }


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = MemoryStore(root / ".pico" / "memory")
        store.remember(
            "user-prefers-tofu",
            "用户喜欢的食物",
            "用户喜欢吃豆腐。",
            type="user",
            tags=["preference", "food"],
        )

        read_args = {
            "path": ".pico/memory/items/user-prefers-tofu.md",
            "start": 1,
            "end": 80,
        }
        turns = [
            NativeTurn(
                text="",
                tool_calls=[ToolCall(id="call_1", name="read_file", args=read_args)],
                stop_reason="tool_use",
                raw_content=[make_tool_use("call_1", "read_file", read_args)],
            ),
            NativeTurn(
                text="<final>用户喜欢吃豆腐。</final>",
                tool_calls=[],
                stop_reason="end_turn",
                raw_content=[
                    {"type": "text", "text": "<final>用户喜欢吃豆腐。</final>"}
                ],
            ),
        ]
        agent, client = make_agent(root, turns)
        progress = []
        agent.progress = progress.append

        answer = agent.ask("我喜欢啥")

        check("native loop: final answer", answer == "用户喜欢吃豆腐。")
        check("native loop: one tool step", agent.current_task_state.tool_steps == 1)
        check("native loop: two model calls", len(client.calls) == 2)
        check(
            "native loop: system is prefix",
            "You are pico" in client.calls[0]["system"],
        )
        check(
            "native loop: system uses native protocol",
            "native tool calling" in client.calls[0]["system"],
        )
        check(
            "native loop: system asks for plain final",
            "plain final answer" in client.calls[0]["system"],
        )
        first_content = client.calls[0]["messages"][0]["content"]
        check(
            "native loop: memory block in first message",
            any("<memory>" in str(block.get("text", "")) for block in first_content),
        )
        check(
            "native loop: current request in first message",
            any("我喜欢啥" in str(block.get("text", "")) for block in first_content),
        )
        second_messages = client.calls[1]["messages"]
        check("native loop: assistant tool_use replayed", second_messages[1]["role"] == "assistant")
        tool_result_block = second_messages[2]["content"][0]
        check("native loop: tool_result paired", tool_result_block["tool_use_id"] == "call_1")
        check(
            "native loop: tool_result content",
            "用户喜欢吃豆腐" in str(tool_result_block["content"]),
        )
        check(
            "native loop: progress shows tool",
            any("tool read_file" in line for line in progress),
        )

        native_messages = agent.session["native_messages"]
        check("native_messages persisted", len(native_messages) == 4)
        check(
            "native_messages user has no memory block",
            "<memory>" not in str(native_messages[0]["content"]),
        )
        check(
            "native_messages tool result persisted",
            "用户喜欢吃豆腐" in str(native_messages[2]["content"][0]["content"]),
        )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        invalid_args = {"start": 1}
        turns = [
            NativeTurn(
                text="",
                tool_calls=[
                    ToolCall(id="call_bad", name="read_file", args=invalid_args)
                ],
                stop_reason="tool_use",
                raw_content=[make_tool_use("call_bad", "read_file", invalid_args)],
            ),
            NativeTurn(
                text="recovered",
                tool_calls=[],
                stop_reason="end_turn",
                raw_content=[{"type": "text", "text": "recovered"}],
            ),
        ]
        agent, client = make_agent(root, turns)
        answer = agent.ask("trigger a rejected call")

        check("rejected native: final answer", answer == "recovered")
        check("rejected native: not counted", agent.current_task_state.tool_steps == 0)
        check(
            "rejected native: is_error result",
            client.calls[1]["messages"][2]["content"][0].get("is_error") is True,
        )

    print("ALL NATIVE LOOP CHECKS PASSED")


if __name__ == "__main__":
    main()
