"""Verify deterministic native message compression."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pico.native_context import (
    compress_native_messages,
    estimate_messages_tokens,
    estimate_tokens,
)


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def has_summary(message):
    content = message.get("content", [])
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and "<history_summary>" in str(block.get("text", ""))
        for block in content
    )


def main():
    check("estimate: English", estimate_tokens("a" * 400) < 200)
    check("estimate: Chinese", estimate_tokens("中文" * 100) >= 200)

    messages = [{"role": "user", "content": [{"type": "text", "text": "初始目标"}]}]
    for index in range(10):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"call_{index}",
                        "name": "read_file",
                        "input": {"path": f"file_{index}.py"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"call_{index}",
                        "content": f"content {'x' * 400}",
                    }
                ],
            }
        )
    messages.append({"role": "assistant", "content": [{"type": "text", "text": "阶段完成"}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": "当前请求"}]})

    before = estimate_messages_tokens(messages)
    compressed, metadata = compress_native_messages(messages, max_tokens=300, keep_recent=6)
    after = estimate_messages_tokens(compressed)

    check("compression: triggered", metadata["compressed"] is True)
    check("compression: tokens reduced", after < before)
    check("compression: summary present", has_summary(compressed[0]))
    check("compression: current request kept", str(compressed[-1]).find("当前请求") >= 0)
    summary_text = ""
    for block in compressed[0].get("content", []):
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and "<history_summary>" in str(block.get("text", ""))
        ):
            summary_text = str(block.get("text", ""))
            break
    check(
        "compression: summary has tool history",
        "tool_use" in summary_text or "tool_result" in summary_text,
    )

    # Every kept tool_result must have a preceding assistant tool_use.
    paired = True
    for index, message in enumerate(compressed):
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        if not any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        ):
            continue
        if index == 0 or compressed[index - 1].get("role") != "assistant":
            paired = False
            break
    check("compression: tool pairs preserved", paired)

    small = messages[:2]
    unchanged, small_metadata = compress_native_messages(small, max_tokens=100000)
    check("compression: no-op under budget", small_metadata["compressed"] is False)
    check("compression: no-op keeps messages", unchanged == small)

    print("ALL NATIVE COMPRESSION CHECKS PASSED")


if __name__ == "__main__":
    main()
