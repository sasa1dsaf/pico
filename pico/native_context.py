"""Token estimation and deterministic compression for native messages."""

from __future__ import annotations

import json

from .workspace import clip

NATIVE_INPUT_BUDGET_TOKENS = 64000
SAFETY_TOKENS = 4000


def estimate_tokens(text):
    text = str(text)
    cjk = 0
    for character in text:
        code = ord(character)
        if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            cjk += 1
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def estimate_messages_tokens(messages):
    return sum(estimate_tokens(json.dumps(message, ensure_ascii=False)) for message in messages)


def _has_tool_result(message):
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _render_summary(messages):
    lines = []
    for message in messages:
        role = str(message.get("role", "message"))
        content = message.get("content", "")
        if isinstance(content, str):
            lines.append(f"{role}: {clip(content, 200)}")
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                lines.append(f"{role}: {clip(block.get('text', ''), 200)}")
            elif block_type == "tool_use":
                lines.append(
                    f"tool_use {block.get('name')}: "
                    f"{clip(json.dumps(block.get('input', {}), ensure_ascii=False), 200)}"
                )
            elif block_type == "tool_result":
                lines.append(f"tool_result: {clip(block.get('content', ''), 200)}")
    return "\n".join(line for line in lines if line)


def compress_native_messages(messages, max_tokens, keep_recent=12):
    before = estimate_messages_tokens(messages)
    if before <= max_tokens or len(messages) <= keep_recent:
        return list(messages), {
            "compressed": False,
            "before_tokens": before,
            "after_tokens": before,
            "summarized_messages": 0,
            "kept_messages": len(messages),
        }

    cut = max(0, len(messages) - keep_recent)
    while cut > 0 and _has_tool_result(messages[cut]):
        cut -= 1
    if cut <= 0:
        return list(messages), {
            "compressed": False,
            "before_tokens": before,
            "after_tokens": before,
            "summarized_messages": 0,
            "kept_messages": len(messages),
        }

    old_messages = messages[:cut]
    kept_messages = list(messages[cut:])
    summary = _render_summary(old_messages)
    if not summary:
        return list(messages), {
            "compressed": False,
            "before_tokens": before,
            "after_tokens": before,
            "summarized_messages": 0,
            "kept_messages": len(messages),
        }

    summary_block = {
        "type": "text",
        "text": f"<history_summary>\n{summary}\n</history_summary>",
    }
    if kept_messages and kept_messages[0].get("role") == "user":
        first = kept_messages[0]
        content = first.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        result = [
            {"role": "user", "content": [summary_block, *content]},
            *kept_messages[1:],
        ]
    else:
        result = [{"role": "user", "content": [summary_block]}, *kept_messages]

    after = estimate_messages_tokens(result)
    return result, {
        "compressed": True,
        "before_tokens": before,
        "after_tokens": after,
        "summarized_messages": len(old_messages),
        "kept_messages": len(kept_messages),
    }
