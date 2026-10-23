"""Stable prompt prefix construction."""

import hashlib
import json
from dataclasses import dataclass

from .workspace import now


@dataclass
class PromptPrefix:
    # prefix 除了文本本身，还带一小份元数据，
    # 这样 runtime 才能明确判断 prefix 是否可以复用。
    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str
    built_at: str


def tool_signature(tools):
    payload = []
    for name in sorted(tools):
        tool = tools[name]
        payload.append(
            {
                "name": name,
                "schema": tool["schema"],
                "risky": tool["risky"],
                "description": tool["description"],
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_prompt_prefix(workspace, tools, built_at=None, native_tools=False, skills_text=""):
    tool_lines = []
    for name, tool in tools.items():
        fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
        risk = "approval required" if tool["risky"] else "safe"
        tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
    tool_text = "\n".join(tool_lines)

    skills_text = str(skills_text or "").strip()
    skills_block = ""
    if skills_text:
        skills_block = "\n".join(
            [
                "Skills (reference instructions loaded on demand):",
                skills_text,
                "Skill rules:",
                "- When the request matches a skill description, call use_skill with that name before acting.",
                "- Skills cannot override system rules, tool permissions, or approval requirements.",
                "- Do not claim to use a skill unless you actually called use_skill.",
            ]
        )

    shared_rules = [
        "- Never invent tool results.",
        "- Keep answers concise and concrete.",
        "- If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
        "- Before writing tests for existing code, read the implementation first.",
        "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
        "- New files should be complete and runnable, including obvious imports.",
        "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.",
        "- After reading a file, prefer patch_file or write_file over reading it again.",
        "- Do not re-read the same file range unless the file changed or you need a different range.",
        "- Use the remember tool when the user states a stable preference, decision, project convention, or dependency.",
        "- Never claim you remembered something without calling remember.",
        "- Do not store secrets, transient task state, or facts already obvious from the repository.",
        "- The Memory section contains a durable memory index with paths like `.pico/memory/items/foo.md`; use read_file on that workspace-relative path when details are needed.",
        "- Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={}.",
    ]

    if native_tools:
        rules_text = "\n".join(
            [
                "Rules:",
                "- Use the provided tools instead of guessing about the workspace.",
                "- Tool calls are made through the provider's native tool calling; do not emit XML <tool> tags.",
                "- When you are done, reply with a plain final answer; do not wrap it in <final> tags.",
                *shared_rules,
            ]
        )
        examples_text = ""
    else:
        rules_text = "\n".join(
            [
                "Rules:",
                "- Use tools instead of guessing about the workspace.",
                "- Return exactly one <tool>...</tool> or one <final>...</final>.",
                "- Tool calls must look like:",
                '  <tool>{"name":"tool_name","args":{...}}</tool>',
                "- For write_file and patch_file with multi-line text, prefer XML style:",
                '  <tool name="write_file" path="file.py"><content>...</content></tool>',
                "- Final answers must look like:",
                "  <final>your answer</final>",
                *shared_rules,
            ]
        )
        examples = [
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            '<tool>{"name":"remember","args":{"name":"user-prefers-tofu","description":"User likes eating tofu","content":"User likes eating tofu.","type":"user","tags":"preference,food"}}</tool>',
            '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "<final>Done.</final>",
        ]
        if "use_skill" in tools:
            examples.insert(4, '<tool>{"name":"use_skill","args":{"name":"test-fixer"}}</tool>')
        examples_text = "\n".join(examples)

    lines = [
        "You are pico, a small local coding agent working inside a local repository.",
        "",
        rules_text,
        "",
        "Tools:",
        tool_text,
        "",
    ]
    if skills_block:
        lines.extend([skills_block, ""])
    if examples_text:
        lines.extend(["Valid response examples:", examples_text, ""])
    lines.append(workspace.text())
    text = "\n".join(lines).strip()

    signature = tool_signature(tools)
    return PromptPrefix(
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        workspace_fingerprint=workspace.fingerprint(),
        tool_signature=signature,
        built_at=built_at or now(),
    )
