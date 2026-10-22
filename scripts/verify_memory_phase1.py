"""Verify the Phase 1 durable memory store and remember tool."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pico import Pico, SessionStore, WorkspaceContext


class ScriptedClient:
    supports_prompt_cache = False

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("scripted model ran out of outputs")
        return self.outputs.pop(0)


def make_agent(root, outputs):
    client = ScriptedClient(outputs)
    agent = Pico(
        model_client=client,
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".pico" / "sessions"),
        approval_policy="auto",
    )
    return agent, client


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        agent, _ = make_agent(
            root,
            [
                '<tool>{"name":"remember","args":{"name":"user-prefers-tofu","description":"用户喜欢吃豆腐","content":"用户喜欢吃豆腐。","type":"user","tags":"preference,food"}}</tool>',
                "<final>已记住。</final>",
            ],
        )

        answer = agent.ask("记住：我喜欢吃豆腐")
        item_path = root / ".pico" / "memory" / "items" / "user-prefers-tofu.md"
        index_path = root / ".pico" / "memory" / "MEMORY.md"

        check("remember: final answer", answer == "已记住。")
        check("remember: item file exists", item_path.exists())
        check("remember: item content", "用户喜欢吃豆腐" in item_path.read_text(encoding="utf-8"))
        check("remember: index exists", index_path.exists())
        check("remember: index entry", "user-prefers-tofu" in index_path.read_text(encoding="utf-8"))

        # A new session in the same workspace must see the index in its prompt.
        agent2, _ = make_agent(root, ["<final>ok</final>"])
        prompt = agent2.prompt("我晚饭吃什么")
        check("recall: index in prompt", "durable_memory_index" in prompt)
        check("recall: item listed in prompt", "user-prefers-tofu" in prompt)
        check(
            "recall: workspace-relative item path",
            ".pico/memory/items/user-prefers-tofu.md" in prompt,
        )
        check("recall: no bare items path", "](items/user-prefers-tofu.md)" not in prompt)
        check("tool: remember in prompt", "- remember(" in prompt)

        read_result = agent2.execute_tool(
            "read_file",
            {"path": ".pico/memory/items/user-prefers-tofu.md"},
        ).content
        check("recall: item readable", "用户喜欢吃豆腐" in read_result)

        # Simulate the model following the workspace-relative path from the index.
        recall_agent, _ = make_agent(
            root,
            [
                '<tool>{"name":"read_file","args":{"path":".pico/memory/items/user-prefers-tofu.md","start":1,"end":80}}</tool>',
                "<final>用户喜欢吃豆腐。</final>",
            ],
        )
        recall_answer = recall_agent.ask("我喜欢啥")
        check("recall: scripted end-to-end", recall_answer == "用户喜欢吃豆腐。")
        check(
            "recall: read_file succeeded",
            any(
                item.get("role") == "tool"
                and item.get("name") == "read_file"
                and "用户喜欢吃豆腐" in str(item.get("content", ""))
                for item in recall_agent.session["history"]
            ),
        )

        # Updating by the same name must not duplicate the index entry.
        original_item = item_path.read_text(encoding="utf-8")
        created_line = next(
            line for line in original_item.splitlines() if line.startswith("created_at:")
        )
        agent3, _ = make_agent(root, [])
        update_result = agent3.execute_tool(
            "remember",
            {
                "name": "user-prefers-tofu",
                "description": "用户喜欢吃豆腐",
                "content": "用户喜欢吃豆腐，尤其是麻婆豆腐。",
                "type": "user",
                "tags": "preference,food",
            },
        ).content
        updated_item = item_path.read_text(encoding="utf-8")
        updated_index = index_path.read_text(encoding="utf-8")

        check("update: tool result", update_result.startswith("remembered user-prefers-tofu"))
        check("update: content changed", "麻婆豆腐" in updated_item)
        check("update: created_at preserved", created_line in updated_item)
        check(
            "update: index has one entry",
            sum(
                1
                for line in updated_index.splitlines()
                if line.startswith("- [user-prefers-tofu]")
            )
            == 1,
        )

        # Secrets must not be written.
        secret_result = agent3.execute_tool(
            "remember",
            {
                "name": "leaked-key",
                "description": "api key",
                "content": "sk-abc123456789",
                "type": "reference",
                "tags": "secret",
            },
        ).content
        check("secret: rejected", "secret" in secret_result)
        check("secret: no item file", not (root / ".pico" / "memory" / "items" / "leaked-key.md").exists())

    print("ALL MEMORY PHASE 1 CHECKS PASSED")


if __name__ == "__main__":
    main()
