"""Phase 1 budget/guard verification.

Run with the project venv:
    .venv/Scripts/python.exe scripts/verify_phase1_budget.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pico import Pico, SessionStore, WorkspaceContext
from pico.cli import build_arg_parser


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


def make_agent(root, outputs, progress=None, max_steps=32):
    client = ScriptedClient(outputs)
    agent = Pico(
        model_client=client,
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".pico" / "sessions"),
        approval_policy="auto",
        max_steps=max_steps,
        progress=progress,
    )
    return agent, client


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def make_puzzle_file(root):
    lines = [f"LINE_{number:03d} = {number}" for number in range(1, 121)]
    lines.append('END_OF_PUZZLE_FILE = "marker"')
    body = "\n".join(lines) + "\n"
    body += "# padding to approximate the 3.8K puzzle_game.py\n"
    body += ("#" + "X" * 100 + "\n") * 14
    path = root / "puzzle_game.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_read_once_then_patch():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        puzzle = make_puzzle_file(root)
        progress = []
        agent, client = make_agent(
            root,
            [
                '<tool>{"name":"read_file","args":{"path":"puzzle_game.py","start":1,"end":200}}</tool>',
                '<tool>{"name":"patch_file","args":{"path":"puzzle_game.py","old_text":"LINE_001 = 1","new_text":"LINE_001 = 999"}}</tool>',
                "<final>patched</final>",
            ],
            progress.append,
        )

        answer = agent.ask("read once and patch")

        check("small file: final answer", answer == "patched")
        check("small file: patch applied", "LINE_001 = 999" in puzzle.read_text(encoding="utf-8"))
        check("small file: tool_steps == 2", agent.current_task_state.tool_steps == 2)
        check("small file: full content visible in next prompt", "END_OF_PUZZLE_FILE" in client.prompts[1])
        check("small file: prompt under new budget", len(client.prompts[1]) < 128000)
        check("small file: waiting progress", any("waiting for model" in line for line in progress))
        check("small file: responded progress", any("model responded" in line for line in progress))
        check(
            "small file: no repeated-read rejection",
            not any(
                "repeated identical" in str(item.get("content", ""))
                for item in agent.session["history"]
                if item.get("role") == "tool"
            ),
        )


def test_rejected_calls_do_not_consume_budget():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_puzzle_file(root)
        agent, _ = make_agent(
            root,
            [
                '<tool>{"name":"read_file","args":{"start":1}}</tool>',
                '<tool>{"name":"read_file","args":{"start":2}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"puzzle_game.py","start":1,"end":1}}</tool>',
                "<final>ok</final>",
            ],
        )

        answer = agent.ask("recover from rejected calls")

        check("rejected calls: final answer", answer == "ok")
        check("rejected calls: only executed call counted", agent.current_task_state.tool_steps == 1)


def test_identical_reads_are_allowed():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_puzzle_file(root)
        agent, _ = make_agent(
            root,
            [
                '<tool>{"name":"read_file","args":{"path":"puzzle_game.py","start":1,"end":1}}</tool>',
                '<tool>{"name":"read_file","args":{"path":"puzzle_game.py","start":1,"end":1}}</tool>',
                "<final>ok</final>",
            ],
        )

        answer = agent.ask("read the same range twice")

        check("identical reads: final answer", answer == "ok")
        check("identical reads: both executed", agent.current_task_state.tool_steps == 2)
        check(
            "identical reads: no repeat rejection",
            not any(
                "repeated identical" in str(item.get("content", ""))
                for item in agent.session["history"]
                if item.get("role") == "tool"
            ),
        )


def test_large_read_has_continuation_marker():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        big = root / "big.py"
        big.write_text(
            "\n".join(f"VALUE_{number:04d} = {number:04d} " + "X" * 40 for number in range(1, 2001)) + "\n",
            encoding="utf-8",
        )
        agent, _ = make_agent(root, [])
        result = agent.execute_tool(
            "read_file",
            {"path": "big.py", "start": 1, "end": 2000},
        ).content

        check("large read: output capped", len(result) <= 16000)
        check("large read: continuation marker", "continue with start=" in result)
        check("large read: header has line range", "(lines 1-" in result)


def test_defaults():
    args = build_arg_parser().parse_args([])
    check("defaults: CLI max_steps == 32", args.max_steps == 32)
    check("defaults: CLI max_new_tokens == 16384", args.max_new_tokens == 16384)

    with tempfile.TemporaryDirectory() as td:
        agent, _ = make_agent(Path(td), [])
        check("defaults: Pico max_steps == 32", agent.max_steps == 32)
        check("defaults: Pico max_new_tokens == 16384", agent.max_new_tokens == 16384)


def main():
    test_read_once_then_patch()
    test_rejected_calls_do_not_consume_budget()
    test_identical_reads_are_allowed()
    test_large_read_has_continuation_marker()
    test_defaults()
    print("ALL PHASE 1 CHECKS PASSED")


if __name__ == "__main__":
    main()
