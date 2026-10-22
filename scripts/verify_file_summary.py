"""Verify whole-file structural summaries (Plan A)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pico import Pico, SessionStore, WorkspaceContext
from pico.features.memory import summarize_file


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


PYTHON_SOURCE = '''"""Widget storage helpers.

This module persists widgets.
"""
import os

LIMIT = 3


class Store:
    pass


def load(path):
    return path


async def save():
    return None
'''


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        module = root / "module.py"
        module.write_text(PYTHON_SOURCE, encoding="utf-8")

        summary = summarize_file("module.py", root)
        check("python: docstring", "Widget storage helpers." in summary)
        check("python: class", "class Store" in summary)
        check("python: sync function", "load()" in summary)
        check("python: async function", "save()" in summary)
        check("python: constant", "LIMIT" in summary)
        check("python: line count", "lines" in summary)

        agent = Pico(
            model_client=object(),
            workspace=WorkspaceContext.build(root),
            session_store=SessionStore(root / ".pico" / "sessions"),
            approval_policy="auto",
        )
        # Simulate reading only a middle chunk of the file.
        fake_result = "# module.py (lines 10-12 of 20)\n  10: def load..."
        agent.update_memory_after_tool("read_file", {"path": "module.py"}, fake_result)
        stored = agent.memory.to_dict()["file_summaries"]["module.py"]["summary"]
        check("middle read: whole-file summary", "Widget storage helpers." in stored)
        check("middle read: no range header", "lines 10-12" not in stored)
        notes = [note["text"] for note in agent.memory.to_dict()["episodic_notes"]]
        check("middle read: one note", len(notes) == 1)

        agent.update_memory_after_tool(
            "read_file", {"path": "./module.py", "start": 10, "end": 12}, fake_result
        )
        notes_again = [note["text"] for note in agent.memory.to_dict()["episodic_notes"]]
        check("repeat read: no duplicate note", len(notes_again) == 1)

        old_freshness = agent.memory.to_dict()["file_summaries"]["module.py"]["freshness"]
        module.write_text(
            PYTHON_SOURCE.replace("LIMIT = 3", "LIMIT = 9") + "\nclass Cache:\n    pass\n",
            encoding="utf-8",
        )
        agent.update_memory_after_tool("read_file", {"path": "module.py"}, fake_result)
        entry = agent.memory.to_dict()["file_summaries"]["module.py"]
        check("changed file: freshness updated", entry["freshness"] != old_freshness)
        check("changed file: summary refreshed", "class Cache" in entry["summary"])
        notes_changed = [note["text"] for note in agent.memory.to_dict()["episodic_notes"]]
        check("changed file: new note appended", len(notes_changed) == 2)

        js = root / "app.js"
        js.write_text(
            "/** App entry. */\nexport function startApp() {}\nexport class Router {}\n",
            encoding="utf-8",
        )
        js_summary = summarize_file("app.js", root)
        check("js: doc comment", "App entry." in js_summary)
        check("js: exported function", "startApp" in js_summary)
        check("js: exported class", "Router" in js_summary)

        text_file = root / "facts.txt"
        text_file.write_text("deploy key is red\n", encoding="utf-8")
        check(
            "text: first line", summarize_file("facts.txt", root).startswith("deploy key is red")
        )

        empty = root / "empty.py"
        empty.write_text("", encoding="utf-8")
        check("empty file", summarize_file("empty.py", root) == "(empty file)")

        check("missing file", summarize_file("nope.py", root) == "")

        rendered = agent.memory.render_memory_text()
        check("prompt memory shows symbols", "class Store" in rendered)

    print("ALL FILE SUMMARY CHECKS PASSED")


if __name__ == "__main__":
    main()
