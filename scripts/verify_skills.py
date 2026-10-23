"""Verify Claude-compatible skill discovery and the use_skill tool."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pico import FakeModelClient, Pico, SessionStore, WorkspaceContext
from pico.providers.native import to_json_schema
from pico.skills import SkillRegistry, parse_skill_document


def check(name, condition):
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        raise AssertionError(name)


def write_skill(root, name, description, body, files=None):
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    for relative, content in (files or {}).items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return directory


def make_agent(root, skills_dirs, outputs=(), **kwargs):
    return Pico(
        model_client=FakeModelClient(list(outputs)),
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".pico" / "sessions"),
        approval_policy="auto",
        skills_dirs=skills_dirs,
        **kwargs,
    )


def main():
    sample = (
        chr(0xFEFF)
        + "---\r\nname: demo\r\ndescription: >-\r\n  line one\r\n  line two\r\n---\r\n\r\nBody\r\n"
    )
    metadata, body = parse_skill_document(sample)
    check("frontmatter: BOM + CRLF", metadata.get("name") == "demo")
    check("frontmatter: folded description", metadata.get("description") == "line one line two")
    check("frontmatter: body", body == "Body")

    metadata, body = parse_skill_document("No frontmatter here\n")
    check("frontmatter: absent", metadata == {} and body == "No frontmatter here")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        global_root = root / "global-skills"
        extra_root = root / "extra-skills"
        global_root.mkdir()
        extra_root.mkdir()
        os.environ["PICO_SKILLS_DIR"] = str(global_root)

        write_skill(
            global_root,
            "pdf",
            "Use for PDF reading and manipulation.",
            "# PDF\n\nRead the reference first.",
            files={
                "references/readme.md": "PDF reference content.\n",
                "scripts/run.sh": "#!/bin/sh\necho pdf\n",
                "bin.dat": b"\x00\x01\x02",
            },
        )
        write_skill(
            global_root,
            "test-fixer",
            "Original description.",
            "# Original\n\nOriginal body.",
        )
        write_skill(
            extra_root,
            "test-fixer",
            "Use when a test fails and you need to reproduce and fix it.",
            "# Test Fixer\n\nReproduce the failure first.",
        )
        write_skill(
            extra_root,
            "extra-only",
            "Only available in the extra directory.",
            "# Extra",
        )
        write_skill(
            extra_root,
            "long-skill",
            "Long body test.",
            "Y" * 12000,
        )
        write_skill(
            extra_root,
            "long-desc",
            "word " * 100,
            "# Long description",
        )
        (global_root / "secret.txt").write_text("secret\n", encoding="utf-8")

        registry = SkillRegistry(extra_dirs=[extra_root])
        check("registry: global skill", "pdf" in registry.skills)
        check("registry: extra override", "Original description" not in registry.index_text())
        check("registry: override description", "reproduce and fix" in registry.index_text())
        check("registry: extra-only", "extra-only" in registry.index_text())
        check("registry: index format", "- pdf:" in registry.index_text())

        long_desc_line = registry.get("long-desc").index_line()
        check("index line: single line", "\n" not in long_desc_line)
        check("index line: plain ellipsis", long_desc_line.endswith("..."))
        check("index line: no truncation marker", "truncated" not in long_desc_line)
        check(
            "index line: bounded",
            len(long_desc_line) <= len("- long-desc: ") + 240,
        )
        check(
            "description stored in full",
            len(registry.get("long-desc").description) > 240,
        )
        check("index has no clip markers", "truncated" not in registry.index_text())

        rendered = registry.render("pdf")
        check("skill body loaded", "Read the reference first." in rendered)
        check("skill resources listed", "references/readme.md" in rendered)
        check(
            "skill resource read",
            "PDF reference content." in registry.read_resource("pdf", "references/readme.md"),
        )
        check(
            "unknown skill message",
            "unknown skill" in registry.unknown_skill_message("nope"),
        )

        for relative in ("../secret.txt", "C:/Windows/win.ini", "/etc/passwd"):
            try:
                registry.read_resource("pdf", relative)
            except ValueError:
                pass
            else:
                raise AssertionError(f"path escape allowed: {relative}")
        check("resource path sandbox", True)

        try:
            registry.read_resource("pdf", "bin.dat")
        except ValueError as exc:
            check("binary resource rejected", "binary" in str(exc))
        else:
            raise AssertionError("binary resource was returned")

        agent = make_agent(
            root,
            [extra_root],
            outputs=[
                '<tool>{"name":"use_skill","args":{"name":"test-fixer"}}</tool>',
                "<final>Loaded the skill.</final>",
            ],
        )
        check("tool registered", "use_skill" in agent.tools)
        check("prompt index injected", "Skills (reference instructions" in agent.prefix)
        check("prompt lists extra skill", "test-fixer:" in agent.prefix)
        check("prompt lists global skill", "pdf:" in agent.prefix)

        answer = agent.ask("Fix the failing test")
        check("text loop final", answer == "Loaded the skill.")
        tool_items = [item for item in agent.session["history"] if item["role"] == "tool"]
        check("text loop tool", tool_items and tool_items[0]["name"] == "use_skill")
        check("text loop body", "Reproduce the failure first." in tool_items[0]["content"])

        long_result = agent.run_tool("use_skill", {"name": "long-skill"})
        check("use_skill output not clipped to 4K", len(long_result) > 12000)
        check("use_skill keeps body tail", "Y" * 11000 in long_result)

        agent.validate_tool("use_skill", {"name": "test-fixer"})
        try:
            agent.validate_tool("use_skill", {"name": "missing-skill"})
        except ValueError:
            check("validator: unknown skill", True)
        else:
            raise AssertionError("unknown skill passed validation")
        try:
            agent.validate_tool("use_skill", {"name": "pdf", "file": "../secret.txt"})
        except ValueError:
            check("validator: traversal rejected", True)
        else:
            raise AssertionError("traversal passed validation")

        native_defs = {item["name"]: item for item in to_json_schema(agent.tools)}
        schema = native_defs["use_skill"]["input_schema"]
        check("native schema required name", schema.get("required") == ["name"])
        check("native schema file optional", "file" in schema["properties"])

        late_dir = extra_root / "late-skill"
        late_dir.mkdir()
        (late_dir / "SKILL.md").write_text(
            "---\nname: late-skill\ndescription: Added after startup.\n---\n\nLate body.\n",
            encoding="utf-8",
        )
        refresh = agent.refresh_prefix()
        check("refresh detects skill change", refresh["skills_changed"] is True)
        check("refresh rebuilds prefix", "late-skill:" in agent.prefix)

        restricted = make_agent(root, [extra_root], allowed_tools=["read_file"])
        check("allowlist removes tool", "use_skill" not in restricted.tools)
        check("allowlist hides skills index", "Skills (reference instructions" not in restricted.prefix)

        agent.session["history"] = [
            {
                "role": "tool",
                "name": "use_skill",
                "args": {"name": "test-fixer"},
                "content": "X" * 12000,
            }
        ]
        prompt, _ = agent._build_prompt_and_metadata("continue")
        check("use_skill history not clipped to 4K", "X" * 9000 in prompt)

    print("ALL SKILL CHECKS PASSED")


if __name__ == "__main__":
    main()
