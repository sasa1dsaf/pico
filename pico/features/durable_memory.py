"""User-level durable memory backed by MEMORY.md plus one file per fact."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

MAX_ITEM_CONTENT_CHARS = 4000
MAX_INDEX_CHARS = 6000
SECRET_SHAPED_TEXT_PATTERN = re.compile(
    r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})"
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _one_line(text):
    return " ".join(str(text).split())


def _slugify(name):
    text = str(name).strip().lower()
    slug = re.sub(r"[^a-z0-9一-鿿]+", "-", text).strip("-")
    if not slug:
        slug = "memory-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return slug[:80]


def _parse_frontmatter(text):
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    metadata = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, parts[2].lstrip("\n")


class MemoryStore:
    """One Markdown file per durable fact, with a compact index."""

    def __init__(self, root):
        self.root = Path(root)
        self.items_dir = self.root / "items"
        self.index_path = self.root / "MEMORY.md"

    def item_path(self, name):
        return self.items_dir / f"{_slugify(name)}.md"

    def remember(self, name, description, content, type="user", tags=()):
        name = str(name).strip()
        description = _one_line(description)
        content = str(content).strip()
        if not name or not description or not content:
            raise ValueError("name, description and content must not be empty")
        if SECRET_SHAPED_TEXT_PATTERN.search(f"{name}\n{description}\n{content}"):
            raise ValueError("memory looks like it contains a secret; not stored")

        item_type = str(type).strip() or "user"
        tag_list = [str(tag).strip() for tag in tags if str(tag).strip()]
        content = content[:MAX_ITEM_CONTENT_CHARS]

        item_path = self.item_path(name)
        created_at = _now()
        if item_path.exists():
            existing, _ = _parse_frontmatter(item_path.read_text(encoding="utf-8"))
            existing_name = existing.get("name") or ""
            if existing_name and existing_name != name:
                # Two different names can slugify to the same filename. Keep
                # both instead of silently overwriting an unrelated memory.
                suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
                item_path = self.items_dir / f"{item_path.stem}-{suffix}.md"
                if item_path.exists():
                    existing, _ = _parse_frontmatter(item_path.read_text(encoding="utf-8"))
            created_at = existing.get("created_at") or created_at

        self.items_dir.mkdir(parents=True, exist_ok=True)
        updated_at = _now()
        payload = (
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"type: {item_type}\n"
            f"tags: {', '.join(tag_list)}\n"
            f"created_at: {created_at}\n"
            f"updated_at: {updated_at}\n"
            "---\n\n"
            f"{content}\n"
        )
        item_path.write_text(payload, encoding="utf-8")
        self._rebuild_index()
        try:
            relative = item_path.relative_to(self.root.parent.parent).as_posix()
        except ValueError:
            relative = item_path.as_posix()
        return f"remembered {name} -> {relative}"

    def read(self, name):
        path = self.item_path(name)
        if not path.exists():
            raise ValueError(f"memory not found: {name}")
        return path.read_text(encoding="utf-8")

    def index_text(self):
        if not self.index_path.exists():
            return "-"
        text = self.index_path.read_text(encoding="utf-8").strip()
        if not text:
            return "-"
        if len(text) > MAX_INDEX_CHARS:
            text = text[:MAX_INDEX_CHARS] + "\n...[truncated; read an item for details]"
        return text

    def _rebuild_index(self):
        items = []
        if self.items_dir.exists():
            for path in sorted(self.items_dir.glob("*.md")):
                metadata, _ = _parse_frontmatter(path.read_text(encoding="utf-8"))
                name = metadata.get("name") or path.stem
                description = metadata.get("description") or ""
                items.append((name, path.name, description))
        lines = ["# Durable Memory Index", ""]
        for name, filename, description in sorted(items, key=lambda item: item[0].lower()):
            lines.append(f"- [{name}](items/{filename}) — {description}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
