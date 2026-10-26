"""Prompt versions live in /prompts as plain text files (v1.md, v2.md).

Keeping them as files makes every change visible in a diff, and the eval
records the prompt's hash so results are always tied to the exact text used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import json_schema

#: Copy shipped inside the wheel, so `billmeter` works outside the repository.
PACKAGED_PROMPTS = Path(__file__).resolve().parent / "prompts"


def prompts_dir() -> Path:
    """`prompts/` in the folder you run from, else the copy shipped with the package."""
    local = Path.cwd() / "prompts"
    return local if local.is_dir() else PACKAGED_PROMPTS


def load_prompt(version: str = "v2") -> str:
    directory = prompts_dir()
    path = Path(version) if version.endswith(".md") else directory / f"{version}.md"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in directory.glob("*.md"))) or "none"
        raise FileNotFoundError(f"Prompt '{version}' not found in {directory}. Available: {available}")
    template = path.read_text(encoding="utf-8")
    return template.replace("{{SCHEMA}}", json.dumps(json_schema(), indent=2))


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
