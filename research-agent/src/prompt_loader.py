from __future__ import annotations

from pathlib import Path
from typing import Optional


def bundled_prompt_dir() -> Path:
    """Return the repository's bundled prompt directory."""
    return Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str, prompt_dir: Optional[str] = None) -> str:
    """Load *name* from a custom prompt directory or bundled defaults."""
    search_paths = []
    if prompt_dir:
        search_paths.append(Path(prompt_dir) / name)
    search_paths.append(bundled_prompt_dir() / name)

    for path in search_paths:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()

    searched = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(f"Prompt file not found: {name} (searched: {searched})")
