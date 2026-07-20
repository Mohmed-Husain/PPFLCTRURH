"""
Shared utility functions.

Kept verbatim from Phase 1 & Phase 2 notebooks:
  - save_json / load_json
  - timestamp
  - generate_chunk_id
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def save_json(data: Any, filepath: str | Path, silent: bool = False) -> None:
    """Save *data* as a formatted JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if not silent:
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ Saved: {filepath}  ({size_kb:.1f} KB)")


def load_json(filepath: str | Path) -> Any:
    """Load a JSON file and return the parsed data."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


def generate_chunk_id(*args: Any) -> str:
    """Generate a deterministic short ID from input arguments."""
    content = "|".join(str(a) for a in args)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
