"""Append-only JSONL files so that interrupted Colab sessions can resume."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially written last line from a killed process: ignore it.
                continue
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def done_ids(path: Path, key: str = "id") -> set:
    return {r[key] for r in read_jsonl(path) if key in r}
