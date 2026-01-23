from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


TZ = ZoneInfo("America/New_York")


def now_run_id() -> str:
    return datetime.now(TZ).strftime("%Y%m%d_%H%M%S")


def repo_root_from_src_file(src_file: Path) -> Path:
    # src/a6_dossier_eval/io.py -> repo root is three parents up
    return src_file.resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: dict) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class TurnRecord:
    turn_index: int
    role: str
    content: str
    node_id: str | None = None
    phase: str | None = None

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "content": self.content,
            "node_id": self.node_id,
            "phase": self.phase,
        }
