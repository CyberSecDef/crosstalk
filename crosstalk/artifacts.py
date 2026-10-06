"""Atomic checkpoints and append-only run logs. Never persist credentials."""

import hashlib
import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

_LOCK = threading.RLock()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a") as handle:
        handle.write(json.dumps(value, default=str) + "\n")


def new_run(topic: str, root: Path = Path("runs")) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60] or "episode"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"{stamp}_{slug}"
    path.mkdir(parents=True)
    write_json(path / "run.json", {"topic": topic, "created": stamp, "stage": "research"})
    return path


class CallLimitReached(RuntimeError):
    pass


class RunStore:
    """A persisted request limit includes failed attempts and survives resume."""

    def __init__(self, path: Path, max_calls: int = 100, budget=None):
        self.path = path
        self.max_calls = max_calls
        self.budget = budget
        path.mkdir(parents=True, exist_ok=True)

    def reserve(self, provider: str, model: str, task: str):
        with _LOCK:
            ledger = self.path / "requests.jsonl"
            count = len(ledger.read_text().splitlines()) if ledger.exists() else 0
            if count >= self.max_calls:
                raise CallLimitReached(f"Run request limit ({self.max_calls}) reached")
            append_jsonl(
                ledger,
                {
                    "provider": provider,
                    "model": model,
                    "task": task,
                    "time": datetime.now(UTC).isoformat(),
                },
            )

    def cache_path(self, namespace: str, key: str) -> Path:
        return self.path / "cache" / namespace / f"{key}.json"

    def usage(self, **entry):
        append_jsonl(self.path / "usage.jsonl", entry)

    def decision(self, **entry):
        append_jsonl(self.path / "decisions.jsonl", entry)
