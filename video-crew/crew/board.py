"""Task board: one JSON file per task, claimed with an exclusive lock file.

This is the whole coordination mechanism. There is no manager process and
no messages between agents. Anyone can read the entire board at any time,
which is what makes the verifier and composer possible without summaries.

Status flow: open -> claimed -> done | failed
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Task:
    id: str
    kind: str                     # describe | verify | compose
    payload: dict
    status: str = "open"
    owner: Optional[str] = None
    result: Optional[dict] = None
    depends_on: list = field(default_factory=list)
    history: list = field(default_factory=list)


class Board:
    def __init__(self, root: str):
        self.root = root
        self.tasks_dir = os.path.join(root, "tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)

    # -- persistence -------------------------------------------------------
    def _path(self, task_id: str) -> str:
        return os.path.join(self.tasks_dir, f"{task_id}.json")

    def _write(self, t: Task) -> None:
        tmp = self._path(t.id) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(t), f, indent=2)
        os.replace(tmp, self._path(t.id))  # atomic on POSIX

    def get(self, task_id: str) -> Task:
        with open(self._path(task_id)) as f:
            return Task(**json.load(f))

    def all(self) -> list[Task]:
        ids = sorted(f[:-5] for f in os.listdir(self.tasks_dir) if f.endswith(".json"))
        return [self.get(i) for i in ids]

    # -- lifecycle ----------------------------------------------------------
    def post(self, task: Task, by: str) -> Task:
        task.history.append({"t": time.time(), "by": by, "event": "posted"})
        self._write(task)
        return task

    def ready(self, kind: Optional[str] = None) -> list[Task]:
        done = {t.id for t in self.all() if t.status == "done"}
        return [t for t in self.all()
                if t.status == "open" and (kind is None or t.kind == kind)
                and all(d in done for d in t.depends_on)]

    def claim(self, task_id: str, by: str) -> Optional[Task]:
        """Exclusive claim via O_EXCL lock creation. Safe across processes."""
        lock = self._path(task_id) + ".lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        os.write(fd, by.encode())
        os.close(fd)
        t = self.get(task_id)
        if t.status != "open":
            return None
        t.status, t.owner = "claimed", by
        t.history.append({"t": time.time(), "by": by, "event": "claimed"})
        self._write(t)
        return t

    def finish(self, task_id: str, by: str, result: dict, ok: bool = True) -> Task:
        t = self.get(task_id)
        t.status = "done" if ok else "failed"
        t.result = result
        t.history.append({"t": time.time(), "by": by, "event": t.status})
        self._write(t)
        return t

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for t in self.all():
            counts[t.status] = counts.get(t.status, 0) + 1
        return counts
