"""Standing permissions ("允许并记住") for agent steps of the `ask` level, in memory.db's `grants` table.

A grant is an exact match on tool + host + command: approving `ssh 196 df -h` once with "remember" lets
exactly that run again without asking - not `df -h /` and not the same command on another machine.
Steps of the `danger` level are never granted (learning.agent asks every time). The owner can see and
revoke grants in the web page's memory panel (ui/server.py).
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS grants (
    id INTEGER PRIMARY KEY, tool TEXT NOT NULL, host TEXT NOT NULL DEFAULT '', cmd TEXT NOT NULL,
    created REAL, last_used REAL, uses INTEGER DEFAULT 0, UNIQUE(tool, host, cmd));
"""


def key_of(step: dict) -> tuple[str, str, str]:
    tool = str(step.get("tool") or "")
    host = str(step.get("host") or "") if tool == "ssh" else ""
    return tool, host, str(step.get("cmd") or "").strip()


class GrantStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5)
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def allowed(self, step: dict) -> bool:
        tool, host, cmd = key_of(step)
        if not cmd:
            return False
        with self._lock:
            row = self._db.execute("SELECT id FROM grants WHERE tool = ? AND host = ? AND cmd = ?", (tool, host, cmd)).fetchone()
            if row:
                self._db.execute("UPDATE grants SET uses = uses + 1, last_used = ? WHERE id = ?", (time.time(), row[0]))
                self._db.commit()
        return bool(row)

    def add(self, step: dict) -> int:
        tool, host, cmd = key_of(step)
        if tool not in ("shell", "ssh") or not cmd:
            return 0
        now = time.time()
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO grants(tool, host, cmd, created, last_used) VALUES (?, ?, ?, ?, ?)",
                             (tool, host, cmd, now, now))
            self._db.commit()
            row = self._db.execute("SELECT id FROM grants WHERE tool = ? AND host = ? AND cmd = ?", (tool, host, cmd)).fetchone()
        return int(row[0]) if row else 0

    def revoke(self, gid: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM grants WHERE id = ?", (gid,))
            self._db.commit()

    def all(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT id, tool, host, cmd, created, last_used, uses FROM grants ORDER BY created DESC").fetchall()
        return [dict(zip(("id", "tool", "host", "cmd", "created", "last_used", "uses"), r)) for r in rows]
