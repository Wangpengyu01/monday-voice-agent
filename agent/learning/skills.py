"""Skill library: how 星期一 did something it had to learn (procedural memory, next to the facts in memory.db).

A skill is a task in plain words ("SSH 到 196 服务器看磁盘") plus the steps that worked, in order,
each a tool call ({"tool": "shell", "cmd": "df -h"}) with a one-line purpose, and short notes on
what to watch out for. Skills are written by learning.agent after a task succeeded (and the teacher
agreed it did), found again by similar wording, and turned into training samples for the LoRA.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3

from learning.memory import _fts_query, similarity

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY, task TEXT NOT NULL, steps TEXT NOT NULL, notes TEXT DEFAULT '',
    created REAL, updated REAL, uses INTEGER DEFAULT 0, successes INTEGER DEFAULT 0, failures INTEGER DEFAULT 0);
CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(task, notes, content='skills', content_rowid='id', tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
    INSERT INTO skills_fts(rowid, task, notes) VALUES (new.id, new.task, new.notes); END;
CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, task, notes) VALUES ('delete', old.id, old.task, old.notes); END;
CREATE TRIGGER IF NOT EXISTS skills_au AFTER UPDATE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, task, notes) VALUES ('delete', old.id, old.task, old.notes);
    INSERT INTO skills_fts(rowid, task, notes) VALUES (new.id, new.task, new.notes); END;
"""


@dataclass
class Skill:
    id: int
    task: str
    steps: list[dict]
    notes: str = ""
    successes: int = 0
    failures: int = 0
    score: float = 0.0

    def describe(self) -> str:
        """Steps as text, for the planner prompt and for training data."""
        def how(s: dict) -> str:
            if s.get("tool") == "ssh":
                return f"ssh {s.get('host', '')} {s.get('cmd', '')}"
            if s.get("tool") == "note":
                return f"写笔记 {s.get('name', '')}"
            return s.get("cmd", "")

        lines = [f"{i + 1}. {s.get('purpose', '')}：{how(s)}".strip() for i, s in enumerate(self.steps)]
        return "\n".join(lines) + (f"\n注意：{self.notes}" if self.notes else "")


@dataclass
class SkillStore:
    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def _rows(self, sql: str, args: tuple = ()) -> list[Skill]:
        out = []
        for r in self._db.execute(sql, args).fetchall():
            try:
                steps = json.loads(r[2])
            except ValueError:
                steps = []
            out.append(Skill(id=r[0], task=r[1], steps=steps if isinstance(steps, list) else [], notes=r[3] or "",
                             successes=r[4] or 0, failures=r[5] or 0))
        return out

    def search(self, task: str, k: int = 3) -> list[Skill]:
        cols = "s.id, s.task, s.steps, s.notes, s.successes, s.failures"
        with self._lock:
            q = _fts_query(task)
            rows = self._rows(f"SELECT {cols} FROM skills_fts f JOIN skills s ON s.id = f.rowid "
                              "WHERE skills_fts MATCH ? LIMIT 20", (q,)) if q else []
            if len(rows) < k:
                seen = {r.id for r in rows}
                rows += [r for r in self._rows(f"SELECT {cols} FROM skills s ORDER BY updated DESC LIMIT 300") if r.id not in seen]
        for r in rows:
            r.score = similarity(task, r.task)
        rows.sort(key=lambda r: (r.score, r.successes - r.failures), reverse=True)
        return rows[:k]

    def save(self, task: str, steps: list[dict], notes: str = "", *, update_id: int = 0) -> int:
        """Store a skill that worked; the same task (update_id, or very similar wording) is updated, not duplicated."""
        task = task.strip()
        if not task or not steps:
            return 0
        same = [s for s in self.search(task, k=1) if s.score >= 0.9]
        if update_id:
            same = [s for s in self.all(1000) if s.id == update_id] or same
        now = time.time()
        blob = json.dumps(steps, ensure_ascii=False)
        with self._lock:
            if same:
                self._db.execute("UPDATE skills SET steps = ?, notes = ?, updated = ?, successes = successes + 1, uses = uses + 1 "
                                 "WHERE id = ?", (blob, notes or same[0].notes, now, same[0].id))
                rid = same[0].id
            else:
                cur = self._db.execute("INSERT INTO skills(task, steps, notes, created, updated, successes) VALUES (?, ?, ?, ?, ?, 1)",
                                       (task, blob, notes, now, now))
                rid = int(cur.lastrowid)
            self._db.commit()
        return rid

    def record(self, skill_id: int, ok: bool) -> None:
        with self._lock:
            col = "successes" if ok else "failures"
            self._db.execute(f"UPDATE skills SET uses = uses + 1, {col} = {col} + 1, updated = ? WHERE id = ?",
                             (time.time(), skill_id))
            self._db.commit()

    def all(self, limit: int = 200) -> list[Skill]:
        with self._lock:
            return self._rows("SELECT id, task, steps, notes, successes, failures FROM skills ORDER BY updated DESC LIMIT ?", (limit,))
