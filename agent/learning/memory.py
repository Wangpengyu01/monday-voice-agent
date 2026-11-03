"""Local long-term memory (SQLite, Chinese full-text search via the FTS5 trigram tokenizer).

Three kinds of memory, like a person keeps:
- knowledge: what the teacher model answered ("澳大利亚的首都是堪培拉"), with an expiry
- facts:     things about the user ("主人五点下课", "走回宿舍要十五分钟")
- lessons:   corrections ("不对，应该是…"): what was wrong and what is right

Runs on the RK3588 CPU; nothing here touches the RK1828.
"""
from __future__ import annotations

import difflib
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_PUNCT = re.compile(r"[\s，,。.！!？?、~～…：:；;\"“”'‘’（）()]+")
_FILLER = re.compile(r"^(?:请问|那个|你知道|你知不知道|告诉我|我想知道|帮我查一下|查一下)+|(?:吗|呢|呀|啊|吧)+$")


# same question, different words: "北京去上海坐高铁多长时间" ~ "从北京到上海的高铁要多久"
_SYNONYMS = [
    (re.compile(r"(?:需要|要|得)?(?:多长时间|多少时间|多久|多长时候|几个小时)"), "多久"),
    (re.compile(r"(?:需要|要|得)?(?:多少钱|什么价|啥价)"), "多少钱"),
    (re.compile(r"是什么|是啥|是哪个|是哪一个|是哪条|是哪座|是哪里|在哪儿|在哪里|在哪"), "是哪"),
    (re.compile(r"(?<=[一-鿿])(?:去|往|至)(?=[一-鿿])"), "到"),
    (re.compile(r"^从"), ""),
    (re.compile(r"坐|乘|搭|的"), ""),
]


def normalize(text: str) -> str:
    t = _FILLER.sub("", _PUNCT.sub("", text or ""))
    for pat, rep in _SYNONYMS:
        t = pat.sub(rep, t)
    return t


def similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _fts_query(text: str) -> str | None:
    t = normalize(text)
    grams = {t[i:i + 3] for i in range(max(0, len(t) - 2))}
    grams = {g.replace('"', "") for g in grams if g.strip()}
    if not grams:
        return None
    return " OR ".join(f'"{g}"' for g in sorted(grams))


@dataclass
class Knowledge:
    id: int
    question: str
    answer: str
    source: str
    expires: float  # 0 = never
    score: float = 0.0  # similarity to the query


SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL, source TEXT,
    created REAL, expires REAL DEFAULT 0, uses INTEGER DEFAULT 0, last_used REAL DEFAULT 0);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    question, answer, content='knowledge', content_rowid='id', tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(rowid, question, answer) VALUES (new.id, new.question, new.answer); END;
CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, question, answer) VALUES ('delete', old.id, old.question, old.answer); END;
CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, question, answer) VALUES ('delete', old.id, old.question, old.answer);
    INSERT INTO knowledge_fts(rowid, question, answer) VALUES (new.id, new.question, new.answer); END;
CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, fact TEXT NOT NULL, created REAL);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY, question TEXT, wrong TEXT, right TEXT, created REAL);
"""


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._db.commit()

    # ---- knowledge
    def search(self, question: str, k: int = 3, now: float | None = None) -> list[Knowledge]:
        q = _fts_query(question)
        if not q:
            return []
        now = now or time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT k.id, k.question, k.answer, k.source, k.expires FROM knowledge_fts f "
                "JOIN knowledge k ON k.id = f.rowid WHERE knowledge_fts MATCH ? ORDER BY bm25(knowledge_fts) LIMIT 20",
                (q,),
            ).fetchall()
            if len(rows) < k and self._db.execute("SELECT count(*) FROM knowledge").fetchone()[0] <= 5000:
                # no shared trigram (paraphrase): a small memory is cheap to scan whole
                seen = {r[0] for r in rows}
                rows += [r for r in self._db.execute(
                    "SELECT id, question, answer, source, expires FROM knowledge") if r[0] not in seen]
        out = [Knowledge(*r) for r in rows if not r[4] or r[4] > now]
        for item in out:
            item.score = similarity(question, item.question)
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:k]

    def best_answer(self, question: str, threshold: float = 0.85) -> Knowledge | None:
        hits = self.search(question, k=1)
        if hits and hits[0].score >= threshold:
            with self._lock:
                self._db.execute("UPDATE knowledge SET uses = uses + 1, last_used = ? WHERE id = ?",
                                 (time.time(), hits[0].id))
                self._db.commit()
            return hits[0]
        return None

    def remember(self, question: str, answer: str, *, days: float | None = None, source: str = "teacher") -> int:
        """Store (or update) one learned answer. days=None: keep forever."""
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            return 0
        expires = time.time() + days * 86400 if days else 0
        same = [h for h in self.search(question, k=3) if h.score >= 0.9]
        with self._lock:
            if same:
                self._db.execute("UPDATE knowledge SET answer = ?, source = ?, expires = ?, created = ? WHERE id = ?",
                                 (answer, source, expires, time.time(), same[0].id))
                rid = same[0].id
            else:
                cur = self._db.execute(
                    "INSERT INTO knowledge(question, answer, source, created, expires) VALUES (?, ?, ?, ?, ?)",
                    (question, answer, source, time.time(), expires))
                rid = int(cur.lastrowid)
            self._db.commit()
        return rid

    # ---- facts about the user
    def add_fact(self, fact: str) -> bool:
        fact = fact.strip()
        if not fact:
            return False
        for existing in self.facts(limit=200):
            if similarity(fact, existing) >= 0.85:
                return False
        with self._lock:
            self._db.execute("INSERT INTO facts(fact, created) VALUES (?, ?)", (fact, time.time()))
            self._db.commit()
        return True

    def facts(self, limit: int = 8) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute(
                "SELECT fact FROM facts ORDER BY created DESC LIMIT ?", (limit,)).fetchall()]

    # ---- lessons (corrections)
    def add_lesson(self, question: str, wrong: str, right: str) -> None:
        with self._lock:
            self._db.execute("INSERT INTO lessons(question, wrong, right, created) VALUES (?, ?, ?, ?)",
                             (question, wrong, right, time.time()))
            self._db.commit()

    def lessons(self, limit: int = 5) -> list[tuple[str, str, str]]:
        with self._lock:
            return list(self._db.execute(
                "SELECT question, wrong, right FROM lessons ORDER BY created DESC LIMIT ?", (limit,)).fetchall())

    # ---- prompt context
    def context_for(self, question: str, *, max_knowledge: int = 3, min_score: float = 0.35) -> str:
        """A short block for the system prompt: related knowledge + what we know about the user."""
        parts: list[str] = []
        facts = self.facts(limit=6)
        if facts:
            parts.append("你记得关于主人的事：" + "；".join(facts) + "。")
        related = [k for k in self.search(question, k=max_knowledge) if k.score >= min_score]
        if related:
            parts.append("你以前学到的知识：" + "；".join(f"{k.question}——{k.answer}" for k in related))
        return "\n".join(parts)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {t: self._db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("knowledge", "facts", "lessons")}
