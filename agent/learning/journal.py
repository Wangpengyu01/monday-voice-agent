"""Every conversation turn, one JSON line per turn in run/learning/turns/YYYY-MM-DD.jsonl.

The nightly reflection (learning/reflect.py) reads a day back to have the teacher grade the answers
the local model gave on its own. Kept small: question, answer, who answered, why.
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


def day_file(journal_dir: Path, day: datetime.date) -> Path:
    return Path(journal_dir) / f"{day.isoformat()}.jsonl"


def log_turn(journal_dir: Path, question: str, answer: str, *, who: str, reason: str = "") -> None:
    """who: local / teacher / memory / quick. Never raises (a full disk must not break a conversation)."""
    question, answer = (question or "").strip(), (answer or "").strip()
    if not question or not answer:
        return
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "who": who, "reason": reason, "q": question, "a": answer}
    try:
        path = day_file(journal_dir, datetime.date.today())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as exc:
        LOG.warning("[journal] write failed: %s", exc)


def read_day(journal_dir: Path, day: datetime.date) -> list[dict]:
    path = day_file(journal_dir, day)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("q") and rec.get("a"):
            out.append(rec)
    return out
