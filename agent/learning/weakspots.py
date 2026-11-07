"""Weak spots found by real tests, kept until a retest of the same task passes.

The school's daily self-review guesses gaps from conversations, and a gap lesson counts as mastered
once she answers its quiz. That said "no weak spots" on 2026-11-07 while the v5 eval had just caught
her calling running services dead. This list holds tasks she actually got wrong when tried for real:
  - practice drills (learning.drill agent_drill / coding_drill): a failed task whose teacher run is
    right (so the task itself is fine) goes in;
  - tools/lora/board_eval.py: failed agent and coding items go in;
  - by hand: python3 -m learning.weakspots add --mode agent --task "..." --why "...".
The drills retest due items first (the same task, judged the same way, at most once a day each). An item
leaves the list only when that retest passes; one the teacher can't do right either is dropped as a
bad task. The page shows the list under 排队要补的短板 (ui/server.py /api/school), and the school's
self-review sees it as recent troubles.

    cd /userdata/agent && python3 -m learning.weakspots list
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

FILE = "weak_spots.json"      # in run/learning
MAX_OPEN = 30
KEEP_CLOSED = 20
MODES = ("agent", "coding")
SOURCE_NAMES = {"drill": "练习", "eval": "评测", "manual": "人工"}


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def path(learning_dir: Path) -> Path:
    return Path(learning_dir) / FILE


def load(learning_dir: Path) -> dict:
    try:
        d = json.loads(path(learning_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    return {"open": list(d.get("open") or []), "closed": list(d.get("closed") or [])}


def save(learning_dir: Path, d: dict) -> None:
    p = path(learning_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"open": d["open"][:MAX_OPEN], "closed": d["closed"][-KEEP_CLOSED:]},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _find(items: list[dict], mode: str, task: str) -> dict | None:
    return next((x for x in items if x.get("mode") == mode and x.get("task") == task), None)


def add(learning_dir: Path, mode: str, task: str, why: str, source: str = "drill", *, tests: str = "",
        solution: str = "", day: str | None = None) -> bool:
    """Put a failed task on the list (False if it is already there or the input is empty)."""
    task = (task or "").strip()
    if mode not in MODES or not task:
        return False
    d = load(learning_dir)
    if _find(d["open"], mode, task):
        return False
    item = {"mode": mode, "task": task, "why": (why or "").strip()[:200], "source": source,
            "found": day or _today(), "tries": 0, "last_try": ""}
    if tests:
        item["tests"], item["solution"] = tests, solution
    d["open"].append(item)
    d["open"] = d["open"][-MAX_OPEN:]
    save(learning_dir, d)
    return True


def due(learning_dir: Path, mode: str, n: int, day: str | None = None) -> list[dict]:
    """Up to n open items of this mode not retested today, oldest first."""
    day = day or _today()
    return [x for x in load(learning_dir)["open"] if x.get("mode") == mode and x.get("last_try") != day][:max(0, n)]


def record(learning_dir: Path, mode: str, task: str, passed: bool, why: str = "", *, dropped: bool = False,
           day: str | None = None) -> None:
    """A retest result: passed (or dropped as a bad task) closes the item, otherwise it stays with the new reason."""
    day = day or _today()
    d = load(learning_dir)
    item = _find(d["open"], mode, task)
    if item is None:
        return
    item["tries"] = int(item.get("tries", 0)) + 1
    item["last_try"] = day
    if passed or dropped:
        d["open"].remove(item)
        d["closed"].append({**item, "closed": day, "result": "fixed" if passed else "dropped"})
    elif why:
        item["why"] = why.strip()[:200]
    save(learning_dir, d)


def summary(learning_dir: Path, n: int = 8) -> str:
    """Open items as lines for the school's self-review ("最近做砸的")."""
    label = {"agent": "办事", "coding": "写代码"}
    return "\n".join(f"- 真考没过（{label.get(x['mode'], x['mode'])}）：{x['task'][:50]}，{x.get('why', '')[:60]}"
                     for x in load(learning_dir)["open"][:n])


def main() -> None:
    ap = argparse.ArgumentParser(description="星期一's weak spots found by real tests")
    ap.add_argument("--dir", default="/userdata/agent/run/learning")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("--mode", choices=MODES, required=True)
    a.add_argument("--task", required=True)
    a.add_argument("--why", default="")
    a.add_argument("--source", default="manual", choices=sorted(SOURCE_NAMES))
    r = sub.add_parser("remove")
    r.add_argument("--mode", choices=MODES, required=True)
    r.add_argument("--task", required=True)
    args = ap.parse_args()
    learning_dir = Path(args.dir)
    if args.cmd == "add":
        print("added" if add(learning_dir, args.mode, args.task, args.why, args.source) else "already listed")
    elif args.cmd == "remove":
        record(learning_dir, args.mode, args.task, False, dropped=True)
    d = load(learning_dir)
    for x in d["open"]:
        print(f"[{SOURCE_NAMES.get(x.get('source'), x.get('source'))}·{x['mode']}] {x['task']} | {x.get('why', '')} | "
              f"found {x.get('found')} tries {x.get('tries', 0)}")
    print(f"{len(d['open'])} open, {len(d['closed'])} closed")


if __name__ == "__main__":
    main()
