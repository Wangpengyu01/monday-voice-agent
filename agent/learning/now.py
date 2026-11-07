"""What 星期一 is studying right now, for the web page (上课 tab): run/learning/now.json.

    with doing(learning_dir, "lesson", "Python 网络请求", detail="第 42 节 · 课本"):
        ...
        step("做第 2/3 题")

The file names the process that wrote it; the page treats it as stale once that process is gone, so a
crash never leaves "正在上课" stuck on screen.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

NAME = "now.json"
_CUR: Path | None = None   # set by begin(): step() from deep inside a lesson needs no directory


def _write(learning_dir: Path, data: dict) -> None:
    try:
        learning_dir.mkdir(parents=True, exist_ok=True)
        tmp = learning_dir / (NAME + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(learning_dir / NAME)
    except OSError:
        pass  # the page just shows nothing; never break a lesson over it


def begin(learning_dir: Path, kind: str, title: str, detail: str = "") -> None:
    global _CUR
    _CUR = learning_dir
    _write(learning_dir, {"kind": kind, "title": title, "detail": detail, "step": "", "pid": os.getpid(),
                          "since": time.time(), "updated": time.time()})


def step(text: str, learning_dir: Path | None = None) -> None:
    learning_dir = learning_dir or _CUR
    if learning_dir is None:
        return
    cur = read(learning_dir, check_pid=False)
    if cur and cur.get("pid") == os.getpid():
        cur.update(step=text, updated=time.time())
        _write(learning_dir, cur)


def end(learning_dir: Path) -> None:
    cur = read(learning_dir, check_pid=False)
    if cur and cur.get("pid") == os.getpid():
        try:
            (learning_dir / NAME).unlink()
        except OSError:
            pass


@contextmanager
def doing(learning_dir: Path, kind: str, title: str, detail: str = ""):
    begin(learning_dir, kind, title, detail)
    try:
        yield
    finally:
        end(learning_dir)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read(learning_dir: Path, *, check_pid: bool = True) -> dict:
    try:
        d = json.loads((learning_dir / NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    if check_pid and not _alive(int(d.get("pid") or 0)):
        return {}
    return d
