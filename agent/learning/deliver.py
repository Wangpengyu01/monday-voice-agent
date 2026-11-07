"""Put a project's result into 星期一's real self, and take it back out on its own if anything gets worse.

Owner's choice (2026-11-07): "自动上线，带回滚" - no question asked, but she must not be able to break
herself. One delivery:
  1. check   - every file goes under the agent root, none is protected (PROTECTED: the permission gate,
               the sandboxes, the exam and the self-tune guards, the budget, the teacher client, snapshots
               and self-heal, the project runner and this file, config, secrets, systemd units), sizes
               are sane;
  2. before  - the test suite (which tests fail already) and the exam (learning.exam) as they are now;
  3. node    - a snapshot (ops.snapshot, "历史节点") of code, config and units;
  4. apply   - the files are copied in;
  5. after   - the test suite again: no test may fail that passed before;
               the services the change touches are restarted (only once the owner has been quiet for a
               while) and must come back up; the exam must not drop by more than EXAM_SLACK;
  6. keep or roll back - any failure restores the snapshot (code only, units untouched), removes the
               files that were new, restarts the same services and says what went wrong.
Every delivery is logged to run/learning/deliveries.jsonl.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    "learning/agent.py", "learning/sandbox.py", "learning/projbox.py", "learning/projbox_fwd.py",
    "learning/egress_proxy.py", "learning/project.py", "learning/deliver.py", "learning/exam.py",
    "learning/selfprompt.py", "learning/selftune.py", "learning/usage.py", "learning/teacher.py",
    "learning/credentials.py", "learning/grants.py", "ops/", "config/", "systemd/", "run/", "logs/",
    "tests/test_selftune.py", "tests/test_sandbox.py", "tests/test_projects.py", "tests/test_agent.py",
)
MAX_FILES = 40
MAX_BYTES = 512 * 1024
EXAM_SLACK = 2
SERVICES = {"orchestrator/": "r1-orchestrator", "learning/": "r1-orchestrator", "tts/": "r1-orchestrator",
            "llm/": "r1-orchestrator", "audio/": "r1-orchestrator", "mcp_servers/": "r1-orchestrator",
            "ui/": "r1-agent-ui", "devices/": "r1-devices"}
LOG_FILE = "deliveries.jsonl"
_FAIL = re.compile(r"^(FAIL|ERROR): (\S+) \(([\w.]+)\)", re.M)


def check(files: list[dict], root: Path = AGENT_ROOT) -> list[str]:
    """files: [{"src": Path, "dst": "path relative to the agent root"}] -> problems (empty: may go in)."""
    problems = []
    if not files:
        problems.append("没有要上线的文件")
    if len(files) > MAX_FILES:
        problems.append(f"文件太多（{len(files)}，最多 {MAX_FILES}）")
    for f in files:
        dst = str(f.get("dst") or "").strip().lstrip("/")
        p = Path(dst)
        if not dst or ".." in p.parts or p.is_absolute():
            problems.append(f"路径不对：{dst!r}")
            continue
        if any(dst == x or dst.startswith(x) for x in PROTECTED):
            problems.append(f"受保护，不能改：{dst}")
        src = Path(f["src"])
        if not src.is_file():
            problems.append(f"找不到文件：{src.name}")
        elif src.stat().st_size > MAX_BYTES:
            problems.append(f"文件太大：{dst}")
    return problems


def services_for(dsts: list[str]) -> list[str]:
    out = []
    for d in dsts:
        for prefix, unit in SERVICES.items():
            if d.startswith(prefix) and unit not in out:
                out.append(unit)
    return out


def failing_tests(root: Path = AGENT_ROOT, timeout: float = 900) -> set[str] | None:
    """Ids of the tests that fail or error now (None if the suite could not run at all)."""
    try:
        p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=root,
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = p.stdout + p.stderr
    if "Ran " not in out:
        return None
    return {f"{m.group(3)}.{m.group(2)}" for m in _FAIL.finditer(out)}


def restart(units: list[str]) -> str:
    """'' if every unit is active again ~20 s after its restart, else what is down."""
    down = []
    for u in units:
        subprocess.run(["sudo", "-n", "systemctl", "restart", u], capture_output=True, timeout=120)
    time.sleep(20)
    for u in units:
        st = subprocess.run(["systemctl", "is-active", u], capture_output=True, text=True).stdout.strip()
        if st != "active":
            down.append(f"{u}={st}")
    return "、".join(down)


def deliver(files: list[dict], why: str, *, project: str, learning_dir: Path,
            exam_score: Callable[[], int | None], wait_quiet: Callable[[], bool],
            root: Path = AGENT_ROOT, tests: Callable[[], set[str] | None] = failing_tests,
            snapshot: Callable[[str, str], str] | None = None, restore: Callable[[str], None] | None = None,
            restart_units: Callable[[list[str]], str] = restart) -> dict:
    """-> {"ok": bool, "stage": ..., "why": ...}. Never leaves a half-applied change behind."""
    if snapshot is None or restore is None:
        sys.path.insert(0, str(root))
        from ops import snapshot as snap_mod
        snapshot = snapshot or (lambda label, reason: snap_mod.create(label, reason).id)
        restore = restore or (lambda sid: snap_mod.restore(sid, write_units=False))
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "project": project, "why": why[:200],
           "files": [f["dst"] for f in files]}

    def log(**kw) -> dict:
        rec.update(kw)
        with (learning_dir / LOG_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    problems = check(files, root)
    if problems:
        return log(ok=False, stage="check", reason="；".join(problems)[:300])
    units = services_for([f["dst"] for f in files])
    if units and not wait_quiet():
        return log(ok=False, stage="wait", reason="主人一直在说话，要重启服务，这次先不上线")
    before_fail = tests()
    if before_fail is None:
        return log(ok=False, stage="before", reason="测试跑不起来，先不上线")
    before_exam = exam_score()
    sid = snapshot(f"before-project-{project}"[:60], f"上线项目 {project} 的成果：{why[:80]}")
    new_files = [f["dst"] for f in files if not (root / f["dst"]).exists()]
    rec["snapshot"] = sid

    def roll_back(stage: str, reason: str) -> dict:
        restore(sid)
        for d in new_files:
            (root / d).unlink(missing_ok=True)
        down = restart_units(units) if units else ""
        return log(ok=False, stage=stage, reason=reason, rolled_back=True, after_rollback_down=down)

    try:
        for f in files:
            dst = root / f["dst"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.deliver.tmp")
            shutil.copyfile(f["src"], tmp)
            os.replace(tmp, dst)
    except OSError as exc:
        return roll_back("apply", f"复制文件出错：{exc}")
    after_fail = tests()
    if after_fail is None:
        return roll_back("tests", "改完以后测试跑不起来")
    broke = sorted(after_fail - before_fail)
    if broke:
        return roll_back("tests", "这些原来通过的测试失败了：" + "、".join(broke[:5]))
    if units:
        down = restart_units(units)
        if down:
            return roll_back("restart", f"重启后服务没起来：{down}")
    after_exam = exam_score()
    if before_exam is not None and after_exam is not None and after_exam < before_exam - EXAM_SLACK:
        return roll_back("exam", f"考卷从 {before_exam} 降到 {after_exam}")
    return log(ok=True, stage="done", reason=f"测试没有新失败，考卷 {before_exam} → {after_exam}", restarted=units)
