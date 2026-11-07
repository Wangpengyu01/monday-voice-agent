#!/usr/bin/env python3
"""星期一 HUD web server (stdlib only): http://<board IPv4>/ on port 80.

Serves ui/index.html plus a small JSON API the page reads:
  GET  /api/system                  CPU / memory / temperature / disk / uptime / IPv4 / services
  GET  /api/memory?q=&limit=        learned knowledge, facts about the user, corrections
  POST /api/memory/delete           {"table": "knowledge|facts|lessons|skills|grants", "id": n}  (grants = 撤销授权)
  GET  /api/files?root=&path=       directory listing inside one of FILE_ROOTS
  GET  /api/file?root=&path=        file content (inline; &download=1 to save)
  POST /api/files/upload?path=      raw body, X-Filename header (writable roots only)
  POST /api/files/delete            {"root", "path"} (writable roots only)
  GET  /api/logs?name=&lines=       tail of an agent log
  GET  /api/school                  the school: syllabus progress, self-review, lessons, class notes
  GET  /api/progress                LoRA training/export status + work reports (run/learning/progress)
  GET  /api/owner                   主人 tab: profile, timetable, her questions, files she read, daily reports
  POST /api/owner/answer            {id, answer}: the owner answers one of her questions (picked up by learning.inbox)
  POST /api/study/start             {what: lesson | practice}: 马上上一节课 / 马上练一轮 (same run as the timers)
  GET  /ws                         proxied to the assistant's WebSocket (:8765): state, transcript, chat, settings

Only FILE_ROOTS are browsable: secrets (/userdata/secrets) and the device vault (run/devices)
are never reachable, and every path is resolved and checked against its root.
"""
from __future__ import annotations

import http.server
import json
import logging
import mimetypes
import os
import shutil
import socket
import socketserver
import sqlite3
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

LOG = logging.getLogger("ui_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

UI_DIR = Path(__file__).parent.resolve()
AGENT_ROOT = Path(os.environ.get("AGENT_ROOT", str(UI_DIR.parent)))
MEMORY_DB = Path(os.environ.get("MEMORY_DB", str(AGENT_ROOT / "run" / "learning" / "memory.db")))

# id -> (label, directory, writable)
FILE_ROOTS: dict[str, tuple[str, Path, bool]] = {
    "files": ("我的文件", Path(os.environ.get("UI_FILES_DIR", "/userdata/files")), True),
    "learning": ("学习数据", AGENT_ROOT / "run" / "learning", False),
    "logs": ("日志", AGENT_ROOT / "logs", False),
}
LOGS = {
    "orchestrator": "orchestrator.log",
    "tts_daemon": "tts_daemon.log",
    "llm_daemon": "llm_daemon.log",
    "selfheal": "selfheal.log",
    "backup": "backup.log",
}
SNAP_DIR = Path(os.environ.get("SNAPSHOT_DIR", "/userdata/backups/snapshots"))
SERVICES = [
    ("r1-orchestrator", "核心"),
    ("r1-llm-daemon", "大模型"),
    ("r1-tts-daemon", "语音合成"),
    ("r1-devices", "设备"),
    ("r1-agent-ui", "界面"),
]
WS_UPSTREAM = ("127.0.0.1", int(os.environ.get("AGENT_WS_PORT", 8765)))
MAX_VIEW = 2 * 1024 * 1024  # bytes shown inline as text
MAX_UPLOAD = 200 * 1024 * 1024


# ---------------------------------------------------------------- system
def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


_cpu_prev: tuple[int, int] | None = None


def _cpu_percent() -> float:
    global _cpu_prev
    try:
        vals = [int(x) for x in _read("/proc/stat").splitlines()[0].split()[1:]]
    except (IndexError, ValueError):
        return 0.0
    idle, total = vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)
    prev, _cpu_prev = _cpu_prev, (idle, total)
    if not prev or total == prev[1]:
        return 0.0
    return round(100.0 * (1 - (idle - prev[0]) / (total - prev[1])), 1)


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in _read("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        try:
            out[k] = int(v.split()[0]) * 1024
        except (IndexError, ValueError):
            pass
    return out


def _temps() -> float | None:
    best = None
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            t = int(z.read_text().strip()) / 1000
        except (OSError, ValueError):
            continue
        best = t if best is None else max(best, t)
    return round(best, 1) if best is not None else None


def _ipv4() -> list[str]:
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"],
                             capture_output=True, text=True, timeout=2).stdout
        return [ln.split()[3].split("/")[0] for ln in out.splitlines() if len(ln.split()) > 3]
    except (OSError, subprocess.SubprocessError):
        return []


def _services() -> list[dict[str, str]]:
    names = [s for s, _ in SERVICES]
    try:
        states = subprocess.run(["systemctl", "is-active", *names], capture_output=True, text=True,
                                timeout=3).stdout.split()
    except (OSError, subprocess.SubprocessError):
        states = []
    states += ["unknown"] * (len(names) - len(states))
    return [{"id": n, "label": label, "state": st} for (n, label), st in zip(SERVICES, states)]


def system_info() -> dict[str, Any]:
    mem = _meminfo()
    du = shutil.disk_usage("/userdata" if Path("/userdata").is_dir() else "/")
    load = _read("/proc/loadavg").split()
    return {
        "time": time.time(),
        "hostname": socket.gethostname(),
        "ipv4": _ipv4(),
        "uptime_s": float((_read("/proc/uptime").split() or ["0"])[0]),
        "cpu_percent": _cpu_percent(),
        "cpu_count": os.cpu_count() or 1,
        "load": [float(x) for x in load[:3]] if len(load) >= 3 else [],
        "mem_total": mem.get("MemTotal", 0),
        "mem_available": mem.get("MemAvailable", 0),
        "temp_c": _temps(),
        "disk_total": du.total,
        "disk_free": du.free,
        "services": _services(),
    }


# ---------------------------------------------------------------- memory
def _db(readonly: bool = True) -> sqlite3.Connection:
    if not MEMORY_DB.is_file():
        raise FileNotFoundError("还没有记忆数据")
    uri = f"file:{MEMORY_DB}?mode={'ro' if readonly else 'rw'}"
    con = sqlite3.connect(uri, uri=True, timeout=3)
    con.row_factory = sqlite3.Row
    return con


def memory_info(q: str, limit: int) -> dict[str, Any]:
    con = _db()
    try:
        like = f"%{q}%"
        where = " WHERE question LIKE ? OR answer LIKE ?" if q else ""
        args: tuple = (like, like) if q else ()
        knowledge = [dict(r) for r in con.execute(
            "SELECT id, question, answer, source, created, expires, uses FROM knowledge" + where +
            " ORDER BY created DESC LIMIT ?", (*args, limit))]
        facts = [dict(r) for r in con.execute(
            "SELECT id, fact, created FROM facts" + (" WHERE fact LIKE ?" if q else "") +
            " ORDER BY created DESC LIMIT ?", ((like,) if q else ()) + (limit,))]
        lessons = [dict(r) for r in con.execute(
            "SELECT id, question, wrong, right, created FROM lessons" +
            (" WHERE question LIKE ? OR right LIKE ?" if q else "") + " ORDER BY created DESC LIMIT ?",
            args + (limit,))]
        stats = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("knowledge", "facts", "lessons")}
        skills = []
        if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='skills'").fetchone():
            skills = [dict(r) for r in con.execute(
                "SELECT id, task, steps, notes, successes, failures, updated AS created FROM skills" +
                (" WHERE task LIKE ? OR notes LIKE ?" if q else "") + " ORDER BY updated DESC LIMIT ?", args + (limit,))]
            for sk in skills:
                try:
                    sk["steps"] = json.loads(sk["steps"])
                except ValueError:
                    sk["steps"] = []
            stats["skills"] = con.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        grants = []
        if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grants'").fetchone():
            grants = [dict(r) for r in con.execute(
                "SELECT id, tool, host, cmd, created, last_used, uses FROM grants ORDER BY created DESC LIMIT ?", (limit,))]
    finally:
        con.close()
    return {"stats": stats, "knowledge": knowledge, "facts": facts, "lessons": lessons, "skills": skills,
            "grants": grants, "snapshots": snapshots_info()}


def snapshots_info(limit: int = 30) -> list[dict[str, Any]]:
    """The newest "历史节点" (ops/snapshot.py metadata files), read-only."""
    out = []
    try:
        metas = sorted(SNAP_DIR.glob("*.json"), reverse=True)[:limit]
    except OSError:
        return out
    for m in metas:
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
            out.append({k: d.get(k) for k in ("id", "created", "label", "reason", "good", "size")})
        except (OSError, ValueError):
            continue
    return out


def memory_delete(table: str, rid: int) -> None:
    if table not in ("knowledge", "facts", "lessons", "skills", "grants"):
        raise ValueError("bad table")
    con = _db(readonly=False)
    try:
        con.execute(f"DELETE FROM {table} WHERE id = ?", (rid,))  # FTS is kept in sync by trigger
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------- files
def resolve(root_id: str, rel: str) -> tuple[Path, Path, bool]:
    if root_id not in FILE_ROOTS:
        raise PermissionError("unknown root")
    _, root, writable = FILE_ROOTS[root_id]
    root = root.resolve()
    target = (root / rel.lstrip("/")).resolve()
    if target != root and root not in target.parents:
        raise PermissionError("outside root")
    return root, target, writable


def list_dir(root_id: str, rel: str) -> dict[str, Any]:
    root, d, writable = resolve(root_id, rel)
    if root_id == "files":
        root.mkdir(parents=True, exist_ok=True)
    if not d.is_dir():
        raise FileNotFoundError(rel)
    entries = []
    for p in sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        try:
            st = p.stat()
        except OSError:
            continue
        entries.append({"name": p.name, "dir": p.is_dir(), "size": st.st_size, "mtime": st.st_mtime,
                        "type": mimetypes.guess_type(p.name)[0] or ""})
    rel_path = "" if d == root else d.relative_to(root).as_posix()
    return {"root": root_id, "path": rel_path, "writable": writable, "entries": entries}


def school_info() -> dict[str, Any]:
    """The school (learning/school.py): syllabus progress, today's self-review, queued lessons, history, notes."""
    sdir = AGENT_ROOT / "run" / "learning" / "school"
    try:
        state = json.loads((sdir / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    try:
        import sys as _sys
        if str(AGENT_ROOT) not in _sys.path:
            _sys.path.insert(0, str(AGENT_ROOT))
        from learning.school import SYLLABUS
    except Exception:  # noqa: BLE001 - the page still works without the syllabus names
        SYLLABUS = []
    notes = sorted(sdir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True) if sdir.is_dir() else []
    history = state.get("history", [])
    try:   # weak spots found by real tests (learning/weakspots.py), retested by the practice drills
        ws = json.loads((sdir.parent / "weak_spots.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ws = {}
    try:   # the rules she wrote for herself (learning/selfprompt.py) and the self-tune rounds (learning/selftune.py)
        rules = json.loads((sdir.parent / "self_prompt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rules = {}
    tune_log = []
    tune_file = sdir.parent / "selftune.jsonl"
    for line in (tune_file.read_text(encoding="utf-8").splitlines()[-12:] if tune_file.is_file() else []):
        try:
            tune_log.append(json.loads(line))
        except ValueError:
            continue
    return {
        "syllabus": SYLLABUS,
        "index": int(state.get("index", 0)),
        "sessions": int(state.get("sessions", 0)),
        "day": state.get("day", ""),
        "review": state.get("review", {}),
        "pending": state.get("pending", []),
        "weak_spots": {"open": ws.get("open") or [], "closed": list(reversed((ws.get("closed") or [])[-6:]))},
        "self_rules": {"version": rules.get("version", 0), "updated": rules.get("updated", ""),
                       "agent": rules.get("agent") or [], "voice": rules.get("voice") or []},
        "self_tune": list(reversed(tune_log)),
        "history": list(reversed(history[-80:])),
        "mastered": sum(1 for h in history if h.get("mastered")),
        "notes": [{"name": p.name, "mtime": p.stat().st_mtime, "size": p.stat().st_size} for p in notes[:120]],
        "now": study_now(),
    }


def _systemctl(*args: str) -> str:
    try:
        return subprocess.run(["systemctl", *args], capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _project_mod():
    import sys as _sys
    if str(AGENT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(AGENT_ROOT))
    from learning import project
    return project


def projects_info() -> dict[str, Any]:
    """Her projects (learning/project.py): the list, and the latest one's recent steps and notes."""
    project = _project_mod()
    items = project.list_projects()[:12]
    out: dict[str, Any] = {"projects": items, "steps": [], "notes": ""}
    if items:
        pdir = project.projbox.project_dir(items[0]["slug"])
        out["steps"] = [{k: h.get(k) for k in ("n", "time", "tool", "brief", "status", "thought")}
                        for h in project.steps(pdir)[-15:]][::-1]
        notes = pdir / "notes.md"
        out["notes"] = notes.read_text(encoding="utf-8")[-1500:] if notes.is_file() else ""
    return out


def start_project(goal: str) -> dict[str, Any]:
    """The owner typed a goal on the page: same path as saying "开个项目：…"."""
    goal = " ".join((goal or "").split())[:300]
    if len(goal) < 4:
        return {"ok": False, "error": "目标太短了"}
    project = _project_mod()
    say = project.project_command(f"开个项目：{goal}")
    reply = say() if say else "没听懂这个目标"
    return {"ok": reply.startswith("好，"), "say": reply}


_MANUAL = {"lesson": "learning.school", "practice": "learning.practice"}


def start_study(what: str) -> dict[str, Any]:
    """The owner pressed 马上上一节课 / 马上练一轮: run it now, the same way the timers do (service user,
    teacher key from its env file, the shared study lock), with --force because the owner asked for it.
    Fixed command lines only - nothing from the request reaches the shell."""
    module = _MANUAL.get(what)
    if module is None:
        return {"ok": False, "error": "不认识的操作"}
    from learning import usage as _usage
    if _usage.is_peak():   # the owner's rule: no teacher calls in DeepSeek's peak hours
        return {"ok": False, "error": "现在是高峰时段，不找老师，12 点 / 18 点以后再学"}
    if study_now().get("doing"):
        return {"ok": False, "error": "她正在学，等这一节结束再来"}
    if _systemctl("list-units", "--no-legend", "r1-manual-*") .strip():
        return {"ok": False, "error": "刚才那一节还在排队"}
    log = "school" if what == "lesson" else "practice"
    cmd = ["sudo", "-n", "systemd-run", "--no-block", "--quiet", f"--unit=r1-manual-{what}-{int(time.time())}",
           "--uid=youyeetoo", "--gid=youyeetoo", "-p", "EnvironmentFile=-/userdata/secrets/deepseek.env",
           "-p", f"WorkingDirectory={AGENT_ROOT}", "-p", f"StandardOutput=append:{AGENT_ROOT}/logs/{log}.log",
           "-p", f"StandardError=append:{AGENT_ROOT}/logs/{log}.log",
           "/usr/bin/flock", "-w", "900", "/tmp/monday-study.lock", "/usr/bin/python3", "-m", module, "--force"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc)}
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or "启动失败").strip()[-200:]}
    LOG.info("[study] manual %s started", what)
    return {"ok": True}


def study_now() -> dict[str, Any]:
    """What she is studying this minute (learning/now.py), whether a lesson is queued behind practice,
    and when the next lesson / practice round starts."""
    try:
        import sys as _sys
        if str(AGENT_ROOT) not in _sys.path:
            _sys.path.insert(0, str(AGENT_ROOT))
        from learning import now as now_mod
        cur = now_mod.read(AGENT_ROOT / "run" / "learning")
    except Exception:  # noqa: BLE001
        cur = {}
    out: dict[str, Any] = {"doing": cur}
    if cur:
        out["elapsed_s"] = int(time.time() - float(cur.get("since") or time.time()))
    out["school_waiting"] = (not cur or cur.get("kind") != "lesson") and _systemctl("is-active", "r1-school.service") in ("active", "activating")
    out["loop_on"] = _systemctl("is-active", "r1-study.service") == "active"
    out["loop"] = _jload(AGENT_ROOT / "run" / "learning" / "study_loop.json", {})
    if not out["loop_on"]:
        for name in ("school", "practice"):
            out[f"next_{name}"] = _systemctl("show", f"r1-{name}.timer", "-p", "NextElapseUSecRealtime", "--value")
    return out


def spend_info() -> dict[str, Any]:
    """Teacher (DeepSeek) spend from learning/usage.py: today, yesterday, this month."""
    try:
        import datetime as _dt
        import sys as _sys
        if str(AGENT_ROOT) not in _sys.path:
            _sys.path.insert(0, str(AGENT_ROOT))
        from learning import usage as _usage
        udir = AGENT_ROOT / "run" / "learning" / "usage"
        today = _dt.date.today()
        month = sum(_usage.summary(f.stem, udir)["total"]["yuan"] for f in udir.glob(today.strftime("%Y-%m-") + "*.jsonl"))
        return {"today": _usage.summary(today.isoformat(), udir),
                "yesterday": _usage.summary((today - _dt.timedelta(days=1)).isoformat(), udir)["total"],
                "month_yuan": round(month, 2), "peak_now": _usage.is_peak()}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def progress_info() -> dict[str, Any]:
    """Self-improvement progress (run/learning/progress): LoRA training/export status pushed from the GPU
    server (lora.json, written by tools/lora/sync_progress.py) and the work reports (reports/*.md)."""
    pdir = AGENT_ROOT / "run" / "learning" / "progress"
    try:
        lora = json.loads((pdir / "lora.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        lora = {}
    rdir = pdir / "reports"
    reports = sorted(rdir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True) if rdir.is_dir() else []
    return {"lora": lora, "spend": spend_info(), "distill": _jload(AGENT_ROOT / "run" / "learning" / "distill" / "state.json", {}),
            "reports": [{"name": p.name, "path": "progress/reports/" + p.name, "mtime": p.stat().st_mtime} for p in reports[:60]]}


OWNER_DIR = AGENT_ROOT / "run" / "learning" / "owner"
INBOX_DIR_NAME = "给星期一"


def _jload(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def owner_info() -> dict[str, Any]:
    """What 星期一 knows about the owner (learning/owner.py, learning/inbox.py)."""
    days = sorted((OWNER_DIR / "days").glob("*.md"), reverse=True) if (OWNER_DIR / "days").is_dir() else []
    inbox = FILE_ROOTS["files"][1] / INBOX_DIR_NAME
    waiting = []
    if inbox.is_dir():
        seen = _jload(OWNER_DIR / "inbox_seen.json", {})
        waiting = [f.name for f in inbox.iterdir() if f.is_file() and f.name not in seen and not f.name.endswith(".part")]
    return {"profile": _jload(OWNER_DIR / "profile.json", {}), "schedule": _jload(OWNER_DIR / "schedule.json", []),
            "questions": list(reversed(_jload(OWNER_DIR / "questions.json", [])))[:20],
            "files": _jload(OWNER_DIR / "inbox_seen.json", {}), "waiting": waiting, "inbox": INBOX_DIR_NAME,
            "stats": _jload(OWNER_DIR / "stats.json", {}),
            "days": [{"name": p.name, "path": "owner/days/" + p.name} for p in days[:30]]}


def owner_answer(qid: str, answer: str) -> None:
    answer = answer.strip()[:500]
    if not qid or not answer:
        raise ValueError("回答为空")
    if qid not in {q.get("id") for q in _jload(OWNER_DIR / "questions.json", [])}:
        raise ValueError("没有这个问题")
    OWNER_DIR.mkdir(parents=True, exist_ok=True)
    with (OWNER_DIR / "answers.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": qid, "answer": answer, "time": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False) + "\n")
    qs = _jload(OWNER_DIR / "questions.json", [])   # show it as answered right away; the learner stores it as a fact
    for q in qs:
        if q.get("id") == qid:
            q["answer"] = answer
    (OWNER_DIR / "questions.json").write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")


def roots_info() -> list[dict[str, Any]]:
    return [{"id": k, "label": v[0], "writable": v[2]} for k, v in FILE_ROOTS.items()]


def tail(path: Path, lines: int) -> str:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 256 * 1024))
        data = f.read().decode("utf-8", "replace")
    return "\n".join(data.splitlines()[-lines:])


# ---------------------------------------------------------------- http
class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if 0 < n < 65536 else {}

    def _guard(self, fn) -> None:
        try:
            fn()
        except PermissionError as exc:
            self._json({"error": f"不允许访问：{exc}"}, 403)
        except FileNotFoundError as exc:
            self._json({"error": f"找不到：{exc}"}, 404)
        except (ValueError, sqlite3.Error) as exc:
            self._json({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            LOG.exception("api error")
            self._json({"error": str(exc)}, 500)

    def do_GET(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        qs = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            self._proxy_ws()
            return
        if not u.path.startswith("/api/"):
            if u.path in ("/", "/index.html"):
                self.path = "/index.html"
            elif u.path in ("/showcase", "/showcase/"):
                self.path = "/showcase.html"   # read-only results page for visitors
            elif u.path.lstrip("/") not in {"index.html", "showcase.html", "favicon.svg"}:
                self.send_error(404)  # only the page itself is static; everything else is API
                return
            return super().do_GET()
        routes = {
            "/api/system": lambda: self._json(system_info()),
            "/api/memory": lambda: self._json(memory_info(qs.get("q", "").strip(), min(500, int(qs.get("limit", 100))))),
            "/api/roots": lambda: self._json(roots_info()),
            "/api/school": lambda: self._json(school_info()),
            "/api/projects": lambda: self._json(projects_info()),
            "/api/progress": lambda: self._json(progress_info()),
            "/api/owner": lambda: self._json(owner_info()),
            "/api/files": lambda: self._json(list_dir(qs.get("root", "files"), qs.get("path", ""))),
            "/api/file": lambda: self._send_file(qs.get("root", "files"), qs.get("path", ""), qs.get("download") == "1"),
            "/api/logs": lambda: self._logs(qs.get("name", "orchestrator"), min(2000, int(qs.get("lines", 200)))),
        }
        fn = routes.get(u.path)
        if fn is None:
            self._json({"error": "no such api"}, 404)
        else:
            self._guard(fn)

    def do_POST(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        qs = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path == "/api/memory/delete":
            def run() -> None:
                b = self._body_json()
                memory_delete(str(b.get("table")), int(b.get("id")))
                self._json({"ok": True})
            self._guard(run)
        elif u.path == "/api/study/start":
            def run_study() -> None:
                res = start_study(str(self._body_json().get("what") or ""))
                self._json(res, 200 if res.get("ok") else 409)
            self._guard(run_study)
        elif u.path == "/api/project/start":
            def run_project() -> None:
                res = start_project(str(self._body_json().get("goal") or ""))
                self._json(res, 200 if res.get("ok") else 409)
            self._guard(run_project)
        elif u.path == "/api/owner/answer":
            def run() -> None:
                b = self._body_json()
                owner_answer(str(b.get("id", "")), str(b.get("answer", "")))
                self._json({"ok": True})
            self._guard(run)
        elif u.path == "/api/files/upload":
            self._guard(lambda: self._upload(qs.get("root", "files"), qs.get("path", "")))
        elif u.path == "/api/files/delete":
            def run() -> None:
                b = self._body_json()
                root, p, writable = resolve(str(b.get("root", "files")), str(b.get("path", "")))
                if not writable or p == root:
                    raise PermissionError("只读目录")
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                self._json({"ok": True})
            self._guard(run)
        else:
            self._json({"error": "no such api"}, 404)

    def _proxy_ws(self) -> None:
        """Pass /ws through to the assistant's WebSocket, so the page needs only port 80."""
        try:
            up = socket.create_connection(WS_UPSTREAM, timeout=5)
        except OSError:
            self.send_error(502, "assistant offline")
            return
        head = self.raw_requestline + "".join(f"{k}: {v}\r\n" for k, v in self.headers.items()).encode("latin1") + b"\r\n"
        up.sendall(head)
        up.settimeout(None)
        client = self.connection
        client.settimeout(None)
        self.close_connection = True

        def pipe(src: socket.socket, dst: socket.socket) -> None:
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                for s in (src, dst):
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        t = threading.Thread(target=pipe, args=(up, client), daemon=True)
        t.start()
        pipe(client, up)
        t.join(timeout=2)
        up.close()

    def _send_file(self, root_id: str, rel: str, download: bool) -> None:
        _, p, _ = resolve(root_id, rel)
        if not p.is_file():
            raise FileNotFoundError(rel)
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or p.suffix.lower() in (".log", ".jsonl", ".json", ".yaml", ".yml", ".md", ".txt", ".py", ".sh"):
            ctype = "text/plain; charset=utf-8"
        size = p.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("X-Content-Type-Options", "nosniff")
        disp = "attachment" if download else "inline"
        self.send_header("Content-Disposition", f"{disp}; filename*=UTF-8''{urllib.parse.quote(p.name)}")
        self.end_headers()
        with open(p, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def _upload(self, root_id: str, rel: str) -> None:
        name = urllib.parse.unquote(self.headers.get("X-Filename", "")).strip()
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("文件名不合法")
        root, d, writable = resolve(root_id, rel)
        if not writable:
            raise PermissionError("只读目录")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_UPLOAD:
            raise ValueError("文件太大或为空")
        d.mkdir(parents=True, exist_ok=True)
        _, target, _ = resolve(root_id, str((d / name).relative_to(root)))
        tmp = target.with_name(target.name + ".part")
        with open(tmp, "wb") as f:
            left = n
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    raise ValueError("上传中断")
                f.write(chunk)
                left -= len(chunk)
        tmp.replace(target)
        self._json({"ok": True, "name": name})

    def _logs(self, name: str, lines: int) -> None:
        if name not in LOGS:
            raise PermissionError("unknown log")
        p = AGENT_ROOT / "logs" / LOGS[name]
        if not p.is_file():
            raise FileNotFoundError(LOGS[name])
        self._json({"name": name, "text": tail(p, lines), "logs": list(LOGS)})

    def log_message(self, fmt: str, *args: Any) -> None:
        if "/api/system" not in str(args):
            super().log_message(fmt, *args)


def main() -> None:
    port = int(os.environ.get("UI_PORT", 80))
    host = os.environ.get("UI_HOST", "0.0.0.0")
    server = Server((host, port), Handler)
    LOG.info("星期一 HUD on http://%s:%d/", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
