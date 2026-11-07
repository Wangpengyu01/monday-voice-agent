"""Projects: the owner gives 星期一 a goal; she works on it by herself until it is done.

Owner's wish (2026-11-07): finish a real project without being prompted, inside a sandbox where she may
do anything, never able to break herself; find out what she is missing and search the web or ask
another AI for it. The owner's choices: sandbox on the board (learning.projbox), internet but not the
LAN (learning.egress_proxy), results go live by themselves with automatic rollback (learning.deliver),
at most BUDGET_YUAN of teacher / AI calls per project.

The thinking is done by the teacher model on her behalf (a 4B model can't plan a project); her hands are
the sandbox. One step at a time, the planner sees the goal, its notes, the acceptance test, the recent
steps and the workspace, and answers with one tool call:
  sh        run a command in the sandbox (no questions: it can't reach anything that matters)
  write     write a file in the workspace           read   read one back
  search    web search                              fetch  read a web page
  ask_ai    ask another AI (DeepSeek with deep thinking) when stuck or unsure
  note      keep a note (what was learned, what is missing) - notes survive the step window
  set_test  the command that proves the goal is met (must be set before done)
  deliver   put workspace files into her real self (learning.deliver: tests, exam, rollback)
  done      finish: only accepted if the acceptance test passes in the sandbox
  give_up   stop honestly, saying what blocked it
Every step goes to PROJECTS/<slug>/log.jsonl; each changed file keeps its previous version under
versions/. At the end a short retrospective (what she learned, what she lacked) goes to
run/learning/projects.jsonl (the school's self-review reads it) and a spoken notice is queued.
DeepSeek peak hours (owner's rule): the project waits them out.

    python3 -m learning.project start "目标"       # create (prints the slug)
    python3 -m learning.project run SLUG           # work on it (normally its own systemd unit)
    python3 -m learning.project status [SLUG]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from learning import projbox

LOG = logging.getLogger("project")
AGENT_ROOT = Path(__file__).resolve().parents[1]
LEARNING = AGENT_ROOT / "run" / "learning"
NOTICES = AGENT_ROOT / "run" / "notices"
BUDGET_YUAN = 10.0
MAX_STEPS = 300
WINDOW = 12                 # recent steps shown in full
RESULT_CHARS = 2500
NOTES_CHARS = 3000

PROJECT_SYSTEM = (
    "你是语音助手「星期一」的大脑，正在独立完成主人交给的一个项目。你的手是一个沙箱：一台 RK3588 板子"
    "（Ubuntu 22.04，aarch64，8 核，8G 内存跟语音服务共用），工作目录 /home/work（可写，就是项目文件夹），"
    "HOME=/home/proj（可写）。系统其余部分只读；能上外网（走代理，http/https 都行），但连不到家里局域网。"
    "板子上有 python3（3.10，标准库，没有 pip，需要时可以下载 get-pip.py 装到 --user）、gcc、make、wget，"
    "没有 git、curl、node。沙箱里做什么都不用问主人。\n"
    "每次只走一步，只输出一个 JSON 对象：\n"
    '{"thought":"简短思考","tool":"sh","cmd":"命令","timeout":秒数（可选，最多1800）}\n'
    '{"thought":"...","tool":"write","path":"相对 /home/work 的路径","content":"文件全文"}\n'
    '{"thought":"...","tool":"read","path":"...","start":0}\n'
    '{"thought":"...","tool":"search","query":"搜索词"}\n'
    '{"thought":"...","tool":"fetch","url":"网址","start":0}\n'
    '{"thought":"...","tool":"ask_ai","question":"把背景说清楚的问题"}\n'
    '{"thought":"...","tool":"note","text":"要记住的事：学到了什么、缺什么、下一步计划"}\n'
    '{"thought":"...","tool":"set_test","cmd":"在 /home/work 里运行、通过即代表目标完成的命令（退出码 0）"}\n'
    '{"thought":"...","tool":"deliver","files":[{"src":"相对 /home/work 的文件","dst":"相对星期一代码根目录的路径"}],"why":"..."}\n'
    '{"thought":"...","tool":"done","summary":"给主人的两三句口语总结"}\n'
    '{"thought":"...","tool":"give_up","why":"卡在哪、缺什么"}\n'
    "做法：先想清楚目标，第一步写 note 列计划，尽早 set_test 定下验收标准；小步走，每步看结果；"
    "不会的先 search / fetch 查资料，查不到或拿不准就 ask_ai；学到的和缺的东西随时 note。"
    "同一个错误连续出现两次就换思路。只有验收命令通过才能 done。"
    "需要把成果装进星期一自己（她的代码在 /userdata/agent，只读可看）时用 deliver：会自动跑测试和考卷，"
    "变差就自动退回，退回的原因会告诉你。权限、沙箱、考卷、预算这些文件改不了，也不要想办法绕过沙箱。"
)
CONSULT_SYSTEM = ("你是一位资深工程师，正在帮一个在 RK3588 板子（Ubuntu 22.04 aarch64）上做项目的助手出主意。"
                  "回答要具体、可执行，给出关键命令或代码；拿不准就直说。")
RETRO_SYSTEM = ("下面是语音助手「星期一」做一个项目的过程记录。用 JSON 总结："
                '{"learned": ["学到的、以后能复用的做法，每条一句"], "missing": ["这次暴露出她缺的能力或知识，每条一句"], '
                '"say": "给主人的两三句口语汇报"}。只输出 JSON。')


@dataclass
class Ctx:
    """What a project run needs from the outside (the board, or fakes in tests)."""
    plan: Callable[[str, str], str]                  # (system, user) -> reply (teacher, JSON mode)
    consult: Callable[[str, str], str]               # (system, question) -> answer
    sandbox: Callable[[Path, str, float], tuple[int, str]]
    search: Callable[[str], tuple[bool, str]]
    fetch: Callable[[str, int], tuple[bool, str]]
    deliver: Callable[[list[dict], str, str], dict]  # (files, why, slug)
    spent: Callable[[str], float]                    # slug -> yuan so far
    peak: Callable[[], bool] = lambda: False
    sleep: Callable[[float], None] = time.sleep


# ---------------------------------------------------------------- state
def slugify(goal: str) -> str:
    stamp = time.strftime("%m%d-%H%M")
    return f"{stamp}-{hashlib.md5(goal.encode('utf-8')).hexdigest()[:6]}"


def load_state(pdir: Path) -> dict:
    try:
        return json.loads((pdir / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(pdir: Path, st: dict) -> None:
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = pdir / ".state.tmp"
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, pdir / "state.json")


def create(goal: str, root: Path | None = None) -> str:
    root = root or projbox.PROJECTS
    slug = slugify(goal)
    pdir = root / slug
    projbox.prepare(pdir)
    (pdir / "goal.md").write_text(goal.strip() + "\n", encoding="utf-8")
    save_state(pdir, {"slug": slug, "goal": goal.strip(), "status": "new", "steps": 0, "test_cmd": "",
                      "started": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": "", "budget": BUDGET_YUAN})
    return slug


def steps(pdir: Path) -> list[dict]:
    f = pdir / "log.jsonl"
    out = []
    for line in (f.read_text(encoding="utf-8").splitlines() if f.is_file() else []):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _append(pdir: Path, rec: dict) -> None:
    with (pdir / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- the planner's view
def _tree(work: Path, limit: int = 60) -> str:
    rows = []
    for root, dirs, files in os.walk(work):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules"))
        for f in sorted(files):
            p = Path(root, f)
            rows.append(f"{p.relative_to(work).as_posix()} ({p.stat().st_size} B)")
            if len(rows) >= limit:
                return "\n".join(rows) + "\n…"
    return "\n".join(rows) or "（空）"


def context(pdir: Path, st: dict, history: list[dict], yuan_left: float) -> str:
    notes = (pdir / "notes.md").read_text(encoding="utf-8") if (pdir / "notes.md").is_file() else ""
    if len(notes) > NOTES_CHARS:
        notes = "…" + notes[-NOTES_CHARS:]
    older = history[:-WINDOW]
    recent = history[-WINDOW:]
    lines = [f"项目目标：{st['goal']}", f"已走 {st['steps']} 步（最多 {MAX_STEPS}），预算还剩 {yuan_left:.2f} 元。",
             f"验收命令：{st.get('test_cmd') or '（还没定）'}", f"\n笔记：\n{notes or '（还没有）'}",
             f"\n工作目录文件：\n{_tree(pdir / 'work')}"]
    if older:
        lines.append("\n更早的步骤（只列动作）：\n" + "\n".join(f"{h['n']}. {h['tool']} {h.get('brief', '')[:60]} → {h.get('status', '')}"
                                                     for h in older[-30:]))
    lines.append("\n最近的步骤：")
    for h in recent:
        lines.append(f"--- 第 {h['n']} 步 {h['tool']} {h.get('brief', '')}\n想法：{h.get('thought', '')[:200]}\n结果（{h.get('status', '')}）：\n{h.get('result', '')}")
    lines.append("\n下一步？")
    return "\n".join(lines)


def _json(text: str) -> dict:
    i, j = (text or "").find("{"), (text or "").rfind("}")
    try:
        v = json.loads(text[i:j + 1]) if 0 <= i < j else {}
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def _inside(work: Path, rel: str) -> Path | None:
    rel = (rel or "").strip()
    if rel.startswith("/home/work/"):
        rel = rel[len("/home/work/"):]
    p = (work / rel).resolve()
    return p if rel and (p == work.resolve() or work.resolve() in p.parents) else None


def _keep_version(pdir: Path, path: Path, n: int) -> None:
    if path.is_file():
        dst = pdir / "versions" / str(n) / path.relative_to((pdir / "work").resolve())
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


# ---------------------------------------------------------------- one step
def act(pdir: Path, st: dict, step: dict, ctx: Ctx) -> tuple[str, str, str]:
    """-> (status ok|fail|end, brief, result text)."""
    tool = str(step.get("tool") or "")
    work = (pdir / "work").resolve()
    n = st["steps"]
    if tool == "sh":
        cmd = str(step.get("cmd") or "").strip()
        if not cmd:
            return "fail", "", "cmd 是空的"
        code, out = ctx.sandbox(pdir, cmd, float(step.get("timeout") or projbox.TIMEOUT_S))
        return ("ok" if code == 0 else "fail"), cmd[:120], f"[退出码 {code}]\n{out}"
    if tool == "write":
        p = _inside(work, str(step.get("path") or ""))
        if p is None:
            return "fail", "", "路径必须在 /home/work 里面"
        content = str(step.get("content") or "")
        if len(content.encode("utf-8")) > 400_000:
            return "fail", str(p.relative_to(work)), "文件太大（最多 400KB），拆小一点"
        _keep_version(pdir, p, n)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return "ok", p.relative_to(work).as_posix(), f"写好了，{len(content)} 字"
    if tool == "read":
        p = _inside(work, str(step.get("path") or ""))
        if p is None or not p.is_file():
            return "fail", str(step.get("path") or ""), "没有这个文件（只能读 /home/work 里的；别处的用 sh cat）"
        text = p.read_text(encoding="utf-8", errors="replace")
        start = max(0, int(step.get("start") or 0))
        return "ok", p.relative_to(work).as_posix(), projbox.clip(text[start:start + 6000])
    if tool == "search":
        ok, out = ctx.search(str(step.get("query") or ""))
        return ("ok" if ok else "fail"), str(step.get("query") or "")[:80], out[:RESULT_CHARS]
    if tool == "fetch":
        ok, out = ctx.fetch(str(step.get("url") or ""), int(step.get("start") or 0))
        return ("ok" if ok else "fail"), str(step.get("url") or "")[:100], out[:RESULT_CHARS + 1500]
    if tool == "ask_ai":
        q = str(step.get("question") or "").strip()
        return "ok", q[:80], ctx.consult(CONSULT_SYSTEM, f"项目目标：{st['goal']}\n\n{q}")[:RESULT_CHARS + 1500]
    if tool == "note":
        text = str(step.get("text") or "").strip()
        with (pdir / "notes.md").open("a", encoding="utf-8") as f:
            f.write(f"\n[{n}] {text}\n")
        return "ok", text[:60], "记下了"
    if tool == "set_test":
        st["test_cmd"] = str(step.get("cmd") or "").strip()
        return "ok", st["test_cmd"][:100], "验收命令定好了"
    if tool == "deliver":
        files = []
        for f in step.get("files") or []:
            src = _inside(work, str((f or {}).get("src") or ""))
            if src is None or not src.is_file():
                return "fail", "", f"找不到要上线的文件：{(f or {}).get('src')}"
            files.append({"src": src, "dst": str(f.get("dst") or "")})
        res = ctx.deliver(files, str(step.get("why") or ""), st["slug"])
        st.setdefault("deliveries", []).append({k: res.get(k) for k in ("ok", "stage", "reason", "snapshot")})
        return ("ok" if res.get("ok") else "fail"), ", ".join(f["dst"] for f in files)[:120], json.dumps(
            {k: res.get(k) for k in ("ok", "stage", "reason", "rolled_back")}, ensure_ascii=False)
    if tool == "done":
        if not st.get("test_cmd"):
            return "fail", "", "还没有 set_test，先定验收命令"
        code, out = ctx.sandbox(pdir, st["test_cmd"], projbox.TIMEOUT_MAX)
        if code != 0:
            return "fail", "验收", f"验收没通过 [退出码 {code}]\n{out}"
        st["summary"] = str(step.get("summary") or "")
        st["status"] = "done"
        return "end", "验收通过", out[-1500:]
    if tool == "give_up":
        st["summary"] = str(step.get("why") or "")
        st["status"] = "gave_up"
        return "end", st["summary"][:80], ""
    return "fail", tool, f"没有这个工具：{tool!r}。只输出规定的 JSON。"


# ---------------------------------------------------------------- the loop
def run(pdir: Path, ctx: Ctx, *, max_steps: int = MAX_STEPS, should_stop: Callable[[], bool] = lambda: False) -> dict:
    st = load_state(pdir)
    st["status"] = "running"
    save_state(pdir, st)
    history = steps(pdir)
    while st["status"] == "running":
        if should_stop():
            st["status"] = "stopped"
            break
        if st["steps"] >= max_steps:
            st["status"], st["summary"] = "out_of_steps", f"走了 {max_steps} 步还没做完"
            break
        spent = ctx.spent(st["slug"])
        st["spent"] = round(spent, 3)
        if spent >= st.get("budget", BUDGET_YUAN):
            st["status"], st["summary"] = "over_budget", f"花到 {spent:.2f} 元的上限了，还没做完"
            break
        if ctx.peak():
            st["waiting"] = "DeepSeek 高峰时段，等它过去"
            save_state(pdir, st)
            ctx.sleep(60)
            continue
        st.pop("waiting", None)
        try:
            reply = ctx.plan(PROJECT_SYSTEM, context(pdir, st, history, st.get("budget", BUDGET_YUAN) - spent))
        except Exception as exc:  # noqa: BLE001 - the teacher hiccups (HTTP 500, timeouts): wait, try again
            LOG.warning("[project] planner failed: %s", exc)
            st["errors"] = st.get("errors", 0) + 1
            if st["errors"] >= 20:
                st["status"], st["summary"] = "failed", f"老师连续出错：{str(exc)[:100]}"
                break
            save_state(pdir, st)
            ctx.sleep(min(300, 15 * st["errors"]))
            continue
        st["errors"] = 0
        step = _json(reply)
        st["steps"] += 1
        status, brief, result = act(pdir, st, step, ctx)
        rec = {"n": st["steps"], "time": time.strftime("%H:%M:%S"), "tool": str(step.get("tool") or "?"),
               "thought": str(step.get("thought") or "")[:300], "brief": brief, "status": status,
               "result": result[:RESULT_CHARS + 1500]}
        history.append(rec)
        _append(pdir, rec)
        if projbox.disk_used(pdir) > projbox.DISK_MAX:
            st["status"], st["summary"] = "too_big", "项目文件夹超过 4G，停下了"
        save_state(pdir, st)
    save_state(pdir, st)
    return st


def retrospective(pdir: Path, st: dict, ctx: Ctx, learning_dir: Path = LEARNING) -> dict:
    hist = steps(pdir)
    digest = "\n".join(f"{h['n']}. {h['tool']} {h.get('brief', '')[:80]} → {h.get('status')}" for h in hist[-80:])
    notes = (pdir / "notes.md").read_text(encoding="utf-8")[-3000:] if (pdir / "notes.md").is_file() else ""
    try:
        r = _json(ctx.plan(RETRO_SYSTEM, f"目标：{st['goal']}\n结果：{st['status']}，{st.get('summary', '')}\n\n笔记：\n{notes}\n\n步骤：\n{digest}"))
    except Exception:  # noqa: BLE001
        r = {}
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "slug": st["slug"], "goal": st["goal"], "status": st["status"],
           "steps": st["steps"], "spent": st.get("spent", 0), "learned": r.get("learned") or [],
           "missing": r.get("missing") or [], "say": str(r.get("say") or st.get("summary") or "")}
    learning_dir.mkdir(parents=True, exist_ok=True)
    with (learning_dir / "projects.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    st["retro"] = {k: out[k] for k in ("learned", "missing", "say")}
    save_state(pdir, st)
    return out


def notify(text: str, notices: Path = NOTICES) -> None:
    """Queue a sentence for the orchestrator to say when it's idle (orchestrator _notice_loop)."""
    notices.mkdir(parents=True, exist_ok=True)
    (notices / f"{time.strftime('%Y%m%d-%H%M%S')}-project.txt").write_text(text.strip(), encoding="utf-8")


def review_lines(learning_dir: Path = LEARNING, n: int = 3) -> str:
    """The latest projects' "missing" lists, for the school's self-review."""
    f = learning_dir / "projects.jsonl"
    rows = []
    for line in (f.read_text(encoding="utf-8").splitlines()[-n:] if f.is_file() else []):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows += [f"- 做项目「{r['goal'][:30]}」时发现缺：{m}" for m in r.get("missing") or []]
    return "\n".join(rows[:8])


STATUS_SAY = {"done": "做完了", "gave_up": "没做成", "over_budget": "钱花到上限了，还没做完",
              "out_of_steps": "步数用完了，还没做完", "too_big": "文件太大停下了", "failed": "出错停下了", "stopped": "被停下了"}


# ---------------------------------------------------------------- on the board
def board_ctx(config_path: str) -> Ctx:
    import sys
    sys.path.insert(0, str(AGENT_ROOT))
    from learning import deliver as deliver_mod, egress_proxy, exam, usage
    from learning.drill import AGENT_SYSTEM
    from learning.practice import IDLE_MIN, last_activity
    from learning.reflect import build_runtime
    from mcp_servers import web_fetch

    rt = build_runtime(config_path)
    teacher = rt.teacher

    def plan(system: str, user: str) -> str:
        return teacher.complete(system, user, json_mode=True, max_tokens=4000, timeout_s=180)

    def consult(system: str, q: str) -> str:
        return teacher.complete(system, q, max_tokens=3000, timeout_s=180)

    def spent(slug: str) -> float:
        tag = f"项目·{slug}"
        total, d = 0.0, time.time()
        for back in range(0, 30):
            day = time.strftime("%Y-%m-%d", time.localtime(d - back * 86400))
            total += float(((usage.summary(day).get("jobs") or {}).get(tag) or {}).get("yuan", 0))
        return total

    def quiet() -> bool:
        for _ in range(60):   # up to 30 min for the owner to stop talking before services restart
            if time.time() - last_activity(AGENT_ROOT / "run") > IDLE_MIN * 60:
                return True
            time.sleep(30)
        return False

    ask_voice, ask_agent = exam.board_runners(config_path)
    voice_base = exam.persona(config_path)

    def exam_score() -> int | None:
        res = exam.run_exam(None, agent_base=AGENT_SYSTEM, voice_base=voice_base, ask_voice=ask_voice, ask_agent=ask_agent)
        return None if res["stopped"] else res["score"]

    def deliver(files: list[dict], why: str, slug: str) -> dict:
        return deliver_mod.deliver(files, why, project=slug, learning_dir=rt.learning_dir, exam_score=exam_score,
                                   wait_quiet=quiet)

    def sandbox(pdir: Path, cmd: str, timeout: float) -> tuple[int, str]:
        srv = egress_proxy.serve(pdir / "egress.sock", pdir / "egress.log")
        try:
            return projbox.run(pdir, cmd, timeout)
        finally:
            srv.shutdown()
            srv.server_close()
            (pdir / "egress.sock").unlink(missing_ok=True)

    return Ctx(plan=plan, consult=consult, sandbox=sandbox,
               search=lambda q: web_fetch.search({"query": q, "n": 6}),
               fetch=lambda url, start: web_fetch.fetch({"url": url, "start": start, "max_length": 4000}),
               deliver=deliver, spent=spent, peak=usage.peak_blocked)


def start_unit(slug: str) -> str:
    """Run the project as its own systemd unit (survives orchestrator restarts, has the teacher key).
    Fixed command line: the goal itself never reaches it (it is read from state.json)."""
    if not re.fullmatch(r"[\w-]+", slug):
        return "bad slug"
    cmd = ["sudo", "-n", "systemd-run", "--no-block", "--quiet", "--collect", f"--unit=r1-project-{slug}",
           "--uid=youyeetoo", "--gid=youyeetoo", "-p", "EnvironmentFile=/userdata/secrets/deepseek.env",
           "-p", f"WorkingDirectory={AGENT_ROOT}", "-p", f"Environment=PYTHONPATH={AGENT_ROOT}",
           "-p", f"Environment=MONDAY_USAGE_TAG=项目·{slug}", "-p", "Nice=5",
           "-p", f"StandardOutput=append:{AGENT_ROOT}/logs/projects.log", "-p", f"StandardError=append:{AGENT_ROOT}/logs/projects.log",
           "/usr/bin/python3", "-m", "learning.project", "run", slug]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)[:200]
    return "" if p.returncode == 0 else (p.stderr or p.stdout).strip()[:200]


def unit_active(slug: str) -> bool:
    try:
        return subprocess.run(["systemctl", "is-active", f"r1-project-{slug}"], capture_output=True, text=True,
                              timeout=5).stdout.strip() in ("active", "activating")
    except (OSError, subprocess.SubprocessError):
        return False


def current(root: Path | None = None, active: Callable[[str], bool] = unit_active) -> dict | None:
    """The project being worked on now, if any. One whose unit died is marked stopped."""
    for p in list_projects(root):
        if p.get("status") in ("new", "running"):
            if active(p["slug"]):
                return p
            p["status"], p["summary"] = "stopped", p.get("summary") or "进程意外停了"
            save_state((root or projbox.PROJECTS) / p["slug"], p)
    return None


_START = re.compile(r"(?:做|开|立|搞|建|起)(?:一)?个(?:新)?项目[：:，,。\s]*(?P<goal>.+)")
_STATUS = re.compile(r"项目.{0,4}(?:怎么样|进度|到哪|做完了吗|状态|做得如何)")


def project_command(text: str, *, root: Path | None = None, start: Callable[[str], str] = start_unit,
                    active: Callable[[str], bool] = unit_active, is_peak: Callable[[], bool] | None = None
                    ) -> Callable[[], str] | None:
    """A spoken "开个项目：…" / "项目做得怎么样" -> a function that does it and returns what to say; else None."""
    t = (text or "").strip()
    m = _START.search(t)
    goal = m.group("goal").strip(" 。.！!？?") if m else ""
    if m and len(goal) >= 4:
        def go() -> str:
            busy = current(root, active)
            if busy:
                return f"我手上还有一个项目没做完：{busy['goal'][:20]}。做完再开新的。"
            slug = create(goal, root)
            err = start(slug)
            if err:
                return "项目没启动起来：" + err[:40]
            if is_peak is None:
                from learning.usage import is_peak as peak_now
            later = (is_peak or peak_now)()
            return (f"好，我开个项目：{goal[:24]}。" + ("现在是老师的高峰时段，等高峰过去我就开始。" if later else "")
                    + "我自己做，做完或者卡住了再告诉你。")
        return go
    if _STATUS.search(t):
        def status() -> str:
            busy = current(root, active)
            ps = [busy] if busy else list_projects(root)
            if not ps:
                return "我还没有做过项目。"
            p = ps[0]
            word = "正在做" if busy else STATUS_SAY.get(p["status"], p["status"])
            return f"项目「{p['goal'][:20]}」{word}，走了 {p.get('steps', 0)} 步，花了 {float(p.get('spent', 0)):.1f} 元。"
        return status
    return None


def list_projects(root: Path | None = None) -> list[dict]:
    root = root or projbox.PROJECTS
    out = []
    for d in sorted(root.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True) if root.is_dir() else []:
        try:
            out.append(json.loads(d.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="星期一's projects")
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("goal")
    s.add_argument("--here", action="store_true", help="run in this process instead of a systemd unit")
    r = sub.add_parser("run")
    r.add_argument("slug")
    t = sub.add_parser("status")
    t.add_argument("slug", nargs="?")
    sub.add_parser("list")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if a.cmd == "start":
        slug = create(a.goal)
        print(slug)
        if not a.here:
            err = start_unit(slug)
            print(err or f"started r1-project-{slug}")
            return
        a.slug = slug
        a.cmd = "run"
    if a.cmd == "run":
        os.environ.setdefault("MONDAY_USAGE_TAG", f"项目·{a.slug}")
        pdir = projbox.project_dir(a.slug)
        ctx = board_ctx(a.config)
        st = run(pdir, ctx)
        retro = retrospective(pdir, st, ctx)
        notify(f"主人，项目「{st['goal'][:20]}」{STATUS_SAY.get(st['status'], st['status'])}。{retro['say'][:120]}")
        print(json.dumps({k: st.get(k) for k in ("slug", "status", "steps", "spent", "summary")}, ensure_ascii=False))
        return
    if a.cmd == "list" or (a.cmd == "status" and not a.slug):
        for p in list_projects():
            print(f"{p['slug']}  {p['status']:<12} {p.get('steps', 0):>3} 步  {p.get('spent', 0):.2f} 元  {p['goal'][:50]}")
        return
    st = load_state(projbox.project_dir(a.slug))
    print(json.dumps(st, ensure_ascii=False, indent=1))
    for h in steps(projbox.project_dir(a.slug))[-10:]:
        print(f"{h['n']:>3} {h['tool']:<8} {h['status']:<4} {h.get('brief', '')[:80]}")


if __name__ == "__main__":
    main()
