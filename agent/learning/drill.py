"""Hands-on drills for 星期一's local model: real coding and real agent work, graded by results.

Run from learning.practice (its timer slots rotate knowledge / speaking / coding / agent):

coding - the teacher writes small Python tasks with assert tests and its own solution; a task is kept
         only if that solution passes the sandbox (learning.school.run_sandboxed). The local model
         answers under CODING_SYSTEM; its code runs against the tests. Passed: its own answer becomes
         training data. Failed: the teacher's verified solution does.
agent  - the teacher writes owner requests a person would solve with a few read-only commands on the
         board ("板子现在多热"). The local model plans one step at a time as JSON under AGENT_SYSTEM
         (shell + done). Only steps learning.agent.classify() calls "safe" are executed; anything else
         is not run and the try counts as failed. The teacher judges the finished run (right numbers,
         speakable answer). Passed: every step becomes a training sample (context -> step JSON).
         Failed: the teacher solves the same task in the same format, under the same rules; if that
         run is judged right, its steps become the samples.
Samples go to run/learning/lora_data.jsonl under their own system prompt (lora_prep keeps the modes
apart); one summary line per drill goes to run/learning/drill.jsonl (local pass rate over time).
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

LOG = logging.getLogger("drill")

AGENT_MAX_STEPS = 4
CODE_TASKS = 3
AGENT_TASKS = 2

# What the local model sees when it works as an agent. Kept short: a whole run has to fit
# tools/lora/train_lora.py's 1024-token max_len, which cuts long samples from the front (the system prompt).
# 2026-11-07: commands now run in the read-only sandbox (learning.sandbox), and wrong arithmetic was the
# main reason left for failed drills (51769 millidegrees, KB -> MB, ...): a 4B model gets mental maths
# wrong but writes `python3 -c 'print(51769/1000)'` fine, so it is told to compute instead.
# 2026-11-07 (v5 eval): asked whether r1-orchestrator / r1-tts-daemon were alive, she ran
# `ps | grep <service name>`, found nothing (their processes are python3 -m orchestrator.main and
# rknn_qwen3_tts) and said they were down, so service questions now go through systemctl is-active.
AGENT_SYSTEM = (
    "你是星期一，在主人的 RK3588 板子（Ubuntu，用户 youyeetoo）上帮主人办事。每次只走一步，只输出一行 JSON：\n"
    '{"tool":"shell","cmd":"一条命令"} 在板子上执行；\n'
    '{"tool":"done","say":"给主人的一两句口语结论"} 结束。\n'
    "命令在只读沙箱里跑：df、free、uptime、ps、ls、du、cat /proc 或 /sys 下的文件、systemctl status、ip addr、ss、lsblk、grep、awk 都行，"
    "日志在 /userdata/agent/logs/ 下（用 tail、grep 读，不用 journalctl）。不用 sudo。"
    "问服务在不在跑，用 systemctl is-active 服务名，active 就是在跑；进程名和服务名不一样（r1-orchestrator 是 python3 -m orchestrator.main，"
    "r1-tts-daemon 是 rknn_qwen3_tts，r1-llm-daemon 是 llm_daemon），ps 里 grep 服务名搜不到不等于它挂了。"
    "要算数或换算（温度是千分之一度、内存是 KB、开机时间是秒、算百分比）不要心算，用 python3 -c 'print(51769/1000)' 这样算出来再说。"
    "看到结果够回答了就 done，说结果里的真实数字，不要念命令。"
)
# Earlier wordings of AGENT_SYSTEM: samples logged under them are still agent samples, and lora_prep
# trains them under the current wording (learning.lora_prep.route / build). Newest first.
AGENT_SYSTEM_OLD = (
    "你是星期一，在主人的 RK3588 板子（Ubuntu，用户 youyeetoo）上帮主人办事。每次只走一步，只输出一行 JSON：\n"
    '{"tool":"shell","cmd":"一条命令"} 在板子上执行；\n'
    '{"tool":"done","say":"给主人的一两句口语结论"} 结束。\n'
    "命令在只读沙箱里跑：df、free、uptime、ps、ls、du、cat /proc 或 /sys 下的文件、systemctl status、ip addr、ss、lsblk、grep、awk 都行，"
    "日志在 /userdata/agent/logs/ 下（用 tail、grep 读，不用 journalctl）。不用 sudo。"
    "要算数或换算（温度是千分之一度、内存是 KB、开机时间是秒、算百分比）不要心算，用 python3 -c 'print(51769/1000)' 这样算出来再说。"
    "看到结果够回答了就 done，说结果里的真实数字，不要念命令。",
    "你是星期一，在主人的 RK3588 板子（Ubuntu，用户 youyeetoo）上帮主人办事。每次只走一步，只输出一行 JSON：\n"
    '{"tool":"shell","cmd":"一条只读命令"} 在板子上执行；\n'
    '{"tool":"done","say":"给主人的一两句口语结论"} 结束。\n'
    "只用只读命令（df、free、uptime、ps、ls、du、cat /proc 或 /sys 下的文件、systemctl status、ip addr、ss、lsblk），"
    "日志在 /userdata/agent/logs/ 下（用 tail、grep 读，不用 journalctl）。一次一条，不用 sudo、分号和重定向。看到结果够回答了就 done，说结果里的真实数字，不要念命令。",
)

CODE_TASK_SYSTEM = (
    "你在给一个四B小模型出 Python 编程题，考它真实的编程能力。每道题：一句话说清要写的函数（给出函数名和参数），"
    "附 3 到 5 条 assert 测试（覆盖边界情况），再附你自己的标准解。只用标准库里的纯计算（不读写文件、不联网、"
    "不导入 os/sys/subprocess），每个解不超过 25 行。难度从入门到中等，类型轮换：字符串、列表字典、排序查找、"
    "递归、简单算法、日期时间计算、文本解析。不要和最近出过的重复。\n"
    '只输出 JSON：{"tasks": [{"q": "题目", "tests": "assert ...\\nassert ...", "solution": "def ..."}]}'
)
AGENT_TASK_SYSTEM = (
    "你在给语音助手「星期一」出实操题：主人随口交代的小事，星期一要在自己那块 RK3588 Linux 板子上用一两条只读命令"
    "查清楚再回答。例如磁盘还剩多少、内存够不够、开机多久了、CPU 温度、哪个进程最占内存、某个服务在不在跑"
    "（服务名以 r1- 开头，比如 r1-llm-daemon、r1-orchestrator、r1-tts-daemon）、本机 IP、某个目录多大"
    "（/userdata/files、/userdata/agent/logs）、监听了哪些端口。大约一半的题要两步才能答清（整道题最多三条命令就够，因为只给四步）：先找再查"
    "（比如先找出最占内存的进程，再看它是哪个服务；先看哪个分区最满，再看里面哪个目录最大；先看服务状态，再到 /userdata/agent/logs/ 下它的日志里找最后的报错）。"
    "大约三分之一的题要把命令输出换算或算一下才能答（CPU 温度文件里是千分之一度、/proc/meminfo 是 KB、"
    "/proc/uptime 是秒、某个占比是百分之几、两个数差多少），主人要的是换算好的结果。"
    "题目像主人随口说话，一句话，中文，答案必须能从命令输出里得到；每道题的开头和说法都要不一样，"
    "不要用「帮我看看板子上现在」这类固定套话开头，主人不会每次都这么说。"
    "不要需要改系统、联网、sudo 或密码的事，不要和最近出过的重复。\n"
    '只输出 JSON：{"tasks": ["...", "..."]}'
)
# How the owner might actually say it: one style per request, so the tasks do not all read
# "帮我看看板子上现在……" (most drill and distill tasks did, 2026-11-06).
AGENT_TASK_STYLES = (
    "很短的随口一问，几个字到十来个字，比如「板子烫不烫」「内存还够用吗」",
    "带情绪或抱怨，比如「怎么感觉今天反应好慢，看看是谁在占 CPU」",
    "先交代一点背景再问，比如「我待会要往板子上拷个大文件，看看哪个盘空间够」",
    "像随手吩咐，比如「去瞅一眼语音那个服务还活着没」",
    "问是不是、有没有、多不多，比如「日志目录是不是又变大了」",
    "说法口语、省略主语，比如「看下开机多久了，顺便看看负载」",
    "两件相关的事连着问，比如「哪个进程最吃内存，它是咱们自己的服务吗」",
)


def agent_task_request(n: int, recent: list[str]) -> str:
    """The user message for AGENT_TASK_SYSTEM: how many tasks, which phrasing style, what to avoid."""
    import random
    style = random.choice(AGENT_TASK_STYLES)
    return (f"出 {n} 道题。这一批的说法风格：{style}（每道题开头不一样）。"
            f"最近出过（不要重复，也不要照着它们的句式）：{'；'.join(recent) or '无'}")
AGENT_JUDGE_SYSTEM = (
    "你在检查语音助手「星期一」有没有把主人交代的小事真正办对。给你任务、它执行的每条命令和输出、它最后对主人说的话。"
    "判对的条件：命令用得合理，最后的话把任务里的每一问都答到了（问了两件事就要答两件），里面的数字和事实与命令输出一致，没有编造，适合直接念出来（不念命令和符号）。"
    "换算要对（单位和量级，比如 51769 是 51.8 度不是 517 度）；用 python3 或 awk 把数算出来是好做法，不算多余步骤。"
    "服务在不在跑以 systemctl 的结果为准：只凭 ps 里 grep 服务名搜不到就说服务挂了，算错（进程名和服务名常常不一样）。"
    '只输出 JSON：{"ok": true或false, "why": "一句话"}'
)

_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def _json(text: str) -> dict:
    m = _JSON_OBJ.search(text or "")
    if not m:
        return {}
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def _strip_think(text: str) -> str:
    return re.sub(r"(?s)<think>.*?</think>", "", text or "").strip()


def _step(text: str) -> None:
    from learning import now
    now.step(text)


def log_sample(learning_dir: Path, system: str, user: str, reply: str, source: str) -> None:
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user},
        {"role": "assistant", "content": reply.strip()}]}
    with (learning_dir / "lora_data.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _log_summary(learning_dir: Path, summary: dict) -> None:
    with (learning_dir / "drill.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def recent_tasks(learning_dir: Path, mode: str, n: int = 30) -> list[str]:
    f = learning_dir / "drill.jsonl"
    if not f.is_file():
        return []
    out: list[str] = []
    for line in f.read_text(encoding="utf-8").splitlines()[-60:]:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("mode") == mode:
            out += [str(r.get("task", "")) for r in d.get("results", [])]
    return out[-n:]


# ---------------------------------------------------------------- coding
def coding_drill(*, teacher, learning_dir: Path, student: Callable[[str], str], idle: Callable[[], bool],
                 n: int = CODE_TASKS) -> dict:
    from learning import weakspots
    from learning.school import CODING_SYSTEM, _code_of, run_sandboxed

    # weak spots she failed before come back first, the same task and tests (learning.weakspots)
    items = [{"q": x["task"], "tests": x.get("tests", ""), "solution": x.get("solution", ""), "again": True}
             for x in weakspots.due(learning_dir, "coding", n)]
    if len(items) < n:
        before = recent_tasks(learning_dir, "coding")
        res = _json(teacher.complete(CODE_TASK_SYSTEM, f"出 {n - len(items)} 道题。最近出过（不要重复）：{'；'.join(t[:40] for t in before[-15:]) or '无'}",
                                     json_mode=True, max_tokens=2500))
        items += [{**t, "again": False} for t in (res.get("tasks") or [])[:n - len(items)] if isinstance(t, dict)]
    results, stopped = [], False
    for t in items:
        q, tests, sol = str(t.get("q") or "").strip(), str(t.get("tests") or "").strip(), _code_of(str(t.get("solution") or ""))
        again = bool(t.get("again"))
        if not (q and tests and (sol or again)):
            if again:   # nothing to check it against: can't be retested
                weakspots.record(learning_dir, "coding", q, False, dropped=True)
            continue
        if sol:
            ok, out = run_sandboxed(sol, tests)
            if not ok:  # the teacher's own task is broken: skip it rather than teach something wrong
                LOG.info("[drill] coding task dropped (teacher solution fails): %s | %s", q[:60], out[:120])
                if again:
                    weakspots.record(learning_dir, "coding", q, False, dropped=True)
                continue
        if not idle():
            stopped = True
            break
        _step(f"{'复测' if again else '写代码'}：{q[:40]}")
        answer = _strip_think(student(q))
        code = _code_of(answer)
        passed, why = run_sandboxed(code, tests) if code else (False, "没有写出代码")
        if passed or sol:
            log_sample(learning_dir, CODING_SYSTEM, q, code if passed else sol, "drill-coding")
        why = "" if passed else why[-160:]
        if again:
            weakspots.record(learning_dir, "coding", q, passed, why)
        elif not passed:
            weakspots.add(learning_dir, "coding", q, why, "drill", tests=tests, solution=sol)
        results.append({"task": q, "passed": passed, "why": why, "retest": again})
    summary = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "coding", "results": results, "stopped": stopped,
               "passed": sum(r["passed"] for r in results), "total": len(results)}
    _log_summary(learning_dir, summary)
    return summary


# ---------------------------------------------------------------- agent
REPEATED = "（这条命令前面已经执行过，结果不会变，没有再跑：换一条命令，或者用已有结果 done）"
LAST_STEP = "\n命令次数用完了，现在必须用 done，根据上面的结果回答。"
# Command output in the planner's context. It used to be a flat 240 characters per step, cut head+tail,
# so the useful middle of `ps` / `df` was often gone and the model ran the command again to look.
# Now the padding is squeezed out first and the newest output gets more room than older ones
# (3 x 200 + 360 = the same 960 characters as before, so a run still fits train_lora's 1024 tokens).
AGENT_KEEP_CHARS = 600     # stored per step (what the judge sees)
AGENT_LATEST_CHARS = 360   # the step just run
AGENT_OLDER_CHARS = 200    # earlier steps


def squeeze(text: str) -> str:
    """Column padding and blank lines out: `ps` / `df` / `ss` output is mostly spaces."""
    lines = (re.sub(r"[ \t]+", " ", ln).strip() for ln in (text or "").splitlines())
    return "\n".join(ln for ln in lines if ln)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rstrip() + "…（后面省略）"


def agent_execute(execute: Callable[[dict], tuple[int, str]]) -> Callable[[dict], tuple[int, str]]:
    """Wrap a raw step runner for run_agent: squeeze, then redact secrets and cap the length."""
    from learning import agent as agent_mod

    def run(step: dict) -> tuple[int, str]:
        code, out = execute(step)
        return code, agent_mod.redact(squeeze(out), AGENT_KEEP_CHARS)
    return run


def local_planner(sock_path: str = "/tmp/r1-llm.sock") -> Callable[[str], str]:
    """The local model as an agent planner under AGENT_SYSTEM plus the rules she wrote for herself
    (learning.selfprompt; drills, eval, local-first live tasks). Samples are still logged under the bare
    AGENT_SYSTEM, so training never carries the rules."""
    from learning import selfprompt
    from llm.client import llm_chat

    def plan(ctx: str) -> str:
        system = selfprompt.apply(AGENT_SYSTEM, "agent")
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n"
        return str(llm_chat(prompt, sock_path=sock_path, max_new_tokens=120).get("text", ""))
    return plan


def agent_context(task: str, history: list[dict]) -> str:
    """What the planner sees for the next step. It says how many commands are left and, at the end, that
    it must answer now: without that, the local model *and* the teacher kept running commands until the
    step budget ran out (most failed drills, 2026-11-06)."""
    ctx = f"任务：{task}"
    if history:
        last = len(history) - 1
        ctx += "\n已执行：\n" + "\n".join(
            f"{i + 1}. {h['cmd']} -> 返回码 {h['code']}\n{_clip(h['output'], AGENT_LATEST_CHARS if i == last else AGENT_OLDER_CHARS)}"
            for i, h in enumerate(history))
    left = AGENT_MAX_STEPS - len(history)
    if left <= 0:
        return ctx + LAST_STEP
    return ctx + (f"\n还能执行 {left} 条命令。下一步？" if history else "\n下一步？")


_OLD_TAIL = "\n下一步？"
_STEP_LINE = re.compile(r"(?m)^(\d+)\. .* -> 返回码 -?\d+$")


def upgrade_context(ctx: str) -> str:
    """An agent-step context written before the step budget was shown (ends "\\n下一步？" after some
    executed steps) -> the current wording, so old and new training samples agree. Idempotent."""
    if "\n已执行：\n" not in ctx or not ctx.endswith(_OLD_TAIL):
        return ctx
    n = 0
    for m in _STEP_LINE.finditer(ctx):
        if int(m.group(1)) == n + 1:
            n += 1
    if not n:
        return ctx
    left = AGENT_MAX_STEPS - n
    return ctx[: -len(_OLD_TAIL)] + (LAST_STEP if left <= 0 else f"\n还能执行 {left} 条命令。下一步？")


def parse_step(text: str) -> dict:
    d = _json(_strip_think(text))
    step = d.get("step") if isinstance(d.get("step"), dict) else d
    tool = step.get("tool")
    if tool == "shell" and str(step.get("cmd") or "").strip():
        return {"tool": "shell", "cmd": str(step["cmd"]).strip()}
    if tool == "done" and str(step.get("say") or "").strip():
        return {"tool": "done", "say": str(step["say"]).strip()}
    return {}


def run_agent(task: str, plan: Callable[[str], str], *, execute: Callable[[dict], tuple[int, str]],
              classify: Callable[[dict], tuple[str, str]],
              sandbox: Callable[[dict], tuple[int, str]] | None = None) -> dict:
    """One try at a task: -> {"steps": [(context, step_json)], "history", "say", "error"}.
    Safe steps run with `execute`; with a `sandbox` (learning.sandbox, wrapped by agent_execute) an
    ask-level shell step runs there too, the way the live agent does; anything else fails the try."""
    from learning.sandbox import eligible
    history: list[dict] = []
    steps: list[tuple[str, str]] = []
    for _ in range(AGENT_MAX_STEPS + 1):
        ctx = agent_context(task, history)
        step = parse_step(plan(ctx))
        if not step:
            return {"steps": steps, "history": history, "say": "", "error": "输出的不是规定的 JSON 步骤"}
        steps.append((ctx, json.dumps(step, ensure_ascii=False)))
        if step["tool"] == "done":
            return {"steps": steps, "history": history, "say": step["say"], "error": ""}
        if len(history) >= AGENT_MAX_STEPS:
            break
        risk, why = classify(step)
        boxed = risk == "ask" and sandbox is not None and eligible(step)
        if risk != "safe" and not boxed:
            return {"steps": steps, "history": history, "say": "", "error": f"命令不在只读范围（{why}）：{step['cmd']}"}
        if any(h["cmd"] == step["cmd"] for h in history):  # v4 reran the same command in ~40% of its runs
            history.append({"cmd": step["cmd"], "code": -1, "output": REPEATED})
            continue
        code, out = (sandbox if boxed else execute)(step)
        history.append({"cmd": step["cmd"], "code": code, "output": out[:AGENT_KEEP_CHARS]})
    return {"steps": steps, "history": history, "say": "", "error": "步数用完还没给结论"}


def judge_agent(teacher, task: str, run: dict) -> tuple[bool, str]:
    from learning.lora_prep import speakable

    if run["error"]:
        return False, run["error"]
    if not speakable(run["say"]):
        return False, "结论里念了命令或符号"
    listing = "\n".join(f"{i + 1}. $ {h['cmd']}（返回码 {h['code']}）\n{h['output']}" for i, h in enumerate(run["history"])) or "（没有执行命令）"
    v = _json(teacher.complete(AGENT_JUDGE_SYSTEM, f"任务：{task}\n执行：\n{listing}\n对主人说：{run['say']}",
                               json_mode=True, max_tokens=300))
    return bool(v.get("ok")), str(v.get("why") or "")


def default_sandbox() -> Callable[[dict], tuple[int, str]] | None:
    """learning.sandbox wrapped for run_agent, or None where there is no bwrap (tests, Windows)."""
    from learning import sandbox
    return agent_execute(sandbox.run_step) if sandbox.available() else None


def agent_drill(*, teacher, learning_dir: Path, local_plan: Callable[[str], str], idle: Callable[[], bool],
                execute: Callable[[dict], tuple[int, str]] | None = None,
                classify: Callable[[dict], tuple[str, str]] | None = None, n: int = AGENT_TASKS,
                sandbox: Callable[[dict], tuple[int, str]] | None = None) -> dict:
    from learning import agent as agent_mod

    execute_redacted = agent_execute(execute or (lambda s: agent_mod.run_step(s, timeout=20.0)))
    if execute is None and sandbox is None:   # the real thing on the board; tests pass their own
        sandbox = default_sandbox()
    classify = classify or agent_mod.classify
    from learning import weakspots

    # weak spots she failed before come back first, the same task judged the same way (learning.weakspots)
    tasks = [(x["task"], True) for x in weakspots.due(learning_dir, "agent", n)]
    if len(tasks) < n:
        before = recent_tasks(learning_dir, "agent")
        res = _json(teacher.complete(AGENT_TASK_SYSTEM, agent_task_request(n - len(tasks), before[-15:]),
                                     json_mode=True, max_tokens=600))
        tasks += [(t, False) for t in [str(t).strip() for t in (res.get("tasks") or []) if str(t).strip()][:n - len(tasks)]]

    def teacher_plan(ctx: str) -> str:
        return teacher.complete(AGENT_SYSTEM, ctx, json_mode=True, max_tokens=300)

    results, stopped = [], False
    for task, again in tasks:
        if not idle():
            stopped = True
            break
        _step(f"{'复测' if again else '自己动手'}：{task[:40]}")
        mine = run_agent(task, local_plan, execute=execute_redacted, classify=classify, sandbox=sandbox)
        passed, why = judge_agent(teacher, task, mine)
        used = mine if passed else None
        doable = True
        if not passed:
            demo = run_agent(task, teacher_plan, execute=execute_redacted, classify=classify, sandbox=sandbox)
            ok, demo_why = judge_agent(teacher, task, demo)
            used, doable = (demo if ok else None), ok
            if not ok:
                LOG.info("[drill] agent task dropped (teacher run not right either): %s | %s", task, demo_why)
        for ctx, reply in (used or {}).get("steps", []):
            log_sample(learning_dir, AGENT_SYSTEM, ctx, reply, "drill-agent")
        why = "" if passed else why[:160]
        if again:
            weakspots.record(learning_dir, "agent", task, passed, why, dropped=not doable)
        elif not passed and doable:
            weakspots.add(learning_dir, "agent", task, why, "drill")
        results.append({"task": task, "passed": passed, "why": why, "retest": again,
                        "cmds": [h["cmd"] for h in mine["history"]], "say": mine["say"][:120], "learned": used is not None})
    summary = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "agent", "results": results, "stopped": stopped,
               "passed": sum(r["passed"] for r in results), "total": len(results)}
    _log_summary(learning_dir, summary)
    return summary
