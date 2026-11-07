"""Distil the teacher (DeepSeek) into 星期一's own training data, in bulk, checked before it is kept.

The study loop only learns from what the small model gets wrong (a few hundred samples a day). This job
produces thousands at once, shaped like the work she actually does, each one verified:

  voice  - questions the owner might ask in daily life; the teacher answers *as 星期一* (her persona
           prompt) for the ear: kept only if learning.lora_prep.speakable() and short enough; an answer
           that reads badly gets one spoken rewrite, then is kept or dropped.
  self   - questions about herself (her board, hardware, her own code), answered only from her survey
           (run/learning/self/hardware.md) and her code (learning.selfknow.code_units).
  coding - Python tasks with assert tests; the teacher's solution must pass the sandbox
           (learning.school.run_sandboxed) or the task is dropped.
  agent  - owner requests answered by reading the board: the teacher works them out step by step in
           the same JSON format the local model uses (learning.drill.AGENT_SYSTEM), commands really run
           (only what learning.agent.classify calls "safe"), and the finished run is judged; only
           judged-right runs are kept, every step as one sample.

Out: run/learning/distill/<kind>.jsonl (picked up by learning.lora_prep). Progress for the page:
run/learning/distill/state.json. Rerunning continues where it stopped (targets count what is there).
Guards: stops at the yuan budget (--max-yuan, counted from learning.usage), and by default only runs in
DeepSeek's off-peak hours (stops when peak starts; r1-distill.timer resumes it in the next off-peak window). Workers default to 2: the account allows 3 calls at
once and one is left for conversation.

    cd /userdata/agent && python3 -m learning.distill [--voice 3000 --self 600 --coding 1500 --agent 1600]
                                                      [--max-yuan 40] [--workers 2] [--any-time]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from learning import usage

LOG = logging.getLogger("distill")
VOICE_MAX_CHARS = 150

VOICE_AREAS = [
    "日常起居和作息", "做饭买菜和吃什么", "出门、交通和天气以外的出行准备", "学习方法和考试复习", "大学生活和上课",
    "时间安排和做计划", "健康和运动常识（不做诊断）", "情绪、压力和睡眠", "朋友、家人和人际相处", "花钱和省钱",
    "手机和电脑怎么用、怎么设置", "家里的网络、路由器和智能设备", "编程和技术概念用口语讲清楚", "电脑和板子的运维（只要结论或一句话的做法）",
    "科学常识", "中国历史和地理常识", "成语、俗语和语文", "数学心算、单位换算、日期星期推算", "生活和安全常识",
    "推荐和比较（怎么选）", "做事的步骤（怎么办一件事）", "闲聊、开玩笑和斗嘴", "关于你自己：你是谁、能干什么", "鼓励、安慰和陪伴",
]
VOICE_Q_SYSTEM = (
    "你在给语音助手「星期一」造训练题：主人在家随口对它说的话。每句一行意思，中文口语，像真人说话，长短不一，"
    "有问题、有吩咐、有闲聊，偶尔带语气词或说得不太完整。不要需要实时信息的（天气、新闻、价格、比分、今天几号），"
    "不要涉及账号密码，不要和给出的已有例子重复。\n"
    '只输出 JSON：{"questions": ["...", "..."]}'
)
VOICE_ANSWER_RULES = (
    "\n\n（补充要求：这条回答会直接念给主人听。一到三句，先说结论；纯口语，不用列表、序号、括号、引号、符号、代码、"
    "命令和路径；数字和单位用中文说法；需要实时信息或你确实不知道的，就直说并说你去问问老师，不编造。"
    "你没有提醒、定时、闹钟、倒计时、日程这些功能，到点也不会主动开口；主人让你做这些，直说现在还做不到，给个替代办法"
    "（比如用手机定个闹钟），绝不答应「到时候提醒你」「记下了」。）"
)
SELF_QA_SYSTEM = (
    "你在给语音助手「星期一」造训练题，让它了解自己。下面是它自己的材料（它跑在什么板子和硬件上，或者它自己的一段代码）。"
    "出 6 个主人可能问它关于它自己的问题，并以星期一的口吻用第一人称、口语回答：一到三句，只依据材料，材料里没有的不要编；"
    "只问不会变的事：它是什么板子、有哪些硬件部件和接口、某个服务或某段代码是干什么的、怎么工作的；"
    "不要问此刻的状态（在不在运行、开没开、温度、占用、剩多少、有没有报错），这些每次都要现查，不能背答案；"
    "不念代码、命令、路径和符号，函数名可以换成说它是做什么的。\n"
    '只输出 JSON：{"pairs": [{"q": "...", "a": "..."}]}'
)
# "is X running / how hot / how much is left" changes all the time: she must look it up, never recite it
_LIVE_STATE = re.compile(r"在运行|在跑|跑着|运行着|开着|没启动|没在|停止|状态|待命|现在|目前|此刻|还剩|剩下|还够|用了|占用|温度|多热|正常|报错|宽裕")
CODE_BATCH = 5
CODE_PER_FUNC = 5      # the teacher keeps coming back to its favourite tasks; more copies only teach memorising
# answers that promise something she cannot do: she has no reminders / timers, never speaks up on her own
_PROMISE = re.compile(r"[设定]好了|(分钟|小时|钟头)后.{0,4}(提醒|叫)|点.{0,4}(提醒|叫)(你|您)|记下了|记好了|记住了|已经?设好|帮你设|帮您设|倒计时.{0,4}开始|到点我|我会提醒|我提醒|会叫你|叫您|帮您记|帮你记|时间一到|我.{0,10}提醒(你|您)|(到时候?|到点|一会儿?|待会儿?|回头).{0,6}(提醒|叫)(你|您)")
_CANNOT = re.compile(r"不|没法|没有|做不到|还不能|定不了|记不了")
DECLINES = re.compile(r"做不到|没法|还不会|不会主动|不能|没有.{0,6}(功能|能力)|定不了|办不到|还不行")


def promises(text: str) -> bool:
    """Does the answer promise a reminder / timer she cannot do? Judged clause by clause, so an honest
    「我没法到点提醒你」 is fine but 「……或者我帮您记下这件事？」 is not."""
    return any(_PROMISE.search(c) and not _CANNOT.search(c) for c in re.split(r"[，。！？；,!?;]", text or ""))
CLEAN_VOICE_SYSTEM = (
    "语音助手「星期一」没有提醒、定时、闹钟、倒计时、日程功能，到点也不会主动开口说话。下面是主人的一句话和她的回答。"
    "如果回答答应了到时候提醒、计时、叫人、记着事回头再说这类她做不到的事，就以她的口吻重写：一到两句纯口语，"
    "直说现在还做不到，给个替代办法（比如用手机定个闹钟），别的有用内容可以保留；没有这种问题就保持不变。\n"
    '只输出 JSON：{"ok": true} 或 {"ok": false, "a": "重写后的回答"}'
)
AGENT_BATCH = 8
VOICE_BATCH = 20


class Budget:
    """Stops the job at max_yuan: everything the distillation ever spent (all runs, from learning.usage),
    so stopping at peak hours and resuming later never adds up past the budget."""

    def __init__(self, max_yuan: float, usage_dir: Path | None = None) -> None:
        self.max_yuan, self.t0, self.dir = max_yuan, time.time(), usage_dir or usage.USAGE_DIR
        self._spent, self._checked = 0.0, 0.0

    def spent(self) -> float:
        if time.time() - self._checked > 20:
            total = 0.0
            for f in sorted(self.dir.glob("*.jsonl")):
                for raw in f.read_text(encoding="utf-8").splitlines():
                    try:
                        r = json.loads(raw)
                    except ValueError:
                        continue
                    if r.get("job") == "蒸馏":
                        total += float(r.get("yuan") or 0)
            self._spent, self._checked = total, time.time()
        return self._spent

    def ok(self) -> bool:
        return self.spent() < self.max_yuan


class Store:
    """run/learning/distill/<kind>.jsonl, thread-safe appends, question dedupe across runs."""

    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.count: dict[str, int] = {}
        self.seen: set[str] = set()
        self.funcs: dict[str, int] = {}
        for f in self.dir.glob("*.jsonl"):
            n = 0
            for raw in f.read_text(encoding="utf-8").splitlines():
                try:
                    m = json.loads(raw)["messages"]
                except (ValueError, KeyError):
                    continue
                n += 1
                self.seen.add(norm(m[-2]["content"]))
                if f.stem == "coding":
                    fn = func_name(m[-1]["content"])
                    self.funcs[fn] = self.funcs.get(fn, 0) + 1
            self.count[f.stem] = n

    def add(self, kind: str, system: str, user: str, reply: str, *, dedupe: bool = True) -> bool:
        key = norm(user)
        with self.lock:
            if dedupe and key in self.seen:
                return False
            self.seen.add(key)
            rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": f"distill-{kind}", "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user},
                {"role": "assistant", "content": reply.strip()}]}
            with (self.dir / f"{kind}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.count[kind] = self.count.get(kind, 0) + 1
            return True


def func_name(code: str) -> str:
    g = re.search(r"def (\w+)", code or "")
    return g.group(1) if g else "?"


def agent_commands(steps) -> list[str]:
    """The shell commands of one agent run, in order (steps = [(context, reply json)])."""
    out = []
    for _, reply in steps:
        try:
            d = json.loads(reply)
        except (ValueError, TypeError):
            continue
        if isinstance(d, dict) and d.get("tool") == "shell":
            out.append(str(d.get("cmd") or "").strip())
    return out


def norm(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?~～…]+", "", text or "").lower()


def _json(text: str) -> dict:
    from learning.drill import _json as j
    return j(text)


def voice_ok(text: str) -> bool:
    from learning.lora_prep import speakable
    return bool(text) and speakable(text) and len(text) <= VOICE_MAX_CHARS


# ---------------------------------------------------------------- one unit of work per kind
def voice_unit(teacher, store: Store, persona: str, area: str, stats: dict) -> None:
    from learning.lora_prep import rewrite
    examples = random.sample(sorted(store.seen), min(8, len(store.seen))) if store.seen else []
    res = _json(teacher.complete(VOICE_Q_SYSTEM, f"话题：{area}。出 {VOICE_BATCH} 句。已有例子（别重复这类说法）：{'；'.join(examples) or '无'}",
                                 json_mode=True, max_tokens=1500))
    for q in [str(x).strip() for x in res.get("questions") or [] if str(x).strip()]:
        if norm(q) in store.seen:
            continue
        a = (teacher.complete(persona + VOICE_ANSWER_RULES, q, max_tokens=300) or "").strip()
        if not voice_ok(a):
            a = rewrite(teacher, q, a) if a else ""
        if voice_ok(a) and not promises(a):
            store.add("voice", persona, q, a)
        else:
            stats["dropped"] = stats.get("dropped", 0) + 1


def self_unit(teacher, store: Store, persona: str, material: str, stats: dict) -> None:
    res = _json(teacher.complete(SELF_QA_SYSTEM, material[:5000], json_mode=True, max_tokens=1500))
    for p in res.get("pairs") or []:
        if not isinstance(p, dict):
            continue
        q, a = str(p.get("q") or "").strip(), str(p.get("a") or "").strip()
        if q and voice_ok(a) and not _LIVE_STATE.search(q + a):
            store.add("self", persona, q, a)
        else:
            stats["dropped"] = stats.get("dropped", 0) + 1


def coding_unit(teacher, store: Store, stats: dict) -> None:
    from learning.drill import CODE_TASK_SYSTEM
    from learning.school import CODING_SYSTEM, _code_of, run_sandboxed
    recent = random.sample(sorted(store.seen), min(10, len(store.seen))) if store.seen else []
    res = _json(teacher.complete(CODE_TASK_SYSTEM, f"出 {CODE_BATCH} 道题。最近出过（不要重复）：{'；'.join(r[:30] for r in recent) or '无'}",
                                 json_mode=True, max_tokens=3000))
    for t in res.get("tasks") or []:
        if not isinstance(t, dict):
            continue
        q, tests, sol = str(t.get("q") or "").strip(), str(t.get("tests") or "").strip(), _code_of(str(t.get("solution") or ""))
        fn = func_name(sol)
        if store.funcs.get(fn, 0) >= CODE_PER_FUNC:
            stats["dropped"] = stats.get("dropped", 0) + 1
        elif q and tests and sol and run_sandboxed(sol, tests)[0]:
            if store.add("coding", CODING_SYSTEM, q, sol):
                with store.lock:
                    store.funcs[fn] = store.funcs.get(fn, 0) + 1
        else:
            stats["dropped"] = stats.get("dropped", 0) + 1


def agent_unit(teacher, store: Store, stats: dict) -> None:
    from learning import agent as agent_mod
    from learning.drill import (AGENT_SYSTEM, AGENT_TASK_SYSTEM, agent_execute, agent_task_request, default_sandbox,
                                judge_agent, run_agent)

    execute = agent_execute(lambda s: agent_mod.run_step(s, timeout=20.0))
    sandbox = default_sandbox()
    recent = random.sample(sorted(store.seen), min(10, len(store.seen))) if store.seen else []
    res = _json(teacher.complete(AGENT_TASK_SYSTEM, agent_task_request(AGENT_BATCH, recent),
                                 json_mode=True, max_tokens=800))
    for task in [str(t).strip() for t in res.get("tasks") or [] if str(t).strip()]:
        if norm(task) in store.seen:
            continue
        run = run_agent(task, lambda ctx: teacher.complete(AGENT_SYSTEM, ctx, json_mode=True, max_tokens=300),
                        execute=execute, classify=agent_mod.classify, sandbox=sandbox)
        cmds = agent_commands(run["steps"])
        ok = len(cmds) == len(set(cmds)) and judge_agent(teacher, task, run)[0]   # running the same command again teaches going in circles
        if not ok:
            stats["dropped"] = stats.get("dropped", 0) + 1
            continue
        for i, (ctx, reply) in enumerate(run["steps"]):
            store.add("agent", AGENT_SYSTEM, ctx, reply, dedupe=(i == 0))


# ---------------------------------------------------------------- the job
def self_materials(learning_dir: Path) -> list[str]:
    from learning.selfknow import code_units
    out = []
    hw = learning_dir / "self" / "hardware.md"
    if hw.is_file():
        text = hw.read_text(encoding="utf-8")
        out += [text[i:i + 3000] for i in range(0, len(text), 3000)]
    try:
        out += [f"我的代码 {u.path}（{u.topic}）：\n{u.text}" for u in code_units()]
    except Exception as exc:  # noqa: BLE001 - hardware material alone still works
        LOG.warning("[distill] code units: %s", exc)
    random.shuffle(out)
    return out


def save_state(out_dir: Path, data: dict) -> None:
    # one temp file per thread: workers save state at the same time, and with a shared "state.tmp" one
    # renamed it away under the other (FileNotFoundError ended the 2026-11-07 run at "stopped")
    tmp = out_dir / f"state.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_dir / "state.json")


def run(*, teacher, persona: str, learning_dir: Path, targets: dict[str, int], max_yuan: float, workers: int,
        any_time: bool, units: dict | None = None) -> dict:
    out_dir = learning_dir / "distill"
    store = Store(out_dir)
    budget = Budget(max_yuan)
    stats: dict = {}
    materials = self_materials(learning_dir) if targets.get("self") else []
    units = units or {
        "voice": lambda: voice_unit(teacher, store, persona, random.choice(VOICE_AREAS), stats),
        "self": lambda: self_unit(teacher, store, persona, materials.pop() if materials else "", stats),
        "coding": lambda: coding_unit(teacher, store, stats),
        "agent": lambda: agent_unit(teacher, store, stats),
    }
    stop = {"why": ""}

    def state(status: str) -> None:
        save_state(out_dir, {"status": status, "why": stop["why"], "targets": targets, "done": dict(store.count),
                             "dropped": stats.get("dropped", 0), "yuan": round(budget.spent(), 3), "max_yuan": max_yuan,
                             "started": budget.t0, "updated": time.time()})

    def should_stop() -> bool:
        if not budget.ok():
            stop["why"] = f"花到预算上限 {max_yuan} 元"
        elif not any_time and usage.is_peak():
            stop["why"] = "到了高峰时段，空闲时段再接着跑"
        elif all(store.count.get(k, 0) >= n for k, n in targets.items()):
            stop["why"] = "全部做完"
        return bool(stop["why"])

    def worker() -> None:
        while not should_stop():
            todo = [k for k, n in targets.items() if store.count.get(k, 0) < n and (k != "self" or materials)]
            if not todo:
                stop["why"] = stop["why"] or "全部做完"
                return
            kind = min(todo, key=lambda k: store.count.get(k, 0) / max(1, targets[k]))  # keep the kinds balanced
            try:
                units[kind]()
            except Exception as exc:  # noqa: BLE001 - one bad batch (network, odd JSON) never stops the job
                LOG.warning("[distill] %s batch failed: %s", kind, exc)
                time.sleep(5)
            state("running")
            LOG.info("[distill] %s · %.2f 元", " ".join(f"{k} {store.count.get(k, 0)}/{n}" for k, n in targets.items()), budget.spent())

    state("running")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(worker) for _ in range(workers)]:
            f.result()
    state("stopped")
    LOG.info("[distill] stopped: %s", stop["why"])
    return {"done": dict(store.count), "dropped": stats.get("dropped", 0), "yuan": budget.spent(), "why": stop["why"]}


def _read(f: Path) -> list[dict]:
    if not f.is_file():
        return []
    out = []
    for raw in f.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(raw))
        except ValueError:
            continue
    return out


def _write(f: Path, rows: list[dict]) -> None:
    tmp = f.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(f)


def fix_promise(teacher, q: str, a: str) -> str | None:
    """The answer as it should be: unchanged, honestly rewritten, or None (drop it)."""
    res = _json(teacher.complete(CLEAN_VOICE_SYSTEM, f"主人：{q}\n她：{a}", json_mode=True, max_tokens=300))
    if res.get("ok") is True:
        return a
    new = str(res.get("a") or "").strip()
    return new if voice_ok(new) and not promises(new) else None


def clean(teacher, out_dir: Path, workers: int = 2) -> dict:
    """One pass over what is already distilled, with the rules the generator now follows:
    voice answers promising reminders / timers are rewritten honestly (or dropped), coding keeps at most
    CODE_PER_FUNC tasks per function name, agent steps that repeat a command are dropped."""
    res = {}
    vf = out_dir / "voice.jsonl"
    rows = _read(vf)
    todo = [i for i, r in enumerate(rows) if promises(r["messages"][-1]["content"])]

    def one(i: int) -> tuple[int, str | None]:
        m = rows[i]["messages"]
        try:
            return i, fix_promise(teacher, m[-2]["content"], m[-1]["content"])
        except Exception as exc:  # noqa: BLE001 - a failed call drops the sample rather than keeping a lie
            LOG.warning("[distill] clean: %s", exc)
            return i, None

    with ThreadPoolExecutor(max_workers=max(1, min(3, workers))) as pool:
        fixed = dict(pool.map(one, todo))
    rewritten = sum(1 for i, v in fixed.items() if v is not None and v != rows[i]["messages"][-1]["content"])
    kept = []
    for i, r in enumerate(rows):
        if i in fixed:
            if fixed[i] is None:
                continue
            r["messages"][-1]["content"] = fixed[i]
        kept.append(r)
    _write(vf, kept)
    res["voice"] = {"checked": len(todo), "rewritten": rewritten,
                    "dropped": sum(1 for v in fixed.values() if v is None), "left": len(kept)}

    cf = out_dir / "coding.jsonl"
    rows, per, kept = _read(cf), {}, []
    for r in rows:
        fn = func_name(r["messages"][-1]["content"])
        per[fn] = per.get(fn, 0) + 1
        if per[fn] <= CODE_PER_FUNC:
            kept.append(r)
    _write(cf, kept)
    res["coding"] = {"before": len(rows), "left": len(kept), "functions": len(per)}

    af = out_dir / "agent.jsonl"
    rows, kept = _read(af), []
    for r in rows:
        m = r["messages"]
        cmds = re.findall(r"^\d+\. (.+?) -> 返回码", m[-2]["content"], re.M) + agent_commands([("", m[-1]["content"])])
        if len(cmds) == len(set(cmds)):
            kept.append(r)
    _write(af, kept)
    res["agent"] = {"before": len(rows), "left": len(kept)}
    return res


def main() -> None:
    from learning.reflect import build_runtime

    ap = argparse.ArgumentParser(description="distil the teacher into checked training data")
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    for k, n in (("voice", 3000), ("self", 600), ("coding", 1500), ("agent", 1600)):   # round 2: agent runs were the weak spot
        ap.add_argument(f"--{k}", type=int, default=n)
    ap.add_argument("--max-yuan", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--any-time", action="store_true", help="also run in DeepSeek's peak hours (double price)")
    ap.add_argument("--clean", action="store_true", help="re-check what is already distilled (see clean()) and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rt = build_runtime(args.config)
    if args.clean:
        LOG.info("[distill] clean: %s", clean(rt.teacher, rt.learning_dir / "distill", args.workers))
        return
    targets = {k: getattr(args, k) for k in ("voice", "self", "coding", "agent") if getattr(args, k) > 0}
    res = run(teacher=rt.teacher, persona=rt.system, learning_dir=rt.learning_dir, targets=targets,
              max_yuan=args.max_yuan, workers=max(1, min(3, args.workers)), any_time=args.any_time)
    LOG.info("[distill] result: %s", res)


if __name__ == "__main__":
    main()
