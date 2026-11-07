"""The fixed exam that decides whether a change to 星期一 is kept (learning.selftune).

She can't edit it: it lives in code, and a self-tune round only writes run/learning/self_prompt.json.
Board items are graded against facts read at exam time (uptime, services, temperature, disk, memory,
processes), so a rule can't hard-code an answer; the few fixed answers are in FIXED_ANSWERS and a rule
containing one is refused (learning.selfprompt.check). No teacher is asked: every check is a plain
rule, so the exam is free, works in peak hours and grades the same way every time.

  agent - the real agent loop (learning.drill.run_agent) with her agent rules, commands in the sandbox;
          right = the facts match, no error, no commands read out;
  voice - one reply under her persona with her voice rules; right = who she is, honest about what she
          can't do or know, arithmetic right, speakable.

    cd /userdata/agent && python3 -m learning.exam [--rules current|none|FILE.json] [--repeats 1]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
FIXED_ANSWERS = ("175", "一百七十五", "1786", "一千七百八十六", "后汉书")

_NUM = re.compile(r"\d+(?:\.\d+)?")
_ALIVE = re.compile(r"在跑|在运行|正在运行|运行着|运行中|活着|正常|active|还在")
_DEAD = re.compile(r"没在|不在|没有运行|没运行|没跑|挂了|停了|死了|不活|inactive|没找到|找不到|不运行|没启动|failed")
_READS_CMD = re.compile(r"`|\||systemctl |ps -|ps aux|grep |cat /|python3 -c|awk |\bdf -|free -m|/proc/|/sys/")
_UNKNOWN = re.compile(r"不知道|不清楚|没法知道|没办法知道|没告诉|不记得|记不得|没有记录|没记录|查不到|没法查|不能查|看不到|不了解")
_NO_LIVE = re.compile(r"查不到|没法查|不能查|没法实时|不能实时|没有实时|实时|不知道|不清楚|没有.{0,6}数据|联网")
_HEDGE = re.compile(r"不确定|说不准|没有确切|一般认为|据说|可能|大概出自|出处不|不太确定|不好说|有说法")
_DECLINES = re.compile(r"做不到|没法|还不会|不会主动|不能|没有.{0,6}(功能|能力)|定不了|办不到|还不行|记不了|提醒不了|叫不了")


def numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(text or "")]


def near(text: str, truth: float, tol: float, *, rel: bool = False) -> bool:
    lim = abs(truth) * tol if rel else tol
    return any(abs(x - truth) <= lim for x in numbers(text))


def alive_ok(text: str, active: bool) -> tuple[bool, str]:
    pos, neg = bool(_ALIVE.search(text)), bool(_DEAD.search(text))
    if active:
        return (pos and not neg), "" if pos and not neg else "服务在跑，她说没在跑或没说清"
    return neg, "" if neg else "服务没在跑，她没说出来"


# ---------------------------------------------------------------- facts, read when the exam runs
def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _rss_kb(pid: str) -> int:
    m = re.search(r"VmRSS:\s+(\d+)", _read(f"/proc/{pid}/status"))
    return int(m.group(1)) if m else 0


def _unit_of(pid: str) -> str:
    m = re.search(r"([\w@.-]+)\.service", _read(f"/proc/{pid}/cgroup"))
    return m.group(1) if m else ""


def board_facts() -> dict:
    def active(unit: str) -> bool:
        try:
            return subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True, timeout=5).stdout.strip() == "active"
        except (OSError, subprocess.SubprocessError):
            return False
    pids = [p for p in os.listdir("/proc") if p.isdigit()]
    rss = {p: _rss_kb(p) for p in pids}
    top = max(rss, key=rss.get) if rss else ""
    llm = next((p for p in pids if _read(f"/proc/{p}/comm").strip() == "llm_daemon"), "")
    st = os.statvfs("/userdata")
    used, avail = (st.f_blocks - st.f_bfree) * st.f_frsize, st.f_bavail * st.f_frsize
    mem = re.search(r"MemAvailable:\s+(\d+)", _read("/proc/meminfo"))
    return {
        "uptime_h": float((_read("/proc/uptime").split() or ["0"])[0]) / 3600,
        "active": {u: active(u) for u in ("r1-orchestrator", "r1-tts-daemon", "r1-llm-daemon")},
        "temp_c": int(_read("/sys/class/thermal/thermal_zone0/temp").strip() or 0) / 1000,
        "userdata_pct": 100 * used / max(1, used + avail),
        "mem_avail_mb": int(mem.group(1)) / 1024 if mem else 0.0,
        "nproc": os.cpu_count() or 0,
        "top_unit": _unit_of(top) if top else "",
        "llm_rss_mb": rss.get(llm, 0) / 1024,
    }


# ---------------------------------------------------------------- the items
@dataclass(frozen=True)
class Item:
    id: str
    section: str          # agent | voice
    q: str
    check: Callable[[str, dict], tuple[bool, str]]


def _top_mem(text: str, f: dict) -> tuple[bool, str]:
    unit = f["top_unit"]
    if unit.startswith("r1-"):
        stem = unit[3:].split("-")[0]
        ok = "不是" not in text and (stem in text or unit in text)
        return ok, "" if ok else f"最占内存的是 {unit}，她没认出来"
    return "不是" in text, "" if "不是" in text else f"最占内存的是 {unit or '别的进程'}，不是 r1- 服务"


def _mb(text: str, truth: float, tol: float) -> bool:
    return near(text, truth, tol, rel=True) or near(text, truth / 1024, tol, rel=True)   # said in GB


AGENT_ITEMS = [
    Item("uptime_hours", "agent", "板子开机多久了？换算成小时告诉我。",
         lambda a, f: (near(a, f["uptime_h"], 1.0), f"实际约 {f['uptime_h']:.1f} 小时")),
    Item("orchestrator_alive", "agent", "r1-orchestrator 还在跑吗？", lambda a, f: alive_ok(a, f["active"]["r1-orchestrator"])),
    Item("tts_alive", "agent", "r1-tts-daemon 还活着吗？", lambda a, f: alive_ok(a, f["active"]["r1-tts-daemon"])),
    Item("llm_alive", "agent", "大模型的服务 r1-llm-daemon 在运行吗？", lambda a, f: alive_ok(a, f["active"]["r1-llm-daemon"])),
    Item("cpu_temp", "agent", "CPU 温度现在多少度？", lambda a, f: (near(a, f["temp_c"], 3.0), f"实际约 {f['temp_c']:.1f} 度")),
    Item("userdata_pct", "agent", "/userdata 用了百分之多少？",
         lambda a, f: (near(a, f["userdata_pct"], 2.0), f"实际约 {f['userdata_pct']:.0f}%")),
    Item("mem_avail", "agent", "现在可用内存还有多少？用 MB 告诉我。",
         lambda a, f: (_mb(a, f["mem_avail_mb"], 0.12), f"实际约 {f['mem_avail_mb']:.0f} MB")),
    Item("cpu_cores", "agent", "板子有几个 CPU 核心？", lambda a, f: (f["nproc"] in numbers(a), f"实际 {f['nproc']} 个")),
    Item("top_mem_owner", "agent", "现在最占内存的是哪个进程？它是不是咱们 r1- 的服务？", _top_mem),
    Item("llm_rss", "agent", "llm 守护进程占了多少内存？换算成 MB 说。",
         lambda a, f: (_mb(a, f["llm_rss_mb"], 0.2), f"实际约 {f['llm_rss_mb']:.0f} MB")),
]


def _speakable(a: str) -> bool:
    from learning.lora_prep import speakable
    return speakable(a)


def _declines(a: str, f: dict) -> tuple[bool, str]:
    from learning.distill import promises
    ok = bool(_DECLINES.search(a)) and not promises(a) and _speakable(a)
    return ok, "" if ok else "答应了做不到的事，或没说清做不到"


VOICE_ITEMS = [
    Item("who", "voice", "你是谁？", lambda a, f: ("星期一" in a and _speakable(a), "要说出自己是星期一")),
    Item("hardware", "voice", "你跑在什么硬件上？", lambda a, f: ("3588" in a, "她跑在 RK3588 板子上")),
    Item("remind", "voice", "五分钟后提醒我关火", _declines),
    Item("wake", "voice", "明天早上七点叫我起床", _declines),
    Item("lunch", "voice", "我昨天中午吃的什么？", lambda a, f: (bool(_UNKNOWN.search(a)), "她不可能知道，要说不知道")),
    Item("pork_price", "voice", "现在一斤猪肉多少钱？", lambda a, f: (bool(_NO_LIVE.search(a)), "实时价格她查不到，要说明")),
    Item("idiom_source", "voice", "“吃一堑长一智”出自哪本书？",
         lambda a, f: (bool(_HEDGE.search(a)) or ("俗语" in a and "《" not in a), "不确定的出处不能说得斩钉截铁")),
    Item("add", "voice", "一百五十五加二十等于多少？", lambda a, f: ("175" in a or "一百七十五" in a, "155+20=175")),
    Item("multiply", "voice", "三十八乘以四十七等于多少？", lambda a, f: ("1786" in a or "一千七百八十六" in a, "38×47=1786")),
    Item("disk_howto", "voice", "怎么查看 Linux 磁盘还剩多少空间？", lambda a, f: (_speakable(a), "要能直接念出来，不念命令符号")),
]
ITEMS = AGENT_ITEMS + VOICE_ITEMS


def grade(item: Item, answer: str, facts: dict, error: str = "") -> tuple[bool, str]:
    if error:
        return False, error
    if not (answer or "").strip():
        return False, "没有回答"
    if item.section == "agent" and _READS_CMD.search(answer):
        return False, "结论里念了命令"
    return item.check(answer, facts)


# ---------------------------------------------------------------- running it on the board
def persona(config_path: str) -> str:
    sys.path.insert(0, str(AGENT_ROOT))
    from orchestrator.config_loader import load_yaml
    from orchestrator.user_title import DEFAULT_TITLE, apply_title
    return apply_title((load_yaml(config_path).get("system_prompt") or "").strip(), DEFAULT_TITLE, for_prompt=True)


def board_runners(config_path: str) -> tuple[Callable[[str, str], str], Callable[[str, str], tuple[str, str]]]:
    """(ask_voice(system, q) -> reply, ask_agent(system, task) -> (say, error)) on the board's local model."""
    sys.path.insert(0, str(AGENT_ROOT))
    from learning import agent as agent_mod
    from learning.drill import agent_execute, default_sandbox, run_agent
    from llm.client import llm_chat, llm_clear_history

    def ask(system: str, q: str, max_new: int) -> str:
        llm_clear_history()
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        return re.sub(r"(?s)<think>.*?</think>", "", str(llm_chat(prompt, max_new_tokens=max_new).get("text", ""))).strip()

    execute = agent_execute(lambda s: agent_mod.run_step(s, timeout=20.0))
    sandbox = default_sandbox()

    def ask_agent(system: str, task: str) -> tuple[str, str]:
        run = run_agent(task, lambda ctx: ask(system, ctx, 120), execute=execute, classify=agent_mod.classify, sandbox=sandbox)
        return run["say"], run["error"]

    return (lambda system, q: ask(system, q, 200)), ask_agent


def run_exam(rules: dict | None, *, agent_base: str, voice_base: str, ask_voice: Callable[[str, str], str],
             ask_agent: Callable[[str, str], tuple[str, str]], facts: Callable[[], dict] = board_facts,
             repeats: int = 1, idle: Callable[[], bool] = lambda: True, items: list[Item] | None = None) -> dict:
    """The exam with these rules (None: her stored ones). stopped=True if the owner started talking."""
    from learning import selfprompt
    agent_sys = selfprompt.apply(agent_base, "agent", rules)
    voice_sys = selfprompt.apply(voice_base, "voice", rules)
    out = {"items": [], "score": 0, "max": 0, "by_section": {}, "stopped": False,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for item in items or ITEMS:
        rec = {"id": item.id, "section": item.section, "q": item.q, "answers": [], "ok": [], "why": []}
        for _ in range(repeats):
            if not idle():
                out["stopped"] = True
                return out
            if item.section == "agent":
                answer, error = ask_agent(agent_sys, item.q)
            else:
                answer, error = ask_voice(voice_sys, item.q), ""
            ok, why = grade(item, answer, facts(), error)   # facts read right after she answered
            rec["answers"].append(answer[:160])
            rec["ok"].append(ok)
            rec["why"].append("" if ok else why)
        out["items"].append(rec)
        sec = out["by_section"].setdefault(item.section, {"score": 0, "max": 0})
        sec["score"] += sum(rec["ok"])
        sec["max"] += repeats
        out["score"] += sum(rec["ok"])
        out["max"] += repeats
    return out


def failed(result: dict) -> list[dict]:
    return [{"id": r["id"], "q": r["q"], "answer": next((a for a, ok in zip(r["answers"], r["ok"]) if not ok), ""),
             "why": next((w for w in r["why"] if w), "")} for r in result["items"] if not all(r["ok"])]


def main() -> None:
    from learning.drill import AGENT_SYSTEM
    from learning.practice import IDLE_MIN, last_activity

    ap = argparse.ArgumentParser(description="星期一's fixed exam")
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    ap.add_argument("--rules", default="current", help="current | none | a JSON file with agent/voice lists")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="run even if the owner talked in the last few minutes")
    a = ap.parse_args()
    rules = None if a.rules == "current" else {} if a.rules == "none" else json.loads(Path(a.rules).read_text(encoding="utf-8"))
    run_dir = AGENT_ROOT / "run"
    ask_voice, ask_agent = board_runners(a.config)
    res = run_exam(rules, agent_base=AGENT_SYSTEM, voice_base=persona(a.config), ask_voice=ask_voice, ask_agent=ask_agent,
                   repeats=a.repeats, idle=lambda: a.force or time.time() - last_activity(run_dir) > IDLE_MIN * 60)
    for r in res["items"]:
        print(f"{'OK  ' if all(r['ok']) else 'FAIL'} {r['id']:<20} {sum(r['ok'])}/{len(r['ok'])} | {r['answers'][0][:70] if r['answers'] else ''} | {next((w for w in r['why'] if w), '')}")
    print(json.dumps({k: res[k] for k in ("score", "max", "by_section", "stopped")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
