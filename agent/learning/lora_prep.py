"""Build the LoRA training set from what 星期一 collected (runs on the board, with the teacher).

    cd /userdata/agent && python3 -m learning.lora_prep [--config config/agent.yaml]

Sources: run/learning/lora_curated.jsonl (graded turns) + lora_data.jsonl (class and practice)
+ distill/*.jsonl (bulk distillation from the teacher, learning/distill.py).

Two modes, kept apart by the system prompt:
- voice (her persona prompt): everything is read aloud, so replies must be speakable;
- coding (learning.school.CODING_SYSTEM): she writes code herself (class now, her own scripts later),
  so replies stay real code;
- agent (learning.drill.AGENT_SYSTEM): one step of a tool-using task, the reply is step JSON, kept as is.
Older class samples were logged under the voice prompt, which taught her to *say* code. Routing:
- sample already under the coding prompt -> kept as is;
- voice sample whose reply is not speakable and whose question is a coding exercise ("写函数…",
  "这段代码输出什么", "给我讲讲…") -> moved to the coding prompt, original code answer kept;
- any other voice reply that is not speakable -> rewritten by the teacher into spoken Chinese (same
  content, no symbols); still not speakable afterwards -> dropped.
Rewrites are cached (run/learning/lora_spoken_cache.json), so a rerun only asks about new samples.
Samples that touch account details are dropped. Output: run/learning/lora_train.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOG = logging.getLogger("lora_prep")
AGENT_ROOT = Path(__file__).resolve().parents[1]
SAME_QUESTION_MAX = 2   # per mode: copies of one question kept for training

SPOKEN_SYSTEM = """你是语音助手「星期一」的话术编辑。下面这条回答会被直接朗读出来，但里面有代码、命令、符号或格式，念出来很别扭。
把它改写成适合朗读的口语中文：
- 意思和结论必须完全保持正确，不要加新信息，也不要丢掉关键答案；
- 不出现代码块、反引号、方括号、花括号、引号包着的字符串、路径、缩进；
- 代码题：用一两句话说清楚怎么写，比如「用 n 对 2 取余，余数是零就返回真」；
- 输出结果题：直接说结果，比如列表 [3, 2, 1] 说成「三、二、一」，['12', '3'] 说成「两个字符串，十二和三」；
- 命令可以说命令名，比如「用 df 命令加上 h 参数」，不要念横杠、斜杠这类符号；
- 简短，最多三句；不要开场白，不要「好的」「主人」之类的客套（原文有称呼可以保留）。
只输出 JSON：{"spoken": "改写后的回答"}"""

_CODE_HINTS = re.compile(
    r"```|`|^\s*(def|class|import|from|return|for|while|if|elif|else:|print\(|with|try:|except)\b|"
    r"[\[\]{}=*@]|!=|<=|>=|->|\d+\.\d+\.\d+|\w+\([^)]*\)|(^|\s)-{1,2}[a-zA-Z]|(^|\s)[~.]?/[\w.-]+|\\[nt]|^\s*[#$>]\s",
    re.M)
_CODING_Q = re.compile(
    r"^\s*(写|编写|实现|定义|给我讲讲)|这段代码|下面的代码|代码[:：]|输出(是)?什么|返回(值)?(是)?什么|会输出|运行结果|"
    r"^\s*(def|import|from|print\()|```|\w+\([^)]*\)", re.M)
_ACCOUNT = re.compile(r"密码已隐藏|账号是|密码是|password\s*[:=]", re.I)


def speakable(text: str) -> bool:
    return not _CODE_HINTS.search(text)


def coding_question(q: str) -> bool:
    return bool(_CODING_Q.search(q))


def coding_system() -> str:
    from learning.school import CODING_SYSTEM
    return CODING_SYSTEM


def agent_system() -> str:
    from learning.drill import AGENT_SYSTEM
    return AGENT_SYSTEM


def agent_systems() -> set[str]:
    """The current agent prompt and its earlier wordings (samples logged under any of them are agent steps)."""
    from learning.drill import AGENT_SYSTEM, AGENT_SYSTEM_OLD
    return {t.strip() for t in (AGENT_SYSTEM, *AGENT_SYSTEM_OLD)}


def route(m: list[dict]) -> str:
    """'coding' | 'agent' | 'voice' | 'rewrite' for one sample (see the module docstring)."""
    q, a = m[-2]["content"], m[-1]["content"]
    from learning.selfprompt import strip
    sys_text = strip(m[0].get("content", "")).strip() if m[0].get("role") == "system" else ""
    if sys_text == coding_system():
        return "coding"
    if sys_text in agent_systems():
        return "agent"
    if speakable(a):
        return "voice"
    return "coding" if coding_question(q) else "rewrite"


def _key(q: str, a: str) -> str:
    return hashlib.sha1(f"{q}\x00{a}".encode("utf-8")).hexdigest()[:16]


def load_rows(learning: Path) -> list[dict]:
    seen, rows = set(), []
    sources = [learning / "lora_curated.jsonl", learning / "lora_data.jsonl"]
    sources += sorted((learning / "distill").glob("*.jsonl")) if (learning / "distill").is_dir() else []
    for f in sources:   # graded turns, class and practice, then the bulk distillation (learning.distill)
        name = f.name
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                m = r["messages"]
                q, a = m[-2]["content"].strip(), m[-1]["content"].strip()
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            if not a or (q, a) in seen or _ACCOUNT.search(q + "\n" + a):
                continue
            seen.add((q, a))
            rows.append({"messages": m, "source": r.get("source") or name.split(".")[0]})
    return rows


def rewrite(teacher, q: str, a: str) -> str:
    raw = teacher.complete(SPOKEN_SYSTEM, f"问题：{q}\n\n原回答：\n{a}", json_mode=True, max_tokens=300)
    try:
        return str(json.loads(raw).get("spoken", "")).strip()
    except (ValueError, AttributeError):
        return ""


def build(teacher, learning: Path, *, workers: int = 4) -> dict:
    cache_f = learning / "lora_spoken_cache.json"
    try:
        cache = json.loads(cache_f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    rows = load_rows(learning)
    todo = [r for r in rows if route(r["messages"]) == "rewrite"
            and _key(r["messages"][-2]["content"], r["messages"][-1]["content"]) not in cache]

    def job(r):
        q, a = r["messages"][-2]["content"].strip(), r["messages"][-1]["content"].strip()
        try:
            return _key(q, a), rewrite(teacher, q, a)
        except Exception as exc:  # noqa: BLE001 - one failed call just leaves that sample out this time
            LOG.warning("rewrite failed: %s", exc)
            return _key(q, a), None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, spoken in ex.map(job, todo):
            if spoken is not None:
                cache[k] = spoken
    cache_f.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")

    out, stats = [], {"total": len(rows), "voice": 0, "coding": 0, "agent": 0, "rewritten": 0, "dropped": 0}
    per_q: dict[tuple[str, str], int] = {}
    for r in rows:
        m = [dict(x) for x in r["messages"]]
        if m and m[0].get("role") == "system":   # her own prompt rules never go into training (learning.selfprompt)
            from learning.selfprompt import strip
            m[0]["content"] = strip(m[0]["content"])
        q, a = m[-2]["content"].strip(), m[-1]["content"].strip()
        mode = route(m)
        k = (mode, re.sub(r"\s+", "", q))
        per_q[k] = per_q.get(k, 0) + 1
        if per_q[k] > SAME_QUESTION_MAX:   # the same lecture taught ten times is learnt by heart, not understood
            stats["repeats"] = stats.get("repeats", 0) + 1
            continue
        if mode == "voice":
            stats["voice"] += 1
        elif mode == "coding":
            m = [{"role": "system", "content": coding_system()}] + [x for x in m if x.get("role") != "system"]
            stats["coding"] += 1
        elif mode == "agent":
            stats["agent"] += 1   # step JSON for the agent loop: reply kept as is
            # older contexts predate the "还能执行 N 条命令" line: rewrite them so all samples agree
            # and older system wordings are trained under the current one
            from learning.drill import upgrade_context
            m[-2]["content"] = upgrade_context(m[-2]["content"])
            m = [{"role": "system", "content": agent_system()}] + [x for x in m if x.get("role") != "system"]
        else:
            spoken = cache.get(_key(q, a), "")
            if not spoken or not speakable(spoken):
                stats["dropped"] += 1
                continue
            m[-1]["content"] = spoken
            stats["rewritten"] += 1
        out.append({"messages": m, "source": r["source"]})
    with (learning / "lora_train.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["written"] = len(out)
    return stats


def main() -> None:
    import sys

    sys.path.insert(0, str(AGENT_ROOT))
    from learning.teacher import DeepSeekTeacher
    from orchestrator.config_loader import load_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(AGENT_ROOT / "config" / "agent.yaml"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_yaml(a.config)
    t = cfg.get("teacher") or {}
    teacher = DeepSeekTeacher(base_url=t.get("base_url", "https://api.deepseek.com"), model=t.get("model", "deepseek-flash"),
                              api_key_env=t.get("api_key_env", "DEEPSEEK_API_KEY"), timeout_s=float(t.get("timeout_s", 20)))
    if not teacher.available:
        raise SystemExit("no teacher key: run with the deepseek EnvironmentFile")
    learning = Path((cfg.get("learning") or {}).get("db", str(AGENT_ROOT / "run/learning/memory.db"))).parent
    stats = build(teacher, learning)
    LOG.info("[lora_prep] %s -> %s", stats, learning / "lora_train.jsonl")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
