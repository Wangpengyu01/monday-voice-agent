"""One self-tuning round: 星期一 rewrites the rules she adds to her own prompts, and keeps the change
only if the fixed exam says she got better.

  1. baseline - the exam (learning.exam) with her current rules (learning.selfprompt);
  2. evidence - the items she failed, her open weak spots (learning.weakspots), the rules she has now;
  3. propose  - the teacher thinks it through for her (a 4B model diagnosing itself is too weak) and
                rewrites the rule lists: add, change or remove, with a reason. learning.selfprompt.check
                refuses anything too long, anything about the owner's rules, or an exam answer;
  4. try      - the exam again with the proposed rules;
  5. decide   - kept if the total rose by at least MIN_GAIN and no section went down, or if the score
                held and the rules got clearly shorter; otherwise the old rules stay.
Every round, kept or not, is appended to run/learning/selftune.jsonl (the page shows it as 自我优化记录).
Only files under run/learning are written. It doesn't start in DeepSeek peak hours (owner's rule), and
it stops without changing anything if the owner starts talking (the exam needs the local model).

    cd /userdata/agent && python3 -m learning.selftune [--repeats 2]
    python3 -m learning.selfprompt rollback      # undo the last kept change
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from learning import exam, selfprompt, weakspots

LOG = logging.getLogger("selftune")
LOG_FILE = "selftune.jsonl"
MIN_GAIN = 2           # out of (items x repeats): one lucky answer is not an improvement
SHORTER = 0.8          # same score with rules at most this long (in characters) also counts as better

TUNE_SYSTEM = (
    "你在帮语音助手「星期一」改进她写给自己的经验规则。她的大脑是一个四B本地小模型，跑在 RK3588 板子上。"
    "她有两段固定的基础提示词（不能改）：agent 是在板子上一步步执行只读命令办事，voice 是平时说话。"
    "每段后面可以附上她自己总结的经验规则，每段最多 8 条，每条一句话、不超过 60 字。\n"
    "给你：两段基础提示词、她现在的规则、考卷上她做错的题（题目、她的回答、错在哪）、最近真考没过的事。"
    "请重写两段规则（可以增、改、删），让她在这类题上做对，同时不影响已经做对的题。要求：\n"
    "- 规则要具体可执行，针对错误的原因（比如“问服务在不在跑，用 systemctl is-active 服务名”），不要空话；\n"
    "- 不要把某道题的答案或数字写进规则，要写方法；\n"
    "- 不涉及权限、确认、sudo、密钥密码、联网、删改文件、重启服务：那些是主人定的规矩；\n"
    "- 没用的旧规则删掉，规则越少越好；四B模型读不懂长句。\n"
    '只输出 JSON：{"agent": ["..."], "voice": ["..."], "why": "一两句话：改了什么、为什么"}'
)


def _rules_text(rules: dict) -> str:
    return "\n".join(f"[{s}]\n" + ("\n".join(f"- {r}" for r in rules.get(s) or []) or "（还没有）") for s in selfprompt.SECTIONS)


def evidence(result: dict, learning_dir: Path, rules: dict, agent_base: str, voice_base: str) -> str:
    fails = exam.failed(result)
    lines = [f"- [{f['id']}] {f['q']} → 她答：{f['answer'][:100] or '（没答）'}；错在：{f['why']}" for f in fails]
    return (f"基础提示词 agent：\n{agent_base}\n\n基础提示词 voice：\n{voice_base[:800]}\n\n"
            f"她现在的规则：\n{_rules_text(rules)}\n\n"
            f"考卷 {result['score']}/{result['max']}，做错的：\n" + ("\n".join(lines) or "（全对）") + "\n\n"
            f"最近真考没过的事：\n{weakspots.summary(learning_dir) or '（没有）'}")


def _sections(res: dict) -> dict:
    return {s: v["score"] for s, v in res.get("by_section", {}).items()}


def _chars(rules: dict) -> int:
    return sum(len(r) for s in selfprompt.SECTIONS for r in rules.get(s) or [])


def decide(base: dict, cand: dict, old: dict, new: dict) -> tuple[bool, str]:
    gain = cand["score"] - base["score"]
    drops = [s for s, v in _sections(base).items() if _sections(cand).get(s, 0) < v]
    if drops:
        return False, f"{'、'.join(drops)} 退步了"
    if gain >= MIN_GAIN:
        return True, f"多对了 {gain} 题"
    if gain >= 0 and _chars(new) <= SHORTER * _chars(old) and _chars(old) > 0:
        return True, "分数没降，规则更短"
    return False, f"只多对 {gain} 题，不够 {MIN_GAIN}" if gain > 0 else "没有变好"


def run_round(*, teacher, learning_dir: Path, exam_with: Callable[[dict | None], dict], agent_base: str, voice_base: str,
              peak_blocked: Callable[[], bool] = lambda: False, retry_wait: float = 30.0) -> dict:
    """One round. exam_with(rules) runs the exam (rules None = her stored ones)."""
    old = selfprompt.load(learning_dir)
    rec: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "version": old["version"],
                 "before": {s: old[s] for s in selfprompt.SECTIONS}}

    def finish(decision: str, why: str, **extra) -> dict:
        rec.update(decision=decision, reason=why, **extra)
        with (learning_dir / LOG_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        LOG.info("[selftune] %s: %s", decision, why)
        return rec

    if peak_blocked():
        return finish("skipped", "DeepSeek 高峰时段，不请老师")
    base = exam_with(None)
    rec["base"] = {"score": base["score"], "max": base["max"], "by_section": _sections(base),
                   "failed": [f["id"] for f in exam.failed(base)]}
    if base["stopped"]:
        return finish("aborted", "主人开始说话了，考试中断，什么都没改")
    if not exam.failed(base):
        return finish("skipped", "考卷全对，不用改")
    from learning.teacher import TeacherUnavailable
    ev = evidence(base, learning_dir, old, agent_base, voice_base)
    raw = ""
    for attempt in range(3):   # 2026-11-07 the teacher gave a null body, HTTP 500s and a read timeout in a row
        try:
            raw = teacher.complete(TUNE_SYSTEM, ev, json_mode=True, max_tokens=2000, timeout_s=180)
            break
        except TeacherUnavailable as exc:
            if attempt == 2 or "peak" in str(exc):
                return finish("skipped", f"老师没回应：{str(exc)[:120]}")
            LOG.info("[selftune] teacher failed (%s), retry in %ds", str(exc)[:80], retry_wait * (attempt + 1))
            time.sleep(retry_wait * (attempt + 1))
    try:
        prop = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except ValueError:
        return finish("refused", "老师给的不是 JSON")
    new = selfprompt.clean(prop)
    rec["after"], rec["why"] = new, str(prop.get("why") or "")[:200]
    problems = selfprompt.check(new, exam.FIXED_ANSWERS)
    if problems:
        return finish("refused", "；".join(problems)[:200])
    if all(new[s] == old[s] for s in selfprompt.SECTIONS):
        return finish("skipped", "老师没提出改动")
    cand = exam_with(new)
    rec["cand"] = {"score": cand["score"], "max": cand["max"], "by_section": _sections(cand),
                   "failed": [f["id"] for f in exam.failed(cand)]}
    if cand["stopped"]:
        return finish("aborted", "主人开始说话了，考试中断，什么都没改")
    keep, why = decide(base, cand, old, new)
    if not keep:
        return finish("dropped", why)
    rec["new_version"] = selfprompt.save(new, rec["why"] or why, banned=exam.FIXED_ANSWERS, learning_dir=learning_dir,
                                         scores={"before": base["score"], "after": cand["score"], "max": cand["max"]})
    return finish("kept", why)


def recent(learning_dir: Path, n: int = 10) -> list[dict]:
    f = learning_dir / LOG_FILE
    rows = []
    for line in (f.read_text(encoding="utf-8").splitlines()[-n:] if f.is_file() else []):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def main() -> None:
    from learning.drill import AGENT_SYSTEM
    from learning.practice import IDLE_MIN, last_activity
    from learning.reflect import build_runtime
    from learning.usage import peak_blocked

    ap = argparse.ArgumentParser(description="星期一 tunes her own prompt rules")
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="run even if the owner talked in the last few minutes")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rt = build_runtime(a.config)
    run_dir = exam.AGENT_ROOT / "run"
    voice_base = exam.persona(a.config)
    ask_voice, ask_agent = exam.board_runners(a.config)

    def idle() -> bool:
        return a.force or time.time() - last_activity(run_dir) > IDLE_MIN * 60

    def exam_with(rules: dict | None) -> dict:
        return exam.run_exam(rules, agent_base=AGENT_SYSTEM, voice_base=voice_base, ask_voice=ask_voice,
                             ask_agent=ask_agent, repeats=a.repeats, idle=idle)

    if not idle():
        LOG.info("[selftune] owner active in the last %d min: skip", IDLE_MIN)
        return
    r = run_round(teacher=rt.teacher, learning_dir=rt.learning_dir, exam_with=exam_with, agent_base=AGENT_SYSTEM,
                  voice_base=voice_base, peak_blocked=peak_blocked)
    print(json.dumps({k: r.get(k) for k in ("decision", "reason", "why", "base", "cand", "after", "new_version")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
