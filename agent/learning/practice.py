"""Quiet practice: while nobody is talking to it, 星期一 keeps hunting for what its small local model gets
wrong, and learns it.

Every run (one practice round of the study loop, learning/study_loop.py; or --slot by hand): rounds rotate
knowledge -> speaking -> coding -> agent. Coding and agent rounds are hands-on drills graded by results (learning.drill); the steps below are
the knowledge and speaking slots:
  1. idle?   - skip unless the owner has been quiet for IDLE_MIN minutes (run/last_activity) and
               the local model is free; re-checked before every question, so speaking to it wins.
  2. ask     - the teacher writes questions aimed at what a 4B model typically fails (a rotating
               area, plus recent mistakes and tomorrow's focus), never repeating recent ones.
  3. answer  - the local model answers, exactly as it would in a conversation.
  4. learn   - the teacher grades, a second pass verifies every "wrong"; confirmed mistakes become
               lessons + corrected memory cards (used from the next conversation on) and LoRA data.
  5. speak   - every reply is read aloud, so an answer that is right but unspeakable (commands,
               symbols, code) or too long is rewritten by the teacher into short spoken Chinese and
               added to the LoRA data too (no lesson: the content was fine).
Nothing is spoken; one summary line per run goes to run/learning/practice.jsonl.

    cd /userdata/agent && python3 -m learning.practice [--n 10] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from learning.reflect import _json, build_dataset, grade, learn_from
from learning import now as now_status

LOG = logging.getLogger("practice")

IDLE_MIN = 5   # minutes without a real sentence from the owner before background study runs
SLOT_S = 300          # one run per timer slot
MODES = ("knowledge", "speaking", "coding", "agent")  # slot % 4
MAX_SPOKEN = 120      # a spoken reply longer than this (characters) gets a shorter version
AREAS = [
    "多步心算和应用题（买东西找零、打折、速度路程时间）",
    "日期和星期推算（过几天是几号星期几、闰年、月底跨月）",
    "单位换算（长度、面积、重量、温度、容积、时间）",
    "时间计算（几点到几点多久、跨午夜、时区）",
    "中国地理和历史常识",
    "生活和安全常识（用电、燃气、急救常识，不做诊断）",
    "科学常识（物理、化学、生物、天文的基本事实）",
    "成语俗语的意思和用法",
    "容易想当然的逻辑陷阱题",
    "数字和数量的比较与排序",
]
PRACTICE_SYSTEM = (
    "你在帮语音助手「星期一」查漏补缺。它的本地大脑是一个四B小模型，常见弱点：多步计算出错、日期星期推错、"
    "单位换算出错、记错冷门事实、被题目里的陷阱带偏、把不知道的编出来。\n"
    "请出题专门考这些弱点：题目要像主人日常随口会问的话，一句话，中文，答案明确、能用一两句话说清；"
    "不要实时信息（天气、新闻、价格），不要开放式聊天题，不要和最近问过的重复。\n"
    '只输出 JSON：{"questions": ["...", "..."]}'
)
SPOKEN_AREAS = [
    "电脑和板子的运维（看磁盘、内存、进程、网络、服务、日志），主人只想听结论或一句话的做法",
    "手机电脑和家里设备怎么操作、怎么设置",
    "用口语解释一个技术概念（接口、端口、进程、内存、编码、协议等）",
    "主人的日常安排、提醒、出门准备",
    "主人情绪不好、累了、烦了时随口说的话",
    "比较和推荐（两样东西怎么选、哪个更好）",
    "怎么做一件事的步骤（做饭、学习方法、办事流程）",
    "关于你自己：你是谁、你能做什么、你跑在什么硬件上",
]
SPOKEN_PRACTICE_SYSTEM = (
    "你在帮语音助手「星期一」练说话。它的回答会被直接朗读出来，本地大脑是一个四B小模型，常见毛病："
    "把命令、代码、路径、符号原样念出来，回答太长，列一堆要点，开场白和客套太多，不知道的编出来。\n"
    "请出题专门引出这些毛病：题目要像主人日常随口会说的话，一句话，中文；"
    "可以是怎么做、为什么、帮我想想、陪我聊两句这类问题；不要实时信息（天气、新闻、价格），不要和最近问过的重复。\n"
    '只输出 JSON：{"questions": ["...", "..."]}'
)


def needs_spoken(text: str) -> bool:
    from learning.lora_prep import speakable
    return not speakable(text) or len(text) > MAX_SPOKEN


def spoken_versions(teacher, items: list[dict]) -> list[dict]:
    """items: [{"q", "a"}] whose answer is fine in content but bad to read aloud -> [{"q", "a", "right"}]."""
    from learning.lora_prep import rewrite, speakable
    out = []
    for it in items:
        try:
            spoken = rewrite(teacher, it["q"], it["a"])
        except Exception as exc:  # noqa: BLE001 - one failed call just skips that sample
            LOG.warning("[practice] spoken rewrite failed: %s", exc)
            continue
        if spoken and speakable(spoken):
            out.append({**it, "right": spoken})
    return out


def last_activity(run_dir: Path) -> float:
    try:
        return (run_dir / "last_activity").stat().st_mtime
    except OSError:
        return 0.0


def recent_questions(log: Path, n: int = 40) -> list[str]:
    if not log.is_file():
        return []
    qs: list[str] = []
    for line in log.read_text(encoding="utf-8").splitlines()[-30:]:
        try:
            qs += json.loads(line).get("questions", [])
        except ValueError:
            continue
    return qs[-n:]


def practice(*, teacher, store, learning_dir: Path, local_answer: Callable[[str], str], system: str,
             idle: Callable[[], bool], n: int = 10, now: float | None = None,
             student: Callable[[str], str] | None = None, local_plan: Callable[[str], str] | None = None,
             slot: int | None = None) -> dict:
    now = now or time.time()
    log = learning_dir / "practice.jsonl"
    slot = int(now // SLOT_S) if slot is None else slot   # the study loop counts rounds; the timer used the clock
    mode = MODES[slot % len(MODES)]
    if mode == "coding" and student is not None:
        from learning.drill import coding_drill
        now_status.begin(learning_dir, "practice", "编程真考", "老师出题，代码在沙箱里真跑")
        return coding_drill(teacher=teacher, learning_dir=learning_dir, student=student, idle=idle)
    if mode == "agent" and local_plan is not None:
        from learning.drill import agent_drill
        now_status.begin(learning_dir, "practice", "动手办事真考", "自己一步步执行只读命令")
        return agent_drill(teacher=teacher, learning_dir=learning_dir, local_plan=local_plan, idle=idle)
    speaking = mode == "speaking"
    areas = SPOKEN_AREAS if speaking else AREAS
    area = areas[(slot // len(MODES)) % len(areas)]  # a different area every run
    now_status.begin(learning_dir, "practice", "说话练习" if speaking else "知识练习", area)
    now_status.step("老师在出题")
    weak = [f"{q}（答成：{w}；正确：{r}）" for q, w, r in store.lessons(limit=5)]
    focus_file = learning_dir / "focus.json"
    if focus_file.is_file():
        try:
            weak += [str(f) for f in json.loads(focus_file.read_text(encoding="utf-8")).get("focus", [])]
        except ValueError:
            pass
    asked_before = recent_questions(log)
    res = _json(teacher.complete(SPOKEN_PRACTICE_SYSTEM if speaking else PRACTICE_SYSTEM,
                                 f"这次的重点：{area}。\n最近答错过的：{'；'.join(weak) or '无'}。\n"
                                 f"最近问过（不要重复）：{'；'.join(asked_before[-20:]) or '无'}。\n出 {n} 道题。",
                                 json_mode=True, max_tokens=1200))
    questions = [str(q).strip() for q in res.get("questions", []) if str(q).strip() and str(q).strip() not in asked_before][:n]
    pairs, stopped = [], False
    for i, q in enumerate(questions, 1):
        now_status.step(f"第 {i}/{len(questions)} 题：{q[:40]}")
        if not idle():
            stopped = True  # the owner started talking: give the local model back
            break
        pairs.append({"q": q, "a": local_answer(q) or "（没有回答）"})
    wrong = learn_from(store, grade(teacher, pairs), dry_run=False, teacher=teacher) if pairs else []
    wrong_q = {w["q"] for w in wrong}
    # right answers that read badly, plus corrections that themselves read badly: teach the spoken form
    clumsy = [p for p in pairs if p["q"] not in wrong_q and needs_spoken(p["a"])]
    clumsy += [{"q": w["q"], "a": w["right"]} for w in wrong if needs_spoken(w["right"])]
    spoken = spoken_versions(teacher, clumsy)
    by_q = {w["q"]: w for w in wrong}
    by_q.update({s["q"]: s for s in spoken})
    corrections = list(by_q.values())
    size = build_dataset(learning_dir, system, corrections) if corrections else 0
    summary = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), "area": area,
               "mode": "speaking" if speaking else "knowledge",
               "questions": [p["q"] for p in pairs], "wrong": len(wrong), "spoken": len(spoken), "stopped": stopped,
               "fixed": [{"q": w["q"], "was": w["a"][:80], "now": w["right"][:120]} for w in wrong]}
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    if size:
        summary["dataset"] = size
    return summary


def local_drillers(config_path: str) -> tuple[Callable[[str], str], Callable[[str], str]]:
    """The local model as a coder (CODING_SYSTEM) and as an agent planner (drill.AGENT_SYSTEM)."""
    from learning.drill import AGENT_SYSTEM
    from learning.school import CODING_SYSTEM, STUDENT_MAX_TOKENS
    from llm.client import llm_chat
    from orchestrator.config_loader import load_yaml
    sock = str(load_yaml(config_path).get("socket_path", "/tmp/r1-llm.sock"))

    def ask(system: str, user: str, max_new: int) -> str:
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        return str(llm_chat(prompt, sock_path=sock, max_new_tokens=max_new).get("text", ""))

    return (lambda q: ask(CODING_SYSTEM, q[:900], STUDENT_MAX_TOKENS)), (lambda ctx: ask(AGENT_SYSTEM, ctx, 120))


def main() -> None:
    from learning.reflect import build_runtime

    ap = argparse.ArgumentParser(description="星期一 quiet practice")
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="run even if the owner was active recently")
    ap.add_argument("--slot", type=int, help="practice round number (picks the mode; default: from the clock)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rt = build_runtime(args.config)
    run_dir = rt.learning_dir.parent

    def idle() -> bool:
        return args.force or time.time() - last_activity(run_dir) > IDLE_MIN * 60

    if not idle():
        LOG.info("[practice] owner active in the last %d min: skip", IDLE_MIN)
        return
    student, local_plan = local_drillers(args.config)
    try:
        s = practice(teacher=rt.teacher, store=rt.store, learning_dir=rt.learning_dir, local_answer=rt.local_answer,
                     system=rt.system, idle=idle, n=args.n, student=student, local_plan=local_plan, slot=args.slot)
    finally:
        now_status.end(rt.learning_dir)
    if s["mode"] in ("coding", "agent"):
        LOG.info("[practice] %s drill: passed %d/%d%s", s["mode"], s["passed"], s["total"], " (stopped: owner busy)" if s["stopped"] else "")
        for r in s["results"]:
            LOG.info("[practice] %s%s %s | %s", "PASS" if r["passed"] else "FAIL", " (复测)" if r.get("retest") else "",
                     r["task"][:60], r.get("why", "")[:80])
        return
    LOG.info("[practice] %s %s: asked %d, wrong %d, spoken %d%s", s["mode"], s["area"], len(s["questions"]), s["wrong"],
             s["spoken"], " (stopped: owner busy)" if s["stopped"] else "")
    for f in s["fixed"]:
        LOG.info("[practice] fixed: %s | was: %s | now: %s", f["q"], f["was"][:60], f["now"][:60])


if __name__ == "__main__":
    main()
