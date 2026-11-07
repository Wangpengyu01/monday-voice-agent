"""Nightly self-reflection: 星期一 reviews its day with the teacher (DeepSeek) and learns from it.

  1. review   - the teacher grades every answer the local model gave on its own yesterday; every
                "wrong" gets a second, reasoned check (graders misjudge too), and the confirmed ones
                become a lesson + a corrected memory card (used from the next question on).
  2. quiz     - the teacher writes questions aimed at yesterday's weak spots, the local model
                answers them, the teacher grades; mistakes are learned the same way.
  3. diary    - the teacher writes a short first-person diary (what went wrong, what was learned,
                what to watch tomorrow) to run/learning/diary/YYYY-MM-DD.md; tomorrow's quiz reads
                its focus list.
  4. dataset  - teacher answers and verified corrections are merged, de-duplicated, into
                run/learning/lora_curated.jsonl: the training set for the LoRA adapter.

Only memory and files under run/learning are written; nothing else on the board is touched.
Runs from r1-reflect.timer (03:30); by hand:
    cd /userdata/agent && python3 -m learning.reflect [--day 2026-11-04] [--quiz 10] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from learning.journal import read_day
from learning.memory import MemoryStore, normalize

LOG = logging.getLogger("reflect")

GRADE_SYSTEM = (
    "你是严格又务实的老师，在批改家用语音助手「星期一」的回答。星期一是一个跑在板子上的四B小模型，"
    "回答会被直接念出来，所以简短口语是对的，不要因为简短扣分。\n"
    "逐条判断：事实、计算、常识、时间推理有错就是 wrong；回答回避了问题、答非所问、编造也是 wrong；"
    "闲聊、安慰、寒暄只要得体就是 ok；拿不准是 unsure。\n"
    "verdict 不是 ok 时，right 给出正确、适合直接朗读的一到两句中文口语回答（数字和单位用中文说法）。\n"
    "remember 表示正确答案值得长期记住（常识、知识）；会变化的信息给 days 天数，闲聊为 false。\n"
    '只输出 JSON：{"items": [{"i": 序号, "verdict": "ok|wrong|unsure", "right": "", "remember": true, '
    '"days": null, "why": "一句话原因"}]}'
)
QUIZ_SYSTEM = (
    "你在给家用语音助手「星期一」（四B小模型）出练习题，目标是找出并补上它的薄弱点。"
    "题目要像主人平时会随口问的话，一句话，中文。覆盖给出的薄弱点，再补一些常见的："
    "生活常识、单位换算、简单心算、日期推理、地理历史常识、安全常识。不要出需要实时信息（天气、新闻、价格）的题。"
    '只输出 JSON：{"questions": ["...", "..."]}'
)
VERIFY_SYSTEM = (
    "你在复核一位老师的批改，防止误判。每条给出：问题、学生（小模型）的原回答、老师认为的正确答案。"
    "先在 reason 里一步步推理出这道题真正的答案，再判断：原回答在意思上是否和正确答案一致（说法不同、更简短、"
    "数字写法不同都算一致）。只有原回答确实错误或答非所问时 really_wrong 才为 true；"
    "如果老师的正确答案本身有错，也给出修正后的 right。\n"
    '只输出 JSON：{"items": [{"i": 序号, "reason": "推理", "really_wrong": true, "right": "修正后的正确答案"}]}'
)
DIARY_SYSTEM = (
    "你是语音助手「星期一」，在写今天的学习日记，第一人称，像认真复盘的人。"
    "内容：今天和主人聊了什么方面、哪里答错了和为什么、学到了什么、明天要注意什么。"
    "三到六段短文，中文，不用表格。最后另起一行输出："
    'FOCUS: {"focus": ["明天练习的薄弱点1", "..."]}'
)


@dataclass
class Report:
    day: str
    reviewed: int = 0
    wrong: list[dict] = field(default_factory=list)
    quizzed: int = 0
    quiz_wrong: list[dict] = field(default_factory=list)
    dataset_size: int = 0
    diary: str = ""
    errors: list[str] = field(default_factory=list)


def _json(text: str) -> dict:
    text = (text or "").strip()
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return {}
    try:
        obj = json.loads(text[i:j + 1])
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def grade(teacher, pairs: list[dict], batch: int = 12) -> list[dict]:
    """pairs: [{"q", "a"}] -> same list with verdict/right/remember/days/why added."""
    out: list[dict] = []
    for k in range(0, len(pairs), batch):
        chunk = pairs[k:k + batch]
        listing = "\n".join(f"{i}. 问：{p['q']}\n   答：{p['a']}" for i, p in enumerate(chunk))
        res = _json(teacher.complete(GRADE_SYSTEM, f"今天是{datetime.date.today().isoformat()}。请批改：\n{listing}",
                                     json_mode=True, max_tokens=2000))
        verdicts = {int(it.get("i", -1)): it for it in res.get("items", []) if isinstance(it, dict)}
        for i, p in enumerate(chunk):
            v = verdicts.get(i, {})
            out.append({**p, "verdict": str(v.get("verdict", "unsure")), "right": str(v.get("right") or "").strip(),
                        "remember": bool(v.get("remember")), "days": v.get("days"), "why": str(v.get("why") or "")})
    return out


def verify(teacher, wrong: list[dict]) -> list[dict]:
    """Second look at every "wrong": keep only the ones that really are wrong (graders misjudge too)."""
    if not wrong:
        return []
    listing = "\n".join(f"{i}. 问：{w['q']}\n   原回答：{w['a']}\n   老师的正确答案：{w['right']}"
                        for i, w in enumerate(wrong))
    res = _json(teacher.complete(VERIFY_SYSTEM, listing, json_mode=True, max_tokens=2500))
    checked = {int(it.get("i", -1)): it for it in res.get("items", []) if isinstance(it, dict)}
    kept = []
    for i, w in enumerate(wrong):
        c = checked.get(i)
        if c is None or not c.get("really_wrong"):
            LOG.info("[reflect] verify: not wrong after all: %s | %s", w["q"], w["a"][:60])
            continue
        if str(c.get("right") or "").strip():
            w = {**w, "right": str(c["right"]).strip()}
        kept.append(w)
    return kept


def learn_from(store: MemoryStore, graded: list[dict], *, dry_run: bool, teacher=None) -> list[dict]:
    """Store every (verified) wrong answer's correction; return the wrong ones."""
    wrong = [g for g in graded if g["verdict"] == "wrong" and g["right"]]
    if teacher is not None:
        wrong = verify(teacher, wrong)
    if dry_run:
        return wrong
    for g in wrong:
        store.add_lesson(g["q"], g["a"], g["right"])
        days = g["days"] if isinstance(g["days"], (int, float)) and g["days"] > 0 else None
        if g["remember"] or days:
            store.remember(g["q"], g["right"], days=days, source="reflect")
    return wrong


def build_dataset(learning_dir: Path, system: str, corrections: list[dict]) -> int:
    """lora_data.jsonl (teacher answers) + reviewed corrections + previous curated set, latest wins per question."""
    curated = learning_dir / "lora_curated.jsonl"
    by_q: dict[str, dict] = {}

    def add(rec: dict) -> None:
        msgs = rec.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        reply = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        if user.strip() and reply.strip():
            by_q[normalize(user)] = rec

    for src in (curated, learning_dir / "lora_data.jsonl"):
        if src.is_file():
            for line in src.read_text(encoding="utf-8").splitlines():
                try:
                    add(json.loads(line))
                except (ValueError, TypeError, KeyError):
                    continue
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for c in corrections:
        add({"time": now, "source": "reflect", "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": c["q"]},
            {"role": "assistant", "content": c["right"]},
        ]})
    tmp = curated.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in by_q.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(curated)
    return len(by_q)


def reflect(*, day: datetime.date, teacher, store: MemoryStore, learning_dir: Path, journal_dir: Path,
            local_answer: Callable[[str], str], system: str, n_quiz: int = 10, max_review: int = 80,
            dry_run: bool = False) -> Report:
    rep = Report(day=day.isoformat())
    # 1. review what the local model said on its own
    turns = [t for t in read_day(journal_dir, day) if t.get("who") == "local"][-max_review:]
    if turns:
        try:
            graded = grade(teacher, [{"q": t["q"], "a": t["a"]} for t in turns])
            rep.reviewed = len(graded)
            rep.wrong = learn_from(store, graded, dry_run=dry_run, teacher=teacher)
        except Exception as exc:  # noqa: BLE001 - one failing step must not stop the others
            rep.errors.append(f"review: {exc}")
    # 2. quiz on weak spots
    focus_file = learning_dir / "focus.json"
    focus = []
    if focus_file.is_file():
        try:
            focus = json.loads(focus_file.read_text(encoding="utf-8")).get("focus", [])
        except ValueError:
            focus = []
    weak = [f"{w['q']}（答成了：{w['a']}）" for w in rep.wrong] + [str(f) for f in focus]
    if n_quiz > 0:
        try:
            res = _json(teacher.complete(QUIZ_SYSTEM, f"薄弱点：{'；'.join(weak) or '暂无，出通用题'}。出 {n_quiz} 道题。",
                                         json_mode=True, max_tokens=1200))
            questions = [str(q).strip() for q in res.get("questions", []) if str(q).strip()][:n_quiz]
            pairs = []
            for q in questions:
                try:
                    pairs.append({"q": q, "a": local_answer(q) or "（没有回答）"})
                except Exception as exc:  # noqa: BLE001
                    rep.errors.append(f"local answer: {exc}")
                    break
            if pairs:
                graded = grade(teacher, pairs)
                rep.quizzed = len(graded)
                rep.quiz_wrong = learn_from(store, graded, dry_run=dry_run, teacher=teacher)
        except Exception as exc:  # noqa: BLE001
            rep.errors.append(f"quiz: {exc}")
    # 3. diary + tomorrow's focus
    try:
        mistakes = "\n".join(f"- 问：{w['q']}；我答：{w['a']}；正确：{w['right']}；原因：{w['why']}"
                             for w in rep.wrong + rep.quiz_wrong) or "- 没有发现错误"
        topics = "；".join(t["q"] for t in read_day(journal_dir, day)[-40:])
        topics = f"今天主人问过：{topics}" if topics else "今天主人没有和我对话，只做了自我练习"
        text = teacher.complete(DIARY_SYSTEM, f"日期：{rep.day}\n{topics}\n"
                                f"复盘了 {rep.reviewed} 条自己的回答，练习了 {rep.quizzed} 道题。\n"
                                f"经过复核确认的错误（只写这些，不要编造其他错误或改判）：\n{mistakes}",
                                max_tokens=1200)
        body, _, tail = text.partition("FOCUS:")
        rep.diary = body.strip()
        new_focus = _json(tail).get("focus", [])
        if not dry_run:
            diary_dir = learning_dir / "diary"
            diary_dir.mkdir(parents=True, exist_ok=True)
            stats = (f"> 复盘 {rep.reviewed} 条 · 答错 {len(rep.wrong)} 条 · 练习 {rep.quizzed} 题 · "
                     f"练习答错 {len(rep.quiz_wrong)} 题")
            (diary_dir / f"{rep.day}.md").write_text(f"# 星期一的学习日记 {rep.day}\n\n{stats}\n\n{rep.diary}\n", encoding="utf-8")
            if new_focus:
                focus_file.write_text(json.dumps({"day": rep.day, "focus": new_focus}, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        rep.errors.append(f"diary: {exc}")
    # 4. training set
    if not dry_run:
        try:
            rep.dataset_size = build_dataset(learning_dir, system, rep.wrong + rep.quiz_wrong)
        except OSError as exc:
            rep.errors.append(f"dataset: {exc}")
    return rep


@dataclass
class Runtime:
    teacher: object
    store: MemoryStore
    learning_dir: Path
    journal_dir: Path
    system: str
    local_answer: Callable[[str], str]


def build_runtime(config_path: str) -> Runtime:
    """Teacher, memory and the local model, set up the way the assistant itself uses them."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from llm.client import llm_chat
    from learning.teacher import DeepSeekTeacher
    from orchestrator.config_loader import load_yaml
    from orchestrator.user_title import DEFAULT_TITLE, apply_title
    from tts.sentence_split import clean_llm_reply

    cfg = load_yaml(config_path)
    tcfg, lcfg = cfg.get("teacher") or {}, cfg.get("learning") or {}
    teacher = DeepSeekTeacher(base_url=tcfg.get("base_url", "https://api.deepseek.com"), model=tcfg.get("model", "deepseek-flash"),
                              api_key_env=tcfg.get("api_key_env", "DEEPSEEK_API_KEY"), timeout_s=float(tcfg.get("timeout_s", 20)))
    if not teacher.available:
        raise SystemExit("no teacher key (DEEPSEEK_API_KEY): nothing to learn from")
    db = Path(lcfg.get("db", "/userdata/agent/run/learning/memory.db"))
    store = MemoryStore(db)
    system = apply_title((cfg.get("system_prompt") or "").strip(), DEFAULT_TITLE, for_prompt=True)
    sock = str(cfg.get("socket_path", "/tmp/r1-llm.sock"))
    max_new = int(cfg.get("max_new_tokens", 200))

    def local_answer(q: str) -> str:
        ctx = store.context_for(q)
        sys_text = f"{system}\n\n{ctx}" if ctx else system
        prompt = f"<|im_start|>system\n{sys_text}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        return clean_llm_reply(q, str(llm_chat(prompt, sock_path=sock, max_new_tokens=max_new).get("text", "")))

    return Runtime(teacher, store, db.parent, Path(lcfg.get("journal_dir", str(db.parent / "turns"))), system, local_answer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    ap.add_argument("--day", help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--quiz", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="grade and print, write nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rt = build_runtime(args.config)
    day = datetime.date.fromisoformat(args.day) if args.day else datetime.date.today() - datetime.timedelta(days=1)
    rep = reflect(day=day, teacher=rt.teacher, store=rt.store, learning_dir=rt.learning_dir, journal_dir=rt.journal_dir,
                  local_answer=rt.local_answer, system=rt.system, n_quiz=args.quiz, dry_run=args.dry_run)
    LOG.info("[reflect] %s reviewed=%d wrong=%d quizzed=%d quiz_wrong=%d dataset=%d errors=%s", rep.day, rep.reviewed,
             len(rep.wrong), rep.quizzed, len(rep.quiz_wrong), rep.dataset_size, rep.errors or "none")
    for w in rep.wrong + rep.quiz_wrong:
        LOG.info("[reflect] fixed: %s | was: %s | now: %s", w["q"], w["a"][:60], w["right"][:60])
    if args.dry_run and rep.diary:
        print(rep.diary)


if __name__ == "__main__":
    main()
