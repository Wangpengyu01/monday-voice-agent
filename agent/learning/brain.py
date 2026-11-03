"""Decide who answers (memory / local model / teacher), and learn from the teacher's answers.

Flow per question:
  1. memory has a close, unexpired answer          -> say it (no model call)
  2. chat, comfort, persona, simple requests        -> local model (with memory context)
  3. facts, reasoning, anything time-sensitive, personal info worth remembering,
     a correction ("不对…"), or the local model is unsure -> teacher model
The teacher appends a memory card after its spoken answer:
     【记忆】{"remember": true, "q": "...", "a": "...", "days": 365, "facts": ["..."]}
which is cut out of the speech, then stored. Every teacher Q/A is also appended to a JSONL file in
chat format - training data for a future LoRA adapter for the local model.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from learning.memory import normalize

MEMO_MARK = "【记忆】"

_TIME_SENSITIVE = re.compile(r"今天|明天|昨天|现在|最近|最新|今年|天气|气温|新闻|股价|汇率|比分|比赛|票价|价格|多少钱|上映|发布")
_PERSONAL = re.compile(r"^我(?:的|是|在|住|叫|每天|每周|周[一二三四五六日天]|喜欢|不喜欢|讨厌|习惯|一般|通常|今年|下课|上课|上班|下班|宿舍|生日)"
                       r"|我.{0,6}(?:下课|上课|上班|下班|宿舍|生日|喜欢|过敏)")
_CORRECTION = re.compile(r"^(?:不对|错了|你说错了|说错了|不是这样|不是的|你搞错了|答错了)")
# a plain fact question ("世界上最长的河是哪条") the small model may get wrong
_FACTUAL = re.compile(r"是哪|哪条|哪个|哪座|哪年|哪一年|哪国|多少|多久|多远|多高|多大|多长|几[个年月岁号点]|谁(?:发明|写|是)|什么时候")
_ABOUT_YOU = re.compile(r"你(?!知道|知不知道|觉得)")  # "你几岁了" is about 星期一, answered by the persona
_UNSURE = re.compile(r"不确定|不太清楚|不知道|无法回答|没有相关信息|我查不到|没法确定|可能是|大概是")


@dataclass
class Route:
    who: str  # "memory" | "local" | "teacher"
    reason: str
    memory_answer: str = ""


def route(question: str, kind: str, *, teacher_ok: bool, memory_answer: str = "",
          correction: bool = False) -> Route:
    """kind is quick_reply.ack_kind(): request / lookup / explain / comfort / cheer / chat."""
    if correction and teacher_ok:
        return Route("teacher", "correction")
    if memory_answer and not _TIME_SENSITIVE.search(question):
        return Route("memory", "learned before", memory_answer)
    if not teacher_ok:
        return Route("local", "no teacher")
    if kind in ("comfort", "cheer"):
        return Route("local", kind)  # "我今天好累": answer the feeling, nothing to look up
    if _TIME_SENSITIVE.search(question):
        return Route("teacher", "time-sensitive")
    if _PERSONAL.search(question):
        return Route("teacher", "personal info to remember")
    if _ABOUT_YOU.search(question):
        return Route("local", "about 星期一")
    if kind in ("lookup", "explain"):
        return Route("teacher", kind)
    if kind == "chat" and _FACTUAL.search(question):
        return Route("teacher", "fact")
    return Route("local", kind)


def is_correction(text: str) -> bool:
    return bool(_CORRECTION.search((text or "").strip()))


def local_answer_unsure(answer: str) -> bool:
    return bool(_UNSURE.search(answer or ""))


def teacher_system(base_system: str, memory_context: str) -> str:
    memo_rules = (
        "回答完以后，另起一行输出一张记忆卡片，格式严格为：\n"
        f'{MEMO_MARK}{{"remember": true或false, "q": "标准问法", "a": "一句话答案", "days": 有效天数或null, "facts": ["关于主人的新信息"]}}\n'
        "规则：常识、知识类问题 remember 为 true，days 为 null（长期有效）；会变化的信息（价格、时刻表）给天数；"
        "天气、新闻、闲聊 remember 为 false。主人透露的个人信息（作息、住处、喜好、课表）写进 facts，没有就给空列表。"
        "卡片只给程序看，不会被念出来。"
    )
    parts = [base_system.strip()]
    if memory_context:
        parts.append(memory_context)
    parts.append(memo_rules)
    return "\n\n".join(parts)


class MemoFilter:
    """Pass spoken text through; hold back everything from the memo marker on."""

    def __init__(self) -> None:
        self._pending = ""
        self.memo = ""
        self._in_memo = False

    def feed(self, piece: str) -> str:
        if self._in_memo:
            self.memo += piece
            return ""
        buf = self._pending + piece
        i = buf.find(MEMO_MARK)
        if i >= 0:
            self._in_memo = True
            self.memo += buf[i + len(MEMO_MARK):]
            self._pending = ""
            return buf[:i]
        # keep a tail that could be the start of the marker (e.g. "【记")
        keep = 0
        for n in range(min(len(MEMO_MARK) - 1, len(buf)), 0, -1):
            if MEMO_MARK.startswith(buf[-n:]):
                keep = n
                break
        self._pending = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep] if keep else buf

    def finish(self) -> str:
        out, self._pending = self._pending, ""
        return "" if self._in_memo else out


def parse_memo(raw: str) -> dict:
    raw = (raw or "").strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        memo = json.loads(m.group(0))
    except ValueError:
        return {}
    return memo if isinstance(memo, dict) else {}


@dataclass
class Learner:
    store: object  # MemoryStore
    lora_log: Path
    learned: list[str] = field(default_factory=list)

    def learn(self, question: str, spoken: str, memo: dict, *, system: str, wrong: str = "",
              keep_if_no_card: bool = False) -> None:
        """Keep what the teacher taught: knowledge, facts about the user, the correction, training data."""
        if (wrong or keep_if_no_card) and not memo:
            # a correction or a plain fact question, but the teacher left out the memory card:
            # keep the answer it gave
            memo = dict(memo, remember=True, a=spoken.strip(), q=memo.get("q") or question)
        if memo.get("remember") and memo.get("a"):
            days = memo.get("days")
            days = float(days) if isinstance(days, (int, float)) and days > 0 else None
            canon = str(memo.get("q") or question)
            self.store.remember(canon, str(memo["a"]), days=days)
            if normalize(question) != normalize(canon):
                # also keep how the user actually put it ("珠峰多高" for "珠穆朗玛峰有多高")
                self.store.remember(question, str(memo["a"]), days=days)
            self.learned.append(str(memo.get("q") or question))
        for fact in memo.get("facts") or []:
            if isinstance(fact, str):
                self.store.add_fact(fact)
        if wrong:
            self.store.add_lesson(question, wrong, spoken)
        self.log_training(question, spoken, system=system)

    def log_training(self, question: str, answer: str, *, system: str) -> None:
        if not answer.strip():
            return
        self.lora_log.parent.mkdir(parents=True, exist_ok=True)
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer.strip()},
        ]}
        with self.lora_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
