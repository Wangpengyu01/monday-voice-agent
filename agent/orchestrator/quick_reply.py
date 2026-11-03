"""Instant replies for trivial utterances, and a short spoken ack before LLM answers.

Two ways to sound less like a machine waiting on a model:

1. quick_reply_for(): "你是谁", "几点了", "谢谢" ... don't need the LLM at all.
   Only whole short utterances match, so "你好，帮我查一下天气" still goes to the model.
   Time and date are computed locally (the model only parrots the prompt anyway).

2. ack_for(): for everything that does go to the LLM, react right away from a pre-synthesized
   clip ("好嘞，" / "我看看，" / "这个嘛，" / "哎呀，" ... depending on what was said), then the
   answer, like a person would.
   strip_leading_ack() removes the model's own "好的，主人。" so it isn't said twice.
"""
from __future__ import annotations

import datetime
import random
import re

from orchestrator.persona import CALL_NAME_RE, INTRODUCTION
from tts.number_cn import int_cn

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

_WAKE_PREFIX = re.compile(r"^(?:(?:嘿|嗨|喂|哈喽|你好)?" + CALL_NAME_RE + r"|主人)+")
_TRAIL = re.compile(r"[呀啊呢吧嘛哈哦啦]+$")
_PUNCT = re.compile(r"[\s，,。.！!？?、~～…]+")


_REPEAT = re.compile(r"(.)\1+")


def normalize_utterance(text: str) -> str:
    """Strip punctuation, wake words and trailing particles; collapse 谢谢 -> 谢.

    The ASR/chat path already collapses repeated characters (normalize_user_text),
    so tables and input are both collapsed here to match either form.
    """
    t = _PUNCT.sub("", text or "")
    t = _WAKE_PREFIX.sub("", t)
    t = _TRAIL.sub("", t)
    return _REPEAT.sub(r"\1", t)


def _period(hour: int) -> str:
    if hour < 5:
        return "凌晨"
    if hour < 9:
        return "早上"
    if hour < 12:
        return "上午"
    if hour == 12:
        return "中午"
    if hour < 18:
        return "下午"
    return "晚上"


def _clock_cn(now: datetime.datetime) -> str:
    h12 = now.hour % 12 or 12
    h = "两" if h12 == 2 else int_cn(h12)
    if now.minute == 0:
        m = "整"
    elif now.minute < 10:
        m = "零" + int_cn(now.minute) + "分"
    else:
        m = int_cn(now.minute) + "分"
    return f"{_period(now.hour)}{h}点{m}"


def _date_cn(now: datetime.datetime) -> str:
    return f"{int_cn(now.month)}月{int_cn(now.day)}日，{_WEEKDAYS[now.weekday()]}"


_GREETING = {"你好", "您好", "嗨", "哈喽", "早", "早上好", "早安", "上午好", "中午好", "下午好", "晚上好", "在吗你好"}
_WHO = {"你是谁", "你叫什么", "你叫啥", "你叫什么名字", "你是何方神圣", "你的名字", "你的名字是什么", "你是什么", "介绍一下你自己", "你介绍一下自己", "自我介绍一下"}
_THANKS = {"谢谢", "谢谢你", "谢谢小揽", "谢谢星期一", "多谢", "辛苦了", "辛苦你了", "感谢"}
_BYE = {"再见", "拜拜", "回头见", "我走了"}
_NIGHT = {"晚安", "我睡了", "我去睡觉了"}
_TIME = {"几点了", "现在几点", "现在几点了", "现在什么时间", "现在是什么时间", "几点钟了", "现在几点钟"}
_DATE = {"今天几号", "今天是几号", "今天几月几号", "今天是几月几号", "今天日期", "今天星期几", "今天是星期几", "今天周几", "今天礼拜几", "今天是什么日子"}
_SKILLS = {"你会什么", "你能做什么", "你会干什么", "你有什么功能", "你都会什么", "你能干什么", "你可以做什么"}


def _collapsed(words: set[str]) -> set[str]:
    return {_REPEAT.sub(r"\1", w) for w in words}


_GREETING, _WHO, _THANKS, _BYE, _NIGHT, _TIME, _DATE, _SKILLS = (
    _collapsed(x) for x in (_GREETING, _WHO, _THANKS, _BYE, _NIGHT, _TIME, _DATE, _SKILLS)
)


def quick_reply_for(text: str, now: datetime.datetime | None = None) -> str | None:
    u = normalize_utterance(text)
    if not u or len(u) > 10:
        return None
    now = now or datetime.datetime.now()
    if u in _WHO:
        return INTRODUCTION
    if u in _GREETING:
        return f"{_period(now.hour)}好，主人。"
    if u in _THANKS:
        return "不客气，主人。"
    if u in _BYE:
        return "好的，主人，回头见。"
    if u in _NIGHT:
        return "晚安，主人，好梦。"
    if u in _TIME:
        return f"现在是{_clock_cn(now)}。"
    if u in _DATE:
        return f"今天是{_date_cn(now)}。"
    if u in _SKILLS:
        return "我可以陪您聊天、回答问题、算算数、讲讲知识，有事尽管吩咐。"
    return None


# --- spoken ack before an LLM answer -------------------------------------
#
# People don't open every answer with the same "嗯": they react to what was said. Being asked to do
# something gets "好嘞", a factual question "我看看", a why/how question "这个嘛", a complaint
# "哎呀", good news "哇". Each kind has a few interchangeable phrases and the same phrase is not used
# twice in a row. Every phrase is pre-synthesized at start-up (orchestrator._warm_ack_cache).

ACK_GROUPS: dict[str, tuple[str, ...]] = {
    "request": ("好嘞，", "好的，", "没问题，", "行，"),
    "lookup": ("我看看，", "让我看看，", "我查一下，"),
    "explain": ("这个嘛，", "我想想，", "嗯，这个问题嘛，"),
    "comfort": ("哎呀，", "唉，"),
    "cheer": ("哇，", "真的呀，"),
    "chat": ("嗯，", "嗯嗯，"),
}
ACK_PHRASES: tuple[str, ...] = tuple(p for group in ACK_GROUPS.values() for p in group)

_REQUEST = re.compile(r"帮我|给我|请你|麻烦|能不能|可不可以|可以帮|讲个|讲一个|说个|说一个|推荐|来一个|来个|唱|打开|关掉|关闭|调大|调小|设置|设个|提醒我|播放|放一首|切换")
_EXPLAIN = re.compile(r"为什么|为啥|怎么回事|怎么样才|怎么做|怎么办|如何|原理|区别|你觉得|你认为|你怎么看|有什么建议|应该|值不值|好不好")
_LOOKUP = re.compile(r"多少|多远|多久|多长|多大|多高|几个|几点|几号|几岁|哪里|哪儿|哪个|什么时候|是什么|是谁|叫什么|天气|气温|价格|多少钱|查一下|搜一下|有没有")
_COMFORT = re.compile(r"好累|太累|累死|好烦|烦死|难过|伤心|不开心|不舒服|生病|头疼|郁闷|失眠|压力|倒霉|无聊|饿了|好饿|冷死|热死")
_CHEER = re.compile(r"开心|高兴|太好了|好消息|升职|加薪|考上|通过了|中奖|成功了|放假了|过生日")

_last_ack = ""


def ack_kind(text: str) -> str:
    u = normalize_utterance(text)
    # a complaint beats everything ("能不能安慰我一下，我好难过" -> 哎呀); a request beats good news
    # ("帮我放首开心的歌" -> 好嘞)
    if _COMFORT.search(u):
        return "comfort"
    if _REQUEST.search(u):
        return "request"
    if _CHEER.search(u):
        return "cheer"
    if _EXPLAIN.search(u):
        return "explain"
    if _LOOKUP.search(u):
        return "lookup"
    return "chat"


def ack_for(text: str, rng: random.Random | None = None) -> str:
    """Pick an ack that fits what was said; avoid repeating the previous one."""
    global _last_ack
    choices = [p for p in ACK_GROUPS[ack_kind(text)] if p != _last_ack] or list(ACK_GROUPS[ack_kind(text)])
    phrase = (rng or random).choice(choices)
    _last_ack = phrase
    return phrase


_LEADING_ACK = re.compile(
    r"^(?:好的|好嘞|好呀|好|嗯+|行|没问题|当然可以|当然|收到|让我看看|我看看|我想想|这个嘛)(?=[，,。！!～~\s]|主人)"  # not 好莱坞
    r"(?:[，,。！!～~\s]*主人)?[，,。！!～~\s]*"
)


def strip_leading_ack(chunk: str) -> str:
    """Drop the model's own opening "好的，主人。" after we already said an ack."""
    return _LEADING_ACK.sub("", chunk or "", count=1)
