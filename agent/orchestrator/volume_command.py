"""Spoken volume commands ("音量调到三十", "大点声", "小一点", "现在音量多少").

Handled by the orchestrator itself (the same setting as the web page's speaker slider), never by the
agent: on the ES8388 board the ALSA 'PCM' control is the *capture* side, so poking the mixer does not
change how loud 星期一 speaks.
"""
from __future__ import annotations

import re

STEP = 15
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "幺": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

_ABOUT = re.compile(r"音量|声音|声儿|大声|小声|大点声|小点声|轻点声|吵")
_ASK = re.compile(r"(音量|声音).{0,4}(多少|多大|几)|现在.{0,3}(音量|声音)")
_SET = re.compile(r"(?:音量|声音)(?:调|设|开|放|改)?(?:到|成|为|在)?\s*(?:百分之)?([0-9一二两三四五六七八九十百零〇幺]+)\s*(?:%|％)?")
_MAX = re.compile(r"(音量|声音).{0,4}(最大|开满|拉满)|最大声")
_MIN = re.compile(r"(音量|声音).{0,4}(最小)|最小声")
_HALF = re.compile(r"(音量|声音).{0,4}(一半|中间)")
_UP = re.compile(r"(大|响|高)(一)?(点|些)|调大|开大|加大|提高|太小|听不(见|清)|大声")
_DOWN = re.compile(r"(小|轻|低)(一)?(点|些)|调小|关小|降低|太大|太吵|吵|小声|轻声")
_NOT_COMMAND = re.compile(r"为什么|原理|怎么产生|传播|速度|分贝是")


def cn_number(text: str) -> int | None:
    """'30' / '三十' / '三十五' / '一百' / '五' -> int."""
    if text.isdigit():
        return int(text)
    if text == "百" or text == "一百":
        return 100
    total, cur = 0, 0
    for ch in text:
        if ch in _CN_DIGIT:
            cur = _CN_DIGIT[ch]
        elif ch == "十":
            total += (cur or 1) * 10
            cur = 0
        elif ch == "百":
            total += (cur or 1) * 100
            cur = 0
        else:
            return None
    return total + cur


def parse_volume_command(text: str, current: int) -> tuple[int | None, str] | None:
    """-> (new percent or None for a query, reply) when this is a volume command, else None."""
    t = re.sub(r"\s+", "", text or "")
    if not t or len(t) > 24 or not _ABOUT.search(t) or _NOT_COMMAND.search(t):
        return None
    if _ASK.search(t) and not re.search(r"调|设|改|开大|关小|[大小响轻](一)?[点些]", t):
        return None, f"现在音量是百分之{current}。"
    if _MAX.search(t):
        target = 100
    elif _MIN.search(t):
        target = 10
    elif _HALF.search(t):
        target = 50
    elif (m := _SET.search(t)) and (n := cn_number(m.group(1))) is not None:
        target = n
    elif _UP.search(t):
        target = current + STEP
    elif _DOWN.search(t):
        target = current - STEP
    else:
        return None
    target = max(5, min(100, int(target)))  # never fully silent by voice: she must still be able to answer
    if target == current:
        return target, f"音量已经是百分之{current}了。"
    return target, f"好，音量调到百分之{target}了。"
