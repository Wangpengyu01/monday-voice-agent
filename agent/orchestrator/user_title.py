"""How the assistant addresses the user ("主人" by default, changeable: "以后叫我王哥").

Strings across the code base say "主人"; instead of threading a variable through every one of
them, apply_title() swaps it at the few places text leaves the agent (TTS, UI events, the LLM
system prompt). parse_title_command() recognizes spoken requests to change it.
"""
from __future__ import annotations

import re

DEFAULT_TITLE = "主人"

# "主人公" / "主人翁" / "主人家" are ordinary words, not the form of address
_TITLE_WORD = re.compile(r"主人(?![公翁家])")

_COMMAND = re.compile(
    r"^(?:小揽[，,\s]*)?(?:你)?(?:以后|今后|从现在开始|从今天起|往后)?[，,\s]*(?:就|都)?(?:可以|要)?"
    r"(?:叫我|喊我|称呼我|管我叫)(?P<title>.+?)(?:吧|就行|就好|好了|可以了|行吗|好吗|呗)?[。！!？?，,\s]*$"
)
# "叫我起床" / "叫我一声" are not forms of address
_NOT_TITLE = re.compile(
    r"小揽|什么|啥|名字|干嘛|怎么|起床|起来|吃饭|睡觉|回家|出门|上班|下班|过去|过来|一声|一下|去|做"
)
_PUNCT = re.compile(r"[\s，,。.！!？?、~～…\"“”'‘’]+")


def clean_title(raw: str) -> str | None:
    t = _PUNCT.sub("", raw or "")
    t = re.sub(r"(?:吧|呀|啊|呢|哦|就行|就好)$", "", t)
    if not t or len(t) > 8 or _NOT_TITLE.search(t):
        return None
    return t


def parse_title_command(text: str) -> str | None:
    """'以后叫我王哥' -> '王哥'; '' (no address) for '别叫我主人了'; None if not a title request."""
    u = (text or "").strip()
    if re.search(r"(?:别|不要|不用)(?:再)?(?:叫我|喊我|称呼我)", u):
        m = re.search(r"(?:别|不要|不用)(?:再)?(?:叫我|喊我|称呼我)(?P<old>[^，,。.！!？?]*)", u)
        rest = u[m.end():] if m else ""
        m2 = re.search(r"(?:叫我|喊我|称呼我)(?P<title>[^，,。.！!？?]+)", rest)
        if m2:
            return clean_title(m2.group("title"))
        return ""
    m = _COMMAND.match(u)
    if not m:
        return None
    return clean_title(m.group("title"))


_DROP_ADDRESS = re.compile(r"[，,]?\s*主人(?![公翁家])\s*[，,]?")


def apply_title(text: str, title: str | None, *, for_prompt: bool = False) -> str:
    """Swap "主人" for the user's title. title == "" means: no form of address at all."""
    if not text or title is None or title == DEFAULT_TITLE:
        return text
    if title:
        return _TITLE_WORD.sub(title, text)
    if for_prompt:
        text = text.replace("称呼对方为主人", "不用特别的称呼")
        return _TITLE_WORD.sub("对方", text)

    # spoken text: drop the address and keep the sentence punctuation sensible
    def drop(m: re.Match[str]) -> str:
        s = m.group(0)
        start, end = m.start(), m.end()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if start == 0 or before in "。！？!?":
            return ""  # "主人，现在…" -> "现在…"
        if after in "。！？!?" or end == len(text):
            return ""  # "好的，主人。" -> "好的。"
        return "，" if ("，" in s or "," in s) else ""

    return _DROP_ADDRESS.sub(drop, text)
