"""Read Arabic numerals the way people say them in Chinese, for TTS.

sanitize_tts_text used to map digits one by one, so the model's
"2026年11月2日 下午1点32分" was spoken as "二零二六年一一月二日 下午一点三二分",
"3.5公里" as "三五公里" and "50%" lost its meaning. Rules here:

- year before 年 (3-4 digits): digit by digit        2026年 -> 二零二六年
- time H:MM                    : X点Y分              13:05 -> 十三点零五分
- percent                      : 百分之...           12.5% -> 百分之十二点五
- decimal                      : integer + 点 + digits 3.14 -> 三点一四
- 5+ digits or leading 0       : digit by digit (phone numbers, codes)
- other integers               : quantity            11 -> 十一, 105 -> 一百零五
"""
from __future__ import annotations

import re

_DIGITS = "零一二三四五六七八九"
_SMALL_UNITS = ["", "十", "百", "千"]
_BIG_UNITS = ["", "万", "亿", "万亿"]


def digits_cn(s: str) -> str:
    return "".join(_DIGITS[int(c)] for c in s if c.isdigit())


def _four(n: int) -> str:
    """0 < n < 10000 without the 两 rule applied."""
    out = ""
    zero = False
    for pos in range(3, -1, -1):
        d = (n // 10**pos) % 10
        if d == 0:
            if out:
                zero = True
            continue
        if zero:
            out += "零"
            zero = False
        out += _DIGITS[d] + _SMALL_UNITS[pos]
    return out


def int_cn(n: int) -> str:
    if n == 0:
        return "零"
    if n < 0:
        return "负" + int_cn(-n)
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    out = ""
    pending_zero = False
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            pending_zero = bool(out)
            continue
        if out and (pending_zero or g < 1000):
            out += "零"
        out += _four(g) + _BIG_UNITS[i]
        pending_zero = False
    if out.startswith("一十"):
        out = out[1:]  # 十一, not 一十一
    # 两百 / 两千 / 两万 sound natural; keep 二 for 二十 and digits.
    out = re.sub(r"二(?=[百千万亿])", "两", out)
    return out


def _decimal_cn(int_part: str, frac: str) -> str:
    return int_cn(int(int_part)) + "点" + digits_cn(frac)


def _integer_token(tok: str) -> str:
    if len(tok) >= 5 or (len(tok) > 1 and tok.startswith("0")):
        return digits_cn(tok)
    return int_cn(int(tok))


_YEAR = re.compile(r"(?<!\d)(\d{3,4})(?=\s*年)")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:：]([0-5]\d)(?!\d)")
_PERCENT = re.compile(r"(?<![\d.])(\d+)(?:\.(\d+))?\s*[%％]")
_DECIMAL = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])")
_THOUSANDS = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+)(?!\d)")
_INTEGER = re.compile(r"\d+")
_LIANG_MEASURES = set("个点次天位本只件种条岁倍张台辆层名部句周对双年米斤克吨秒")
# multi-char units where a bare 2 is said 两: 两公里 两小时 两分钟 两公斤
_LIANG_UNITS = ("公里", "千米", "小时", "分钟", "公斤", "千克", "毫米", "厘米", "平方", "立方", "星期", "礼拜")
# 2 before a magnitude: 两千 两万 两百 两亿 (2千 used to come out as 二千)
_LIANG_MAGNITUDES = set("百千万亿")


def verbalize_numbers(text: str) -> str:
    if not text or not any(c.isdigit() for c in text):
        return text
    text = _THOUSANDS.sub(lambda m: m.group(1).replace(",", ""), text)
    text = _YEAR.sub(lambda m: digits_cn(m.group(1)), text)

    def time_repl(m: re.Match[str]) -> str:
        h, mm = int(m.group(1)), m.group(2)
        minute = "整" if mm == "00" else (("零" + _DIGITS[int(mm[1])]) if mm[0] == "0" else int_cn(int(mm))) + "分"
        return int_cn(h) + "点" + minute

    text = _TIME.sub(time_repl, text)
    text = _PERCENT.sub(
        lambda m: "百分之" + (_decimal_cn(m.group(1), m.group(2)) if m.group(2) else int_cn(int(m.group(1)))),
        text,
    )
    text = _DECIMAL.sub(lambda m: _decimal_cn(m.group(1), m.group(2)), text)

    def int_repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        prev = text[m.start() - 1 : m.start()] if m.start() else ""
        nxt = text[m.end() : m.end() + 2]
        if tok == "2" and prev != "第" and (
            nxt[:1] in _LIANG_MEASURES or nxt[:1] in _LIANG_MAGNITUDES or nxt.startswith(_LIANG_UNITS)
        ):
            return "两"  # 两个 / 两点 / 两千 / 两公里
        return _integer_token(tok)

    text = _INTEGER.sub(int_repl, text)
    return text
