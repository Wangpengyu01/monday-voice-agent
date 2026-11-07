"""Understand what the owner wants before deciding who handles it ("先分类").

The teacher (DeepSeek) reads the last few turns plus what was just said and returns
    intent:  task | knowledge | ask_teacher | chat | unclear
    rewrite: the request as one self-contained sentence - "这个/刚才" resolved from the recent turns,
             speech-recognition slips fixed ("SH" -> "SSH", "一零点零点零点幺五" -> "10.0.0.15").
Then: task -> the agent; knowledge / ask_teacher -> the teacher (with the recent turns); chat -> the
local model; unclear -> "没听清" (only for real garble). Without the teacher (offline, or the call
fails or is slow), fallback() keeps the old keyword rules so the assistant still works.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from learning.agent import is_agent_task

LOG = logging.getLogger(__name__)

KINDS = ("task", "knowledge", "ask_teacher", "chat", "unclear")
INTENT_SYSTEM = (
    "你是语音助手「星期一」的调度员。星期一跑在一块 Linux 板子上，能自己动手（执行命令、SSH 到别的机器、扫描网络、"
    "装软件、整理文件、在服务器上训练模型），也能问老师（你这样的大模型），还有一个本地小模型负责闲聊。\n"
    "根据最近几轮对话和主人刚说的话，判断主人要什么：\n"
    "- task：要星期一在电脑、服务器、网络或设备上动手做事；也包括给上一件没做成的事补充信息（地址、账号、选哪个）。\n"
    "- knowledge：问知识、事实、计算、时效信息、建议、怎么做某事（只要讲不用动手）。\n"
    "- ask_teacher：主人明确让去问老师/DeepSeek，或嫌上一句答得不好要重答。\n"
    "- chat：闲聊、寒暄、情绪、关于星期一自己的轻松话题。\n"
    "- unclear：只有这句话明显是语音识别出的乱码、几个字完全连不成意思时才用；能猜出合理意思就不要用。\n"
    "rewrite：把主人的意思写成一句完整、不依赖上下文的话：补全「这个」「刚才」「它」指的内容；语音识别常把数字和英文听错"
    "（SH 是 SSH，「一零点零点零点幺五」是 10.0.0.15，depsek 是 DeepSeek），按最合理的意思改正；"
    "ask_teacher 时写成要问老师的完整问题；[密码已隐藏] 原样保留，不要改写或猜测密码。\n"
    '只输出 JSON：{"intent": "task|knowledge|ask_teacher|chat|unclear", "rewrite": "...", "why": "一句话"}'
)
_ASK_TEACHER = re.compile(r"(?i)问(问|一下)?(你的?|一下)?(老师|deep\s*seek|deepseek|depsek|dipsik)|去问|查一下资料")


@dataclass
class Intent:
    kind: str
    rewrite: str
    why: str = ""
    by: str = "teacher"  # "teacher" or "rules"


def _json(text: str) -> dict:
    i, j = (text or "").find("{"), (text or "").rfind("}")
    try:
        obj = json.loads(text[i:j + 1]) if 0 <= i < j else {}
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def fallback(text: str) -> Intent:
    if is_agent_task(text):
        return Intent("task", text, "keyword", "rules")
    if _ASK_TEACHER.search(text or ""):
        return Intent("ask_teacher", text, "keyword", "rules")
    return Intent("chat", text, "default", "rules")


def recent_block(recent: list[dict]) -> str:
    lines = []
    for t in recent[-6:]:
        who = {"agent": "星期一（动手）", "teacher": "星期一（问了老师）"}.get(t.get("who", ""), "星期一")
        lines.append(f"主人：{t['q']}\n{who}：{t['a'][:200]}")
    return "\n".join(lines)


def classify(teacher, text: str, recent: list[dict], hosts: list[str] | None = None) -> Intent:
    """Blocking (call from a worker thread). Never raises: falls back to keyword rules.
    hosts: configured SSH aliases ("196（gp@192.168.100.196）"), so the rewrite keeps them instead of guessing IPs."""
    if not getattr(teacher, "available", False):
        return fallback(text)
    ctx = recent_block(recent)
    user = f"最近的对话：\n{ctx}\n\n" if ctx else ""
    if hosts:
        user += f"星期一已配置的机器（别名（账号@地址））：{'、'.join(hosts)}。主人说到这些名字时 rewrite 里保留别名，不要改成别的地址。\n\n"
    user += f"主人刚说：{text}"
    try:
        res = _json(teacher.complete(INTENT_SYSTEM, user, json_mode=True, max_tokens=300, timeout_s=8))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[intent] teacher failed (%s): keyword rules", exc)
        return fallback(text)
    kind = str(res.get("intent") or "")
    if kind not in KINDS:
        return fallback(text)
    rewrite = str(res.get("rewrite") or "").strip() or text
    if "[密码已隐藏]" in text and "[密码已隐藏]" not in rewrite and "密码" in rewrite:
        rewrite += "（密码已提供）"
    return Intent(kind, rewrite, str(res.get("why") or ""))
