"""Who the assistant is. Change the name / introduction here."""
from __future__ import annotations

ASSISTANT_NAME = "星期一"
FULL_NAME = "M.O.N.D.A.Y."
FULL_NAME_EXPANDED = "Male Operational Network Digital Assistant Youth"

# spoken self-introduction ("你是谁"); the letters are separated so TTS spells them out
INTRODUCTION = (
    f"您好，我是 M O N D A Y，全称是 {FULL_NAME_EXPANDED}，男性运行网络数字助理青年版。"
    f"您也可以直接叫我{ASSISTANT_NAME}。虽然大家都讨厌星期一，但我保证会让您的每一天都轻松高效。"
)

# names the user may call it by: the current one, plus the old name people may still use
CALL_NAMES = (ASSISTANT_NAME, "小揽")
CALL_NAME_RE = "(?:" + "|".join(CALL_NAMES) + ")"
