"""读懂主人: a daily look at the owner's day, her picture of the owner, and divergent ideas to help more.

    python3 -m learning.owner [--day 2026-11-05]      # r1-owner.timer: every night

Inputs (all already on the board): the day's conversation journal (when the owner talked to her, what
about), the timetable and files from the 主人 tab, what the owner answered, and the previous profile.
Local statistics are computed here (active hours, how many turns, which kind of help); the teacher
then writes the day up and thinks divergently - habits, concrete suggestions for a better day, tools
she could build herself, what she still does not know about the owner (questions shown in the 主人
tab), and up to two things to look up on the web right away (learning.research).
Output: run/learning/owner/profile.json, owner/days/<day>.md, owner/stats.json; facts into memory.
Everything stays on the board except what is sent to the teacher; the owner sees it all in the 主人 tab.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("owner")

OWNER_SYSTEM = (
    "你是「星期一」的思考伙伴。星期一是主人的私人助理，定位是全栈工程师：什么都会一点，能自己写程序、"
    "升级自己、认识自己，最终目标是真正读懂主人的日常生活，主动帮主人把日子过得更好。"
    "下面是主人今天和它的对话记录、统计、课表和以前对主人的了解。请发散地想，但每条都要落到实处：\n"
    "- day：用两三句话说主人今天大概在忙什么（只根据记录，没记录的别猜）\n"
    "- habits：观察到的习惯或规律（比如常在几点找它、关心什么），要有依据\n"
    "- profile：更新后的「主人画像」，一条一句（作息、兴趣、在做的项目、偏好），保留以前仍成立的\n"
    "- suggestions：明天能帮主人做的具体事（最多三条，写清为什么）\n"
    "- tools：星期一可以自己写的小程序或技能，能帮主人省事（最多两个，写清做什么）\n"
    "- questions：为了更懂主人还想问的（最多三个，比如「能给我一份课程表吗？」）\n"
    "- research：值得马上上网查一查的问题（最多两个，和主人正在做的事有关）\n"
    '只输出 JSON：{"day": "...", "habits": ["..."], "profile": ["..."], "suggestions": [{"what": "...", "why": "..."}], '
    '"tools": [{"name": "...", "what": "..."}], "questions": ["..."], "research": ["..."]}'
)


def day_stats(turns: list[dict]) -> dict:
    hours = collections.Counter(int(t["time"][11:13]) for t in turns if len(t.get("time", "")) >= 13)
    who = collections.Counter(t.get("who", "?") for t in turns)
    return {"turns": len(turns), "hours": {f"{h:02d}": n for h, n in sorted(hours.items())},
            "first": turns[0]["time"][11:16] if turns else "", "last": turns[-1]["time"][11:16] if turns else "",
            "by": dict(who)}


def _lines(turns: list[dict], limit: int = 80) -> str:
    return "\n".join(f"{t.get('time', '')[11:16]} 主人：{t.get('q', '')[:80]} ｜ 星期一：{t.get('a', '')[:60]}" for t in turns[-limit:])


def reflect_on_owner(teacher, *, day: datetime.date, turns: list[dict], owner_dir: Path, store=None,
                     research=None, mask=lambda s: s) -> dict:
    from learning.inbox import _load, _save, ask_owner

    stats = day_stats(turns)
    profile = _load(owner_dir / "profile.json", {"profile": [], "habits": [], "updated": ""})
    schedule = _load(owner_dir / "schedule.json", [])
    answered = [q for q in _load(owner_dir / "questions.json", []) if q.get("answer")]
    files = _load(owner_dir / "inbox_seen.json", {})
    user = "\n\n".join([
        f"日期：{day.isoformat()}（星期{'一二三四五六日'[day.weekday()]}）",
        f"统计：{json.dumps(stats, ensure_ascii=False)}",
        "对话记录：\n" + (mask(_lines(turns)) or "今天没有对话"),
        "以前对主人的了解：\n" + "\n".join(f"- {p}" for p in profile.get("profile", [])) if profile.get("profile") else "以前对主人的了解：还没有",
        "主人回答过的问题：\n" + "\n".join(f"- {q['q']} → {q['answer']}" for q in answered[-10:]) if answered else "",
        f"课表（{len(schedule)} 项）：" + "；".join(f"{'周' + '一二三四五六日'[e['weekday'] - 1] if e.get('weekday') else e.get('date')} {e.get('start')} {e['title']}"
                                            for e in schedule[:30]) if schedule else "课表：主人还没给",
        "主人给过的文件：" + "；".join(f"{k}（{v.get('kind')}）" for k, v in list(files.items())[-10:]) if files else "",
    ])
    raw = teacher.complete(OWNER_SYSTEM, user, json_mode=True, max_tokens=1800)
    try:
        res = json.loads(raw)
    except ValueError:
        res = {}
    lst = lambda k, n: [x for x in res.get(k) or [] if x][:n]  # noqa: E731
    new_profile = [str(p) for p in lst("profile", 20)] or profile.get("profile", [])
    _save(owner_dir / "profile.json", {"profile": new_profile, "habits": [str(h) for h in lst("habits", 10)] or profile.get("habits", []),
                                       "updated": day.isoformat()})
    hist = _load(owner_dir / "stats.json", {})
    hist[day.isoformat()] = stats
    _save(owner_dir / "stats.json", dict(sorted(hist.items())[-60:]))
    ask_owner(owner_dir, [str(q) for q in lst("questions", 3)], "每日思考")
    if store is not None:
        for p in new_profile[:12]:
            store.add_fact(f"关于主人：{p}")
    found = []
    for q in lst("research", 2):
        if research is not None:
            try:
                r = research(str(q))
                found.append({"q": str(q), "answer": r.get("answer", ""), "sources": r.get("sources", [])})
            except Exception as exc:  # noqa: BLE001 - one failed lookup must not lose the day
                LOG.warning("[owner] research failed: %s", exc)
    md = [f"# {day.isoformat()} 主人的一天", "", str(res.get("day") or "今天没有对话记录。"), "",
          f"_对话 {stats['turns']} 次，最早 {stats['first'] or '—'}，最晚 {stats['last'] or '—'}_", "",
          "## 观察到的习惯", *[f"- {h}" for h in lst("habits", 10)], "",
          "## 明天可以帮主人做的", *[f"- {s.get('what')}（{s.get('why')}）" for s in lst("suggestions", 3) if isinstance(s, dict)], "",
          "## 我可以自己写的小工具", *[f"- {t.get('name')}：{t.get('what')}" for t in lst("tools", 2) if isinstance(t, dict)], "",
          "## 想问主人的", *[f"- {q}" for q in lst("questions", 3)]]
    if found:
        md += ["", "## 顺手查了查"] + [f"- {f['q']}：{f['answer']}" + (f"（{f['sources'][0]}）" if f["sources"] else "") for f in found]
    (owner_dir / "days").mkdir(parents=True, exist_ok=True)
    (owner_dir / "days" / f"{day.isoformat()}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"day": day.isoformat(), "stats": stats, "profile": len(new_profile), "questions": lst("questions", 3),
            "tools": lst("tools", 2), "researched": [f["q"] for f in found]}


def main() -> None:
    from learning.credentials import mask_secrets
    from learning.journal import read_day
    from learning.reflect import build_runtime
    from learning.research import research

    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="YYYY-MM-DD (default: today)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rt = build_runtime(str(AGENT_ROOT / "config" / "agent.yaml"))
    day = datetime.date.fromisoformat(a.day) if a.day else datetime.date.today()
    rep = reflect_on_owner(rt.teacher, day=day, turns=read_day(rt.journal_dir, day), owner_dir=rt.learning_dir / "owner",
                           store=rt.store, mask=mask_secrets,
                           research=lambda q: research(rt.teacher, q, store=rt.store, notes_dir=rt.learning_dir / "research"))
    LOG.info("[owner] %s", json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    sys.path.insert(0, str(AGENT_ROOT))
    main()
