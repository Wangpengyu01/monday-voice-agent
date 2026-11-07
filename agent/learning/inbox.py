"""The owner's file window: files dropped into /userdata/files/给星期一 (web page -> 主人 tab) get read and learned.

    python3 -m learning.inbox        # r1-inbox.timer runs this every few minutes; no new file -> no teacher call

Per new or changed file: text is extracted (txt/md/csv/json/ics/log/code, pdf via pdftotext, docx/xlsx/pptx
straight from their XML), secrets are masked, and the teacher says what it is. A schedule (课程表, 日程)
becomes structured events in run/learning/owner/schedule.json; facts go to memory (source "inbox");
a note per file goes to run/learning/owner/files/. Answers the owner typed into the 主人 tab
(run/learning/owner/answers.jsonl) are stored as facts too.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
INBOX_NAME = "给星期一"
TEXT_EXT = {".txt", ".md", ".csv", ".tsv", ".json", ".ics", ".log", ".yaml", ".yml", ".py", ".sh", ".html", ".htm", ".xml"}
MAX_TEXT = 12000
LOG = logging.getLogger("inbox")

INBOX_SYSTEM = (
    "主人把一个文件交给了语音助手「星期一」，让它读懂并记住。星期一的定位是全栈工程师型的私人助理："
    "要了解主人的生活和工作习惯，好主动帮忙。根据文件内容回答（看不懂或无关的就如实说）：\n"
    "- kind：schedule（课程表、日程、排班）| notes（主人的笔记、计划、清单）| document（资料、文档）| data（数据表）| other\n"
    "- summary：两三句说清这是什么、对了解主人有什么用\n"
    "- facts：值得记住的关于主人或这份资料的事实（问答形式，最多八条）\n"
    "- events：如果是日程/课程表，列出每一项（每周重复的写 weekday 1-7；只有某天的写 date YYYY-MM-DD；时间用 24 小时 HH:MM）\n"
    "- questions：读完后还想问主人的问题（最多三个，比如「这学期到哪周结束？」）\n"
    '只输出 JSON：{"kind": "...", "summary": "...", "facts": [{"q": "...", "a": "..."}], '
    '"events": [{"title": "...", "weekday": 1, "date": "", "start": "08:00", "end": "09:40", "place": "", "note": ""}], '
    '"questions": ["..."]}'
)


# ---------------------------------------------------------------- text extraction
def _xml_text(xml: str) -> str:
    xml = re.sub(r"</w:p>|</a:p>|<w:tab/>|<w:br/>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml)


def _xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = [re.sub(r"<[^>]+>", "", s) for s in
                      re.findall(r"(?s)<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode("utf-8", "replace"))]
        out = []
        for name in sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)):
            out.append(f"## {name.rsplit('/', 1)[-1]}")
            for row in re.findall(r"(?s)<row[^>]*>(.*?)</row>", z.read(name).decode("utf-8", "replace")):
                cells = []
                for attrs, body in re.findall(r'(?s)<c([^>]*)>(.*?)</c>', row):
                    v = re.search(r"(?s)<v>(.*?)</v>", body) or re.search(r"(?s)<t[^>]*>(.*?)</t>", body)
                    val = v.group(1) if v else ""
                    if 't="s"' in attrs and val.isdigit() and int(val) < len(shared):
                        val = shared[int(val)]
                    cells.append(val)
                if any(c.strip() for c in cells):
                    out.append("\t".join(cells))
        return "\n".join(out)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext in TEXT_EXT:
            raw = path.read_bytes()[:400_000]
            for enc in ("utf-8", "gbk"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", "replace")
        if ext == ".pdf":
            r = subprocess.run(["pdftotext", "-layout", "-l", "30", str(path), "-"], capture_output=True, timeout=60)
            return r.stdout.decode("utf-8", "replace")
        if ext == ".docx":
            with zipfile.ZipFile(path) as z:
                return _xml_text(z.read("word/document.xml").decode("utf-8", "replace"))
        if ext == ".pptx":
            with zipfile.ZipFile(path) as z:
                slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                                key=lambda n: int(re.findall(r"\d+", n)[-1]))
                return "\n\n".join(f"## 第{i}页\n" + _xml_text(z.read(n).decode("utf-8", "replace")) for i, n in enumerate(slides, 1))
        if ext == ".xlsx":
            return _xlsx(path)
    except (OSError, zipfile.BadZipFile, KeyError, subprocess.SubprocessError) as exc:
        LOG.warning("[inbox] cannot read %s: %s", path.name, exc)
    return ""


# ---------------------------------------------------------------- processing
def _sig(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def merge_events(schedule: list[dict], events: list[dict], source: str) -> list[dict]:
    """Events from one file replace that file's previous events (re-uploading a timetable updates it)."""
    keep = [e for e in schedule if e.get("source") != source]
    for e in events:
        if not isinstance(e, dict) or not str(e.get("title") or "").strip():
            continue
        wd = e.get("weekday")
        item = {"title": str(e["title"]).strip()[:60], "start": str(e.get("start") or "")[:5], "end": str(e.get("end") or "")[:5],
                "place": str(e.get("place") or "")[:40], "note": str(e.get("note") or "")[:80], "source": source}
        if isinstance(wd, int) and 1 <= wd <= 7:
            item["weekday"] = wd
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(e.get("date") or "")):
            item["date"] = e["date"]
        else:
            continue
        keep.append(item)
    return sorted(keep, key=lambda e: (e.get("date") or f"w{e.get('weekday', 9)}", e.get("start") or ""))


def ask_owner(owner_dir: Path, questions: list[str], source: str) -> None:
    qf = owner_dir / "questions.json"
    qs = _load(qf, [])
    have = {q["q"] for q in qs}
    for q in questions[:3]:
        q = str(q).strip()
        if q and q not in have:
            qs.append({"id": hashlib.sha1(q.encode()).hexdigest()[:10], "q": q, "from": source,
                       "time": time.strftime("%Y-%m-%d %H:%M"), "answer": ""})
    _save(qf, qs[-40:])


def take_answers(owner_dir: Path, store) -> int:
    """The owner's answers typed in the 主人 tab -> facts; the question is marked answered."""
    af = owner_dir / "answers.jsonl"
    if not af.is_file():
        return 0
    lines = af.read_text(encoding="utf-8").splitlines()
    af.write_text("", encoding="utf-8")
    qs = _load(owner_dir / "questions.json", [])
    by_id = {q["id"]: q for q in qs}
    n = 0
    for line in lines:
        try:
            a = json.loads(line)
        except ValueError:
            continue
        q = by_id.get(str(a.get("id")))
        text = str(a.get("answer") or "").strip()[:500]
        if not q or not text:
            continue
        q["answer"] = text
        store.remember(q["q"], text, source="owner")
        store.add_fact(f"{q['q']} —— 主人说：{text}")
        n += 1
    _save(owner_dir / "questions.json", qs)
    return n


def process(teacher, store, *, files_dir: Path, owner_dir: Path, mask=lambda s: s) -> dict:
    inbox = files_dir / INBOX_NAME
    inbox.mkdir(parents=True, exist_ok=True)
    seen = _load(owner_dir / "inbox_seen.json", {})
    schedule = _load(owner_dir / "schedule.json", [])
    report = {"new": 0, "read": [], "skipped": [], "answers": take_answers(owner_dir, store)}
    for f in sorted(p for p in inbox.iterdir() if p.is_file() and not p.name.endswith(".part")):
        sig = _sig(f)
        if seen.get(f.name, {}).get("sig") == sig:
            continue
        report["new"] += 1
        text = mask(extract_text(f)).strip()
        if len(text) < 20:
            seen[f.name] = {"sig": sig, "kind": "unreadable", "summary": "读不出文字（图片或不支持的格式）", "time": time.strftime("%Y-%m-%d %H:%M")}
            report["skipped"].append(f.name)
            continue
        raw = teacher.complete(INBOX_SYSTEM, f"文件名：{f.name}\n\n内容：\n{text[:MAX_TEXT]}", json_mode=True, max_tokens=2500)
        try:
            res = json.loads(raw)
        except ValueError:
            res = {}
        facts = [x for x in res.get("facts") or [] if isinstance(x, dict) and x.get("q") and x.get("a")][:8]
        for x in facts:
            store.remember(str(x["q"]), str(x["a"]), source="inbox")
        summary = str(res.get("summary") or "").strip()
        if summary:
            store.remember(f"主人给我的文件《{f.stem}》是什么", summary, source="inbox")
        events = res.get("events") or []
        if events:
            schedule = merge_events(schedule, events, f.name)
        ask_owner(owner_dir, [q for q in res.get("questions") or [] if isinstance(q, str)], f.name)
        note = [f"# {f.name}", "", f"类型：{res.get('kind', 'other')}", "", summary, "", "## 记住的事", *[f"- {x['q']}：{x['a']}" for x in facts]]
        if events:
            note += ["", f"## 日程（{len(events)} 项，已放进课表）"]
        (owner_dir / "files").mkdir(parents=True, exist_ok=True)
        (owner_dir / "files" / f"{f.stem}.md").write_text("\n".join(note) + "\n", encoding="utf-8")
        seen[f.name] = {"sig": sig, "kind": str(res.get("kind") or "other"), "summary": summary, "facts": len(facts),
                        "events": len(events), "time": time.strftime("%Y-%m-%d %H:%M")}
        report["read"].append(f.name)
    _save(owner_dir / "schedule.json", schedule)
    _save(owner_dir / "inbox_seen.json", seen)
    return report


def today_events(owner_dir: Path, day=None) -> list[dict]:
    """Today's schedule (weekly items for this weekday + dated items for today), sorted by start."""
    import datetime
    day = day or datetime.date.today()
    sched = _load(owner_dir / "schedule.json", [])
    return sorted([e for e in sched if e.get("weekday") == day.isoweekday() or e.get("date") == day.isoformat()],
                  key=lambda e: e.get("start") or "")


def main() -> None:
    from learning.credentials import mask_secrets
    from learning.reflect import build_runtime

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rt = build_runtime(str(AGENT_ROOT / "config" / "agent.yaml"))
    files_dir = Path("/userdata/files")
    rep = process(rt.teacher, rt.store, files_dir=files_dir, owner_dir=rt.learning_dir / "owner", mask=mask_secrets)
    if rep["new"] or rep["answers"]:
        LOG.info("[inbox] %s", rep)


if __name__ == "__main__":
    sys.path.insert(0, str(AGENT_ROOT))
    main()
