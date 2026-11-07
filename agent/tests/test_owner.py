from __future__ import annotations

import datetime
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import inbox, owner, research  # noqa: E402
from learning.memory import MemoryStore  # noqa: E402
from mcp_servers import web_fetch  # noqa: E402


class Teacher:
    def __init__(self, reply: dict) -> None:
        self.reply, self.calls = reply, []

    def complete(self, system, user, **kw):
        self.calls.append(user)
        return json.dumps(self.reply, ensure_ascii=False)


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.store = MemoryStore(self.dir / "memory.db")
        self.files, self.owner = self.dir / "files", self.dir / "owner"

    def tearDown(self) -> None:
        self.store._db.close()
        self._tmp.cleanup()


class ExtractTests(Base):
    def test_docx_xlsx_text(self) -> None:
        d = self.dir / "a.docx"
        with zipfile.ZipFile(d, "w") as z:
            z.writestr("word/document.xml", "<w:document><w:p><w:t>周一 高数</w:t></w:p><w:p><w:t>周三 英语</w:t></w:p></w:document>")
        self.assertIn("周一 高数\n周三 英语", inbox.extract_text(d))
        x = self.dir / "b.xlsx"
        with zipfile.ZipFile(x, "w") as z:
            z.writestr("xl/sharedStrings.xml", "<sst><si><t>节次</t></si><si><t>高数</t></si></sst>")
            z.writestr("xl/worksheets/sheet1.xml", '<worksheet><sheetData><row><c t="s"><v>0</v></c><c><v>1</v></c></row>'
                                                   '<row><c t="s"><v>1</v></c><c><v>2</v></c></row></sheetData></worksheet>')
        self.assertIn("节次\t1\n高数\t2", inbox.extract_text(x))
        g = self.dir / "c.txt"
        g.write_bytes("课程表".encode("gbk"))
        self.assertEqual(inbox.extract_text(g), "课程表")


class InboxTests(Base):
    def test_schedule_file_becomes_events_facts_questions(self) -> None:
        box = self.files / inbox.INBOX_NAME
        box.mkdir(parents=True)
        (box / "课表.txt").write_text("周一 8:00-9:40 高等数学 教一101\n周三 10:00 英语", encoding="utf-8")
        t = Teacher({"kind": "schedule", "summary": "主人这学期的课程表", "facts": [{"q": "主人周一第一节是什么课", "a": "高等数学"}],
                     "events": [{"title": "高等数学", "weekday": 1, "start": "08:00", "end": "09:40", "place": "教一101"},
                                {"title": "英语", "weekday": 3, "start": "10:00"}, {"title": "坏的", "weekday": 9}],
                     "questions": ["这学期到哪周结束？"]})
        rep = inbox.process(t, self.store, files_dir=self.files, owner_dir=self.owner, mask=lambda s: s.replace("101", "***"))
        self.assertEqual(rep["read"], ["课表.txt"])
        self.assertIn("***", t.calls[0])                                    # masked before it leaves the board
        sched = json.loads((self.owner / "schedule.json").read_text(encoding="utf-8"))
        self.assertEqual([e["title"] for e in sched], ["高等数学", "英语"])  # invalid weekday dropped
        self.assertIsNotNone(self.store.best_answer("主人周一第一节是什么课"))
        self.assertEqual(inbox.today_events(self.owner, datetime.date(2026, 11, 2))[0]["title"], "高等数学")   # a Monday
        # unchanged file: not read again; the owner answers her question -> fact
        self.assertEqual(inbox.process(t, self.store, files_dir=self.files, owner_dir=self.owner)["new"], 0)
        q = json.loads((self.owner / "questions.json").read_text(encoding="utf-8"))[0]
        (self.owner / "answers.jsonl").write_text(json.dumps({"id": q["id"], "answer": "第十八周"}, ensure_ascii=False) + "\n", encoding="utf-8")
        self.assertEqual(inbox.process(t, self.store, files_dir=self.files, owner_dir=self.owner)["answers"], 1)
        self.assertTrue(any("第十八周" in f for f in self.store.facts(20)))

    def test_reupload_replaces_that_files_events(self) -> None:
        s = inbox.merge_events([], [{"title": "旧课", "weekday": 2}], "课表.xlsx")
        s = inbox.merge_events(s + [{"title": "别的", "weekday": 1, "source": "x"}], [{"title": "新课", "weekday": 2}], "课表.xlsx")
        self.assertEqual(sorted(e["title"] for e in s), ["别的", "新课"])


class OwnerTests(Base):
    def test_daily_reflection(self) -> None:
        turns = [{"time": "2026-11-05 08:10:00", "who": "local", "q": "今天天气怎么样", "a": "晴"},
                 {"time": "2026-11-05 23:30:00", "who": "agent", "q": "帮我看看板子磁盘", "a": "还剩二十G"}]
        t = Teacher({"day": "主人早上问天气，深夜还在折腾板子", "habits": ["常在深夜调试板子"], "profile": ["在做语音助手项目"],
                     "suggestions": [{"what": "睡前提醒", "why": "常熬夜"}], "tools": [{"name": "作息统计", "what": "每周作息图"}],
                     "questions": ["能给我一份课程表吗？"], "research": ["RK3588 省电设置"]})
        looked = []
        rep = owner.reflect_on_owner(t, day=datetime.date(2026, 11, 5), turns=turns, owner_dir=self.owner, store=self.store,
                                     research=lambda q: looked.append(q) or {"answer": "可以调低频率", "sources": ["https://x"]})
        self.assertEqual(rep["stats"]["turns"], 2)
        self.assertEqual(rep["stats"]["hours"], {"08": 1, "23": 1})
        self.assertEqual(looked, ["RK3588 省电设置"])
        md = (self.owner / "days" / "2026-11-05.md").read_text(encoding="utf-8")
        for s in ("深夜还在折腾板子", "睡前提醒", "作息统计", "能给我一份课程表吗", "可以调低频率"):
            self.assertIn(s, md)
        self.assertTrue(any("语音助手项目" in f for f in self.store.facts(20)))
        self.assertEqual(json.loads((self.owner / "questions.json").read_text(encoding="utf-8"))[0]["q"], "能给我一份课程表吗？")


class ResearchTests(Base):
    def test_search_read_summarize_remember(self) -> None:
        listing = "1. 甲\nhttps://a.example/1\n摘要\n\n2. 乙\nhttps://b.example/2\n摘要"
        pages = {"https://a.example/1": "正文" * 200, "https://b.example/2": "短"}
        t = Teacher({"answer": "编号等于组乘三十二加偏移。", "facts": [{"q": "GPIO1_A7 编号", "a": "39"}],
                     "sources": ["https://a.example/1"], "confidence": "high"})
        out = research.research(t, "RK3588 GPIO 编号", store=self.store, notes_dir=self.dir / "notes",
                                search=lambda a: (True, listing), fetch=lambda a: (True, pages[a["url"]]))
        self.assertIn("https://a.example/1", t.calls[0])
        self.assertNotIn("【网页 https://b.example/2】", t.calls[0])     # too short to be worth reading
        self.assertIsNotNone(self.store.best_answer("GPIO1_A7 编号"))
        self.assertTrue(next((self.dir / "notes").glob("*.md")).read_text(encoding="utf-8").count("https://a.example/1"))
        self.assertEqual(out["confidence"], "high")

    def test_parsers(self) -> None:
        ddg = ('<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2Fp&amp;rut=1">T <b>1</b></a>'
               '<a class="result__snippet" href="x">snip</a>')
        self.assertEqual(web_fetch.parse_ddg(ddg), [{"title": "T 1", "url": "https://ex.com/p", "snippet": "snip"}])
        bing = '<li class="b_algo"><h2><a href="https://b.com/x" h="ID">B</a></h2><div><p>bs</p></div></li>'
        self.assertEqual(web_fetch.parse_bing(bing), [{"title": "B", "url": "https://b.com/x", "snippet": "bs"}])


if __name__ == "__main__":
    unittest.main()
