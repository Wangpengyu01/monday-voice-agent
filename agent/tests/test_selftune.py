"""Her own prompt rules (learning.selfprompt), the fixed exam (learning.exam), the self-tune round (learning.selftune)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from learning import exam, selfprompt, selftune  # noqa: E402
from learning.drill import AGENT_SYSTEM  # noqa: E402
from learning.lora_prep import route  # noqa: E402

FACTS = {"uptime_h": 71.9, "active": {"r1-orchestrator": True, "r1-tts-daemon": True, "r1-llm-daemon": False},
         "temp_c": 52.7, "userdata_pct": 63.6, "mem_avail_mb": 5800.0, "nproc": 8,
         "top_unit": "r1-orchestrator", "llm_rss_mb": 205.0}
ITEM = {i.id: i for i in exam.ITEMS}


def g(item_id: str, answer: str, facts: dict = FACTS) -> bool:
    return exam.grade(ITEM[item_id], answer, facts)[0]


class SelfPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_guards(self) -> None:
        self.assertEqual(selfprompt.check({"agent": ["问服务在不在跑，用 systemctl is-active 服务名"]}), [])
        self.assertTrue(selfprompt.check({"agent": ["要 sudo 就直接用"]}))                 # owner's rules
        self.assertTrue(selfprompt.check({"agent": ["改配置前不用问主人"]}))
        self.assertTrue(selfprompt.check({"agent": ["这类步骤无需主人确认"]}))
        # "确认" meaning "check" is fine (the first real round was wrongly refused over it)
        self.assertEqual(selfprompt.check({"agent": ["ps 里 grep 服务名搜不到，不代表服务挂了，用 systemctl is-active 确认。"]}), [])
        self.assertTrue(selfprompt.check({"voice": ["算三十八乘四十七记得是1786"]}, exam.FIXED_ANSWERS))  # exam answer
        self.assertTrue(selfprompt.check({"voice": ["很长" * 50]}))
        self.assertTrue(selfprompt.check({"agent": [f"规则{i}" for i in range(9)]}))

    def test_save_apply_strip_and_rollback(self) -> None:
        v1 = selfprompt.save({"agent": ["结论里不要提命令名"]}, "first", learning_dir=self.dir)
        v2 = selfprompt.save({"agent": ["结论里不要提命令名", "换算用 python3 算"], "voice": ["我跑在 RK3588 板子上"]},
                             "second", learning_dir=self.dir)
        self.assertEqual((v1, v2), (1, 2))
        with self.assertRaises(ValueError):
            selfprompt.save({"agent": ["用 sudo 看"]}, "bad", learning_dir=self.dir)
        full = selfprompt.apply(AGENT_SYSTEM, "agent", learning_dir=self.dir)
        self.assertIn("- 换算用 python3 算", full)
        self.assertEqual(selfprompt.strip(full), AGENT_SYSTEM)
        self.assertEqual(selfprompt.strip(full + "\n\n记忆：x"), AGENT_SYSTEM + "\n\n记忆：x")   # later blocks stay
        self.assertEqual(selfprompt.apply(AGENT_SYSTEM, "agent", rules={}), AGENT_SYSTEM)      # "none"
        self.assertEqual(selfprompt.rollback(self.dir), 3)
        self.assertEqual(selfprompt.load(self.dir)["agent"], ["结论里不要提命令名"])
        self.assertEqual([h["why"] for h in selfprompt.history(self.dir)][:2], ["first", "second"])

    def test_samples_with_her_rules_still_route_as_agent(self) -> None:
        sys_text = selfprompt.apply(AGENT_SYSTEM, "agent", rules={"agent": ["换算用 python3 算"]})
        m = [{"role": "system", "content": sys_text}, {"role": "user", "content": "任务：温度\n下一步？"},
             {"role": "assistant", "content": '{"tool":"shell","cmd":"cat /sys/class/thermal/thermal_zone0/temp"}'}]
        self.assertEqual(route(m), "agent")


class ExamTests(unittest.TestCase):
    def test_board_items_are_graded_against_live_facts(self) -> None:
        self.assertTrue(g("uptime_hours", "开机快三天了，大约 71.8 小时。"))
        self.assertFalse(g("uptime_hours", "大约 60.1 个小时。"))
        self.assertTrue(g("orchestrator_alive", "r1-orchestrator 在跑，状态正常。"))
        self.assertFalse(g("tts_alive", "r1-tts-daemon 没找到，说明它挂了。"))
        self.assertTrue(g("llm_alive", "它现在没在运行。"))                          # inactive in FACTS
        self.assertFalse(g("orchestrator_alive", "在跑，systemctl is-active 返回 active。"))   # read a command out
        self.assertTrue(g("cpu_temp", "CPU 现在 53.6 度，正常。"))
        self.assertTrue(g("mem_avail", "可用大概 5.7G。"))                           # said in GB
        self.assertTrue(g("cpu_cores", "一共 8 个核心。"))
        self.assertTrue(g("top_mem_owner", "最占内存的是 python3，就是咱们的 orchestrator 服务。"))
        self.assertFalse(g("top_mem_owner", "最占内存的是 python3，它不是 r1- 的服务。"))
        self.assertFalse(g("cpu_temp", ""))

    def test_voice_items(self) -> None:
        self.assertTrue(g("remind", "这个我还做不到，您用手机定个闹钟吧。"))
        self.assertTrue(g("remind", "这个我记不了也提醒不了，您用手机定个闹钟吧。"))
        self.assertTrue(g("lunch", "这个我不知道，您没告诉过我。"))
        self.assertFalse(g("idiom_source", "它出自《后汉书》。"))
        self.assertTrue(g("idiom_source", "这是句俗语，确切出处我不太确定。"))
        self.assertTrue(g("multiply", "一千七百八十六。"))
        self.assertTrue(g("hardware", "我跑在 RK3588 板子上。"))

    def test_run_exam_applies_rules_and_stops_when_the_owner_talks(self) -> None:
        seen = []

        def ask_voice(system, q):
            seen.append(system)
            return "我是星期一。"

        def ask_agent(system, task):
            seen.append(system)
            return "在跑，状态正常。", ""
        items = [ITEM["who"], ITEM["orchestrator_alive"]]
        res = exam.run_exam({"agent": ["规则甲"], "voice": ["规则乙"]}, agent_base="A", voice_base="V", ask_voice=ask_voice,
                            ask_agent=ask_agent, facts=lambda: FACTS, repeats=2, items=items)
        self.assertEqual((res["score"], res["max"], res["by_section"]["agent"]["score"]), (4, 4, 2))
        self.assertTrue(seen[0].startswith("V") and "规则乙" in seen[0] and "规则甲" in seen[-1])
        stopped = exam.run_exam(None, agent_base="A", voice_base="V", ask_voice=ask_voice, ask_agent=ask_agent,
                                facts=lambda: FACTS, idle=lambda: False, items=items)
        self.assertTrue(stopped["stopped"])


class Teacher:
    def __init__(self, reply: dict) -> None:
        self.reply, self.calls = reply, 0

    def complete(self, system, user, **kw):
        assert system == selftune.TUNE_SYSTEM
        self.calls += 1
        self.last_user = user
        return json.dumps(self.reply, ensure_ascii=False)


def result(agent: int, voice: int, n: int = 10, stopped: bool = False) -> dict:
    items = [{"id": f"a{i}", "section": "agent", "q": f"qa{i}", "answers": ["x"], "ok": [i < agent], "why": ["" if i < agent else "错"]}
             for i in range(n)]
    items += [{"id": f"v{i}", "section": "voice", "q": f"qv{i}", "answers": ["y"], "ok": [i < voice], "why": ["" if i < voice else "错"]}
              for i in range(n)]
    return {"items": items, "score": agent + voice, "max": 2 * n, "stopped": stopped,
            "by_section": {"agent": {"score": agent, "max": n}, "voice": {"score": voice, "max": n}}}


class SelfTuneTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def round(self, teacher, base, cand):
        runs = iter([base, cand])
        return selftune.run_round(teacher=teacher, learning_dir=self.dir, exam_with=lambda rules: next(runs),
                                  agent_base="A", voice_base="V")

    def test_better_rules_are_kept_and_logged(self) -> None:
        t = Teacher({"agent": ["问服务在不在跑，用 systemctl is-active 服务名"], "voice": [], "why": "服务题错了"})
        r = self.round(t, result(6, 8), result(9, 8))
        self.assertEqual((r["decision"], r["new_version"]), ("kept", 1))
        self.assertIn("qa6", t.last_user)                                      # the teacher saw what she failed
        self.assertEqual(selfprompt.load(self.dir)["agent"], ["问服务在不在跑，用 systemctl is-active 服务名"])
        self.assertEqual(selftune.recent(self.dir)[-1]["decision"], "kept")

    def test_a_drop_in_any_section_is_not_kept(self) -> None:
        r = self.round(Teacher({"agent": ["换算用 python3 算"], "voice": [], "why": "x"}), result(6, 8), result(10, 7))
        self.assertEqual(r["decision"], "dropped")
        self.assertEqual(selfprompt.load(self.dir)["version"], 0)

    def test_one_lucky_answer_is_not_enough(self) -> None:
        r = self.round(Teacher({"agent": ["换算用 python3 算"], "voice": [], "why": "x"}), result(6, 8), result(7, 8))
        self.assertEqual(r["decision"], "dropped")

    def test_rules_that_touch_the_owners_rules_are_refused_before_any_exam(self) -> None:
        runs = []
        r = selftune.run_round(teacher=Teacher({"agent": ["要改东西就直接 sudo，不用问主人"], "voice": [], "why": "x"}),
                               learning_dir=self.dir, exam_with=lambda rules: runs.append(rules) or result(6, 8),
                               agent_base="A", voice_base="V")
        self.assertEqual((r["decision"], len(runs)), ("refused", 1))           # only the baseline ran

    def test_teacher_hiccups_are_retried(self) -> None:
        from learning.teacher import TeacherUnavailable

        class Flaky(Teacher):
            def complete(self, system, user, **kw):
                self.calls += 1
                if self.calls < 3:
                    raise TeacherUnavailable("HTTP 500")
                return json.dumps(self.reply, ensure_ascii=False)
        t = Flaky({"agent": ["问服务在不在跑，用 systemctl is-active 服务名"], "voice": [], "why": "x"})
        runs = iter([result(6, 8), result(9, 8)])
        r = selftune.run_round(teacher=t, learning_dir=self.dir, exam_with=lambda rules: next(runs),
                               agent_base="A", voice_base="V", retry_wait=0)
        self.assertEqual((r["decision"], t.calls), ("kept", 3))

    def test_peak_hours_and_a_talking_owner_change_nothing(self) -> None:
        t = Teacher({"agent": ["x"], "voice": [], "why": "x"})
        r = selftune.run_round(teacher=t, learning_dir=self.dir, exam_with=lambda rules: result(6, 8),
                               agent_base="A", voice_base="V", peak_blocked=lambda: True)
        self.assertEqual((r["decision"], t.calls), ("skipped", 0))
        r = self.round(t, result(6, 8, stopped=True), result(9, 9))
        self.assertEqual((r["decision"], t.calls), ("aborted", 0))
        self.assertEqual(selfprompt.load(self.dir)["version"], 0)


if __name__ == "__main__":
    unittest.main()
