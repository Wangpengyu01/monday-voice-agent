from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

import orchestrator.main as m  # noqa: E402


def shell(recent, ctx=768, n=5):
    o = m.Orchestrator.__new__(m.Orchestrator)
    o.history_max_turns, o.llm_context_tokens, o.max_new_tokens, o._recent = n, ctx, 200, recent
    o._system_body = lambda memory_ctx="": "系统" * 120
    return o


class HistoryBudgetTests(unittest.TestCase):
    def test_keeps_five_recent_turns_from_any_route(self) -> None:
        recent = [{"q": f"问题{i}", "a": f"回答{i}", "who": w} for i, w in enumerate(["local", "teacher", "agent", "memory", "local", "local"])]
        o = shell(recent)
        prompt = o._build_llm_prompt("现在的问题")
        self.assertEqual(o._last_history_turns, 5)
        self.assertNotIn("问题0", prompt)              # only the last five
        self.assertLess(prompt.index("问题1"), prompt.index("问题5"))  # oldest first
        self.assertTrue(prompt.endswith("<|im_start|>user\n现在的问题<|im_end|>\n<|im_start|>assistant\n"))

    def test_drops_oldest_when_the_budget_runs_out(self) -> None:
        recent = [{"q": f"问题{i}" + "很长" * 20, "a": "回答" * 60, "who": "local"} for i in range(5)]
        o = shell(recent, ctx=560)
        prompt = o._build_llm_prompt("现在")
        self.assertLess(o._last_history_turns, 5)
        self.assertIn("问题4", prompt)        # the newest survives
        self.assertNotIn("问题0", prompt)
        self.assertIn("…", prompt)            # long answers are clipped

    def test_disabled(self) -> None:
        o = shell([{"q": "a", "a": "b", "who": "local"}], n=0)
        o._build_llm_prompt("x")
        self.assertEqual(o._last_history_turns, 0)


if __name__ == "__main__":
    unittest.main()
