import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.selftest import Check, announcement  # noqa: E402


class AnnouncementTests(unittest.TestCase):
    def test_all_ok_is_one_sentence(self) -> None:
        checks = [Check("大模型", True), Check("语音合成", True), Check("内存", True, "", "内存剩余五点五G")]
        self.assertEqual(announcement(checks), "星期一已就绪，系统自检全部正常。")

    def test_problems_are_named(self) -> None:
        checks = [Check("大模型", True), Check("网络", True), Check("外网", False, "局域网正常，但外网暂时连不上")]
        self.assertEqual(announcement(checks), "星期一已启动。自检发现：局域网正常，但外网暂时连不上。其余正常。")

    def test_full_report_adds_details(self) -> None:
        checks = [Check("内存", True, "", "内存剩余五点五G"), Check("温度", True, "", "芯片温度四十二度")]
        self.assertEqual(announcement(checks, full=True), "星期一已就绪，系统自检全部正常。内存剩余五点五G，芯片温度四十二度。")


if __name__ == "__main__":
    unittest.main()
