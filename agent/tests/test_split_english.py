import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tts.sentence_split as ss  # noqa: E402


def stream(text: str, step: int = 3) -> list[str]:
    sp = ss.StreamingSentenceSplitter()
    out: list[str] = []
    for i in range(0, len(text), step):
        out += sp.feed(text[i:i + step])
    return out + sp.finish()


class SplitEnglishTests(unittest.TestCase):
    def setUp(self) -> None:
        ss.set_keep_latin(True)

    def tearDown(self) -> None:
        ss.set_keep_latin(False)

    def test_does_not_cut_inside_english_words(self) -> None:
        reply = ("主人，我叫星期一是因为我的名字来源于你提到的M.O.N.D.A.Y. Male Operational Network "
                 "Digital Assistant Youth，也就是男性运行网络数字助理青年版。")
        chunks = stream(reply)
        joined = " ".join(chunks)
        for word in ("Operational", "Network", "Digital", "Assistant", "Youth"):
            self.assertIn(word, joined, chunks)
        self.assertNotIn("Operationa ", joined)
        self.assertTrue(any("M O N D A Y" in c for c in chunks), chunks)


if __name__ == "__main__":
    unittest.main()
