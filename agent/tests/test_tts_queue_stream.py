from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from tts.qwen3_tts import Qwen3Chunk  # noqa: E402
from tts.tts_queue import TtsQueue  # noqa: E402


class FakeClient:
    """Stands in for Qwen3TtsClient: each text yields scripted chunks after a delay."""

    def __init__(self, tmp: Path, script: dict[str, list[tuple[str, bool]]], delay: float = 0.01) -> None:
        self.tmp = tmp
        self.script = script
        self.delay = delay
        self.submitted: list[str] = []
        self.cancelled = 0
        self._queues: list[asyncio.Queue] = []

    async def submit(self, text: str):
        self.submitted.append(text)
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)

        async def feed() -> None:
            for name, final in self.script[text]:
                await asyncio.sleep(self.delay)
                p = self.tmp / f"qtts_{name}.wav"
                p.write_bytes(b"x")
                q.put_nowait(Qwen3Chunk(text, p, 0, final))
            q.put_nowait(Qwen3Chunk(text, None))

        asyncio.create_task(feed())
        return text, q

    async def cancel_all(self) -> None:
        self.cancelled += 1
        for q in self._queues:
            q.put_nowait(Qwen3Chunk("", None))

    def cancel_all_nowait(self) -> None:
        self.cancelled += 1


class FakePlayer:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.active = False
        self.write_delay = 0.0

    def start(self, env) -> None:
        self.active = True
        self.events.append("start")

    def write_wav(self, path: Path) -> bool:
        import time

        time.sleep(self.write_delay)
        self.events.append(path.stem)
        return True

    def finish(self, *, restore_mic: bool, env) -> None:
        self.active = False
        self.events.append("finish")

    def stop(self) -> None:
        self.active = False
        self.events.append("stop")


class FakeStreamingEngine:
    streaming = True

    def __init__(self, client: FakeClient) -> None:
        self.qwen3 = client
        self.stream_player = FakePlayer()

    def stream_env(self) -> dict[str, str]:
        return {}

    def stop_playback(self) -> None:
        self.stream_player.stop()


class StreamingQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    async def asyncTearDown(self) -> None:
        self._tmp.cleanup()

    async def test_ack_then_two_sentences_play_in_order_in_one_stream(self) -> None:
        client = FakeClient(self.tmp, {
            "第一句。": [("a0", False), ("a1", True)],
            "第二句。": [("b0", True)],
        })
        engine = FakeStreamingEngine(client)
        queue = TtsQueue(engine)
        await queue.start()
        try:
            ack = self.tmp / "ack.wav"
            ack.write_bytes(b"x")
            queue.mark_utterance_start(1)
            await queue.enqueue_wav("嗯，", ack, generation=1)
            await queue.enqueue("第一句。", generation=1)
            await queue.enqueue("第二句。", generation=1)
            await asyncio.wait_for(queue.wait_done(generation=1), timeout=2)
            self.assertEqual(engine.stream_player.events, ["start", "ack", "qtts_a0", "qtts_a1", "qtts_b0", "finish"])
            # both requests were sent before the first one finished playing (pipelined)
            self.assertEqual(client.submitted, ["第一句。", "第二句。"])
            self.assertFalse((self.tmp / "qtts_a0.wav").exists())
            self.assertTrue(ack.exists())
        finally:
            await queue.stop()

    async def test_request_without_final_chunk_still_finishes_the_stream(self) -> None:
        client = FakeClient(self.tmp, {"一句话。": [("c0", False), ("c1", False)]})
        engine = FakeStreamingEngine(client)
        queue = TtsQueue(engine)
        await queue.start()
        try:
            queue.mark_utterance_start(1)
            await queue.enqueue("一句话。", generation=1)
            await asyncio.wait_for(queue.wait_done(generation=1), timeout=2)
            self.assertEqual(engine.stream_player.events, ["start", "qtts_c0", "qtts_c1", "finish"])
        finally:
            await queue.stop()

    async def test_barge_in_cancels_daemon_and_new_turn_plays(self) -> None:
        client = FakeClient(self.tmp, {
            "很长的旧回答。": [(f"o{i}", i == 9) for i in range(10)],
            "新回答。": [("n0", True)],
        }, delay=0.05)
        engine = FakeStreamingEngine(client)
        queue = TtsQueue(engine)
        await queue.start()
        try:
            queue.mark_utterance_start(1)
            await queue.enqueue("很长的旧回答。", generation=1)
            await asyncio.sleep(0.12)
            await queue.interrupt()
            self.assertEqual(client.cancelled, 1)
            queue.mark_utterance_start(2)
            await queue.enqueue("新回答。", generation=2)
            await asyncio.wait_for(queue.wait_done(generation=2), timeout=2)
            events = engine.stream_player.events
            self.assertIn("stop", events)
            self.assertEqual(events[-2:], ["qtts_n0", "finish"])
            self.assertLess(sum(e.startswith("qtts_o") for e in events), 10)
        finally:
            await queue.stop()


if __name__ == "__main__":
    unittest.main()
