"""Async TTS queue — pipelined synth/play, one sentence per chunk."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tts.pronunciation_fix import fix_tts_pronunciation
from tts.sentence_split import is_speakable, sanitize_tts_text
from tts.tts_engine import TtsEngine

LOG = logging.getLogger(__name__)
TTS_RUN_DIR = Path("/userdata/agent/run/tts")


@dataclass(frozen=True)
class _TextItem:
    generation: int
    text: str
    wav: Path | None = None  # pre-synthesized clip (ack phrases): skip synth


@dataclass(frozen=True)
class _WavItem:
    generation: int
    wav: Path | None  # None (streaming only): end of a text item that produced no final chunk
    text: str
    final: bool = True  # last piece of its text item (streaming backends split one text into chunks)


class TtsQueue:
    def __init__(
        self,
        engine: TtsEngine,
        *,
        merge_max_chars: int = 80,
    ) -> None:
        self.engine = engine
        self.merge_max_chars = merge_max_chars  # kept for config compat
        self._text_q: asyncio.Queue[_TextItem | None] = asyncio.Queue()
        self._wav_q: asyncio.Queue[_WavItem | None] = asyncio.Queue(maxsize=3)
        self._synth_task: asyncio.Task | None = None
        self._play_task: asyncio.Task | None = None
        self._first_play_at: float | None = None
        self._utterance_started_at: float | None = None
        self._slot = 0
        self._interrupted = asyncio.Event()
        self._utterance_chunks = 0
        self._played_chunks = 0
        self._utterance_gain: float | None = None
        self._current_speaking_chunk: str | None = None
        self._current_speaking_generation: int | None = None
        self._active_generation = 0
        self._engine_lock = asyncio.Lock()
        # streaming backend (qwen3 daemon): requests are sent as soon as text arrives and their
        # chunk streams are forwarded to _wav_q strictly in enqueue order
        self._sources: asyncio.Queue = asyncio.Queue()
        self._forward_task: asyncio.Task | None = None
        # applied to every spoken text (the orchestrator swaps "主人" for the user's title here)
        self.text_filter: Callable[[str], str] | None = None
        # pre-synthesized clips for fixed replies, keyed by spoken_form(text): played without synthesis
        self.clip_cache: dict[str, Path] = {}

    @property
    def is_playing(self) -> bool:
        return self._play_task is not None and not self._interrupted.is_set() and (
            self._wav_q.qsize() > 0 or self._text_q.qsize() > 0 or self._current_speaking_chunk is not None
        )

    def set_max_sentences(self, n: int) -> None:
        _ = n

    async def discard_pending(self, *, include_current: bool = False) -> list[str]:
        texts: list[str] = []
        if include_current and self._current_speaking_chunk:
            texts.append(self._current_speaking_chunk)
        while True:
            try:
                item = self._wav_q.get_nowait()
                if item and item.text:
                    texts.append(item.text)
            except asyncio.QueueEmpty:
                break
            else:
                self._wav_q.task_done()
        while True:
            try:
                t = self._text_q.get_nowait()
                if t:
                    texts.append(t.text)
            except asyncio.QueueEmpty:
                break
            else:
                self._text_q.task_done()
        return texts

    async def interrupt(self) -> list[str]:
        self._interrupted.set()
        pending = await self.discard_pending()
        if getattr(self.engine, "streaming", False):
            await self.engine.qwen3.cancel_all()
        await asyncio.to_thread(self.engine.stop_playback)
        LOG.info("[tts] interrupted, saved %d clauses for relay", len(pending))
        return pending

    def abort(self) -> None:
        """Synchronously request abort and discard pending queues."""
        self._interrupted.set()
        while not self._wav_q.empty():
            try:
                self._wav_q.get_nowait()
                self._wav_q.task_done()
            except Exception:
                break
        while not self._text_q.empty():
            try:
                self._text_q.get_nowait()
                self._text_q.task_done()
            except Exception:
                break
        if getattr(self.engine, "streaming", False):
            self.engine.qwen3.cancel_all_nowait()
        try:
            self.engine.stop_playback()
        except Exception:
            pass

    async def start(self) -> None:
        if self._synth_task is None:
            if getattr(self.engine, "streaming", False):
                self._synth_task = asyncio.create_task(self._stream_submit_worker())
                self._forward_task = asyncio.create_task(self._stream_forward_worker())
                self._play_task = asyncio.create_task(self._stream_play_worker())
            else:
                self._synth_task = asyncio.create_task(self._synth_worker())
                self._play_task = asyncio.create_task(self._play_worker())

    def mark_utterance_start(self, generation: int = 0) -> None:
        self._active_generation = generation
        self._utterance_started_at = time.monotonic()
        self._first_play_at = None
        self._utterance_chunks = 0
        self._played_chunks = 0
        self._utterance_gain = None
        self._interrupted.clear()

    @property
    def first_play_latency_s(self) -> float | None:
        if self._utterance_started_at is None or self._first_play_at is None:
            return None
        return self._first_play_at - self._utterance_started_at

    @property
    def first_play_at(self) -> float | None:
        return self._first_play_at

    def spoken_form(self, text: str) -> str:
        """The text exactly as it would be synthesized (title swap, cleanup, pronunciation fixes)."""
        if self.text_filter is not None:
            text = self.text_filter(text)
        return fix_tts_pronunciation(sanitize_tts_text(text))

    async def enqueue(self, text: str, *, generation: int | None = None) -> None:
        generation = self._active_generation if generation is None else generation
        if self._interrupted.is_set() or generation != self._active_generation:
            return
        text = self.spoken_form(text)
        if not is_speakable(text):
            return
        wav = self.clip_cache.get(text)
        if wav is not None and not wav.is_file():
            wav = None
        self._utterance_chunks += 1
        await self._text_q.put(_TextItem(generation, text, wav))

    async def enqueue_wav(self, text: str, wav: Path, *, generation: int | None = None) -> bool:
        """Queue an already-synthesized clip; keeps order with text items."""
        generation = self._active_generation if generation is None else generation
        if self._interrupted.is_set() or generation != self._active_generation or not wav.is_file():
            return False
        self._utterance_chunks += 1
        await self._text_q.put(_TextItem(generation, text, wav))
        return True

    async def synthesize_to(self, text: str, path: Path) -> Path:
        """Synthesize outside the queue (cache warm-up), serialized with the synth worker."""
        async with self._engine_lock:
            return await asyncio.to_thread(self.engine.synthesize, text, str(path))

    async def flush(self) -> None:
        return

    async def wait_done(self, generation: int | None = None) -> None:
        generation = self._active_generation if generation is None else generation
        if (
            self._interrupted.is_set()
            or generation != self._active_generation
            or self._utterance_chunks <= 0
        ):
            return
        await self._text_q.join()
        if self._interrupted.is_set() or generation != self._active_generation:
            return
        await self._wav_q.join()

    async def stop(self) -> None:
        """Cancel queue workers during orchestrator shutdown; safe to call repeatedly."""
        tasks = [t for t in (self._synth_task, self._forward_task, self._play_task) if t is not None]
        if not tasks:
            return
        self._interrupted.set()
        await self.discard_pending()
        await asyncio.to_thread(self.engine.stop_playback)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._synth_task = None
        self._forward_task = None
        self._play_task = None
        self._sources = asyncio.Queue()
        self._current_speaking_chunk = None
        self._current_speaking_generation = None

    async def reconfigure(self, cfg) -> None:
        from tts.tts_engine import TtsConfig

        assert isinstance(cfg, TtsConfig)
        was_streaming = getattr(self.engine, "streaming", False)
        async with self._engine_lock:
            await asyncio.to_thread(self.engine.reconfigure, cfg)
        if was_streaming != getattr(self.engine, "streaming", False) and self._synth_task is not None:
            # backend family changed: restart the workers with the matching pipeline
            await self.stop()
            self._interrupted.clear()
            await self.start()
        LOG.info(
            "[tts] reconfigured backend=%s sid=%d speed=%.2f",
            cfg.backend,
            cfg.sid,
            cfg.speed,
        )

    async def _synth_worker(self) -> None:
        while True:
            item = await self._text_q.get()
            try:
                if item is None:
                    await self._wav_q.put(None)
                    return
                if self._interrupted.is_set() or item.generation != self._active_generation:
                    continue
                if item.wav is not None:
                    await self._wav_q.put(_WavItem(item.generation, item.wav, item.text))
                    continue
                TTS_RUN_DIR.mkdir(parents=True, exist_ok=True)
                path = TTS_RUN_DIR / f"agent_tts_{self._slot}.wav"
                self._slot += 1
                if path.is_file():
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        path = TTS_RUN_DIR / f"agent_tts_{time.monotonic_ns()}_{self._slot}.wav"
                t0 = time.monotonic()
                async with self._engine_lock:
                    wav = await asyncio.to_thread(self.engine.synthesize, item.text, str(path))
                if self._interrupted.is_set() or item.generation != self._active_generation:
                    continue
                LOG.info("[tts] synth %.2fs (%d chars)", time.monotonic() - t0, len(item.text))
                await self._wav_q.put(_WavItem(item.generation, wav, item.text))
            except Exception as exc:
                LOG.error("[tts] synth failed: %s", exc)
            finally:
                self._text_q.task_done()

    async def _play_worker(self) -> None:
        while True:
            item = await self._wav_q.get()
            try:
                if item is None:
                    return
                if self._interrupted.is_set() or item.generation != self._active_generation:
                    continue
                if self._first_play_at is None:
                    self._first_play_at = time.monotonic()
                    lat = self.first_play_latency_s
                    if lat is not None:
                        LOG.info("[tts] first_play latency=%.2fs", lat)
                self._played_chunks += 1
                continuation = self._played_chunks > 1
                last = self._played_chunks >= self._utterance_chunks
                LOG.info("[tts] speak (%d/%d): %s", self._played_chunks, self._utterance_chunks, item.text[:80])
                t0 = time.monotonic()
                self._current_speaking_chunk = item.text
                self._current_speaking_generation = item.generation
                try:
                    gain = await asyncio.to_thread(
                        self.engine.play,
                        item.wav,
                        continuation=continuation,
                        last_in_utterance=last,
                        utterance_gain=self._utterance_gain,
                    )
                finally:
                    if self._current_speaking_generation == item.generation:
                        self._current_speaking_chunk = None
                        self._current_speaking_generation = None
                    try:
                        # only temp files; cached clips (ack) live elsewhere and are reused
                        if item.wav and item.wav.parent == TTS_RUN_DIR and item.wav.is_file():
                            item.wav.unlink(missing_ok=True)
                    except Exception:
                        pass
                if self._utterance_gain is None and gain is not None:
                    self._utterance_gain = gain
                if self._interrupted.is_set():
                    LOG.info("[tts] play aborted (barge-in)")
                else:
                    tag = " cont" if continuation else ""
                    if last:
                        tag += " last"
                    LOG.info("[tts] play %.2fs%s", time.monotonic() - t0, tag)
            except Exception as exc:
                LOG.error("[tts] play failed: %s", exc)
            finally:
                self._wav_q.task_done()

    # ---- streaming backend (qwen3 daemon) ----

    async def _stream_submit_worker(self) -> None:
        """Send each text to the daemon right away so the next sentence generates while the
        current one is still decoding/playing. _text_q.task_done() is left to the forwarder,
        so wait_done() keeps waiting until the audio is queued."""
        while True:
            item = await self._text_q.get()
            if item is None:
                await self._sources.put(None)
                self._text_q.task_done()
                return
            if item.wav is not None or self._interrupted.is_set() or item.generation != self._active_generation:
                await self._sources.put((item, None))
                continue
            try:
                _, chunks = await self.engine.qwen3.submit(item.text)
                LOG.info("[tts] qwen3 submit (%d chars): %s", len(item.text), item.text[:40])
            except Exception as exc:
                LOG.error("[tts] qwen3 submit failed: %s", exc)
                chunks = None
            await self._sources.put((item, chunks))

    def _live(self, generation: int) -> bool:
        return not self._interrupted.is_set() and generation == self._active_generation

    async def _stream_forward_worker(self) -> None:
        while True:
            src = await self._sources.get()
            if src is None:
                await self._wav_q.put(None)
                return
            item, chunks = src
            try:
                if item.wav is not None or chunks is None:
                    # cached clip (ack), or a request that could not be sent
                    if self._live(item.generation):
                        await self._wav_q.put(_WavItem(item.generation, item.wav, item.text, True))
                    continue
                t0 = time.monotonic()
                final_sent = False
                first = True
                while True:
                    c = await chunks.get()
                    if c.path is None:
                        break
                    if not self._live(item.generation) or final_sent:
                        c.path.unlink(missing_ok=True)
                        continue
                    if first:
                        LOG.info("[tts] qwen3 first chunk %.2fs (%d chars)", time.monotonic() - t0, len(item.text))
                        first = False
                    final_sent = c.final
                    await self._wav_q.put(_WavItem(item.generation, c.path, item.text, c.final))
                if self._live(item.generation) and not final_sent:
                    await self._wav_q.put(_WavItem(item.generation, None, item.text, True))
            except Exception as exc:
                LOG.error("[tts] qwen3 forward failed: %s", exc)
            finally:
                self._text_q.task_done()

    async def _stream_play_worker(self) -> None:
        player = self.engine.stream_player
        last_text: str | None = None
        while True:
            item = await self._wav_q.get()
            try:
                if item is None:
                    return
                if not self._live(item.generation):
                    if item.wav is not None and item.wav.name.startswith("qtts_"):
                        item.wav.unlink(missing_ok=True)
                    continue
                env = self.engine.stream_env()
                if item.wav is not None:
                    if self._first_play_at is None:
                        self._first_play_at = time.monotonic()
                        lat = self.first_play_latency_s
                        if lat is not None:
                            LOG.info("[tts] first_play latency=%.2fs", lat)
                    if not player.active:
                        await asyncio.to_thread(player.start, env)
                    if item.text != last_text:
                        LOG.info("[tts] speak (%d/%d): %s", self._played_chunks + 1, self._utterance_chunks, item.text[:80])
                        last_text = item.text
                    self._current_speaking_chunk = item.text
                    self._current_speaking_generation = item.generation
                    try:
                        await asyncio.to_thread(player.write_wav, item.wav)
                    finally:
                        # daemon chunks are temp files; cached clips (ack) are reused
                        if item.wav.name.startswith("qtts_"):
                            item.wav.unlink(missing_ok=True)
                if not item.final:
                    continue
                self._played_chunks += 1
                if self._played_chunks < self._utterance_chunks:
                    continue
                if player.active:
                    # utterance complete: let the device drain, then re-arm the mic
                    t0 = time.monotonic()
                    await asyncio.to_thread(player.finish, restore_mic=True, env=env)
                    LOG.info("[tts] stream drained %.2fs", time.monotonic() - t0)
                last_text = None
                self._current_speaking_chunk = None
                self._current_speaking_generation = None
            except Exception as exc:
                LOG.error("[tts] stream play failed: %s", exc)
            finally:
                self._wav_q.task_done()
