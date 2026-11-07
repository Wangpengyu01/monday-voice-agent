"""Qwen3-TTS backend: client for the resident rknn_qwen3_tts_daemon + gapless stream player.

The daemon (RK1828 talker + RK3588 decoder) streams one request as several ~2 s WAV chunks that
split mid-sentence, so chunks go into one long-running aplay per utterance instead of one
play_wav.sh per file (process start-up would leave audible gaps inside words).

Protocol (JSON lines over a Unix socket):
  -> {"type":"synth","id":..,"text":..}   <- {"type":"chunk","id","seq","path","frames"} ... {"type":"done","id","chunks"}
  -> {"type":"cancel","id":..|"*"}        <- {"type":"cancelled","id"}
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)
DEFAULT_SOCK = "/tmp/r1-tts.sock"
VOICE_SCRIPTS = Path("/userdata/voice/scripts")
SAMPLE_RATE = 24000


@dataclass(frozen=True)
class Qwen3Chunk:
    request_id: str
    path: Path | None  # None: request finished (done/cancelled/error)
    seq: int = 0
    final: bool = False  # daemon: nothing more to decode for this request


class Qwen3TtsClient:
    """asyncio client; one persistent connection, requests pipelined in the daemon."""

    def __init__(self, sock_path: str = DEFAULT_SOCK, connect_timeout: float = 20.0) -> None:
        self.sock_path = sock_path
        self.connect_timeout = connect_timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task | None = None
        self._queues: dict[str, asyncio.Queue[Qwen3Chunk]] = {}
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()

    async def _ensure(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        deadline = time.monotonic() + self.connect_timeout
        while True:
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(self.sock_path)
                break
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                if time.monotonic() >= deadline:
                    raise ConnectionError(f"qwen3 tts daemon not reachable at {self.sock_path}: {exc}") from exc
                await asyncio.sleep(0.5)
        self._read_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    LOG.warning("[qwen3] bad line from daemon: %r", line[:200])
                    continue
                rid = str(msg.get("id", ""))
                q = self._queues.get(rid)
                kind = msg.get("type")
                if q is None:
                    continue
                if kind == "chunk":
                    q.put_nowait(Qwen3Chunk(rid, Path(msg["path"]), int(msg.get("seq", 0)), bool(msg.get("final", False))))
                elif kind in ("done", "cancelled", "error"):
                    q.put_nowait(Qwen3Chunk(rid, None))
                    self._queues.pop(rid, None)
        except Exception as exc:  # connection reset etc.
            LOG.warning("[qwen3] daemon connection lost: %s", exc)
        finally:
            self._writer = None
            for rid, q in list(self._queues.items()):
                q.put_nowait(Qwen3Chunk(rid, None))
            self._queues.clear()

    async def _send(self, msg: dict) -> None:
        async with self._lock:
            await self._ensure()
            assert self._writer is not None
            self._writer.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            await self._writer.drain()

    async def submit(self, text: str) -> tuple[str, asyncio.Queue[Qwen3Chunk]]:
        """Start a request; chunks (then a path=None end marker) arrive on the returned queue."""
        rid = f"o{os.getpid()}_{next(self._ids)}"
        q: asyncio.Queue[Qwen3Chunk] = asyncio.Queue()
        self._queues[rid] = q
        try:
            await self._send({"type": "synth", "id": rid, "text": text})
        except Exception:
            self._queues.pop(rid, None)
            raise
        return rid, q

    def _end_local(self) -> None:
        # the daemon answers a "*" cancel once, not per request: close every local stream now
        # (late chunk lines for these ids are ignored since their queues are gone)
        for rid, q in list(self._queues.items()):
            q.put_nowait(Qwen3Chunk(rid, None))
        self._queues.clear()

    async def cancel_all(self) -> None:
        self._end_local()
        if self._writer is None:
            return
        try:
            await self._send({"type": "cancel", "id": "*"})
        except Exception as exc:
            LOG.warning("[qwen3] cancel failed: %s", exc)

    def cancel_all_nowait(self) -> None:
        """For sync callers (TtsQueue.abort) running on the event loop thread."""
        self._end_local()
        if self._writer is None or self._writer.is_closing():
            return
        try:
            self._writer.write(b'{"type":"cancel","id":"*"}\n')
        except Exception:
            pass


def synthesize_blocking(text: str, out_path: Path, sock_path: str = DEFAULT_SOCK, timeout: float = 60.0) -> Path:
    """Whole-utterance synth to one WAV (ack cache warm-up). Uses its own connection."""
    rid = f"w{os.getpid()}_{time.monotonic_ns()}"
    chunks: list[Path] = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall((json.dumps({"type": "synth", "id": rid, "text": text}, ensure_ascii=False) + "\n").encode())
        f = s.makefile("r", encoding="utf-8")
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("qwen3 tts daemon closed the connection")
            msg = json.loads(line)
            if msg.get("id") != rid:
                continue
            if msg.get("type") == "chunk":
                chunks.append(Path(msg["path"]))
            elif msg.get("type") in ("done", "cancelled", "error"):
                break
    if not chunks:
        raise RuntimeError(f"qwen3 tts produced no audio for {text[:40]!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        for p in chunks:
            with wave.open(str(p), "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))
            p.unlink(missing_ok=True)
    return out_path


def read_pcm(path: Path) -> bytes:
    import audioop  # stdlib up to 3.12 (board runs 3.10); imported lazily so tests run on newer Pythons

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM")
        data = w.readframes(w.getnframes())
        if w.getnchannels() == 2:
            data = audioop.tomono(data, 2, 0.5, 0.5)
        if w.getframerate() != SAMPLE_RATE:
            data, _ = audioop.ratecv(data, 2, 1, w.getframerate(), SAMPLE_RATE, None)
    return data


VOLUME_RANGE_DB = 30.0


def volume_scale(percent) -> float:
    """Owner volume 0-100 % -> amplitude factor: 100 % = as loud as before, every 10 % down about -3 dB."""
    try:
        pct = max(0.0, min(100.0, float(percent)))
    except (TypeError, ValueError):
        return 1.0
    if pct <= 0:
        return 0.0
    return 10 ** (-VOLUME_RANGE_DB * (1 - pct / 100) / 20)


class StreamPlayer:
    """One aplay process per utterance; PCM chunks are written to its stdin back to back."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._stopped = False
        self.gain: float | None = None

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _script_env(self, env: dict[str, str]) -> dict[str, str]:
        e = os.environ.copy()
        e.update(env)
        return e

    def start(self, env: dict[str, str]) -> None:
        e = self._script_env(env)
        subprocess.run(
            ["bash", "-c", f"source {VOICE_SCRIPTS}/voice_env.sh; bash {VOICE_SCRIPTS}/speaker_setup.sh"],
            env=e,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        channels = int(e.get("PLAYBACK_CHANNELS", "2"))
        if e.get("PLAYBACK_MONO", "stereo") == "stereo":
            channels = 2
        self.channels = max(1, min(2, channels))
        cmd = [
            "aplay", "-q", "-D", "plughw:0,0", "-t", "raw", "-f", "S16_LE",
            "-r", str(SAMPLE_RATE), "-c", str(self.channels),
            f"--buffer-time={e.get('PLAYBACK_BUFFER_US', '80000')}",
        ]
        if shutil.which("pasuspender"):
            cmd = ["pasuspender", "--", *cmd]
        with self._lock:
            self._stopped = False
            self.gain = None
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, env=e, start_new_session=True
            )
        self._target_peak = int(e.get("PLAYBACK_TARGET_PEAK", "26000"))
        self._max_gain = float(e.get("PLAYBACK_MAX_GAIN", "8.0"))
        self._volume = volume_scale(e.get("PLAYBACK_VOLUME", "100"))
        # amp / codec settle, same as play_wav.sh lead-in
        self._write_raw(b"\x00\x00" * int(SAMPLE_RATE * 0.08) * self.channels)

    def _write_raw(self, data: bytes) -> bool:
        proc = self._proc
        if proc is None or proc.stdin is None or self._stopped:
            return False
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
            return True
        except (BrokenPipeError, ValueError, OSError):
            return False

    def write_wav(self, path: Path) -> bool:
        import audioop

        pcm = read_pcm(path)
        if self.gain is None:
            # one gain per utterance (from its first chunk), like play_wav.sh's utterance_gain
            peak = audioop.max(pcm, 2) if pcm else 0
            auto = 1.0 if peak <= 0 else max(1.0, min(self._max_gain, self._target_peak / peak))
            self.gain = auto * getattr(self, "_volume", 1.0)   # loudness normalised, then the owner's volume
        if self.gain != 1.0:
            pcm = audioop.mul(pcm, 2, self.gain)
        if self.channels == 2:
            pcm = audioop.tostereo(pcm, 2, 1.0, 1.0)
        return self._write_raw(pcm)

    def finish(self, *, restore_mic: bool, env: dict[str, str]) -> None:
        """Let queued audio drain, then close the device (and re-arm the mic)."""
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        self._write_raw(b"\x00\x00" * int(SAMPLE_RATE * 0.03) * getattr(self, "channels", 2))
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.stop()
        with self._lock:
            if self._proc is proc:
                self._proc = None
        if restore_mic and not self._stopped:
            subprocess.run(
                ["bash", f"{VOICE_SCRIPTS}/mic_setup.sh"],
                env=self._script_env(env),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            proc = self._proc
            self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=1.0)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
