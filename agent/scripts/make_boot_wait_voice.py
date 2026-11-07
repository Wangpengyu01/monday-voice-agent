#!/usr/bin/env python3
"""Pre-make the boot wait clips (assets/boot_wait_*.wav) with the current TTS voice.

Needs the running TTS daemon. Re-run after changing the voice:
    python3 /userdata/agent/scripts/make_boot_wait_voice.py
"""
import json
import socket
import sys
import time
import wave
from pathlib import Path

CLIPS = {
    "boot_wait_start": "星期一正在启动，大约需要一分钟，请稍等。",
    "boot_wait_1": "还在加载中，请再等一下。",
    "boot_wait_2": "马上就好。",
}


def synth(sock_path: str, text: str, out: Path) -> float:
    s = socket.socket(socket.AF_UNIX)
    s.connect(sock_path)
    f = s.makefile("r")
    rid = f"bootwait{int(time.time() * 1000)}"
    s.sendall((json.dumps({"type": "synth", "id": rid, "text": text}) + "\n").encode())
    paths = []
    while True:
        m = json.loads(f.readline())
        if m.get("id") != rid:
            continue
        if m["type"] == "chunk":
            paths.append(m["path"])
        elif m["type"] in ("done", "error", "cancelled"):
            break
    s.close()
    if not paths:
        raise RuntimeError(f"no audio for {text!r}")
    tmp = out.with_suffix(".tmp")
    with wave.open(str(tmp), "wb") as w:
        for i, p in enumerate(paths):
            with wave.open(p) as r:
                if i == 0:
                    w.setparams(r.getparams())
                w.writeframes(r.readframes(r.getnframes()))
            Path(p).unlink(missing_ok=True)
    tmp.replace(out)
    with wave.open(str(out)) as r:
        return r.getnframes() / r.getframerate()


def main() -> None:
    sock = sys.argv[1] if len(sys.argv) > 1 else "/tmp/r1-tts.sock"
    assets = Path(sys.argv[2] if len(sys.argv) > 2 else "/userdata/agent/assets")
    for name, text in CLIPS.items():
        # sampling TTS can run on past the phrase: retry clips that come out too long
        for _ in range(4):
            sec = synth(sock, text, assets / f"{name}.wav")
            if sec <= 0.6 + 0.3 * len(text):
                break
        print(f"{name}: {sec:.2f}s {text}")


if __name__ == "__main__":
    main()
