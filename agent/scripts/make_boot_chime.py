"""Synthesize the boot chime: a deep, full "咚" then a soft little "叮" (~1.5 s), original sound.

python make_boot_chime.py out.wav   -> 48 kHz stereo 16-bit
The board's small speaker cannot play real bass without distorting, so the "咚" is pitched at G3
(196 Hz) but carried mostly by its 2nd-4th harmonics: the ear hears the missing low fundamental,
the speaker only has to reproduce 400-800 Hz. Slightly detuned for warmth, a quick downward pitch
settle at the strike for weight. The "叮" (C6) is about a third as loud and rings briefly.
Peak at ~-8 dBFS so nothing clips.
"""
import math
import random
import struct
import sys
import wave

SR = 48000
DUR = 1.5
N = int(SR * DUR)


def dong(x):
    """Deep warm strike: harmonic partials, the fundamental itself kept low."""
    f0 = 196.0
    attack = int(SR * 0.004)
    # (harmonic, gain, decay s)
    partials = [(1, 0.35, 0.65), (2, 1.0, 0.55), (3, 0.7, 0.4), (4, 0.4, 0.28), (5, 0.18, 0.18), (6, 0.08, 0.12)]
    for h, g, decay in partials:
        for cents in (-2.0, 0.0, 2.0):
            ph = random.random() * 2 * math.pi
            tau = SR * decay
            amp = g / 3
            for i in range(N):
                env = min(1.0, i / attack) * math.exp(-i / tau)
                if env < 1e-4 and i > attack:
                    break
                t = i / SR
                # pitch starts ~1.5% sharp and settles within 40 ms: a "thump" feel
                bend = 1.0 + 0.015 * math.exp(-t / 0.04)
                f = f0 * h * bend * (2 ** (cents / 1200))
                ph += 2 * math.pi * f / SR
                x[i] += amp * env * math.sin(ph)


def ding(x, start, gain):
    f0 = 1046.5
    s0 = int(SR * start)
    attack = int(SR * 0.002)
    for ratio, g, decay in ((1.0, 1.0, 0.45), (2.0, 0.25, 0.18), (2.76, 0.08, 0.1)):
        w = 2 * math.pi * f0 * ratio / SR
        for i in range(N - s0):
            env = min(1.0, i / attack) * math.exp(-i / (SR * decay))
            if env < 1e-4 and i > attack:
                break
            x[s0 + i] += gain * g * env * math.sin(w * i)


def reverb(x, delays=(0.021, 0.033, 0.047), fb=0.3, mix=0.15):
    out = list(x)
    for d in delays:
        k = int(SR * d)
        buf = [0.0] * len(x)
        for i in range(k, len(x)):
            buf[i] = x[i - k] + fb * buf[i - k]
        for i in range(len(x)):
            out[i] += mix * buf[i] / len(delays)
    return out


def main():
    random.seed(5)
    mono = [0.0] * N
    dong(mono)
    dong_peak = max(abs(v) for v in mono) or 1.0
    ding(mono, 0.42, 0.33 * dong_peak)  # the "叮" about a third as loud as the "咚"
    wet = reverb(mono)
    peak = max(abs(v) for v in wet) or 1.0
    scale = 0.4 * 32767 / peak
    fade = int(SR * 0.25)
    with wave.open(sys.argv[1], "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        frames = bytearray()
        for i in range(N):
            f = 1.0 if i < N - fade else (N - i) / fade
            v = int(wet[i] * scale * f)
            frames += struct.pack("<hh", v, v)
        w.writeframes(bytes(frames))


if __name__ == "__main__":
    main()
