import sys
import time
import subprocess

sys.path.insert(0, "/userdata/agent")
from tts.tts_engine import TtsEngine

print("=== 1. Synthesizing '今天是星期几' ===")
engine = TtsEngine()
wav_path = engine.synthesize("今天是星期几", "/tmp/test_dt.wav")
print(f"Generated: {wav_path}")

print("=== 2. Playing ===")
subprocess.run(["bash", "/userdata/voice/scripts/play_wav.sh", str(wav_path)])

print("=== 3. Waiting 8s for Xiao Lan reply ===")
time.sleep(8)

print("=== 4. Last 40 log lines ===")
res = subprocess.run(
    ["tail", "-n", "40", "/userdata/agent/logs/orchestrator.log"],
    capture_output=True,
    text=True,
)
print(res.stdout)

n_partial = res.stdout.count("[asr] partial")
n_mute_blocked = res.stdout.count("mute_mic: blocked")
print(f"ASR partials in log window:    {n_partial}")
print(f"mute_mic blocked events:       {n_mute_blocked}")
if n_partial == 0:
    print("GOOD — no ASR partials during TTS!")
else:
    print("WARN — ASR still processing mic input during TTS")
