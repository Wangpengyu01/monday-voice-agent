import sys
import os
import time
import subprocess
from pathlib import Path

sys.path.insert(0, '/userdata/agent')

from tts.tts_engine import TtsEngine

print("=== [TEST 1] Synthesizing Test Audio Files ===")
engine = TtsEngine()

tests = {
    "shoudu": "首都是哪里",
    "china_shoudu": "中国的首都是哪里",
    "stop": "停",
    "continue": "你说",
    "solar": "介绍一下太阳系",
}

paths = {}
for key, text in tests.items():
    p = f"/tmp/test_{key}.wav"
    wav = engine.synthesize(text, p)
    paths[key] = str(wav)
    print(f"  [+] {key} ('{text}') -> {wav}")

def read_log_tail(n=25):
    try:
        res = subprocess.run(["tail", f"-n{n}", "/userdata/agent/logs/orchestrator.log"], capture_output=True, text=True)
        return res.stdout
    except Exception as e:
        return str(e)

print("\n=== [TEST 2] Playing Question: '首都是哪里' via Speaker ===")
print("Speaker will play audio into microphone...")
subprocess.run(["bash", "/userdata/voice/scripts/play_wav.sh", paths["shoudu"]])

print("Waiting 4s for pipeline response...")
time.sleep(4)

print("\n--- Recent Log Events ---")
print(read_log_tail(30))
