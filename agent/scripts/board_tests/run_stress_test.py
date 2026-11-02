#!/usr/bin/env python3
"""Automated acoustic stress test script.

Evaluates:
1. Zero self-interruption & no repeating loops when speaker is directed at mic.
2. Real barge-in interruption on '停'.
3. Conversational relay on '你说' / '继续'.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

LOG_PATH = Path("/userdata/agent/logs/orchestrator.log")

def get_log_offset():
    if LOG_PATH.exists():
        return LOG_PATH.stat().st_size
    return 0

def read_new_logs(offset):
    if not LOG_PATH.exists():
        return "", offset
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(offset)
        content = f.read()
        new_offset = f.tell()
    return content, new_offset

def play(wav_path):
    print(f"  [AUDIO PLAY] {wav_path}", flush=True)
    # Set PLAYBACK_CONTINUATION=1 to prevent mixer popping
    env = os.environ.copy()
    env["PLAYBACK_CONTINUATION"] = "1"
    subprocess.run(["bash", "/userdata/voice/scripts/play_wav.sh", wav_path], env=env, check=True)

def wait_for_pattern(offset, pattern, timeout=10.0, check_interval=0.1):
    start = time.monotonic()
    collected = ""
    curr_offset = offset
    while time.monotonic() - start < timeout:
        new_text, curr_offset = read_new_logs(curr_offset)
        if new_text:
            collected += new_text
            if pattern in collected:
                return True, collected, curr_offset
        time.sleep(check_interval)
    return False, collected, curr_offset

def main():
    print("==================================================", flush=True)
    print("      R1 ACOUSTIC FEEDBACK & CONVERSATION EVAL    ", flush=True)
    print("==================================================", flush=True)
    
    offset = get_log_offset()
    
    # ----------------------------------------------------
    # TEST 1: Question without interruption + Echo rejection
    # ----------------------------------------------------
    print("\n--- TEST 1: Direct Question & Anti-Echo Loopback ---", flush=True)
    print("Query: '中国的首都是哪里？' (Speaker -> Mic)", flush=True)
    play("/tmp/eval_china_shoudu_16k.wav")
    
    # Wait for TTS to start and finish
    ok, new_logs, offset = wait_for_pattern(offset, "TTS -> LISTEN", timeout=12.0)
    # Wait another 2.5s to catch any secondary tail loops
    time.sleep(2.5)
    tail_logs, offset = read_new_logs(offset)
    test1_logs = new_logs + tail_logs
    
    print("\n[TEST 1 LOGS]:", flush=True)
    for line in test1_logs.strip().splitlines():
        if any(k in line for k in ("asr", "event", "state", "llm", "tts", "barge-in", "relay")):
            print("  " + line, flush=True)
        
    t1_pass = True
    if "北京" in test1_logs:
        print("  [PASS] Correct fact answered ('北京').", flush=True)
    else:
        print("  [FAIL] Did not answer capital question correctly.", flush=True)
        t1_pass = False
        
    if "detected acoustic echo" in test1_logs or "self-acoustic feedback detected" in test1_logs:
        print("  [PASS] Acoustic feedback successfully intercepted and suppressed.", flush=True)
    else:
        print("  [INFO] No acoustic feedback crossed VAD threshold.", flush=True)
        
    # Check if loop happened (more than 1 LLM query)
    llm_count = test1_logs.count("ASR -> LLM")
    if llm_count <= 1:
        print(f"  [PASS] Loop suppression confirmed (LLM query count: {llm_count}).", flush=True)
    else:
        print(f"  [FAIL] Loop occurred (LLM query count: {llm_count})!", flush=True)
        t1_pass = False

    time.sleep(2.0)
    offset = get_log_offset()

    # ----------------------------------------------------
    # TEST 2: Long utterance + Barge-in on '停'
    # ----------------------------------------------------
    print("\n--- TEST 2: Conversational Barge-In on '停' ---", flush=True)
    print("Query: '请详细介绍一下太阳系的各大行星。'", flush=True)
    play("/tmp/eval_solar_16k.wav")
    
    # Wait until speech starts
    ok, pre_logs, offset = wait_for_pattern(offset, "tts_sentence:", timeout=8.0)
    if not ok:
        print("  [FAIL] TTS did not start in time.", flush=True)
    else:
        print("  [+] Assistant started speaking. Waiting 0.8s then issuing '停'...", flush=True)
        time.sleep(0.8)
        play("/tmp/eval_stop_16k.wav")
        
    ok, barge_logs, offset = wait_for_pattern(offset, "TTS -> LISTEN", timeout=8.0)
    time.sleep(2.0)
    tail_barge, offset = read_new_logs(offset)
    test2_logs = pre_logs + barge_logs + tail_barge
    
    print("\n[TEST 2 LOGS]:", flush=True)
    for line in test2_logs.strip().splitlines():
        if any(k in line for k in ("barge-in", "interrupt", "relay", "stop", "tts_sentence", "ASR ->", "user requested stop")):
            print("  " + line, flush=True)
            
    t2_pass = False
    if "interrupt TTS playback immediately" in test2_logs or "barge-in" in test2_logs or "user requested stop" in test2_logs:
        print("  [PASS] Instant barge-in triggered on '停'!", flush=True)
        t2_pass = True
    else:
        print("  [FAIL] Barge-in did not trigger.", flush=True)
        
    if "saved" in test2_logs and "relay" in test2_logs:
        print("  [PASS] Paused clauses saved for conversational relay.", flush=True)
    else:
        print("  [INFO] Paused relay status recorded.", flush=True)

    time.sleep(2.0)
    offset = get_log_offset()

    # ----------------------------------------------------
    # TEST 3: Conversational Relay on '你说吧'
    # ----------------------------------------------------
    print("\n--- TEST 3: Conversational Relay on '你说吧' ---", flush=True)
    print("Command: '你说吧。'", flush=True)
    play("/tmp/eval_relay_16k.wav")
    
    ok, relay_logs, offset = wait_for_pattern(offset, "TTS -> LISTEN", timeout=8.0)
    time.sleep(2.0)
    tail_relay, offset = read_new_logs(offset)
    test3_logs = relay_logs + tail_relay
    
    print("\n[TEST 3 LOGS]:", flush=True)
    for line in test3_logs.strip().splitlines():
        if any(k in line for k in ("relay", "resuming", "speak", "tts_sentence", "TTS ->", "ASR ->")):
            print("  " + line, flush=True)
            
    t3_pass = False
    if "resuming playback" in test3_logs or "relay" in test3_logs:
        print("  [PASS] Conversational relay confirmed!", flush=True)
        t3_pass = True
    else:
        print("  [FAIL] Relay did not trigger.", flush=True)

    print("\n==================================================", flush=True)
    print("                 TEST SUMMARY                     ", flush=True)
    print("==================================================", flush=True)
    print(f"  Test 1 (Echo Rejection & Anti-Loop): {'PASS' if t1_pass else 'FAIL'}", flush=True)
    print(f"  Test 2 (Barge-In on '停'):            {'PASS' if t2_pass else 'FAIL'}", flush=True)
    print(f"  Test 3 (Relay on '你说吧'):           {'PASS' if t3_pass else 'FAIL'}", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()
