#!/bin/bash
# After power-on the models take about a minute to load. Instead of staying silent after the
# chime: say "starting, about a minute" once, then a short "still loading" line every 20 s,
# until the orchestrator's boot self-test marks this boot as announced (run/selftest_boot_id,
# written just before it says "星期一已就绪"). Clips are pre-made by make_boot_wait_voice.py.
AGENT=/userdata/agent
ASSETS="$AGENT/assets"
MARKER="${BOOT_WAIT_MARKER:-$AGENT/run/selftest_boot_id}"
BOOT_ID="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
INTERVAL="${BOOT_WAIT_INTERVAL:-20}"
MAX_SEC=150

ready() { [ -n "$BOOT_ID" ] && [ "$(cat "$MARKER" 2>/dev/null)" = "$BOOT_ID" ]; }

play() {
  [ -f "$1" ] || return 0
  ready && return 0
  for _ in 1 2 3; do
    aplay -q -D plughw:0,0 "$1" 2>/dev/null && return 0
    sleep 1  # device busy: retry
  done
}

ready && exit 0  # restarted after this boot was already announced
play "$ASSETS/boot_wait_start.wav"
start=$(date +%s)
n=0
while [ $(( $(date +%s) - start )) -lt $MAX_SEC ]; do
  for _ in $(seq 1 $INTERVAL); do
    ready && exit 0
    sleep 1
  done
  n=$((n + 1))
  play "$ASSETS/boot_wait_$(( (n - 1) % 2 + 1 )).wav"
done
exit 0
