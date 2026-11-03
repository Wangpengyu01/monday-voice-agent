#!/bin/bash
# Boot chime ("叮咚"): played by r1-boot-chime.service as soon as the sound card is up, long before
# the models are loaded, so the user knows the board is starting.
WAV="${1:-/userdata/agent/assets/boot_chime.wav}"
[ -f "$WAV" ] || exit 0
# route the codec to the speaker (same as TTS playback)
bash -c "source /userdata/voice/scripts/voice_env.sh; bash /userdata/voice/scripts/speaker_setup.sh" >/dev/null 2>&1 || true
for _ in 1 2 3; do
  aplay -q -D plughw:0,0 "$WAV" 2>/dev/null && exit 0
  sleep 1  # device busy (PulseAudio grabbing it during login): retry
done
exit 0
