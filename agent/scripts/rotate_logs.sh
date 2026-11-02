#!/bin/bash
# 小揽日志轮转（板上没装 logrotate）。由 /etc/cron.d/r1-agent-logs 每小时调用。
# 服务用 StandardOutput=append 一直写着日志，所以先复制再清空（copytruncate），不用重启服务。
set -u
LOG_DIR="${1:-/userdata/agent/logs}"
MAX_BYTES="${R1_LOG_MAX_BYTES:-20971520}"   # 20MB
KEEP="${R1_LOG_KEEP:-3}"

for f in "${LOG_DIR}"/*.log; do
  [ -f "$f" ] || continue
  size=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$size" -ge "$MAX_BYTES" ] || continue
  for ((i = KEEP - 1; i >= 1; i--)); do
    [ -f "$f.$i.gz" ] && mv -f "$f.$i.gz" "$f.$((i + 1)).gz"
  done
  rm -f "$f.$((KEEP + 1)).gz"
  cp -p "$f" "$f.1" && : > "$f" && gzip -f "$f.1"
done
