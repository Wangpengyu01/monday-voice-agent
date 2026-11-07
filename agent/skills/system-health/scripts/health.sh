#!/bin/bash
# Read-only health report of the board running 星期一.
set -u
echo "== 服务"
for s in r1-orchestrator r1-llm-daemon r1-tts-daemon r1-agent-ui r1-devices r1-selfheal.timer r1-practice.timer r1-reflect.timer r1-backup.timer; do
  printf "%-22s %s\n" "$s" "$(systemctl is-active "$s" 2>/dev/null)"
done
echo "== 资源"
t=0
for z in /sys/class/thermal/thermal_zone*/temp; do v=$(cat "$z" 2>/dev/null || echo 0); [ "$v" -gt "$t" ] && t=$v; done
echo "芯片最高温度 $((t / 1000)) 度"
echo "负载 $(cut -d' ' -f1-3 /proc/loadavg)"
awk '/MemAvailable/ {printf "内存可用 %.1f G\n", $2/1048576}' /proc/meminfo
df -h / /userdata 2>/dev/null | awk 'NR>1 {print "磁盘 " $6 " 用了 " $5 "，剩 " $4}'
echo "== 最近十分钟的报错"
log=/userdata/agent/logs/orchestrator.log
since=$(date -d '-10 min' +%H:%M 2>/dev/null)
if [ -f "$log" ]; then
  tail -n 3000 "$log" | grep -a -E "ERROR|Traceback|Exception| failed" | awk -v s="$since" '$1 >= s' | tail -n 10
fi
echo "(完)"
