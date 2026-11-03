#!/bin/bash
# 在 R1 板子上应用一次部署（由 deploy_r1_board.sh 上传后以 sudo 调用）。
#
#   sudo bash board_apply_deploy.sh /tmp/r1-deploy/agent
#
# 只适配这块板子（板载 ES8388 麦克风、DSI 屏、只跑 llm_daemon）：
# - 只装 r1-llm-daemon / r1-orchestrator / r1-agent-ui / r1-gpio-poweroff 四个服务，
#   板上有 Qwen3-TTS（/userdata/qwen3tts）时再加 r1-tts-daemon
# - 不装 r1-p4-eth（会改 eth0）、r1-agent-gui（会关 DSI 屏）、r1-webui、r1-vlm-daemon 等
# - 部署前自动备份当前版本到 /userdata/backups/，出问题用 restore_r1_agent.sh 回退
set -euo pipefail

NEW="${1:?usage: board_apply_deploy.sh <staged agent dir>}"
AGENT=/userdata/agent
BACKUPS=/userdata/backups
USER_NAME=youyeetoo
UNITS=(r1-llm-daemon r1-orchestrator r1-agent-ui r1-gpio-poweroff r1-boot-chime)
TTS_BIN=/userdata/qwen3tts/rknn_Qwen3_TTS_demo/rknn_qwen3_tts_daemon
if [ -x "$TTS_BIN" ]; then
  UNITS+=(r1-tts-daemon)
fi

[ "$(id -u)" -eq 0 ] || { echo "需要 sudo" >&2; exit 1; }
[ -f "$NEW/orchestrator/main.py" ] || { echo "不是 agent 目录: $NEW" >&2; exit 1; }

TS=$(date +%Y%m%d-%H%M%S)
echo "== 1/6 部署前备份当前版本"
mkdir -p "$BACKUPS"
chown "$USER_NAME:$USER_NAME" "$BACKUPS"
tar czf "$BACKUPS/r1-agent-$TS.tar.gz" -C / \
  --exclude="userdata/agent/run/chromium-kiosk" \
  userdata/agent \
  $(cd / && ls -d etc/systemd/system/r1-*.service 2>/dev/null) \
  home/$USER_NAME/.config/autostart \
  $(cd / && ls -d home/$USER_NAME/Desktop/*.desktop 2>/dev/null)
(cd "$BACKUPS" && sha256sum "r1-agent-$TS.tar.gz" > "r1-agent-$TS.tar.gz.sha256")
echo "   $BACKUPS/r1-agent-$TS.tar.gz"

daemon_changed=0
if ! cmp -s "$NEW/bin/llm_daemon" "$AGENT/bin/llm_daemon" \
   || ! cmp -s "$NEW/systemd/r1-llm-daemon.service" /etc/systemd/system/r1-llm-daemon.service; then
  daemon_changed=1
fi

tts_changed=0
if [ -x "$TTS_BIN" ] && ! cmp -s "$NEW/systemd/r1-tts-daemon.service" /etc/systemd/system/r1-tts-daemon.service; then
  tts_changed=1
fi

echo "== 2/6 停止服务（llm_daemon 有变化才重启: $daemon_changed）"
systemctl stop r1-orchestrator r1-agent-ui 2>/dev/null || true
if [ "$daemon_changed" -eq 1 ]; then
  systemctl stop r1-llm-daemon 2>/dev/null || true
fi

echo "== 3/6 同步代码（保留 run/ logs/ 和板上旧备份目录）"
rsync -a --delete \
  --exclude=/run/ --exclude=/logs/ \
  --exclude=/.codex-backups/ --exclude=/.codex-build/ --exclude=/backup_stable_1.21s/ \
  --exclude=__pycache__/ \
  "$NEW/" "$AGENT/"
mkdir -p "$AGENT/run" "$AGENT/logs"
chown -R "$USER_NAME:$USER_NAME" "$AGENT"
chmod +x "$AGENT/bin/llm_daemon" "$AGENT"/scripts/*.sh 2>/dev/null || true

echo "== 4/6 安装服务和定时任务"
for u in "${UNITS[@]}"; do
  install -m 644 "$AGENT/systemd/$u.service" /etc/systemd/system/
done
install -m 644 "$AGENT/systemd/r1-agent-logs.cron" /etc/cron.d/r1-agent-logs
# 内核参数预留的 5 GB 大页没人用（RK1828 走 PCIe），开机还给系统；删掉这个文件即恢复
install -m 644 "$AGENT/systemd/90-r1-free-hugepages.conf" /etc/sysctl.d/
sysctl -q -p /etc/sysctl.d/90-r1-free-hugepages.conf || true
install -d -o "$USER_NAME" -g "$USER_NAME" "/home/$USER_NAME/.config/autostart"
install -m 644 -o "$USER_NAME" -g "$USER_NAME" \
  "$AGENT/scripts/xiaolan-gui-autostart.desktop" "/home/$USER_NAME/.config/autostart/"
systemctl daemon-reload
systemctl enable "${UNITS[@]}" >/dev/null 2>&1

echo "== 5/6 启动服务"
LOG="$AGENT/logs/orchestrator.log"
log_off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
if [ "$daemon_changed" -eq 1 ] || ! systemctl is-active --quiet r1-llm-daemon; then
  systemctl start r1-llm-daemon
fi
if [ -x "$TTS_BIN" ] && { [ "$tts_changed" -eq 1 ] || ! systemctl is-active --quiet r1-tts-daemon; }; then
  systemctl restart r1-tts-daemon
fi
systemctl restart r1-gpio-poweroff
systemctl start r1-agent-ui
systemctl start r1-orchestrator

echo "== 6/6 检查"
ok=0
for _ in $(seq 1 90); do
  if tail -c +"$((log_off + 1))" "$LOG" 2>/dev/null | grep -aq "IDLE -> LISTEN"; then ok=1; break; fi
  sleep 2
done
for u in "${UNITS[@]}"; do printf "   %-18s %s\n" "$u" "$(systemctl is-active "$u" || true)"; done
if [ "$ok" -eq 1 ]; then
  echo "部署完成，语音助手已进入监听。回退: sudo bash $BACKUPS/restore_r1_agent.sh"
else
  echo "警告: 3 分钟内没看到 IDLE -> LISTEN，查看 $AGENT/logs/orchestrator.log" >&2
  echo "回退: sudo bash $BACKUPS/restore_r1_agent.sh" >&2
  exit 2
fi
